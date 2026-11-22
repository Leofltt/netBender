#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <random>
#include <cmath>
#include <thread>

//==============================================================================
NetBenderAudioProcessor::NetBenderAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(BusesProperties()
#if !JucePlugin_IsMidiEffect
#if !JucePlugin_IsSynth
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
#endif
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)
#endif
      )
#endif
{
    addParameter(m_dryWetParam = new juce::AudioParameterFloat(
                     juce::ParameterID("dry_wet", 1), "Dry / Wet",
                     juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 1.0f));

    addParameter(m_scaleParam = new juce::AudioParameterFloat(
                     juce::ParameterID("scale", 1), "Layer Scale",
                     juce::NormalisableRange<float>(0.0f, 5.0f, 0.01f), 1.0f));

    addParameter(m_offsetParam = new juce::AudioParameterFloat(
                     juce::ParameterID("offset", 1), "Layer Offset",
                     juce::NormalisableRange<float>(-2.0f, 2.0f, 0.001f), 0.0f));

    addParameter(m_heatParam = new juce::AudioParameterFloat(
                     juce::ParameterID("heat", 1), "Layer Heat / Volatility",
                     juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.0f));

    addParameter(m_memoryParam = new juce::AudioParameterFloat(
                     juce::ParameterID("memory", 1), "Drift Memory / Drag",
                     juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.8f));

    addParameter(m_shortCircuitParam = new juce::AudioParameterBool(
                     juce::ParameterID("short_circuit", 1), "Momentary Short Circuit", false));

    addParameter(m_latentXParam = new juce::AudioParameterFloat(
                     juce::ParameterID("latent_x", 1), "Latent Terrain X",
                     juce::NormalisableRange<float>(-1.0f, 1.0f, 0.001f), 0.0f));

    addParameter(m_latentYParam = new juce::AudioParameterFloat(
                     juce::ParameterID("latent_y", 1), "Latent Terrain Y",
                     juce::NormalisableRange<float>(-1.0f, 1.0f, 0.001f), 0.0f));

    addParameter(m_latentDepthParam = new juce::AudioParameterFloat(
                     juce::ParameterID("latent_depth", 1), "Latent Depth",
                     juce::NormalisableRange<float>(0.0f, 2.0f, 0.01f), 0.5f));

    addParameter(m_decodeGenSmoothnessParam = new juce::AudioParameterFloat(
                     juce::ParameterID("decode_gen_smoothness", 1), "Decode Gen Smoothness",
                     juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.30f));

    m_session.set_latency_callback([this](int latency) { setLatencySamples(latency); });
    m_session.set_error_callback([](const std::string& msg) {
        juce::ignoreUnused(msg);
        DBG("NetBenderSession error: " << msg);
    });
}

NetBenderAudioProcessor::~NetBenderAudioProcessor() {
    m_session.set_latency_callback(nullptr);
    m_session.set_error_callback(nullptr);
    m_session.engine().detach_backend();
}

//==============================================================================
auto NetBenderAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) -> void {
    m_session.prepare(sampleRate, samplesPerBlock,
                      std::max(getTotalNumInputChannels(), getTotalNumOutputChannels()));
}

auto NetBenderAudioProcessor::releaseResources() -> void {
    m_session.release();
}

auto NetBenderAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const -> bool {
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
#else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() &&
        layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

#if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
#endif
}

auto NetBenderAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer& midiMessages) NETBENDER_NONBLOCKING
    -> void {
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Process incoming MIDI triggers (Note On kicks momentary glitch; Note Off releases)
    for (const auto metadata : midiMessages) {
        auto msg = metadata.getMessage();
        if (msg.isNoteOn()) {
            m_session.trigger_short_circuit(true);
        } else if (msg.isNoteOff()) {
            m_session.trigger_short_circuit(false);
        }
    }

    // Clear unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Push DAW automation into the session (wait-free atomic stores)
    m_session.set_dry_wet(m_dryWetParam ? m_dryWetParam->get() : m_session.get_dry_wet());
    m_session.set_bending_automation(
        m_scaleParam ? m_scaleParam->get() : 1.0f, m_offsetParam ? m_offsetParam->get() : 0.0f,
        m_heatParam ? m_heatParam->get() : 0.0f, m_memoryParam ? m_memoryParam->get() : -1.0f,
        m_shortCircuitParam ? m_shortCircuitParam->get() : false);
    if (m_latentXParam && m_latentYParam && m_latentDepthParam) {
        m_session.set_latent_automation(m_latentXParam->get(), m_latentYParam->get(),
                                        m_latentDepthParam->get());
    }
    if (m_decodeGenSmoothnessParam) {
        m_session.set_decode_gen_smoothness(m_decodeGenSmoothnessParam->get());
    }

    // Session zero-fills output when no model is loaded, runs the sidechain envelope follower,
    // and streams through StreamingEngine (resampling, dry/wet, SafetySentry)
    m_session.process(buffer.getArrayOfReadPointers(), totalNumInputChannels,
                      buffer.getArrayOfWritePointers(), totalNumOutputChannels,
                      buffer.getNumSamples());
}

//==============================================================================
auto NetBenderAudioProcessor::getAvailableModes() const -> std::vector<juce::String> {
    std::vector<juce::String> modes;
    for (const auto& m : m_session.get_available_modes())
        modes.emplace_back(juce::String(m));
    return modes;
}

auto NetBenderAudioProcessor::loadModel(const juce::File& file) -> bool {
    return m_session.load_model(file.getFullPathName().toStdString());
}

//==============================================================================
auto NetBenderAudioProcessor::syncParametersFromSession() -> void {
    if (m_dryWetParam)
        *m_dryWetParam = m_session.get_dry_wet();
    if (m_latentXParam)
        *m_latentXParam = m_session.latent().get_x();
    if (m_latentYParam)
        *m_latentYParam = m_session.latent().get_y();
    if (m_latentDepthParam)
        *m_latentDepthParam = m_session.latent().get_depth();
    if (m_decodeGenSmoothnessParam)
        *m_decodeGenSmoothnessParam = m_session.get_decode_gen_smoothness();
    if (m_shortCircuitParam)
        *m_shortCircuitParam = false;

    // Active-layer automation params are authoritative for the active layer, so seed them from
    // the restored layer state (or reset to defaults when no active layer is selected).
    const std::string activeLayer = m_session.get_active_layer();
    if (!activeLayer.empty()) {
        const auto state = m_session.bending().get_layer_state(activeLayer);
        if (m_scaleParam)
            *m_scaleParam = state.scale;
        if (m_offsetParam)
            *m_offsetParam = state.offset;
        if (m_heatParam)
            *m_heatParam = state.heat;
        if (m_memoryParam)
            *m_memoryParam = state.memory;
    } else {
        if (m_scaleParam)
            *m_scaleParam = 1.0f;
        if (m_offsetParam)
            *m_offsetParam = 0.0f;
        if (m_heatParam)
            *m_heatParam = 0.0f;
        if (m_memoryParam)
            *m_memoryParam = 0.8f;
    }
}

auto NetBenderAudioProcessor::getStateInformation(juce::MemoryBlock& destData) -> void {
    // Sync current DAW parameter values into the session before serializing JSON state
    if (m_dryWetParam)
        m_session.set_dry_wet(m_dryWetParam->get());
    if (m_latentXParam && m_latentYParam)
        m_session.latent().set_coords(m_latentXParam->get(), m_latentYParam->get());
    if (m_latentDepthParam)
        m_session.latent().set_depth(m_latentDepthParam->get());
    if (m_decodeGenSmoothnessParam)
        m_session.set_decode_gen_smoothness(m_decodeGenSmoothnessParam->get());

    const std::string activeLayer = m_session.get_active_layer();
    if (!activeLayer.empty()) {
        const float scaleVal = m_scaleParam ? m_scaleParam->get() : 1.0f;
        const float offsetVal = m_offsetParam ? m_offsetParam->get() : 0.0f;
        const float heatVal = m_heatParam ? m_heatParam->get() : 0.0f;
        const float memVal = m_memoryParam ? m_memoryParam->get() : 0.8f;
        const auto allStates = m_session.bending().get_all_layer_states();
        const bool isNonDefault = (std::abs(scaleVal - 1.0f) > 0.0001f ||
                                   std::abs(offsetVal) > 0.0001f || heatVal > 0.0001f);
        if (allStates.count(activeLayer) > 0 || isNonDefault) {
            auto state = m_session.bending().get_layer_state(activeLayer);
            if (std::abs(state.scale - scaleVal) > 0.0001f)
                m_session.bending().set_layer_scale(activeLayer, scaleVal);
            if (std::abs(state.offset - offsetVal) > 0.0001f)
                m_session.bending().set_layer_offset(activeLayer, offsetVal);
            if (std::abs(state.heat - heatVal) > 0.0001f)
                m_session.bending().set_layer_heat(activeLayer, heatVal);
            if (std::abs(state.memory - memVal) > 0.0001f)
                m_session.bending().set_layer_memory(activeLayer, memVal);
        }
    }

    const std::string state = m_session.save_state();
    destData.replaceAll(state.data(), state.size());
}

auto NetBenderAudioProcessor::setStateInformation(const void* data, int sizeInBytes) -> void {
    if (data == nullptr || sizeInBytes <= 0)
        return;
    const std::string state(static_cast<const char*>(data), static_cast<size_t>(sizeInBytes));
    if (m_session.load_state(state)) {
        syncParametersFromSession();
        m_stateRevision.fetch_add(1);
        updateHostDisplay(ChangeDetails().withNonParameterStateChanged(true));
    }
}

//==============================================================================
auto NetBenderAudioProcessor::createEditor() -> juce::AudioProcessorEditor* {
    return new NetBenderAudioProcessorEditor(*this);
}

//==============================================================================
// This creates the filter...
auto JUCE_CALLTYPE createPluginFilter() -> juce::AudioProcessor* {
    return new NetBenderAudioProcessor();
}
