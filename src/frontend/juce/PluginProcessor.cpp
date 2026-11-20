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

    m_backend = BackendFactory::create_default();

    m_engine.set_pre_inference_callback([this](IBackend& backend) {
        std::string activeLayerStr;
        {
            std::lock_guard<std::mutex> stateLock(m_state_mutex);
            activeLayerStr = m_activeLayerName.toStdString();
        }
        const float scaleVal = m_scaleParam ? m_scaleParam->get() : 1.0f;
        const float offsetVal = m_offsetParam ? m_offsetParam->get() : 0.0f;
        const float heatVal = m_heatParam ? m_heatParam->get() : 0.0f;
        const float memVal = m_memoryParam ? m_memoryParam->get() : -1.0f;
        const bool shortVal = m_shortCircuitParam ? m_shortCircuitParam->get() : false;
        m_bending_manager.apply_bending(&backend, activeLayerStr, scaleVal, offsetVal, heatVal,
                                        memVal, shortVal);
    });

    m_engine.set_latent_hook_provider([this]() -> IBackend::LatentHook {
        std::string modeStr;
        IBackend::LatentHook userHook;
        {
            std::lock_guard<std::mutex> stateLock(m_state_mutex);
            modeStr = m_currentMethod.toStdString();
            userHook = m_latentHook;
        }
        if (userHook) {
            return userHook;
        }
        if (modeStr == "decode gen" || m_latent_engine.is_enabled()) {
            const float lx = m_latentXParam ? m_latentXParam->get() : -999.0f;
            const float ly = m_latentYParam ? m_latentYParam->get() : -999.0f;
            const float depth = m_latentDepthParam ? m_latentDepthParam->get() : -999.0f;
            return [this, lx, ly, depth](IBackend::LatentBuffer& buf) {
                m_latent_engine.apply_hook(buf, lx, ly, depth);
            };
        }
        return nullptr;
    });

    m_engine.set_fuse_reset_callback([this] {
        m_engine.get_safety_sentry().reset_fuse();
        m_bending_manager.request_audio_thread_reset_all();
    });

    m_engine.set_error_callback([](const std::string& msg) {
        juce::ignoreUnused(msg);
        DBG("StreamingEngine error: " << msg);
    });
}

NetBenderAudioProcessor::~NetBenderAudioProcessor() {
    m_engine.detach_backend();
}

//==============================================================================
auto NetBenderAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) -> void {
    m_engine.stop();
    m_hostSampleRate.store(sampleRate > 0.0 ? sampleRate : 44100.0);
    m_maxHostBlockSize.store(std::max(256, samplesPerBlock));
    if (m_backend) {
        m_backend->set_sample_rate(sampleRate);
    }
    configure_engine();
}

auto NetBenderAudioProcessor::releaseResources() -> void {
    m_engine.stop();
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
            m_bending_manager.trigger_short_circuit(true);
        } else if (msg.isNoteOff()) {
            m_bending_manager.trigger_short_circuit(false);
        }
    }

    // Clear unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    if (!m_modelLoaded.load() || !m_engine.is_configured()) {
        for (auto i = 0; i < buffer.getNumChannels(); ++i)
            buffer.clear(i, 0, buffer.getNumSamples());
        return;
    }

    const int numSamples = buffer.getNumSamples();

    // Sidechain Audio Envelope Follower for Transient Triggering
    if (totalNumInputChannels > 0 && numSamples > 0) {
        float maxPeak = 0.0f;
        for (int c = 0; c < totalNumInputChannels; ++c) {
            const float peak = buffer.getMagnitude(c, 0, numSamples);
            if (peak > maxPeak)
                maxPeak = peak;
        }
        m_bending_manager.update_audio_envelope(maxPeak);
    }

    m_engine.set_dry_wet(m_dryWetParam ? m_dryWetParam->get() : m_dryWet.load());
    m_engine.process(buffer.getArrayOfReadPointers(), totalNumInputChannels,
                     buffer.getArrayOfWritePointers(), totalNumOutputChannels, numSamples);
}

//==============================================================================
auto NetBenderAudioProcessor::getAvailableModes() const -> std::vector<juce::String> {
    std::vector<juce::String> modes;
    if (!m_modelLoaded.load() || !m_backend)
        return modes;

    auto backendModes = m_backend->get_plugin_modes();
    for (const auto& m : backendModes)
        modes.emplace_back(juce::String(m));
    return modes;
}

auto NetBenderAudioProcessor::loadModel(const juce::File& file) -> bool {
    m_modelLoaded.store(false);
    m_engine.detach_backend();

    std::string path = file.getFullPathName().toStdString();

    try {
        auto newBackend = BackendFactory::create_for_file(path);
        if (!newBackend)
            return false;

        if (file.getFileName().containsIgnoreCase("_coreml.pte") ||
            file.getFileName().containsIgnoreCase("_mps.pte"))
            m_useGpu.store(true);
        else if (file.getFileName().containsIgnoreCase("_xnnpack.pte") ||
                 file.getFileName().containsIgnoreCase("_portable.pte"))
            m_useGpu.store(false);

        newBackend->use_gpu(m_useGpu.load());
        m_backend = std::move(newBackend);

        int err = m_backend->load(path, getSampleRate());
        if (err == 0) {
            m_modelPath = file.getFullPathName();

            // Find default mode: prefer "forward", else "prior", else first available
            auto modes = m_backend->get_plugin_modes();
            if (!modes.empty()) {
                std::string defaultMode = modes[0];
                for (const auto& m : modes) {
                    if (m == "forward") {
                        defaultMode = "forward";
                        break;
                    }
                }
                {
                    std::lock_guard<std::mutex> stateLock(m_state_mutex);
                    m_currentMethod = defaultMode;
                }

                auto params = m_backend->get_mode_params(defaultMode);
                if (params.size() >= 4) {
                    m_model_in = params[0];
                    m_model_out = params[2];
                }
            }

            // Auto-configure buffer size to backend preferred size or model ratio.
            int prefSize = m_backend->get_preferred_buffer_size();
            if (prefSize > 0) {
                m_bufferSize = prefSize;
            } else {
                int higher_ratio = m_backend->get_higher_ratio();
                int reqSize = 1;
                while (reqSize < higher_ratio)
                    reqSize <<= 1;
                int minBufferForRatio = (higher_ratio >= 2048) ? 4096 : 2048;
                m_bufferSize = std::max(minBufferForRatio, reqSize * 2);
            }

            // Reset latent terrain cached matrices and stale layer bending states on new model load
            m_latent_engine.reset();
            m_bending_manager.clear_all_layer_bending();
            {
                std::lock_guard<std::mutex> stateLock(m_state_mutex);
                m_activeLayerName = "";
            }

            const double hostSr = getSampleRate();
            if (hostSr > 0.0) {
                m_hostSampleRate.store(hostSr);
            }

            m_modelLoaded.store(true);
            configure_engine();
            return true;
        }
    } catch (const std::exception& e) {
        DBG("Exception loading model: " << e.what());
    } catch (...) {
        DBG("Unknown exception loading model");
    }

    return false;
}

auto NetBenderAudioProcessor::setCurrentMethod(const juce::String& method) -> void {
    if (!m_backend)
        return;
    std::string methodStr = method.toStdString();
    auto modes = m_backend->get_plugin_modes();
    bool isValidMode = (std::find(modes.begin(), modes.end(), methodStr) != modes.end()) ||
                       m_backend->has_method(methodStr);

    if (isValidMode) {
        int prevIn = m_model_in;
        int prevOut = m_model_out;
        {
            std::lock_guard<std::mutex> stateLock(m_state_mutex);
            m_currentMethod = method;
        }
        auto params = m_backend->get_mode_params(methodStr);
        if (params.size() >= 4) {
            m_model_in = params[0];
            m_model_out = params[2];
        }

        if (prevIn != m_model_in || prevOut != m_model_out) {
            configure_engine();
        } else {
            m_engine.set_method(methodStr);
            if (!m_engine.is_worker_running() && m_backend) {
                m_backend->clear_decode_gen_fifo();
            }
        }
    }
}

auto NetBenderAudioProcessor::setBufferSize(int size) -> void {
    if (size > 0 && (size & (size - 1)) == 0) {  // Power of two check
        m_bufferSize = size;
        configure_engine();
    }
}

auto NetBenderAudioProcessor::setUseGpu(bool enabled) -> void {
    if (m_useGpu.load() == enabled && m_modelLoaded.load())
        return;

    m_useGpu.store(enabled);
    if (!m_modelLoaded.load() || m_modelPath.isEmpty()) {
        if (m_backend)
            m_backend->use_gpu(enabled);
        return;
    }

    juce::String savedMethod = getCurrentMethod();
    juce::File targetFile(m_modelPath);

    // For .pte models, automatically swap between sibling *_xnnpack.pte (CPU) and *_coreml.pte
    // (GPU/ANE) if present
    if (m_modelPath.endsWithIgnoreCase(".pte")) {
        juce::String candidatePath = m_modelPath;
        if (enabled && candidatePath.containsIgnoreCase("_xnnpack.pte"))
            candidatePath = candidatePath.replace("_xnnpack.pte", "_coreml.pte", true);
        else if (!enabled && candidatePath.containsIgnoreCase("_coreml.pte"))
            candidatePath = candidatePath.replace("_coreml.pte", "_xnnpack.pte", true);

        juce::File candidateFile(candidatePath);
        if (candidateFile.existsAsFile())
            targetFile = candidateFile;
    }

    if (loadModel(targetFile)) {
        if (savedMethod.isNotEmpty())
            setCurrentMethod(savedMethod);
    }
}

auto NetBenderAudioProcessor::configure_engine() -> void {
    if (!m_modelLoaded.load() || !m_backend || m_model_out <= 0) {
        m_engine.detach_backend();
        setLatencySamples(0);
        return;
    }

    StreamingConfig cfg;
    {
        std::lock_guard<std::mutex> stateLock(m_state_mutex);
        cfg.method = m_currentMethod.toStdString();
    }
    cfg.dispatch = InferenceDispatch::PluginMode;
    cfg.model_in_channels = m_model_in;
    cfg.model_out_channels = m_model_out;
    cfg.n_batches = 1;
    cfg.model_buffer_size = m_bufferSize;
    cfg.host_sample_rate = m_hostSampleRate.load();
    cfg.model_sample_rate = m_backend->get_sample_rate();
    cfg.max_host_block_size = std::max(256, m_maxHostBlockSize.load());
    cfg.max_host_channels =
        std::max(2, std::max(getTotalNumInputChannels(), getTotalNumOutputChannels()));
    cfg.resample_mode = ResampleMode::Auto;
    cfg.resample_quality = ResamplerQuality::CatmullRom;
    cfg.use_worker_thread = true;
    cfg.auto_start_worker = true;
    cfg.enable_dry_delay = true;
    cfg.enable_safety_sentry = true;
    cfg.duplicate_mono_input = true;
    cfg.duplicate_mono_output = true;

    m_engine.configure(m_backend.get(), cfg);
    setLatencySamples(m_engine.get_latency_samples());
}

//==============================================================================
auto NetBenderAudioProcessor::getStateInformation(juce::MemoryBlock& destData) -> void {
    juce::XmlElement xmlState("NetBenderSettings");
    xmlState.setAttribute("modelPath", m_modelPath);
    xmlState.setAttribute("currentMethod", getCurrentMethod());
    xmlState.setAttribute("bufferSize", m_bufferSize);
    xmlState.setAttribute("activeLayer", getActiveLayerName());
    xmlState.setAttribute("dryWet", (double)getDryWet());
    xmlState.setAttribute("triggerMode", (int)getTriggerMode());
    xmlState.setAttribute("envAttack", (double)getEnvelopeAttack());
    xmlState.setAttribute("envRelease", (double)getEnvelopeRelease());
    xmlState.setAttribute("transientThresh", (double)getTransientThreshold());

    auto* layersElement = xmlState.createNewChildElement("Layers");
    auto allStates = m_bending_manager.get_all_layer_states();
    for (const auto& pair : allStates) {
        auto* layerEl = layersElement->createNewChildElement("Layer");
        layerEl->setAttribute("name", pair.first);
        layerEl->setAttribute("scale", (double)pair.second.scale);
        layerEl->setAttribute("offset", (double)pair.second.offset);
        layerEl->setAttribute("heat", (double)pair.second.heat);
        layerEl->setAttribute("memory", (double)pair.second.memory);
        layerEl->setAttribute("driftMode", (int)pair.second.driftMode);
        layerEl->setAttribute("frozen", pair.second.frozen);
    }

    copyXmlToBinary(xmlState, destData);
}

auto NetBenderAudioProcessor::setStateInformation(const void* data, int sizeInBytes) -> void {
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr) {
        if (xmlState->hasTagName("NetBenderSettings") ||
            xmlState->hasTagName("NNBendingSettings")) {
            m_bufferSize = xmlState->getIntAttribute("bufferSize", 2048);
            setDryWet((float)xmlState->getDoubleAttribute("dryWet", 1.0));
            setActiveLayerName(xmlState->getStringAttribute("activeLayer"));
            setTriggerMode((TriggerMode)xmlState->getIntAttribute("triggerMode", 0));
            setEnvelopeAttack((float)xmlState->getDoubleAttribute("envAttack", 0.25));
            setEnvelopeRelease((float)xmlState->getDoubleAttribute("envRelease", 0.85));
            setTransientThreshold((float)xmlState->getDoubleAttribute("transientThresh", 0.15));

            juce::String path = xmlState->getStringAttribute("modelPath");
            if (path.isNotEmpty()) {
                juce::File file(path);
                if (file.existsAsFile())
                    loadModel(file);
            }
            juce::String method = xmlState->getStringAttribute("currentMethod");
            if (method.isNotEmpty())
                setCurrentMethod(method);

            // Restore per-layer bending states
            auto* layersElement = xmlState->getChildByName("Layers");
            if (layersElement != nullptr) {
                for (auto* layerEl : layersElement->getChildIterator()) {
                    std::string name = layerEl->getStringAttribute("name").toStdString();
                    if (!name.empty()) {
                        float scale = (float)layerEl->getDoubleAttribute("scale", 1.0);
                        float offset = (float)layerEl->getDoubleAttribute("offset", 0.0);
                        double heatVal = layerEl->getDoubleAttribute("heat", -1.0);
                        if (heatVal < 0.0)
                            heatVal = layerEl->getDoubleAttribute("jitter", 0.0);
                        float heat = (float)heatVal;
                        float memory = (float)layerEl->getDoubleAttribute("memory", 0.8);
                        auto driftMode = (DriftMode)layerEl->getIntAttribute("driftMode", 0);
                        bool frozen = layerEl->getBoolAttribute("frozen", false);
                        m_bending_manager.restore_layer_state(name, scale, offset, heat, memory,
                                                              driftMode, frozen);
                    }
                }
            }
        }
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
