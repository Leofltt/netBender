#pragma once

#include "IBackend.h"
#include "BackendFactory.h"
#include "engine/StreamingEngine.h"
#include "bending/BendingManager.h"
#include "bending/LatentTerrainEngine.h"
#include "engine/NetBenderSession.h"
#include <JuceHeader.h>
#include <mutex>
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include "rtsan_utils.h"

class NetBenderAudioProcessor : public juce::AudioProcessor {
public:
    using LayerCategory = ::LayerCategory;
    using DriftMode = ::DriftMode;
    using TriggerMode = ::TriggerMode;
    using LayerBendingState = ::LayerBendingState;
    using LatentMode = ::LatentMode;

    NetBenderAudioProcessor();
    ~NetBenderAudioProcessor() override;

    auto prepareToPlay(double sampleRate, int samplesPerBlock) -> void override;
    auto releaseResources() -> void override;
    auto isBusesLayoutSupported(const BusesLayout& layouts) const -> bool override;
    auto processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) NETBENDER_NONBLOCKING
        -> void override;

    auto createEditor() -> juce::AudioProcessorEditor* override;
    auto hasEditor() const -> bool override { return true; }

    auto getName() const -> const juce::String override { return "NetBender"; }
    auto acceptsMidi() const -> bool override { return false; }
    auto producesMidi() const -> bool override { return false; }
    auto isMidiEffect() const -> bool override { return false; }
    auto getTailLengthSeconds() const -> double override { return 0.0; }

    auto getNumPrograms() -> int override { return 1; }
    auto getCurrentProgram() -> int override { return 0; }
    auto setCurrentProgram(int index) -> void override {}
    auto getProgramName(int index) -> const juce::String override { return {}; }
    auto changeProgramName(int index, const juce::String& newName) -> void override {}

    auto getStateInformation(juce::MemoryBlock& destData) -> void override;
    auto setStateInformation(const void* data, int sizeInBytes) -> void override;

    // Backend methods (forward to NetBenderSession)
    auto loadModel(const juce::File& file) -> bool;
    auto getBackend() -> IBackend& { return m_session.backend(); }
    auto isModelLoaded() const -> bool { return m_session.is_model_loaded(); }
    auto getModelPath() const -> juce::String { return juce::String(m_session.get_model_path()); }
    auto getCurrentMethod() const -> juce::String { return juce::String(m_session.get_method()); }
    auto setCurrentMethod(const juce::String& method) -> void {
        m_session.set_method(method.toStdString());
    }
    auto getCurrentMode() const -> juce::String { return getCurrentMethod(); }
    auto setCurrentMode(const juce::String& mode) -> void { setCurrentMethod(mode); }
    auto getAvailableModes() const -> std::vector<juce::String>;
    auto getBufferSize() const -> int { return m_session.get_buffer_size(); }
    auto setBufferSize(int size) -> void { m_session.set_buffer_size(size); }
    auto getModelSampleRate() const -> double { return m_session.get_model_sample_rate(); }
    auto isResamplingActive() const -> bool { return m_session.is_resampling_active(); }
    auto isUsingGpu() const -> bool { return m_session.is_using_gpu(); }
    auto setUseGpu(bool enabled) -> void { m_session.set_use_gpu(enabled); }

    // Latent bending engine
    auto setLatentHook(IBackend::LatentHook hook) -> void {
        m_session.set_latent_hook(std::move(hook));
    }
    auto setLatentMode(LatentMode mode) -> void { m_session.latent().set_mode(mode); }
    auto getLatentMode() const -> LatentMode { return m_session.latent().get_mode(); }
    auto setLatentEnabled(bool enabled) -> void { m_session.latent().set_enabled(enabled); }
    auto isLatentEnabled() const -> bool { return m_session.latent().is_enabled(); }
    auto setLatentCoords(float x, float y) -> void {
        m_session.latent().set_coords(x, y);
        if (m_latentXParam)
            *m_latentXParam = m_session.latent().get_x();
        if (m_latentYParam)
            *m_latentYParam = m_session.latent().get_y();
    }
    auto getLatentX() const -> float {
        return m_latentXParam ? m_latentXParam->get() : m_session.latent().get_x();
    }
    auto getLatentY() const -> float {
        return m_latentYParam ? m_latentYParam->get() : m_session.latent().get_y();
    }
    auto setLatentDepth(float depth) -> void {
        m_session.latent().set_depth(depth);
        if (m_latentDepthParam)
            *m_latentDepthParam = m_session.latent().get_depth();
    }
    auto getLatentDepth() const -> float {
        return m_latentDepthParam ? m_latentDepthParam->get() : m_session.latent().get_depth();
    }
    auto setLatentSlew(float slew) -> void { m_session.latent().set_slew(slew); }
    auto getLatentSlew() const -> float { return m_session.latent().get_slew(); }
    auto hasAutoencode() -> bool {
        return m_session.has_backend() ? m_session.backend().has_autoencode() : false;
    }

    // Dry / Wet control
    auto getDryWet() const -> float {
        return m_dryWetParam ? m_dryWetParam->get() : m_session.get_dry_wet();
    }
    auto setDryWet(float value) -> void {
        float v = juce::jlimit(0.0f, 1.0f, value);
        m_session.set_dry_wet(v);
        if (m_dryWetParam)
            *m_dryWetParam = v;
    }

    // Weight Bending subsystem (forwards to the session's BendingManager)
    static auto classifyLayer(const std::string& name) -> LayerCategory {
        return BendingManager::classify_layer(name);
    }
    auto getLayerState(const std::string& layerName) const -> LayerBendingState {
        return m_session.bending().get_layer_state(layerName);
    }
    auto setLayerState(const std::string& layerName, const LayerBendingState& state) -> void {
        m_session.bending().set_layer_state(layerName, state);
    }
    auto setLayerScale(const std::string& layerName, float scale) -> void {
        m_session.bending().set_layer_scale(layerName, scale);
        if (m_scaleParam && layerName == m_session.get_active_layer())
            *m_scaleParam = scale;
    }
    auto setLayerOffset(const std::string& layerName, float offset) -> void {
        m_session.bending().set_layer_offset(layerName, offset);
        if (m_offsetParam && layerName == m_session.get_active_layer())
            *m_offsetParam = offset;
    }
    auto setLayerHeat(const std::string& layerName, float heat) -> void {
        m_session.bending().set_layer_heat(layerName, heat);
        if (m_heatParam && layerName == m_session.get_active_layer())
            *m_heatParam = heat;
    }
    auto setLayerJitter(const std::string& layerName, float jitter) -> void {
        setLayerHeat(layerName, jitter);
    }
    auto setLayerMemory(const std::string& layerName, float memory) -> void {
        m_session.bending().set_layer_memory(layerName, memory);
        if (m_memoryParam && layerName == m_session.get_active_layer())
            *m_memoryParam = memory;
    }
    auto setLayerDriftMode(const std::string& layerName, DriftMode mode) -> void {
        m_session.bending().set_layer_drift_mode(layerName, mode);
    }
    auto setLayerFrozen(const std::string& layerName, bool frozen) -> void {
        m_session.bending().set_layer_frozen(layerName, frozen);
    }
    auto setLayerBridgeSource(const std::string& layerName, const std::string& sourceLayer)
        -> void {
        m_session.bending().set_layer_bridge_source(layerName, sourceLayer);
    }
    auto setLayerBridgeDepth(const std::string& layerName, float depth) -> void {
        m_session.bending().set_layer_bridge_depth(layerName, depth);
    }
    auto setLayerDrawnWeights(const std::string& layerName, const std::vector<float>& weights)
        -> void {
        m_session.bending().set_layer_drawn_weights(layerName, weights);
    }
    auto setLayerBaseWeights(const std::string& layerName, const std::vector<float>& weights)
        -> void {
        m_session.bending().set_layer_base_weights(layerName, weights);
    }
    auto setLayerOriginalWeights(const std::string& layerName, const std::vector<float>& weights)
        -> void {
        m_session.bending().set_layer_original_weights(layerName, weights);
    }
    auto clearLayerBending(const std::string& layerName) -> void {
        m_session.clear_layer_bending(layerName);
    }
    auto clearAllLayerBending() -> void { m_session.clear_all_layer_bending(); }
    auto hasActiveBending() const -> bool { return m_session.bending().has_active_bending(); }

    // Safety Sentry & Blown Fuse Control
    auto isFuseBlown() const -> bool { return m_session.safety_sentry().is_fuse_blown(); }
    auto resetFuse() -> void { m_session.reset_fuse(); }
    auto setAutoResetFuse(bool enabled) -> void {
        m_session.safety_sentry().set_auto_reset(enabled);
    }
    auto getAutoResetFuse() const -> bool { return m_session.safety_sentry().get_auto_reset(); }

    // Momentary Short-Circuit Gate Control
    auto triggerShortCircuit(bool active) noexcept NETBENDER_NONBLOCKING -> void {
        m_session.trigger_short_circuit(active);
    }
    auto isShortCircuitActive() const -> bool {
        return m_session.bending().is_short_circuit_active();
    }
    auto getMomentaryEnvelope() const -> float {
        return m_session.bending().get_momentary_envelope();
    }
    auto setTriggerMode(TriggerMode mode) -> void { m_session.bending().set_trigger_mode(mode); }
    auto getTriggerMode() const -> TriggerMode { return m_session.bending().get_trigger_mode(); }
    auto setEnvelopeAttack(float att) -> void { m_session.bending().set_envelope_attack(att); }
    auto getEnvelopeAttack() const -> float { return m_session.bending().get_envelope_attack(); }
    auto setEnvelopeRelease(float rel) -> void { m_session.bending().set_envelope_release(rel); }
    auto getEnvelopeRelease() const -> float { return m_session.bending().get_envelope_release(); }
    auto setTransientThreshold(float th) -> void {
        m_session.bending().set_transient_threshold(th);
    }
    auto getTransientThreshold() const -> float {
        return m_session.bending().get_transient_threshold();
    }

    // Active UI layer target for DAW automation mapping
    auto setActiveLayerName(const juce::String& name) -> void {
        m_session.set_active_layer(name.toStdString());
    }
    auto getActiveLayerName() const -> juce::String {
        return juce::String(m_session.get_active_layer());
    }

    // Subsystem accessors
    auto getSession() -> NetBenderSession& { return m_session; }
    auto getSession() const -> const NetBenderSession& { return m_session; }
    auto getStateRevision() const -> uint64_t { return m_stateRevision.load(); }
    auto getBendingManager() -> BendingManager& { return m_session.bending(); }
    auto getLatentEngine() -> LatentTerrainEngine& { return m_session.latent(); }
    auto getSafetySentry() -> SafetySentry& { return m_session.safety_sentry(); }
    auto getSafetySentry() const -> const SafetySentry& { return m_session.safety_sentry(); }

    // Automatable parameters accessors
    auto getDryWetParam() const -> juce::AudioParameterFloat* { return m_dryWetParam; }
    auto getScaleParam() const -> juce::AudioParameterFloat* { return m_scaleParam; }
    auto getOffsetParam() const -> juce::AudioParameterFloat* { return m_offsetParam; }
    auto getHeatParam() const -> juce::AudioParameterFloat* { return m_heatParam; }
    auto getJitterParam() const -> juce::AudioParameterFloat* { return m_heatParam; }
    auto getMemoryParam() const -> juce::AudioParameterFloat* { return m_memoryParam; }
    auto getShortCircuitParam() const -> juce::AudioParameterBool* { return m_shortCircuitParam; }
    auto getLatentXParam() const -> juce::AudioParameterFloat* { return m_latentXParam; }
    auto getLatentYParam() const -> juce::AudioParameterFloat* { return m_latentYParam; }
    auto getLatentDepthParam() const -> juce::AudioParameterFloat* { return m_latentDepthParam; }

private:
    // DAW Automatable parameters
    juce::AudioParameterFloat* m_dryWetParam{nullptr};
    juce::AudioParameterFloat* m_scaleParam{nullptr};
    juce::AudioParameterFloat* m_offsetParam{nullptr};
    juce::AudioParameterFloat* m_heatParam{nullptr};
    juce::AudioParameterFloat* m_memoryParam{nullptr};
    juce::AudioParameterBool* m_shortCircuitParam{nullptr};
    juce::AudioParameterFloat* m_latentXParam{nullptr};
    juce::AudioParameterFloat* m_latentYParam{nullptr};
    juce::AudioParameterFloat* m_latentDepthParam{nullptr};

    // Frontend-agnostic backend + streaming + bending + latent + JSON state facade
    NetBenderSession m_session;
    std::atomic<uint64_t> m_stateRevision{0};

    auto syncParametersFromSession() -> void;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NetBenderAudioProcessor)
};
