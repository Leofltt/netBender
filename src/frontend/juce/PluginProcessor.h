#pragma once

#include "IBackend.h"
#include "TorchBackend.h"
#include "BackendFactory.h"
#include "dsp/AudioRingBuffer.h"
#include "core/ModelThread.h"
#include "core/BendingManager.h"
#include "core/LatentTerrainEngine.h"
#include "core/SafetySentry.h"
#include <JuceHeader.h>
#include <mutex>
#include <atomic>
#include <vector>
#include <memory>
#include <string>

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
    auto processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) -> void override;

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

    // Backend methods
    auto loadModel(const juce::File& file) -> bool;
    auto getBackend() -> IBackend& { return *m_backend; }
    auto getTorchBackend() -> TorchBackend* { return dynamic_cast<TorchBackend*>(m_backend.get()); }
    auto isModelLoaded() const -> bool { return m_modelLoaded.load(); }
    auto getModelPath() const -> juce::String { return m_modelPath; }
    auto getCurrentMethod() const -> juce::String { return m_currentMethod; }
    auto setCurrentMethod(const juce::String& method) -> void;
    auto getCurrentMode() const -> juce::String { return m_currentMethod; }
    auto setCurrentMode(const juce::String& mode) -> void { setCurrentMethod(mode); }
    auto getAvailableModes() const -> std::vector<juce::String>;
    auto getBufferSize() const -> int { return m_bufferSize; }
    auto setBufferSize(int size) -> void;
    auto getModelSampleRate() const -> double { return m_modelSampleRate.load(); }
    auto isResamplingActive() const -> bool { return m_resamplingActive.load(); }
    auto isUsingGpu() const -> bool { return m_useGpu.load(); }
    auto setUseGpu(bool enabled) -> void;

    // Latent bending engine
    auto setLatentHook(IBackend::LatentHook hook) -> void { m_latentHook = hook; }
    auto setLatentMode(LatentMode mode) -> void { m_latent_engine.set_mode(mode); }
    auto getLatentMode() const -> LatentMode { return m_latent_engine.get_mode(); }
    auto setLatentEnabled(bool enabled) -> void { m_latent_engine.set_enabled(enabled); }
    auto isLatentEnabled() const -> bool { return m_latent_engine.is_enabled(); }
    auto setLatentCoords(float x, float y) -> void {
        m_latent_engine.set_coords(x, y);
        if (m_latentXParam) *m_latentXParam = m_latent_engine.get_x();
        if (m_latentYParam) *m_latentYParam = m_latent_engine.get_y();
    }
    auto getLatentX() const -> float { return m_latentXParam ? m_latentXParam->get() : m_latent_engine.get_x(); }
    auto getLatentY() const -> float { return m_latentYParam ? m_latentYParam->get() : m_latent_engine.get_y(); }
    auto setLatentDepth(float depth) -> void {
        m_latent_engine.set_depth(depth);
        if (m_latentDepthParam) *m_latentDepthParam = m_latent_engine.get_depth();
    }
    auto getLatentDepth() const -> float { return m_latentDepthParam ? m_latentDepthParam->get() : m_latent_engine.get_depth(); }
    auto setLatentSlew(float slew) -> void { m_latent_engine.set_slew(slew); }
    auto getLatentSlew() const -> float { return m_latent_engine.get_slew(); }
    auto hasAutoencode() -> bool { return m_backend ? m_backend->has_autoencode() : false; }

    // Inference loop
    auto runInference() -> void;

    // Dry / Wet control
    auto getDryWet() const -> float { return m_dryWetParam ? m_dryWetParam->get() : m_dryWet.load(); }
    auto setDryWet(float value) -> void {
        float v = juce::jlimit(0.0f, 1.0f, value);
        m_dryWet.store(v);
        if (m_dryWetParam)
            *m_dryWetParam = v;
    }

    // Weight Bending subsystem (forwards to BendingManager)
    static auto classifyLayer(const std::string& name) -> LayerCategory { return BendingManager::classify_layer(name); }
    auto getLayerState(const std::string& layerName) const -> LayerBendingState { return m_bending_manager.get_layer_state(layerName); }
    auto setLayerState(const std::string& layerName, const LayerBendingState& state) -> void { m_bending_manager.set_layer_state(layerName, state); }
    auto setLayerScale(const std::string& layerName, float scale) -> void { m_bending_manager.set_layer_scale(layerName, scale); }
    auto setLayerOffset(const std::string& layerName, float offset) -> void { m_bending_manager.set_layer_offset(layerName, offset); }
    auto setLayerHeat(const std::string& layerName, float heat) -> void { m_bending_manager.set_layer_heat(layerName, heat); }
    auto setLayerJitter(const std::string& layerName, float jitter) -> void { m_bending_manager.set_layer_jitter(layerName, jitter); }
    auto setLayerMemory(const std::string& layerName, float memory) -> void { m_bending_manager.set_layer_memory(layerName, memory); }
    auto setLayerDriftMode(const std::string& layerName, DriftMode mode) -> void { m_bending_manager.set_layer_drift_mode(layerName, mode); }
    auto setLayerFrozen(const std::string& layerName, bool frozen) -> void { m_bending_manager.set_layer_frozen(layerName, frozen); }
    auto setLayerBridgeSource(const std::string& layerName, const std::string& sourceLayer) -> void { m_bending_manager.set_layer_bridge_source(layerName, sourceLayer); }
    auto setLayerBridgeDepth(const std::string& layerName, float depth) -> void { m_bending_manager.set_layer_bridge_depth(layerName, depth); }
    auto setLayerDrawnWeights(const std::string& layerName, const std::vector<float>& weights) -> void { m_bending_manager.set_layer_drawn_weights(layerName, weights); }
    auto setLayerBaseWeights(const std::string& layerName, const std::vector<float>& weights) -> void { m_bending_manager.set_layer_base_weights(layerName, weights); }
    auto setLayerOriginalWeights(const std::string& layerName, const std::vector<float>& weights) -> void { m_bending_manager.set_layer_original_weights(layerName, weights); }
    auto clearLayerBending(const std::string& layerName) -> void { m_bending_manager.clear_layer_bending(layerName); }
    auto clearAllLayerBending() -> void { m_bending_manager.clear_all_layer_bending(); }
    auto hasActiveBending() const -> bool { return m_bending_manager.has_active_bending(); }

    // Safety Sentry & Blown Fuse Control
    auto isFuseBlown() const -> bool { return m_safety_sentry.is_fuse_blown(); }
    auto resetFuse() -> void {
        m_safety_sentry.reset_fuse();
        m_bending_manager.clear_all_layer_bending();
        if (m_backend)
            m_backend->reset_all_layer_weights();
    }
    auto setAutoResetFuse(bool enabled) -> void { m_safety_sentry.set_auto_reset(enabled); }
    auto getAutoResetFuse() const -> bool { return m_safety_sentry.get_auto_reset(); }

    // Momentary Short-Circuit Gate Control
    auto triggerShortCircuit(bool active) -> void { m_bending_manager.trigger_short_circuit(active); }
    auto isShortCircuitActive() const -> bool { return m_bending_manager.is_short_circuit_active(); }
    auto getMomentaryEnvelope() const -> float { return m_bending_manager.get_momentary_envelope(); }
    auto setTriggerMode(TriggerMode mode) -> void { m_bending_manager.set_trigger_mode(mode); }
    auto getTriggerMode() const -> TriggerMode { return m_bending_manager.get_trigger_mode(); }
    auto setEnvelopeAttack(float att) -> void { m_bending_manager.set_envelope_attack(att); }
    auto getEnvelopeAttack() const -> float { return m_bending_manager.get_envelope_attack(); }
    auto setEnvelopeRelease(float rel) -> void { m_bending_manager.set_envelope_release(rel); }
    auto getEnvelopeRelease() const -> float { return m_bending_manager.get_envelope_release(); }
    auto setTransientThreshold(float th) -> void { m_bending_manager.set_transient_threshold(th); }
    auto getTransientThreshold() const -> float { return m_bending_manager.get_transient_threshold(); }

    // Active UI layer target for DAW automation mapping
    auto setActiveLayerName(const juce::String& name) -> void { m_activeLayerName = name; }
    auto getActiveLayerName() const -> juce::String { return m_activeLayerName; }

    // Subsystem accessors
    auto getBendingManager() -> BendingManager& { return m_bending_manager; }
    auto getLatentEngine() -> LatentTerrainEngine& { return m_latent_engine; }
    auto getSafetySentry() -> SafetySentry& { return m_safety_sentry; }

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
    friend class ModelThread;

    std::unique_ptr<IBackend> m_backend;
    IBackend::LatentHook m_latentHook { nullptr };
    std::atomic<bool> m_modelLoaded { false };
    std::atomic<bool> m_useGpu { false };
    juce::String m_modelPath;
    juce::String m_currentMethod;
    int m_bufferSize { 2048 };
    int m_model_in { 0 };
    int m_model_out { 0 };

    std::atomic<float> m_dryWet { 1.0f };

    // Subcomponents
    BendingManager       m_bending_manager;
    LatentTerrainEngine  m_latent_engine;
    SafetySentry         m_safety_sentry;
    ModelThread          m_model_thread;

    // DAW Automatable parameters
    juce::AudioParameterFloat* m_dryWetParam { nullptr };
    juce::AudioParameterFloat* m_scaleParam { nullptr };
    juce::AudioParameterFloat* m_offsetParam { nullptr };
    juce::AudioParameterFloat* m_heatParam { nullptr };
    juce::AudioParameterFloat* m_memoryParam { nullptr };
    juce::AudioParameterBool*  m_shortCircuitParam { nullptr };
    juce::AudioParameterFloat* m_latentXParam { nullptr };
    juce::AudioParameterFloat* m_latentYParam { nullptr };
    juce::AudioParameterFloat* m_latentDepthParam { nullptr };

    juce::String m_activeLayerName;

    // Buffers and synchronization
    std::vector<AudioRingBuffer> m_in_buffers;
    std::vector<AudioRingBuffer> m_out_buffers;
    std::vector<AudioRingBuffer> m_dry_delay_buffers;

    std::vector<std::vector<float>> m_staging_in;
    std::vector<std::vector<float>> m_staging_out;

    std::vector<std::vector<float>> m_worker_in;
    std::vector<std::vector<float>> m_worker_out;

    std::mutex m_staging_mutex;
    std::atomic<bool> m_output_ready { false };

    // Resampling infrastructure (Host SR <-> Model SR)
    std::atomic<double> m_hostSampleRate { 44100.0 };
    std::atomic<double> m_modelSampleRate { 44100.0 };
    std::atomic<bool>   m_resamplingActive { false };

    std::vector<juce::CatmullRomInterpolator> m_input_resamplers;
    std::vector<juce::CatmullRomInterpolator> m_output_resamplers;
    std::vector<std::vector<float>> m_resampled_in_scratch;
    std::vector<std::vector<float>> m_resampled_out_scratch;
    std::vector<float> m_dryBlockScratch;

    void initBuffers();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NetBenderAudioProcessor)
};
