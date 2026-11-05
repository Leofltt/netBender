#pragma once
#include "backend.h"
#include "BackendFactory.h"
#include <JuceHeader.h>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <memory>

class CircularBuffer {
public:
    CircularBuffer() = default;
    
    CircularBuffer(CircularBuffer&& other) noexcept
    {
        m_buffer = std::move(other.m_buffer);
        m_capacity = other.m_capacity;
        m_mask = other.m_mask;
        m_writeIndex = other.m_writeIndex;
        m_readIndex = other.m_readIndex;
        m_available.store(other.m_available.load());

        other.m_capacity = 0;
        other.m_mask = 0;
        other.m_writeIndex = 0;
        other.m_readIndex = 0;
        other.m_available.store(0);
    }
    
    CircularBuffer& operator=(CircularBuffer&& other) noexcept
    {
        if (this != &other)
        {
            m_buffer = std::move(other.m_buffer);
            m_capacity = other.m_capacity;
            m_mask = other.m_mask;
            m_writeIndex = other.m_writeIndex;
            m_readIndex = other.m_readIndex;
            m_available.store(other.m_available.load());

            other.m_capacity = 0;
            other.m_mask = 0;
            other.m_writeIndex = 0;
            other.m_readIndex = 0;
            other.m_available.store(0);
        }
        return *this;
    }
    
    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;

    auto init(int size) -> void {
        // Enforce power-of-two capacity for fast bitwise wrap-around
        int target = std::max(size * 4, 8192);
        m_capacity = 1;
        while (m_capacity < target)
            m_capacity <<= 1;
        m_mask = m_capacity - 1;
        m_buffer.assign(m_capacity, 0.0f);
        m_writeIndex = 0;
        m_readIndex = 0;
        m_available.store(0);
    }
    
    auto put(const float* data, int numSamples) -> void {
        if (m_capacity <= 0 || numSamples <= 0 || m_buffer.empty()) return;
        int toWrite = std::min(numSamples, m_capacity);

        if (data != nullptr) {
            int firstPart = std::min(toWrite, m_capacity - m_writeIndex);
            std::memcpy(m_buffer.data() + m_writeIndex, data, firstPart * sizeof(float));
            if (toWrite > firstPart) {
                std::memcpy(m_buffer.data(), data + firstPart, (toWrite - firstPart) * sizeof(float));
            }
        } else {
            int firstPart = std::min(toWrite, m_capacity - m_writeIndex);
            std::memset(m_buffer.data() + m_writeIndex, 0, firstPart * sizeof(float));
            if (toWrite > firstPart) {
                std::memset(m_buffer.data(), 0, (toWrite - firstPart) * sizeof(float));
            }
        }

        m_writeIndex = (m_writeIndex + toWrite) & m_mask;
        int prevAvail = m_available.fetch_add(toWrite);
        if (prevAvail + toWrite > m_capacity) {
            // Buffer overflow: clamp available and advance readIndex past overwritten samples
            int overflow = (prevAvail + toWrite) - m_capacity;
            m_readIndex = (m_readIndex + overflow) & m_mask;
            m_available.store(m_capacity);
        }
    }
    
    auto get(float* dest, int numSamples) -> void {
        if (m_capacity <= 0 || numSamples <= 0 || dest == nullptr) return;
        int avail = std::min(m_available.load(), m_capacity);
        int toRead = std::min(numSamples, avail);

        if (toRead > 0 && !m_buffer.empty()) {
            int firstPart = std::min(toRead, m_capacity - m_readIndex);
            std::memcpy(dest, m_buffer.data() + m_readIndex, firstPart * sizeof(float));
            if (toRead > firstPart) {
                std::memcpy(dest + firstPart, m_buffer.data(), (toRead - firstPart) * sizeof(float));
            }
            m_readIndex = (m_readIndex + toRead) & m_mask;
            m_available.fetch_sub(toRead);
        }

        if (toRead < numSamples) {
            std::memset(dest + toRead, 0, (numSamples - toRead) * sizeof(float));
        }
    }
    
    auto getAvailable() const -> int { return std::min(m_available.load(), m_capacity); }
    
    auto peek(float* dest, int numSamples) const -> int {
        if (m_capacity <= 0 || numSamples <= 0 || dest == nullptr || m_buffer.empty()) return 0;
        int avail = std::min(m_available.load(), m_capacity);
        int toRead = std::min(numSamples, avail);

        if (toRead > 0) {
            int firstPart = std::min(toRead, m_capacity - m_readIndex);
            std::memcpy(dest, m_buffer.data() + m_readIndex, firstPart * sizeof(float));
            if (toRead > firstPart) {
                std::memcpy(dest + firstPart, m_buffer.data(), (toRead - firstPart) * sizeof(float));
            }
        }
        return toRead;
    }

    auto drop(int numSamples) -> void {
        if (m_capacity <= 0 || numSamples <= 0) return;
        int avail = std::min(m_available.load(), m_capacity);
        int toDrop = std::min(numSamples, avail);
        m_readIndex = (m_readIndex + toDrop) & m_mask;
        m_available.fetch_sub(toDrop);
    }
    
    auto clear() -> void {
        if (m_capacity > 0)
            std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
        m_writeIndex = 0;
        m_readIndex = 0;
        m_available.store(0);
    }
    
private:
    std::vector<float> m_buffer;
    int m_capacity { 0 };
    int m_mask { 0 };
    int m_writeIndex { 0 };
    int m_readIndex { 0 };
    std::atomic<int> m_available { 0 };
};

class NNBendingAudioProcessor;

class ModelThread : public juce::Thread {
public:
    ModelThread(NNBendingAudioProcessor& processor);
    ~ModelThread() override;
    
    auto run() -> void override;
    auto triggerCompute() -> void;
    auto stop() -> void;
    auto isProcessing() const -> bool { return m_processing.load(); }
    
private:
    NNBendingAudioProcessor& m_processor;
    juce::WaitableEvent m_event;
    std::atomic<bool> m_processing { false };
};

class NNBendingAudioProcessor : public juce::AudioProcessor {
public:
    NNBendingAudioProcessor();
    ~NNBendingAudioProcessor() override;

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

    // Latent bending hook (applied during autoencode mode)
    auto setLatentHook(IBackend::LatentHook hook) -> void { m_latentHook = hook; }

    enum class LatentMode
    {
        FourierOrbit = 0, // 2D XY mapped via Random Fourier Features into multidimensional latent space
        LatentSlew,       // Temporal inertia / lowpass filter on latent activations
        Quantize          // Latent space bitcrushing / discretization
    };

    auto setLatentMode(LatentMode mode) -> void { m_latentMode.store(mode); }
    auto getLatentMode() const -> LatentMode { return m_latentMode.load(); }

    auto setLatentEnabled(bool enabled) -> void { m_latentEnabled.store(enabled); }
    auto isLatentEnabled() const -> bool { return m_latentEnabled.load(); }

    auto setLatentCoords(float x, float y) -> void {
        m_latentX.store(juce::jlimit(-1.0f, 1.0f, x));
        m_latentY.store(juce::jlimit(-1.0f, 1.0f, y));
        if (m_latentXParam) *m_latentXParam = m_latentX.load();
        if (m_latentYParam) *m_latentYParam = m_latentY.load();
    }
    auto getLatentX() const -> float { return m_latentXParam ? m_latentXParam->get() : m_latentX.load(); }
    auto getLatentY() const -> float { return m_latentYParam ? m_latentYParam->get() : m_latentY.load(); }

    auto setLatentDepth(float depth) -> void {
        float d = juce::jlimit(0.0f, 2.0f, depth);
        m_latentDepth.store(d);
        if (m_latentDepthParam) *m_latentDepthParam = d;
    }
    auto getLatentDepth() const -> float { return m_latentDepthParam ? m_latentDepthParam->get() : m_latentDepth.load(); }

    auto setLatentSlew(float slew) -> void { m_latentSlew.store(juce::jlimit(0.0f, 0.999f, slew)); }
    auto getLatentSlew() const -> float { return m_latentSlew.load(); }

    auto hasAutoencode() -> bool { return m_backend ? m_backend->has_autoencode() : false; }

    // Thread communication
    auto runInference() -> void;

    // Dry / Wet control
    auto getDryWet() const -> float { return m_dryWetParam ? m_dryWetParam->get() : m_dryWet.load(); }
    auto setDryWet(float value) -> void {
        float v = juce::jlimit(0.0f, 1.0f, value);
        m_dryWet.store(v);
        if (m_dryWetParam)
            *m_dryWetParam = v;
    }

    // Layer classification for Trace Isolation
    enum class LayerCategory
    {
        All = 0,
        Norm,       // LayerNorm, BatchNorm, RMSNorm, weight_g, scale/shift
        Conv,       // Convolution kernels (Conv1d, ConvTranspose1d, ConvNeXt)
        Attention,  // Multi-Head Self/Cross-Attention, QKV projections
        Linear,     // Dense / linear weights
        Activation, // Snake, SnakeBeta, Siren periodic parameters (alpha, beta scalars)
        Recurrent,  // LSTM / GRU weights (Encodec)
        Bias,       // All bias vectors
        Other       // Any remaining tensor parameters
    };

    static auto classifyLayer(const std::string& name) -> LayerCategory {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("norm") != std::string::npos || lower.find("weight_g") != std::string::npos || lower.find("gamma") != std::string::npos)
            return LayerCategory::Norm;
        if (lower.find("bias") != std::string::npos)
            return LayerCategory::Bias;
        if (lower.find("snake") != std::string::npos || lower.find("act") != std::string::npos || lower.find("alpha") != std::string::npos)
            return LayerCategory::Activation;
        if (lower.find("attn") != std::string::npos || lower.find("attention") != std::string::npos || lower.find("qkv") != std::string::npos || lower.find("self_attn") != std::string::npos)
            return LayerCategory::Attention;
        if (lower.find("lstm") != std::string::npos || lower.find("gru") != std::string::npos || lower.find("rnn") != std::string::npos)
            return LayerCategory::Recurrent;
        if (lower.find("conv") != std::string::npos)
            return LayerCategory::Conv;
        if (lower.find("linear") != std::string::npos || lower.find("dense") != std::string::npos || lower.find("fc") != std::string::npos || lower.find("proj") != std::string::npos)
            return LayerCategory::Linear;
        return LayerCategory::Other;
    }

    // Stochastic Drift Modes
    enum class DriftMode
    {
        Noise = 0,    // Classic independent uniform/Gaussian jitter per block
        ThermalOU,    // Ornstein-Uhlenbeck mean-reverting thermal drift
        RandomWalk    // Continuous wandering walk
    };

    // Per-Layer Bending State
    struct LayerBendingState
    {
        float scale { 1.0f };
        float offset { 0.0f };
        float heat { 0.0f };         // Thermal volatility / Jitter depth (sigma)
        float memory { 0.8f };       // Mean reversion drag / elastic pull (theta)
        DriftMode driftMode { DriftMode::ThermalOU };
        bool frozen { false };
        std::string bridgeSourceLayer;      // Target cross-talk source layer name
        float bridgeDepth { 0.0f };         // Cross-talk blend ratio alpha (0.0 = off, 1.0 = 100% source)
        std::vector<float> originalWeights; // Unbent baseline weights W0
        std::vector<float> drawnWeights;    // User-drawn / bent target curve
        std::vector<float> driftOffsets;    // Current stochastic drift vector
        std::vector<float> scratchBridged;  // Persistent scratch buffer for cross-talk composite
        std::vector<float> scratchFinal;    // Persistent scratch buffer for bent weight upload
        bool isApplied { false };           // True if bent weights currently applied to model
        float lastAppliedEnv { -1.0f };     // Previous momentary envelope value applied
    };

    auto getLayerState(const std::string& layerName) const -> LayerBendingState {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        auto it = m_layerStates.find(layerName);
        if (it != m_layerStates.end())
            return it->second;
        return LayerBendingState();
    }

    auto setLayerState(const std::string& layerName, const LayerBendingState& state) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName] = state;
    }

    auto setLayerScale(const std::string& layerName, float scale) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].scale = scale;
        m_layerStates[layerName].isApplied = false;
    }

    auto setLayerOffset(const std::string& layerName, float offset) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].offset = offset;
        m_layerStates[layerName].isApplied = false;
    }

    auto setLayerHeat(const std::string& layerName, float heat) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].heat = std::max(0.0f, heat);
    }
    auto setLayerJitter(const std::string& layerName, float jitter) -> void { setLayerHeat(layerName, jitter); }

    auto setLayerMemory(const std::string& layerName, float memory) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].memory = juce::jlimit(0.0f, 1.0f, memory);
    }

    auto setLayerDriftMode(const std::string& layerName, DriftMode mode) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].driftMode = mode;
    }

    auto setLayerFrozen(const std::string& layerName, bool frozen) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].frozen = frozen;
    }

    auto setLayerBridgeSource(const std::string& layerName, const std::string& sourceLayer) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].bridgeSourceLayer = sourceLayer;
        m_layerStates[layerName].isApplied = false;
    }

    auto setLayerBridgeDepth(const std::string& layerName, float depth) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].bridgeDepth = juce::jlimit(0.0f, 1.0f, depth);
        m_layerStates[layerName].isApplied = false;
    }

    auto setLayerDrawnWeights(const std::string& layerName, const std::vector<float>& weights) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].drawnWeights = weights;
        m_layerStates[layerName].isApplied = false;
    }

    auto setLayerBaseWeights(const std::string& layerName, const std::vector<float>& weights) -> void {
        setLayerDrawnWeights(layerName, weights);
    }

    auto setLayerOriginalWeights(const std::string& layerName, const std::vector<float>& weights) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates[layerName].originalWeights = weights;
    }

    auto clearLayerBending(const std::string& layerName) -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates.erase(layerName);
    }

    auto clearAllLayerBending() -> void {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        m_layerStates.clear();
    }

    auto hasActiveBending() const -> bool {
        std::lock_guard<std::mutex> lock(m_layerStateMutex);
        if (m_layerStates.empty()) return false;
        for (const auto& pair : m_layerStates) {
            const auto& s = pair.second;
            if (std::abs(s.scale - 1.0f) > 0.001f || std::abs(s.offset) > 0.001f || s.heat > 0.001f || s.frozen
                || (s.bridgeDepth > 0.001f && !s.bridgeSourceLayer.empty()))
                return true;
        }
        return false;
    }

    // Safety Sentry & Blown Fuse Control
    auto isFuseBlown() const -> bool { return m_blownFuse.load(); }
    auto resetFuse() -> void {
        m_blownFuse.store(false);
        clearAllLayerBending();
        if (m_backend)
            m_backend->reset_all_layer_weights();
    }
    auto setAutoResetFuse(bool enabled) -> void { m_autoResetFuse.store(enabled); }
    auto getAutoResetFuse() const -> bool { return m_autoResetFuse.load(); }

    // Bending Trigger Modes
    enum class TriggerMode
    {
        Continuous = 0, // Default: Always active / continuous bending
        Momentary,      // UI "Short" Button / Parameter hold
        Transient       // Audio sidechain transient follower
    };

    // Momentary Short-Circuit Gate Control
    auto triggerShortCircuit(bool active) -> void { m_shortCircuitActive.store(active); }
    auto isShortCircuitActive() const -> bool { return m_shortCircuitActive.load(); }
    auto getMomentaryEnvelope() const -> float { return m_momentaryEnvelope.load(); }

    auto setTriggerMode(TriggerMode mode) -> void { m_triggerMode.store(mode); }
    auto getTriggerMode() const -> TriggerMode { return m_triggerMode.load(); }

    auto setEnvelopeAttack(float att) -> void { m_envAttack.store(juce::jlimit(0.01f, 1.0f, att)); }
    auto getEnvelopeAttack() const -> float { return m_envAttack.load(); }

    auto setEnvelopeRelease(float rel) -> void { m_envRelease.store(juce::jlimit(0.5f, 0.999f, rel)); }
    auto getEnvelopeRelease() const -> float { return m_envRelease.load(); }

    auto setTransientThreshold(float th) -> void { m_transientThreshold.store(juce::jlimit(0.01f, 1.0f, th)); }
    auto getTransientThreshold() const -> float { return m_transientThreshold.load(); }

    // Active UI layer target for DAW automation mapping
    auto setActiveLayerName(const juce::String& name) -> void { m_activeLayerName = name; }
    auto getActiveLayerName() const -> juce::String { return m_activeLayerName; }

    // Automatable parameters accessors
    auto getDryWetParam() const -> juce::AudioParameterFloat* { return m_dryWetParam; }
    auto getScaleParam() const -> juce::AudioParameterFloat* { return m_scaleParam; }
    auto getOffsetParam() const -> juce::AudioParameterFloat* { return m_offsetParam; }
    auto getHeatParam() const -> juce::AudioParameterFloat* { return m_heatParam; }
    auto getJitterParam() const -> juce::AudioParameterFloat* { return m_heatParam; } // alias
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

    std::atomic<float> m_dryWet { 1.0f }; // 0.0 = Dry, 1.0 = Wet

    // Latent Terrain State
    std::atomic<bool>        m_latentEnabled { false };
    std::atomic<LatentMode>  m_latentMode { LatentMode::FourierOrbit };
    std::atomic<float>       m_latentX { 0.0f };
    std::atomic<float>       m_latentY { 0.0f };
    std::atomic<float>       m_latentDepth { 0.5f };
    std::atomic<float>       m_latentSlew { 0.0f };
    std::vector<float>       m_prevLatentBuffer;
    std::vector<float>       m_rffMatrix; // Random Fourier Feature projection matrix B [D/2 * 2]
    int                      m_cachedLatentDim { 0 };
    void applyLatentTerrainHook(IBackend::LatentBuffer& buf);

    // Safety Sentry state
    std::atomic<bool> m_blownFuse { false };
    std::atomic<bool> m_autoResetFuse { false };
    std::atomic<int>  m_fuseCooldownBlocks { 0 };

    // Momentary Glitch state
    std::atomic<bool>        m_shortCircuitActive { false };
    std::atomic<float>       m_momentaryEnvelope { 0.0f };
    std::atomic<TriggerMode> m_triggerMode { TriggerMode::Continuous };
    std::atomic<float>       m_envAttack { 0.25f };   // Fast ramp-in step (0.01 - 1.0)
    std::atomic<float>       m_envRelease { 0.85f };  // Exponential relaxation multiplier (0.5 - 0.999)
    std::atomic<float>       m_transientThreshold { 0.15f }; // Sidechain audio envelope follower threshold
    std::atomic<bool>        m_midiGateActive { false };
    std::atomic<float>       m_audioEnvelope { 0.0f };

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

    // Per-layer states & threading
    juce::String m_activeLayerName;
    std::unordered_map<std::string, LayerBendingState> m_layerStates;
    mutable std::mutex m_layerStateMutex;
    juce::Random m_jitterRng;

    // Buffers and synchronization
    std::vector<CircularBuffer> m_in_buffers;
    std::vector<CircularBuffer> m_out_buffers;
    std::vector<CircularBuffer> m_dry_delay_buffers;
    
    std::vector<std::vector<float>> m_staging_in;
    std::vector<std::vector<float>> m_staging_out;

    // Dedicated worker buffers for background model inference
    std::vector<std::vector<float>> m_worker_in;
    std::vector<std::vector<float>> m_worker_out;
    
    std::mutex m_staging_mutex;
    std::atomic<bool> m_output_ready { false };

    ModelThread m_model_thread;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessor)
};

