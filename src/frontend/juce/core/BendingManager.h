#pragma once

#include "IBackend.h"
#include <JuceHeader.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

// Layer classification for Trace Isolation
enum class LayerCategory {
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

// Stochastic Drift Modes
enum class DriftMode {
    Noise = 0,    // Classic independent uniform/Gaussian jitter per block
    ThermalOU,    // Ornstein-Uhlenbeck mean-reverting thermal drift
    RandomWalk    // Continuous wandering walk
};

// Bending Trigger Modes
enum class TriggerMode {
    Continuous = 0, // Default: Always active / continuous bending
    Momentary,      // UI "Short" Button / Parameter hold
    Transient       // Audio sidechain transient follower
};

// Per-Layer Bending State
struct LayerBendingState {
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

class BendingManager {
public:
    BendingManager() = default;

    static auto classify_layer(const std::string& name) -> LayerCategory;
    static auto classifyLayer(const std::string& name) -> LayerCategory { return classify_layer(name); }

    // Layer state accessors
    auto get_layer_state(const std::string& layer_name) const -> LayerBendingState;
    auto set_layer_state(const std::string& layer_name, const LayerBendingState& state) -> void;
    auto set_layer_scale(const std::string& layer_name, float scale) -> void;
    auto set_layer_offset(const std::string& layer_name, float offset) -> void;
    auto set_layer_heat(const std::string& layer_name, float heat) -> void;
    auto set_layer_jitter(const std::string& layer_name, float jitter) -> void { set_layer_heat(layer_name, jitter); }
    auto set_layer_memory(const std::string& layer_name, float memory) -> void;
    auto set_layer_drift_mode(const std::string& layer_name, DriftMode mode) -> void;
    auto set_layer_frozen(const std::string& layer_name, bool frozen) -> void;
    auto set_layer_bridge_source(const std::string& layer_name, const std::string& source_layer) -> void;
    auto set_layer_bridge_depth(const std::string& layer_name, float depth) -> void;
    auto set_layer_drawn_weights(const std::string& layer_name, const std::vector<float>& weights) -> void;
    auto set_layer_base_weights(const std::string& layer_name, const std::vector<float>& weights) -> void { set_layer_drawn_weights(layer_name, weights); }
    auto set_layer_original_weights(const std::string& layer_name, const std::vector<float>& weights) -> void;

    auto clear_layer_bending(const std::string& layer_name) -> void;
    auto clear_all_layer_bending() -> void;
    [[nodiscard]] auto has_active_bending() const -> bool;
    [[nodiscard]] auto get_all_layer_states() const -> std::unordered_map<std::string, LayerBendingState>;
    auto restore_layer_state(const std::string& name, float scale, float offset, float heat, float memory, DriftMode drift_mode, bool frozen) -> void;

    // Momentary / Trigger / Envelope controls
    auto trigger_short_circuit(bool active) -> void { m_short_circuit_active.store(active); }
    [[nodiscard]] auto is_short_circuit_active() const -> bool { return m_short_circuit_active.load(); }
    [[nodiscard]] auto get_momentary_envelope() const -> float { return m_momentary_envelope.load(); }

    auto set_trigger_mode(TriggerMode mode) -> void { m_trigger_mode.store(mode); }
    [[nodiscard]] auto get_trigger_mode() const -> TriggerMode { return m_trigger_mode.load(); }

    auto set_envelope_attack(float att) -> void;
    [[nodiscard]] auto get_envelope_attack() const -> float { return m_env_attack.load(); }

    auto set_envelope_release(float rel) -> void;
    [[nodiscard]] auto get_envelope_release() const -> float { return m_env_release.load(); }

    auto set_transient_threshold(float th) -> void;
    [[nodiscard]] auto get_transient_threshold() const -> float { return m_transient_threshold.load(); }

    auto update_audio_envelope(float peak) -> void;

    // Core weight calculation and upload to backend
    auto apply_bending(
        IBackend* backend,
        const std::string& active_layer_name,
        float param_scale,
        float param_offset,
        float param_heat,
        float param_memory,
        bool param_short_active
    ) -> void;

    // Backward-compatible camelCase aliases
    auto getLayerState(const std::string& name) const -> LayerBendingState { return get_layer_state(name); }
    auto setLayerState(const std::string& name, const LayerBendingState& s) -> void { set_layer_state(name, s); }
    auto setLayerScale(const std::string& name, float s) -> void { set_layer_scale(name, s); }
    auto setLayerOffset(const std::string& name, float o) -> void { set_layer_offset(name, o); }
    auto setLayerHeat(const std::string& name, float h) -> void { set_layer_heat(name, h); }
    auto setLayerJitter(const std::string& name, float j) -> void { set_layer_jitter(name, j); }
    auto setLayerMemory(const std::string& name, float m) -> void { set_layer_memory(name, m); }
    auto setLayerDriftMode(const std::string& name, DriftMode m) -> void { set_layer_drift_mode(name, m); }
    auto setLayerFrozen(const std::string& name, bool f) -> void { set_layer_frozen(name, f); }
    auto setLayerBridgeSource(const std::string& name, const std::string& s) -> void { set_layer_bridge_source(name, s); }
    auto setLayerBridgeDepth(const std::string& name, float d) -> void { set_layer_bridge_depth(name, d); }
    auto setLayerDrawnWeights(const std::string& name, const std::vector<float>& w) -> void { set_layer_drawn_weights(name, w); }
    auto setLayerBaseWeights(const std::string& name, const std::vector<float>& w) -> void { set_layer_base_weights(name, w); }
    auto setLayerOriginalWeights(const std::string& name, const std::vector<float>& w) -> void { set_layer_original_weights(name, w); }
    auto clearLayerBending(const std::string& name) -> void { clear_layer_bending(name); }
    auto clearAllLayerBending() -> void { clear_all_layer_bending(); }
    [[nodiscard]] auto hasActiveBending() const -> bool { return has_active_bending(); }
    auto triggerShortCircuit(bool active) -> void { trigger_short_circuit(active); }
    [[nodiscard]] auto isShortCircuitActive() const -> bool { return is_short_circuit_active(); }
    [[nodiscard]] auto getMomentaryEnvelope() const -> float { return get_momentary_envelope(); }
    auto setTriggerMode(TriggerMode mode) -> void { set_trigger_mode(mode); }
    [[nodiscard]] auto getTriggerMode() const -> TriggerMode { return get_trigger_mode(); }
    auto setEnvelopeAttack(float att) -> void { set_envelope_attack(att); }
    [[nodiscard]] auto getEnvelopeAttack() const -> float { return get_envelope_attack(); }
    auto setEnvelopeRelease(float rel) -> void { set_envelope_release(rel); }
    [[nodiscard]] auto getEnvelopeRelease() const -> float { return get_envelope_release(); }
    auto setTransientThreshold(float th) -> void { set_transient_threshold(th); }
    [[nodiscard]] auto getTransientThreshold() const -> float { return get_transient_threshold(); }

private:
    std::unordered_map<std::string, LayerBendingState> m_layer_states;
    mutable std::mutex m_mutex;
    juce::Random m_jitter_rng;

    // Momentary & Trigger state
    std::atomic<bool>        m_short_circuit_active { false };
    std::atomic<float>       m_momentary_envelope { 0.0f };
    std::atomic<TriggerMode> m_trigger_mode { TriggerMode::Continuous };
    std::atomic<float>       m_env_attack { 0.25f };   // Fast ramp-in step (0.01 - 1.0)
    std::atomic<float>       m_env_release { 0.85f };  // Exponential relaxation multiplier (0.5 - 0.999)
    std::atomic<float>       m_transient_threshold { 0.15f }; // Sidechain audio envelope follower threshold
    std::atomic<float>       m_audio_envelope { 0.0f };
};
