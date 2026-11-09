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
    auto request_audio_thread_reset_all() -> void { m_atomic_reset_all_requested.store(true); }
    [[nodiscard]] auto get_layer_live_weights(const std::string& layer_name) const -> std::vector<float>;
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

private:
    std::unordered_map<std::string, LayerBendingState> m_layer_states;
    std::vector<std::string> m_pending_reset_layers;
    bool m_pending_reset_all { false };
    std::atomic<bool> m_atomic_reset_all_requested { false };
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
