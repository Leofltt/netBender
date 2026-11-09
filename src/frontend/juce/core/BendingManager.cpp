#include "BendingManager.h"
#include <algorithm>
#include <cmath>
#include <cstring>

auto BendingManager::classify_layer(const std::string& name) -> LayerCategory {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("norm") != std::string::npos || lower.find("weight_g") != std::string::npos || lower.find("gamma") != std::string::npos) {
        return LayerCategory::Norm;
    }
    if (lower.find("bias") != std::string::npos) {
        return LayerCategory::Bias;
    }
    if (lower.find("snake") != std::string::npos || lower.find("act") != std::string::npos || lower.find("alpha") != std::string::npos) {
        return LayerCategory::Activation;
    }
    if (lower.find("attn") != std::string::npos || lower.find("attention") != std::string::npos || lower.find("qkv") != std::string::npos || lower.find("self_attn") != std::string::npos) {
        return LayerCategory::Attention;
    }
    if (lower.find("lstm") != std::string::npos || lower.find("gru") != std::string::npos || lower.find("rnn") != std::string::npos) {
        return LayerCategory::Recurrent;
    }
    if (lower.find("conv") != std::string::npos) {
        return LayerCategory::Conv;
    }
    if (lower.find("linear") != std::string::npos || lower.find("dense") != std::string::npos || lower.find("fc") != std::string::npos || lower.find("proj") != std::string::npos) {
        return LayerCategory::Linear;
    }
    return LayerCategory::Other;
}

auto BendingManager::get_layer_state(const std::string& layer_name) const -> LayerBendingState {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_layer_states.find(layer_name);
    if (it != m_layer_states.end()) {
        return it->second;
    }
    return LayerBendingState();
}

auto BendingManager::set_layer_state(const std::string& layer_name, const LayerBendingState& state) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name] = state;
}

auto BendingManager::set_layer_scale(const std::string& layer_name, float scale) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].scale = scale;
    m_layer_states[layer_name].isApplied = false;
}

auto BendingManager::set_layer_offset(const std::string& layer_name, float offset) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].offset = offset;
    m_layer_states[layer_name].isApplied = false;
}

auto BendingManager::set_layer_heat(const std::string& layer_name, float heat) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].heat = std::max(0.0f, heat);
}

auto BendingManager::set_layer_memory(const std::string& layer_name, float memory) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].memory = std::clamp(memory, 0.0f, 1.0f);
}

auto BendingManager::set_layer_drift_mode(const std::string& layer_name, DriftMode mode) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].driftMode = mode;
}

auto BendingManager::set_layer_frozen(const std::string& layer_name, bool frozen) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].frozen = frozen;
}

auto BendingManager::set_layer_bridge_source(const std::string& layer_name, const std::string& source_layer) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].bridgeSourceLayer = source_layer;
    m_layer_states[layer_name].isApplied = false;
}

auto BendingManager::set_layer_bridge_depth(const std::string& layer_name, float depth) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].bridgeDepth = std::clamp(depth, 0.0f, 1.0f);
    m_layer_states[layer_name].isApplied = false;
}

auto BendingManager::set_layer_drawn_weights(const std::string& layer_name, const std::vector<float>& weights) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states[layer_name].drawnWeights = weights;
    m_layer_states[layer_name].isApplied = false;
}

auto BendingManager::set_layer_original_weights(const std::string& layer_name, const std::vector<float>& weights) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& state = m_layer_states[layer_name];
    state.originalWeights = weights;
    if (state.drawnWeights.empty()) {
        state.drawnWeights = weights;
        if (std::abs(state.scale - 1.0f) <= 0.0001f &&
            std::abs(state.offset) <= 0.0001f &&
            state.heat <= 0.0001f &&
            (state.bridgeDepth <= 0.001f || state.bridgeSourceLayer.empty())) {
            state.isApplied = true;
        }
    }
}

auto BendingManager::clear_layer_bending(const std::string& layer_name) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states.erase(layer_name);
    if (!layer_name.empty()) {
        m_pending_reset_layers.push_back(layer_name);
    }
}

auto BendingManager::clear_all_layer_bending() -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_layer_states.clear();
    m_pending_reset_layers.clear();
    m_pending_reset_all = true;
}

auto BendingManager::get_layer_live_weights(const std::string& layer_name) const -> std::vector<float> {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_layer_states.find(layer_name);
    if (it != m_layer_states.end()) {
        if (it->second.isApplied && !it->second.scratchFinal.empty()) {
            return it->second.scratchFinal;
        }
        if (!it->second.originalWeights.empty()) {
            return it->second.originalWeights;
        }
    }
    return {};
}

auto BendingManager::has_active_bending() const -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_layer_states.empty()) {
        return false;
    }
    for (const auto& pair : m_layer_states) {
        const auto& s = pair.second;
        if (std::abs(s.scale - 1.0f) > 0.001f || std::abs(s.offset) > 0.001f || s.heat > 0.001f || s.frozen
            || (s.bridgeDepth > 0.001f && !s.bridgeSourceLayer.empty())) {
            return true;
        }
    }
    return false;
}

auto BendingManager::get_all_layer_states() const -> std::unordered_map<std::string, LayerBendingState> {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_layer_states;
}

auto BendingManager::restore_layer_state(const std::string& name, float scale, float offset, float heat, float memory, DriftMode drift_mode, bool frozen) -> void {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& state = m_layer_states[name];
    state.scale = scale;
    state.offset = offset;
    state.heat = heat;
    state.memory = memory;
    state.driftMode = drift_mode;
    state.frozen = frozen;
}

auto BendingManager::set_envelope_attack(float att) -> void {
    m_env_attack.store(std::clamp(att, 0.01f, 1.0f));
}

auto BendingManager::set_envelope_release(float rel) -> void {
    m_env_release.store(std::clamp(rel, 0.5f, 0.999f));
}

auto BendingManager::set_transient_threshold(float th) -> void {
    m_transient_threshold.store(std::clamp(th, 0.01f, 1.0f));
}

auto BendingManager::update_audio_envelope(float peak) -> void {
    float prev = m_audio_envelope.load();
    m_audio_envelope.store(std::max(peak, prev * 0.95f));
}

auto BendingManager::apply_bending(
    IBackend* backend,
    const std::string& active_layer_name,
    float param_scale,
    float param_offset,
    float param_heat,
    float param_memory,
    bool param_short_active
) -> void {
    if (backend == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_atomic_reset_all_requested.exchange(false)) {
        m_layer_states.clear();
        m_pending_reset_layers.clear();
        m_pending_reset_all = true;
    }

    if (m_pending_reset_all) {
        backend->reset_all_layer_weights();
        m_pending_reset_all = false;
        m_pending_reset_layers.clear();
    } else if (!m_pending_reset_layers.empty()) {
        for (const auto& rst_layer : m_pending_reset_layers) {
            backend->reset_layer_weights(rst_layer);
        }
        m_pending_reset_layers.clear();
    }

    // Update active layer from automated DAW parameters
    if (!active_layer_name.empty()) {
        auto it = m_layer_states.find(active_layer_name);
        bool is_non_default = (std::abs(param_scale - 1.0f) > 0.0001f ||
                               std::abs(param_offset) > 0.0001f ||
                               param_heat > 0.0001f);
        if (it != m_layer_states.end() || is_non_default) {
            auto& state = m_layer_states[active_layer_name];
            float new_scale = param_scale;
            float new_offset = param_offset;
            float new_heat = param_heat;
            float new_memory = (param_memory >= 0.0f) ? param_memory : state.memory;

            if (std::abs(state.scale - new_scale) > 0.0001f ||
                std::abs(state.offset - new_offset) > 0.0001f ||
                std::abs(state.heat - new_heat) > 0.0001f ||
                std::abs(state.memory - new_memory) > 0.0001f) {
                state.scale = new_scale;
                state.offset = new_offset;
                state.heat = new_heat;
                state.memory = new_memory;
                state.isApplied = false;
            }
        }
    }

    // Calculate momentary envelope progression based on TriggerMode:
    TriggerMode mode = m_trigger_mode.load();
    bool short_active = false;

    if (mode == TriggerMode::Momentary) {
        short_active = m_short_circuit_active.load() || param_short_active;
    } else if (mode == TriggerMode::Transient) {
        short_active = (m_audio_envelope.load() >= m_transient_threshold.load());
    }

    float current_env = m_momentary_envelope.load();
    float att = m_env_attack.load();
    float rel = m_env_release.load();

    if (mode == TriggerMode::Continuous) {
        // Continuous mode: bending is always 100% active
        current_env = 1.0f;
    } else {
        if (short_active) {
            current_env = std::min(1.0f, current_env + att);
        } else {
            current_env = std::max(0.0f, current_env * rel);
        }
    }

    m_momentary_envelope.store(current_env);

    for (auto& pair : m_layer_states) {
        const std::string& layer_name = pair.first;
        LayerBendingState& state = pair.second;

        bool has_drift = (state.heat > 0.0001f && !state.frozen);
        if (!has_drift) {
            if (mode == TriggerMode::Continuous && state.isApplied) {
                continue;
            }
            if (mode != TriggerMode::Continuous && current_env <= 0.001f && !state.isApplied) {
                continue;
            }
            if (mode != TriggerMode::Continuous && current_env > 0.001f &&
                std::abs(current_env - state.lastAppliedEnv) <= 0.002f) {
                continue;
            }
        }

        if (state.originalWeights.empty()) {
            state.originalWeights = backend->get_original_layer_weights(layer_name);
        }
        if (state.originalWeights.empty()) {
            continue;
        }

        size_t num_weights = state.originalWeights.size();

        if (state.drawnWeights.size() != num_weights) {
            state.drawnWeights = state.originalWeights;
        }
        if (state.driftOffsets.size() != num_weights) {
            state.driftOffsets.assign(num_weights, 0.0f);
        }

        // Cross-Talk (Trace Bridging): Blend target layer's base shape with source layer
        if (state.scratchBridged.size() != num_weights) {
            state.scratchBridged.resize(num_weights);
        }
        std::memcpy(state.scratchBridged.data(), state.drawnWeights.data(), num_weights * sizeof(float));

        if (state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != layer_name) {
            auto source_weights = backend->get_original_layer_weights(state.bridgeSourceLayer);
            if (!source_weights.empty()) {
                float alpha = std::clamp(state.bridgeDepth, 0.0f, 1.0f);
                size_t src_size = source_weights.size();
                for (size_t i = 0; i < num_weights; ++i) {
                    float src_val = source_weights[i % src_size];
                    state.scratchBridged[i] = (1.0f - alpha) * state.drawnWeights[i] + alpha * src_val;
                }
            }
        }

        if (state.scratchFinal.size() != num_weights) {
            state.scratchFinal.resize(num_weights);
        }

        // Compute stochastic drift when heat > 0 and not frozen
        if (state.heat > 0.0001f && !state.frozen) {
            float heat = state.heat;
            float memory = std::clamp(state.memory, 0.0f, 0.999f);

            for (size_t i = 0; i < num_weights; ++i) {
                float noise = (m_jitter_rng.nextFloat() * 2.0f - 1.0f) * heat;

                if (state.driftMode == DriftMode::ThermalOU) {
                    // Ornstein-Uhlenbeck: dW = -theta * W + sigma * dW_noise
                    state.driftOffsets[i] = (state.driftOffsets[i] * memory) + noise * (1.0f - memory);
                } else if (state.driftMode == DriftMode::RandomWalk) {
                    // Continuous wandering walk with soft bounds
                    state.driftOffsets[i] = std::clamp(state.driftOffsets[i] + noise * 0.1f, -1.0f, 1.0f);
                } else {
                    // Classic Noise / Glitch
                    state.driftOffsets[i] = noise;
                }

                // Bent target weight = (Bridged base * Scale + Offset) + Stochastic Drift
                float bent_weight = (state.scratchBridged[i] * state.scale + state.offset) + state.driftOffsets[i];

                if (mode == TriggerMode::Continuous) {
                    state.scratchFinal[i] = bent_weight;
                } else {
                    // Crossfade from unbent baseline W0 (originalWeights) to target bent W based on envelope
                    state.scratchFinal[i] = state.originalWeights[i] + current_env * (bent_weight - state.originalWeights[i]);
                }
            }
            backend->set_layer_weights(layer_name, state.scratchFinal);
            state.isApplied = true;
            state.lastAppliedEnv = current_env;
        } else if (mode != TriggerMode::Continuous && current_env > 0.001f) {
            // Momentary gate active without heat: only recompute if envelope moved significantly
            if (std::abs(current_env - state.lastAppliedEnv) > 0.002f) {
                juce::FloatVectorOperations::copy(state.scratchFinal.data(), state.scratchBridged.data(), static_cast<int>(num_weights));
                juce::FloatVectorOperations::multiply(state.scratchFinal.data(), state.scale, static_cast<int>(num_weights));
                juce::FloatVectorOperations::add(state.scratchFinal.data(), state.offset, static_cast<int>(num_weights));

                // Vectorized crossfade: finalWeights = originalWeights * (1 - env) + bentWeight * env
                juce::FloatVectorOperations::multiply(state.scratchFinal.data(), current_env, static_cast<int>(num_weights));
                juce::FloatVectorOperations::addWithMultiply(state.scratchFinal.data(), state.originalWeights.data(), 1.0f - current_env, static_cast<int>(num_weights));

                backend->set_layer_weights(layer_name, state.scratchFinal);
                state.isApplied = true;
                state.lastAppliedEnv = current_env;
            }
        } else if (mode == TriggerMode::Continuous) {
            // Continuous mode without drift: apply bent target once and skip redundant uploads
            if (!state.isApplied) {
                juce::FloatVectorOperations::copy(state.scratchFinal.data(), state.scratchBridged.data(), static_cast<int>(num_weights));
                juce::FloatVectorOperations::multiply(state.scratchFinal.data(), state.scale, static_cast<int>(num_weights));
                juce::FloatVectorOperations::add(state.scratchFinal.data(), state.offset, static_cast<int>(num_weights));

                backend->set_layer_weights(layer_name, state.scratchFinal);
                state.isApplied = true;
            }
        } else if (mode != TriggerMode::Continuous && current_env <= 0.001f) {
            // Momentary gate inactive: restore unbent baseline weights once
            if (state.isApplied) {
                backend->set_layer_weights(layer_name, state.originalWeights);
                state.isApplied = false;
                state.lastAppliedEnv = 0.0f;
            }
        }
    }
}
