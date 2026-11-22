#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <cmath>
#include "static_buffer.h"

struct MethodProperties {
    std::string name = "";
    int channels_in = -1;
    int channels_out = -1;
    int ratio_in = -1;
    int ratio_out = -1;
};

struct AttributeProperties {
    std::string name = "";
    std::vector<std::string> attribute_types = {};
};

struct ModelInfo {
    using MethodDict = std::unordered_map<std::string, MethodProperties>;
    using AttributeDict = std::unordered_map<std::string, AttributeProperties>;
    MethodDict method_properties = {};
    AttributeDict attribute_properties = {};
};

struct LockedModel;

/**
 * Abstract interface representing a Neural Network Audio Backend.
 * Allows decoupling high-level audio DSP and bending tools from specific inference runtimes
 * (such as LibTorch TorchScript, ONNX Runtime, or RT-Neural).
 */
class IBackend {
public:
    using DataType = float;
    using BufferMap = std::map<std::string, StaticBuffer<float>>;

    virtual ~IBackend() = default;

    // Lifecycle & Model Loading
    virtual auto load(std::string path, double sample_rate,
                      const std::string* target_method = nullptr) -> int = 0;
    virtual auto reload() -> int = 0;
    virtual auto is_loaded() -> bool = 0;
    virtual auto set_sample_rate(double sample_rate) -> void = 0;
    virtual auto get_sample_rate() -> double = 0;
    virtual auto use_gpu(bool value) -> void = 0;
    virtual auto get_last_error() const -> std::string { return ""; }
    virtual auto get_backend_name() const -> std::string { return "Generic"; }
    virtual auto get_input_label(const std::string& method, int index) -> std::string {
        return "(signal) model input " + std::to_string(index);
    }
    virtual auto get_output_label(const std::string& method, int index) -> std::string {
        return "(signal) model output " + std::to_string(index);
    }

    // Latent Hook Abstraction (backend agnostic: operates on raw float pointer, channels, length)
    struct LatentBuffer {
        float* data;
        int64_t channels;
        int64_t length;
    };
    using LatentHook = std::function<void(LatentBuffer&)>;

    // Inference Execution
    virtual auto perform(std::vector<float*>& in_buffer, std::vector<float*>& out_buffer,
                         std::string method, int n_batches, int n_out_channels, int n_vec)
        -> void = 0;

    virtual auto perform_forward(std::vector<float*>& in_buffer, std::vector<float*>& out_buffer,
                                 int n_batches, int n_out_channels, int n_vec,
                                 LatentHook latent_hook = nullptr) -> void {}

    virtual auto perform_prior_decode(std::vector<float*>& out_buffer, int n_batches,
                                      int n_out_channels, int n_vec) -> void = 0;

    virtual auto perform_decode_gen(std::vector<float*>& out_buffer, int n_batches,
                                    int n_out_channels, int n_vec, LatentHook latent_hook = nullptr)
        -> void {}

    virtual auto clear_decode_gen_fifo() -> void {
        for (auto& ch : m_decode_gen_fifo)
            ch.clear();
        m_decode_gen_fifo.clear();
        m_decode_gen_fifo_read_idx = 0;
        m_decode_gen_latent_state.clear();
        m_decode_gen_ola_delay.clear();
        m_decode_gen_ola_write_idx = 0;
        m_decode_gen_ola_phase = 0;
    }

    virtual auto set_decode_gen_smoothness(float smoothness) noexcept -> void {
        m_decode_gen_smoothness.store(std::clamp(smoothness, 0.0f, 1.0f),
                                      std::memory_order_relaxed);
    }

    [[nodiscard]] virtual auto get_decode_gen_smoothness() const noexcept -> float {
        return m_decode_gen_smoothness.load(std::memory_order_relaxed);
    }

    [[nodiscard]] static auto compute_decode_gen_alpha(float smoothness, int ratio) noexcept
        -> float {
        if (smoothness < 0.0f) {
            return (ratio <= 640) ? 0.94f : 0.86f;
        }
        const float s = std::clamp(smoothness, 0.0f, 1.0f);
        // Nonlinear log-spaced (1 - alpha): s=0 -> 0.50, s=0.30 -> 0.874, s=0.50 -> 0.95, s=1 ->
        // 0.995
        return 1.0f - 0.50f * std::pow(0.01f, s);
    }

    virtual auto set_decode_gen_ola_mode(int mode) noexcept -> void {
        m_decode_gen_ola_mode.store(std::clamp(mode, 0, 3), std::memory_order_relaxed);
    }

    [[nodiscard]] virtual auto get_decode_gen_ola_mode() const noexcept -> int {
        return m_decode_gen_ola_mode.load(std::memory_order_relaxed);
    }

    // Method & Mode Introspection
    virtual auto has_method(std::string method_name) -> bool = 0;
    virtual auto has_prior_decode() -> bool = 0;
    virtual auto has_autoencode() -> bool = 0;
    virtual auto get_plugin_modes() -> std::vector<std::string> = 0;
    virtual auto get_mode_params(std::string mode) -> std::vector<int> = 0;
    virtual auto get_available_methods(LockedModel* model = nullptr)
        -> std::vector<std::string> = 0;
    virtual auto get_method_params(std::string method) -> std::vector<int> = 0;
    virtual auto get_higher_ratio() -> int = 0;
    virtual auto get_preferred_buffer_size() -> int { return 0; }

    // Parameter Introspection & Bending
    virtual auto get_available_layers() -> std::vector<std::string> = 0;
    virtual auto get_layer_weights(std::string layer_name) -> std::vector<float> = 0;
    virtual auto get_original_layer_weights(std::string layer_name) -> std::vector<float> = 0;
    virtual auto get_layer_shape(std::string layer_name) -> std::vector<int64_t> = 0;
    virtual auto set_layer_weights(std::string layer_name, const std::vector<float>& weights)
        -> void = 0;
    virtual auto reset_layer_weights(std::string layer_name) -> void = 0;
    virtual auto reset_all_layer_weights() -> void = 0;
    virtual auto reset_all_weights() -> void { reset_all_layer_weights(); }
    virtual auto save_model(std::string path) -> int = 0;

    // Attributes & Buffer Management (Max/MSP & PureData compatibility)
    virtual auto get_available_attributes() -> std::vector<std::string> = 0;
    virtual auto get_settable_attributes() -> std::vector<std::string> = 0;
    virtual auto has_settable_attribute(std::string attribute) -> bool = 0;
    virtual auto get_attribute_as_string(std::string attribute_name) -> std::string = 0;
    virtual auto set_attribute(std::string attribute_name, std::vector<std::string> attribute_args,
                               const BufferMap& buffer_array) -> void = 0;
    virtual auto get_buffer_attributes() -> std::vector<std::string> = 0;
    virtual auto is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx)
        -> bool = 0;
    virtual auto is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx)
        -> bool = 0;
    virtual auto get_buffer_name(std::string attribute_name, int attribute_elt_idx)
        -> std::string = 0;
    virtual auto update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) -> int = 0;
    virtual auto reset_buffer(std::string buffer_id) -> int = 0;
    virtual auto get_model_info() -> ModelInfo = 0;

protected:
    auto apply_decode_gen_patch_ola(std::vector<float*>& out_buffer, int out_ch, int n_vec,
                                    int ratio) -> void {
        const int mode = get_decode_gen_ola_mode();
        const int effective = (mode == 0) ? ((ratio >= 4096) ? 3 : 1) : mode;
        if (effective <= 1 || out_ch <= 0 || n_vec <= 0)
            return;
        const size_t patch = static_cast<size_t>(std::max(ratio, 1024));
        const int active_ch = std::min(out_ch, static_cast<int>(out_buffer.size()));

        if (effective == 2) {
            const size_t half_patch = patch / 2;
            if (half_patch == 0)
                return;
            if (m_decode_gen_ola_delay.size() != static_cast<size_t>(active_ch) ||
                (!m_decode_gen_ola_delay.empty() &&
                 m_decode_gen_ola_delay[0].size() != half_patch)) {
                m_decode_gen_ola_delay.assign(static_cast<size_t>(active_ch),
                                              std::vector<float>(half_patch, 0.0f));
                m_decode_gen_ola_write_idx = 0;
                m_decode_gen_ola_phase = 0;
            }
            constexpr float kPi = 3.14159265358979323846f;
            const float step = kPi / static_cast<float>(patch);
            for (int i = 0; i < n_vec; ++i) {
                const size_t p = m_decode_gen_ola_phase % patch;
                const size_t p_shifted = (p + half_patch) % patch;
                const float w_a = std::sin(step * static_cast<float>(p));
                const float w_b = std::sin(step * static_cast<float>(p_shifted));
                const size_t idx = m_decode_gen_ola_write_idx;
                for (int c = 0; c < active_ch; ++c) {
                    if (out_buffer[c] == nullptr)
                        continue;
                    const float cur = out_buffer[c][i];
                    const float delayed = m_decode_gen_ola_delay[c][idx];
                    m_decode_gen_ola_delay[c][idx] = cur;
                    out_buffer[c][i] = cur * w_a + delayed * w_b;
                }
                m_decode_gen_ola_write_idx = (idx + 1) % half_patch;
                m_decode_gen_ola_phase = (p + 1) % patch;
            }
            return;
        }

        const size_t q = patch / 4;
        if (q == 0)
            return;
        const size_t delay_len = 3 * q;
        if (m_decode_gen_ola_delay.size() != static_cast<size_t>(active_ch) ||
            (!m_decode_gen_ola_delay.empty() && m_decode_gen_ola_delay[0].size() != delay_len)) {
            m_decode_gen_ola_delay.assign(static_cast<size_t>(active_ch),
                                          std::vector<float>(delay_len, 0.0f));
            m_decode_gen_ola_write_idx = 0;
            m_decode_gen_ola_phase = 0;
        }
        constexpr float kTwoPi = 6.28318530717958647692f;
        // 0.5 / sqrt(1.5) so sum of 4 quarter-shifted squared Hann windows equals 1.0
        constexpr float kHannNorm = 0.40824829046386301637f;
        const float step = kTwoPi / static_cast<float>(patch);
        for (int i = 0; i < n_vec; ++i) {
            const size_t p = m_decode_gen_ola_phase % patch;
            const float w0 = kHannNorm * (1.0f - std::cos(step * static_cast<float>(p)));
            const float w1 =
                kHannNorm * (1.0f - std::cos(step * static_cast<float>((p + 3 * q) % patch)));
            const float w2 =
                kHannNorm * (1.0f - std::cos(step * static_cast<float>((p + 2 * q) % patch)));
            const float w3 =
                kHannNorm * (1.0f - std::cos(step * static_cast<float>((p + q) % patch)));
            const size_t idx = m_decode_gen_ola_write_idx;
            const size_t idx3 = idx;
            const size_t idx2 = (idx + q) % delay_len;
            const size_t idx1 = (idx + 2 * q) % delay_len;
            for (int c = 0; c < active_ch; ++c) {
                if (out_buffer[c] == nullptr)
                    continue;
                const float cur = out_buffer[c][i];
                const float d1 = m_decode_gen_ola_delay[c][idx1];
                const float d2 = m_decode_gen_ola_delay[c][idx2];
                const float d3 = m_decode_gen_ola_delay[c][idx3];
                m_decode_gen_ola_delay[c][idx] = cur;
                out_buffer[c][i] = cur * w0 + d1 * w1 + d2 * w2 + d3 * w3;
            }
            m_decode_gen_ola_write_idx = (idx + 1) % delay_len;
            m_decode_gen_ola_phase = (p + 1) % patch;
        }
    }

    // Decode Gen output FIFO buffer per channel for gapless streaming & latent random walk state
    std::vector<std::vector<float>> m_decode_gen_fifo;
    size_t m_decode_gen_fifo_read_idx{0};
    std::vector<float> m_decode_gen_latent_state;
    std::atomic<float> m_decode_gen_smoothness{-1.0f};
    std::atomic<int> m_decode_gen_ola_mode{0};
    std::vector<std::vector<float>> m_decode_gen_ola_delay;
    size_t m_decode_gen_ola_write_idx{0};
    size_t m_decode_gen_ola_phase{0};
};
