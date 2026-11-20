#pragma once

#include "IBackend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Reusable in-memory mock implementation of IBackend for backend unit and integration tests.
 * Zero dependency on LibTorch, ONNX Runtime, ExecuTorch, JUCE, or model files on disk.
 */
class MockBackend : public IBackend {
public:
    MockBackend() {
        m_layers["layer0.weight"] = LayerEntry{
            {2, 2},
            {1.0f, 0.5f, -0.5f, 0.25f},
            {1.0f, 0.5f, -0.5f, 0.25f},
        };
        m_layers["layer1.bias"] = LayerEntry{
            {2},
            {0.1f, -0.1f},
            {0.1f, -0.1f},
        };
        m_layer_order = {"layer0.weight", "layer1.bias"};

        m_attributes["temperature"] = "1.0";
        m_attributes["readonly_info"] = "mock_v1";
        m_attributes["buf_attr"] = "buf_attr";
        m_buffers["buf_attr"] = StaticBuffer<float>(1, 4, 44100.0);
    }

    // Configurable behavior
    int in_channels{1};
    int out_channels{1};
    int in_ratio{1};
    int out_ratio{1};
    int higher_ratio{2048};
    int preferred_buffer_size{2048};
    int latent_dim{4};
    float gain{1.0f};
    bool has_forward_method{true};
    bool has_prior_method{true};
    bool has_autoencode_methods{true};
    bool fail_next_load{false};
    std::function<void(std::vector<float*>&, std::vector<float*>&, int, int, int)> custom_perform;

    // Inspection counters & state
    int perform_count{0};
    int forward_count{0};
    int prior_count{0};
    int decode_gen_count{0};
    std::string last_method;
    int last_n_batches{0};
    int last_n_out_channels{0};
    int last_n_vec{0};
    bool last_latent_hook_called{false};
    bool gpu_enabled{false};

    [[nodiscard]] auto decode_gen_state_size() const -> size_t {
        return m_decode_gen_latent_state.size();
    }

    // Lifecycle & Model Loading
    auto load(std::string path, double sample_rate, const std::string* target_method = nullptr)
        -> int override {
        if (path.empty()) {
            m_last_error = "MockBackend: empty model path";
            m_loaded = false;
            return -1;
        }
        if (fail_next_load) {
            fail_next_load = false;
            m_last_error = "MockBackend: simulated load failure";
            m_loaded = false;
            return -1;
        }
        if (target_method != nullptr && !target_method->empty() &&
            !has_method_internal(*target_method)) {
            m_last_error = "MockBackend: missing target method '" + *target_method + "'";
            m_loaded = false;
            return -1;
        }

        m_path = std::move(path);
        m_sample_rate = (sample_rate > 0.0) ? sample_rate : 44100.0;
        m_loaded = true;
        m_last_error.clear();
        clear_decode_gen_fifo();
        return 0;
    }

    auto reload() -> int override {
        if (m_path.empty()) {
            m_last_error = "MockBackend: cannot reload without a path";
            return -1;
        }
        return load(m_path, m_sample_rate);
    }

    auto is_loaded() -> bool override { return m_loaded; }

    auto set_sample_rate(double sample_rate) -> void override {
        if (sample_rate > 0.0) {
            m_sample_rate = sample_rate;
        }
    }

    auto get_sample_rate() -> double override { return m_sample_rate; }

    auto use_gpu(bool value) -> void override { gpu_enabled = value; }

    auto get_last_error() const -> std::string override { return m_last_error; }

    auto get_backend_name() const -> std::string override {
        return gpu_enabled ? "MockBackend (GPU)" : "MockBackend";
    }

    // Inference Execution
    auto perform(std::vector<float*>& in_buffer, std::vector<float*>& out_buffer,
                 std::string method, int n_batches, int n_out_channels, int n_vec)
        -> void override {
        ++perform_count;
        last_method = std::move(method);
        last_n_batches = n_batches;
        last_n_out_channels = n_out_channels;
        last_n_vec = n_vec;

        if (custom_perform) {
            custom_perform(in_buffer, out_buffer, n_batches, n_out_channels, n_vec);
            return;
        }

        run_scaled_copy(in_buffer, out_buffer, n_batches, n_out_channels, n_vec, gain);
    }

    auto perform_forward(std::vector<float*>& in_buffer, std::vector<float*>& out_buffer,
                         int n_batches, int n_out_channels, int n_vec,
                         LatentHook latent_hook = nullptr) -> void override {
        ++forward_count;
        last_method = "forward";
        last_n_batches = n_batches;
        last_n_out_channels = n_out_channels;
        last_n_vec = n_vec;
        last_latent_hook_called = false;

        float latent_scale = 1.0f;
        if (latent_hook) {
            const int64_t ch = std::max(1, latent_dim);
            const int64_t len = std::max(1, n_vec / std::max(1, higher_ratio));
            std::vector<float> latent_storage(static_cast<size_t>(ch * len), 1.0f);
            LatentBuffer buf{latent_storage.data(), ch, len};
            latent_hook(buf);
            last_latent_hook_called = true;
            latent_scale = latent_storage[0];
        }

        if (custom_perform) {
            custom_perform(in_buffer, out_buffer, n_batches, n_out_channels, n_vec);
            if (last_latent_hook_called && latent_scale != 1.0f) {
                for (int b = 0; b < n_batches; ++b) {
                    for (int c = 0; c < n_out_channels; ++c) {
                        const int out_idx = b * n_out_channels + c;
                        if (out_idx < static_cast<int>(out_buffer.size()) &&
                            out_buffer[static_cast<size_t>(out_idx)] != nullptr) {
                            float* dst = out_buffer[static_cast<size_t>(out_idx)];
                            for (int i = 0; i < n_vec; ++i) {
                                dst[i] *= latent_scale;
                            }
                        }
                    }
                }
            }
            return;
        }

        run_scaled_copy(in_buffer, out_buffer, n_batches, n_out_channels, n_vec,
                        gain * latent_scale);
    }

    auto perform_prior_decode(std::vector<float*>& out_buffer, int n_batches, int n_out_channels,
                              int n_vec) -> void override {
        ++prior_count;
        last_method = "prior";
        last_n_batches = n_batches;
        last_n_out_channels = n_out_channels;
        last_n_vec = n_vec;

        for (int b = 0; b < n_batches; ++b) {
            for (int c = 0; c < n_out_channels; ++c) {
                const int out_idx = b * n_out_channels + c;
                if (out_idx >= static_cast<int>(out_buffer.size())) {
                    continue;
                }
                float* dst = out_buffer[static_cast<size_t>(out_idx)];
                if (dst == nullptr) {
                    continue;
                }
                for (int i = 0; i < n_vec; ++i) {
                    dst[i] = gain * (0.2f + 0.001f * static_cast<float>(i % 64));
                }
            }
        }
    }

    auto perform_decode_gen(std::vector<float*>& out_buffer, int n_batches, int n_out_channels,
                            int n_vec, LatentHook latent_hook = nullptr) -> void override {
        ++decode_gen_count;
        last_method = "decode_gen";
        last_n_batches = n_batches;
        last_n_out_channels = n_out_channels;
        last_n_vec = n_vec;
        last_latent_hook_called = false;

        const int64_t ch = std::max(1, latent_dim);
        const int64_t len = std::max(1, n_vec / std::max(1, higher_ratio));
        m_decode_gen_latent_state.assign(static_cast<size_t>(ch * len), 1.0f);

        if (latent_hook) {
            LatentBuffer buf{m_decode_gen_latent_state.data(), ch, len};
            latent_hook(buf);
            last_latent_hook_called = true;
        }

        const float latent_scale = m_decode_gen_latent_state[0];
        for (int b = 0; b < n_batches; ++b) {
            for (int c = 0; c < n_out_channels; ++c) {
                const int out_idx = b * n_out_channels + c;
                if (out_idx >= static_cast<int>(out_buffer.size())) {
                    continue;
                }
                float* dst = out_buffer[static_cast<size_t>(out_idx)];
                if (dst == nullptr) {
                    continue;
                }
                for (int i = 0; i < n_vec; ++i) {
                    dst[i] = gain * latent_scale * 0.25f;
                }
            }
        }
    }

    // Method & Mode Introspection
    auto has_method(std::string method_name) -> bool override {
        return m_loaded && has_method_internal(method_name);
    }

    auto has_prior_decode() -> bool override { return m_loaded && has_prior_method; }

    auto has_autoencode() -> bool override { return m_loaded && has_autoencode_methods; }

    auto get_plugin_modes() -> std::vector<std::string> override {
        if (!m_loaded) {
            return {};
        }
        std::vector<std::string> modes;
        if (has_forward_method) {
            modes.emplace_back("forward");
        }
        if (has_autoencode_methods) {
            modes.emplace_back("autoencode");
        }
        if (has_prior_method) {
            modes.emplace_back("prior");
        }
        return modes;
    }

    auto get_mode_params(std::string mode) -> std::vector<int> override {
        if (mode == "autoencode" && has_autoencode_methods) {
            return {in_channels, in_ratio, out_channels, out_ratio};
        }
        if (mode == "prior" && has_prior_method) {
            return {0, 0, out_channels, out_ratio};
        }
        return get_method_params(mode);
    }

    auto get_available_methods(LockedModel* /*model*/ = nullptr)
        -> std::vector<std::string> override {
        if (!m_loaded) {
            return {};
        }
        std::vector<std::string> methods;
        if (has_forward_method) {
            methods.emplace_back("forward");
        }
        if (has_autoencode_methods) {
            methods.emplace_back("encode");
            methods.emplace_back("decode");
        } else if (has_prior_method) {
            methods.emplace_back("decode");
        }
        if (has_prior_method) {
            methods.emplace_back("prior");
        }
        return methods;
    }

    auto get_method_params(std::string method) -> std::vector<int> override {
        if (!has_method_internal(method)) {
            return {};
        }
        if (method == "forward") {
            return {in_channels, in_ratio, out_channels, out_ratio};
        }
        if (method == "encode") {
            return {in_channels, higher_ratio, latent_dim, higher_ratio};
        }
        if (method == "decode") {
            return {latent_dim, higher_ratio, out_channels, higher_ratio};
        }
        if (method == "prior") {
            return {0, 0, latent_dim, higher_ratio};
        }
        return {};
    }

    auto get_higher_ratio() -> int override { return higher_ratio; }

    auto get_preferred_buffer_size() -> int override { return preferred_buffer_size; }

    // Parameter Introspection & Bending
    auto get_available_layers() -> std::vector<std::string> override { return m_layer_order; }

    auto get_layer_weights(std::string layer_name) -> std::vector<float> override {
        auto it = m_layers.find(layer_name);
        return (it != m_layers.end()) ? it->second.current : std::vector<float>{};
    }

    auto get_original_layer_weights(std::string layer_name) -> std::vector<float> override {
        auto it = m_layers.find(layer_name);
        return (it != m_layers.end()) ? it->second.original : std::vector<float>{};
    }

    auto get_layer_shape(std::string layer_name) -> std::vector<int64_t> override {
        auto it = m_layers.find(layer_name);
        return (it != m_layers.end()) ? it->second.shape : std::vector<int64_t>{};
    }

    auto set_layer_weights(std::string layer_name, const std::vector<float>& weights)
        -> void override {
        auto it = m_layers.find(layer_name);
        if (it == m_layers.end() || weights.size() != it->second.current.size()) {
            return;
        }
        it->second.current = weights;
    }

    auto reset_layer_weights(std::string layer_name) -> void override {
        auto it = m_layers.find(layer_name);
        if (it != m_layers.end()) {
            it->second.current = it->second.original;
        }
    }

    auto reset_all_layer_weights() -> void override {
        for (auto& kv : m_layers) {
            kv.second.current = kv.second.original;
        }
    }

    auto save_model(std::string path) -> int override {
        if (!m_loaded || path.empty()) {
            return -1;
        }
        return 0;
    }

    // Attributes & Buffer Management
    auto get_available_attributes() -> std::vector<std::string> override {
        return {"temperature", "readonly_info", "buf_attr"};
    }

    auto get_settable_attributes() -> std::vector<std::string> override {
        return {"temperature", "buf_attr"};
    }

    auto has_settable_attribute(std::string attribute) -> bool override {
        return attribute == "temperature" || attribute == "buf_attr";
    }

    auto get_attribute_as_string(std::string attribute_name) -> std::string override {
        auto it = m_attributes.find(attribute_name);
        return (it != m_attributes.end()) ? it->second : std::string{};
    }

    auto set_attribute(std::string attribute_name, std::vector<std::string> attribute_args,
                       const BufferMap& buffer_array) -> void override {
        if (!has_settable_attribute(attribute_name) || attribute_args.empty()) {
            return;
        }
        m_attributes[attribute_name] = attribute_args[0];
        if (attribute_name == "buf_attr") {
            auto buf_it = buffer_array.find(attribute_args[0]);
            if (buf_it != buffer_array.end()) {
                m_buffers["buf_attr"] = buf_it->second;
            }
        }
    }

    auto get_buffer_attributes() -> std::vector<std::string> override { return {"buf_attr"}; }

    auto is_buffer_element_of_attribute(std::string attribute_name, int attribute_elt_idx)
        -> bool override {
        return attribute_name == "buf_attr" && attribute_elt_idx == 0;
    }

    auto is_tensor_element_of_attribute(std::string attribute_name, int attribute_elt_idx)
        -> bool override {
        return attribute_name == "temperature" && attribute_elt_idx == 0;
    }

    auto get_buffer_name(std::string attribute_name, int attribute_elt_idx)
        -> std::string override {
        if (attribute_name == "buf_attr" && attribute_elt_idx == 0) {
            return "buf_attr";
        }
        return {};
    }

    auto update_buffer(std::string buffer_id, StaticBuffer<float>& buffer) -> int override {
        if (buffer_id.empty()) {
            return -1;
        }
        m_buffers[buffer_id] = buffer;
        return 0;
    }

    auto reset_buffer(std::string buffer_id) -> int override {
        auto it = m_buffers.find(buffer_id);
        if (it == m_buffers.end()) {
            return -1;
        }
        it->second.reset();
        return 0;
    }

    auto get_model_info() -> ModelInfo override {
        ModelInfo info;
        for (const auto& m : get_available_methods()) {
            auto p = get_method_params(m);
            if (p.size() >= 4) {
                info.method_properties[m] = MethodProperties{m, p[0], p[2], p[1], p[3]};
            }
        }
        info.attribute_properties["temperature"] = AttributeProperties{"temperature", {"float"}};
        info.attribute_properties["buf_attr"] = AttributeProperties{"buf_attr", {"buffer"}};
        return info;
    }

private:
    struct LayerEntry {
        std::vector<int64_t> shape;
        std::vector<float> original;
        std::vector<float> current;
    };

    [[nodiscard]] auto has_method_internal(const std::string& method_name) const -> bool {
        if (method_name == "forward") {
            return has_forward_method;
        }
        if (method_name == "encode") {
            return has_autoencode_methods;
        }
        if (method_name == "decode") {
            return has_autoencode_methods || has_prior_method;
        }
        if (method_name == "prior") {
            return has_prior_method;
        }
        return false;
    }

    auto run_scaled_copy(const std::vector<float*>& in_buffer, std::vector<float*>& out_buffer,
                         int n_batches, int n_out_channels, int n_vec, float scale) const -> void {
        const int eff_in_ch = std::max(1, in_channels);
        for (int b = 0; b < n_batches; ++b) {
            for (int c = 0; c < n_out_channels; ++c) {
                const int out_idx = b * n_out_channels + c;
                if (out_idx >= static_cast<int>(out_buffer.size())) {
                    continue;
                }
                float* dst = out_buffer[static_cast<size_t>(out_idx)];
                if (dst == nullptr) {
                    continue;
                }

                const float* src = nullptr;
                if (in_channels > 0) {
                    const int in_idx = b * eff_in_ch + (c % eff_in_ch);
                    if (in_idx < static_cast<int>(in_buffer.size())) {
                        src = in_buffer[static_cast<size_t>(in_idx)];
                    }
                }

                if (src != nullptr) {
                    for (int i = 0; i < n_vec; ++i) {
                        dst[i] = src[i] * scale;
                    }
                } else {
                    std::fill(dst, dst + n_vec, 0.0f);
                }
            }
        }
    }

    bool m_loaded{false};
    std::string m_path;
    double m_sample_rate{44100.0};
    std::string m_last_error;
    std::vector<std::string> m_layer_order;
    std::unordered_map<std::string, LayerEntry> m_layers;
    std::unordered_map<std::string, std::string> m_attributes;
    std::map<std::string, StaticBuffer<float>> m_buffers;
};
