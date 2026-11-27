#include "ExecuTorchSamplerBackend.h"
#include "SamplerBundleLoader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#endif

namespace netbender::sampler {

namespace {

struct SidecarParam {
    std::vector<int64_t> shape;
    std::vector<float> weights;
};

struct SidecarData {
    std::vector<std::string> param_order;
    std::unordered_map<std::string, SidecarParam> params;
    std::unordered_map<std::string, std::vector<std::string>> method_param_inputs;

    auto load(const std::string& bin_path) -> void {
        param_order.clear();
        params.clear();
        method_param_inputs.clear();
        if (bin_path.empty()) {
            return;
        }
        std::ifstream ifs(bin_path, std::ios::binary);
        if (!ifs.is_open()) {
            return;
        }
        char magic[4]{};
        ifs.read(magic, 4);
        if (!ifs.good() || std::memcmp(magic, "NBND", 4) != 0) {
            return;
        }
        uint32_t version = 0;
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!ifs.good()) {
            return;
        }
        if (version >= 2) {
            uint32_t json_len = 0;
            ifs.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));
            if (json_len > 0 && ifs.good()) {
                std::string json_str(json_len, '\0');
                ifs.read(json_str.data(), json_len);
                try {
                    const auto meta = nlohmann::json::parse(json_str);
                    if (meta.contains("method_param_inputs") &&
                        meta["method_param_inputs"].is_object()) {
                        for (auto it = meta["method_param_inputs"].begin();
                             it != meta["method_param_inputs"].end(); ++it) {
                            if (!it.value().is_array()) {
                                continue;
                            }
                            std::vector<std::string> names;
                            for (const auto& item : it.value()) {
                                if (item.is_string()) {
                                    names.push_back(item.get<std::string>());
                                }
                            }
                            method_param_inputs[it.key()] = std::move(names);
                        }
                    } else if (meta.contains("promoted_inputs") &&
                               meta["promoted_inputs"].is_array()) {
                        std::vector<std::string> names;
                        for (const auto& item : meta["promoted_inputs"]) {
                            if (item.is_string()) {
                                names.push_back(item.get<std::string>());
                            }
                        }
                        method_param_inputs["forward"] = std::move(names);
                    }
                } catch (...) {
                }
            }
        }

        uint32_t num_tensors = 0;
        ifs.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
        if (!ifs.good()) {
            return;
        }
        for (uint32_t t = 0; t < num_tensors && ifs.good(); ++t) {
            uint32_t name_len = 0;
            ifs.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            if (name_len > 0) {
                ifs.read(name.data(), name_len);
            }

            uint32_t num_dims = 0;
            ifs.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
            std::vector<int64_t> shape(num_dims);
            for (uint32_t d = 0; d < num_dims; ++d) {
                ifs.read(reinterpret_cast<char*>(&shape[d]), sizeof(int64_t));
            }

            uint64_t num_elems = 0;
            ifs.read(reinterpret_cast<char*>(&num_elems), sizeof(num_elems));
            std::vector<float> data(static_cast<size_t>(num_elems));
            if (num_elems > 0) {
                ifs.read(reinterpret_cast<char*>(data.data()),
                         static_cast<std::streamsize>(num_elems * sizeof(float)));
            }

            param_order.push_back(name);
            params[name] = SidecarParam{std::move(shape), std::move(data)};
        }

        if (method_param_inputs.find("forward") == method_param_inputs.end() &&
            !param_order.empty()) {
            method_param_inputs["forward"] = param_order;
        }
    }
};

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
struct ComponentProgram {
    std::unique_ptr<executorch::extension::Module> module;
    SidecarData sidecar;
};

auto load_et_module(const std::string& pte_path,
                    std::unique_ptr<executorch::extension::Module>& out_mod, std::string& out_error)
    -> bool {
    out_mod = std::make_unique<executorch::extension::Module>(
        pte_path, executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    const auto err = out_mod->load();
    if (err != executorch::runtime::Error::Ok) {
        out_error = "ExecuTorch failed to load '" + pte_path + "' (error " +
                    std::to_string(static_cast<int>(err)) + ")";
        out_mod.reset();
        return false;
    }
    return true;
}

auto append_baseline_params(ComponentProgram& prog, const std::string& method_name,
                            size_t num_model_inputs,
                            std::vector<executorch::runtime::EValue>& inputs,
                            std::vector<executorch::extension::TensorPtr>& keepalive) -> void {
    if (!prog.module) {
        return;
    }
    size_t expected_inputs = num_model_inputs;
    auto meta_res = prog.module->method_meta(method_name);
    if (meta_res.ok()) {
        expected_inputs = meta_res.get().num_inputs();
    }
    if (expected_inputs <= num_model_inputs) {
        return;
    }

    auto pm_it = prog.sidecar.method_param_inputs.find(method_name);
    const auto& names = (pm_it != prog.sidecar.method_param_inputs.end())
                            ? pm_it->second
                            : prog.sidecar.param_order;

    for (const auto& pname : names) {
        if (inputs.size() >= expected_inputs) {
            break;
        }
        auto pit = prog.sidecar.params.find(pname);
        if (pit == prog.sidecar.params.end()) {
            continue;
        }
        std::vector<executorch::aten::SizesType> psizes;
        psizes.reserve(pit->second.shape.size());
        for (int64_t d : pit->second.shape) {
            psizes.push_back(static_cast<executorch::aten::SizesType>(d));
        }
        auto pt = executorch::extension::from_blob(pit->second.weights.data(), std::move(psizes),
                                                   executorch::aten::ScalarType::Float);
        inputs.push_back(pt);
        keepalive.push_back(std::move(pt));
    }
}
#endif

}  // namespace

struct ExecuTorchSamplerBackend::Impl {
#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    ComponentProgram text_encoder;
    ComponentProgram number_cond;
    ComponentProgram dit;
    ComponentProgram ae_main;
    ComponentProgram ae_decode;
    ComponentProgram ae_encode;
    bool separate_decode{false};
    bool separate_encode{false};
#endif
};

ExecuTorchSamplerBackend::ExecuTorchSamplerBackend() : m_impl(std::make_unique<Impl>()) {}

ExecuTorchSamplerBackend::~ExecuTorchSamplerBackend() = default;

auto ExecuTorchSamplerBackend::is_loaded() const -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_loaded;
}

auto ExecuTorchSamplerBackend::get_metadata() const -> const SamplerBundleMetadata& {
    return m_meta;
}

auto ExecuTorchSamplerBackend::get_last_error() const -> std::string {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_last_error;
}

auto ExecuTorchSamplerBackend::get_backend_name() const -> std::string {
    std::string del = m_meta.delegate;
    std::transform(del.begin(), del.end(), del.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (del.empty() || del == "PORTABLE") {
        return "ExecuTorch Sampler";
    }
    return "ExecuTorch Sampler (" + del + ")";
}

auto ExecuTorchSamplerBackend::load_bundle(const std::string& manifest_path, bool use_gpu) -> int {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_loaded = false;
    m_use_gpu = use_gpu;
    m_last_error.clear();

    SamplerBundleMetadata parsed;
    std::string err;
    if (!parse_sampler_bundle_manifest(manifest_path, parsed, err)) {
        m_last_error = err;
        return -1;
    }
    if (parsed.runtime != "executorch") {
        m_last_error =
            "ExecuTorchSamplerBackend requires runtime 'executorch', got '" + parsed.runtime + "'";
        return -1;
    }

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        m_impl = std::make_unique<Impl>();

        if (!load_et_module(parsed.text_encoder_model, m_impl->text_encoder.module, m_last_error)) {
            return -1;
        }
        m_impl->text_encoder.sidecar.load(parsed.text_encoder_sidecar);

        if (!load_et_module(parsed.number_cond_model, m_impl->number_cond.module, m_last_error)) {
            return -1;
        }
        m_impl->number_cond.sidecar.load(parsed.number_cond_sidecar);

        if (!load_et_module(parsed.dit_model, m_impl->dit.module, m_last_error)) {
            return -1;
        }
        m_impl->dit.sidecar.load(parsed.dit_sidecar);

        m_impl->ae_main.sidecar.load(parsed.ae_sidecar);
        m_impl->separate_decode =
            !parsed.ae_decode_model.empty() && parsed.ae_decode_model != parsed.ae_model;
        m_impl->separate_encode =
            !parsed.ae_encode_model.empty() && parsed.ae_encode_model != parsed.ae_model;

        if (!m_impl->separate_decode || !m_impl->separate_encode) {
            if (!load_et_module(parsed.ae_model, m_impl->ae_main.module, m_last_error)) {
                return -1;
            }
        }
        if (m_impl->separate_decode) {
            if (!load_et_module(parsed.ae_decode_model, m_impl->ae_decode.module, m_last_error)) {
                return -1;
            }
            m_impl->ae_decode.sidecar = m_impl->ae_main.sidecar;
        }
        if (m_impl->separate_encode) {
            if (!load_et_module(parsed.ae_encode_model, m_impl->ae_encode.module, m_last_error)) {
                return -1;
            }
            m_impl->ae_encode.sidecar = m_impl->ae_main.sidecar;
        }

        m_meta = std::move(parsed);
        m_loaded = true;
        return 0;
    } catch (const std::exception& e) {
        m_last_error = std::string("Exception loading ExecuTorch sampler bundle: ") + e.what();
        return -1;
    }
#else
    m_meta = std::move(parsed);
    m_last_error =
        "ExecuTorch C++ runtime is not enabled in this build (NETBENDER_HAS_EXECUTORCH=0)";
    return -1;
#endif
}

auto ExecuTorchSamplerBackend::encode_prompt(const TokenizedPrompt& tokens, PromptEmbedding& out)
    -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_loaded) {
        m_last_error = "Sampler bundle not loaded";
        return false;
    }
    if (tokens.input_ids.empty() || tokens.input_ids.size() != tokens.attention_mask.size()) {
        m_last_error = "Invalid tokenized prompt buffer sizes";
        return false;
    }

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        std::vector<int64_t> ids = tokens.input_ids;
        std::vector<int64_t> mask = tokens.attention_mask;
        const auto seq_len = static_cast<executorch::aten::SizesType>(ids.size());

        std::vector<executorch::extension::TensorPtr> keepalive;
        std::vector<executorch::runtime::EValue> inputs;

        auto ids_t = executorch::extension::from_blob(ids.data(), {1, seq_len},
                                                      executorch::aten::ScalarType::Long);
        auto mask_t = executorch::extension::from_blob(mask.data(), {1, seq_len},
                                                       executorch::aten::ScalarType::Long);
        inputs.push_back(ids_t);
        inputs.push_back(mask_t);
        keepalive.push_back(std::move(ids_t));
        keepalive.push_back(std::move(mask_t));

        append_baseline_params(m_impl->text_encoder, "forward", 2, inputs, keepalive);

        auto res = m_impl->text_encoder.module->execute("forward", inputs);
        if (!res.ok() || res.get().empty() || !res.get()[0].isTensor()) {
            m_last_error = "ExecuTorch text_encoder execution failed";
            return false;
        }
        const auto tensor = res.get()[0].toTensor();
        const int64_t out_seq =
            (tensor.dim() >= 2) ? tensor.size(1) : static_cast<int64_t>(seq_len);
        const int64_t out_dim =
            (tensor.dim() >= 3) ? tensor.size(2) : static_cast<int64_t>(m_meta.cond_dim);
        const size_t total = static_cast<size_t>(out_seq * out_dim);

        out.seq_len = static_cast<int>(out_seq);
        out.cond_dim = static_cast<int>(out_dim);
        out.data.resize(total);
        std::memcpy(out.data.data(), tensor.const_data_ptr<float>(), total * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        m_last_error = std::string("Exception in encode_prompt: ") + e.what();
        return false;
    }
#else
    (void)out;
    m_last_error = "ExecuTorch C++ runtime is not enabled";
    return false;
#endif
}

auto ExecuTorchSamplerBackend::encode_duration(float seconds_total, std::vector<float>& out)
    -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_loaded) {
        m_last_error = "Sampler bundle not loaded";
        return false;
    }

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        float sec_val = seconds_total;
        std::vector<executorch::extension::TensorPtr> keepalive;
        std::vector<executorch::runtime::EValue> inputs;

        auto sec_t =
            executorch::extension::from_blob(&sec_val, {1}, executorch::aten::ScalarType::Float);
        inputs.push_back(sec_t);
        keepalive.push_back(std::move(sec_t));

        append_baseline_params(m_impl->number_cond, "forward", 1, inputs, keepalive);

        auto res = m_impl->number_cond.module->execute("forward", inputs);
        if (!res.ok() || res.get().empty() || !res.get()[0].isTensor()) {
            m_last_error = "ExecuTorch number_cond execution failed";
            return false;
        }
        const auto tensor = res.get()[0].toTensor();
        const size_t total = static_cast<size_t>(tensor.numel());
        out.resize(total);
        std::memcpy(out.data(), tensor.const_data_ptr<float>(), total * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        m_last_error = std::string("Exception in encode_duration: ") + e.what();
        return false;
    }
#else
    (void)seconds_total;
    (void)out;
    m_last_error = "ExecuTorch C++ runtime is not enabled";
    return false;
#endif
}

auto ExecuTorchSamplerBackend::run_dit_step(const float* x, int latent_frames, float t,
                                            const float* cross_attn_cond, int cond_seq_len,
                                            const float* global_cond, int global_dim,
                                            const float* input_concat_cond,
                                            const uint8_t* padding_mask, float* v_out) -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_loaded) {
        m_last_error = "Sampler bundle not loaded";
        return false;
    }
    if (x == nullptr || cross_attn_cond == nullptr || v_out == nullptr || latent_frames <= 0 ||
        cond_seq_len <= 0) {
        m_last_error = "Invalid null or non-positive input to run_dit_step";
        return false;
    }

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        const int lat_dim = std::max(1, m_meta.latent_dim);
        const int c_dim = std::max(1, m_meta.cond_dim);
        const int g_dim = std::max(1, (global_dim > 0) ? global_dim : m_meta.global_cond_dim);
        const int ic_dim = std::max(1, m_meta.input_concat_dim);

        std::vector<float> x_buf(x, x + static_cast<size_t>(lat_dim) * latent_frames);
        float t_val = t;
        std::vector<float> ca_buf(cross_attn_cond,
                                  cross_attn_cond + static_cast<size_t>(cond_seq_len) * c_dim);
        std::vector<float> gc_buf(static_cast<size_t>(g_dim), 0.0f);
        if (global_cond != nullptr && global_dim > 0) {
            std::memcpy(gc_buf.data(), global_cond,
                        static_cast<size_t>(std::min(global_dim, g_dim)) * sizeof(float));
        }

        std::vector<float> ic_buf(static_cast<size_t>(ic_dim) * latent_frames, 0.0f);
        if (input_concat_cond != nullptr && m_meta.input_concat_dim > 0) {
            std::memcpy(ic_buf.data(), input_concat_cond,
                        static_cast<size_t>(ic_dim) * latent_frames * sizeof(float));
        }

        std::vector<int64_t> pm_buf(static_cast<size_t>(latent_frames), 1);
        if (padding_mask != nullptr) {
            for (int f = 0; f < latent_frames; ++f) {
                pm_buf[static_cast<size_t>(f)] = padding_mask[f] ? 1 : 0;
            }
        }

        std::vector<executorch::extension::TensorPtr> keepalive;
        std::vector<executorch::runtime::EValue> inputs;
        auto push_float = [&](float* ptr, std::vector<executorch::aten::SizesType> sizes) {
            auto tp = executorch::extension::from_blob(ptr, std::move(sizes),
                                                       executorch::aten::ScalarType::Float);
            inputs.push_back(tp);
            keepalive.push_back(std::move(tp));
        };

        push_float(x_buf.data(), {1, lat_dim, latent_frames});
        push_float(&t_val, {1});
        push_float(ca_buf.data(), {1, cond_seq_len, c_dim});
        push_float(gc_buf.data(), {1, g_dim});
        push_float(ic_buf.data(), {1, ic_dim, latent_frames});

        auto pm_t = executorch::extension::from_blob(pm_buf.data(), {1, latent_frames},
                                                     executorch::aten::ScalarType::Long);
        inputs.push_back(pm_t);
        keepalive.push_back(std::move(pm_t));

        append_baseline_params(m_impl->dit, "forward", 6, inputs, keepalive);

        auto res = m_impl->dit.module->execute("forward", inputs);
        if (!res.ok() || res.get().empty() || !res.get()[0].isTensor()) {
            m_last_error = "ExecuTorch dit forward execution failed";
            return false;
        }
        const auto out_tensor = res.get()[0].toTensor();
        const size_t expected = static_cast<size_t>(lat_dim) * static_cast<size_t>(latent_frames);
        const size_t copy_elems = std::min(expected, static_cast<size_t>(out_tensor.numel()));
        std::memcpy(v_out, out_tensor.const_data_ptr<float>(), copy_elems * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        m_last_error = std::string("Exception in run_dit_step: ") + e.what();
        return false;
    }
#else
    (void)x;
    (void)latent_frames;
    (void)t;
    (void)cross_attn_cond;
    (void)cond_seq_len;
    (void)global_cond;
    (void)global_dim;
    (void)input_concat_cond;
    (void)padding_mask;
    (void)v_out;
    m_last_error = "ExecuTorch C++ runtime is not enabled";
    return false;
#endif
}

auto ExecuTorchSamplerBackend::decode_latents(const float* z, int latent_frames,
                                              std::vector<std::vector<float>>& audio_out) -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_loaded) {
        m_last_error = "Sampler bundle not loaded";
        return false;
    }
    if (z == nullptr || latent_frames <= 0) {
        m_last_error = "Invalid latent buffer in decode_latents";
        return false;
    }

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        const int lat_dim = std::max(1, m_meta.latent_dim);
        std::vector<float> z_buf(z, z + static_cast<size_t>(lat_dim) * latent_frames);

        ComponentProgram& prog = m_impl->separate_decode ? m_impl->ae_decode : m_impl->ae_main;
        const std::string method = m_impl->separate_decode ? "forward" : "decode";
        if (!prog.module) {
            m_last_error = "Autoencoder decode module is not loaded";
            return false;
        }

        std::vector<executorch::extension::TensorPtr> keepalive;
        std::vector<executorch::runtime::EValue> inputs;
        auto z_t = executorch::extension::from_blob(z_buf.data(), {1, lat_dim, latent_frames},
                                                    executorch::aten::ScalarType::Float);
        inputs.push_back(z_t);
        keepalive.push_back(std::move(z_t));

        append_baseline_params(prog, method, 1, inputs, keepalive);

        auto res = prog.module->execute(method, inputs);
        if (!res.ok() || res.get().empty() || !res.get()[0].isTensor()) {
            m_last_error = "ExecuTorch autoencoder decode failed";
            return false;
        }
        const auto out_tensor = res.get()[0].toTensor();
        const int out_ch = (out_tensor.dim() >= 2) ? static_cast<int>(out_tensor.size(1))
                                                   : std::max(1, m_meta.io_channels);
        const int out_samples = (out_tensor.dim() >= 3)
                                    ? static_cast<int>(out_tensor.size(2))
                                    : static_cast<int>(out_tensor.numel() / std::max(1, out_ch));

        const float* src = out_tensor.const_data_ptr<float>();
        audio_out.assign(static_cast<size_t>(out_ch),
                         std::vector<float>(static_cast<size_t>(out_samples), 0.0f));
        for (int c = 0; c < out_ch; ++c) {
            std::memcpy(audio_out[static_cast<size_t>(c)].data(),
                        src + static_cast<size_t>(c) * out_samples,
                        static_cast<size_t>(out_samples) * sizeof(float));
        }
        return true;
    } catch (const std::exception& e) {
        m_last_error = std::string("Exception in decode_latents: ") + e.what();
        return false;
    }
#else
    (void)z;
    (void)latent_frames;
    (void)audio_out;
    m_last_error = "ExecuTorch C++ runtime is not enabled";
    return false;
#endif
}

auto ExecuTorchSamplerBackend::encode_audio(const std::vector<const float*>& audio_in,
                                            int num_samples, std::vector<float>& z_out,
                                            int& out_latent_frames) -> bool {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_loaded) {
        m_last_error = "Sampler bundle not loaded";
        return false;
    }
    if (audio_in.empty() || num_samples <= 0) {
        m_last_error = "Invalid audio input to encode_audio";
        return false;
    }

#if defined(NETBENDER_HAS_EXECUTORCH) && NETBENDER_HAS_EXECUTORCH
    try {
        const int in_ch = std::max(1, m_meta.io_channels);
        const int ratio = std::max(1, m_meta.downsampling_ratio);
        const int target_samples = (m_meta.compiled_latent_frames > 0)
                                       ? (m_meta.compiled_latent_frames * ratio)
                                       : num_samples;

        std::vector<float> audio_buf(static_cast<size_t>(in_ch) * target_samples, 0.0f);
        const int copy_len = std::min(num_samples, target_samples);
        for (int c = 0; c < in_ch && c < static_cast<int>(audio_in.size()); ++c) {
            if (audio_in[static_cast<size_t>(c)] != nullptr) {
                std::memcpy(audio_buf.data() + static_cast<size_t>(c) * target_samples,
                            audio_in[static_cast<size_t>(c)],
                            static_cast<size_t>(copy_len) * sizeof(float));
            }
        }

        ComponentProgram& prog = m_impl->separate_encode ? m_impl->ae_encode : m_impl->ae_main;
        const std::string method = m_impl->separate_encode ? "forward" : "encode";
        if (!prog.module) {
            m_last_error = "Autoencoder encode module is not loaded";
            return false;
        }

        std::vector<executorch::extension::TensorPtr> keepalive;
        std::vector<executorch::runtime::EValue> inputs;
        auto a_t = executorch::extension::from_blob(audio_buf.data(), {1, in_ch, target_samples},
                                                    executorch::aten::ScalarType::Float);
        inputs.push_back(a_t);
        keepalive.push_back(std::move(a_t));

        append_baseline_params(prog, method, 1, inputs, keepalive);

        auto res = prog.module->execute(method, inputs);
        if (!res.ok() || res.get().empty() || !res.get()[0].isTensor()) {
            m_last_error = "ExecuTorch autoencoder encode failed";
            return false;
        }
        const auto out_tensor = res.get()[0].toTensor();
        const int lat_dim = (out_tensor.dim() >= 2) ? static_cast<int>(out_tensor.size(1))
                                                    : std::max(1, m_meta.latent_dim);
        out_latent_frames = (out_tensor.dim() >= 3)
                                ? static_cast<int>(out_tensor.size(2))
                                : static_cast<int>(out_tensor.numel() / std::max(1, lat_dim));
        const size_t total = static_cast<size_t>(out_tensor.numel());
        z_out.resize(total);
        std::memcpy(z_out.data(), out_tensor.const_data_ptr<float>(), total * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        m_last_error = std::string("Exception in encode_audio: ") + e.what();
        return false;
    }
#else
    (void)audio_in;
    (void)num_samples;
    (void)z_out;
    (void)out_latent_frames;
    m_last_error = "ExecuTorch C++ runtime is not enabled";
    return false;
#endif
}

}  // namespace netbender::sampler
