#pragma once

#include "ISamplerBackend.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace netbender::sampler {

/**
 * ExecuTorch `.pte` implementation of `ISamplerBackend` for Stable Audio 3 bundles.
 * Supports CoreML, XNNPACK, and Portable delegates across static bucketed shapes
 * and optional NBND v2 weight sidecars (`--bend-mode none` or `delegate`).
 *
 * All ExecuTorch headers are strictly isolated in `ExecuTorchSamplerBackend.cpp`
 * inside `executorch_backend_obj` (`NETBENDER_NO_TORCH_HEADERS=1`).
 */
class ExecuTorchSamplerBackend final : public ISamplerBackend {
public:
    ExecuTorchSamplerBackend();
    ~ExecuTorchSamplerBackend() override;

    ExecuTorchSamplerBackend(const ExecuTorchSamplerBackend&) = delete;
    ExecuTorchSamplerBackend& operator=(const ExecuTorchSamplerBackend&) = delete;

    auto load_bundle(const std::string& manifest_path, bool use_gpu = true) -> int override;
    [[nodiscard]] auto is_loaded() const -> bool override;
    [[nodiscard]] auto get_metadata() const -> const SamplerBundleMetadata& override;
    [[nodiscard]] auto get_backend_name() const -> std::string override;
    [[nodiscard]] auto get_last_error() const -> std::string override;

    auto encode_prompt(const TokenizedPrompt& tokens, PromptEmbedding& out) -> bool override;
    auto encode_duration(float seconds_total, std::vector<float>& out) -> bool override;
    auto run_dit_step(const float* x, int latent_frames, float t, const float* cross_attn_cond,
                      int cond_seq_len, const float* global_cond, int global_dim,
                      const float* input_concat_cond, const uint8_t* padding_mask, float* v_out)
        -> bool override;
    auto decode_latents(const float* z, int latent_frames,
                        std::vector<std::vector<float>>& audio_out) -> bool override;
    auto encode_audio(const std::vector<const float*>& audio_in, int num_samples,
                      std::vector<float>& z_out, int& out_latent_frames) -> bool override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    mutable std::mutex m_mutex;
    SamplerBundleMetadata m_meta;
    bool m_loaded{false};
    bool m_use_gpu{true};
    std::string m_last_error;
};

}  // namespace netbender::sampler
