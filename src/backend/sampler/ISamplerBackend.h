#pragma once

#include "SamplerTypes.h"

#include <string>
#include <vector>

namespace netbender::sampler {

/**
 * Runtime-agnostic interface for Stable Audio 3 multi-component generative inference
 * (prompting head, DiT velocity predictor, and autoencoder).
 * Implementations (ExecuTorch, ONNX, LibTorch) isolate runtime headers in their `.cpp` files.
 */
class ISamplerBackend {
public:
    virtual ~ISamplerBackend() = default;

    /** Loads a `.nbsampler.json` bundle and prepares inference components. Returns 0 on success. */
    virtual auto load_bundle(const std::string& manifest_path, bool use_gpu = true) -> int = 0;
    [[nodiscard]] virtual auto is_loaded() const -> bool = 0;
    [[nodiscard]] virtual auto get_metadata() const -> const SamplerBundleMetadata& = 0;
    [[nodiscard]] virtual auto get_backend_name() const -> std::string = 0;
    [[nodiscard]] virtual auto get_last_error() const -> std::string = 0;

    /** Encodes padded token IDs + attention mask into prompt conditioning embeddings. */
    virtual auto encode_prompt(const TokenizedPrompt& tokens, PromptEmbedding& out) -> bool = 0;

    /** Encodes `seconds_total` scalar into a single time-conditioning vector `[cond_dim]`. */
    virtual auto encode_duration(float seconds_total, std::vector<float>& out) -> bool = 0;

    /** Executes one DiT step at float32 timestep `t` producing velocity prediction `v_out`. */
    virtual auto run_dit_step(const float* x, int latent_frames, float t,
                              const float* cross_attn_cond, int cond_seq_len,
                              const float* global_cond, int global_dim,
                              const float* input_concat_cond, const uint8_t* padding_mask,
                              float* v_out) -> bool = 0;

    /** Decodes channel-major latents `z [latent_dim, latent_frames]` to multi-channel audio. */
    virtual auto decode_latents(const float* z, int latent_frames,
                                std::vector<std::vector<float>>& audio_out) -> bool = 0;

    /** Encodes multi-channel audio into channel-major latents `z_out [latent_dim * frames]`. */
    virtual auto encode_audio(const std::vector<const float*>& audio_in, int num_samples,
                              std::vector<float>& z_out, int& out_latent_frames) -> bool = 0;
};

}  // namespace netbender::sampler
