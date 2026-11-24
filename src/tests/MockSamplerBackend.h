#pragma once

#include "../backend/sampler/ISamplerBackend.h"
#include "../backend/sampler/SamplerBundleLoader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace netbender::sampler::test {

/**
 * Zero-dependency in-memory mock implementation of `ISamplerBackend` for headless
 * `SamplerPrompter` and pipeline unit tests.
 */
class MockSamplerBackend final : public ISamplerBackend {
public:
    bool force_fail_prompt{false};
    int force_fail_dit_at_step{-1};
    bool force_fail_decode{false};

    int encode_prompt_calls{0};
    int encode_duration_calls{0};
    int dit_step_calls{0};
    int decode_calls{0};
    int encode_audio_calls{0};
    std::vector<float> recorded_timesteps;
    std::vector<int> recorded_dit_frames;
    std::vector<int> recorded_valid_mask_frames;

    auto init_synthetic(const std::string& runtime = "executorch",
                        const std::string& delegate = "coreml", int compiled_latent_frames = 16,
                        int sample_rate = 44100, int downsampling_ratio = 4096, int latent_dim = 8,
                        int cond_dim = 16) -> void {
        m_meta = SamplerBundleMetadata{};
        m_meta.format = "netbender.sampler";
        m_meta.version = 1;
        m_meta.architecture = "stable_audio_3";
        m_meta.variant = "small-music";
        m_meta.runtime = runtime;
        m_meta.delegate = delegate;
        m_meta.sample_rate = sample_rate;
        m_meta.io_channels = 2;
        m_meta.latent_dim = latent_dim;
        m_meta.downsampling_ratio = downsampling_ratio;
        m_meta.latent_align = 2;
        m_meta.cond_dim = cond_dim;
        m_meta.global_cond_dim = cond_dim;
        m_meta.input_concat_dim = latent_dim + 1;
        m_meta.compiled_latent_frames = compiled_latent_frames;
        m_meta.max_duration_sec = 30.0f;
        m_meta.duration_padding_sec = 1.0f;
        m_meta.objective = "rf_denoiser";
        m_meta.default_sampler = SamplerAlgorithm::PingPong;
        m_meta.default_steps = 8;
        m_meta.default_cfg_scale = 1.0f;
        m_meta.default_apg_scale = 1.0f;
        m_meta.max_token_length = 16;
        m_loaded = true;
        m_last_error.clear();
    }

    auto load_bundle(const std::string& manifest_path, bool use_gpu = true) -> int override {
        (void)use_gpu;
        SamplerBundleMetadata parsed;
        std::string err;
        if (!parse_sampler_bundle_manifest(manifest_path, parsed, err)) {
            m_last_error = err;
            m_loaded = false;
            return -1;
        }
        m_meta = std::move(parsed);
        m_loaded = true;
        m_last_error.clear();
        return 0;
    }

    [[nodiscard]] auto is_loaded() const -> bool override { return m_loaded; }
    [[nodiscard]] auto get_metadata() const -> const SamplerBundleMetadata& override {
        return m_meta;
    }
    [[nodiscard]] auto get_backend_name() const -> std::string override {
        return "MockSamplerBackend";
    }
    [[nodiscard]] auto get_last_error() const -> std::string override { return m_last_error; }

    auto encode_prompt(const TokenizedPrompt& tokens, PromptEmbedding& out) -> bool override {
        ++encode_prompt_calls;
        if (force_fail_prompt) {
            m_last_error = "forced encode_prompt failure";
            return false;
        }
        const int seq = std::max(1, static_cast<int>(tokens.input_ids.size()));
        const int dim = std::max(1, m_meta.cond_dim);
        out.seq_len = seq;
        out.cond_dim = dim;
        out.data.assign(static_cast<size_t>(seq) * dim, 0.0f);
        for (int s = 0; s < seq; ++s) {
            const float tok_v = static_cast<float>(tokens.input_ids[static_cast<size_t>(s)] + 1);
            const float m = static_cast<float>(tokens.attention_mask[static_cast<size_t>(s)]);
            for (int d = 0; d < dim; ++d) {
                out.data[static_cast<size_t>(s) * dim + static_cast<size_t>(d)] =
                    m * (0.01f * tok_v + 0.001f * static_cast<float>(d));
            }
        }
        return true;
    }

    auto encode_duration(float seconds_total, std::vector<float>& out) -> bool override {
        ++encode_duration_calls;
        const int dim = std::max(1, m_meta.cond_dim);
        out.assign(static_cast<size_t>(dim), 0.0f);
        for (int d = 0; d < dim; ++d) {
            out[static_cast<size_t>(d)] =
                std::sin(seconds_total * 0.25f + static_cast<float>(d) * 0.1f);
        }
        return true;
    }

    auto run_dit_step(const float* x, int latent_frames, float t, const float* cross_attn_cond,
                      int cond_seq_len, const float* global_cond, int global_dim,
                      const float* input_concat_cond, const uint8_t* padding_mask, float* v_out)
        -> bool override {
        (void)input_concat_cond;
        if (force_fail_dit_at_step >= 0 && dit_step_calls == force_fail_dit_at_step) {
            m_last_error = "forced DiT step failure";
            ++dit_step_calls;
            return false;
        }
        ++dit_step_calls;
        recorded_timesteps.push_back(t);
        recorded_dit_frames.push_back(latent_frames);

        int valid_count = 0;
        if (padding_mask != nullptr) {
            for (int f = 0; f < latent_frames; ++f) {
                if (padding_mask[f] != 0) {
                    ++valid_count;
                }
            }
        }
        recorded_valid_mask_frames.push_back(valid_count);

        float cond_bias = 0.0f;
        const int c_dim = std::max(1, m_meta.cond_dim);
        for (int i = 0; i < std::min(cond_seq_len * c_dim, 64); ++i) {
            cond_bias += cross_attn_cond[i];
        }
        if (global_cond != nullptr && global_dim > 0) {
            cond_bias += 0.1f * global_cond[0];
        }

        // Pull x smoothly toward a bounded target determined by conditioning
        const float target = std::tanh(cond_bias * 0.05f) * 0.5f;
        const int lat_dim = std::max(1, m_meta.latent_dim);
        const size_t n = static_cast<size_t>(lat_dim) * static_cast<size_t>(latent_frames);
        for (size_t i = 0; i < n; ++i) {
            v_out[i] = x[i] - target;
        }
        return true;
    }

    auto decode_latents(const float* z, int latent_frames,
                        std::vector<std::vector<float>>& audio_out) -> bool override {
        ++decode_calls;
        if (force_fail_decode) {
            m_last_error = "forced decode failure";
            return false;
        }
        const int ch = std::max(1, m_meta.io_channels);
        const int ratio = std::max(1, m_meta.downsampling_ratio);
        const int total_samples = std::max(1, latent_frames * ratio);
        audio_out.assign(static_cast<size_t>(ch),
                         std::vector<float>(static_cast<size_t>(total_samples), 0.0f));
        for (int c = 0; c < ch; ++c) {
            for (int f = 0; f < latent_frames; ++f) {
                const float lv =
                    std::tanh(z[static_cast<size_t>(c % std::max(1, m_meta.latent_dim)) *
                                    static_cast<size_t>(latent_frames) +
                                static_cast<size_t>(f)] +
                              0.25f);
                for (int s = 0; s < ratio; ++s) {
                    const int idx = f * ratio + s;
                    audio_out[static_cast<size_t>(c)][static_cast<size_t>(idx)] =
                        0.4f * lv * std::sin(0.05f * static_cast<float>(idx + c));
                }
            }
        }
        return true;
    }

    auto encode_audio(const std::vector<const float*>& audio_in, int num_samples,
                      std::vector<float>& z_out, int& out_latent_frames) -> bool override {
        ++encode_audio_calls;
        (void)audio_in;
        const int ratio = std::max(1, m_meta.downsampling_ratio);
        out_latent_frames = (m_meta.compiled_latent_frames > 0)
                                ? m_meta.compiled_latent_frames
                                : std::max(1, (num_samples + ratio - 1) / ratio);
        z_out.assign(static_cast<size_t>(m_meta.latent_dim) * out_latent_frames, 0.0f);
        return true;
    }

private:
    SamplerBundleMetadata m_meta;
    bool m_loaded{false};
    std::string m_last_error;
};

}  // namespace netbender::sampler::test
