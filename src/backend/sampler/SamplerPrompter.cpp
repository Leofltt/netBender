#include "SamplerPrompter.h"
#include "DiffusionSchedule.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace netbender::sampler {

namespace {

auto is_cancelled(const std::atomic<bool>* cancel_flag) noexcept -> bool {
    return cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire);
}

auto assemble_cross_attn(const std::vector<std::string>& cond_ids, const PromptEmbedding& prompt,
                         const std::vector<float>& time_embed, int cond_dim, int& out_seq_len)
    -> std::vector<float> {
    std::vector<float> out;
    out_seq_len = 0;

    for (const auto& cid : cond_ids) {
        if (cid == "prompt" && prompt.seq_len > 0 && !prompt.data.empty()) {
            const int src_dim = std::max(1, prompt.cond_dim);
            const int copy_dim = std::min(cond_dim, src_dim);
            const size_t base = out.size();
            out.resize(base + static_cast<size_t>(prompt.seq_len) * cond_dim, 0.0f);
            for (int s = 0; s < prompt.seq_len; ++s) {
                std::memcpy(out.data() + base + static_cast<size_t>(s) * cond_dim,
                            prompt.data.data() + static_cast<size_t>(s) * src_dim,
                            static_cast<size_t>(copy_dim) * sizeof(float));
            }
            out_seq_len += prompt.seq_len;
        } else if (cid == "seconds_total" && !time_embed.empty()) {
            const int copy_dim = std::min(cond_dim, static_cast<int>(time_embed.size()));
            const size_t base = out.size();
            out.resize(base + static_cast<size_t>(cond_dim), 0.0f);
            std::memcpy(out.data() + base, time_embed.data(),
                        static_cast<size_t>(copy_dim) * sizeof(float));
            out_seq_len += 1;
        }
    }

    if (out_seq_len == 0) {
        out_seq_len = std::max(1, prompt.seq_len);
        out.assign(static_cast<size_t>(out_seq_len) * cond_dim, 0.0f);
        if (!prompt.data.empty()) {
            const size_t n = std::min(out.size(), prompt.data.size());
            std::memcpy(out.data(), prompt.data.data(), n * sizeof(float));
        }
    }
    return out;
}

}  // namespace

SamplerPrompter::SamplerPrompter(ISamplerBackend& backend, const TextTokenizer& tokenizer)
    : m_backend(backend), m_tokenizer(tokenizer) {}

auto SamplerPrompter::generate(const GenerationRequest& req, GeneratedSample& out,
                               const std::atomic<bool>* cancel_flag, ProgressCallback progress_cb)
    -> bool {
    m_last_error.clear();
    out = GeneratedSample{};
    const auto t_start = std::chrono::steady_clock::now();

    auto emit = [&](GenerationProgress::Stage stage, int cur, int tot, const std::string& msg) {
        if (progress_cb) {
            progress_cb(GenerationProgress{stage, cur, tot, msg});
        }
    };

    if (!m_backend.is_loaded()) {
        m_last_error = "Sampler backend is not loaded";
        emit(GenerationProgress::Stage::Error, 0, 0, m_last_error);
        return false;
    }
    if (!m_tokenizer.is_loaded()) {
        m_last_error = "Text tokenizer is not loaded";
        emit(GenerationProgress::Stage::Error, 0, 0, m_last_error);
        return false;
    }

    const SamplerBundleMetadata& meta = m_backend.get_metadata();
    const float max_dur = std::max(0.1f, meta.max_duration_sec);
    const float duration_sec = std::clamp(req.duration_sec, 0.1f, max_dur);
    const int steps = std::max(1, (req.steps > 0) ? req.steps : meta.default_steps);
    const float cfg_scale = (req.cfg_scale >= 0.0f) ? req.cfg_scale : meta.default_cfg_scale;
    const float apg_scale = (req.apg_scale >= 0.0f) ? req.apg_scale : meta.default_apg_scale;
    const SamplerAlgorithm sampler = req.use_default_sampler ? meta.default_sampler : req.sampler;

    if (is_cancelled(cancel_flag)) {
        m_last_error = "Generation cancelled";
        emit(GenerationProgress::Stage::Cancelled, 0, steps, m_last_error);
        return false;
    }

    // 1. Tokenize positive (and optional negative) prompt
    emit(GenerationProgress::Stage::Tokenizing, 0, steps, "Tokenizing prompt");
    const TokenizedPrompt pos_tokens = m_tokenizer.encode(req.prompt);
    const bool cfg_active =
        (std::abs(cfg_scale - 1.0f) > 1e-5f) && !meta.cross_attn_cond_ids.empty();

    // 2. Encode prompt and duration conditioning
    emit(GenerationProgress::Stage::EncodingPrompt, 0, steps, "Encoding conditioning");
    PromptEmbedding pos_embed;
    if (!m_backend.encode_prompt(pos_tokens, pos_embed)) {
        m_last_error = "encode_prompt failed: " + m_backend.get_last_error();
        emit(GenerationProgress::Stage::Error, 0, steps, m_last_error);
        return false;
    }

    std::vector<float> time_embed;
    if (!m_backend.encode_duration(duration_sec, time_embed)) {
        m_last_error = "encode_duration failed: " + m_backend.get_last_error();
        emit(GenerationProgress::Stage::Error, 0, steps, m_last_error);
        return false;
    }

    if (is_cancelled(cancel_flag)) {
        m_last_error = "Generation cancelled";
        emit(GenerationProgress::Stage::Cancelled, 0, steps, m_last_error);
        return false;
    }

    const int cond_dim = std::max(1, meta.cond_dim);
    int cond_seq_len = 0;
    const std::vector<float> cross_attn_cond = assemble_cross_attn(
        meta.cross_attn_cond_ids, pos_embed, time_embed, cond_dim, cond_seq_len);

    std::vector<float> uncond_cross_attn;
    if (cfg_active) {
        if (!req.negative_prompt.empty()) {
            const TokenizedPrompt neg_tokens = m_tokenizer.encode(req.negative_prompt);
            PromptEmbedding neg_embed;
            if (!m_backend.encode_prompt(neg_tokens, neg_embed)) {
                m_last_error = "negative encode_prompt failed: " + m_backend.get_last_error();
                emit(GenerationProgress::Stage::Error, 0, steps, m_last_error);
                return false;
            }
            int neg_seq_len = 0;
            uncond_cross_attn = assemble_cross_attn(meta.cross_attn_cond_ids, neg_embed, time_embed,
                                                    cond_dim, neg_seq_len);
            if (uncond_cross_attn.size() != cross_attn_cond.size()) {
                uncond_cross_attn.resize(cross_attn_cond.size(), 0.0f);
            }
        } else {
            uncond_cross_attn.assign(cross_attn_cond.size(), 0.0f);
        }
    }

    const int global_dim = std::max(0, meta.global_cond_dim);
    std::vector<float> global_cond(static_cast<size_t>(std::max(1, global_dim)), 0.0f);
    if (global_dim > 0) {
        const bool wants_seconds =
            std::find(meta.global_cond_ids.begin(), meta.global_cond_ids.end(), "seconds_total") !=
            meta.global_cond_ids.end();
        if (wants_seconds && !time_embed.empty()) {
            const size_t n = std::min(global_cond.size(), time_embed.size());
            std::memcpy(global_cond.data(), time_embed.data(), n * sizeof(float));
        }
    }

    // 3. Prepare latent canvas, padding mask, and timestep schedule
    const int latent_dim = std::max(1, meta.latent_dim);
    const int latent_frames = compute_aligned_latent_frames(
        duration_sec, meta.duration_padding_sec, meta.sample_rate, meta.downsampling_ratio,
        meta.latent_align, meta.compiled_latent_frames);
    const int effective_frames =
        compute_effective_latent_frames(duration_sec, meta.sample_rate, meta.downsampling_ratio);
    const std::vector<uint8_t> padding_mask =
        build_padding_mask(latent_frames, effective_frames, meta.duration_padding_sec,
                           meta.sample_rate, meta.downsampling_ratio);

    const int ic_dim = std::max(1, meta.input_concat_dim);
    const std::vector<float> input_concat(
        static_cast<size_t>(ic_dim) * static_cast<size_t>(latent_frames), 0.0f);

    const size_t latent_elems =
        static_cast<size_t>(latent_dim) * static_cast<size_t>(latent_frames);
    std::vector<float> x(latent_elems, 0.0f);
    netbender::bending::Xoshiro128Plus rng(req.seed);
    fill_standard_normal(x.data(), latent_elems, rng);

    const std::vector<float> sigmas =
        build_schedule(steps, meta.dist_shift, effective_frames, 1.0f, true);

    std::vector<float> v_cond(latent_elems, 0.0f);
    std::vector<float> v_uncond;
    std::vector<float> v_eff(latent_elems, 0.0f);
    if (cfg_active) {
        v_uncond.resize(latent_elems, 0.0f);
    }
    std::vector<float> dpmpp_old_denoised;

    // 4. Denoising diffusion loop
    for (int i = 0; i < steps; ++i) {
        if (is_cancelled(cancel_flag)) {
            m_last_error = "Generation cancelled";
            emit(GenerationProgress::Stage::Cancelled, i, steps, m_last_error);
            return false;
        }

        const float t_curr = sigmas[static_cast<size_t>(i)];
        const float t_next = sigmas[static_cast<size_t>(i + 1)];
        const float t_prev = (i > 0) ? sigmas[static_cast<size_t>(i - 1)] : 0.0f;

        emit(GenerationProgress::Stage::Diffusing, i + 1, steps,
             "Step " + std::to_string(i + 1) + "/" + std::to_string(steps));

        if (!m_backend.run_dit_step(x.data(), latent_frames, t_curr, cross_attn_cond.data(),
                                    cond_seq_len, global_cond.data(), global_dim,
                                    input_concat.data(), padding_mask.data(), v_cond.data())) {
            m_last_error = "run_dit_step failed at step " + std::to_string(i) + ": " +
                           m_backend.get_last_error();
            emit(GenerationProgress::Stage::Error, i + 1, steps, m_last_error);
            return false;
        }

        const float* step_v = v_cond.data();
        if (cfg_active) {
            if (!m_backend.run_dit_step(x.data(), latent_frames, t_curr, uncond_cross_attn.data(),
                                        cond_seq_len, global_cond.data(), global_dim,
                                        input_concat.data(), padding_mask.data(),
                                        v_uncond.data())) {
                m_last_error = "uncond run_dit_step failed at step " + std::to_string(i) + ": " +
                               m_backend.get_last_error();
                emit(GenerationProgress::Stage::Error, i + 1, steps, m_last_error);
                return false;
            }
            blend_cfg_apg(x.data(), v_cond.data(), v_uncond.data(), latent_dim, latent_frames,
                          t_curr, cfg_scale, apg_scale, req.cfg_norm_threshold, padding_mask.data(),
                          v_eff.data());
            step_v = v_eff.data();
        }

        switch (sampler) {
            case SamplerAlgorithm::PingPong:
                step_pingpong(x.data(), step_v, latent_elems, t_curr, t_next, rng);
                break;
            case SamplerAlgorithm::Euler:
                step_euler(x.data(), step_v, latent_elems, t_curr, t_next);
                break;
            case SamplerAlgorithm::DpmPP:
                step_dpmpp(x.data(), step_v, latent_elems, t_curr, t_next, t_prev, i > 0,
                           dpmpp_old_denoised);
                break;
        }
    }

    // 5. Optional SAME-S decoding + valid-region masking + duration trim
    if (req.decode_audio) {
        if (is_cancelled(cancel_flag)) {
            m_last_error = "Generation cancelled";
            emit(GenerationProgress::Stage::Cancelled, steps, steps, m_last_error);
            return false;
        }
        emit(GenerationProgress::Stage::Decoding, steps, steps, "Decoding audio");

        std::vector<std::vector<float>> decoded;
        if (!m_backend.decode_latents(x.data(), latent_frames, decoded) || decoded.empty()) {
            m_last_error = "decode_latents failed: " + m_backend.get_last_error();
            emit(GenerationProgress::Stage::Error, steps, steps, m_last_error);
            return false;
        }

        const int ratio = std::max(1, meta.downsampling_ratio);
        const int target_samples = std::max(
            1, static_cast<int>(duration_sec * static_cast<float>(std::max(1, meta.sample_rate))));

        for (auto& ch : decoded) {
            const int n_s = static_cast<int>(ch.size());
            for (int s = 0; s < n_s; ++s) {
                const int frame_idx = std::min(latent_frames - 1, s / ratio);
                if (padding_mask[static_cast<size_t>(frame_idx)] == 0) {
                    ch[static_cast<size_t>(s)] = 0.0f;
                }
            }
            if (static_cast<int>(ch.size()) > target_samples) {
                ch.resize(static_cast<size_t>(target_samples));
            }
        }
        out.audio = std::move(decoded);
    }

    const auto t_end = std::chrono::steady_clock::now();
    out.latents = std::move(x);
    out.latent_dim = latent_dim;
    out.latent_frames = latent_frames;
    out.valid_latent_frames = std::min(latent_frames, effective_frames);
    out.sample_rate = meta.sample_rate;
    out.duration_sec = duration_sec;
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    emit(GenerationProgress::Stage::Done, steps, steps, "Generation complete");
    return true;
}

}  // namespace netbender::sampler
