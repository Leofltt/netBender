#include "SamplerPrompter.h"
#include "DiffusionSchedule.h"
#include "../bending/PromptBendingEngine.h"

#include <algorithm>
#include <array>
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
                               const std::atomic<bool>* cancel_flag, ProgressCallback progress_cb,
                               PromptEmbeddingCache* prompt_cache) -> bool {
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

    // 1. Check cancellation & CFG setup
    emit(GenerationProgress::Stage::Tokenizing, 0, steps, "Tokenizing prompt");
    const bool cfg_active =
        (std::abs(cfg_scale - 1.0f) > 1e-5f) && !meta.cross_attn_cond_ids.empty();

    // 2. Encode prompt (or A/B/C/D morph slots) with optional cache and duration conditioning
    emit(GenerationProgress::Stage::EncodingPrompt, 0, steps, "Encoding conditioning");
    const auto& pb = req.prompt_bend;
    netbender::bending::Xoshiro128Plus rng(req.seed);

    auto encode_cached = [&](const std::string& text, PromptEmbedding& out_emb) -> bool {
        if (prompt_cache != nullptr) {
            const auto it = prompt_cache->find(text);
            if (it != prompt_cache->end()) {
                out_emb = it->second;
                return true;
            }
        }
        const TokenizedPrompt toks = m_tokenizer.encode(text);
        if (!m_backend.encode_prompt(toks, out_emb)) {
            return false;
        }
        if (prompt_cache != nullptr) {
            if (prompt_cache->size() >= 16) {
                prompt_cache->clear();
            }
            (*prompt_cache)[text] = out_emb;
        }
        return true;
    };

    PromptEmbedding pos_embed;
    std::array<PromptEmbedding, 4> bend_slots;
    std::array<bool, 4> active_slots{false, false, false, false};
    int second_active_slot = 0;
    if (pb.enabled) {
        int active_count = 0;
        for (int s = 0; s < 4; ++s) {
            const std::string& p_txt = pb.prompts[static_cast<size_t>(s)];
            if (!p_txt.empty()) {
                active_slots[static_cast<size_t>(s)] = true;
                if (!encode_cached(p_txt, bend_slots[static_cast<size_t>(s)])) {
                    m_last_error = "encode_prompt failed: " + m_backend.get_last_error();
                    emit(GenerationProgress::Stage::Error, 0, steps, m_last_error);
                    return false;
                }
                if (active_count == 1) {
                    second_active_slot = s;
                } else if (active_count == 0) {
                    second_active_slot = s;
                }
                ++active_count;
            }
        }
        if (active_count == 0) {
            active_slots[0] = true;
            if (!encode_cached(req.prompt, bend_slots[0])) {
                m_last_error = "encode_prompt(A) failed: " + m_backend.get_last_error();
                emit(GenerationProgress::Stage::Error, 0, steps, m_last_error);
                return false;
            }
        }
        pos_embed = netbender::bending::PromptBendingEngine::blend_embeddings_masked(
            bend_slots, active_slots, pb.morph_x, pb.morph_y, pb.morph_law);
        if (!pb.drift_per_step) {
            netbender::bending::PromptBendingEngine::apply_embedding_dsp(pos_embed, pb, rng);
        }
    } else {
        if (!encode_cached(req.prompt, pos_embed)) {
            m_last_error = "encode_prompt failed: " + m_backend.get_last_error();
            emit(GenerationProgress::Stage::Error, 0, steps, m_last_error);
            return false;
        }
    }

    std::vector<float> time_embed;
    const float cond_duration_sec = (pb.enabled && pb.duration_lie_sec > 0.0f)
                                        ? std::clamp(pb.duration_lie_sec, 0.5f, max_dur)
                                        : std::min(max_dur, duration_sec + 2.0f);
    if (!m_backend.encode_duration(cond_duration_sec, time_embed)) {
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
    fill_standard_normal(x.data(), latent_elems, rng);
    if (pb.enabled && pb.latent_jitter > 1e-4f) {
        const float a = std::clamp(pb.latent_jitter, 0.0f, 1.0f);
        const float keep = std::sqrt(std::max(0.0f, 1.0f - a * a));
        netbender::bending::Xoshiro128Plus jitter_rng(req.seed ^ 0x9E3779B97F4A7C15ULL);
        std::vector<float> jitter(latent_elems, 0.0f);
        fill_standard_normal(jitter.data(), latent_elems, jitter_rng);
        for (size_t i = 0; i < latent_elems; ++i) {
            x[i] = keep * x[i] + a * jitter[i];
        }
    }

    std::vector<float> sigmas =
        build_schedule(steps, meta.dist_shift, effective_frames, 1.0f, true);
    if (pb.enabled) {
        netbender::bending::PromptBendingEngine::warp_schedule(sigmas, pb.schedule_warp);
    }

    const bool dynamic_step_cond = pb.enabled && (pb.drift_per_step || pb.step_morph_ab > 1e-4f);
    std::vector<float> step_cross_attn = cross_attn_cond;
    const float noise_scale = pb.enabled ? pb.pingpong_noise_scale : 1.0f;

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

        const float* active_cross_attn = cross_attn_cond.data();
        if (dynamic_step_cond) {
            PromptEmbedding step_emb = pos_embed;
            if (pb.step_morph_ab > 1e-4f) {
                const float frac =
                    (steps > 1) ? static_cast<float>(i) / static_cast<float>(steps - 1) : 1.0f;
                const float t_ab = std::clamp(frac * pb.step_morph_ab, 0.0f, 1.0f);
                const auto& target_emb = active_slots[1]
                                             ? bend_slots[1]
                                             : bend_slots[static_cast<size_t>(second_active_slot)];
                step_emb = netbender::bending::PromptBendingEngine::blend_pair(
                    pos_embed, target_emb, t_ab, pb.morph_law);
            }
            if (pb.drift_per_step) {
                netbender::bending::PromptBendingEngine::apply_embedding_dsp(step_emb, pb, rng);
            }
            int step_seq_len = 0;
            step_cross_attn = assemble_cross_attn(meta.cross_attn_cond_ids, step_emb, time_embed,
                                                  cond_dim, step_seq_len);
            active_cross_attn = step_cross_attn.data();
        }

        if (!m_backend.run_dit_step(x.data(), latent_frames, t_curr, active_cross_attn,
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
                step_pingpong(x.data(), step_v, latent_elems, t_curr, t_next, rng, noise_scale);
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
