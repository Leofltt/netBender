#include "DiffusionSchedule.h"

#include <algorithm>
#include <cmath>

namespace netbender::sampler {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

auto log_snr_clamped(float t) noexcept -> float {
    const float num = std::max(1.0f - t, 1e-10f);
    const float den = std::max(t, 1e-10f);
    return std::log(num / den);
}

}  // namespace

auto apply_distribution_shift(const std::vector<float>& t, int seq_len, const DistShiftConfig& cfg)
    -> std::vector<float> {
    std::vector<float> out = t;
    const int safe_seq_len = std::max(1, seq_len);

    switch (cfg.type) {
        case DistShiftType::Identity:
            return out;

        case DistShiftType::LogSNR: {
            const float anchor_len = std::max(1.0f, cfg.anchor_length);
            const float logsnr_start =
                cfg.anchor_logsnr -
                cfg.rate * std::log2(static_cast<float>(safe_seq_len) / anchor_len);
            for (size_t i = 0; i < t.size(); ++i) {
                const float ti = t[i];
                if (ti <= 0.0f) {
                    out[i] = 0.0f;
                } else if (ti >= 1.0f) {
                    out[i] = 1.0f;
                } else {
                    const float logsnr = cfg.logsnr_end - ti * (cfg.logsnr_end - logsnr_start);
                    out[i] = 1.0f / (1.0f + std::exp(logsnr));
                }
            }
            return out;
        }

        case DistShiftType::Flux: {
            const float min_l = std::max(1.0f, cfg.min_length);
            const float max_l = std::max(min_l, cfg.max_length);
            const float clamped = std::clamp(static_cast<float>(safe_seq_len), min_l, max_l);
            float denom = std::log(max_l) - std::log(min_l);
            if (denom == 0.0f) {
                denom = 1e-8f;
            }
            const float frac = (std::log(clamped) - std::log(min_l)) / denom;
            const float log_a_min = std::log(std::max(cfg.alpha_min, 1e-8f));
            const float log_a_max = std::log(std::max(cfg.alpha_max, 1e-8f));
            const float alpha = std::exp(log_a_min + frac * (log_a_max - log_a_min));
            for (size_t i = 0; i < t.size(); ++i) {
                const float ti = t[i];
                out[i] = alpha * ti / (1.0f + (alpha - 1.0f) * ti);
            }
            return out;
        }

        case DistShiftType::Full: {
            const float min_l = std::max(1.0f, cfg.min_length);
            const float max_l = std::max(min_l, cfg.max_length);
            const float clamped = std::clamp(static_cast<float>(safe_seq_len), min_l, max_l);
            const float denom = std::max(1e-8f, max_l - min_l);
            const float mu =
                -(cfg.base_shift + (cfg.max_shift - cfg.base_shift) * (clamped - min_l) / denom);
            const float exp_mu = std::exp(mu);
            for (size_t i = 0; i < t.size(); ++i) {
                const float ti = t[i];
                if (ti <= 0.0f) {
                    out[i] = 0.0f;
                } else if (ti >= 1.0f) {
                    out[i] = 1.0f;
                } else {
                    const float t_safe = std::clamp(ti, 1e-7f, 1.0f - 1e-7f);
                    float shifted = 1.0f - exp_mu / (exp_mu + (1.0f / (1.0f - t_safe) - 1.0f));
                    if (cfg.use_sine) {
                        shifted = std::sin(shifted * kPi * 0.5f);
                    }
                    out[i] = shifted;
                }
            }
            return out;
        }
    }
    return out;
}

auto build_schedule(int steps, const DistShiftConfig& cfg, int effective_seq_len, float sigma_max,
                    bool include_endpoint) -> std::vector<float> {
    const int safe_steps = std::max(1, steps);
    const int n_points = include_endpoint ? (safe_steps + 1) : safe_steps;
    std::vector<float> t(static_cast<size_t>(n_points), 0.0f);

    if (include_endpoint) {
        for (int i = 0; i < n_points; ++i) {
            t[static_cast<size_t>(i)] =
                sigma_max * (1.0f - static_cast<float>(i) / static_cast<float>(safe_steps));
        }
    } else {
        for (int i = 0; i < n_points; ++i) {
            t[static_cast<size_t>(i)] =
                sigma_max * (1.0f - static_cast<float>(i) / static_cast<float>(n_points));
        }
    }

    t = apply_distribution_shift(t, std::max(1, effective_seq_len), cfg);
    if (!t.empty()) {
        t[0] = sigma_max;
    }
    return t;
}

auto compute_effective_latent_frames(float duration_sec, int sample_rate, int downsampling_ratio)
    -> int {
    const int sr = std::max(1, sample_rate);
    const int ratio = std::max(1, downsampling_ratio);
    const int audio_samples = std::max(1, static_cast<int>(duration_sec * static_cast<float>(sr)));
    return std::max(1, (audio_samples + ratio - 1) / ratio);
}

auto compute_aligned_latent_frames(float duration_sec, float duration_padding_sec, int sample_rate,
                                   int downsampling_ratio, int latent_align,
                                   int compiled_latent_frames) -> int {
    if (compiled_latent_frames > 0) {
        return compiled_latent_frames;
    }
    const int sr = std::max(1, sample_rate);
    const int ratio = std::max(1, downsampling_ratio);
    const int align = std::max(1, latent_align);
    const int unit = ratio * align;
    const int raw_samples =
        std::max(unit, static_cast<int>((duration_sec + std::max(0.0f, duration_padding_sec)) *
                                        static_cast<float>(sr)));
    const int aligned_samples = ((raw_samples + unit - 1) / unit) * unit;
    return std::max(1, aligned_samples / ratio);
}

auto build_padding_mask(int latent_frames, int effective_frames, float duration_padding_sec,
                        int sample_rate, int downsampling_ratio) -> std::vector<uint8_t> {
    const int frames = std::max(1, latent_frames);
    const int headroom = std::max(
        0, static_cast<int>(duration_padding_sec * static_cast<float>(std::max(1, sample_rate)) /
                            static_cast<float>(std::max(1, downsampling_ratio))));
    const int valid_len = std::clamp(effective_frames + headroom, 1, frames);

    std::vector<uint8_t> mask(static_cast<size_t>(frames), 0);
    for (int i = 0; i < valid_len; ++i) {
        mask[static_cast<size_t>(i)] = 1;
    }
    return mask;
}

auto fill_standard_normal(float* dst, size_t count, netbender::bending::Xoshiro128Plus& rng)
    -> void {
    if (dst == nullptr || count == 0) {
        return;
    }
    size_t i = 0;
    while (i < count) {
        const float u1 = std::max(1e-7f, rng.next_float());
        const float u2 = rng.next_float();
        const float r = std::sqrt(-2.0f * std::log(u1));
        const float theta = kTwoPi * u2;
        dst[i++] = r * std::cos(theta);
        if (i < count) {
            dst[i++] = r * std::sin(theta);
        }
    }
}

auto apg_project(const float* v0, const float* v1, int channels, int frames,
                 const uint8_t* padding_mask, float* parallel_out, float* orthogonal_out) -> void {
    if (v0 == nullptr || v1 == nullptr || channels <= 0 || frames <= 0) {
        return;
    }

    double v1_sq_sum = 0.0;
    double dot = 0.0;

    for (int c = 0; c < channels; ++c) {
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(frames);
        for (int t = 0; t < frames; ++t) {
            if (padding_mask != nullptr && padding_mask[t] == 0) {
                continue;
            }
            const double a = static_cast<double>(v0[base + static_cast<size_t>(t)]);
            const double b = static_cast<double>(v1[base + static_cast<size_t>(t)]);
            v1_sq_sum += b * b;
            dot += a * b;
        }
    }

    const double v1_norm = std::max(std::sqrt(v1_sq_sum), 1e-8);
    const double proj_scalar = dot / (v1_norm * v1_norm);

    for (int c = 0; c < channels; ++c) {
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(frames);
        for (int t = 0; t < frames; ++t) {
            const float m = (padding_mask == nullptr || padding_mask[t] != 0) ? 1.0f : 0.0f;
            const size_t idx = base + static_cast<size_t>(t);
            const float par = static_cast<float>(proj_scalar * static_cast<double>(v1[idx]) *
                                                 static_cast<double>(m));
            const float orth = (v0[idx] - par) * m;
            if (parallel_out != nullptr) {
                parallel_out[idx] = par;
            }
            if (orthogonal_out != nullptr) {
                orthogonal_out[idx] = orth;
            }
        }
    }
}

auto blend_cfg_apg(const float* x, const float* v_cond, const float* v_uncond, int channels,
                   int frames, float sigma, float cfg_scale, float apg_scale,
                   float cfg_norm_threshold, const uint8_t* padding_mask, float* v_out) -> void {
    if (x == nullptr || v_cond == nullptr || v_uncond == nullptr || v_out == nullptr ||
        channels <= 0 || frames <= 0) {
        return;
    }

    const size_t total = static_cast<size_t>(channels) * static_cast<size_t>(frames);
    const float safe_sigma = std::max(sigma, 1e-8f);

    std::vector<float> cond_denoised(total);
    std::vector<float> diff(total);
    for (size_t i = 0; i < total; ++i) {
        const float cd = x[i] - v_cond[i] * safe_sigma;
        const float ud = x[i] - v_uncond[i] * safe_sigma;
        cond_denoised[i] = cd;
        diff[i] = cd - ud;
    }

    if (cfg_norm_threshold > 0.0f) {
        double diff_sq = 0.0;
        for (int c = 0; c < channels; ++c) {
            const size_t base = static_cast<size_t>(c) * static_cast<size_t>(frames);
            for (int t = 0; t < frames; ++t) {
                if (padding_mask != nullptr && padding_mask[t] == 0) {
                    continue;
                }
                const double d = static_cast<double>(diff[base + static_cast<size_t>(t)]);
                diff_sq += d * d;
            }
        }
        const float diff_norm = static_cast<float>(std::sqrt(diff_sq));
        if (diff_norm > cfg_norm_threshold && diff_norm > 0.0f) {
            const float scale_factor = cfg_norm_threshold / diff_norm;
            for (float& d : diff) {
                d *= scale_factor;
            }
        }
    }

    std::vector<float> diff_orth;
    if (apg_scale != 0.0f) {
        diff_orth.resize(total);
        apg_project(diff.data(), cond_denoised.data(), channels, frames, padding_mask, nullptr,
                    diff_orth.data());
    }

    const float guidance = cfg_scale - 1.0f;
    for (size_t i = 0; i < total; ++i) {
        float cfg_diff = diff[i];
        if (apg_scale == 1.0f) {
            cfg_diff = diff_orth[i];
        } else if (apg_scale != 0.0f) {
            cfg_diff = apg_scale * diff_orth[i] + (1.0f - apg_scale) * diff[i];
        }
        const float cfg_denoised = cond_denoised[i] + guidance * cfg_diff;
        v_out[i] = (x[i] - cfg_denoised) / safe_sigma;
    }
}

auto step_pingpong(float* x, const float* v_pred, size_t count, float t_curr, float t_next,
                   netbender::bending::Xoshiro128Plus& rng) -> void {
    if (x == nullptr || v_pred == nullptr || count == 0) {
        return;
    }

    const float one_minus_next = 1.0f - t_next;
    std::vector<float> eps;
    if (t_next != 0.0f) {
        eps.resize(count);
        fill_standard_normal(eps.data(), count, rng);
        for (size_t i = 0; i < count; ++i) {
            const float denoised = x[i] - t_curr * v_pred[i];
            x[i] = one_minus_next * denoised + t_next * eps[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            x[i] = x[i] - t_curr * v_pred[i];
        }
    }
}

auto step_euler(float* x, const float* v_pred, size_t count, float t_curr, float t_next) -> void {
    if (x == nullptr || v_pred == nullptr || count == 0) {
        return;
    }
    const float dt = t_next - t_curr;
    for (size_t i = 0; i < count; ++i) {
        x[i] += dt * v_pred[i];
    }
}

auto step_dpmpp(float* x, const float* v_pred, size_t count, float t_curr, float t_next,
                float t_prev, bool has_prev, std::vector<float>& old_denoised) -> void {
    if (x == nullptr || v_pred == nullptr || count == 0) {
        return;
    }

    std::vector<float> denoised(count);
    for (size_t i = 0; i < count; ++i) {
        denoised[i] = x[i] - t_curr * v_pred[i];
    }

    const float alpha_t = 1.0f - t_next;
    const float dt = t_next - t_curr;
    const float safe_curr = std::max(t_curr, 1e-10f);
    const float dpmpp_coeff = dt / (std::max(1.0f - t_next, 1e-10f) * safe_curr);
    const float ratio = t_next / safe_curr;

    const bool is_first = !has_prev || old_denoised.size() != count;
    const bool is_last = (t_next == 0.0f);

    if (is_first || is_last) {
        for (size_t i = 0; i < count; ++i) {
            x[i] = ratio * x[i] - alpha_t * dpmpp_coeff * denoised[i];
        }
    } else {
        const float h = log_snr_clamped(t_next) - log_snr_clamped(t_curr);
        const float h_last = log_snr_clamped(t_curr) - log_snr_clamped(t_prev);
        const float r = (h != 0.0f) ? (h_last / h) : 1.0f;
        const float c1 = 1.0f + 1.0f / (2.0f * r);
        const float c2 = 1.0f / (2.0f * r);
        for (size_t i = 0; i < count; ++i) {
            const float denoised_d = c1 * denoised[i] - c2 * old_denoised[i];
            x[i] = ratio * x[i] - alpha_t * dpmpp_coeff * denoised_d;
        }
    }

    old_denoised = std::move(denoised);
}

}  // namespace netbender::sampler
