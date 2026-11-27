#include "PromptBendingEngine.h"

#include <algorithm>
#include <cmath>

namespace netbender::bending {

auto PromptBendingEngine::blend_pair(const netbender::sampler::PromptEmbedding& a,
                                     const netbender::sampler::PromptEmbedding& b, float t,
                                     netbender::sampler::PromptMorphLaw law)
    -> netbender::sampler::PromptEmbedding {
    const float w = std::clamp(t, 0.0f, 1.0f);
    if (a.data.empty()) {
        return b;
    }
    if (b.data.empty() || w <= 1e-6f) {
        return a;
    }
    if (w >= 1.0f - 1e-6f) {
        return b;
    }

    const int seq = std::min(a.seq_len, b.seq_len);
    const int dim = std::min(a.cond_dim, b.cond_dim);
    netbender::sampler::PromptEmbedding out;
    out.seq_len = seq;
    out.cond_dim = dim;
    out.data.resize(static_cast<size_t>(seq) * static_cast<size_t>(dim), 0.0f);

    if (law == netbender::sampler::PromptMorphLaw::TokenSplice) {
        const int stride = 4;
        for (int s = 0; s < seq; ++s) {
            const float phase = static_cast<float>(s % stride) / static_cast<float>(stride);
            const auto& src = (phase < w) ? b : a;
            for (int d = 0; d < dim; ++d) {
                out.data[static_cast<size_t>(s) * dim + d] =
                    src.data[static_cast<size_t>(s) * src.cond_dim + d];
            }
        }
        return out;
    }

    if (law == netbender::sampler::PromptMorphLaw::FreqSplit) {
        const int split_dim = static_cast<int>(w * static_cast<float>(dim));
        for (int s = 0; s < seq; ++s) {
            for (int d = 0; d < dim; ++d) {
                const auto& src = (d < split_dim) ? b : a;
                out.data[static_cast<size_t>(s) * dim + d] =
                    src.data[static_cast<size_t>(s) * src.cond_dim + d];
            }
        }
        return out;
    }

    if (law == netbender::sampler::PromptMorphLaw::Linear) {
        const float one_minus_w = 1.0f - w;
        for (int s = 0; s < seq; ++s) {
            for (int d = 0; d < dim; ++d) {
                const float va = a.data[static_cast<size_t>(s) * a.cond_dim + d];
                const float vb = b.data[static_cast<size_t>(s) * b.cond_dim + d];
                out.data[static_cast<size_t>(s) * dim + d] = one_minus_w * va + w * vb;
            }
        }
        return out;
    }

    // Norm-preserving SLERP per token vector
    for (int s = 0; s < seq; ++s) {
        double na2 = 0.0;
        double nb2 = 0.0;
        double dot = 0.0;
        const size_t base_a = static_cast<size_t>(s) * a.cond_dim;
        const size_t base_b = static_cast<size_t>(s) * b.cond_dim;
        const size_t base_o = static_cast<size_t>(s) * dim;

        for (int d = 0; d < dim; ++d) {
            const double va = static_cast<double>(a.data[base_a + d]);
            const double vb = static_cast<double>(b.data[base_b + d]);
            na2 += va * va;
            nb2 += vb * vb;
            dot += va * vb;
        }

        const double na = std::sqrt(na2);
        const double nb = std::sqrt(nb2);
        if (na < 1e-8 || nb < 1e-8) {
            const float one_minus_w = 1.0f - w;
            for (int d = 0; d < dim; ++d) {
                out.data[base_o + d] = one_minus_w * a.data[base_a + d] + w * b.data[base_b + d];
            }
            continue;
        }

        const double cos_theta = std::clamp(dot / (na * nb), -0.99999, 0.99999);
        const double theta = std::acos(cos_theta);
        const double sin_theta = std::sin(theta);
        const double target_norm = (1.0 - w) * na + w * nb;

        double sa = 1.0 - w;
        double sb = w;
        if (sin_theta > 1e-5) {
            sa = std::sin((1.0 - w) * theta) / sin_theta;
            sb = std::sin(w * theta) / sin_theta;
        }

        for (int d = 0; d < dim; ++d) {
            const double ua = static_cast<double>(a.data[base_a + d]) / na;
            const double ub = static_cast<double>(b.data[base_b + d]) / nb;
            out.data[base_o + d] = static_cast<float>(target_norm * (sa * ua + sb * ub));
        }
    }
    return out;
}

auto PromptBendingEngine::blend_embeddings(
    const std::array<netbender::sampler::PromptEmbedding, 4>& slots, float x, float y,
    netbender::sampler::PromptMorphLaw law) -> netbender::sampler::PromptEmbedding {
    const auto top = blend_pair(slots[0], slots[1], x, law);
    const auto bottom = blend_pair(slots[2], slots[3], x, law);
    return blend_pair(top, bottom, y, law);
}

auto PromptBendingEngine::blend_embeddings_masked(
    const std::array<netbender::sampler::PromptEmbedding, 4>& slots,
    const std::array<bool, 4>& active, float x, float y, netbender::sampler::PromptMorphLaw law)
    -> netbender::sampler::PromptEmbedding {
    constexpr std::array<float, 4> k_cx{0.0f, 1.0f, 0.0f, 1.0f};
    constexpr std::array<float, 4> k_cy{0.0f, 0.0f, 1.0f, 1.0f};

    std::array<int, 4> indices{};
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        if (active[static_cast<size_t>(i)] && !slots[static_cast<size_t>(i)].data.empty()) {
            indices[static_cast<size_t>(count++)] = i;
        }
    }

    if (count == 0) {
        return slots[0];
    }
    if (count == 1) {
        return slots[static_cast<size_t>(indices[0])];
    }
    if (count == 4) {
        return blend_embeddings(slots, x, y, law);
    }

    const float px = std::clamp(x, 0.0f, 1.0f);
    const float py = std::clamp(y, 0.0f, 1.0f);

    if (count == 2) {
        const int i0 = indices[0];
        const int i1 = indices[1];
        const float dx = k_cx[static_cast<size_t>(i1)] - k_cx[static_cast<size_t>(i0)];
        const float dy = k_cy[static_cast<size_t>(i1)] - k_cy[static_cast<size_t>(i0)];
        const float len2 = std::max(1e-6f, dx * dx + dy * dy);
        const float t = std::clamp(((px - k_cx[static_cast<size_t>(i0)]) * dx +
                                    (py - k_cy[static_cast<size_t>(i0)]) * dy) /
                                       len2,
                                   0.0f, 1.0f);
        return blend_pair(slots[static_cast<size_t>(i0)], slots[static_cast<size_t>(i1)], t, law);
    }

    // 3 active corners: inverse-distance-squared weighting with corner snap
    std::array<float, 3> w{};
    float w_sum = 0.0f;
    for (int k = 0; k < 3; ++k) {
        const int idx = indices[static_cast<size_t>(k)];
        const float dx = px - k_cx[static_cast<size_t>(idx)];
        const float dy = py - k_cy[static_cast<size_t>(idx)];
        const float d2 = dx * dx + dy * dy;
        if (d2 < 1e-5f) {
            return slots[static_cast<size_t>(idx)];
        }
        w[static_cast<size_t>(k)] = 1.0f / d2;
        w_sum += w[static_cast<size_t>(k)];
    }

    const float w01 = w[0] + w[1];
    auto out = blend_pair(slots[static_cast<size_t>(indices[0])],
                          slots[static_cast<size_t>(indices[1])], w[1] / w01, law);
    out = blend_pair(out, slots[static_cast<size_t>(indices[2])], w[2] / w_sum, law);
    return out;
}

auto PromptBendingEngine::apply_embedding_dsp(netbender::sampler::PromptEmbedding& emb,
                                              const netbender::sampler::PromptBendingConfig& cfg,
                                              Xoshiro128Plus& rng) -> void {
    if (emb.data.empty()) {
        return;
    }

    const bool do_drift = cfg.emb_drift > 1e-5f;
    const bool do_quant = cfg.emb_quantize_levels >= 2;
    const bool do_fold = cfg.emb_wavefold > 1e-5f;
    const bool do_drop = cfg.token_dropout > 1e-5f && emb.seq_len > 1 && emb.cond_dim > 0;
    if (!do_drift && !do_quant && !do_fold && !do_drop) {
        return;
    }

    // Estimate RMS scale of embedding so drift/quant/fold are calibrated to T5Gemma's magnitude
    double sq_sum = 0.0;
    for (float v : emb.data) {
        sq_sum += static_cast<double>(v) * static_cast<double>(v);
    }
    const float rms = std::max(
        1e-4f, static_cast<float>(std::sqrt(sq_sum / static_cast<double>(emb.data.size()))));

    constexpr float k_two_pi = 6.28318530717958647692f;
    const float drift_amp = cfg.emb_drift * rms;
    const float fold_gain = 1.0f + cfg.emb_wavefold * 2.5f;
    const float q_step =
        do_quant ? (4.0f * rms / static_cast<float>(cfg.emb_quantize_levels - 1)) : 1.0f;

    size_t i = 0;
    const size_t n = emb.data.size();
    while (i < n) {
        float g0 = 0.0f;
        float g1 = 0.0f;
        if (do_drift) {
            const float u1 = std::max(1e-7f, rng.next_float());
            const float u2 = rng.next_float();
            const float r = std::sqrt(-2.0f * std::log(u1));
            const float theta = k_two_pi * u2;
            g0 = r * std::cos(theta) * drift_amp;
            g1 = r * std::sin(theta) * drift_amp;
        }

        const float g_vals[2]{g0, g1};
        for (int k = 0; k < 2 && i < n; ++k, ++i) {
            float v = emb.data[i];
            if (do_drift) {
                v += g_vals[k];
            }
            if (do_quant) {
                v = std::round(v / q_step) * q_step;
            }
            if (do_fold) {
                v = rms * std::sin((v / rms) * fold_gain);
            }
            emb.data[i] = v;
        }
    }

    if (do_drop) {
        const float drop_p = std::clamp(cfg.token_dropout, 0.0f, 0.95f);
        const size_t dim = static_cast<size_t>(emb.cond_dim);
        for (int s = 1; s < emb.seq_len; ++s) {
            if (rng.next_float() < drop_p) {
                const size_t base = static_cast<size_t>(s) * dim;
                const size_t end = std::min(base + dim, emb.data.size());
                std::fill(emb.data.begin() + static_cast<std::ptrdiff_t>(base),
                          emb.data.begin() + static_cast<std::ptrdiff_t>(end), 0.0f);
            }
        }
    }
}

auto PromptBendingEngine::warp_schedule(std::vector<float>& sigmas, float warp) -> void {
    if (sigmas.size() <= 2) {
        return;
    }
    const float w = std::clamp(warp, 0.25f, 3.0f);
    if (std::abs(w - 1.0f) <= 1e-4f) {
        return;
    }
    for (size_t i = 1; i + 1 < sigmas.size(); ++i) {
        const float s = std::clamp(sigmas[i], 0.0f, 1.0f);
        sigmas[i] = std::pow(s, w);
    }
}

}  // namespace netbender::bending
