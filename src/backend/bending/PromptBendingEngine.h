#pragma once

#include "../sampler/SamplerTypes.h"
#include "BendRandom.h"

#include <array>
#include <vector>

namespace netbender::bending {

/**
 * Stateless DSP & trajectory transformations for Stable Audio 3 prompt embeddings
 * and diffusion schedules (`Phase 3 — Axis A`).
 */
class PromptBendingEngine {
public:
    /** Blend two `PromptEmbedding` tensors `[seq_len, cond_dim]` with `t in [0, 1]`. */
    static auto blend_pair(const netbender::sampler::PromptEmbedding& a,
                           const netbender::sampler::PromptEmbedding& b, float t,
                           netbender::sampler::PromptMorphLaw law)
        -> netbender::sampler::PromptEmbedding;

    /** Bilinear 2D semantic blend across 4 prompt slots `[A, B, C, D]`. */
    static auto blend_embeddings(const std::array<netbender::sampler::PromptEmbedding, 4>& slots,
                                 float x, float y, netbender::sampler::PromptMorphLaw law)
        -> netbender::sampler::PromptEmbedding;

    /**
     * Blend any active subset (1 to 4) of `[A, B, C, D]` slots:
     *   - 1 active: returns that slot
     *   - 2 active: projects `(x, y)` onto the segment connecting the two active corners
     *   - 3 active: inverse-distance-squared weighted blend across the 3 active corners
     *   - 4 active: full 2D bilinear `blend_embeddings`
     */
    static auto blend_embeddings_masked(
        const std::array<netbender::sampler::PromptEmbedding, 4>& slots,
        const std::array<bool, 4>& active, float x, float y, netbender::sampler::PromptMorphLaw law)
        -> netbender::sampler::PromptEmbedding;

    /** Apply in-place embedding drift, quantization, and sinusoidal wavefolding. */
    static auto apply_embedding_dsp(netbender::sampler::PromptEmbedding& emb,
                                    const netbender::sampler::PromptBendingConfig& cfg,
                                    Xoshiro128Plus& rng) -> void;

    /** Warp interior diffusion timesteps `sigmas` (`pow(s, warp)`) preserving endpoints. */
    static auto warp_schedule(std::vector<float>& sigmas, float warp) -> void;
};

}  // namespace netbender::bending
