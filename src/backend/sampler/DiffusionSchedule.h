#pragma once

#include "SamplerTypes.h"
#include "../bending/BendRandom.h"

#include <cstdint>
#include <vector>

namespace netbender::sampler {

/**
 * Exact C++20 ports of Stable Audio 3's timestep distribution shifts,
 * schedule builder, CFG/APG projection, and diffusion integrators.
 */

auto apply_distribution_shift(const std::vector<float>& t, int seq_len, const DistShiftConfig& cfg)
    -> std::vector<float>;

auto build_schedule(int steps, const DistShiftConfig& cfg, int effective_seq_len,
                    float sigma_max = 1.0f, bool include_endpoint = true) -> std::vector<float>;

[[nodiscard]] auto compute_effective_latent_frames(float duration_sec, int sample_rate,
                                                   int downsampling_ratio) -> int;

[[nodiscard]] auto compute_aligned_latent_frames(float duration_sec, float duration_padding_sec,
                                                 int sample_rate, int downsampling_ratio,
                                                 int latent_align, int compiled_latent_frames = 0)
    -> int;

auto build_padding_mask(int latent_frames, int effective_frames, float duration_padding_sec,
                        int sample_rate, int downsampling_ratio) -> std::vector<uint8_t>;

auto fill_standard_normal(float* dst, size_t count, netbender::bending::Xoshiro128Plus& rng)
    -> void;

auto apg_project(const float* v0, const float* v1, int channels, int frames,
                 const uint8_t* padding_mask, float* parallel_out, float* orthogonal_out) -> void;

auto blend_cfg_apg(const float* x, const float* v_cond, const float* v_uncond, int channels,
                   int frames, float sigma, float cfg_scale, float apg_scale,
                   float cfg_norm_threshold, const uint8_t* padding_mask, float* v_out) -> void;

auto step_pingpong(float* x, const float* v_pred, size_t count, float t_curr, float t_next,
                   netbender::bending::Xoshiro128Plus& rng) -> void;

auto step_euler(float* x, const float* v_pred, size_t count, float t_curr, float t_next) -> void;

auto step_dpmpp(float* x, const float* v_pred, size_t count, float t_curr, float t_next,
                float t_prev, bool has_prev, std::vector<float>& old_denoised) -> void;

}  // namespace netbender::sampler
