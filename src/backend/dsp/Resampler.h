#pragma once

#include "rtsan_utils.h"

enum class ResamplerQuality {
    CatmullRom = 0,
    Linear = 1,
};

struct ResampleResult {
    int input_consumed{0};
    int output_produced{0};
};

/**
 * Stateful, allocation-free streaming fractional resampler supporting Catmull-Rom (4-point cubic
 * Hermite matching JUCE's CatmullRomInterpolator) and linear interpolation.
 */
class Resampler {
public:
    explicit Resampler(ResamplerQuality quality = ResamplerQuality::CatmullRom) noexcept
        : m_quality(quality) {}

    auto reset() noexcept NETBENDER_NONBLOCKING -> void;

    auto set_quality(ResamplerQuality quality) noexcept NETBENDER_NONBLOCKING -> void {
        m_quality = quality;
    }

    [[nodiscard]] auto get_quality() const noexcept NETBENDER_NONBLOCKING -> ResamplerQuality {
        return m_quality;
    }

    [[nodiscard]] auto get_base_latency() const noexcept NETBENDER_NONBLOCKING -> float {
        return (m_quality == ResamplerQuality::CatmullRom) ? 2.0f : 1.0f;
    }

    [[nodiscard]] auto get_sub_sample_pos() const noexcept NETBENDER_NONBLOCKING -> double {
        return m_sub_sample_pos;
    }

    // Safe upper bound on input samples required to produce num_output_samples at speed_ratio.
    [[nodiscard]] auto required_input_samples(
        double speed_ratio, int num_output_samples) const noexcept NETBENDER_NONBLOCKING -> int;

    // Pull-side resampling: produces exact num_output_samples, never reading past
    // max_input_samples (feeds 0.0f if input runs short).
    auto process_for_output(double speed_ratio, const float* input, int max_input_samples,
                            float* output, int num_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

    auto process_for_output(double speed_ratio, const float* input, int max_input_samples,
                            double* output, int num_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

    auto process_for_output(double speed_ratio, const double* input, int max_input_samples,
                            float* output, int num_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

    // Bounded push-side resampling: consumes at most num_input_samples and produces up to
    // max_output_samples without ever over-reading input.
    auto process_for_input(double speed_ratio, const float* input, int num_input_samples,
                           float* output, int max_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

    auto process_for_input(double speed_ratio, const double* input, int num_input_samples,
                           float* output, int max_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

private:
    auto push_sample(float sample) noexcept NETBENDER_NONBLOCKING -> void {
        m_history[m_write_idx] = sample;
        m_write_idx = (m_write_idx + 1) & 3;
    }

    [[nodiscard]] auto interpolate(float offset) const noexcept NETBENDER_NONBLOCKING -> float;

    template <typename InT, typename OutT>
    auto process_for_output_impl(double speed_ratio, const InT* input, int max_input_samples,
                                 OutT* output,
                                 int num_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

    template <typename InT>
    auto process_for_input_impl(double speed_ratio, const InT* input, int num_input_samples,
                                float* output,
                                int max_output_samples) noexcept NETBENDER_NONBLOCKING
        -> ResampleResult;

    float m_history[4]{0.0f, 0.0f, 0.0f, 0.0f};
    int m_write_idx{0};
    double m_sub_sample_pos{1.0};
    ResamplerQuality m_quality{ResamplerQuality::CatmullRom};
};
