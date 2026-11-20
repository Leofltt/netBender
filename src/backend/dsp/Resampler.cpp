#include "Resampler.h"

#include <algorithm>
#include <cmath>

auto Resampler::reset() noexcept NETBENDER_NONBLOCKING -> void {
    m_history[0] = 0.0f;
    m_history[1] = 0.0f;
    m_history[2] = 0.0f;
    m_history[3] = 0.0f;
    m_write_idx = 0;
    m_sub_sample_pos = 1.0;
}

auto Resampler::required_input_samples(double speed_ratio,
                                       int num_output_samples) const noexcept NETBENDER_NONBLOCKING
    -> int {
    if (num_output_samples <= 0 || speed_ratio <= 0.0) {
        return 0;
    }
    const double total =
        m_sub_sample_pos + static_cast<double>(num_output_samples - 1) * speed_ratio + 1e-9;
    return (total < 1.0) ? 0 : static_cast<int>(std::floor(total));
}

auto Resampler::interpolate(float offset) const noexcept NETBENDER_NONBLOCKING -> float {
    int idx = m_write_idx;
    const float y0 = m_history[idx];
    idx = (idx + 1) & 3;
    const float y1 = m_history[idx];
    idx = (idx + 1) & 3;
    const float y2 = m_history[idx];
    idx = (idx + 1) & 3;
    const float y3 = m_history[idx];

    if (m_quality == ResamplerQuality::Linear) {
        return y3 * offset + y2 * (1.0f - offset);
    }

    // 4-point Catmull-Rom cubic Hermite matching juce::CatmullRomTraits::valueAtOffset.
    const float half_y0 = 0.5f * y0;
    const float half_y3 = 0.5f * y3;
    return y1 + offset * ((0.5f * y2 - half_y0) +
                          offset * (((y0 + 2.0f * y2) - (half_y3 + 2.5f * y1)) +
                                    offset * ((half_y3 + 1.5f * y1) - (half_y0 + 1.5f * y2))));
}

template <typename InT, typename OutT>
auto Resampler::process_for_output_impl(double speed_ratio, const InT* input, int max_input_samples,
                                        OutT* output,
                                        int num_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    if (num_output_samples <= 0 || speed_ratio <= 0.0) {
        return {0, 0};
    }

    const int safe_max_in = std::max(0, max_input_samples);
    int in_idx = 0;
    double pos = m_sub_sample_pos;

    for (int i = 0; i < num_output_samples; ++i) {
        while (pos >= 1.0) {
            const float sample = (input != nullptr && in_idx < safe_max_in)
                                     ? static_cast<float>(input[in_idx])
                                     : 0.0f;
            ++in_idx;
            push_sample(sample);
            pos -= 1.0;
        }
        if (output != nullptr) {
            output[i] = static_cast<OutT>(interpolate(static_cast<float>(pos)));
        }
        pos += speed_ratio;
    }

    m_sub_sample_pos = pos;
    return {std::min(in_idx, safe_max_in), num_output_samples};
}

template <typename InT>
auto Resampler::process_for_input_impl(double speed_ratio, const InT* input, int num_input_samples,
                                       float* output,
                                       int max_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    if (num_input_samples < 0 || max_output_samples <= 0 || speed_ratio <= 0.0) {
        return {0, 0};
    }

    int in_idx = 0;
    int out_idx = 0;
    double pos = m_sub_sample_pos;

    while (out_idx < max_output_samples) {
        const int need = (pos >= 1.0) ? static_cast<int>(std::floor(pos)) : 0;
        if (in_idx + need > num_input_samples) {
            break;
        }
        while (pos >= 1.0) {
            const float sample = (input != nullptr) ? static_cast<float>(input[in_idx]) : 0.0f;
            ++in_idx;
            push_sample(sample);
            pos -= 1.0;
        }
        if (output != nullptr) {
            output[out_idx] = interpolate(static_cast<float>(pos));
        }
        ++out_idx;
        pos += speed_ratio;
    }

    m_sub_sample_pos = pos;
    return {in_idx, out_idx};
}

auto Resampler::process_for_output(double speed_ratio, const float* input, int max_input_samples,
                                   float* output,
                                   int num_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    return process_for_output_impl(speed_ratio, input, max_input_samples, output,
                                   num_output_samples);
}

auto Resampler::process_for_output(double speed_ratio, const float* input, int max_input_samples,
                                   double* output,
                                   int num_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    return process_for_output_impl(speed_ratio, input, max_input_samples, output,
                                   num_output_samples);
}

auto Resampler::process_for_output(double speed_ratio, const double* input, int max_input_samples,
                                   float* output,
                                   int num_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    return process_for_output_impl(speed_ratio, input, max_input_samples, output,
                                   num_output_samples);
}

auto Resampler::process_for_input(double speed_ratio, const float* input, int num_input_samples,
                                  float* output,
                                  int max_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    return process_for_input_impl(speed_ratio, input, num_input_samples, output,
                                  max_output_samples);
}

auto Resampler::process_for_input(double speed_ratio, const double* input, int num_input_samples,
                                  float* output,
                                  int max_output_samples) noexcept NETBENDER_NONBLOCKING
    -> ResampleResult {
    return process_for_input_impl(speed_ratio, input, num_input_samples, output,
                                  max_output_samples);
}
