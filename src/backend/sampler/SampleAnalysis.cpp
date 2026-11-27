#include "SampleAnalysis.h"
#include "SamplerTypes.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace netbender::sampler {

auto compute_waveform_peaks(const std::vector<std::vector<float>>& audio, int num_samples)
    -> std::vector<float> {
    std::vector<float> peaks(SampleSlot::k_num_waveform_bins, 0.0f);
    if (audio.empty() || num_samples <= 0) {
        return peaks;
    }
    const int bins = SampleSlot::k_num_waveform_bins;
    for (int b = 0; b < bins; ++b) {
        const int i0 = static_cast<int>((static_cast<int64_t>(b) * num_samples) / bins);
        const int i1 =
            std::max(i0 + 1, static_cast<int>((static_cast<int64_t>(b + 1) * num_samples) / bins));
        float max_abs = 0.0f;
        for (const auto& ch : audio) {
            const int limit = std::min(i1, static_cast<int>(ch.size()));
            for (int i = i0; i < limit; ++i) {
                max_abs = std::max(max_abs, std::abs(ch[static_cast<size_t>(i)]));
            }
        }
        peaks[static_cast<size_t>(b)] = std::min(1.0f, max_abs);
    }
    return peaks;
}

auto compute_spectrogram(const std::vector<std::vector<float>>& audio, int num_samples,
                         int sample_rate) -> std::vector<float> {
    const int cols = SampleSlot::k_spec_cols;
    const int bins = SampleSlot::k_spec_bins;
    std::vector<float> spec(static_cast<size_t>(cols * bins), 0.0f);
    if (audio.empty() || num_samples <= 0) {
        return spec;
    }

    constexpr int k_fft_size = 256;
    constexpr int k_half_fft = k_fft_size / 2;
    constexpr float k_two_pi = 6.28318530717958647692f;

    std::array<float, k_fft_size> hann{};
    for (int i = 0; i < k_fft_size; ++i) {
        hann[static_cast<size_t>(i)] = 0.5f * (1.0f - std::cos(k_two_pi * static_cast<float>(i) /
                                                               static_cast<float>(k_fft_size - 1)));
    }

    const int n_ch = static_cast<int>(audio.size());
    const float inv_ch = 1.0f / static_cast<float>(std::max(1, n_ch));
    const float sr = static_cast<float>(std::max(8000, sample_rate));
    const float nyquist = sr * 0.5f;
    const float min_freq = 40.0f;
    const float freq_ratio = nyquist / min_freq;

    std::array<float, k_fft_size> re{};
    std::array<float, k_fft_size> im{};
    std::array<float, k_half_fft> mag{};
    float max_peak = 1e-5f;

    for (int c = 0; c < cols; ++c) {
        const int center =
            static_cast<int>(((static_cast<double>(c) + 0.5) * static_cast<double>(num_samples)) /
                             static_cast<double>(cols));
        const int s0 = center - k_half_fft;

        for (int i = 0; i < k_fft_size; ++i) {
            const int idx = s0 + i;
            float mono = 0.0f;
            if (idx >= 0 && idx < num_samples) {
                for (int ch = 0; ch < n_ch; ++ch) {
                    if (idx < static_cast<int>(audio[static_cast<size_t>(ch)].size())) {
                        mono += audio[static_cast<size_t>(ch)][static_cast<size_t>(idx)];
                    }
                }
                mono *= inv_ch;
            }
            re[static_cast<size_t>(i)] = mono * hann[static_cast<size_t>(i)];
            im[static_cast<size_t>(i)] = 0.0f;
        }

        for (int i = 1, j = 0; i < k_fft_size; ++i) {
            int bit = k_fft_size >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                std::swap(re[static_cast<size_t>(i)], re[static_cast<size_t>(j)]);
                std::swap(im[static_cast<size_t>(i)], im[static_cast<size_t>(j)]);
            }
        }

        for (int len = 2; len <= k_fft_size; len <<= 1) {
            const float ang = -k_two_pi / static_cast<float>(len);
            const float wlen_re = std::cos(ang);
            const float wlen_im = std::sin(ang);
            for (int i = 0; i < k_fft_size; i += len) {
                float w_re = 1.0f;
                float w_im = 0.0f;
                const int half = len >> 1;
                for (int j = 0; j < half; ++j) {
                    const size_t u = static_cast<size_t>(i + j);
                    const size_t v = static_cast<size_t>(i + j + half);
                    const float t_re = re[v] * w_re - im[v] * w_im;
                    const float t_im = re[v] * w_im + im[v] * w_re;
                    re[v] = re[u] - t_re;
                    im[v] = im[u] - t_im;
                    re[u] += t_re;
                    im[u] += t_im;
                    const float next_w_re = w_re * wlen_re - w_im * wlen_im;
                    w_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = next_w_re;
                }
            }
        }

        for (int k = 1; k < k_half_fft; ++k) {
            const float r = re[static_cast<size_t>(k)];
            const float m = im[static_cast<size_t>(k)];
            mag[static_cast<size_t>(k)] =
                std::sqrt(r * r + m * m) / (static_cast<float>(k_fft_size) * 0.5f);
        }

        for (int b = 0; b < bins; ++b) {
            const float f0 =
                min_freq * std::pow(freq_ratio, static_cast<float>(b) / static_cast<float>(bins));
            const float f1 = min_freq * std::pow(freq_ratio, static_cast<float>(b + 1) /
                                                                 static_cast<float>(bins));
            const int k0 =
                std::clamp(static_cast<int>(std::floor(f0 * k_fft_size / sr)), 1, k_half_fft - 1);
            const int k1 =
                std::clamp(static_cast<int>(std::ceil(f1 * k_fft_size / sr)), k0, k_half_fft - 1);
            float peak = 0.0f;
            for (int k = k0; k <= k1; ++k) {
                peak = std::max(peak, mag[static_cast<size_t>(k)]);
            }
            max_peak = std::max(max_peak, peak);
            spec[static_cast<size_t>(c * bins + b)] = peak;
        }
    }

    for (float& v : spec) {
        const float rel = std::max(v / max_peak, 1e-4f);
        const float db = 20.0f * std::log10(rel);
        v = std::clamp((db + 66.0f) / 66.0f, 0.0f, 1.0f);
    }

    return spec;
}

auto detect_nonsilent_region(const std::vector<std::vector<float>>& audio, int num_samples,
                             int sample_rate, float threshold_db) -> std::pair<float, float> {
    if (audio.empty() || num_samples <= 0) {
        return {0.0f, 1.0f};
    }
    const float thr = std::pow(10.0f, threshold_db / 20.0f);
    const int win = std::max(1, sample_rate / 100);  // ~10 ms windows
    const int num_windows = (num_samples + win - 1) / win;

    auto window_peak = [&](int w_idx) -> float {
        const int i0 = w_idx * win;
        const int i1 = std::min(num_samples, i0 + win);
        float peak = 0.0f;
        for (const auto& ch : audio) {
            const int limit = std::min(i1, static_cast<int>(ch.size()));
            for (int i = i0; i < limit; ++i) {
                peak = std::max(peak, std::abs(ch[static_cast<size_t>(i)]));
            }
        }
        return peak;
    };

    int first_w = -1;
    for (int w = 0; w < num_windows; ++w) {
        if (window_peak(w) >= thr) {
            first_w = w;
            break;
        }
    }
    if (first_w < 0) {
        return {0.0f, 1.0f};
    }

    int last_w = first_w;
    for (int w = num_windows - 1; w >= first_w; --w) {
        if (window_peak(w) >= thr) {
            last_w = w;
            break;
        }
    }

    const float start_norm = std::clamp(
        static_cast<float>(first_w * win) / static_cast<float>(num_samples), 0.0f, 0.98f);
    const float end_sample = static_cast<float>(std::min(num_samples, (last_w + 1) * win));
    const float end_norm =
        std::clamp(end_sample / static_cast<float>(num_samples), start_norm + 0.02f, 1.0f);
    return {start_norm, end_norm};
}

}  // namespace netbender::sampler
