#pragma once

#include <utility>
#include <vector>

namespace netbender::sampler {

/** Compute normalized peak waveform envelope (`SampleSlot::k_num_waveform_bins` bins in `[0, 1]`).
 */
auto compute_waveform_peaks(const std::vector<std::vector<float>>& audio, int num_samples)
    -> std::vector<float>;

/**
 * Compute a `SampleSlot::k_spec_cols x SampleSlot::k_spec_bins` log-frequency spectrogram
 * normalized peak-to-floor over `66 dB` into `[0, 1]`.
 */
auto compute_spectrogram(const std::vector<std::vector<float>>& audio, int num_samples,
                         int sample_rate) -> std::vector<float>;

/**
 * Scan ~10 ms windows to find the first and last non-silent normalized positions `[start, end]`
 * above `threshold_db` dBFS. Returns `{0.0f, 1.0f}` if silent or empty.
 */
auto detect_nonsilent_region(const std::vector<std::vector<float>>& audio, int num_samples,
                             int sample_rate, float threshold_db = -84.0f)
    -> std::pair<float, float>;

}  // namespace netbender::sampler
