#include "SafetySentry.h"
#include <cmath>
#include <algorithm>

auto SafetySentry::process_channel(float* wet_dest, const float* dry_src, int num_samples, float wet_gain, float dry_gain) -> void {
    if (wet_dest == nullptr || dry_src == nullptr || num_samples <= 0) {
        return;
    }

    if (m_blown_fuse.load()) {
        // Blown fuse: bypass wet model output and pass pure dry audio to protect listeners
        for (int i = 0; i < num_samples; ++i) {
            wet_dest[i] = dry_src[i];
        }
        return;
    }

    for (int i = 0; i < num_samples; ++i) {
        float raw_wet = wet_dest[i];

        // NaN & Inf Sentry Check
        if (std::isnan(raw_wet) || std::isinf(raw_wet)) {
            m_blown_fuse.store(true);
            wet_dest[i] = 0.0f;
            continue;
        }

        // Transparent soft headroom saturation on wet model output:
        // Linear unity gain for |x| <= 0.6, smoothly compressing transient peaks to <= 1.3f
        float sat_wet = (std::abs(raw_wet) < 0.6f) ? raw_wet : (std::tanh(raw_wet * 0.77f) * 1.3f);
        float sample = sat_wet * wet_gain + dry_src[i] * dry_gain;

        // Hard cutoff safety check for runaway signals
        if (std::isnan(sample) || std::isinf(sample) || std::abs(sample) > 2.0f) {
            m_blown_fuse.store(true);
            sample = 0.0f; // Instantly silence runaway sample
        } else {
            // Soft clamp to +/- 1.5 to protect output converters
            sample = std::clamp(sample, -1.5f, 1.5f);
        }
        wet_dest[i] = sample;
    }
}

auto SafetySentry::tick_recovery(const std::function<void()>& on_reset) -> void {
    if (m_blown_fuse.load() && m_auto_reset.load()) {
        int cooldown = m_cooldown_blocks.fetch_add(1);
        if (cooldown > 40) { // ~0.5-1.0s depending on block size
            m_cooldown_blocks.store(0);
            if (on_reset) {
                on_reset();
            }
        }
    }
}

