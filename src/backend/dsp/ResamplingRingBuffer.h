#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include "AudioRingBuffer.h"
#include "Resampler.h"
#include "rtsan_utils.h"

enum class ResampleMode {
    Auto = 0,          // Active when |src_sr - dst_sr| > 1e-3 Hz
    ForceEnabled = 1,  // Always resample (useful for testing 1.0 ratio or forced path)
    Disabled = 2,      // Direct pass-through bypass without touching resampler
};

/**
 * Combines a wait-free SPSC AudioRingBuffer, stateful Resampler, and pre-allocated scratch
 * buffer so either producer or consumer can push/pull seamlessly across sample-rate boundaries
 * with zero heap allocations on the real-time thread.
 */
class ResamplingRingBuffer {
public:
    ResamplingRingBuffer() = default;
    ResamplingRingBuffer(ResamplingRingBuffer&& other) noexcept;
    auto operator=(ResamplingRingBuffer&& other) noexcept -> ResamplingRingBuffer&;

    ResamplingRingBuffer(const ResamplingRingBuffer&) = delete;
    auto operator=(const ResamplingRingBuffer&) -> ResamplingRingBuffer& = delete;
    ~ResamplingRingBuffer() = default;

    auto init(int capacity_src_samples, double src_sr, double dst_sr,
              ResampleMode mode = ResampleMode::Auto,
              ResamplerQuality quality = ResamplerQuality::CatmullRom, int scratch_samples = 0)
        -> void;

    auto set_sample_rates(double src_sr, double dst_sr) -> void;
    auto set_resample_mode(ResampleMode mode) -> void;
    auto clear() -> void;

    // Producer (wait-free, source sample rate)
    auto push(const float* src, int num_src_samples) NETBENDER_NONBLOCKING -> int;
    auto push(const double* src, int num_src_samples) NETBENDER_NONBLOCKING -> int;
    auto push_zeros(int num_src_samples) NETBENDER_NONBLOCKING -> int;

    // Consumer (wait-free, destination sample rate; chunked over pre-allocated scratch)
    auto pull(float* dst, int num_dst_samples) NETBENDER_NONBLOCKING -> int;
    auto pull(double* dst, int num_dst_samples) NETBENDER_NONBLOCKING -> int;

    // Introspection
    [[nodiscard]] auto is_resampling_active() const noexcept NETBENDER_NONBLOCKING -> bool {
        return m_resampling_active;
    }
    [[nodiscard]] auto get_speed_ratio() const noexcept NETBENDER_NONBLOCKING -> double {
        return m_speed_ratio;
    }
    [[nodiscard]] auto get_src_sample_rate() const noexcept NETBENDER_NONBLOCKING -> double {
        return m_src_sr;
    }
    [[nodiscard]] auto get_dst_sample_rate() const noexcept NETBENDER_NONBLOCKING -> double {
        return m_dst_sr;
    }
    [[nodiscard]] auto get_resample_mode() const noexcept NETBENDER_NONBLOCKING -> ResampleMode {
        return m_mode;
    }

    [[nodiscard]] auto get_available_src() const NETBENDER_NONBLOCKING -> int {
        return m_ring.get_available();
    }
    [[nodiscard]] auto get_free_space_src() const NETBENDER_NONBLOCKING -> int {
        return m_ring.get_free_space();
    }
    [[nodiscard]] auto get_capacity_src() const NETBENDER_NONBLOCKING -> int {
        return m_ring.get_capacity();
    }

    [[nodiscard]] auto required_src_samples_for_pull(int num_dst_samples) const noexcept
        NETBENDER_NONBLOCKING -> int;

    [[nodiscard]] auto can_pull(int num_dst_samples) const NETBENDER_NONBLOCKING -> bool;

    [[nodiscard]] auto get_overrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
        return m_ring.get_overrun_count();
    }
    [[nodiscard]] auto get_underrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
        return m_ring.get_underrun_count() + m_resample_underruns.load(std::memory_order_relaxed);
    }
    auto reset_counters() NETBENDER_NONBLOCKING -> void {
        m_ring.reset_counters();
        m_resample_underruns.store(0, std::memory_order_relaxed);
    }

private:
    friend class ResamplingRingBufferBank;

    auto update_rates_and_mode(double src_sr, double dst_sr, ResampleMode mode) -> void;

    [[nodiscard]] auto max_chunk_out_samples() const noexcept NETBENDER_NONBLOCKING -> int;

    template <typename OutT>
    auto pull_impl(OutT* dst, int num_dst_samples) NETBENDER_NONBLOCKING -> int;

    template <typename OutT>
    auto pull_chunk_locked(OutT* dst, int chunk_out, int max_src_available) NETBENDER_NONBLOCKING
        -> int;

    AudioRingBuffer m_ring;
    Resampler m_resampler;
    std::vector<float> m_scratch;

    double m_src_sr{44100.0};
    double m_dst_sr{44100.0};
    double m_speed_ratio{1.0};
    ResampleMode m_mode{ResampleMode::Auto};
    bool m_resampling_active{false};

    std::atomic<uint64_t> m_resample_underruns{0};
};

/**
 * Multi-channel bank of ResamplingRingBuffer instances with strict lockstep consumption
 * across all channels so stereo and multichannel streams remain phase-locked.
 */
class ResamplingRingBufferBank {
public:
    ResamplingRingBufferBank() = default;

    auto init(int num_channels, int capacity_src_samples, double src_sr, double dst_sr,
              ResampleMode mode = ResampleMode::Auto,
              ResamplerQuality quality = ResamplerQuality::CatmullRom, int scratch_samples = 0)
        -> void;

    auto set_sample_rates(double src_sr, double dst_sr) -> void;
    auto set_resample_mode(ResampleMode mode) -> void;
    auto clear() -> void;

    auto push_channel(int channel, const float* src, int num_src_samples) NETBENDER_NONBLOCKING
        -> int;
    auto push_channel(int channel, const double* src, int num_src_samples) NETBENDER_NONBLOCKING
        -> int;
    auto push_zeros_all(int num_src_samples) NETBENDER_NONBLOCKING -> void;

    auto pull(float* const* dst_channels, int num_dst_channels,
              int num_dst_samples) NETBENDER_NONBLOCKING -> int;
    auto pull(double* const* dst_channels, int num_dst_channels,
              int num_dst_samples) NETBENDER_NONBLOCKING -> int;

    [[nodiscard]] auto get_num_channels() const noexcept NETBENDER_NONBLOCKING -> int {
        return static_cast<int>(m_channels.size());
    }
    [[nodiscard]] auto is_resampling_active() const noexcept NETBENDER_NONBLOCKING -> bool {
        return !m_channels.empty() && m_channels[0].is_resampling_active();
    }
    [[nodiscard]] auto get_available_src() const NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto get_free_space_src() const NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto required_src_samples_for_pull(int num_dst_samples) const
        NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto can_pull(int num_dst_samples) const NETBENDER_NONBLOCKING -> bool;

    [[nodiscard]] auto get_overrun_count() const NETBENDER_NONBLOCKING -> uint64_t;
    [[nodiscard]] auto get_underrun_count() const NETBENDER_NONBLOCKING -> uint64_t;

private:
    template <typename OutT>
    auto pull_bank_impl(OutT* const* dst_channels, int num_dst_channels,
                        int num_dst_samples) NETBENDER_NONBLOCKING -> int;

    std::vector<ResamplingRingBuffer> m_channels;
};
