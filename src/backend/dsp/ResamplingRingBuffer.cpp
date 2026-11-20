#include "ResamplingRingBuffer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

ResamplingRingBuffer::ResamplingRingBuffer(ResamplingRingBuffer&& other) noexcept
    : m_ring(std::move(other.m_ring)),
      m_resampler(other.m_resampler),
      m_scratch(std::move(other.m_scratch)),
      m_src_sr(other.m_src_sr),
      m_dst_sr(other.m_dst_sr),
      m_speed_ratio(other.m_speed_ratio),
      m_mode(other.m_mode),
      m_resampling_active(other.m_resampling_active),
      m_resample_underruns(other.m_resample_underruns.load(std::memory_order_relaxed)) {
    other.m_resampler.reset();
    other.m_src_sr = 44100.0;
    other.m_dst_sr = 44100.0;
    other.m_speed_ratio = 1.0;
    other.m_resampling_active = false;
    other.m_resample_underruns.store(0, std::memory_order_relaxed);
}

auto ResamplingRingBuffer::operator=(ResamplingRingBuffer&& other) noexcept
    -> ResamplingRingBuffer& {
    if (this != &other) {
        m_ring = std::move(other.m_ring);
        m_resampler = other.m_resampler;
        m_scratch = std::move(other.m_scratch);
        m_src_sr = other.m_src_sr;
        m_dst_sr = other.m_dst_sr;
        m_speed_ratio = other.m_speed_ratio;
        m_mode = other.m_mode;
        m_resampling_active = other.m_resampling_active;
        m_resample_underruns.store(other.m_resample_underruns.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);

        other.m_resampler.reset();
        other.m_src_sr = 44100.0;
        other.m_dst_sr = 44100.0;
        other.m_speed_ratio = 1.0;
        other.m_resampling_active = false;
        other.m_resample_underruns.store(0, std::memory_order_relaxed);
    }
    return *this;
}

auto ResamplingRingBuffer::update_rates_and_mode(double src_sr, double dst_sr, ResampleMode mode)
    -> void {
    const bool was_active = m_resampling_active;
    m_src_sr = (src_sr > 0.0) ? src_sr : 44100.0;
    m_dst_sr = (dst_sr > 0.0) ? dst_sr : 44100.0;
    m_mode = mode;

    const bool auto_active = std::abs(m_src_sr - m_dst_sr) > 1e-3;
    switch (m_mode) {
        case ResampleMode::Auto:
            m_resampling_active = auto_active;
            break;
        case ResampleMode::ForceEnabled:
            m_resampling_active = true;
            break;
        case ResampleMode::Disabled:
            m_resampling_active = false;
            break;
    }

    m_speed_ratio = m_resampling_active ? (m_src_sr / m_dst_sr) : 1.0;
    if (was_active != m_resampling_active) {
        m_resampler.reset();
    }
}

auto ResamplingRingBuffer::init(int capacity_src_samples, double src_sr, double dst_sr,
                                ResampleMode mode, ResamplerQuality quality, int scratch_samples)
    -> void {
    m_ring.init_with_capacity(std::max(1, capacity_src_samples));
    m_resampler.set_quality(quality);
    m_resampler.reset();
    m_resampling_active = false;
    update_rates_and_mode(src_sr, dst_sr, mode);

    const int default_scratch = std::clamp(m_ring.get_capacity(), 1024, 8192);
    const int min_ratio_scratch = static_cast<int>(std::ceil(std::max(1.0, m_speed_ratio))) + 16;
    const int target_scratch =
        std::max(scratch_samples > 0 ? scratch_samples : default_scratch, min_ratio_scratch);
    m_scratch.assign(static_cast<size_t>(target_scratch), 0.0f);
    m_resample_underruns.store(0, std::memory_order_relaxed);
}

auto ResamplingRingBuffer::set_sample_rates(double src_sr, double dst_sr) -> void {
    update_rates_and_mode(src_sr, dst_sr, m_mode);
    const size_t min_scratch =
        static_cast<size_t>(static_cast<int>(std::ceil(std::max(1.0, m_speed_ratio))) + 16);
    if (m_scratch.size() < min_scratch) {
        m_scratch.assign(min_scratch, 0.0f);
    }
}

auto ResamplingRingBuffer::set_resample_mode(ResampleMode mode) -> void {
    update_rates_and_mode(m_src_sr, m_dst_sr, mode);
    const size_t min_scratch =
        static_cast<size_t>(static_cast<int>(std::ceil(std::max(1.0, m_speed_ratio))) + 16);
    if (m_scratch.size() < min_scratch) {
        m_scratch.assign(min_scratch, 0.0f);
    }
}

auto ResamplingRingBuffer::clear() -> void {
    m_ring.clear();
    m_resampler.reset();
    m_resample_underruns.store(0, std::memory_order_relaxed);
}

auto ResamplingRingBuffer::push(const float* src, int num_src_samples) NETBENDER_NONBLOCKING
    -> int {
    return m_ring.put(src, num_src_samples);
}

auto ResamplingRingBuffer::push(const double* src, int num_src_samples) NETBENDER_NONBLOCKING
    -> int {
    return m_ring.put(src, num_src_samples);
}

auto ResamplingRingBuffer::push_zeros(int num_src_samples) NETBENDER_NONBLOCKING -> int {
    return m_ring.put_zeros(num_src_samples);
}

auto ResamplingRingBuffer::required_src_samples_for_pull(int num_dst_samples) const noexcept
    NETBENDER_NONBLOCKING -> int {
    if (num_dst_samples <= 0) {
        return 0;
    }
    if (!m_resampling_active) {
        return num_dst_samples;
    }
    return m_resampler.required_input_samples(m_speed_ratio, num_dst_samples);
}

auto ResamplingRingBuffer::can_pull(int num_dst_samples) const NETBENDER_NONBLOCKING -> bool {
    if (num_dst_samples <= 0) {
        return true;
    }
    if (m_ring.get_capacity() <= 0) {
        return false;
    }
    return m_ring.get_available() >= required_src_samples_for_pull(num_dst_samples);
}

auto ResamplingRingBuffer::max_chunk_out_samples() const noexcept NETBENDER_NONBLOCKING -> int {
    const int scratch_cap = static_cast<int>(m_scratch.size());
    if (scratch_cap <= 0 || m_speed_ratio <= 0.0) {
        return 0;
    }

    const double budget = static_cast<double>(scratch_cap) - m_resampler.get_sub_sample_pos() - 1.0;
    int chunk_out = 1;
    if (budget > 0.0) {
        chunk_out = std::max(1, static_cast<int>(std::floor(budget / m_speed_ratio)) + 1);
    }

    while (chunk_out > 1 && required_src_samples_for_pull(chunk_out) > scratch_cap) {
        --chunk_out;
    }
    return chunk_out;
}

template <typename OutT>
auto ResamplingRingBuffer::pull_chunk_locked(OutT* dst, int chunk_out,
                                             int max_src_available) NETBENDER_NONBLOCKING -> int {
    if (chunk_out <= 0) {
        return 0;
    }

    const int scratch_cap = static_cast<int>(m_scratch.size());
    const int req = m_resampler.required_input_samples(m_speed_ratio, chunk_out);
    const int avail = std::clamp(max_src_available, 0, scratch_cap);
    const int to_peek = std::min(req, avail);

    const int peeked = (to_peek > 0) ? m_ring.peek(m_scratch.data(), to_peek) : 0;
    if (peeked < req) {
        m_resample_underruns.fetch_add(static_cast<uint64_t>(req - peeked),
                                       std::memory_order_relaxed);
    }

    const ResampleResult res =
        m_resampler.process_for_output(m_speed_ratio, m_scratch.data(), peeked, dst, chunk_out);
    if (res.input_consumed > 0) {
        (void)m_ring.drop(res.input_consumed);
    }
    return res.output_produced;
}

template <typename OutT>
auto ResamplingRingBuffer::pull_impl(OutT* dst, int num_dst_samples) NETBENDER_NONBLOCKING -> int {
    if (num_dst_samples <= 0) {
        return 0;
    }

    if (m_ring.get_capacity() <= 0) {
        if (dst != nullptr) {
            std::fill(dst, dst + num_dst_samples, OutT{0});
        }
        return 0;
    }

    if (!m_resampling_active) {
        if (dst == nullptr) {
            return m_ring.drop(num_dst_samples);
        }
        return m_ring.get(dst, num_dst_samples);
    }

    int offset = 0;
    while (offset < num_dst_samples) {
        const int max_chunk = max_chunk_out_samples();
        if (max_chunk <= 0) {
            if (dst != nullptr) {
                std::fill(dst + offset, dst + num_dst_samples, OutT{0});
            }
            return offset;
        }
        const int chunk = std::min(num_dst_samples - offset, max_chunk);
        const int avail = m_ring.get_available();
        OutT* chunk_dst = (dst != nullptr) ? (dst + offset) : nullptr;
        offset += pull_chunk_locked(chunk_dst, chunk, avail);
    }
    return offset;
}

auto ResamplingRingBuffer::pull(float* dst, int num_dst_samples) NETBENDER_NONBLOCKING -> int {
    return pull_impl(dst, num_dst_samples);
}

auto ResamplingRingBuffer::pull(double* dst, int num_dst_samples) NETBENDER_NONBLOCKING -> int {
    return pull_impl(dst, num_dst_samples);
}

//==============================================================================
auto ResamplingRingBufferBank::init(int num_channels, int capacity_src_samples, double src_sr,
                                    double dst_sr, ResampleMode mode, ResamplerQuality quality,
                                    int scratch_samples) -> void {
    const int n = std::max(0, num_channels);
    m_channels.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        m_channels[static_cast<size_t>(i)].init(capacity_src_samples, src_sr, dst_sr, mode, quality,
                                                scratch_samples);
    }
}

auto ResamplingRingBufferBank::set_sample_rates(double src_sr, double dst_sr) -> void {
    for (auto& ch : m_channels) {
        ch.set_sample_rates(src_sr, dst_sr);
    }
}

auto ResamplingRingBufferBank::set_resample_mode(ResampleMode mode) -> void {
    for (auto& ch : m_channels) {
        ch.set_resample_mode(mode);
    }
}

auto ResamplingRingBufferBank::clear() -> void {
    for (auto& ch : m_channels) {
        ch.clear();
    }
}

auto ResamplingRingBufferBank::push_channel(int channel, const float* src,
                                            int num_src_samples) NETBENDER_NONBLOCKING -> int {
    if (channel < 0 || channel >= static_cast<int>(m_channels.size())) {
        return 0;
    }
    return m_channels[static_cast<size_t>(channel)].push(src, num_src_samples);
}

auto ResamplingRingBufferBank::push_channel(int channel, const double* src,
                                            int num_src_samples) NETBENDER_NONBLOCKING -> int {
    if (channel < 0 || channel >= static_cast<int>(m_channels.size())) {
        return 0;
    }
    return m_channels[static_cast<size_t>(channel)].push(src, num_src_samples);
}

auto ResamplingRingBufferBank::push_zeros_all(int num_src_samples) NETBENDER_NONBLOCKING -> void {
    for (auto& ch : m_channels) {
        (void)ch.push_zeros(num_src_samples);
    }
}

auto ResamplingRingBufferBank::get_available_src() const NETBENDER_NONBLOCKING -> int {
    if (m_channels.empty()) {
        return 0;
    }
    int min_avail = std::numeric_limits<int>::max();
    for (const auto& ch : m_channels) {
        min_avail = std::min(min_avail, ch.get_available_src());
    }
    return min_avail;
}

auto ResamplingRingBufferBank::get_free_space_src() const NETBENDER_NONBLOCKING -> int {
    if (m_channels.empty()) {
        return 0;
    }
    int min_free = std::numeric_limits<int>::max();
    for (const auto& ch : m_channels) {
        min_free = std::min(min_free, ch.get_free_space_src());
    }
    return min_free;
}

auto ResamplingRingBufferBank::required_src_samples_for_pull(int num_dst_samples) const
    NETBENDER_NONBLOCKING -> int {
    if (m_channels.empty()) {
        return 0;
    }
    return m_channels[0].required_src_samples_for_pull(num_dst_samples);
}

auto ResamplingRingBufferBank::can_pull(int num_dst_samples) const NETBENDER_NONBLOCKING -> bool {
    if (num_dst_samples <= 0) {
        return true;
    }
    if (m_channels.empty() || m_channels[0].get_capacity_src() <= 0) {
        return false;
    }
    return get_available_src() >= required_src_samples_for_pull(num_dst_samples);
}

auto ResamplingRingBufferBank::get_overrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
    uint64_t total = 0;
    for (const auto& ch : m_channels) {
        total += ch.get_overrun_count();
    }
    return total;
}

auto ResamplingRingBufferBank::get_underrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
    uint64_t total = 0;
    for (const auto& ch : m_channels) {
        total += ch.get_underrun_count();
    }
    return total;
}

template <typename OutT>
auto ResamplingRingBufferBank::pull_bank_impl(OutT* const* dst_channels, int num_dst_channels,
                                              int num_dst_samples) NETBENDER_NONBLOCKING -> int {
    const int n_ch = static_cast<int>(m_channels.size());
    if (num_dst_samples <= 0 || n_ch <= 0) {
        for (int c = 0; c < num_dst_channels; ++c) {
            if (dst_channels != nullptr && dst_channels[c] != nullptr && num_dst_samples > 0) {
                std::fill(dst_channels[c], dst_channels[c] + num_dst_samples, OutT{0});
            }
        }
        return 0;
    }

    // Clear any caller output channels beyond the bank size
    for (int c = n_ch; c < num_dst_channels; ++c) {
        if (dst_channels != nullptr && dst_channels[c] != nullptr) {
            std::fill(dst_channels[c], dst_channels[c] + num_dst_samples, OutT{0});
        }
    }

    if (!is_resampling_active()) {
        const int common_avail = get_available_src();
        const int to_pull = std::min(common_avail, num_dst_samples);
        const int shortfall = num_dst_samples - to_pull;

        for (int c = 0; c < n_ch; ++c) {
            auto& ch = m_channels[static_cast<size_t>(c)];
            OutT* dst =
                (dst_channels != nullptr && c < num_dst_channels) ? dst_channels[c] : nullptr;
            if (dst != nullptr) {
                if (to_pull > 0) {
                    (void)ch.m_ring.get(dst, to_pull);
                }
                if (shortfall > 0) {
                    std::fill(dst + to_pull, dst + num_dst_samples, OutT{0});
                    ch.m_resample_underruns.fetch_add(static_cast<uint64_t>(shortfall),
                                                      std::memory_order_relaxed);
                }
            } else if (to_pull > 0) {
                (void)ch.m_ring.drop(to_pull);
            }
        }
        return to_pull;
    }

    // Resampling active: process in chunks bounded by scratch size, locking common_avail across
    // all channels so every channel consumes identical source samples and keeps identical phase.
    int offset = 0;
    while (offset < num_dst_samples) {
        const int max_chunk = m_channels[0].max_chunk_out_samples();
        if (max_chunk <= 0) {
            for (int c = 0; c < std::min(n_ch, num_dst_channels); ++c) {
                if (dst_channels != nullptr && dst_channels[c] != nullptr) {
                    std::fill(dst_channels[c] + offset, dst_channels[c] + num_dst_samples, OutT{0});
                }
            }
            return offset;
        }

        const int chunk = std::min(num_dst_samples - offset, max_chunk);
        const int common_avail = get_available_src();

        for (int c = 0; c < n_ch; ++c) {
            OutT* dst =
                (dst_channels != nullptr && c < num_dst_channels && dst_channels[c] != nullptr)
                    ? (dst_channels[c] + offset)
                    : nullptr;
            (void)m_channels[static_cast<size_t>(c)].pull_chunk_locked(dst, chunk, common_avail);
        }

        offset += chunk;
    }
    return offset;
}

auto ResamplingRingBufferBank::pull(float* const* dst_channels, int num_dst_channels,
                                    int num_dst_samples) NETBENDER_NONBLOCKING -> int {
    return pull_bank_impl(dst_channels, num_dst_channels, num_dst_samples);
}

auto ResamplingRingBufferBank::pull(double* const* dst_channels, int num_dst_channels,
                                    int num_dst_samples) NETBENDER_NONBLOCKING -> int {
    return pull_bank_impl(dst_channels, num_dst_channels, num_dst_samples);
}
