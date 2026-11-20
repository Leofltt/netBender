#include "AudioRingBuffer.h"

#include <algorithm>
#include <cstring>

namespace {

constexpr int64_t k_max_capacity = static_cast<int64_t>(1) << 30;

auto next_power_of_two_clamped(int64_t target) -> int {
    const int64_t clamped = std::clamp<int64_t>(target, 2, k_max_capacity);
    int cap = 1;
    while (cap < clamped) {
        cap <<= 1;
    }
    return cap;
}

auto copy_to_ring(float* dst, const float* src, size_t offset, size_t count) NETBENDER_NONBLOCKING
    -> void {
    std::memcpy(dst, src + offset, count * sizeof(float));
}

auto copy_to_ring(float* dst, const double* src, size_t offset, size_t count) NETBENDER_NONBLOCKING
    -> void {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[offset + i]);
    }
}

auto copy_to_ring(float* dst, std::nullptr_t, size_t /*offset*/, size_t count) NETBENDER_NONBLOCKING
    -> void {
    std::memset(dst, 0, count * sizeof(float));
}

auto copy_from_ring(float* dst, size_t offset, const float* src, size_t count) NETBENDER_NONBLOCKING
    -> void {
    std::memcpy(dst + offset, src, count * sizeof(float));
}

auto copy_from_ring(double* dst, size_t offset, const float* src,
                    size_t count) NETBENDER_NONBLOCKING -> void {
    for (size_t i = 0; i < count; ++i) {
        dst[offset + i] = static_cast<double>(src[i]);
    }
}

auto zero_fill_dest(float* dst, size_t offset, size_t count) NETBENDER_NONBLOCKING -> void {
    std::memset(dst + offset, 0, count * sizeof(float));
}

auto zero_fill_dest(double* dst, size_t offset, size_t count) NETBENDER_NONBLOCKING -> void {
    for (size_t i = 0; i < count; ++i) {
        dst[offset + i] = 0.0;
    }
}

auto compute_available(uint64_t write_pos, uint64_t read_pos, int capacity) NETBENDER_NONBLOCKING
    -> uint64_t {
    if (capacity <= 0 || write_pos <= read_pos) {
        return 0;
    }
    return std::min<uint64_t>(write_pos - read_pos, static_cast<uint64_t>(capacity));
}

template <typename DstPtr>
auto read_ring_segments(const std::vector<float>& buffer, int capacity, uint64_t mask,
                        uint64_t read_pos, size_t to_read, DstPtr dest) NETBENDER_NONBLOCKING
    -> void {
    const size_t read_idx = static_cast<size_t>(read_pos & mask);
    const size_t first_part = std::min(to_read, static_cast<size_t>(capacity) - read_idx);
    const size_t second_part = to_read - first_part;

    copy_from_ring(dest, 0, buffer.data() + read_idx, first_part);
    if (second_part > 0) {
        copy_from_ring(dest, first_part, buffer.data(), second_part);
    }
}

template <typename SrcPtr>
auto put_typed(std::vector<float>& buffer, int capacity, uint64_t mask,
               std::atomic<uint64_t>& write_pos_atom, const std::atomic<uint64_t>& read_pos_atom,
               std::atomic<uint64_t>& overruns_atom, SrcPtr data,
               int num_samples) NETBENDER_NONBLOCKING -> int {
    if (capacity <= 0 || num_samples <= 0 || buffer.empty()) {
        return 0;
    }

    const uint64_t write_pos = write_pos_atom.load(std::memory_order_relaxed);
    const uint64_t read_pos = read_pos_atom.load(std::memory_order_acquire);
    const uint64_t cap = static_cast<uint64_t>(capacity);
    const uint64_t used = write_pos - read_pos;
    const uint64_t free_space = (used < cap) ? (cap - used) : 0;
    const uint64_t requested = static_cast<uint64_t>(num_samples);
    const uint64_t to_write = std::min(requested, free_space);

    if (to_write < requested) {
        overruns_atom.fetch_add(requested - to_write, std::memory_order_relaxed);
    }

    if (to_write == 0) {
        return 0;
    }

    const size_t write_idx = static_cast<size_t>(write_pos & mask);
    const size_t first_part =
        std::min(static_cast<size_t>(to_write), static_cast<size_t>(capacity) - write_idx);
    const size_t second_part = static_cast<size_t>(to_write) - first_part;

    copy_to_ring(buffer.data() + write_idx, data, 0, first_part);
    if (second_part > 0) {
        copy_to_ring(buffer.data(), data, first_part, second_part);
    }

    write_pos_atom.store(write_pos + to_write, std::memory_order_release);
    return static_cast<int>(to_write);
}

template <typename DstPtr>
auto get_typed(const std::vector<float>& buffer, int capacity, uint64_t mask,
               const std::atomic<uint64_t>& write_pos_atom, std::atomic<uint64_t>& read_pos_atom,
               std::atomic<uint64_t>& underruns_atom, DstPtr dest,
               int num_samples) NETBENDER_NONBLOCKING -> int {
    if (dest == nullptr || num_samples <= 0) {
        return 0;
    }

    if (capacity <= 0 || buffer.empty()) {
        zero_fill_dest(dest, 0, static_cast<size_t>(num_samples));
        return 0;
    }

    const uint64_t read_pos = read_pos_atom.load(std::memory_order_relaxed);
    const uint64_t write_pos = write_pos_atom.load(std::memory_order_acquire);
    const uint64_t avail = compute_available(write_pos, read_pos, capacity);
    const uint64_t requested = static_cast<uint64_t>(num_samples);
    const uint64_t to_read = std::min(requested, avail);

    if (to_read > 0) {
        read_ring_segments(buffer, capacity, mask, read_pos, static_cast<size_t>(to_read), dest);
        read_pos_atom.store(read_pos + to_read, std::memory_order_release);
    }

    if (to_read < requested) {
        underruns_atom.fetch_add(requested - to_read, std::memory_order_relaxed);
        zero_fill_dest(dest, static_cast<size_t>(to_read),
                       static_cast<size_t>(requested - to_read));
    }

    return static_cast<int>(to_read);
}

}  // namespace

AudioRingBuffer::AudioRingBuffer(AudioRingBuffer&& other) noexcept
    : m_buffer(std::move(other.m_buffer)),
      m_capacity(other.m_capacity),
      m_mask(other.m_mask),
      m_write_pos(other.m_write_pos.load(std::memory_order_relaxed)),
      m_read_pos(other.m_read_pos.load(std::memory_order_relaxed)),
      m_overruns(other.m_overruns.load(std::memory_order_relaxed)),
      m_underruns(other.m_underruns.load(std::memory_order_relaxed)) {
    other.m_capacity = 0;
    other.m_mask = 0;
    other.m_write_pos.store(0, std::memory_order_relaxed);
    other.m_read_pos.store(0, std::memory_order_relaxed);
    other.m_overruns.store(0, std::memory_order_relaxed);
    other.m_underruns.store(0, std::memory_order_relaxed);
}

auto AudioRingBuffer::operator=(AudioRingBuffer&& other) noexcept -> AudioRingBuffer& {
    if (this != &other) {
        m_buffer = std::move(other.m_buffer);
        m_capacity = other.m_capacity;
        m_mask = other.m_mask;
        m_write_pos.store(other.m_write_pos.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
        m_read_pos.store(other.m_read_pos.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        m_overruns.store(other.m_overruns.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        m_underruns.store(other.m_underruns.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);

        other.m_capacity = 0;
        other.m_mask = 0;
        other.m_write_pos.store(0, std::memory_order_relaxed);
        other.m_read_pos.store(0, std::memory_order_relaxed);
        other.m_overruns.store(0, std::memory_order_relaxed);
        other.m_underruns.store(0, std::memory_order_relaxed);
    }
    return *this;
}

auto AudioRingBuffer::init(int size) -> void {
    const int64_t scaled = static_cast<int64_t>(std::max(size, 0)) * 4;
    const int64_t target = std::max<int64_t>(scaled, 8192);
    init_with_capacity(static_cast<int>(std::min(target, k_max_capacity)));
}

auto AudioRingBuffer::init_with_capacity(int min_capacity) -> void {
    m_capacity = next_power_of_two_clamped(static_cast<int64_t>(min_capacity));
    m_mask = static_cast<uint64_t>(m_capacity - 1);
    m_buffer.assign(static_cast<size_t>(m_capacity), 0.0f);
    m_write_pos.store(0, std::memory_order_relaxed);
    m_read_pos.store(0, std::memory_order_relaxed);
    m_overruns.store(0, std::memory_order_relaxed);
    m_underruns.store(0, std::memory_order_relaxed);
}

auto AudioRingBuffer::put(const float* data, int num_samples) NETBENDER_NONBLOCKING -> int {
    if (data == nullptr) {
        return put_zeros(num_samples);
    }
    return put_typed(m_buffer, m_capacity, m_mask, m_write_pos, m_read_pos, m_overruns, data,
                     num_samples);
}

auto AudioRingBuffer::put(const double* data, int num_samples) NETBENDER_NONBLOCKING -> int {
    if (data == nullptr) {
        return put_zeros(num_samples);
    }
    return put_typed(m_buffer, m_capacity, m_mask, m_write_pos, m_read_pos, m_overruns, data,
                     num_samples);
}

auto AudioRingBuffer::put(std::nullptr_t, int num_samples) NETBENDER_NONBLOCKING -> int {
    return put_zeros(num_samples);
}

auto AudioRingBuffer::put_zeros(int num_samples) NETBENDER_NONBLOCKING -> int {
    return put_typed(m_buffer, m_capacity, m_mask, m_write_pos, m_read_pos, m_overruns, nullptr,
                     num_samples);
}

auto AudioRingBuffer::get(float* dest, int num_samples) NETBENDER_NONBLOCKING -> int {
    return get_typed(m_buffer, m_capacity, m_mask, m_write_pos, m_read_pos, m_underruns, dest,
                     num_samples);
}

auto AudioRingBuffer::get(double* dest, int num_samples) NETBENDER_NONBLOCKING -> int {
    return get_typed(m_buffer, m_capacity, m_mask, m_write_pos, m_read_pos, m_underruns, dest,
                     num_samples);
}

auto AudioRingBuffer::peek(float* dest, int num_samples) const NETBENDER_NONBLOCKING -> int {
    if (m_capacity <= 0 || num_samples <= 0 || dest == nullptr || m_buffer.empty()) {
        return 0;
    }

    const uint64_t read_pos = m_read_pos.load(std::memory_order_relaxed);
    const uint64_t write_pos = m_write_pos.load(std::memory_order_acquire);
    const uint64_t avail = compute_available(write_pos, read_pos, m_capacity);
    const uint64_t to_read = std::min<uint64_t>(static_cast<uint64_t>(num_samples), avail);

    if (to_read > 0) {
        read_ring_segments(m_buffer, m_capacity, m_mask, read_pos, static_cast<size_t>(to_read),
                           dest);
    }

    return static_cast<int>(to_read);
}

auto AudioRingBuffer::drop(int num_samples) NETBENDER_NONBLOCKING -> int {
    if (m_capacity <= 0 || num_samples <= 0) {
        return 0;
    }

    const uint64_t read_pos = m_read_pos.load(std::memory_order_relaxed);
    const uint64_t write_pos = m_write_pos.load(std::memory_order_acquire);
    const uint64_t avail = compute_available(write_pos, read_pos, m_capacity);
    const uint64_t to_drop = std::min<uint64_t>(static_cast<uint64_t>(num_samples), avail);

    m_read_pos.store(read_pos + to_drop, std::memory_order_release);
    return static_cast<int>(to_drop);
}

auto AudioRingBuffer::clear() -> void {
    if (m_capacity > 0 && !m_buffer.empty()) {
        std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
    }
    m_write_pos.store(0, std::memory_order_relaxed);
    m_read_pos.store(0, std::memory_order_relaxed);
    m_overruns.store(0, std::memory_order_relaxed);
    m_underruns.store(0, std::memory_order_relaxed);
}

auto AudioRingBuffer::get_available() const NETBENDER_NONBLOCKING -> int {
    const uint64_t write_pos = m_write_pos.load(std::memory_order_acquire);
    const uint64_t read_pos = m_read_pos.load(std::memory_order_acquire);
    return static_cast<int>(compute_available(write_pos, read_pos, m_capacity));
}

auto AudioRingBuffer::get_free_space() const NETBENDER_NONBLOCKING -> int {
    if (m_capacity <= 0) {
        return 0;
    }
    return m_capacity - get_available();
}

auto AudioRingBuffer::get_capacity() const NETBENDER_NONBLOCKING -> int {
    return m_capacity;
}

auto AudioRingBuffer::get_overrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
    return m_overruns.load(std::memory_order_relaxed);
}

auto AudioRingBuffer::get_underrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
    return m_underruns.load(std::memory_order_relaxed);
}

auto AudioRingBuffer::reset_counters() NETBENDER_NONBLOCKING -> void {
    m_overruns.store(0, std::memory_order_relaxed);
    m_underruns.store(0, std::memory_order_relaxed);
}
