#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "rtsan_utils.h"

/**
 * Wait-free single-producer / single-consumer (SPSC) lock-free audio ring buffer.
 *
 * Thread-safety model:
 * - Producer thread (or audio callback) may call put*(), get_free_space(), get_overrun_count().
 * - Consumer thread (or audio callback) may call get*(), peek(), drop(), get_available(),
 *   get_underrun_count().
 * - init(), init_with_capacity(), and clear() must only be called while both producer and
 *   consumer are quiescent.
 */
class AudioRingBuffer {
public:
    AudioRingBuffer() = default;
    AudioRingBuffer(AudioRingBuffer&& other) noexcept;
    auto operator=(AudioRingBuffer&& other) noexcept -> AudioRingBuffer&;

    AudioRingBuffer(const AudioRingBuffer&) = delete;
    auto operator=(const AudioRingBuffer&) -> AudioRingBuffer& = delete;
    ~AudioRingBuffer() = default;

    auto init(int size) -> void;
    auto init_with_capacity(int min_capacity) -> void;

    auto put(const float* data, int num_samples) NETBENDER_NONBLOCKING -> int;
    auto put(const double* data, int num_samples) NETBENDER_NONBLOCKING -> int;
    auto put(std::nullptr_t, int num_samples) NETBENDER_NONBLOCKING -> int;
    auto put_zeros(int num_samples) NETBENDER_NONBLOCKING -> int;

    auto get(float* dest, int num_samples) NETBENDER_NONBLOCKING -> int;
    auto get(double* dest, int num_samples) NETBENDER_NONBLOCKING -> int;
    auto peek(float* dest, int num_samples) const NETBENDER_NONBLOCKING -> int;
    auto drop(int num_samples) NETBENDER_NONBLOCKING -> int;

    auto clear() -> void;

    [[nodiscard]] auto get_available() const NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto get_free_space() const NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto get_capacity() const NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto get_overrun_count() const NETBENDER_NONBLOCKING -> uint64_t;
    [[nodiscard]] auto get_underrun_count() const NETBENDER_NONBLOCKING -> uint64_t;
    auto reset_counters() NETBENDER_NONBLOCKING -> void;

private:
    std::vector<float> m_buffer;
    int m_capacity{0};
    uint64_t m_mask{0};

    alignas(64) std::atomic<uint64_t> m_write_pos{0};
    alignas(64) std::atomic<uint64_t> m_read_pos{0};
    alignas(64) std::atomic<uint64_t> m_overruns{0};
    alignas(64) std::atomic<uint64_t> m_underruns{0};
};
