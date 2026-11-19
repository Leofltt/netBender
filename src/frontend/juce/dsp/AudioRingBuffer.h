#pragma once

#include <vector>
#include <atomic>
#include <cstddef>
#include "rtsan_utils.h"

class AudioRingBuffer {
public:
    AudioRingBuffer() = default;
    AudioRingBuffer(AudioRingBuffer&& other) noexcept;
    auto operator=(AudioRingBuffer&& other) noexcept -> AudioRingBuffer&;

    AudioRingBuffer(const AudioRingBuffer&) = delete;
    auto operator=(const AudioRingBuffer&) -> AudioRingBuffer& = delete;
    ~AudioRingBuffer() = default;

    auto init(int size) -> void;
    auto put(const float* data, int num_samples) NETBENDER_NONBLOCKING -> void;
    auto get(float* dest, int num_samples) NETBENDER_NONBLOCKING -> void;
    auto peek(float* dest, int num_samples) const NETBENDER_NONBLOCKING -> int;
    auto drop(int num_samples) NETBENDER_NONBLOCKING -> void;
    auto clear() -> void;

    [[nodiscard]] auto get_available() const NETBENDER_NONBLOCKING -> int;
    [[nodiscard]] auto get_capacity() const NETBENDER_NONBLOCKING -> int;

private:
    std::vector<float> m_buffer;
    int m_capacity { 0 };
    int m_mask { 0 };
    int m_write_index { 0 };
    int m_read_index { 0 };
    std::atomic<int> m_available { 0 };
};

