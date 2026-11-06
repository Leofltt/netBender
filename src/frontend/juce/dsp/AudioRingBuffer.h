#pragma once

#include <vector>
#include <atomic>
#include <cstddef>

class AudioRingBuffer {
public:
    AudioRingBuffer() = default;
    AudioRingBuffer(AudioRingBuffer&& other) noexcept;
    auto operator=(AudioRingBuffer&& other) noexcept -> AudioRingBuffer&;

    AudioRingBuffer(const AudioRingBuffer&) = delete;
    auto operator=(const AudioRingBuffer&) -> AudioRingBuffer& = delete;
    ~AudioRingBuffer() = default;

    auto init(int size) -> void;
    auto put(const float* data, int num_samples) -> void;
    auto get(float* dest, int num_samples) -> void;
    auto peek(float* dest, int num_samples) const -> int;
    auto drop(int num_samples) -> void;
    auto clear() -> void;

    [[nodiscard]] auto get_available() const -> int;
    [[nodiscard]] auto get_capacity() const -> int;

private:
    std::vector<float> m_buffer;
    int m_capacity { 0 };
    int m_mask { 0 };
    int m_write_index { 0 };
    int m_read_index { 0 };
    std::atomic<int> m_available { 0 };
};

