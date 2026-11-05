#include "AudioRingBuffer.h"
#include <algorithm>
#include <cstring>

AudioRingBuffer::AudioRingBuffer(AudioRingBuffer&& other) noexcept {
    m_buffer = std::move(other.m_buffer);
    m_capacity = other.m_capacity;
    m_mask = other.m_mask;
    m_write_index = other.m_write_index;
    m_read_index = other.m_read_index;
    m_available.store(other.m_available.load());

    other.m_capacity = 0;
    other.m_mask = 0;
    other.m_write_index = 0;
    other.m_read_index = 0;
    other.m_available.store(0);
}

auto AudioRingBuffer::operator=(AudioRingBuffer&& other) noexcept -> AudioRingBuffer& {
    if (this != &other) {
        m_buffer = std::move(other.m_buffer);
        m_capacity = other.m_capacity;
        m_mask = other.m_mask;
        m_write_index = other.m_write_index;
        m_read_index = other.m_read_index;
        m_available.store(other.m_available.load());

        other.m_capacity = 0;
        other.m_mask = 0;
        other.m_write_index = 0;
        other.m_read_index = 0;
        other.m_available.store(0);
    }
    return *this;
}

auto AudioRingBuffer::init(int size) -> void {
    // Enforce power-of-two capacity for fast bitwise wrap-around
    int target = std::max(size * 4, 8192);
    m_capacity = 1;
    while (m_capacity < target) {
        m_capacity <<= 1;
    }
    m_mask = m_capacity - 1;
    m_buffer.assign(static_cast<size_t>(m_capacity), 0.0f);
    m_write_index = 0;
    m_read_index = 0;
    m_available.store(0);
}

auto AudioRingBuffer::put(const float* data, int num_samples) -> void {
    if (m_capacity <= 0 || num_samples <= 0 || m_buffer.empty()) {
        return;
    }

    int to_write = std::min(num_samples, m_capacity);

    if (data != nullptr) {
        int first_part = std::min(to_write, m_capacity - m_write_index);
        std::memcpy(m_buffer.data() + m_write_index, data, static_cast<size_t>(first_part) * sizeof(float));
        if (to_write > first_part) {
            std::memcpy(m_buffer.data(), data + first_part, static_cast<size_t>(to_write - first_part) * sizeof(float));
        }
    } else {
        int first_part = std::min(to_write, m_capacity - m_write_index);
        std::memset(m_buffer.data() + m_write_index, 0, static_cast<size_t>(first_part) * sizeof(float));
        if (to_write > first_part) {
            std::memset(m_buffer.data(), 0, static_cast<size_t>(to_write - first_part) * sizeof(float));
        }
    }

    m_write_index = (m_write_index + to_write) & m_mask;
    int prev_avail = m_available.fetch_add(to_write);
    if (prev_avail + to_write > m_capacity) {
        // Buffer overflow: clamp available and advance read_index past overwritten samples
        int overflow = (prev_avail + to_write) - m_capacity;
        m_read_index = (m_read_index + overflow) & m_mask;
        m_available.store(m_capacity);
    }
}

auto AudioRingBuffer::get(float* dest, int num_samples) -> void {
    if (m_capacity <= 0 || num_samples <= 0 || dest == nullptr) {
        return;
    }

    int avail = std::min(m_available.load(), m_capacity);
    int to_read = std::min(num_samples, avail);

    if (to_read > 0 && !m_buffer.empty()) {
        int first_part = std::min(to_read, m_capacity - m_read_index);
        std::memcpy(dest, m_buffer.data() + m_read_index, static_cast<size_t>(first_part) * sizeof(float));
        if (to_read > first_part) {
            std::memcpy(dest + first_part, m_buffer.data(), static_cast<size_t>(to_read - first_part) * sizeof(float));
        }
        m_read_index = (m_read_index + to_read) & m_mask;
        m_available.fetch_sub(to_read);
    }

    if (to_read < num_samples) {
        std::memset(dest + to_read, 0, static_cast<size_t>(num_samples - to_read) * sizeof(float));
    }
}

auto AudioRingBuffer::peek(float* dest, int num_samples) const -> int {
    if (m_capacity <= 0 || num_samples <= 0 || dest == nullptr || m_buffer.empty()) {
        return 0;
    }

    int avail = std::min(m_available.load(), m_capacity);
    int to_read = std::min(num_samples, avail);

    if (to_read > 0) {
        int first_part = std::min(to_read, m_capacity - m_read_index);
        std::memcpy(dest, m_buffer.data() + m_read_index, static_cast<size_t>(first_part) * sizeof(float));
        if (to_read > first_part) {
            std::memcpy(dest + first_part, m_buffer.data(), static_cast<size_t>(to_read - first_part) * sizeof(float));
        }
    }
    return to_read;
}

auto AudioRingBuffer::drop(int num_samples) -> void {
    if (m_capacity <= 0 || num_samples <= 0) {
        return;
    }

    int avail = std::min(m_available.load(), m_capacity);
    int to_drop = std::min(num_samples, avail);
    m_read_index = (m_read_index + to_drop) & m_mask;
    m_available.fetch_sub(to_drop);
}

auto AudioRingBuffer::clear() -> void {
    if (m_capacity > 0) {
        std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
    }
    m_write_index = 0;
    m_read_index = 0;
    m_available.store(0);
}

auto AudioRingBuffer::get_available() const -> int {
    return std::min(m_available.load(), m_capacity);
}

auto AudioRingBuffer::get_capacity() const -> int {
    return m_capacity;
}

