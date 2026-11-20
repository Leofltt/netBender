#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include "rtsan_utils.h"

/**
 * Lock-free reader/reconfigure gate for real-time audio callbacks.
 *
 * Audio threads acquire ScopedRtReader (wait-free/lock-free CAS with zero syscalls).
 * Non-RT reconfiguration threads acquire ScopedReconfigureGate to block new readers
 * and drain any active audio-thread readers before mutating shared buffers.
 */
class RtGate {
public:
    static constexpr uint32_t k_reconfiguring_bit = 0x80000000u;

    RtGate() = default;
    ~RtGate() = default;

    RtGate(const RtGate&) = delete;
    auto operator=(const RtGate&) -> RtGate& = delete;
    RtGate(RtGate&&) = delete;
    auto operator=(RtGate&&) -> RtGate& = delete;

private:
    friend class ScopedRtReader;
    friend class ScopedReconfigureGate;

    std::atomic<uint32_t> m_state{0};
};

class ScopedRtReader {
public:
    explicit ScopedRtReader(RtGate& gate) NETBENDER_NONBLOCKING : m_gate(gate.m_state) {
        uint32_t state = m_gate.load(std::memory_order_acquire);
        while ((state & RtGate::k_reconfiguring_bit) == 0u) {
            if (m_gate.compare_exchange_weak(state, state + 1u, std::memory_order_acquire,
                                             std::memory_order_relaxed)) {
                m_acquired = true;
                break;
            }
        }
    }

    ~ScopedRtReader() NETBENDER_NONBLOCKING {
        if (m_acquired) {
            m_gate.fetch_sub(1u, std::memory_order_release);
        }
    }

    ScopedRtReader(const ScopedRtReader&) = delete;
    auto operator=(const ScopedRtReader&) -> ScopedRtReader& = delete;
    ScopedRtReader(ScopedRtReader&&) = delete;
    auto operator=(ScopedRtReader&&) -> ScopedRtReader& = delete;

    [[nodiscard]] auto is_acquired() const NETBENDER_NONBLOCKING -> bool { return m_acquired; }

private:
    std::atomic<uint32_t>& m_gate;
    bool m_acquired{false};
};

class ScopedReconfigureGate {
public:
    explicit ScopedReconfigureGate(RtGate& gate) : m_gate(gate.m_state) {
        m_gate.fetch_or(RtGate::k_reconfiguring_bit, std::memory_order_acq_rel);
        while ((m_gate.load(std::memory_order_acquire) & ~RtGate::k_reconfiguring_bit) != 0u) {
            std::this_thread::yield();
        }
    }

    ~ScopedReconfigureGate() {
        m_gate.fetch_and(~RtGate::k_reconfiguring_bit, std::memory_order_release);
    }

    ScopedReconfigureGate(const ScopedReconfigureGate&) = delete;
    auto operator=(const ScopedReconfigureGate&) -> ScopedReconfigureGate& = delete;
    ScopedReconfigureGate(ScopedReconfigureGate&&) = delete;
    auto operator=(ScopedReconfigureGate&&) -> ScopedReconfigureGate& = delete;

private:
    std::atomic<uint32_t>& m_gate;
};
