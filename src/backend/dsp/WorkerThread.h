#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include "rtsan_utils.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#elif !defined(_WIN32)
#include <semaphore.h>
#endif

/**
 * Frontend-agnostic background worker thread triggered from a real-time audio callback
 * via an allocation-free native OS semaphore.
 */
class WorkerThread {
public:
    explicit WorkerThread(std::function<void()> compute_fn);
    ~WorkerThread();

    WorkerThread(const WorkerThread&) = delete;
    auto operator=(const WorkerThread&) -> WorkerThread& = delete;
    WorkerThread(WorkerThread&&) = delete;
    auto operator=(WorkerThread&&) -> WorkerThread& = delete;

    auto start() -> void;
    auto stop() -> void;
    auto trigger() NETBENDER_NONBLOCKING -> void;

    [[nodiscard]] auto is_running() const NETBENDER_NONBLOCKING -> bool {
        return m_running.load(std::memory_order_acquire);
    }

    [[nodiscard]] auto is_processing() const NETBENDER_NONBLOCKING -> bool {
        return m_processing.load(std::memory_order_acquire) ||
               m_pending.load(std::memory_order_acquire);
    }

private:
    auto init_semaphore() -> void;
    auto destroy_semaphore() -> void;
    auto drain_semaphore() -> void;
    auto signal_semaphore() NETBENDER_NONBLOCKING -> void;
    auto wait_semaphore() -> void;
    auto run() -> void;

    std::function<void()> m_compute_fn;
    std::thread m_thread;
#if defined(__APPLE__)
    dispatch_semaphore_t m_sem{nullptr};
#elif defined(_WIN32)
    void* m_sem{nullptr};
#else
    sem_t m_sem{};
#endif
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_should_exit{false};
    std::atomic<bool> m_pending{false};
    std::atomic<bool> m_processing{false};
};
