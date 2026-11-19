#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <functional>
#include "rtsan_utils.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#elif !defined(_WIN32)
#include <semaphore.h>
#endif

class NetBenderAudioProcessor;

class ModelThread : public juce::Thread {
public:
    explicit ModelThread(std::function<void()> compute_fn);
    explicit ModelThread(NetBenderAudioProcessor& processor);
    ~ModelThread() override;

    auto run() -> void override;
    auto trigger_compute() NETBENDER_NONBLOCKING -> void;
    auto stop() -> void;
    [[nodiscard]] auto is_processing() const NETBENDER_NONBLOCKING -> bool {
        return m_processing.load(std::memory_order_acquire);
    }

private:
    auto init_semaphore() -> void;
    auto destroy_semaphore() -> void;
    auto signal_semaphore() NETBENDER_NONBLOCKING -> void;
    auto wait_semaphore() -> void;

    std::function<void()> m_compute_fn;
#if defined(__APPLE__)
    dispatch_semaphore_t m_sem { nullptr };
#elif defined(_WIN32)
    void* m_sem { nullptr };
#else
    sem_t m_sem {};
#endif
    std::atomic<bool> m_processing { false };
};

