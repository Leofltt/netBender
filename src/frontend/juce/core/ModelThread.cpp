#include "ModelThread.h"
#include "PluginProcessor.h"

#if defined(_WIN32)
#include <windows.h>
#elif !defined(__APPLE__)
#include <cerrno>
#endif

ModelThread::ModelThread(std::function<void()> compute_fn)
    : juce::Thread("Model Thread"), m_compute_fn(std::move(compute_fn)) {
    init_semaphore();
}

ModelThread::ModelThread(NetBenderAudioProcessor& processor)
    : juce::Thread("Model Thread"), m_compute_fn([&processor] { processor.run_inference(); }) {
    init_semaphore();
}

ModelThread::~ModelThread() {
    stop();
    destroy_semaphore();
}

auto ModelThread::init_semaphore() -> void {
#if defined(__APPLE__)
    m_sem = dispatch_semaphore_create(0);
#elif defined(_WIN32)
    m_sem = CreateSemaphoreA(nullptr, 0, LONG_MAX, nullptr);
#else
    sem_init(&m_sem, 0, 0);
#endif
}

auto ModelThread::destroy_semaphore() -> void {
#if defined(__APPLE__)
    if (m_sem != nullptr) {
        dispatch_release(m_sem);
        m_sem = nullptr;
    }
#elif defined(_WIN32)
    if (m_sem != nullptr) {
        CloseHandle(static_cast<HANDLE>(m_sem));
        m_sem = nullptr;
    }
#else
    sem_destroy(&m_sem);
#endif
}

auto ModelThread::signal_semaphore() NETBENDER_NONBLOCKING -> void {
#if defined(__APPLE__)
    if (m_sem != nullptr) {
        dispatch_semaphore_signal(m_sem);
    }
#elif defined(_WIN32)
    if (m_sem != nullptr) {
        ReleaseSemaphore(static_cast<HANDLE>(m_sem), 1, nullptr);
    }
#else
    sem_post(&m_sem);
#endif
}

auto ModelThread::wait_semaphore() -> void {
#if defined(__APPLE__)
    if (m_sem != nullptr) {
        dispatch_semaphore_wait(m_sem, DISPATCH_TIME_FOREVER);
    }
#elif defined(_WIN32)
    if (m_sem != nullptr) {
        WaitForSingleObject(static_cast<HANDLE>(m_sem), INFINITE);
    }
#else
    while (sem_wait(&m_sem) == -1 && errno == EINTR) {}
#endif
}

auto ModelThread::stop() -> void {
    signalThreadShouldExit();
    signal_semaphore();
    stopThread(-1);
    m_processing.store(false, std::memory_order_release);
}

auto ModelThread::run() -> void {
    while (!threadShouldExit()) {
        wait_semaphore();

        if (threadShouldExit()) {
            break;
        }

        if (m_processing.load(std::memory_order_acquire)) {
#if JUCE_MAC || JUCE_IOS
            const juce::ScopedAutoReleasePool pool;
#endif
            if (m_compute_fn) {
                m_compute_fn();
            }
            m_processing.store(false, std::memory_order_release);
        }
    }
}

auto ModelThread::trigger_compute() NETBENDER_NONBLOCKING -> void {
    m_processing.store(true, std::memory_order_release);
    signal_semaphore();
}
