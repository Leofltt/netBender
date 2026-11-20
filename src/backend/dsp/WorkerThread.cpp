#include "WorkerThread.h"

#if defined(__APPLE__)
#include <pthread/qos.h>

extern "C" {
void* objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void* pool);
}

namespace {

class ScopedAutoreleasePool {
public:
    ScopedAutoreleasePool() : m_pool(objc_autoreleasePoolPush()) {}
    ~ScopedAutoreleasePool() {
        if (m_pool != nullptr) {
            objc_autoreleasePoolPop(m_pool);
        }
    }

    ScopedAutoreleasePool(const ScopedAutoreleasePool&) = delete;
    auto operator=(const ScopedAutoreleasePool&) -> ScopedAutoreleasePool& = delete;

private:
    void* m_pool{nullptr};
};

}  // namespace
#elif defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#endif

WorkerThread::WorkerThread(std::function<void()> compute_fn) : m_compute_fn(std::move(compute_fn)) {
    init_semaphore();
}

WorkerThread::~WorkerThread() {
    stop();
    destroy_semaphore();
}

auto WorkerThread::init_semaphore() -> void {
#if defined(__APPLE__)
    m_sem = dispatch_semaphore_create(0);
#elif defined(_WIN32)
    m_sem = CreateSemaphoreA(nullptr, 0, LONG_MAX, nullptr);
#else
    sem_init(&m_sem, 0, 0);
#endif
}

auto WorkerThread::destroy_semaphore() -> void {
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

auto WorkerThread::drain_semaphore() -> void {
#if defined(__APPLE__)
    if (m_sem != nullptr) {
        while (dispatch_semaphore_wait(m_sem, DISPATCH_TIME_NOW) == 0) {
        }
    }
#elif defined(_WIN32)
    if (m_sem != nullptr) {
        while (WaitForSingleObject(static_cast<HANDLE>(m_sem), 0) == WAIT_OBJECT_0) {
        }
    }
#else
    while (sem_trywait(&m_sem) == 0) {
    }
#endif
}

auto WorkerThread::signal_semaphore() NETBENDER_NONBLOCKING -> void {
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

auto WorkerThread::wait_semaphore() -> void {
#if defined(__APPLE__)
    if (m_sem != nullptr) {
        dispatch_semaphore_wait(m_sem, DISPATCH_TIME_FOREVER);
    }
#elif defined(_WIN32)
    if (m_sem != nullptr) {
        WaitForSingleObject(static_cast<HANDLE>(m_sem), INFINITE);
    }
#else
    while (sem_wait(&m_sem) == -1 && errno == EINTR) {
    }
#endif
}

auto WorkerThread::start() -> void {
    if (m_thread.joinable()) {
        return;
    }
    drain_semaphore();
    m_should_exit.store(false, std::memory_order_release);
    m_pending.store(false, std::memory_order_release);
    m_processing.store(false, std::memory_order_release);
    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&WorkerThread::run, this);
}

auto WorkerThread::stop() -> void {
    if (!m_thread.joinable()) {
        m_running.store(false, std::memory_order_release);
        m_pending.store(false, std::memory_order_release);
        m_processing.store(false, std::memory_order_release);
        return;
    }
    m_should_exit.store(true, std::memory_order_release);
    signal_semaphore();
    m_thread.join();
    m_running.store(false, std::memory_order_release);
    m_pending.store(false, std::memory_order_release);
    m_processing.store(false, std::memory_order_release);
}

auto WorkerThread::trigger() NETBENDER_NONBLOCKING -> void {
    // Coalesce concurrent triggers into at most one pending wakeup so permits stay bounded
    // and triggers issued mid-compute never lose a wakeup.
    if (!m_pending.exchange(true, std::memory_order_acq_rel)) {
        signal_semaphore();
    }
}

auto WorkerThread::run() -> void {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif

    while (!m_should_exit.load(std::memory_order_acquire)) {
        wait_semaphore();

        if (m_should_exit.load(std::memory_order_acquire)) {
            break;
        }

        if (m_pending.load(std::memory_order_acquire)) {
            m_processing.store(true, std::memory_order_release);
            while (!m_should_exit.load(std::memory_order_acquire) &&
                   m_pending.exchange(false, std::memory_order_acq_rel)) {
#if defined(__APPLE__)
                const ScopedAutoreleasePool pool;
#endif
                if (m_compute_fn) {
                    m_compute_fn();
                }
            }
            m_processing.store(false, std::memory_order_release);
        }
    }
}
