#include "ModelThread.h"
#include "PluginProcessor.h"

ModelThread::ModelThread(std::function<void()> compute_fn)
    : juce::Thread("Model Thread"), m_compute_fn(std::move(compute_fn)) {
}

ModelThread::ModelThread(NetBenderAudioProcessor& processor)
    : juce::Thread("Model Thread"), m_compute_fn([&processor] { processor.runInference(); }) {
}

ModelThread::~ModelThread() {
    stop();
}

auto ModelThread::stop() -> void {
    signalThreadShouldExit();
    m_event.signal();
    stopThread(-1);
}

auto ModelThread::run() -> void {
    while (!threadShouldExit()) {
        m_event.wait(20);

        if (threadShouldExit()) {
            break;
        }

        if (m_processing.load()) {
#if JUCE_MAC || JUCE_IOS
            const juce::ScopedAutoReleasePool pool;
#endif
            if (m_compute_fn) {
                m_compute_fn();
            }
            m_processing.store(false);
        }
    }
}

auto ModelThread::trigger_compute() -> void {
    m_processing.store(true);
    m_event.signal();
}
