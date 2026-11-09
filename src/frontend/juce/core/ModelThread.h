#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <functional>

class NetBenderAudioProcessor;

class ModelThread : public juce::Thread {
public:
    explicit ModelThread(std::function<void()> compute_fn);
    explicit ModelThread(NetBenderAudioProcessor& processor);
    ~ModelThread() override;

    auto run() -> void override;
    auto trigger_compute() -> void;
    auto stop() -> void;
    [[nodiscard]] auto is_processing() const -> bool { return m_processing.load(); }

private:
    std::function<void()> m_compute_fn;
    juce::WaitableEvent m_event;
    std::atomic<bool> m_processing { false };
};

