#include "PluginProcessor.h"
#include "dsp/AudioRingBuffer.h"
#include "dsp/SafetySentry.h"
#include "rtsan_utils.h"

#include <JuceHeader.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace {

auto run_ring_buffer_rt(AudioRingBuffer& rb, const float* input, float* output,
                        int block_size) NETBENDER_NONBLOCKING -> void {
    for (int i = 0; i < 64; ++i) {
        rb.put(input, block_size);
        (void)rb.get_available();
        (void)rb.peek(output, block_size);
        rb.drop(block_size / 2);
        rb.get(output, block_size / 2);
    }
}

auto run_safety_sentry_rt(SafetySentry& sentry, float* wet, const float* dry,
                          int block_size) NETBENDER_NONBLOCKING -> void {
    // Normal signal + soft saturation
    sentry.process_channel(wet, dry, block_size, 0.8f, 0.2f);

    // Inject NaN to trip the fuse
    wet[0] = std::numeric_limits<float>::quiet_NaN();
    sentry.process_channel(wet, dry, block_size, 1.0f, 0.0f);

    // Tick recovery across cooldown blocks (no-capture reset callback)
    for (int i = 0; i < 50; ++i) {
        sentry.tick_recovery([&sentry] { sentry.reset_fuse(); });
    }
}

auto test_dsp_primitives() -> void {
    std::cout << "[1/3] Checking AudioRingBuffer & SafetySentry under RTSan...\n";

    constexpr int k_block_size = 512;
    AudioRingBuffer rb;
    rb.init(2048);

    std::vector<float> input(k_block_size, 0.25f);
    std::vector<float> output(k_block_size, 0.0f);
    run_ring_buffer_rt(rb, input.data(), output.data(), k_block_size);

    SafetySentry sentry;
    sentry.set_auto_reset(true);
    std::vector<float> wet(k_block_size, 0.9f);
    std::vector<float> dry(k_block_size, 0.1f);
    run_safety_sentry_rt(sentry, wet.data(), dry.data(), k_block_size);

    std::cout << "      PASS\n";
}

auto test_processor_bypass() -> void {
    std::cout << "[2/3] Checking NetBenderAudioProcessor::processBlock (no model)...\n";

    constexpr double k_sample_rate = 48000.0;
    constexpr int k_block_size = 512;

    NetBenderAudioProcessor processor;
    processor.prepareToPlay(k_sample_rate, k_block_size);

    juce::AudioBuffer<float> audio(2, k_block_size);
    juce::MidiBuffer midi_on;
    midi_on.addEvent(juce::MidiMessage::noteOn(1, 60, 0.8f), 0);
    juce::MidiBuffer midi_off;
    midi_off.addEvent(juce::MidiMessage::noteOff(1, 60), 0);
    juce::MidiBuffer midi_empty;

    for (int block = 0; block < 100; ++block) {
        for (int ch = 0; ch < audio.getNumChannels(); ++ch) {
            float* data = audio.getWritePointer(ch);
            for (int i = 0; i < k_block_size; ++i) {
                float phase = static_cast<float>(block * k_block_size + i) * 0.05f;
                data[i] = 0.25f * std::sin(phase);
            }
        }

        auto& midi = (block == 10) ? midi_on : ((block == 30) ? midi_off : midi_empty);
        processor.processBlock(audio, midi);
    }

    processor.releaseResources();
    std::cout << "      PASS\n";
}

auto test_processor_with_model(const juce::File& model_file) -> bool {
    std::cout << "[3/3] Checking NetBenderAudioProcessor::processBlock with model: "
              << model_file.getFullPathName() << "\n";

    constexpr double k_sample_rate = 48000.0;
    constexpr int k_block_size = 512;
    constexpr int k_large_block_size = 16384;

    NetBenderAudioProcessor processor;
    processor.prepareToPlay(k_sample_rate, k_block_size);

    if (!processor.loadModel(model_file)) {
        std::cerr << "      ERROR: Failed to load model: " << model_file.getFullPathName() << "\n";
        return false;
    }

    processor.setAutoResetFuse(true);

    juce::AudioBuffer<float> audio(2, k_block_size);
    juce::AudioBuffer<float> large_audio(2, k_large_block_size);
    large_audio.clear();
    juce::MidiBuffer midi_on;
    midi_on.addEvent(juce::MidiMessage::noteOn(1, 60, 0.9f), 0);
    juce::MidiBuffer midi_off;
    midi_off.addEvent(juce::MidiMessage::noteOff(1, 60), 0);
    juce::MidiBuffer midi_empty;

    for (int block = 0; block < 200; ++block) {
        for (int ch = 0; ch < audio.getNumChannels(); ++ch) {
            float* data = audio.getWritePointer(ch);
            for (int i = 0; i < k_block_size; ++i) {
                float phase = static_cast<float>(block * k_block_size + i) * 0.05f;
                data[i] = 0.25f * std::sin(phase);
            }
        }

        if (block == 120) {
            // Exercise blown-fuse bypass + auto-recovery cooldown on the audio thread
            processor.getSafetySentry().trip_fuse();
        }

        auto& midi = (block == 20) ? midi_on : ((block == 60) ? midi_off : midi_empty);
        processor.processBlock(audio, midi);

        // Give ModelThread time to run concurrently and exercise lock/ringbuffer handoff
        if (block % 4 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // Exercise a host block larger than pre-allocated scratch (verifies zero heap allocation)
    processor.processBlock(large_audio, midi_empty);

    processor.releaseResources();
    std::cout << "      PASS\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    juce::ScopedJuceInitialiser_GUI juce_init;

    std::cout << "=== NetBender RealtimeSanitizer (RTSan) Test Suite ===\n";
    std::cout << "RTSan instrumentation active: "
              << (netbender::rtsan::is_enabled() ? "YES (-fsanitize=realtime)" : "NO (dry run)")
              << "\n\n";

    test_dsp_primitives();
    test_processor_bypass();

    if (argc > 1) {
        juce::File model_file(argv[1]);
        if (!model_file.existsAsFile()) {
            std::cerr << "[3/3] Model file not found: " << argv[1] << "\n";
            return 1;
        }
        if (!test_processor_with_model(model_file)) {
            return 1;
        }
    } else {
        std::cout << "[3/3] Skipped loaded-model test (pass a model path, e.g. "
                     "models/darbouka_xnnpack.pte, to exercise ModelThread + resampling).\n";
    }

    std::cout << "\nAll RTSan checks completed.\n";
    return 0;
}
