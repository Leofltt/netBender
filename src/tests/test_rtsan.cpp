#include "PluginProcessor.h"
#include "dsp/AudioRingBuffer.h"
#include "dsp/SafetySentry.h"
#include "dsp/Resampler.h"
#include "dsp/ResamplingRingBuffer.h"
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

auto run_resampler_rt(Resampler& resampler, ResamplingRingBufferBank& bank, const float* input,
                      float* output, int block_size, int oversized_block) NETBENDER_NONBLOCKING
    -> void {
    (void)resampler.process_for_output(48000.0 / 44100.0, input, block_size, output,
                                       block_size / 2);
    (void)resampler.process_for_input(48000.0 / 44100.0, input, block_size / 2, output, block_size);

    (void)bank.push_channel(0, input, oversized_block);
    (void)bank.push_channel(1, input, oversized_block);
    float* dst_ptrs[2]{output, output};
    if (bank.can_pull(oversized_block / 2)) {
        (void)bank.pull(dst_ptrs, 2, oversized_block / 2);
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
    std::cout << "[1/3] Checking AudioRingBuffer, Resampler, ResamplingRingBufferBank & "
                 "SafetySentry under RTSan...\n";

    constexpr int k_block_size = 512;
    AudioRingBuffer rb;
    rb.init(2048);

    std::vector<float> input(k_block_size, 0.25f);
    std::vector<float> output(k_block_size, 0.0f);
    run_ring_buffer_rt(rb, input.data(), output.data(), k_block_size);

    Resampler resampler(ResamplerQuality::CatmullRom);
    ResamplingRingBufferBank bank;
    bank.init(2, 16384, 48000.0, 44100.0, ResampleMode::Auto, ResamplerQuality::CatmullRom, 512);
    constexpr int k_oversized = 4096;
    std::vector<float> oversized_in(k_oversized * 2, 0.2f);
    std::vector<float> oversized_out(k_oversized, 0.0f);
    (void)bank.push_channel(0, oversized_in.data(), k_oversized);
    (void)bank.push_channel(1, oversized_in.data(), k_oversized);
    run_resampler_rt(resampler, bank, oversized_in.data(), oversized_out.data(), k_block_size,
                     k_oversized);

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

    // Verify full VST3/AU/Standalone preset state save & restore round-trip (non-RT)
    auto require = [](bool cond, const char* msg) {
        if (!cond) {
            std::cerr << "      FAIL: " << msg << "\n";
            std::exit(1);
        }
    };

    processor.setDryWet(0.37f);
    processor.setTriggerMode(NetBenderAudioProcessor::TriggerMode::Momentary);
    processor.setEnvelopeAttack(0.45f);
    processor.setEnvelopeRelease(0.77f);
    processor.setTransientThreshold(0.28f);
    processor.setLatentEnabled(true);
    processor.setLatentMode(NetBenderAudioProcessor::LatentMode::Quantize);
    processor.setLatentCoords(-0.6f, 0.4f);
    processor.setLatentDepth(1.25f);
    processor.setLatentSlew(0.65f);
    processor.setActiveLayerName("decoder.conv0.weight");
    processor.setLayerScale("decoder.conv0.weight", 1.85f);
    processor.setLayerOffset("decoder.conv0.weight", -0.12f);
    processor.setLayerHeat("decoder.conv0.weight", 0.31f);
    processor.setLayerMemory("decoder.conv0.weight", 0.55f);
    processor.setLayerFrozen("decoder.conv0.weight", true);
    processor.setLayerBridgeSource("decoder.conv0.weight", "decoder.conv1.weight");
    processor.setLayerBridgeDepth("decoder.conv0.weight", 0.42f);

    juce::MemoryBlock savedBlock;
    processor.getStateInformation(savedBlock);
    require(savedBlock.getSize() > 0, "saved preset MemoryBlock must be non-empty");

    // Simulate trailing NUL padding produced by some hosts when persisting string blobs
    juce::MemoryBlock paddedBlock(savedBlock);
    const char zeros[2]{'\0', '\0'};
    paddedBlock.append(zeros, sizeof(zeros));

    NetBenderAudioProcessor restored;
    restored.prepareToPlay(k_sample_rate, k_block_size);
    restored.setStateInformation(paddedBlock.getData(), static_cast<int>(paddedBlock.getSize()));

    require(restored.getStateRevision() == 1, "state revision must increment on load");
    require(std::abs(restored.getDryWet() - 0.37f) < 1e-4f, "dryWet must round-trip");
    require(restored.getTriggerMode() == NetBenderAudioProcessor::TriggerMode::Momentary,
            "triggerMode must round-trip");
    require(std::abs(restored.getEnvelopeAttack() - 0.45f) < 1e-4f,
            "envelopeAttack must round-trip");
    require(std::abs(restored.getEnvelopeRelease() - 0.77f) < 1e-4f,
            "envelopeRelease must round-trip");
    require(restored.isLatentEnabled(), "latentEnabled must round-trip");
    require(restored.getLatentMode() == NetBenderAudioProcessor::LatentMode::Quantize,
            "latentMode must round-trip");
    require(std::abs(restored.getLatentX() - (-0.6f)) < 1e-3f, "latentX must round-trip");
    require(std::abs(restored.getLatentY() - 0.4f) < 1e-3f, "latentY must round-trip");
    require(std::abs(restored.getLatentDepth() - 1.25f) < 1e-3f, "latentDepth must round-trip");
    require(std::abs(restored.getLatentSlew() - 0.65f) < 1e-4f, "latentSlew must round-trip");
    require(restored.getActiveLayerName() == "decoder.conv0.weight",
            "activeLayerName must round-trip");

    const auto restoredLayer = restored.getLayerState("decoder.conv0.weight");
    require(std::abs(restoredLayer.scale - 1.85f) < 1e-4f, "layer scale must round-trip");
    require(std::abs(restoredLayer.offset - (-0.12f)) < 1e-4f, "layer offset must round-trip");
    require(std::abs(restoredLayer.heat - 0.31f) < 1e-4f, "layer heat must round-trip");
    require(std::abs(restoredLayer.memory - 0.55f) < 1e-4f, "layer memory must round-trip");
    require(restoredLayer.frozen, "layer frozen flag must round-trip");
    require(restoredLayer.bridgeSourceLayer == "decoder.conv1.weight",
            "layer bridgeSource must round-trip");
    require(std::abs(restoredLayer.bridgeDepth - 0.42f) < 1e-4f,
            "layer bridgeDepth must round-trip");
    restored.releaseResources();

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

        // Give StreamingEngine worker time to run concurrently and exercise lock/ringbuffer handoff
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
                     "models/darbouka_xnnpack.pte, to exercise StreamingEngine + resampling).\n";
    }

    std::cout << "\nAll RTSan checks completed.\n";
    return 0;
}
