#include "MockBackend.h"
#include "bending/BendRandom.h"
#include "bending/BendingManager.h"
#include "bending/LatentTerrainEngine.h"
#include "dsp/rtsan_utils.h"
#include "engine/NetBenderSession.h"

#if defined(TORCH_API) || defined(TORCH_VERSION) || defined(C10_MACROS_MACROS_H_) ||               \
    defined(AT_DISPATCH_ALL_TYPES) || defined(ORT_API_VERSION) ||                                  \
    defined(ONNXRUNTIME_CXX_API_H) || defined(JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED) ||             \
    defined(JUCE_CORE_H_INCLUDED)
#error "Public NetBender bending/session headers leaked Torch, ONNX Runtime, or JUCE headers"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " #cond << "\n";           \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

template <typename Pred>
auto wait_until(Pred&& pred, int timeout_ms = 2000) -> bool {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(250));
    }
    return true;
}

auto approx(float a, float b, float tol = 1e-4f) -> bool {
    return std::abs(a - b) <= tol;
}

auto exercise_bending_and_session_rt(BendingManager& bending,
                                     netbender::bending::Xoshiro128Plus& rng,
                                     NetBenderSession& session, const float* const* f_in,
                                     float* const* f_out, const double* const* d_in,
                                     double* const* d_out, int n_small,
                                     int n_oversized) NETBENDER_NONBLOCKING -> void {
    bending.update_audio_envelope(0.5f);
    bending.trigger_short_circuit(true);
    bending.trigger_short_circuit(false);
    bending.request_audio_thread_reset_all();
    (void)rng.next_bipolar();

    session.set_dry_wet(0.75f);
    session.set_bending_automation(1.2f, 0.05f, 0.0f, 0.8f, true);
    session.set_latent_automation(0.25f, -0.5f, 0.6f);
    session.trigger_short_circuit(true);
    session.trigger_short_circuit(false);

    session.process(f_in, 2, f_out, 2, n_small);
    session.process(f_in, 2, f_out, 2, n_oversized);
    session.process(d_in, 2, d_out, 2, n_small);
}

//==============================================================================
auto test_xoshiro_prng() -> void {
    using netbender::bending::Xoshiro128Plus;

    Xoshiro128Plus a(12345ULL);
    Xoshiro128Plus b(12345ULL);
    Xoshiro128Plus c(99999ULL);

    bool any_diff_seed = false;
    double sum_bipolar = 0.0;
    constexpr int k_samples = 100000;
    for (int i = 0; i < k_samples; ++i) {
        const uint32_t ua = a.next_u32();
        const uint32_t ub = b.next_u32();
        const uint32_t uc = c.next_u32();
        CHECK(ua == ub);
        if (ua != uc) {
            any_diff_seed = true;
        }
        const float f = a.next_float();
        const float bip = b.next_bipolar();
        CHECK(f >= 0.0f && f < 1.0f);
        CHECK(bip >= -1.0f && bip < 1.0f);
        sum_bipolar += bip;
    }
    CHECK(any_diff_seed);
    CHECK(std::abs(sum_bipolar / k_samples) < 0.02);

    // Zero seed must still produce non-degenerate output
    Xoshiro128Plus z(0ULL);
    uint32_t non_zero = 0;
    for (int i = 0; i < 16; ++i) {
        non_zero |= z.next_u32();
    }
    CHECK(non_zero != 0U);

    std::cout << "  [1/8] Xoshiro128Plus PRNG determinism & distribution: PASS\n";
}

//==============================================================================
auto test_layer_classification() -> void {
    CHECK(BendingManager::classify_layer("decoder.norm1.weight") == LayerCategory::Norm);
    CHECK(BendingManager::classify_layer("ENCODER.WEIGHT_G") == LayerCategory::Norm);
    CHECK(BendingManager::classify_layer("post.gamma") == LayerCategory::Norm);

    CHECK(BendingManager::classify_layer("decoder.layer.bias") == LayerCategory::Bias);
    CHECK(BendingManager::classify_layer("decoder.snake.alpha") == LayerCategory::Activation);
    CHECK(BendingManager::classify_layer("act.beta") == LayerCategory::Activation);
    CHECK(BendingManager::classify_layer("transformer.self_attn.out") == LayerCategory::Attention);
    CHECK(BendingManager::classify_layer("qkv_layer") == LayerCategory::Attention);
    CHECK(BendingManager::classify_layer("encoder.lstm.weight_ih") == LayerCategory::Recurrent);
    CHECK(BendingManager::classify_layer("GRU.hh") == LayerCategory::Recurrent);
    CHECK(BendingManager::classify_layer("decoder.rnn.cell") == LayerCategory::Recurrent);
    CHECK(BendingManager::classify_layer("encoder.conv1.weight") == LayerCategory::Conv);
    CHECK(BendingManager::classify_layer("linear_head") == LayerCategory::Linear);
    CHECK(BendingManager::classify_layer("dense_0") == LayerCategory::Linear);
    CHECK(BendingManager::classify_layer("fc1.weight") == LayerCategory::Linear);
    CHECK(BendingManager::classify_layer("proj.out") == LayerCategory::Linear);
    CHECK(BendingManager::classify_layer("latent.prior.sigma") == LayerCategory::Other);

    std::cout << "  [2/8] BendingManager::classify_layer all 8 categories: PASS\n";
}

//==============================================================================
auto test_static_bending_bridge_and_reset() -> void {
    MockBackend backend;
    CHECK(backend.load("virtual/bending.mock", 48000.0) == 0);

    BendingManager bm;
    CHECK(!bm.has_active_bending());

    // Continuous scale + offset on layer0.weight ({1.0, 0.5, -0.5, 0.25})
    bm.set_layer_scale("layer0.weight", 2.0f);
    bm.set_layer_offset("layer0.weight", -0.5f);
    CHECK(bm.has_active_bending());

    bm.apply_bending(&backend, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
    const auto live = bm.get_layer_live_weights("layer0.weight");
    const auto applied = backend.get_layer_weights("layer0.weight");
    CHECK(live.size() == 4);
    CHECK(applied.size() == 4);
    // expected: w0 * 2 - 0.5 -> {1.5, 0.5, -1.5, 0.0}
    CHECK(approx(applied[0], 1.5f));
    CHECK(approx(applied[1], 0.5f));
    CHECK(approx(applied[2], -1.5f));
    CHECK(approx(applied[3], 0.0f));
    for (size_t i = 0; i < 4; ++i) {
        CHECK(approx(live[i], applied[i]));
    }

    // Custom drawn weights + cross-size trace bridge with layer1.bias ({0.1, -0.1}, len 2)
    bm.set_layer_scale("layer0.weight", 1.0f);
    bm.set_layer_offset("layer0.weight", 0.0f);
    bm.set_layer_drawn_weights("layer0.weight", {0.2f, 0.4f, 0.6f, 0.8f});
    bm.set_layer_bridge_source("layer0.weight", "layer1.bias");
    bm.set_layer_bridge_depth("layer0.weight", 0.5f);
    bm.apply_bending(&backend, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
    const auto bridged = backend.get_layer_weights("layer0.weight");
    // Expected: 0.5 * drawn[i] + 0.5 * src[i % 2] -> {0.15, 0.15, 0.35, 0.35}
    CHECK(approx(bridged[0], 0.15f));
    CHECK(approx(bridged[1], 0.15f));
    CHECK(approx(bridged[2], 0.35f));
    CHECK(approx(bridged[3], 0.35f));

    // Per-layer reset
    bm.clear_layer_bending("layer0.weight");
    bm.apply_bending(&backend, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
    const auto orig0 = backend.get_original_layer_weights("layer0.weight");
    const auto after_clear0 = backend.get_layer_weights("layer0.weight");
    for (size_t i = 0; i < orig0.size(); ++i) {
        CHECK(approx(after_clear0[i], orig0[i]));
    }

    // Audio-thread atomic reset all
    bm.set_layer_scale("layer1.bias", 3.0f);
    bm.apply_bending(&backend, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
    CHECK(approx(backend.get_layer_weights("layer1.bias")[0], 0.3f));

    bm.request_audio_thread_reset_all();
    bm.apply_bending(&backend, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
    CHECK(!bm.has_active_bending());
    CHECK(approx(backend.get_layer_weights("layer1.bias")[0], 0.1f));

    std::cout << "  [3/8] Static bending, drawn weights, trace bridge & resets: PASS\n";
}

//==============================================================================
auto test_drift_modes_and_freeze() -> void {
    auto run_sequence = [](DriftMode mode, float memory) -> std::vector<std::vector<float>> {
        MockBackend b;
        CHECK(b.load("virtual/drift.mock", 48000.0) == 0);
        BendingManager bm;
        bm.set_random_seed(424242ULL);
        bm.set_layer_drift_mode("layer0.weight", mode);
        bm.set_layer_heat("layer0.weight", 0.4f);
        bm.set_layer_memory("layer0.weight", memory);

        std::vector<std::vector<float>> trace;
        for (int step = 0; step < 32; ++step) {
            bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
            trace.push_back(b.get_layer_weights("layer0.weight"));
        }
        return trace;
    };

    for (DriftMode mode : {DriftMode::Noise, DriftMode::ThermalOU, DriftMode::RandomWalk}) {
        const auto t1 = run_sequence(mode, 0.8f);
        const auto t2 = run_sequence(mode, 0.8f);
        CHECK(t1 == t2);  // Exact deterministic repeatability under fixed seed
    }

    // ThermalOU: high memory smooths step-to-step variance vs low memory
    auto mean_step_delta = [](const std::vector<std::vector<float>>& t) -> float {
        float total = 0.0f;
        for (size_t s = 1; s < t.size(); ++s) {
            for (size_t i = 0; i < t[s].size(); ++i) {
                total += std::abs(t[s][i] - t[s - 1][i]);
            }
        }
        return total / static_cast<float>((t.size() - 1) * t[0].size());
    };
    const auto ou_low_mem = run_sequence(DriftMode::ThermalOU, 0.1f);
    const auto ou_high_mem = run_sequence(DriftMode::ThermalOU, 0.95f);
    CHECK(mean_step_delta(ou_high_mem) < mean_step_delta(ou_low_mem));

    // RandomWalk bounds over 500 steps and freeze lock
    {
        MockBackend b;
        CHECK(b.load("virtual/rw.mock", 48000.0) == 0);
        BendingManager bm;
        bm.set_random_seed(777ULL);
        bm.set_layer_drift_mode("layer0.weight", DriftMode::RandomWalk);
        bm.set_layer_heat("layer0.weight", 0.9f);
        const auto orig = b.get_original_layer_weights("layer0.weight");

        for (int s = 0; s < 500; ++s) {
            bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
            const auto st = bm.get_layer_state("layer0.weight");
            for (float d : st.driftOffsets) {
                CHECK(d >= -1.0f && d <= 1.0f);
            }
        }

        // Freeze locks weights completely across subsequent apply_bending calls
        bm.set_layer_frozen("layer0.weight", true);
        bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        const auto frozen_w = b.get_layer_weights("layer0.weight");
        for (int s = 0; s < 10; ++s) {
            bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
            CHECK(b.get_layer_weights("layer0.weight") == frozen_w);
        }
        CHECK(frozen_w != orig);
    }

    std::cout << "  [4/8] Drift modes (Noise/ThermalOU/RandomWalk) & freeze: PASS\n";
}

//==============================================================================
auto test_trigger_modes_and_envelope() -> void {
    // 1. Momentary mode: attack ramp, param_short_active, release decay back to baseline W0
    {
        MockBackend b;
        CHECK(b.load("virtual/mom.mock", 48000.0) == 0);
        const auto orig = b.get_original_layer_weights("layer0.weight");

        BendingManager bm;
        bm.set_trigger_mode(TriggerMode::Momentary);
        bm.set_envelope_attack(0.5f);
        bm.set_envelope_release(0.5f);
        bm.set_layer_scale("layer0.weight", 2.0f);

        // Inactive: stays at baseline
        bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        CHECK(approx(bm.get_momentary_envelope(), 0.0f));
        CHECK(b.get_layer_weights("layer0.weight") == orig);

        // Step 1 active -> env = 0.5 -> crossfade halfway between W0 and 2*W0 = 1.5 * W0
        bm.trigger_short_circuit(true);
        bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        CHECK(approx(bm.get_momentary_envelope(), 0.5f));
        const auto half_w = b.get_layer_weights("layer0.weight");
        for (size_t i = 0; i < orig.size(); ++i) {
            CHECK(approx(half_w[i], orig[i] * 1.5f));
        }

        // Step 2 active via param_short_active while manual trigger false -> env = 1.0
        bm.trigger_short_circuit(false);
        bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, true);
        CHECK(approx(bm.get_momentary_envelope(), 1.0f));
        const auto full_w = b.get_layer_weights("layer0.weight");
        for (size_t i = 0; i < orig.size(); ++i) {
            CHECK(approx(full_w[i], orig[i] * 2.0f));
        }

        // Release until envelope decays <= 0.001f -> restores exact W0
        for (int i = 0; i < 20; ++i) {
            bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        }
        CHECK(bm.get_momentary_envelope() <= 0.001f);
        CHECK(b.get_layer_weights("layer0.weight") == orig);
    }

    // 2. Transient mode: triggered by sidechain peak >= threshold
    {
        MockBackend b;
        CHECK(b.load("virtual/trans.mock", 48000.0) == 0);
        const auto orig = b.get_original_layer_weights("layer0.weight");

        BendingManager bm;
        bm.set_trigger_mode(TriggerMode::Transient);
        bm.set_transient_threshold(0.3f);
        bm.set_envelope_attack(1.0f);
        bm.set_envelope_release(0.5f);
        bm.set_layer_scale("layer0.weight", 3.0f);

        bm.update_audio_envelope(0.1f);
        bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        CHECK(approx(bm.get_momentary_envelope(), 0.0f));

        bm.update_audio_envelope(0.6f);
        bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        CHECK(approx(bm.get_momentary_envelope(), 1.0f));
        CHECK(approx(b.get_layer_weights("layer0.weight")[0], orig[0] * 3.0f));

        // Let sidechain envelope decay naturally below threshold
        for (int i = 0; i < 40; ++i) {
            bm.update_audio_envelope(0.0f);
            bm.apply_bending(&b, "", 1.0f, 0.0f, 0.0f, -1.0f, false);
        }
        CHECK(bm.get_audio_envelope() < 0.3f);
        CHECK(bm.get_momentary_envelope() <= 0.001f);
        CHECK(b.get_layer_weights("layer0.weight") == orig);
    }

    std::cout << "  [5/8] Trigger modes (Momentary/Transient) & envelopes: PASS\n";
}

//==============================================================================
auto test_latent_terrain_engine() -> void {
    // 1. FourierOrbit determinism, coordinate dependence, zero-depth identity, odd/even ch & len
    {
        LatentTerrainEngine lte;
        lte.set_mode(LatentMode::FourierOrbit);
        lte.set_coords(0.6f, -0.4f);
        lte.set_depth(0.8f);

        std::vector<float> buf1(5 * 4, 1.0f);
        std::vector<float> buf2(5 * 4, 1.0f);
        IBackend::LatentBuffer lb1{buf1.data(), 5, 4};
        IBackend::LatentBuffer lb2{buf2.data(), 5, 4};
        lte.apply_hook(lb1);
        lte.apply_hook(lb2);
        CHECK(buf1 == buf2);

        bool changed = false;
        for (float v : buf1) {
            if (!approx(v, 1.0f, 1e-5f)) {
                changed = true;
            }
        }
        CHECK(changed);

        // Overrides take precedence over stored coordinates/depth
        std::vector<float> buf_override(5 * 4, 1.0f);
        IBackend::LatentBuffer lb_over{buf_override.data(), 5, 4};
        lte.apply_hook(lb_over, -0.5f, 0.9f, 1.2f);
        CHECK(buf_override != buf1);

        // Depth == 0 -> identity
        std::vector<float> buf_zero(1 * 3, 2.5f);
        IBackend::LatentBuffer lb_zero{buf_zero.data(), 1, 3};
        lte.apply_hook(lb_zero, 0.5f, 0.5f, 0.0f);
        for (float v : buf_zero) {
            CHECK(approx(v, 2.5f, 1e-6f));
        }

        // Different channel count re-sizes RFF matrix cleanly
        std::vector<float> buf16(16 * 2, 0.0f);
        IBackend::LatentBuffer lb16{buf16.data(), 16, 2};
        lte.apply_hook(lb16);
    }

    // 2. LatentSlew seeding, exponential smoothing & reset()
    {
        LatentTerrainEngine lte;
        lte.set_mode(LatentMode::LatentSlew);
        lte.set_slew(0.75f);

        std::vector<float> step1(4, 1.0f);
        IBackend::LatentBuffer lb1{step1.data(), 2, 2};
        lte.apply_hook(lb1);
        for (float v : step1) {
            CHECK(approx(v, 1.0f));  // First frame seeds history unchanged
        }

        std::vector<float> step2(4, 0.0f);
        IBackend::LatentBuffer lb2{step2.data(), 2, 2};
        lte.apply_hook(lb2);
        for (float v : step2) {
            CHECK(approx(v, 0.75f));  // 0.25 * 0 + 0.75 * 1.0 = 0.75
        }

        lte.reset();
        std::vector<float> step3(4, 2.0f);
        IBackend::LatentBuffer lb3{step3.data(), 2, 2};
        lte.apply_hook(lb3);
        for (float v : step3) {
            CHECK(approx(v, 2.0f));  // Reset clears prior history
        }
    }

    // 3. Quantize
    {
        LatentTerrainEngine lte;
        lte.set_mode(LatentMode::Quantize);
        lte.set_depth(1.0f);  // q_step = 0.5
        std::vector<float> buf{0.12f, 0.37f, -0.62f, 1.49f};
        IBackend::LatentBuffer lb{buf.data(), 4, 1};
        lte.apply_hook(lb);
        CHECK(approx(buf[0], 0.0f));
        CHECK(approx(buf[1], 0.5f));
        CHECK(approx(buf[2], -0.5f));
        CHECK(approx(buf[3], 1.5f));
    }

    std::cout << "  [6/8] LatentTerrainEngine (FourierOrbit/LatentSlew/Quantize): PASS\n";
}

//==============================================================================
auto test_session_streaming_and_json_persistence() -> void {
    // Part A: Sync streaming with NetBenderSession + bending/latent automation + fuse reset
    {
        SessionConfig cfg;
        cfg.use_worker_thread = false;
        NetBenderSession session(cfg);

        int last_latency = -1;
        session.set_latency_callback([&](int lat) { last_latency = lat; });

        MockBackend* mock = nullptr;
        session.set_backend_creator([&](const std::string&) {
            auto b = std::make_unique<MockBackend>();
            b->higher_ratio = 64;
            b->preferred_buffer_size = 64;
            b->gain = 1.0f;
            mock = b.get();
            return b;
        });

        session.prepare(48000.0, 64, 2);
        CHECK(session.load_model("virtual/session_sync.mock"));
        CHECK(session.is_model_loaded());
        CHECK(session.get_model_path() == "virtual/session_sync.mock");
        CHECK(session.get_method() == "forward");
        CHECK(last_latency >= 64);
        CHECK(mock != nullptr);

        // Invalid non-power-of-two buffer size is rejected
        CHECK(!session.set_buffer_size(100));
        CHECK(session.get_buffer_size() == 64);
        CHECK(session.set_buffer_size(128));
        CHECK(session.get_buffer_size() == 128);
        CHECK(session.set_buffer_size(64));

        // Active layer automation mutates MockBackend weights during process()
        session.set_active_layer("layer0.weight");
        session.set_bending_automation(2.0f, 0.25f, 0.0f, 0.8f, false);
        session.latent().set_enabled(true);
        session.latent().set_mode(LatentMode::FourierOrbit);
        session.set_latent_automation(0.75f, -0.25f, 1.0f);

        std::array<float, 64> in_l{}, in_r{}, out_l{}, out_r{};
        std::fill(in_l.begin(), in_l.end(), 0.25f);
        std::fill(in_r.begin(), in_r.end(), -0.25f);
        const float* in_ptrs[2]{in_l.data(), in_r.data()};
        float* out_ptrs[2]{out_l.data(), out_r.data()};

        session.process(in_ptrs, 2, out_ptrs, 2, 64);
        session.process(in_ptrs, 2, out_ptrs, 2, 64);
        CHECK(mock->forward_count >= 1);
        CHECK(mock->last_latent_hook_called);
        CHECK(approx(mock->get_layer_weights("layer0.weight")[0], 2.25f));

        // Double precision process() path
        std::array<double, 64> d_in_l{}, d_in_r{}, d_out_l{}, d_out_r{};
        std::fill(d_in_l.begin(), d_in_l.end(), 0.25);
        std::fill(d_in_r.begin(), d_in_r.end(), -0.25);
        const double* d_in_ptrs[2]{d_in_l.data(), d_in_r.data()};
        double* d_out_ptrs[2]{d_out_l.data(), d_out_r.data()};
        session.process(d_in_ptrs, 2, d_out_ptrs, 2, 64);

        // SafetySentry fuse trip + session reset_fuse()
        session.safety_sentry().trip_fuse();
        CHECK(session.safety_sentry().is_fuse_blown());
        session.reset_fuse();
        CHECK(!session.safety_sentry().is_fuse_blown());
        CHECK(!session.bending().has_active_bending());
        CHECK(approx(mock->get_layer_weights("layer0.weight")[0], 1.0f));
    }

    // Part B: JSON state save / load round-trip + error handling + stale-layer clearing
    {
        SessionConfig cfg;
        cfg.use_worker_thread = false;
        NetBenderSession src(cfg);
        src.set_backend_creator([](const std::string&) {
            auto b = std::make_unique<MockBackend>();
            b->higher_ratio = 64;
            b->preferred_buffer_size = 64;
            return b;
        });
        src.prepare(44100.0, 64, 2);
        CHECK(src.load_model("virtual/preset_test.mock"));
        CHECK(src.set_method("autoencode"));
        CHECK(src.set_buffer_size(256));
        src.set_use_gpu(true);
        src.set_dry_wet(0.42f);
        src.safety_sentry().set_auto_reset(false);
        src.set_active_layer("layer0.weight");

        src.bending().set_trigger_mode(TriggerMode::Transient);
        src.bending().set_envelope_attack(0.6f);
        src.bending().set_envelope_release(0.72f);
        src.bending().set_transient_threshold(0.35f);

        src.latent().set_enabled(true);
        src.latent().set_mode(LatentMode::LatentSlew);
        src.latent().set_coords(-0.35f, 0.7f);
        src.latent().set_depth(1.4f);
        src.latent().set_slew(0.85f);

        LayerBendingState ls;
        ls.scale = 1.75f;
        ls.offset = -0.15f;
        ls.heat = 0.22f;
        ls.memory = 0.65f;
        ls.driftMode = DriftMode::RandomWalk;
        ls.frozen = true;
        ls.bridgeSourceLayer = "layer1.bias";
        ls.bridgeDepth = 0.4f;
        ls.originalWeights = {1.0f, 0.5f, -0.5f, 0.25f};
        ls.drawnWeights = {0.9f, 0.1f, -0.2f, 0.8f};
        src.bending().set_layer_state("layer0.weight", ls);

        const auto full_json = src.to_json(true);
        CHECK(full_json["layers"]["layer0.weight"].contains("drawn_weights"));
        const auto compact_json = src.to_json(false);
        CHECK(!compact_json["layers"]["layer0.weight"].contains("drawn_weights"));

        const std::string saved = src.save_state(true);

        NetBenderSession dst(cfg);
        MockBackend* dst_mock = nullptr;
        int create_count = 0;
        dst.set_backend_creator([&](const std::string&) {
            ++create_count;
            auto b = std::make_unique<MockBackend>();
            b->higher_ratio = 64;
            b->preferred_buffer_size = 64;
            dst_mock = b.get();
            return b;
        });
        dst.prepare(44100.0, 64, 2);

        // Pre-populate stale layer and stale host automation before loading saved preset
        dst.bending().set_layer_scale("stale.layer", 9.0f);
        dst.set_bending_automation(5.0f, 1.0f, 0.5f, 0.1f, true);

        CHECK(dst.load_state(saved, true));
        CHECK(create_count == 1);
        CHECK(dst.is_model_loaded());
        CHECK(dst.get_model_path() == "virtual/preset_test.mock");
        CHECK(dst.get_method() == "autoencode");
        CHECK(dst.get_buffer_size() == 256);
        CHECK(dst.is_using_gpu());
        CHECK(approx(dst.get_dry_wet(), 0.42f));
        CHECK(!dst.safety_sentry().get_auto_reset());
        CHECK(dst.get_active_layer() == "layer0.weight");

        CHECK(dst.bending().get_trigger_mode() == TriggerMode::Transient);
        CHECK(approx(dst.bending().get_envelope_attack(), 0.6f));
        CHECK(approx(dst.bending().get_envelope_release(), 0.72f));
        CHECK(approx(dst.bending().get_transient_threshold(), 0.35f));

        CHECK(dst.latent().is_enabled());
        CHECK(dst.latent().get_mode() == LatentMode::LatentSlew);
        CHECK(approx(dst.latent().get_x(), -0.35f));
        CHECK(approx(dst.latent().get_y(), 0.7f));
        CHECK(approx(dst.latent().get_depth(), 1.4f));
        CHECK(approx(dst.latent().get_slew(), 0.85f));

        const auto all_states = dst.bending().get_all_layer_states();
        CHECK(all_states.count("stale.layer") == 0);
        CHECK(all_states.count("layer0.weight") == 1);
        const auto& r = all_states.at("layer0.weight");
        CHECK(approx(r.scale, 1.75f));
        CHECK(approx(r.offset, -0.15f));
        CHECK(approx(r.heat, 0.22f));
        CHECK(approx(r.memory, 0.65f));
        CHECK(r.driftMode == DriftMode::RandomWalk);
        CHECK(r.frozen);
        CHECK(r.bridgeSourceLayer == "layer1.bias");
        CHECK(approx(r.bridgeDepth, 0.4f));
        CHECK(r.drawnWeights.size() == 4 && approx(r.drawnWeights[1], 0.1f));

        // Same model path & GPU flag: preset switch avoids recreating backend
        auto same_model_preset = nlohmann::json::parse(saved);
        same_model_preset["model"]["method"] = "forward";
        same_model_preset["mix"]["dry_wet"] = 0.85f;
        same_model_preset["layers"] = nlohmann::json{
            {"layer1.bias", {{"scale", 2.5f}, {"offset", 0.05f}}},
        };
        CHECK(dst.from_json(same_model_preset, true));
        CHECK(create_count == 1);
        CHECK(dst.get_method() == "forward");
        CHECK(approx(dst.get_dry_wet(), 0.85f));
        const auto switched_layers = dst.bending().get_all_layer_states();
        CHECK(switched_layers.count("layer0.weight") == 0);
        CHECK(switched_layers.count("layer1.bias") == 1);
        CHECK(approx(switched_layers.at("layer1.bias").scale, 2.5f));

        // Trailing NUL bytes (as padded by some DAW hosts) still parse cleanly
        CHECK(dst.load_state(saved + std::string(4, '\0'), true));
        CHECK(create_count == 1);
        CHECK(dst.get_method() == "autoencode");

        // Changing GPU flag with the same model path reloads backend cleanly
        auto gpu_changed = nlohmann::json::parse(saved);
        gpu_changed["model"]["use_gpu"] = !dst.is_using_gpu();
        CHECK(dst.from_json(gpu_changed, true));
        CHECK(create_count == 2);
        CHECK(!dst.is_using_gpu());

        // Even with stale.layer cleared and restore_model=false, layer table replaces cleanly
        auto minimal = nlohmann::json::parse(saved);
        minimal["layers"] = nlohmann::json::object();
        CHECK(dst.from_json(minimal, false));
        CHECK(dst.bending().get_all_layer_states().empty());
        CHECK(dst_mock != nullptr);

        // Reject invalid payloads and set last error
        CHECK(!dst.load_state("{not valid json"));
        CHECK(!dst.get_last_error().empty());

        nlohmann::json wrong_fmt = minimal;
        wrong_fmt["format"] = "other.plugin";
        CHECK(!dst.from_json(wrong_fmt));

        nlohmann::json future_ver = minimal;
        future_ver["version"] = NetBenderSession::k_state_version + 99;
        CHECK(!dst.from_json(future_ver));
        CHECK(!dst.get_last_error().empty());
    }

    std::cout << "  [7/8] NetBenderSession streaming & JSON persistence: PASS\n";
}

//==============================================================================
auto test_bending_and_session_rtsan() -> void {
    SessionConfig cfg;
    cfg.use_worker_thread = true;
    NetBenderSession session(cfg);

    MockBackend* mock = nullptr;
    session.set_backend_creator([&](const std::string&) {
        auto b = std::make_unique<MockBackend>();
        b->in_channels = 1;
        b->out_channels = 1;
        b->higher_ratio = 256;
        b->preferred_buffer_size = 256;
        mock = b.get();
        return b;
    });

    session.prepare(48000.0, 256, 2);
    CHECK(session.load_model("virtual/rtsan_session.mock"));
    session.set_active_layer("layer0.weight");
    session.bending().set_layer_heat("layer0.weight", 0.25f);
    session.latent().set_enabled(true);

    constexpr int k_small = 256;
    constexpr int k_oversized = 8192;
    std::vector<float> fl(k_oversized, 0.2f), fr(k_oversized, -0.2f);
    std::vector<float> fol(k_oversized, 0.0f), fop(k_oversized, 0.0f);
    std::vector<double> dl(k_small, 0.2), dr(k_small, -0.2);
    std::vector<double> dol(k_small, 0.0), dor(k_small, 0.0);

    const float* f_in[2]{fl.data(), fr.data()};
    float* f_out[2]{fol.data(), fop.data()};
    const double* d_in[2]{dl.data(), dr.data()};
    double* d_out[2]{dol.data(), dor.data()};

    BendingManager rt_bending;
    netbender::bending::Xoshiro128Plus rt_rng(2026ULL);

    exercise_bending_and_session_rt(rt_bending, rt_rng, session, f_in, f_out, d_in, d_out, k_small,
                                    k_oversized);

    // Ensure background worker consumes input and exercises apply_bending + latent hook cleanly
    CHECK(wait_until([&] { return session.engine().get_inference_count() >= 1; }));
    session.release();
    CHECK(mock != nullptr && mock->forward_count >= 1);

    std::cout << "  [8/8] BendingManager & NetBenderSession RTSan safety: PASS (rtsan="
              << (netbender::rtsan::is_enabled() ? "enabled" : "off") << ")\n";
}

}  // namespace

int main() {
    std::cout << "=== NetBender Bending & Session Unit Tests ===\n";

    test_xoshiro_prng();
    test_layer_classification();
    test_static_bending_bridge_and_reset();
    test_drift_modes_and_freeze();
    test_trigger_modes_and_envelope();
    test_latent_terrain_engine();
    test_session_streaming_and_json_persistence();
    test_bending_and_session_rtsan();

    std::cout << "All 8/8 bending & session tests passed.\n";
    return 0;
}
