#include "MockBackend.h"
#include "dsp/rtsan_utils.h"
#include "engine/StreamingEngine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
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

constexpr double k_pi = 3.14159265358979323846;

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

auto exercise_streaming_rt(StreamingEngine& engine, const float* const* f_in, float* const* f_out,
                           const double* const* d_in, double* const* d_out, int n_small,
                           int n_oversized) NETBENDER_NONBLOCKING -> void {
    engine.set_dry_wet(0.85f);
    engine.process(f_in, 2, f_out, 2, n_small);
    engine.process(f_in, 2, f_out, 2, n_oversized);
    engine.process(d_in, 2, d_out, 2, n_small);
}

auto test_sync_mode_resampling_off_and_on() -> void {
    struct Case {
        double host_sr;
        double model_sr;
        ResampleMode mode;
        bool expect_resampling;
    };
    const std::array<Case, 5> cases{
        Case{48000.0, 48000.0, ResampleMode::Auto, false},
        Case{48000.0, 24000.0, ResampleMode::Disabled, false},
        Case{48000.0, 24000.0, ResampleMode::Auto, true},
        Case{48000.0, 44100.0, ResampleMode::Auto, true},
        Case{96000.0, 44100.0, ResampleMode::Auto, true},
    };

    constexpr int k_model_buf = 256;
    constexpr int k_host_block = 64;
    constexpr float k_gain = 0.75f;
    constexpr float k_amp = 0.6f;

    for (const auto& c : cases) {
        MockBackend backend;
        backend.in_channels = 1;
        backend.out_channels = 1;
        backend.higher_ratio = k_model_buf;
        backend.preferred_buffer_size = k_model_buf;
        backend.gain = k_gain;
        CHECK(backend.load("virtual/sync.mock", c.model_sr) == 0);

        StreamingEngine engine;
        StreamingConfig cfg;
        cfg.method = "forward";
        cfg.dispatch = InferenceDispatch::PluginMode;
        cfg.model_in_channels = 1;
        cfg.model_out_channels = 1;
        cfg.n_batches = 1;
        cfg.model_buffer_size = k_model_buf;
        cfg.host_sample_rate = c.host_sr;
        cfg.model_sample_rate = c.model_sr;
        cfg.max_host_block_size = 256;
        cfg.max_host_channels = 1;
        cfg.resample_mode = c.mode;
        cfg.use_worker_thread = false;
        cfg.auto_start_worker = false;
        cfg.enable_dry_delay = false;
        cfg.enable_safety_sentry = false;
        engine.configure(&backend, cfg);

        CHECK(engine.is_configured());
        CHECK(!engine.is_worker_running());
        CHECK(engine.is_resampling_active() == c.expect_resampling);

        const int expected_latency =
            c.expect_resampling
                ? static_cast<int>(std::round(k_model_buf * (c.host_sr / c.model_sr)))
                : k_model_buf;
        CHECK(engine.get_latency_samples() == expected_latency);

        const int total_samples = expected_latency + 4096;
        const int n_blocks = (total_samples + k_host_block - 1) / k_host_block;

        std::array<float, k_host_block> in_buf{};
        std::array<float, k_host_block> out_buf{};
        const float* in_ptrs[1]{in_buf.data()};
        float* out_ptrs[1]{out_buf.data()};

        double sum_sq = 0.0;
        int steady_count = 0;
        int sample_cursor = 0;

        for (int b = 0; b < n_blocks; ++b) {
            for (int i = 0; i < k_host_block; ++i) {
                const double t = static_cast<double>(sample_cursor + i) / c.host_sr;
                in_buf[static_cast<size_t>(i)] =
                    static_cast<float>(k_amp * std::sin(2.0 * k_pi * 100.0 * t));
            }
            engine.process(in_ptrs, 1, out_ptrs, 1, k_host_block);

            for (int i = 0; i < k_host_block; ++i) {
                const float y = out_buf[static_cast<size_t>(i)];
                CHECK(std::isfinite(y));
                const int global_k = sample_cursor + i;
                if (global_k >= expected_latency + 256) {
                    sum_sq += static_cast<double>(y) * static_cast<double>(y);
                    ++steady_count;
                }
            }
            sample_cursor += k_host_block;
        }

        CHECK(engine.get_inference_count() > 0);
        CHECK(engine.get_in_overrun_count() == 0);
        CHECK(engine.get_out_underrun_count() == 0);

        const double actual_rms = std::sqrt(sum_sq / static_cast<double>(steady_count));
        const double expected_rms = (static_cast<double>(k_amp) * k_gain) / std::sqrt(2.0);
        CHECK(std::abs(actual_rms - expected_rms) < expected_rms * 0.03);
    }
}

auto test_async_worker_mode_resampling_off_and_on() -> void {
    struct Case {
        double host_sr;
        double model_sr;
    };
    const std::array<Case, 3> cases{
        Case{48000.0, 48000.0},
        Case{48000.0, 44100.0},
        Case{48000.0, 24000.0},
    };

    constexpr int k_model_buf = 512;
    constexpr int k_host_block = 256;
    constexpr float k_gain = 0.8f;

    for (const auto& c : cases) {
        MockBackend backend;
        backend.in_channels = 2;
        backend.out_channels = 2;
        backend.higher_ratio = k_model_buf;
        backend.preferred_buffer_size = k_model_buf;
        backend.gain = k_gain;
        CHECK(backend.load("virtual/async.mock", c.model_sr) == 0);

        StreamingEngine engine;
        StreamingConfig cfg;
        cfg.method = "forward";
        cfg.dispatch = InferenceDispatch::PluginMode;
        cfg.model_in_channels = 2;
        cfg.model_out_channels = 2;
        cfg.n_batches = 1;
        cfg.model_buffer_size = k_model_buf;
        cfg.host_sample_rate = c.host_sr;
        cfg.model_sample_rate = c.model_sr;
        cfg.max_host_block_size = 512;
        cfg.max_host_channels = 2;
        cfg.resample_mode = ResampleMode::Auto;
        cfg.use_worker_thread = true;
        cfg.auto_start_worker = true;
        cfg.enable_dry_delay = false;
        cfg.enable_safety_sentry = false;
        engine.configure(&backend, cfg);

        CHECK(engine.is_configured());
        CHECK(engine.is_worker_running());

        const double in_ratio = engine.is_resampling_active() ? (c.host_sr / c.model_sr) : 1.0;
        const int latency = engine.get_latency_samples();
        const int total_samples =
            latency + static_cast<int>(std::ceil(k_model_buf * 5.0 * in_ratio));
        const int n_blocks = (total_samples + k_host_block - 1) / k_host_block;

        std::array<float, k_host_block> in_l{}, in_r{}, out_l{}, out_r{};
        const float* in_ptrs[2]{in_l.data(), in_r.data()};
        float* out_ptrs[2]{out_l.data(), out_r.data()};

        double sum_sq_l = 0.0;
        double sum_sq_r = 0.0;
        int steady_count = 0;
        int pushed = 0;
        uint64_t waited_inferences = 0;

        for (int b = 0; b < n_blocks; ++b) {
            for (int i = 0; i < k_host_block; ++i) {
                const double t = static_cast<double>(pushed + i) / c.host_sr;
                in_l[static_cast<size_t>(i)] =
                    static_cast<float>(0.5 * std::sin(2.0 * k_pi * 220.0 * t));
                in_r[static_cast<size_t>(i)] =
                    static_cast<float>(0.5 * std::cos(2.0 * k_pi * 220.0 * t));
            }
            engine.process(in_ptrs, 2, out_ptrs, 2, k_host_block);

            for (int i = 0; i < k_host_block; ++i) {
                CHECK(std::isfinite(out_l[static_cast<size_t>(i)]));
                CHECK(std::isfinite(out_r[static_cast<size_t>(i)]));
                if (pushed + i >= latency + 256) {
                    sum_sq_l += static_cast<double>(out_l[static_cast<size_t>(i)]) *
                                static_cast<double>(out_l[static_cast<size_t>(i)]);
                    sum_sq_r += static_cast<double>(out_r[static_cast<size_t>(i)]) *
                                static_cast<double>(out_r[static_cast<size_t>(i)]);
                    ++steady_count;
                }
            }

            pushed += k_host_block;
            const int next_trigger_at =
                static_cast<int>(std::ceil(static_cast<double>(waited_inferences + 1) *
                                           k_model_buf * in_ratio)) +
                16;
            if (pushed >= next_trigger_at) {
                const uint64_t target = waited_inferences + 1;
                CHECK(wait_until([&]() { return engine.get_inference_count() >= target; }));
                waited_inferences = target;
            }
        }

        engine.detach_backend();
        CHECK(!engine.is_worker_running());
        CHECK(engine.get_inference_count() >= 4);
        CHECK(engine.get_in_overrun_count() == 0);
        CHECK(engine.get_out_underrun_count() == 0);

        const double expected_rms = (0.5 * k_gain) / std::sqrt(2.0);
        const double rms_l = std::sqrt(sum_sq_l / static_cast<double>(steady_count));
        const double rms_r = std::sqrt(sum_sq_r / static_cast<double>(steady_count));
        CHECK(std::abs(rms_l - expected_rms) < expected_rms * 0.04);
        CHECK(std::abs(rms_r - expected_rms) < expected_rms * 0.04);
    }
}

auto test_dry_wet_alignment_under_fractional_resampling() -> void {
    struct SrPair {
        double host_sr;
        double model_sr;
    };
    const std::array<SrPair, 2> pairs{
        SrPair{48000.0, 44100.0},
        SrPair{48000.0, 24000.0},
    };

    constexpr int k_model_buf = 256;
    constexpr int k_host_block = 64;
    constexpr float k_amp = 0.4f;
    constexpr double k_freq = 50.0;

    for (const auto& pair : pairs) {
        MockBackend backend;
        backend.in_channels = 1;
        backend.out_channels = 1;
        backend.higher_ratio = k_model_buf;
        backend.preferred_buffer_size = k_model_buf;
        backend.gain = -1.0f;  // Polarity inversion tests phase/latency alignment via cancellation
        CHECK(backend.load("virtual/drywet.mock", pair.model_sr) == 0);

        StreamingConfig cfg;
        cfg.method = "forward";
        cfg.dispatch = InferenceDispatch::PluginMode;
        cfg.model_in_channels = 1;
        cfg.model_out_channels = 1;
        cfg.n_batches = 1;
        cfg.model_buffer_size = k_model_buf;
        cfg.host_sample_rate = pair.host_sr;
        cfg.model_sample_rate = pair.model_sr;
        cfg.max_host_block_size = 256;
        cfg.max_host_channels = 1;
        cfg.resample_mode = ResampleMode::Auto;
        cfg.use_worker_thread = false;
        cfg.auto_start_worker = false;
        cfg.enable_dry_delay = true;
        cfg.enable_safety_sentry = false;

        auto run_with_mix = [&](float dry_wet, int& out_latency) -> std::vector<float> {
            StreamingEngine engine;
            engine.configure(&backend, cfg);
            engine.set_dry_wet(dry_wet);
            out_latency = engine.get_latency_samples();

            const int total = out_latency + 4096;
            const int n_blocks = (total + k_host_block - 1) / k_host_block;
            std::vector<float> all_in(static_cast<size_t>(n_blocks * k_host_block), 0.0f);
            std::vector<float> all_out(static_cast<size_t>(n_blocks * k_host_block), 0.0f);
            for (size_t i = 0; i < all_in.size(); ++i) {
                const double t = static_cast<double>(i) / pair.host_sr;
                all_in[i] = static_cast<float>(k_amp * std::sin(2.0 * k_pi * k_freq * t));
            }
            for (int b = 0; b < n_blocks; ++b) {
                const float* in_ptr = all_in.data() + b * k_host_block;
                float* out_ptr = all_out.data() + b * k_host_block;
                engine.process(&in_ptr, 1, &out_ptr, 1, k_host_block);
            }
            return all_out;
        };

        int latency = 0;
        const auto out_dry = run_with_mix(0.0f, latency);
        const auto out_wet = run_with_mix(1.0f, latency);
        const auto out_mix = run_with_mix(0.5f, latency);

        // Verify dry path matches exact host_latency integer delay
        float max_dry_err = 0.0f;
        double sum_dry_sq = 0.0;
        double sum_wet_sq = 0.0;
        double sum_mix_sq = 0.0;
        int count = 0;

        for (size_t k = static_cast<size_t>(latency); k < out_dry.size(); ++k) {
            const double t = static_cast<double>(k - static_cast<size_t>(latency)) / pair.host_sr;
            const float expected_dry =
                static_cast<float>(k_amp * std::sin(2.0 * k_pi * k_freq * t));
            max_dry_err = std::max(max_dry_err, std::abs(out_dry[k] - expected_dry));

            if (k >= static_cast<size_t>(latency + 256)) {
                sum_dry_sq += static_cast<double>(out_dry[k]) * static_cast<double>(out_dry[k]);
                sum_wet_sq += static_cast<double>(out_wet[k]) * static_cast<double>(out_wet[k]);
                sum_mix_sq += static_cast<double>(out_mix[k]) * static_cast<double>(out_mix[k]);
                ++count;
            }
        }
        CHECK(max_dry_err < 1e-6f);

        const double rms_dry = std::sqrt(sum_dry_sq / static_cast<double>(count));
        const double rms_wet = std::sqrt(sum_wet_sq / static_cast<double>(count));
        const double rms_mix = std::sqrt(sum_mix_sq / static_cast<double>(count));
        CHECK(std::abs(rms_wet - rms_dry) < rms_dry * 0.02);
        CHECK((rms_mix / rms_dry) < 0.1);
    }
}

auto test_double_batched_raw_method() -> void {
    constexpr int k_batches = 3;
    constexpr int k_in_ch = 2;
    constexpr int k_out_ch = 2;
    constexpr int k_model_buf = 256;
    constexpr int k_host_block = 64;

    const std::array<std::array<float, k_out_ch>, k_batches> scales{{
        {{0.2f, 0.4f}},
        {{0.6f, 0.8f}},
        {{1.1f, 1.4f}},
    }};

    MockBackend backend;
    backend.in_channels = k_in_ch;
    backend.out_channels = k_out_ch;
    backend.higher_ratio = k_model_buf;
    backend.preferred_buffer_size = k_model_buf;
    backend.custom_perform = [&](std::vector<float*>& in_buf, std::vector<float*>& out_buf,
                                 int n_batches, int n_out_channels, int n_vec) {
        CHECK(n_batches == k_batches);
        CHECK(n_out_channels == k_out_ch);
        for (int b = 0; b < n_batches; ++b) {
            for (int c = 0; c < n_out_channels; ++c) {
                const float* src = in_buf[static_cast<size_t>(c * n_batches + b)];
                float* dst = out_buf[static_cast<size_t>(b * n_out_channels + c)];
                CHECK(src != nullptr);
                CHECK(dst != nullptr);
                for (int i = 0; i < n_vec; ++i) {
                    dst[i] = src[i] * scales[static_cast<size_t>(b)][static_cast<size_t>(c)];
                }
            }
        }
    };
    CHECK(backend.load("virtual/batch_raw.mock", 44100.0) == 0);

    StreamingEngine engine;
    StreamingConfig cfg;
    cfg.method = "custom_batch_op";
    cfg.dispatch = InferenceDispatch::RawMethod;
    cfg.model_in_channels = k_in_ch;
    cfg.model_out_channels = k_out_ch;
    cfg.n_batches = k_batches;
    cfg.model_buffer_size = k_model_buf;
    cfg.host_sample_rate = 48000.0;
    cfg.model_sample_rate = 44100.0;
    cfg.max_host_block_size = 256;
    cfg.max_host_channels = 6;
    cfg.resample_mode = ResampleMode::Auto;
    cfg.use_worker_thread = false;
    cfg.auto_start_worker = false;
    cfg.enable_dry_delay = false;
    cfg.enable_safety_sentry = false;
    engine.configure(&backend, cfg);

    std::array<std::vector<double>, 6> in_ch;
    std::array<std::vector<double>, 6> out_ch;
    std::array<const double*, 6> in_ptrs{};
    std::array<double*, 6> out_ptrs{};
    for (int s = 0; s < 6; ++s) {
        in_ch[static_cast<size_t>(s)].assign(k_host_block, 0.1 * static_cast<double>(s + 1));
        out_ch[static_cast<size_t>(s)].assign(k_host_block, 0.0);
        in_ptrs[static_cast<size_t>(s)] = in_ch[static_cast<size_t>(s)].data();
        out_ptrs[static_cast<size_t>(s)] = out_ch[static_cast<size_t>(s)].data();
    }

    const int latency = engine.get_latency_samples();
    const int n_blocks = (latency + 1024 + k_host_block - 1) / k_host_block;
    for (int iter = 0; iter < n_blocks; ++iter) {
        engine.process(in_ptrs.data(), 6, out_ptrs.data(), 6, k_host_block);
    }

    CHECK(backend.perform_count > 0);
    CHECK(backend.last_method == "custom_batch_op");
    CHECK(backend.last_n_batches == k_batches);
    CHECK(backend.last_n_out_channels == k_out_ch);

    for (int b = 0; b < k_batches; ++b) {
        for (int c = 0; c < k_out_ch; ++c) {
            const int in_idx = c * k_batches + b;
            const int out_idx = b * k_out_ch + c;
            const double expected =
                0.1 * static_cast<double>(in_idx + 1) *
                static_cast<double>(scales[static_cast<size_t>(b)][static_cast<size_t>(c)]);
            for (int i = 0; i < k_host_block; ++i) {
                CHECK(std::abs(out_ch[static_cast<size_t>(out_idx)][static_cast<size_t>(i)] -
                               expected) < 1e-4);
            }
        }
    }
}

auto test_generative_modes_hooks_and_error_callback() -> void {
    // 1. "prior" mode (0 inputs -> 2 outputs)
    {
        MockBackend backend;
        backend.in_channels = 0;
        backend.out_channels = 2;
        backend.higher_ratio = 64;
        backend.preferred_buffer_size = 128;
        backend.gain = 1.0f;
        CHECK(backend.load("virtual/prior.mock", 44100.0) == 0);

        StreamingEngine engine;
        StreamingConfig cfg;
        cfg.method = "prior";
        cfg.dispatch = InferenceDispatch::PluginMode;
        cfg.model_in_channels = 0;
        cfg.model_out_channels = 2;
        cfg.n_batches = 1;
        cfg.model_buffer_size = 128;
        cfg.host_sample_rate = 48000.0;
        cfg.model_sample_rate = 44100.0;
        cfg.max_host_block_size = 128;
        cfg.max_host_channels = 2;
        cfg.use_worker_thread = false;
        cfg.auto_start_worker = false;
        engine.configure(&backend, cfg);

        CHECK(engine.get_latency_samples() == 0);

        std::array<float, 64> out_l{}, out_r{};
        float* out_ptrs[2]{out_l.data(), out_r.data()};
        double sum_sq = 0.0;
        for (int b = 0; b < 8; ++b) {
            engine.process(nullptr, 0, out_ptrs, 2, 64);
            for (int i = 0; i < 64; ++i) {
                CHECK(std::isfinite(out_l[static_cast<size_t>(i)]));
                CHECK(std::isfinite(out_r[static_cast<size_t>(i)]));
                sum_sq += static_cast<double>(out_l[static_cast<size_t>(i)]) *
                          static_cast<double>(out_l[static_cast<size_t>(i)]);
            }
        }
        CHECK(backend.prior_count > 0);
        CHECK(engine.get_out_underrun_count() == 0);
        CHECK(std::sqrt(sum_sq / (8.0 * 64.0)) > 0.05);
    }

    // 2. "decode gen" with pre_inference_callback, latent_hook_provider,
    // request_clear_decode_gen_fifo
    {
        MockBackend backend;
        backend.in_channels = 0;
        backend.out_channels = 2;
        backend.higher_ratio = 64;
        backend.preferred_buffer_size = 128;
        backend.gain = 1.0f;
        CHECK(backend.load("virtual/decodegen.mock", 44100.0) == 0);

        StreamingEngine engine;
        int pre_cb_count = 0;
        bool check_cleared_next = false;
        bool saw_cleared_fifo = false;

        engine.set_pre_inference_callback([&](IBackend& b) {
            ++pre_cb_count;
            auto& mb = static_cast<MockBackend&>(b);
            if (check_cleared_next && mb.decode_gen_state_size() == 0) {
                saw_cleared_fifo = true;
            }
        });
        engine.set_latent_hook_provider([]() -> IBackend::LatentHook {
            return [](IBackend::LatentBuffer& buf) {
                for (int64_t i = 0; i < buf.channels * buf.length; ++i) {
                    buf.data[i] *= 2.0f;
                }
            };
        });

        StreamingConfig cfg;
        cfg.method = "decode gen";
        cfg.dispatch = InferenceDispatch::PluginMode;
        cfg.model_in_channels = 0;
        cfg.model_out_channels = 2;
        cfg.n_batches = 1;
        cfg.model_buffer_size = 128;
        cfg.host_sample_rate = 48000.0;
        cfg.model_sample_rate = 44100.0;
        cfg.max_host_block_size = 128;
        cfg.max_host_channels = 2;
        cfg.use_worker_thread = false;
        cfg.auto_start_worker = false;
        engine.configure(&backend, cfg);

        std::array<float, 64> out_l{}, out_r{};
        float* out_ptrs[2]{out_l.data(), out_r.data()};
        for (int b = 0; b < 6; ++b) {
            engine.process(nullptr, 0, out_ptrs, 2, 64);
        }
        CHECK(backend.decode_gen_count >= 2);
        CHECK(pre_cb_count >= 2);
        CHECK(backend.last_latent_hook_called);
        // Expected output: gain(1.0) * latent_scale(2.0) * 0.25 = 0.5f
        for (int i = 0; i < 64; ++i) {
            CHECK(std::abs(out_l[static_cast<size_t>(i)] - 0.5f) < 1e-4f);
            CHECK(std::abs(out_r[static_cast<size_t>(i)] - 0.5f) < 1e-4f);
        }

        CHECK(backend.decode_gen_state_size() > 0);
        check_cleared_next = true;
        engine.request_clear_decode_gen_fifo();
        const int before_count = backend.decode_gen_count;
        for (int b = 0; b < 6 && backend.decode_gen_count == before_count; ++b) {
            engine.process(nullptr, 0, out_ptrs, 2, 64);
        }
        CHECK(backend.decode_gen_count > before_count);
        CHECK(saw_cleared_fifo);
    }

    // 3. Exception handling -> error_callback + zeroed safe output
    {
        MockBackend backend;
        backend.in_channels = 1;
        backend.out_channels = 1;
        backend.higher_ratio = 64;
        backend.preferred_buffer_size = 64;
        backend.custom_perform = [](std::vector<float*>&, std::vector<float*>&, int, int, int) {
            throw std::runtime_error("boom");
        };
        CHECK(backend.load("virtual/err.mock", 48000.0) == 0);

        StreamingEngine engine;
        std::string captured_err;
        engine.set_error_callback([&](const std::string& msg) { captured_err = msg; });

        StreamingConfig cfg;
        cfg.method = "forward";
        cfg.dispatch = InferenceDispatch::PluginMode;
        cfg.model_in_channels = 1;
        cfg.model_out_channels = 1;
        cfg.n_batches = 1;
        cfg.model_buffer_size = 64;
        cfg.host_sample_rate = 48000.0;
        cfg.model_sample_rate = 48000.0;
        cfg.max_host_block_size = 64;
        cfg.max_host_channels = 1;
        cfg.use_worker_thread = false;
        cfg.auto_start_worker = false;
        engine.configure(&backend, cfg);

        std::array<float, 64> in_buf{}, out_buf{};
        in_buf.fill(0.5f);
        out_buf.fill(99.0f);
        const float* in_ptr = in_buf.data();
        float* out_ptr = out_buf.data();
        for (int b = 0; b < 4; ++b) {
            engine.process(&in_ptr, 1, &out_ptr, 1, 64);
        }

        CHECK(captured_err == "boom");
        for (float y : out_buf) {
            CHECK(y == 0.0f);
        }
    }
}

auto test_concurrent_live_reconfiguration() -> void {
    MockBackend backend;
    backend.in_channels = 2;
    backend.out_channels = 2;
    backend.higher_ratio = 256;
    backend.preferred_buffer_size = 256;
    backend.gain = 0.5f;
    CHECK(backend.load("virtual/reconfig.mock", 44100.0) == 0);

    StreamingEngine engine;
    StreamingConfig cfg;
    cfg.method = "forward";
    cfg.dispatch = InferenceDispatch::PluginMode;
    cfg.model_in_channels = 2;
    cfg.model_out_channels = 2;
    cfg.n_batches = 1;
    cfg.model_buffer_size = 256;
    cfg.host_sample_rate = 48000.0;
    cfg.model_sample_rate = 44100.0;
    cfg.max_host_block_size = 512;
    cfg.max_host_channels = 2;
    cfg.resample_mode = ResampleMode::Auto;
    cfg.use_worker_thread = true;
    cfg.auto_start_worker = true;
    cfg.enable_dry_delay = true;
    cfg.enable_safety_sentry = true;
    engine.configure(&backend, cfg);

    std::atomic<bool> keep_running{true};
    std::atomic<bool> saw_nonfinite{false};
    std::atomic<int> audio_blocks{0};

    std::thread audio_thread([&]() {
        constexpr int k_blk = 256;
        std::array<float, k_blk> in_l{}, in_r{}, out_l{}, out_r{};
        in_l.fill(0.3f);
        in_r.fill(-0.3f);
        const float* in_ptrs[2]{in_l.data(), in_r.data()};
        float* out_ptrs[2]{out_l.data(), out_r.data()};
        while (keep_running.load(std::memory_order_acquire)) {
            engine.process(in_ptrs, 2, out_ptrs, 2, k_blk);
            for (int i = 0; i < k_blk; ++i) {
                if (!std::isfinite(out_l[static_cast<size_t>(i)]) ||
                    !std::isfinite(out_r[static_cast<size_t>(i)])) {
                    saw_nonfinite.store(true, std::memory_order_release);
                }
            }
            audio_blocks.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
    });

    for (int round = 0; round < 3; ++round) {
        engine.set_host_sample_rate(44100.0);
        engine.set_host_sample_rate(96000.0);
        engine.set_host_sample_rate(48000.0);
        engine.set_buffer_size(512);
        engine.set_buffer_size(1024);
        engine.set_buffer_size(256);
        engine.set_resample_mode(ResampleMode::Disabled);
        engine.set_resample_mode(ResampleMode::ForceEnabled);
        engine.set_resample_mode(ResampleMode::Auto);
        engine.set_method("prior");
        engine.set_method("forward");
        engine.detach_backend();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        engine.configure(&backend, cfg);
    }

    CHECK(wait_until([&]() { return audio_blocks.load(std::memory_order_relaxed) >= 40; }));
    keep_running.store(false, std::memory_order_release);
    audio_thread.join();

    CHECK(!saw_nonfinite.load(std::memory_order_acquire));

    // Verify steady-state audio resumes cleanly after reconfiguration finishes
    constexpr int k_blk = 256;
    std::array<float, k_blk> in_l{}, in_r{}, out_l{}, out_r{};
    in_l.fill(0.4f);
    in_r.fill(0.4f);
    const float* in_ptrs[2]{in_l.data(), in_r.data()};
    float* out_ptrs[2]{out_l.data(), out_r.data()};

    engine.configure(&backend, cfg);
    const uint64_t base_inf = engine.get_inference_count();
    const double r_in = 48000.0 / 44100.0;
    uint64_t waited = 0;
    int pushed = 0;
    for (int b = 0; b < 8; ++b) {
        engine.process(in_ptrs, 2, out_ptrs, 2, k_blk);
        pushed += k_blk;
        const int next_trigger_at =
            static_cast<int>(std::ceil(static_cast<double>(waited + 1) * 256.0 * r_in)) + 16;
        if (pushed >= next_trigger_at) {
            const uint64_t target = base_inf + waited + 1;
            CHECK(wait_until([&]() { return engine.get_inference_count() >= target; }, 500));
            ++waited;
        }
    }
    engine.detach_backend();
    double tail_sq = 0.0;
    for (int i = 0; i < k_blk; ++i) {
        tail_sq += static_cast<double>(out_l[static_cast<size_t>(i)]) *
                   static_cast<double>(out_l[static_cast<size_t>(i)]);
    }
    CHECK(std::sqrt(tail_sq / static_cast<double>(k_blk)) > 0.05);
}

auto test_safety_sentry_trip_and_auto_recovery() -> void {
    MockBackend backend;
    backend.in_channels = 1;
    backend.out_channels = 1;
    backend.higher_ratio = 128;
    backend.preferred_buffer_size = 128;

    std::atomic<bool> inject_nan{true};
    backend.custom_perform = [&](std::vector<float*>& in_buf, std::vector<float*>& out_buf, int,
                                 int, int n_vec) {
        if (inject_nan.load(std::memory_order_acquire)) {
            std::fill(out_buf[0], out_buf[0] + n_vec, std::numeric_limits<float>::quiet_NaN());
        } else {
            for (int i = 0; i < n_vec; ++i) {
                out_buf[0][i] = in_buf[0][i] * 0.5f;
            }
        }
    };
    CHECK(backend.load("virtual/sentry.mock", 44100.0) == 0);

    StreamingEngine engine;
    std::atomic<int> reset_cb_count{0};
    engine.get_safety_sentry().set_auto_reset(true);
    engine.set_fuse_reset_callback([&]() {
        reset_cb_count.fetch_add(1, std::memory_order_relaxed);
        engine.get_safety_sentry().reset_fuse();
    });

    StreamingConfig cfg;
    cfg.method = "forward";
    cfg.dispatch = InferenceDispatch::PluginMode;
    cfg.model_in_channels = 1;
    cfg.model_out_channels = 1;
    cfg.n_batches = 1;
    cfg.model_buffer_size = 128;
    cfg.host_sample_rate = 48000.0;
    cfg.model_sample_rate = 44100.0;
    cfg.max_host_block_size = 64;
    cfg.max_host_channels = 1;
    cfg.resample_mode = ResampleMode::Auto;
    cfg.use_worker_thread = false;
    cfg.auto_start_worker = false;
    cfg.enable_dry_delay = true;
    cfg.enable_safety_sentry = true;
    engine.configure(&backend, cfg);
    engine.set_dry_wet(1.0f);

    constexpr int k_blk = 64;
    std::array<float, k_blk> in_buf{}, out_buf{};
    const float* in_ptr = in_buf.data();
    float* out_ptr = out_buf.data();
    int sample_cursor = 0;

    auto pump_block = [&]() {
        for (int i = 0; i < k_blk; ++i) {
            const double t = static_cast<double>(sample_cursor + i) / 48000.0;
            in_buf[static_cast<size_t>(i)] =
                static_cast<float>(0.3 * std::sin(2.0 * k_pi * 220.0 * t));
        }
        engine.process(&in_ptr, 1, &out_ptr, 1, k_blk);
        for (float y : out_buf) {
            CHECK(std::isfinite(y));
        }
        sample_cursor += k_blk;
    };

    // Phase A: Pump blocks until blown fuse trips
    for (int b = 0; b < 12; ++b) {
        pump_block();
    }
    CHECK(engine.get_safety_sentry().is_fuse_blown());

    // Phase B: Restore healthy output and pump through cooldown (~42 blocks) -> auto recovery
    inject_nan.store(false, std::memory_order_release);
    double wet_sq = 0.0;
    int wet_count = 0;
    for (int b = 0; b < 60; ++b) {
        pump_block();
        if (b >= 52) {
            for (float y : out_buf) {
                wet_sq += static_cast<double>(y) * static_cast<double>(y);
                ++wet_count;
            }
        }
    }

    CHECK(reset_cb_count.load(std::memory_order_relaxed) >= 1);
    CHECK(!engine.get_safety_sentry().is_fuse_blown());
    CHECK(std::sqrt(wet_sq / static_cast<double>(wet_count)) > 0.05);
}

auto test_rtsan_streaming_engine_process() -> void {
    MockBackend backend;
    backend.in_channels = 2;
    backend.out_channels = 2;
    backend.higher_ratio = 256;
    backend.preferred_buffer_size = 256;
    CHECK(backend.load("virtual/rtsan.mock", 44100.0) == 0);

    StreamingEngine engine;
    StreamingConfig cfg;
    cfg.method = "forward";
    cfg.dispatch = InferenceDispatch::PluginMode;
    cfg.model_in_channels = 2;
    cfg.model_out_channels = 2;
    cfg.n_batches = 1;
    cfg.model_buffer_size = 256;
    cfg.host_sample_rate = 48000.0;
    cfg.model_sample_rate = 44100.0;
    cfg.max_host_block_size = 512;
    cfg.max_host_channels = 2;
    cfg.resample_mode = ResampleMode::Auto;
    cfg.use_worker_thread = true;
    cfg.auto_start_worker = true;
    cfg.enable_dry_delay = true;
    cfg.enable_safety_sentry = true;
    engine.configure(&backend, cfg);

    constexpr int k_small = 256;
    constexpr int k_oversized = 16384;
    std::vector<float> f_in0(k_oversized, 0.25f);
    std::vector<float> f_in1(k_oversized, -0.25f);
    std::vector<float> f_out0(k_oversized, 0.0f);
    std::vector<float> f_out1(k_oversized, 0.0f);
    const float* f_in_ptrs[2]{f_in0.data(), f_in1.data()};
    float* f_out_ptrs[2]{f_out0.data(), f_out1.data()};

    std::vector<double> d_in0(k_small, 0.25);
    std::vector<double> d_in1(k_small, -0.25);
    std::vector<double> d_out0(k_small, 0.0);
    std::vector<double> d_out1(k_small, 0.0);
    const double* d_in_ptrs[2]{d_in0.data(), d_in1.data()};
    double* d_out_ptrs[2]{d_out0.data(), d_out1.data()};

    exercise_streaming_rt(engine, f_in_ptrs, f_out_ptrs, d_in_ptrs, d_out_ptrs, k_small,
                          k_oversized);

    engine.detach_backend();
}

}  // namespace

auto main() -> int {
    std::cout << "Running test_streaming_engine (RTSan: "
              << (netbender::rtsan::is_enabled() ? "ON" : "OFF") << ")...\n";

    test_sync_mode_resampling_off_and_on();
    std::cout << "  [1/8] Sync mode with resampling OFF & ON: OK\n";

    test_async_worker_mode_resampling_off_and_on();
    std::cout << "  [2/8] Async worker mode stereo with resampling OFF & ON: OK\n";

    test_dry_wet_alignment_under_fractional_resampling();
    std::cout << "  [3/8] Dry/wet delay alignment under fractional resampling: OK\n";

    test_double_batched_raw_method();
    std::cout << "  [4/8] double* batched RawMethod routing: OK\n";

    test_generative_modes_hooks_and_error_callback();
    std::cout << "  [5/8] Generative modes, hooks, FIFO reset & error callback: OK\n";

    test_concurrent_live_reconfiguration();
    std::cout << "  [6/8] Concurrent live SR/buffer/method reconfiguration: OK\n";

    test_safety_sentry_trip_and_auto_recovery();
    std::cout << "  [7/8] SafetySentry NaN trip & auto-recovery: OK\n";

    test_rtsan_streaming_engine_process();
    std::cout << "  [8/8] NETBENDER_NONBLOCKING RTSan process() check: OK\n";

    std::cout << "All test_streaming_engine tests passed!\n";
    return 0;
}
