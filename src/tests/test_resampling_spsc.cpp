#include "ResamplingRingBuffer.h"
#include "rtsan_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

auto exercise_spsc_rt(Resampler& resampler, ResamplingRingBuffer& single_rb,
                      ResamplingRingBufferBank& bank, const float* in_f, const double* in_d,
                      float* out_f, double* out_d, int n_small,
                      int n_oversized) NETBENDER_NONBLOCKING -> void {
    const ResampleResult r_out_f =
        resampler.process_for_output(48000.0 / 44100.0, in_f, n_small, out_f, n_small / 2);
    const ResampleResult r_out_d =
        resampler.process_for_output(48000.0 / 44100.0, in_d, n_small, out_f, n_small / 2);
    const ResampleResult r_in_f =
        resampler.process_for_input(48000.0 / 44100.0, in_f, n_small, out_f, n_small);
    (void)r_out_f;
    (void)r_out_d;
    (void)r_in_f;

    (void)single_rb.required_src_samples_for_pull(n_small);
    (void)single_rb.push(in_f, n_oversized);
    if (single_rb.can_pull(n_oversized)) {
        (void)single_rb.pull(out_f, n_oversized);
    }

    (void)bank.push_channel(0, in_f, n_small);
    (void)bank.push_channel(1, in_d, n_small);
    (void)bank.push_channel(2, in_f, n_small);
    float* bank_f_ptrs[3]{out_f, nullptr, out_f};
    if (bank.can_pull(n_small / 2)) {
        (void)bank.pull(bank_f_ptrs, 3, n_small / 2);
    }
    double* bank_d_ptrs[3]{out_d, out_d, nullptr};
    if (bank.can_pull(n_small / 4)) {
        (void)bank.pull(bank_d_ptrs, 3, n_small / 4);
    }
}

auto test_resample_modes_and_bypass() -> void {
    ResamplingRingBuffer rb;
    rb.init(4096, 48000.0, 48000.0, ResampleMode::Auto, ResamplerQuality::CatmullRom);
    CHECK(!rb.is_resampling_active());
    CHECK(rb.get_speed_ratio() == 1.0);
    CHECK(rb.get_src_sample_rate() == 48000.0);
    CHECK(rb.get_dst_sample_rate() == 48000.0);
    CHECK(rb.get_capacity_src() >= 4096);

    // Near-equal within 1e-3 Hz stays bypassed in Auto mode
    rb.set_sample_rates(48000.0, 48000.0005);
    CHECK(!rb.is_resampling_active());
    CHECK(rb.get_speed_ratio() == 1.0);

    // Mismatched activates Auto resampling
    rb.set_sample_rates(48000.0, 44100.0);
    CHECK(rb.is_resampling_active());
    CHECK(std::abs(rb.get_speed_ratio() - (48000.0 / 44100.0)) < 1e-9);

    // Disabled bypasses even at 48k -> 24k with bit-exact pass-through
    rb.clear();
    rb.set_sample_rates(48000.0, 24000.0);
    rb.set_resample_mode(ResampleMode::Disabled);
    CHECK(!rb.is_resampling_active());
    CHECK(rb.get_resample_mode() == ResampleMode::Disabled);

    std::array<float, 64> src{};
    std::array<float, 64> dst{};
    for (int i = 0; i < 64; ++i) {
        src[static_cast<size_t>(i)] = static_cast<float>(i) * 0.01f - 0.3f;
    }
    CHECK(rb.push(src.data(), 64) == 64);
    CHECK(rb.can_pull(64));
    CHECK(rb.pull(dst.data(), 64) == 64);
    for (int i = 0; i < 64; ++i) {
        CHECK(dst[static_cast<size_t>(i)] == src[static_cast<size_t>(i)]);
    }

    // ForceEnabled activates resampling even at equal 48k -> 48k
    rb.set_sample_rates(48000.0, 48000.0);
    rb.set_resample_mode(ResampleMode::ForceEnabled);
    CHECK(rb.is_resampling_active());
    CHECK(rb.push(src.data(), 64) == 64);
    CHECK(rb.get_available_src() == 64);
    rb.clear();
    CHECK(rb.get_available_src() == 0);
    rb.reset_counters();
    CHECK(rb.get_overrun_count() == 0);
    CHECK(rb.get_underrun_count() == 0);

    // Move constructor and move assignment
    ResamplingRingBuffer rb_moved(std::move(rb));
    CHECK(rb_moved.is_resampling_active());
    CHECK(rb_moved.get_capacity_src() >= 4096);
    CHECK(!rb.is_resampling_active());

    ResamplingRingBuffer rb_assigned;
    rb_assigned = std::move(rb_moved);
    CHECK(rb_assigned.is_resampling_active());
    CHECK(rb_assigned.get_capacity_src() >= 4096);
    CHECK(!rb_moved.is_resampling_active());
}

auto test_roundtrip_host_model_host() -> void {
    struct SrPair {
        double host_sr;
        double model_sr;
    };
    const std::array<SrPair, 4> pairs{
        SrPair{48000.0, 24000.0},
        SrPair{48000.0, 44100.0},
        SrPair{96000.0, 44100.0},
        SrPair{44100.0, 48000.0},
    };

    constexpr int k_host_block = 256;
    constexpr int k_model_block = 2048;
    constexpr float k_amp = 0.5f;
    constexpr double k_freq = 440.0;

    for (const auto& pair : pairs) {
        ResamplingRingBuffer in_ring;
        ResamplingRingBuffer out_ring;
        in_ring.init(65536, pair.host_sr, pair.model_sr, ResampleMode::Auto,
                     ResamplerQuality::CatmullRom);
        out_ring.init(65536, pair.model_sr, pair.host_sr, ResampleMode::Auto,
                      ResamplerQuality::CatmullRom);

        // Prefill sync latency so output ring pulls cleanly from t = 0
        CHECK(out_ring.push_zeros(k_model_block) == k_model_block);

        const int host_latency =
            static_cast<int>(std::round(k_model_block * (pair.host_sr / pair.model_sr)));
        const int total_host_samples = host_latency + 16384;
        const int num_blocks = (total_host_samples + k_host_block - 1) / k_host_block;

        std::vector<float> host_in(k_host_block, 0.0f);
        std::vector<float> model_buf(k_model_block, 0.0f);
        std::vector<float> host_out(k_host_block, 0.0f);
        std::vector<float> all_out;
        all_out.reserve(static_cast<size_t>(num_blocks * k_host_block));

        int sample_idx = 0;
        for (int b = 0; b < num_blocks; ++b) {
            for (int i = 0; i < k_host_block; ++i) {
                const double t = static_cast<double>(sample_idx + i) / pair.host_sr;
                host_in[static_cast<size_t>(i)] =
                    static_cast<float>(k_amp * std::sin(2.0 * k_pi * k_freq * t));
            }
            CHECK(in_ring.push(host_in.data(), k_host_block) == k_host_block);

            while (in_ring.can_pull(k_model_block)) {
                CHECK(in_ring.pull(model_buf.data(), k_model_block) == k_model_block);
                CHECK(out_ring.push(model_buf.data(), k_model_block) == k_model_block);
            }

            CHECK(out_ring.can_pull(k_host_block));
            CHECK(out_ring.pull(host_out.data(), k_host_block) == k_host_block);
            for (float s : host_out) {
                CHECK(std::isfinite(s));
            }
            all_out.insert(all_out.end(), host_out.begin(), host_out.end());
            sample_idx += k_host_block;
        }

        CHECK(in_ring.get_overrun_count() == 0);
        CHECK(in_ring.get_underrun_count() == 0);
        CHECK(out_ring.get_overrun_count() == 0);
        CHECK(out_ring.get_underrun_count() == 0);

        const double exact_delay =
            static_cast<double>(k_model_block) * (pair.host_sr / pair.model_sr) + 2.0 +
            2.0 * (pair.host_sr / pair.model_sr);
        const int eval_start = static_cast<int>(std::ceil(exact_delay)) + 128;
        const int eval_end = static_cast<int>(all_out.size()) - 128;
        CHECK(eval_end > eval_start + 4096);

        double sum_sq = 0.0;
        double err_sq = 0.0;
        double ref_sq = 0.0;
        for (int k = eval_start; k < eval_end; ++k) {
            const double t = (static_cast<double>(k) - exact_delay) / pair.host_sr;
            const double ideal = k_amp * std::sin(2.0 * k_pi * k_freq * t);
            const double actual = static_cast<double>(all_out[static_cast<size_t>(k)]);
            const double diff = actual - ideal;
            sum_sq += actual * actual;
            ref_sq += ideal * ideal;
            err_sq += diff * diff;
        }

        const int eval_count = eval_end - eval_start;
        const double actual_rms = std::sqrt(sum_sq / static_cast<double>(eval_count));
        const double expected_rms = static_cast<double>(k_amp) / std::sqrt(2.0);
        CHECK(std::abs(actual_rms - expected_rms) < expected_rms * 0.02);

        const double snr_db = 10.0 * std::log10(ref_sq / std::max(1e-18, err_sq));
        CHECK(snr_db > 55.0);
    }
}

auto test_oversized_16384_blocks_small_scratch() -> void {
    struct SrPair {
        double src_sr;
        double dst_sr;
    };
    const std::array<SrPair, 2> pairs{
        SrPair{48000.0, 44100.0},
        SrPair{44100.0, 96000.0},
    };

    constexpr int k_src_len = 32768;
    constexpr int k_pull_total = 16384;
    std::vector<float> src(k_src_len, 0.0f);
    for (int i = 0; i < k_src_len; ++i) {
        src[static_cast<size_t>(i)] =
            static_cast<float>(0.6 * std::sin(2.0 * k_pi * 330.0 * i / 48000.0));
    }

    for (const auto& pair : pairs) {
        ResamplingRingBuffer rb_big;
        ResamplingRingBuffer rb_ref;
        rb_big.init(65536, pair.src_sr, pair.dst_sr, ResampleMode::Auto,
                    ResamplerQuality::CatmullRom, 512);
        rb_ref.init(65536, pair.src_sr, pair.dst_sr, ResampleMode::Auto,
                    ResamplerQuality::CatmullRom, 512);

        CHECK(rb_big.push(src.data(), k_src_len) == k_src_len);
        CHECK(rb_ref.push(src.data(), k_src_len) == k_src_len);

        CHECK(rb_big.can_pull(k_pull_total));
        std::vector<float> out_big(k_pull_total, 0.0f);
        CHECK(rb_big.pull(out_big.data(), k_pull_total) == k_pull_total);

        std::vector<float> out_ref(k_pull_total, 0.0f);
        for (int off = 0; off < k_pull_total; off += 256) {
            CHECK(rb_ref.can_pull(256));
            CHECK(rb_ref.pull(out_ref.data() + off, 256) == 256);
        }

        CHECK(rb_big.get_underrun_count() == 0);
        CHECK(rb_ref.get_underrun_count() == 0);

        float max_diff = 0.0f;
        for (int i = 0; i < k_pull_total; ++i) {
            max_diff = std::max(max_diff, std::abs(out_big[static_cast<size_t>(i)] -
                                                   out_ref[static_cast<size_t>(i)]));
        }
        CHECK(max_diff < 1e-6f);
    }
}

auto test_multichannel_quadrature_phase_lock() -> void {
    struct SrPair {
        double src_sr;
        double dst_sr;
    };
    const std::array<SrPair, 2> pairs{
        SrPair{48000.0, 44100.0},
        SrPair{96000.0, 24000.0},
    };

    for (const auto& pair : pairs) {
        ResamplingRingBufferBank bank;
        bank.init(4, 32768, pair.src_sr, pair.dst_sr, ResampleMode::Auto,
                  ResamplerQuality::CatmullRom);
        CHECK(bank.get_num_channels() == 4);
        CHECK(bank.is_resampling_active());

        constexpr int k_src_len = 8192;
        std::vector<float> s_pos(k_src_len), c_pos(k_src_len), s_neg(k_src_len), c_neg(k_src_len);
        for (int i = 0; i < k_src_len; ++i) {
            const double phase = 2.0 * k_pi * 440.0 * static_cast<double>(i) / pair.src_sr;
            s_pos[static_cast<size_t>(i)] = static_cast<float>(std::sin(phase));
            c_pos[static_cast<size_t>(i)] = static_cast<float>(std::cos(phase));
            s_neg[static_cast<size_t>(i)] = -s_pos[static_cast<size_t>(i)];
            c_neg[static_cast<size_t>(i)] = -c_pos[static_cast<size_t>(i)];
        }
        CHECK(bank.push_channel(0, s_pos.data(), k_src_len) == k_src_len);
        CHECK(bank.push_channel(1, c_pos.data(), k_src_len) == k_src_len);
        CHECK(bank.push_channel(2, s_neg.data(), k_src_len) == k_src_len);
        CHECK(bank.push_channel(3, c_neg.data(), k_src_len) == k_src_len);

        constexpr int k_blk = 128;
        std::vector<float> d0(k_blk), d1(k_blk), d2(k_blk), d3(k_blk);
        float* ptrs[4]{d0.data(), d1.data(), d2.data(), d3.data()};

        int total_pulled = 0;
        for (int iter = 0; iter < 12; ++iter) {
            CHECK(bank.can_pull(k_blk));
            CHECK(bank.pull(ptrs, 4, k_blk) == k_blk);
            for (int i = 0; i < k_blk; ++i) {
                const int global_out = total_pulled + i;
                CHECK(std::abs(d0[static_cast<size_t>(i)] + d2[static_cast<size_t>(i)]) < 1e-5f);
                CHECK(std::abs(d1[static_cast<size_t>(i)] + d3[static_cast<size_t>(i)]) < 1e-5f);
                if (global_out >= 16) {
                    const float mag_sq = d0[static_cast<size_t>(i)] * d0[static_cast<size_t>(i)] +
                                         d1[static_cast<size_t>(i)] * d1[static_cast<size_t>(i)];
                    CHECK(mag_sq > 0.97f && mag_sq < 1.03f);
                }
            }
            total_pulled += k_blk;
        }

        // nullptr destination on ch2 while keeping ch0/ch1/ch3 in strict lockstep
        float* ptrs_with_null[4]{d0.data(), d1.data(), nullptr, d3.data()};
        CHECK(bank.can_pull(k_blk));
        CHECK(bank.pull(ptrs_with_null, 4, k_blk) == k_blk);
        for (int i = 0; i < k_blk; ++i) {
            CHECK(std::abs(d1[static_cast<size_t>(i)] + d3[static_cast<size_t>(i)]) < 1e-5f);
            const float mag_sq = d0[static_cast<size_t>(i)] * d0[static_cast<size_t>(i)] +
                                 d1[static_cast<size_t>(i)] * d1[static_cast<size_t>(i)];
            CHECK(mag_sq > 0.97f && mag_sq < 1.03f);
        }

        // Push extra data onto ch0 only, verify common_avail locks all channels identically
        const int avail_before = bank.get_available_src();
        std::array<float, 256> extra{};
        extra.fill(0.25f);
        CHECK(bank.push_channel(0, extra.data(), 256) == 256);
        CHECK(bank.get_available_src() == avail_before);
        CHECK(bank.can_pull(k_blk));
        CHECK(bank.pull(ptrs, 4, k_blk) == k_blk);
        for (int i = 0; i < k_blk; ++i) {
            CHECK(std::abs(d1[static_cast<size_t>(i)] + d3[static_cast<size_t>(i)]) < 1e-5f);
        }
        CHECK(bank.get_underrun_count() == 0);
        CHECK(bank.get_overrun_count() == 0);
    }
}

auto test_double_io_path() -> void {
    struct SrPair {
        double src_sr;
        double dst_sr;
    };
    const std::array<SrPair, 2> pairs{
        SrPair{48000.0, 44100.0},
        SrPair{44100.0, 48000.0},
    };

    for (const auto& pair : pairs) {
        ResamplingRingBufferBank bank_f;
        ResamplingRingBufferBank bank_d;
        bank_f.init(2, 16384, pair.src_sr, pair.dst_sr, ResampleMode::Auto);
        bank_d.init(2, 16384, pair.src_sr, pair.dst_sr, ResampleMode::Auto);

        constexpr int k_src = 4096;
        std::vector<float> in_f0(k_src), in_f1(k_src);
        std::vector<double> in_d0(k_src), in_d1(k_src);
        for (int i = 0; i < k_src; ++i) {
            const double ph = 2.0 * k_pi * 512.0 * static_cast<double>(i) / pair.src_sr;
            in_d0[static_cast<size_t>(i)] = 0.7 * std::sin(ph);
            in_d1[static_cast<size_t>(i)] = 0.7 * std::cos(ph);
            in_f0[static_cast<size_t>(i)] = static_cast<float>(in_d0[static_cast<size_t>(i)]);
            in_f1[static_cast<size_t>(i)] = static_cast<float>(in_d1[static_cast<size_t>(i)]);
        }

        CHECK(bank_f.push_channel(0, in_f0.data(), k_src) == k_src);
        CHECK(bank_f.push_channel(1, in_f1.data(), k_src) == k_src);
        CHECK(bank_d.push_channel(0, in_d0.data(), k_src) == k_src);
        CHECK(bank_d.push_channel(1, in_d1.data(), k_src) == k_src);

        constexpr int k_pull = 2048;
        std::vector<float> out_f0(k_pull), out_f1(k_pull);
        std::vector<double> out_d0(k_pull), out_d1(k_pull);
        float* f_ptrs[2]{out_f0.data(), out_f1.data()};
        double* d_ptrs[2]{out_d0.data(), out_d1.data()};

        CHECK(bank_f.pull(f_ptrs, 2, k_pull) == k_pull);
        CHECK(bank_d.pull(d_ptrs, 2, k_pull) == k_pull);

        for (int i = 0; i < k_pull; ++i) {
            CHECK(std::abs(static_cast<double>(out_f0[static_cast<size_t>(i)]) -
                           out_d0[static_cast<size_t>(i)]) < 1e-5);
            CHECK(std::abs(static_cast<double>(out_f1[static_cast<size_t>(i)]) -
                           out_d1[static_cast<size_t>(i)]) < 1e-5);
        }
    }
}

auto test_two_thread_spsc_resampling_stress() -> void {
    struct SrPair {
        double src_sr;
        double dst_sr;
    };
    const std::array<SrPair, 2> pairs{
        SrPair{48000.0, 44100.0},
        SrPair{48000.0, 24000.0},
    };

    constexpr int k_total_src = 500000;
    constexpr int k_pull_block = 512;
    constexpr double k_freq = 220.0;

    for (const auto& pair : pairs) {
        ResamplingRingBuffer rb;
        rb.init(32768, pair.src_sr, pair.dst_sr, ResampleMode::Auto, ResamplerQuality::CatmullRom);

        std::atomic<bool> producer_done{false};
        std::thread producer([&]() {
            int produced = 0;
            bool flip = false;
            std::array<float, 257> chunk{};
            while (produced < k_total_src) {
                const int blk = std::min(flip ? 127 : 257, k_total_src - produced);
                flip = !flip;
                for (int i = 0; i < blk; ++i) {
                    const double t = static_cast<double>(produced + i) / pair.src_sr;
                    chunk[static_cast<size_t>(i)] =
                        static_cast<float>(0.8 * std::sin(2.0 * k_pi * k_freq * t));
                }
                while (rb.get_free_space_src() < blk) {
                    std::this_thread::yield();
                }
                CHECK(rb.push(chunk.data(), blk) == blk);
                produced += blk;
            }
            producer_done.store(true, std::memory_order_release);
        });

        int total_pulled = 0;
        float prev_sample = 0.0f;
        bool have_prev = false;
        float max_step = 0.0f;
        std::vector<float> pull_buf(k_pull_block, 0.0f);

        while (!producer_done.load(std::memory_order_acquire) || rb.can_pull(k_pull_block)) {
            if (!rb.can_pull(k_pull_block)) {
                std::this_thread::yield();
                continue;
            }
            CHECK(rb.pull(pull_buf.data(), k_pull_block) == k_pull_block);
            for (int i = 0; i < k_pull_block; ++i) {
                const float s = pull_buf[static_cast<size_t>(i)];
                CHECK(std::isfinite(s));
                if (have_prev && total_pulled > 32) {
                    max_step = std::max(max_step, std::abs(s - prev_sample));
                }
                prev_sample = s;
                have_prev = true;
                ++total_pulled;
            }
        }

        producer.join();

        const double expected_pulled =
            static_cast<double>(k_total_src) * (pair.dst_sr / pair.src_sr);
        CHECK(total_pulled > static_cast<int>(expected_pulled * 0.95));
        CHECK(rb.get_overrun_count() == 0);
        CHECK(rb.get_underrun_count() == 0);

        const double max_expected_step = 0.8 * (2.0 * k_pi * k_freq / pair.dst_sr) * 1.25;
        CHECK(max_step < max_expected_step);
    }
}

auto test_rtsan_nonblocking_paths() -> void {
    Resampler resampler(ResamplerQuality::CatmullRom);
    ResamplingRingBuffer single_rb;
    single_rb.init(32768, 48000.0, 44100.0, ResampleMode::Auto, ResamplerQuality::CatmullRom, 512);

    ResamplingRingBufferBank bank;
    bank.init(3, 8192, 48000.0, 44100.0, ResampleMode::Auto, ResamplerQuality::CatmullRom, 256);

    constexpr int k_small = 256;
    constexpr int k_oversized = 4096;
    std::vector<float> in_f(k_oversized * 2, 0.2f);
    std::vector<double> in_d(k_small, 0.2);
    std::vector<float> out_f(k_oversized, 0.0f);
    std::vector<double> out_d(k_small, 0.0);

    // Pre-populate slightly so can_pull executes pull paths cleanly under RTSan
    (void)single_rb.push(in_f.data(), k_oversized * 2);
    (void)bank.push_channel(0, in_f.data(), k_small * 2);
    (void)bank.push_channel(1, in_f.data(), k_small * 2);
    (void)bank.push_channel(2, in_f.data(), k_small * 2);

    exercise_spsc_rt(resampler, single_rb, bank, in_f.data(), in_d.data(), out_f.data(),
                     out_d.data(), k_small, k_oversized);
}

}  // namespace

auto main() -> int {
    std::cout << "Running test_resampling_spsc (RTSan: "
              << (netbender::rtsan::is_enabled() ? "ON" : "OFF") << ")...\n";

    test_resample_modes_and_bypass();
    std::cout << "  [1/7] ResampleMode Auto/ForceEnabled/Disabled & move semantics: OK\n";

    test_roundtrip_host_model_host();
    std::cout << "  [2/7] Round-trip host<->model<->host fractional SNR: OK\n";

    test_oversized_16384_blocks_small_scratch();
    std::cout << "  [3/7] Oversized 16384-sample chunked pull over 512-sample scratch: OK\n";

    test_multichannel_quadrature_phase_lock();
    std::cout << "  [4/7] 4-channel quadrature phase lock & nullptr lockstep: OK\n";

    test_double_io_path();
    std::cout << "  [5/7] Double I/O parity on ResamplingRingBufferBank: OK\n";

    test_two_thread_spsc_resampling_stress();
    std::cout << "  [6/7] 500k-sample 2-thread SPSC + fractional resampling stress: OK\n";

    test_rtsan_nonblocking_paths();
    std::cout << "  [7/7] NETBENDER_NONBLOCKING RTSan path: OK\n";

    std::cout << "All test_resampling_spsc tests passed!\n";
    return 0;
}
