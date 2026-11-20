#include "Resampler.h"

#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
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

enum class SignalKind {
    Dc = 0,
    Ramp = 1,
    Sine440 = 2,
    Chirp = 3,
    Impulse = 4,
    Noise = 5,
};

auto generate_signal(SignalKind kind, int length, double sample_rate = 48000.0)
    -> std::vector<float> {
    std::vector<float> out(static_cast<size_t>(length), 0.0f);
    std::mt19937 rng(1337u);
    std::uniform_real_distribution<float> dist(-0.95f, 0.95f);

    for (int i = 0; i < length; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        switch (kind) {
            case SignalKind::Dc:
                out[static_cast<size_t>(i)] = 0.73f;
                break;
            case SignalKind::Ramp:
                out[static_cast<size_t>(i)] =
                    (length > 1)
                        ? (-1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(length - 1))
                        : 0.0f;
                break;
            case SignalKind::Sine440:
                out[static_cast<size_t>(i)] =
                    static_cast<float>(0.8 * std::sin(2.0 * k_pi * 440.0 * t));
                break;
            case SignalKind::Chirp: {
                const double duration = static_cast<double>(length) / sample_rate;
                const double f0 = 100.0;
                const double f1 = 4000.0;
                const double k = (duration > 0.0) ? ((f1 - f0) / duration) : 0.0;
                const double phase = 2.0 * k_pi * (f0 * t + 0.5 * k * t * t);
                out[static_cast<size_t>(i)] = static_cast<float>(0.75 * std::sin(phase));
                break;
            }
            case SignalKind::Impulse:
                out[static_cast<size_t>(i)] = (i == 64) ? 1.0f : 0.0f;
                break;
            case SignalKind::Noise:
                out[static_cast<size_t>(i)] = dist(rng);
                break;
        }
    }
    return out;
}

template <typename JuceInterp>
auto run_parity_comparison(ResamplerQuality quality, double ratio, const std::vector<float>& src,
                           int total_out, const std::vector<int>& block_pattern) -> void {
    Resampler nb(quality);
    JuceInterp juce_interp;
    nb.reset();
    juce_interp.reset();

    std::vector<float> out_nb(static_cast<size_t>(total_out), 0.0f);
    std::vector<float> out_juce(static_cast<size_t>(total_out), 0.0f);

    int out_pos = 0;
    int in_pos = 0;
    size_t pat_idx = 0;

    while (out_pos < total_out) {
        const int blk =
            std::min(block_pattern[pat_idx % block_pattern.size()], total_out - out_pos);
        ++pat_idx;

        const int avail = static_cast<int>(src.size()) - in_pos;
        const int req = nb.required_input_samples(ratio, blk);
        CHECK(avail >= req);

        const ResampleResult res =
            nb.process_for_output(ratio, src.data() + in_pos, avail, out_nb.data() + out_pos, blk);
        const int juce_used =
            juce_interp.process(ratio, src.data() + in_pos, out_juce.data() + out_pos, blk);

        CHECK(res.output_produced == blk);
        CHECK(res.input_consumed == juce_used);

        in_pos += res.input_consumed;
        out_pos += blk;
    }

    float max_abs_diff = 0.0f;
    double sum_sq_diff = 0.0;
    for (int i = 0; i < total_out; ++i) {
        const float diff =
            std::abs(out_nb[static_cast<size_t>(i)] - out_juce[static_cast<size_t>(i)]);
        max_abs_diff = std::max(max_abs_diff, diff);
        sum_sq_diff += static_cast<double>(diff) * static_cast<double>(diff);
    }
    const double rmse = std::sqrt(sum_sq_diff / static_cast<double>(total_out));
    CHECK(max_abs_diff < 1e-5f);
    CHECK(rmse < 1e-6);
}

auto test_catmull_rom_exact_parity_matrix() -> void {
    const std::array<double, 13> ratios{
        48000.0 / 44100.0,
        44100.0 / 48000.0,
        48000.0 / 24000.0,
        24000.0 / 48000.0,
        44100.0 / 24000.0,
        24000.0 / 44100.0,
        96000.0 / 44100.0,
        44100.0 / 96000.0,
        88200.0 / 48000.0,
        48000.0 / 88200.0,
        1.0,
        std::sqrt(2.0),
        1.0 / std::sqrt(2.0),
    };

    const std::array<SignalKind, 6> signals{
        SignalKind::Dc,    SignalKind::Ramp,    SignalKind::Sine440,
        SignalKind::Chirp, SignalKind::Impulse, SignalKind::Noise,
    };

    const std::vector<std::vector<int>> block_patterns{
        {32}, {64}, {127}, {256}, {512}, {2048}, {16384}, {31, 64, 127, 256, 509, 1024},
    };

    constexpr int k_total_out = 16384;
    constexpr int k_max_in_len = static_cast<int>(k_total_out * 2.25) + 256;

    std::vector<std::vector<float>> generated_signals;
    generated_signals.reserve(signals.size());
    for (SignalKind kind : signals) {
        generated_signals.push_back(generate_signal(kind, k_max_in_len, 48000.0));
    }

    for (double ratio : ratios) {
        for (const auto& src : generated_signals) {
            for (const auto& pattern : block_patterns) {
                run_parity_comparison<juce::CatmullRomInterpolator>(
                    ResamplerQuality::CatmullRom, ratio, src, k_total_out, pattern);
            }
        }
    }
}

auto test_linear_exact_parity() -> void {
    const std::array<double, 5> ratios{
        48000.0 / 44100.0, 44100.0 / 48000.0, 48000.0 / 24000.0, 1.0, std::sqrt(2.0),
    };

    constexpr int k_total_out = 16384;
    constexpr int k_max_in_len = static_cast<int>(k_total_out * 2.1) + 256;
    const auto sine_sig = generate_signal(SignalKind::Sine440, k_max_in_len, 48000.0);
    const auto noise_sig = generate_signal(SignalKind::Noise, k_max_in_len, 48000.0);
    const std::vector<int> mixed_blocks{31, 64, 127, 256, 509, 1024};

    for (double ratio : ratios) {
        run_parity_comparison<juce::LinearInterpolator>(ResamplerQuality::Linear, ratio, sine_sig,
                                                        k_total_out, mixed_blocks);
        run_parity_comparison<juce::LinearInterpolator>(ResamplerQuality::Linear, ratio, noise_sig,
                                                        k_total_out, mixed_blocks);
    }
}

auto test_block_seam_continuity() -> void {
    constexpr int k_total_out = 32768;
    const std::array<double, 3> ratios{
        48000.0 / 44100.0,
        44100.0 / 96000.0,
        std::sqrt(2.0),
    };
    const std::vector<int> irregular_blocks{1, 7, 32, 127, 256, 509, 2048, 16384};

    const int max_in = static_cast<int>(std::ceil(k_total_out * 1.5)) + 256;
    auto sine = generate_signal(SignalKind::Sine440, max_in, 48000.0);
    auto noise = generate_signal(SignalKind::Noise, max_in, 48000.0);
    std::vector<float> src(static_cast<size_t>(max_in));
    for (int i = 0; i < max_in; ++i) {
        src[static_cast<size_t>(i)] =
            0.5f * sine[static_cast<size_t>(i)] + 0.5f * noise[static_cast<size_t>(i)];
    }

    for (double ratio : ratios) {
        Resampler oneshot(ResamplerQuality::CatmullRom);
        std::vector<float> out_oneshot(static_cast<size_t>(k_total_out), 0.0f);
        const ResampleResult r_oneshot = oneshot.process_for_output(
            ratio, src.data(), static_cast<int>(src.size()), out_oneshot.data(), k_total_out);
        CHECK(r_oneshot.output_produced == k_total_out);

        Resampler chunked(ResamplerQuality::CatmullRom);
        std::vector<float> out_chunked(static_cast<size_t>(k_total_out), 0.0f);
        int out_pos = 0;
        int in_pos = 0;
        size_t pat_idx = 0;
        while (out_pos < k_total_out) {
            const int blk = std::min(irregular_blocks[pat_idx % irregular_blocks.size()],
                                     k_total_out - out_pos);
            ++pat_idx;
            const ResampleResult r = chunked.process_for_output(
                ratio, src.data() + in_pos, static_cast<int>(src.size()) - in_pos,
                out_chunked.data() + out_pos, blk);
            CHECK(r.output_produced == blk);
            in_pos += r.input_consumed;
            out_pos += blk;
        }

        CHECK(in_pos == r_oneshot.input_consumed);
        float max_abs_diff = 0.0f;
        for (int i = 0; i < k_total_out; ++i) {
            max_abs_diff = std::max(max_abs_diff, std::abs(out_oneshot[static_cast<size_t>(i)] -
                                                           out_chunked[static_cast<size_t>(i)]));
        }
        CHECK(max_abs_diff == 0.0f);
    }
}

auto test_sine_roundtrip_snr_and_impulse_latency() -> void {
    struct SrPair {
        double host_sr;
        double model_sr;
    };
    const std::array<SrPair, 2> pairs{
        SrPair{48000.0, 44100.0},
        SrPair{48000.0, 24000.0},
    };

    for (const auto& pair : pairs) {
        const double r_in = pair.host_sr / pair.model_sr;
        const double r_out = pair.model_sr / pair.host_sr;

        constexpr int k_host_in = 16384;
        constexpr int k_rt_out = 8192;
        const int num_model =
            static_cast<int>(std::floor(12000.0 * (pair.model_sr / pair.host_sr)));

        const auto host_src = generate_signal(SignalKind::Sine440, k_host_in, pair.host_sr);
        std::vector<float> model_nb(static_cast<size_t>(num_model), 0.0f);
        std::vector<float> model_j(static_cast<size_t>(num_model), 0.0f);
        std::vector<float> rt_nb(static_cast<size_t>(k_rt_out), 0.0f);
        std::vector<float> rt_j(static_cast<size_t>(k_rt_out), 0.0f);

        Resampler nb_in(ResamplerQuality::CatmullRom);
        Resampler nb_out(ResamplerQuality::CatmullRom);
        juce::CatmullRomInterpolator j_in;
        juce::CatmullRomInterpolator j_out;

        const ResampleResult r1 = nb_in.process_for_output(
            r_in, host_src.data(), static_cast<int>(host_src.size()), model_nb.data(), num_model);
        const int juce1 = j_in.process(r_in, host_src.data(), model_j.data(), num_model);
        CHECK(r1.input_consumed == juce1);
        CHECK(r1.output_produced == num_model);

        const ResampleResult r2 =
            nb_out.process_for_output(r_out, model_nb.data(), num_model, rt_nb.data(), k_rt_out);
        const int juce2 = j_out.process(r_out, model_j.data(), rt_j.data(), k_rt_out);
        CHECK(r2.input_consumed == juce2);
        CHECK(r2.output_produced == k_rt_out);

        float max_abs_diff = 0.0f;
        for (int i = 0; i < k_rt_out; ++i) {
            max_abs_diff = std::max(max_abs_diff, std::abs(rt_nb[static_cast<size_t>(i)] -
                                                           rt_j[static_cast<size_t>(i)]));
        }
        CHECK(max_abs_diff < 1e-5f);

        const double total_delay = 2.0 + 2.0 * (pair.host_sr / pair.model_sr);
        double sig_energy = 0.0;
        double err_energy = 0.0;
        for (int k = 128; k < k_rt_out - 128; ++k) {
            const double t = (static_cast<double>(k) - total_delay) / pair.host_sr;
            const double ideal = 0.8 * std::sin(2.0 * k_pi * 440.0 * t);
            const double actual = static_cast<double>(rt_nb[static_cast<size_t>(k)]);
            const double err = actual - ideal;
            sig_energy += ideal * ideal;
            err_energy += err * err;
        }
        const double snr_db = 10.0 * std::log10(sig_energy / std::max(1e-18, err_energy));
        CHECK(snr_db > 60.0);
    }

    // Impulse latency check matching JUCE (floor((64 + 2) / ratio))
    const std::array<double, 2> impulse_ratios{1.0, 0.5};
    const auto impulse = generate_signal(SignalKind::Impulse, 512, 48000.0);
    for (double ratio : impulse_ratios) {
        constexpr int k_out = 256;
        std::vector<float> out_nb(k_out, 0.0f);
        std::vector<float> out_j(k_out, 0.0f);
        Resampler nb(ResamplerQuality::CatmullRom);
        juce::CatmullRomInterpolator j;
        (void)nb.process_for_output(ratio, impulse.data(), static_cast<int>(impulse.size()),
                                    out_nb.data(), k_out);
        (void)j.process(ratio, impulse.data(), out_j.data(), k_out);

        const int peak_nb = static_cast<int>(
            std::distance(out_nb.begin(), std::max_element(out_nb.begin(), out_nb.end())));
        const int peak_j = static_cast<int>(
            std::distance(out_j.begin(), std::max_element(out_j.begin(), out_j.end())));
        const int expected = static_cast<int>(std::floor((64.0 + 2.0) / ratio));
        CHECK(peak_nb == expected);
        CHECK(peak_j == expected);

        float max_diff = 0.0f;
        for (int i = 0; i < k_out; ++i) {
            max_diff = std::max(max_diff, std::abs(out_nb[i] - out_j[i]));
        }
        CHECK(max_diff < 1e-6f);
    }
}

auto test_one_million_sample_drift() -> void {
    constexpr int k_total_samples = 1000000;
    const std::array<double, 3> ratios{
        48000.0 / 44100.0,
        44100.0 / 48000.0,
        std::sqrt(2.0),
    };
    const std::array<int, 5> blocks{64, 127, 256, 512, 1024};

    constexpr int k_window_cap = 4096;
    std::vector<float> src_window(k_window_cap, 0.0f);
    std::vector<float> out_nb(1024, 0.0f);
    std::vector<float> out_j(1024, 0.0f);

    for (double ratio : ratios) {
        Resampler nb(ResamplerQuality::CatmullRom);
        juce::CatmullRomInterpolator j;

        int64_t total_consumed = 0;
        int64_t total_juce_consumed = 0;
        int64_t produced = 0;
        int64_t next_global_idx = 0;
        int unread = 0;
        size_t blk_idx = 0;
        float max_abs_diff = 0.0f;

        auto refill_window = [&]() {
            while (unread < 2048) {
                src_window[static_cast<size_t>(unread)] = static_cast<float>(
                    0.8 * std::sin(2.0 * k_pi * 0.0123 * static_cast<double>(next_global_idx)));
                ++next_global_idx;
                ++unread;
            }
        };

        refill_window();

        while (produced < k_total_samples) {
            const int blk = static_cast<int>(
                std::min<int64_t>(blocks[blk_idx % blocks.size()], k_total_samples - produced));
            ++blk_idx;

            refill_window();

            const ResampleResult r =
                nb.process_for_output(ratio, src_window.data(), unread, out_nb.data(), blk);
            const int juce_used = j.process(ratio, src_window.data(), out_j.data(), blk);

            CHECK(r.output_produced == blk);
            CHECK(r.input_consumed == juce_used);

            for (int i = 0; i < blk; ++i) {
                max_abs_diff = std::max(max_abs_diff, std::abs(out_nb[static_cast<size_t>(i)] -
                                                               out_j[static_cast<size_t>(i)]));
            }

            if (r.input_consumed > 0) {
                const int remain = unread - r.input_consumed;
                std::move(src_window.begin() + r.input_consumed, src_window.begin() + unread,
                          src_window.begin());
                unread = remain;
            }

            total_consumed += r.input_consumed;
            total_juce_consumed += juce_used;
            produced += blk;
        }

        CHECK(total_consumed == total_juce_consumed);
        const int64_t ideal_consumed = std::llround(static_cast<double>(k_total_samples) * ratio);
        CHECK(std::abs(total_consumed - ideal_consumed) <= 2);
        CHECK(max_abs_diff < 1e-5f);
    }
}

auto test_bounded_input_and_double_overloads() -> void {
    constexpr double k_ratio = 48000.0 / 44100.0;
    constexpr int k_n_out = 256;
    const auto src = generate_signal(SignalKind::Sine440, 1024, 48000.0);

    // 1. Short-input boundary: max_in < required_input_samples
    {
        Resampler nb(ResamplerQuality::CatmullRom);
        juce::CatmullRomInterpolator j;
        const int req = nb.required_input_samples(k_ratio, k_n_out);
        const int max_in = std::max(1, req / 2);
        CHECK(max_in < req);

        std::vector<float> out_nb(k_n_out, 0.0f);
        std::vector<float> out_j(k_n_out, 0.0f);
        const ResampleResult r =
            nb.process_for_output(k_ratio, src.data(), max_in, out_nb.data(), k_n_out);
        const int j_used = j.process(k_ratio, src.data(), out_j.data(), k_n_out, max_in, 0);

        CHECK(r.output_produced == k_n_out);
        CHECK(r.input_consumed <= max_in);
        CHECK(r.input_consumed == j_used);
        for (int i = 0; i < k_n_out; ++i) {
            CHECK(std::abs(out_nb[static_cast<size_t>(i)] - out_j[static_cast<size_t>(i)]) < 1e-5f);
        }
    }

    // 2. Zero max_input_samples & nullptr safety on Resampler
    {
        Resampler nb(ResamplerQuality::CatmullRom);
        std::vector<float> out_zero(64, 99.0f);
        const ResampleResult r_zero =
            nb.process_for_output(k_ratio, src.data(), 0, out_zero.data(), 64);
        CHECK(r_zero.input_consumed == 0);
        CHECK(r_zero.output_produced == 64);
        for (float v : out_zero) {
            CHECK(v == 0.0f);
        }

        nb.reset();
        out_zero.assign(64, 99.0f);
        const ResampleResult r_null = nb.process_for_output(
            k_ratio, static_cast<const float*>(nullptr), 128, out_zero.data(), 64);
        CHECK(r_null.output_produced == 64);
        for (float v : out_zero) {
            CHECK(v == 0.0f);
        }
    }

    // 3. float->double and double->float overloads
    {
        std::vector<double> src_d(src.begin(), src.end());
        std::vector<float> out_ff(k_n_out, 0.0f);
        std::vector<double> out_fd(k_n_out, 0.0);
        std::vector<float> out_df(k_n_out, 0.0f);

        Resampler r_ff(ResamplerQuality::CatmullRom);
        Resampler r_fd(ResamplerQuality::CatmullRom);
        Resampler r_df(ResamplerQuality::CatmullRom);

        const auto res_ff = r_ff.process_for_output(
            k_ratio, src.data(), static_cast<int>(src.size()), out_ff.data(), k_n_out);
        const auto res_fd = r_fd.process_for_output(
            k_ratio, src.data(), static_cast<int>(src.size()), out_fd.data(), k_n_out);
        const auto res_df = r_df.process_for_output(
            k_ratio, src_d.data(), static_cast<int>(src_d.size()), out_df.data(), k_n_out);

        CHECK(res_ff.input_consumed == res_fd.input_consumed);
        CHECK(res_ff.input_consumed == res_df.input_consumed);
        for (int i = 0; i < k_n_out; ++i) {
            CHECK(std::abs(static_cast<double>(out_ff[static_cast<size_t>(i)]) -
                           out_fd[static_cast<size_t>(i)]) < 1e-6);
            CHECK(std::abs(out_ff[static_cast<size_t>(i)] - out_df[static_cast<size_t>(i)]) <
                  1e-6f);
        }
    }

    // 4. process_for_input (push-side float* and double*) bounded & continuous parity
    {
        std::vector<double> src_d(src.begin(), src.end());
        std::vector<float> out_push_f;
        std::vector<float> out_push_d;

        Resampler r_in_f(ResamplerQuality::CatmullRom);
        Resampler r_in_d(ResamplerQuality::CatmullRom);

        constexpr int k_chunk_in = 53;
        int pos = 0;
        std::vector<float> chunk_out(256, 0.0f);
        while (pos < 530) {
            const int n_in = std::min(k_chunk_in, 530 - pos);
            const ResampleResult rf =
                r_in_f.process_for_input(k_ratio, src.data() + pos, n_in, chunk_out.data(),
                                         static_cast<int>(chunk_out.size()));
            CHECK(rf.input_consumed <= n_in);
            if (rf.input_consumed == 0) {
                break;
            }
            CHECK(rf.output_produced > 0);
            out_push_f.insert(out_push_f.end(), chunk_out.begin(),
                              chunk_out.begin() + rf.output_produced);

            const ResampleResult rd =
                r_in_d.process_for_input(k_ratio, src_d.data() + pos, n_in, chunk_out.data(),
                                         static_cast<int>(chunk_out.size()));
            CHECK(rd.input_consumed == rf.input_consumed);
            CHECK(rd.output_produced == rf.output_produced);
            out_push_d.insert(out_push_d.end(), chunk_out.begin(),
                              chunk_out.begin() + rd.output_produced);

            pos += rf.input_consumed;
        }
        CHECK(pos >= 526);

        // Compare produced prefix directly against pull-side continuous output
        Resampler r_ref(ResamplerQuality::CatmullRom);
        std::vector<float> out_ref(out_push_f.size(), 0.0f);
        const ResampleResult rref =
            r_ref.process_for_output(k_ratio, src.data(), static_cast<int>(src.size()),
                                     out_ref.data(), static_cast<int>(out_ref.size()));
        CHECK(rref.output_produced == static_cast<int>(out_push_f.size()));
        for (size_t i = 0; i < out_push_f.size(); ++i) {
            CHECK(std::abs(out_push_f[i] - out_ref[i]) < 1e-5f);
            CHECK(std::abs(out_push_d[i] - out_ref[i]) < 1e-5f);
        }
    }
}

}  // namespace

auto main() -> int {
    std::cout << "Running test_resampler_parity...\n";

    test_catmull_rom_exact_parity_matrix();
    std::cout << "  [1/6] CatmullRom parity matrix vs juce::CatmullRomInterpolator: OK\n";

    test_linear_exact_parity();
    std::cout << "  [2/6] Linear parity vs juce::LinearInterpolator: OK\n";

    test_block_seam_continuity();
    std::cout << "  [3/6] Chunked vs single-shot seam continuity: OK\n";

    test_sine_roundtrip_snr_and_impulse_latency();
    std::cout << "  [4/6] Round-trip SNR & impulse latency: OK\n";

    test_one_million_sample_drift();
    std::cout << "  [5/6] 1,000,000-sample drift check vs JUCE: OK\n";

    test_bounded_input_and_double_overloads();
    std::cout << "  [6/6] Bounded short-input, double, and push-side overloads: OK\n";

    std::cout << "All test_resampler_parity tests passed!\n";
    return 0;
}
