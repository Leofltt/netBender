#include "CsoundSessionAdapter.h"
#include "MockBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " #cond << "\n";           \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

using netbender::csound_fe::CsoundSessionAdapter;
using netbender::csound_fe::OpenOptions;
using netbender::csound_fe::resolve_model_path;

class CsoundMockBackend : public MockBackend {
public:
    bool expose_decode_gen{true};
    double fixed_model_sr{44100.0};

    auto get_sample_rate() -> double override {
        return (fixed_model_sr > 0.0) ? fixed_model_sr : MockBackend::get_sample_rate();
    }

    auto get_plugin_modes() -> std::vector<std::string> override {
        auto modes = MockBackend::get_plugin_modes();
        if (expose_decode_gen &&
            std::find(modes.begin(), modes.end(), "decode gen") == modes.end()) {
            modes.emplace_back("decode gen");
        }
        return modes;
    }

    auto get_mode_params(std::string mode) -> std::vector<int> override {
        if (mode == "decode gen") {
            return {0, 1, out_channels, out_ratio};
        }
        return MockBackend::get_mode_params(std::move(mode));
    }
};

struct TempDir {
    std::filesystem::path path;

    explicit TempDir(const std::string& name) {
        const auto unique_tag =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        path =
            std::filesystem::temp_directory_path() / ("nb_csound_test_" + name + "_" + unique_tag);
        std::filesystem::create_directories(path);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

auto touch_file(const std::filesystem::path& p) -> void {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream ofs(p);
    ofs << "mock";
}

auto test_resolve_model_path() -> void {
    TempDir tmp("resolve");
    const auto direct_dir = tmp.path / "direct";
    const auto ssdir = tmp.path / "ssdir";
    const auto home_dir = tmp.path / "home_models";

    const auto direct_pte = direct_dir / "voice.pte";
    const auto ssdir_onnx = ssdir / "synth.onnx";
    const auto home_ts = home_dir / "perc.ts";
    const auto prio_pte = direct_dir / "multi.pte";
    const auto prio_onnx = direct_dir / "multi.onnx";

    touch_file(direct_pte);
    touch_file(ssdir_onnx);
    touch_file(home_ts);
    touch_file(prio_pte);
    touch_file(prio_onnx);

    auto ssdir_finder = [&ssdir](const std::string& cand) -> std::string {
        const auto p = ssdir / cand;
        std::error_code ec;
        return std::filesystem::is_regular_file(p, ec) ? p.string() : std::string{};
    };

    // 1. Exact path match
    CHECK(resolve_model_path(direct_pte.string(), ssdir_finder, home_dir.string()) ==
          direct_pte.string());

    // 2. Extension probing priority (.pte > .onnx > .ts)
    CHECK(resolve_model_path((direct_dir / "multi").string(), ssdir_finder, home_dir.string()) ==
          prio_pte.string());

    // 3. Fallback to host finder (Csound FindInputFile)
    CHECK(resolve_model_path("synth", ssdir_finder, home_dir.string()) == ssdir_onnx.string());
    CHECK(resolve_model_path("synth.onnx", ssdir_finder, home_dir.string()) == ssdir_onnx.string());

    // 4. Fallback to ~/.netbender/models
    CHECK(resolve_model_path("perc", ssdir_finder, home_dir.string()) == home_ts.string());
    CHECK(resolve_model_path("perc.ts", ssdir_finder, home_dir.string()) == home_ts.string());

    // 5. Missing file returns empty string
    CHECK(resolve_model_path("does_not_exist", ssdir_finder, home_dir.string()).empty());
}

auto test_open_defaults_and_modes() -> void {
    // 1. Realtime output -> auto async worker, sentry on, dry delay on for forward
    {
        CsoundSessionAdapter adapter(
            [](const std::string&) { return std::make_unique<CsoundMockBackend>(); });
        OpenOptions opts;
        opts.model_path = "mock_forward.pte";
        opts.method = "forward";
        opts.host_sample_rate = 48000.0;
        opts.ksmps = 128;
        opts.is_realtime_output = true;

        std::string err;
        CHECK(adapter.open(opts, err));
        CHECK(err.empty());
        CHECK(adapter.is_open());

        const auto summary = adapter.get_summary();
        CHECK(summary.method == "forward");
        CHECK(summary.in_channels == 1);
        CHECK(summary.out_channels == 1);
        CHECK(summary.worker_thread_enabled);
        CHECK(summary.safety_sentry_enabled);
        CHECK(summary.dry_delay_enabled);
        CHECK(summary.resampling_active);  // 48 kHz host vs 44.1 kHz MockBackend default
        CHECK(summary.layers.size() == 2);
        CHECK(!summary.methods.empty());

        const std::string dump = adapter.format_dump();
        CHECK(dump.find("NetBender Model Info") != std::string::npos);
        CHECK(dump.find("layer0.weight") != std::string::npos);

        adapter.close();
        CHECK(!adapter.is_open());
    }

    // 2. Offline output -> auto sync inline, encode disables sentry & dry delay
    {
        CsoundSessionAdapter adapter(
            [](const std::string&) { return std::make_unique<CsoundMockBackend>(); });
        OpenOptions opts;
        opts.model_path = "mock_encode.pte";
        opts.method = "encode";
        opts.host_sample_rate = 44100.0;
        opts.ksmps = 64;
        opts.is_realtime_output = false;

        std::string err;
        CHECK(adapter.open(opts, err));
        const auto summary = adapter.get_summary();
        CHECK(summary.method == "encode");
        CHECK(summary.in_channels == 1);
        CHECK(summary.out_channels == 4);  // MockBackend::latent_dim
        CHECK(!summary.worker_thread_enabled);
        CHECK(!summary.safety_sentry_enabled);
        CHECK(!summary.dry_delay_enabled);
        CHECK(!summary.resampling_active);
    }

    // 3. buffer_size == 0 forces synchronous inline even when realtime output is true
    {
        CsoundSessionAdapter adapter(
            [](const std::string&) { return std::make_unique<CsoundMockBackend>(); });
        OpenOptions opts;
        opts.model_path = "mock_sync.pte";
        opts.buffer_size = 0;
        opts.is_realtime_output = true;

        std::string err;
        CHECK(adapter.open(opts, err));
        CHECK(!adapter.get_summary().worker_thread_enabled);
    }

    // 4. Unsupported method fails cleanly with descriptive error
    {
        CsoundSessionAdapter adapter(
            [](const std::string&) { return std::make_unique<CsoundMockBackend>(); });
        OpenOptions opts;
        opts.model_path = "mock_bad_method.pte";
        opts.method = "nonexistent_method";

        std::string err;
        CHECK(!adapter.open(opts, err));
        CHECK(!adapter.is_open());
        CHECK(err.find("nonexistent_method") != std::string::npos);
    }
}

auto test_process_resampling_and_0dbfs() -> void {
    // Synchronous inline mode at 48 kHz host vs 44.1 kHz model with 0dbfs = 32768.0
    CsoundSessionAdapter adapter([](const std::string&) {
        auto b = std::make_unique<CsoundMockBackend>();
        b->gain = 0.5f;
        return b;
    });

    OpenOptions opts;
    opts.model_path = "mock_resamp.pte";
    opts.method = "forward";
    opts.buffer_size = 2048;
    opts.thread_mode = 0;    // sync inline for deterministic test
    opts.safety_sentry = 0;  // bypass soft-clipping for linear gain verification
    opts.dry_delay = 1;
    opts.host_sample_rate = 48000.0;
    opts.ksmps = 128;

    std::string err;
    CHECK(adapter.open(opts, err));
    CHECK(adapter.get_summary().resampling_active);

    constexpr int kKsmps = 128;
    constexpr int kBlocks = 120;
    constexpr double kZeroDbfs = 32768.0;

    std::vector<double> in_l(kKsmps);
    std::vector<double> out_l(kKsmps);
    std::vector<double> out_r(kKsmps);
    double max_abs_out = 0.0;

    for (int blk = 0; blk < kBlocks; ++blk) {
        for (int i = 0; i < kKsmps; ++i) {
            const int n = blk * kKsmps + i;
            in_l[static_cast<size_t>(i)] = kZeroDbfs * 0.5 * std::sin(0.05 * n);
        }
        const double* in_ptrs[1] = {in_l.data()};
        double* out_ptrs[2] = {out_l.data(), out_r.data()};

        adapter.process(in_ptrs, 1, out_ptrs, 2, kKsmps, kZeroDbfs, 1.0f, true);

        if (blk > 50) {
            for (int i = 0; i < kKsmps; ++i) {
                CHECK(std::isfinite(out_l[static_cast<size_t>(i)]));
                CHECK(std::abs(out_l[static_cast<size_t>(i)] - out_r[static_cast<size_t>(i)]) <
                      1e-6);
                max_abs_out = std::max(max_abs_out, std::abs(out_l[static_cast<size_t>(i)]));
            }
        }
    }

    // With 0dbfs = 32768, 0.5 input amp * 0.5 model gain -> ~8192 peak
    CHECK(max_abs_out > 4000.0);
    CHECK(max_abs_out < 12000.0);

    // Enabled == false must zero outputs cleanly
    std::fill(out_l.begin(), out_l.end(), 123.0);
    std::fill(out_r.begin(), out_r.end(), -456.0);
    const double* in_ptrs[1] = {in_l.data()};
    double* out_ptrs[2] = {out_l.data(), out_r.data()};
    adapter.process(in_ptrs, 1, out_ptrs, 2, kKsmps, kZeroDbfs, 1.0f, false);
    for (int i = 0; i < kKsmps; ++i) {
        CHECK(out_l[static_cast<size_t>(i)] == 0.0);
        CHECK(out_r[static_cast<size_t>(i)] == 0.0);
    }

    // Reset clears internal buffers without error
    adapter.reset();
    CHECK(adapter.is_open());
}

auto test_encode_decode_multichannel_and_float_overload() -> void {
    CsoundSessionAdapter adapter([](const std::string&) {
        auto b = std::make_unique<CsoundMockBackend>();
        b->latent_dim = 4;
        b->in_channels = 1;
        b->out_channels = 1;
        return b;
    });

    OpenOptions opts;
    opts.model_path = "mock_enc.pte";
    opts.method = "encode";
    opts.buffer_size = 2048;
    opts.thread_mode = 0;
    opts.host_sample_rate = 44100.0;
    opts.ksmps = 256;

    std::string err;
    CHECK(adapter.open(opts, err));
    CHECK(adapter.get_in_channels() == 1);
    CHECK(adapter.get_out_channels() == 4);

    constexpr int kFrames = 256;
    std::vector<float> in(kFrames, 0.25f);
    std::vector<std::vector<float>> outs(4, std::vector<float>(kFrames, 0.0f));
    const float* in_ptrs[1] = {in.data()};
    float* out_ptrs[4] = {outs[0].data(), outs[1].data(), outs[2].data(), outs[3].data()};

    float peak = 0.0f;
    for (int blk = 0; blk < 20; ++blk) {
        adapter.process(in_ptrs, 1, out_ptrs, 4, kFrames, 1.0, 1.0f, true);
        for (int c = 0; c < 4; ++c) {
            for (float v : outs[static_cast<size_t>(c)]) {
                CHECK(std::isfinite(v));
                peak = std::max(peak, std::abs(v));
            }
        }
    }
    CHECK(peak > 0.05f);
}

auto test_generate_modes() -> void {
    for (const std::string& method : {"prior", "decode_gen"}) {
        CsoundSessionAdapter adapter(
            [](const std::string&) { return std::make_unique<CsoundMockBackend>(); });
        OpenOptions opts;
        opts.model_path = "mock_gen.pte";
        opts.method = method;
        opts.buffer_size = 2048;
        opts.thread_mode = 0;
        opts.host_sample_rate = 44100.0;
        opts.ksmps = 128;

        std::string err;
        CHECK(adapter.open(opts, err));
        CHECK(adapter.get_in_channels() == 0);
        CHECK(adapter.get_out_channels() == 1);

        std::vector<double> out(128, 0.0);
        double* out_ptrs[1] = {out.data()};
        double peak = 0.0;
        for (int blk = 0; blk < 10; ++blk) {
            adapter.generate(out_ptrs, 1, 128, 1.0, true);
            for (double v : out) {
                CHECK(std::isfinite(v));
                peak = std::max(peak, std::abs(v));
            }
        }
        CHECK(peak > 0.05);
    }
}

auto test_csound_io_buffer_sizing_and_dump() -> void {
    CsoundSessionAdapter adapter(
        [](const std::string&) { return std::make_unique<CsoundMockBackend>(); });

    OpenOptions opts;
    opts.model_path = "mock_buffers.pte";
    opts.method = "forward";
    opts.buffer_size = 2048;
    opts.host_sample_rate = 48000.0;
    opts.ksmps = 32;
    opts.sw_buffer_frames = 1024;
    opts.hw_buffer_frames = 2048;
    opts.is_realtime_output = true;

    std::string err;
    CHECK(adapter.open(opts, err));
    CHECK(adapter.session() != nullptr);
    CHECK(adapter.session()->engine().get_config().max_host_block_size >= 1024);

    const auto summary = adapter.get_summary();
    CHECK(summary.ksmps == 32);
    CHECK(summary.sw_buffer_frames == 1024);
    CHECK(summary.hw_buffer_frames == 2048);

    const std::string dump = adapter.format_dump();
    CHECK(dump.find("ksmps=32") != std::string::npos);
    CHECK(dump.find("sw_buf(-b)=1024 frames") != std::string::npos);
    CHECK(dump.find("hw_buf(-B)=2048 frames") != std::string::npos);
}

}  // namespace

int main() {
    test_resolve_model_path();
    test_open_defaults_and_modes();
    test_process_resampling_and_0dbfs();
    test_encode_decode_multichannel_and_float_overload();
    test_generate_modes();
    test_csound_io_buffer_sizing_and_dump();
    std::cout << "[PASS] test_csound_adapter\n";
    return 0;
}
