#include "RtGate.h"
#include "SafetySentry.h"
#include "WorkerThread.h"
#include "rtsan_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
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
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

auto test_sentry_normal_saturation_and_clamp() -> void {
    SafetySentry sentry;
    CHECK(!sentry.is_fuse_blown());
    CHECK(!sentry.get_auto_reset());

    // Linear region (|x| < 0.6f) -> exact wet * wet_gain + dry * dry_gain
    std::array<float, 6> linear_wet{-0.59f, -0.3f, 0.0f, 0.25f, 0.5f, 0.59f};
    std::array<float, 6> linear_dry{0.1f, -0.2f, 0.4f, -0.1f, 0.05f, -0.05f};
    std::array<float, 6> expected_linear{};
    for (size_t i = 0; i < linear_wet.size(); ++i) {
        expected_linear[i] = linear_wet[i] * 0.8f + linear_dry[i] * 0.5f;
    }
    sentry.process_channel(linear_wet.data(), linear_dry.data(),
                           static_cast<int>(linear_wet.size()), 0.8f, 0.5f);
    CHECK(!sentry.is_fuse_blown());
    for (size_t i = 0; i < linear_wet.size(); ++i) {
        CHECK(std::abs(linear_wet[i] - expected_linear[i]) < 1e-6f);
    }

    // Soft saturation region (|x| >= 0.6f) -> tanh(x * 0.77) * 1.3 and antisymmetry
    std::array<float, 4> sat_wet{0.6f, -0.6f, 1.2f, -1.2f};
    std::array<float, 4> sat_dry{0.0f, 0.0f, 0.0f, 0.0f};
    sentry.process_channel(sat_wet.data(), sat_dry.data(), static_cast<int>(sat_wet.size()), 1.0f,
                           0.0f);
    CHECK(!sentry.is_fuse_blown());
    CHECK(std::abs(sat_wet[0] - (std::tanh(0.6f * 0.77f) * 1.3f)) < 1e-6f);
    CHECK(std::abs(sat_wet[1] + sat_wet[0]) < 1e-6f);
    CHECK(std::abs(sat_wet[2] - (std::tanh(1.2f * 0.77f) * 1.3f)) < 1e-6f);
    CHECK(std::abs(sat_wet[3] + sat_wet[2]) < 1e-6f);

    // Output clamp region (1.5f < |sample| <= 2.0f) clamps to +/-1.5f without tripping fuse
    std::array<float, 2> clamp_wet{5.0f, -5.0f};
    std::array<float, 2> clamp_dry{0.5f, -0.5f};
    sentry.process_channel(clamp_wet.data(), clamp_dry.data(), static_cast<int>(clamp_wet.size()),
                           1.0f, 1.0f);
    CHECK(!sentry.is_fuse_blown());
    CHECK(clamp_wet[0] == 1.5f);
    CHECK(clamp_wet[1] == -1.5f);
}

auto test_sentry_fault_trip_bypass_and_null_safety() -> void {
    SafetySentry sentry;

    // Null and non-positive counts are safe no-ops
    std::array<float, 4> dummy{0.25f, 0.25f, 0.25f, 0.25f};
    sentry.process_channel(nullptr, dummy.data(), 4, 1.0f, 0.0f);
    sentry.process_channel(dummy.data(), nullptr, 4, 1.0f, 0.0f);
    sentry.process_channel(dummy.data(), dummy.data(), 0, 1.0f, 0.0f);
    sentry.process_channel(dummy.data(), dummy.data(), -4, 1.0f, 0.0f);
    CHECK(!sentry.is_fuse_blown());

    // Helper verifying fault value trips fuse, zeroes offending sample, and bypasses thereafter
    const std::array<float, 4> fault_values{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        3.0f,  // Wait: triggers |raw_wet| >= 0.6 -> sat_wet ~ 1.27, amplified below to > 2.0
    };

    for (size_t idx = 0; idx < fault_values.size(); ++idx) {
        sentry.reset_fuse();
        CHECK(!sentry.is_fuse_blown());

        std::array<float, 4> wet{0.1f, fault_values[idx], 0.2f, 0.3f};
        std::array<float, 4> dry{0.0f, 0.0f, 0.0f, 0.0f};
        const float wet_gain = (idx == 3) ? 2.0f : 1.0f;  // Force runaway > 2.0 on index 3
        sentry.process_channel(wet.data(), dry.data(), 4, wet_gain, 0.0f);

        CHECK(sentry.is_fuse_blown());
        CHECK(wet[1] == 0.0f);

        // Next call while blown must bypass wet (even NaNs) and copy pure dry regardless of gains
        std::array<float, 4> next_wet{std::numeric_limits<float>::quiet_NaN(), 99.0f, -99.0f, 0.5f};
        std::array<float, 4> next_dry{0.125f, -0.25f, 0.375f, -0.5f};
        sentry.process_channel(next_wet.data(), next_dry.data(), 4, 1.0f, 0.0f);
        CHECK(sentry.is_fuse_blown());
        for (size_t i = 0; i < 4; ++i) {
            CHECK(next_wet[i] == next_dry[i]);
        }
    }

    // Manual reset restores clean wet processing
    sentry.reset_fuse();
    CHECK(!sentry.is_fuse_blown());
    std::array<float, 2> recovered_wet{0.4f, -0.4f};
    std::array<float, 2> recovered_dry{0.1f, 0.1f};
    sentry.process_channel(recovered_wet.data(), recovered_dry.data(), 2, 1.0f, 0.5f);
    CHECK(!sentry.is_fuse_blown());
    CHECK(std::abs(recovered_wet[0] - 0.45f) < 1e-6f);
    CHECK(std::abs(recovered_wet[1] - (-0.35f)) < 1e-6f);
}

auto test_sentry_auto_recovery_cooldown() -> void {
    SafetySentry sentry;
    int reset_count = 0;
    auto on_reset = [&] {
        ++reset_count;
        sentry.reset_fuse();
    };

    // Auto-reset disabled by default -> never recovers
    sentry.trip_fuse();
    CHECK(sentry.is_fuse_blown());
    for (int i = 0; i < 100; ++i) {
        sentry.tick_recovery(on_reset);
    }
    CHECK(reset_count == 0);
    CHECK(sentry.is_fuse_blown());

    // Enabled while fuse is NOT blown -> never triggers callback
    sentry.reset_fuse();
    sentry.set_auto_reset(true);
    CHECK(sentry.get_auto_reset());
    for (int i = 0; i < 100; ++i) {
        sentry.tick_recovery(on_reset);
    }
    CHECK(reset_count == 0);

    // Exact cooldown: 41 ticks (0..40) do not trigger; 42nd tick (41 > 40) fires callback
    sentry.trip_fuse();
    for (int i = 0; i < 41; ++i) {
        sentry.tick_recovery(on_reset);
        CHECK(reset_count == 0);
        CHECK(sentry.is_fuse_blown());
    }
    sentry.tick_recovery(on_reset);
    CHECK(reset_count == 1);
    CHECK(!sentry.is_fuse_blown());

    // Manual reset_fuse() mid-cooldown clears cooldown counter back to 0
    sentry.trip_fuse();
    for (int i = 0; i < 25; ++i) {
        sentry.tick_recovery(on_reset);
    }
    CHECK(reset_count == 1);
    sentry.reset_fuse();

    sentry.trip_fuse();
    for (int i = 0; i < 41; ++i) {
        sentry.tick_recovery(on_reset);
        CHECK(reset_count == 1);
    }
    sentry.tick_recovery(on_reset);
    CHECK(reset_count == 2);
    CHECK(!sentry.is_fuse_blown());

    // Null/empty callback does not crash at cooldown completion
    sentry.trip_fuse();
    for (int i = 0; i < 42; ++i) {
        sentry.tick_recovery(nullptr);
    }
    CHECK(sentry.is_fuse_blown());
}

auto test_worker_lifecycle_and_restart() -> void {
    std::atomic<int> run_count{0};
    WorkerThread worker([&] { run_count.fetch_add(1, std::memory_order_acq_rel); });

    CHECK(!worker.is_running());
    CHECK(!worker.is_processing());

    // Idempotent stop before start
    worker.stop();
    CHECK(!worker.is_running());
    CHECK(!worker.is_processing());

    // Start + idempotent double start
    worker.start();
    CHECK(worker.is_running());
    CHECK(!worker.is_processing());
    worker.start();
    CHECK(worker.is_running());
    CHECK(run_count.load(std::memory_order_acquire) == 0);

    // Trigger once
    worker.trigger();
    CHECK(wait_until(
        [&] { return run_count.load(std::memory_order_acquire) == 1 && !worker.is_processing(); }));

    // Stop + idempotent double stop
    worker.stop();
    CHECK(!worker.is_running());
    CHECK(!worker.is_processing());
    worker.stop();
    CHECK(!worker.is_running());

    // Restart after stop
    worker.start();
    CHECK(worker.is_running());
    worker.trigger();
    CHECK(wait_until(
        [&] { return run_count.load(std::memory_order_acquire) == 2 && !worker.is_processing(); }));

    // Destruction while running must cleanly stop without hanging
    {
        std::atomic<int> scoped_runs{0};
        WorkerThread scoped_worker([&] { scoped_runs.fetch_add(1, std::memory_order_acq_rel); });
        scoped_worker.start();
        scoped_worker.trigger();
        CHECK(wait_until([&] { return scoped_runs.load(std::memory_order_acquire) == 1; }));
    }
}

auto test_worker_no_lost_wakeup_when_busy() -> void {
    std::atomic<int> run_count{0};
    std::atomic<bool> entered_first{false};
    std::atomic<bool> release_first{false};

    WorkerThread worker([&] {
        const int run = run_count.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (run == 1) {
            entered_first.store(true, std::memory_order_release);
            while (!release_first.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    });

    worker.start();
    worker.trigger();
    CHECK(wait_until([&] { return entered_first.load(std::memory_order_acquire); }));
    CHECK(worker.is_processing());

    // Trigger again while run #1 is still executing inside m_compute_fn
    worker.trigger();
    CHECK(worker.is_processing());

    release_first.store(true, std::memory_order_release);

    CHECK(wait_until(
        [&] { return run_count.load(std::memory_order_acquire) == 2 && !worker.is_processing(); }));
    worker.stop();
}

auto test_worker_rapid_burst_and_null_callback() -> void {
    std::atomic<int> published_seq{0};
    std::atomic<int> observed_seq{0};

    WorkerThread worker([&] {
        observed_seq.store(published_seq.load(std::memory_order_acquire),
                           std::memory_order_release);
    });

    worker.start();
    for (int i = 1; i <= 500; ++i) {
        published_seq.store(i, std::memory_order_release);
        worker.trigger();
    }
    CHECK(wait_until([&] {
        return observed_seq.load(std::memory_order_acquire) == 500 && !worker.is_processing();
    }));
    worker.stop();

    WorkerThread null_worker(nullptr);
    null_worker.start();
    for (int i = 0; i < 8; ++i) {
        null_worker.trigger();
    }
    CHECK(wait_until([&] { return !null_worker.is_processing(); }));
    null_worker.stop();
}

auto test_rt_gate_basic_and_deterministic_drain() -> void {
    RtGate gate;

    // Concurrent readers can acquire simultaneously
    {
        ScopedRtReader r1(gate);
        ScopedRtReader r2(gate);
        CHECK(r1.is_acquired());
        CHECK(r2.is_acquired());
    }

    // ScopedReconfigureGate blocks new readers and restores access on destruction
    {
        ScopedReconfigureGate reconf(gate);
        ScopedRtReader rejected(gate);
        CHECK(!rejected.is_acquired());
    }
    {
        ScopedRtReader r_after(gate);
        CHECK(r_after.is_acquired());
    }

    // Deterministic drain: reconfigurer waits until active reader exits scope
    std::atomic<bool> reconf_started{false};
    std::atomic<bool> reconf_entered{false};
    std::thread reconf_thread;

    {
        ScopedRtReader active_reader(gate);
        CHECK(active_reader.is_acquired());

        reconf_thread = std::thread([&] {
            reconf_started.store(true, std::memory_order_release);
            ScopedReconfigureGate reconf(gate);
            reconf_entered.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        });

        CHECK(wait_until([&] { return reconf_started.load(std::memory_order_acquire); }));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(!reconf_entered.load(std::memory_order_acquire));
    }

    reconf_thread.join();
    CHECK(reconf_entered.load(std::memory_order_acquire));
}

auto test_rt_gate_multithread_stress() -> void {
    RtGate gate;
    std::atomic<int> active_readers{0};
    std::atomic<int> active_reconf{0};
    std::atomic<uint64_t> reader_entries{0};
    std::atomic<bool> done{false};

    auto reader_fn = [&]() {
        while (!done.load(std::memory_order_acquire)) {
            ScopedRtReader guard(gate);
            if (guard.is_acquired()) {
                active_readers.fetch_add(1, std::memory_order_acq_rel);
                CHECK(active_reconf.load(std::memory_order_acquire) == 0);
                reader_entries.fetch_add(1, std::memory_order_relaxed);
                active_readers.fetch_sub(1, std::memory_order_acq_rel);
            } else {
                std::this_thread::yield();
            }
        }
    };

    std::thread r1(reader_fn);
    std::thread r2(reader_fn);
    CHECK(wait_until([&] { return reader_entries.load(std::memory_order_acquire) > 0; }));

    std::thread reconf_thread([&]() {
        for (int cycle = 0; cycle < 200; ++cycle) {
            {
                ScopedReconfigureGate reconf(gate);
                CHECK(active_readers.load(std::memory_order_acquire) == 0);
                active_reconf.fetch_add(1, std::memory_order_acq_rel);
                std::this_thread::yield();
                CHECK(active_readers.load(std::memory_order_acquire) == 0);
                active_reconf.fetch_sub(1, std::memory_order_acq_rel);
            }
            if ((cycle % 10) == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else {
                std::this_thread::yield();
            }
        }
        done.store(true, std::memory_order_release);
    });

    reconf_thread.join();
    r1.join();
    r2.join();

    CHECK(active_readers.load(std::memory_order_acquire) == 0);
    CHECK(active_reconf.load(std::memory_order_acquire) == 0);
    CHECK(reader_entries.load(std::memory_order_acquire) > 0);
}

auto exercise_sentry_rt(SafetySentry& sentry, float* wet, const float* dry, int block_size,
                        const std::function<void()>& reset_cb) NETBENDER_NONBLOCKING -> void {
    // 1. Normal + saturated pass
    sentry.process_channel(wet, dry, block_size, 0.8f, 0.2f);
    (void)sentry.is_fuse_blown();

    // 2. Trip via NaN
    wet[0] = std::numeric_limits<float>::quiet_NaN();
    sentry.process_channel(wet, dry, block_size, 1.0f, 0.0f);

    // 3. Blown-fuse bypass pass
    sentry.process_channel(wet, dry, block_size, 1.0f, 0.0f);

    // 4. Cooldown auto-recovery pass (resets via pre-built callback)
    for (int i = 0; i < 45; ++i) {
        sentry.tick_recovery(reset_cb);
    }

    // 5. Explicit trip & reset
    sentry.trip_fuse();
    sentry.reset_fuse();
}

auto exercise_rt_gate_acquired(RtGate& gate) NETBENDER_NONBLOCKING -> bool {
    ScopedRtReader outer(gate);
    ScopedRtReader inner(gate);
    return outer.is_acquired() && inner.is_acquired();
}

auto exercise_rt_gate_rejected(RtGate& gate) NETBENDER_NONBLOCKING -> bool {
    ScopedRtReader rejected(gate);
    return rejected.is_acquired();
}

auto exercise_worker_trigger_rt(WorkerThread& worker) NETBENDER_NONBLOCKING -> bool {
    const bool running = worker.is_running();
    (void)worker.is_processing();
    worker.trigger();
    return running;
}

auto test_dsp_core_rtsan() -> void {
    constexpr int k_block = 64;
    SafetySentry sentry;
    sentry.set_auto_reset(true);
    std::function<void()> reset_cb = [&sentry] { sentry.reset_fuse(); };

    std::vector<float> wet(k_block, 0.9f);
    std::vector<float> dry(k_block, 0.1f);
    exercise_sentry_rt(sentry, wet.data(), dry.data(), k_block, reset_cb);
    CHECK(!sentry.is_fuse_blown());

    RtGate gate;
    CHECK(exercise_rt_gate_acquired(gate));
    {
        ScopedReconfigureGate reconf(gate);
        CHECK(!exercise_rt_gate_rejected(gate));
    }

    std::atomic<int> rt_runs{0};
    WorkerThread worker([&] { rt_runs.fetch_add(1, std::memory_order_acq_rel); });
    worker.start();
    CHECK(exercise_worker_trigger_rt(worker));
    CHECK(wait_until([&] { return rt_runs.load(std::memory_order_acquire) >= 1; }));
    worker.stop();
}

}  // namespace

auto main() -> int {
    std::cout << "=== NetBender DSP Core Unit Tests ===\n";
    std::cout << "RTSan instrumentation active: "
              << (netbender::rtsan::is_enabled() ? "YES (-fsanitize=realtime)" : "NO") << "\n";

    test_sentry_normal_saturation_and_clamp();
    std::cout << "  [1/9] SafetySentry saturation & clamp: PASS\n";

    test_sentry_fault_trip_bypass_and_null_safety();
    std::cout << "  [2/9] SafetySentry NaN/Inf/runaway trip & bypass: PASS\n";

    test_sentry_auto_recovery_cooldown();
    std::cout << "  [3/9] SafetySentry auto-recovery cooldown: PASS\n";

    test_worker_lifecycle_and_restart();
    std::cout << "  [4/9] WorkerThread lifecycle & restart: PASS\n";

    test_worker_no_lost_wakeup_when_busy();
    std::cout << "  [5/9] WorkerThread no lost wakeup when busy: PASS\n";

    test_worker_rapid_burst_and_null_callback();
    std::cout << "  [6/9] WorkerThread rapid burst & null callback: PASS\n";

    test_rt_gate_basic_and_deterministic_drain();
    std::cout << "  [7/9] RtGate basic & deterministic drain: PASS\n";

    test_rt_gate_multithread_stress();
    std::cout << "  [8/9] RtGate multithread mutual-exclusion stress: PASS\n";

    test_dsp_core_rtsan();
    std::cout << "  [9/9] RTSan nonblocking path (Sentry/RtGate/WorkerThread): PASS\n";

    std::cout << "All DSP Core tests passed.\n";
    return 0;
}
