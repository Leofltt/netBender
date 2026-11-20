#include "AudioRingBuffer.h"
#include "rtsan_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
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

auto is_power_of_two(int v) -> bool {
    return v > 0 && (v & (v - 1)) == 0;
}

auto test_uninitialized_state() -> void {
    AudioRingBuffer rb;
    CHECK(rb.get_capacity() == 0);
    CHECK(rb.get_available() == 0);
    CHECK(rb.get_free_space() == 0);
    CHECK(rb.get_overrun_count() == 0);
    CHECK(rb.get_underrun_count() == 0);

    std::array<float, 4> f_in{1.0f, 2.0f, 3.0f, 4.0f};
    std::array<double, 4> d_in{1.0, 2.0, 3.0, 4.0};
    CHECK(rb.put(f_in.data(), 4) == 0);
    CHECK(rb.put(d_in.data(), 4) == 0);
    CHECK(rb.put(nullptr, 4) == 0);
    CHECK(rb.put_zeros(4) == 0);
    CHECK(rb.peek(f_in.data(), 4) == 0);
    CHECK(rb.drop(4) == 0);

    std::array<float, 4> f_out{9.0f, 9.0f, 9.0f, 9.0f};
    std::array<double, 4> d_out{9.0, 9.0, 9.0, 9.0};
    CHECK(rb.get(f_out.data(), 4) == 0);
    CHECK(rb.get(d_out.data(), 4) == 0);
    for (int i = 0; i < 4; ++i) {
        CHECK(f_out[static_cast<size_t>(i)] == 0.0f);
        CHECK(d_out[static_cast<size_t>(i)] == 0.0);
    }
}

auto test_capacity_rounding() -> void {
    AudioRingBuffer rb;

    rb.init(512);
    CHECK(rb.get_capacity() == 8192);
    CHECK(is_power_of_two(rb.get_capacity()));
    CHECK(rb.get_free_space() == 8192);

    rb.init(4096);
    CHECK(rb.get_capacity() == 16384);
    CHECK(is_power_of_two(rb.get_capacity()));

    rb.init_with_capacity(13);
    CHECK(rb.get_capacity() == 16);
    CHECK(is_power_of_two(rb.get_capacity()));

    rb.init_with_capacity(16);
    CHECK(rb.get_capacity() == 16);
    CHECK(is_power_of_two(rb.get_capacity()));

    rb.init_with_capacity(1);
    CHECK(rb.get_capacity() == 2);
    CHECK(is_power_of_two(rb.get_capacity()));
}

auto test_prime_chunk_wrap_around() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(128);
    CHECK(rb.get_capacity() == 128);

    constexpr int k_total_samples = 20000;
    constexpr std::array<int, 6> k_primes{7, 13, 17, 23, 31, 41};

    int next_write_val = 0;
    int next_read_val = 0;
    size_t write_step_idx = 0;
    size_t read_step_idx = 2;

    std::vector<float> tmp_in(64);
    std::vector<float> tmp_out(64);

    while (next_read_val < k_total_samples) {
        if (next_write_val < k_total_samples) {
            const int chunk = std::min(k_primes[write_step_idx % k_primes.size()],
                                       k_total_samples - next_write_val);
            if (rb.get_free_space() >= chunk) {
                for (int i = 0; i < chunk; ++i) {
                    tmp_in[static_cast<size_t>(i)] = static_cast<float>(next_write_val + i);
                }
                const int written = rb.put(tmp_in.data(), chunk);
                CHECK(written == chunk);
                next_write_val += chunk;
                ++write_step_idx;
            }
        }

        const int read_chunk = k_primes[read_step_idx % k_primes.size()];
        if (rb.get_available() >= read_chunk ||
            (next_write_val == k_total_samples && rb.get_available() > 0)) {
            const int to_read = std::min(read_chunk, rb.get_available());
            const int got = rb.get(tmp_out.data(), to_read);
            CHECK(got == to_read);
            for (int i = 0; i < to_read; ++i) {
                CHECK(tmp_out[static_cast<size_t>(i)] == static_cast<float>(next_read_val + i));
            }
            next_read_val += to_read;
            ++read_step_idx;
        }
    }

    CHECK(rb.get_available() == 0);
    CHECK(rb.get_free_space() == 128);
    CHECK(rb.get_overrun_count() == 0);
    CHECK(rb.get_underrun_count() == 0);
}

auto test_peek_drop_and_zero_pad() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(16);

    // Advance read/write cursors near wrap boundary (12/16)
    std::vector<float> seed(12);
    std::iota(seed.begin(), seed.end(), 100.0f);
    CHECK(rb.put(seed.data(), 12) == 12);
    std::vector<float> discard(12);
    CHECK(rb.get(discard.data(), 12) == 12);

    // Write 10 samples across the wrap boundary (slots 12..15, 0..5)
    std::vector<float> wrapped(10);
    std::iota(wrapped.begin(), wrapped.end(), 1.0f);
    CHECK(rb.put(wrapped.data(), 10) == 10);
    CHECK(rb.get_available() == 10);

    // Non-destructive peek across wrap boundary
    std::vector<float> peek_buf(10, -1.0f);
    CHECK(rb.peek(peek_buf.data(), 10) == 10);
    for (int i = 0; i < 10; ++i) {
        CHECK(peek_buf[static_cast<size_t>(i)] == static_cast<float>(i + 1));
    }
    CHECK(rb.get_available() == 10);

    // Partial drop (still crosses wrap edge: 4 samples -> index 16 == 0)
    CHECK(rb.drop(6) == 6);
    CHECK(rb.get_available() == 4);

    // Request 8 when only 4 remain -> returns 4, zero-pads [4..7], increments underruns by 4
    std::vector<float> out(8, -99.0f);
    CHECK(rb.get(out.data(), 8) == 4);
    CHECK(rb.get_available() == 0);
    CHECK(rb.get_underrun_count() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(out[static_cast<size_t>(i)] == static_cast<float>(7 + i));
    }
    for (int i = 4; i < 8; ++i) {
        CHECK(out[static_cast<size_t>(i)] == 0.0f);
    }
}

auto test_put_nullptr_and_zeros() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(16);

    // Fill with non-zero pattern and consume 12 so write cursor sits at 12
    std::vector<float> ones(12, 7.0f);
    CHECK(rb.put(ones.data(), 12) == 12);
    std::vector<float> tmp(12);
    CHECK(rb.get(tmp.data(), 12) == 12);

    // Write 6 via nullptr and 6 via put_zeros across wrap boundary, then 8 more (only 4 fit)
    CHECK(rb.put(nullptr, 6) == 6);
    CHECK(rb.put_zeros(6) == 6);
    CHECK(rb.put_zeros(8) == 4);
    CHECK(rb.get_overrun_count() == 4);
    CHECK(rb.get_available() == 16);

    std::vector<float> out(16, 1.0f);
    CHECK(rb.get(out.data(), 16) == 16);
    for (float s : out) {
        CHECK(s == 0.0f);
    }
}

auto test_overflow_non_corruption() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(16);

    std::vector<float> initial(16);
    std::iota(initial.begin(), initial.end(), 1.0f);
    CHECK(rb.put(initial.data(), 16) == 16);
    CHECK(rb.get_free_space() == 0);

    std::vector<float> extra(10, 999.0f);
    CHECK(rb.put(extra.data(), 10) == 0);
    CHECK(rb.get_overrun_count() == 10);

    std::vector<float> out(16, 0.0f);
    CHECK(rb.get(out.data(), 16) == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(out[static_cast<size_t>(i)] == static_cast<float>(i + 1));
    }

    // Partial write when 5 free slots remain
    rb.reset_counters();
    CHECK(rb.get_overrun_count() == 0);
    std::vector<float> eleven(11, 2.0f);
    CHECK(rb.put(eleven.data(), 11) == 11);
    CHECK(rb.get_free_space() == 5);

    std::vector<float> nine(9);
    std::iota(nine.begin(), nine.end(), 50.0f);
    CHECK(rb.put(nine.data(), 9) == 5);
    CHECK(rb.get_overrun_count() == 4);
    CHECK(rb.get_available() == 16);

    CHECK(rb.drop(11) == 11);
    std::vector<float> tail(5, 0.0f);
    CHECK(rb.get(tail.data(), 5) == 5);
    for (int i = 0; i < 5; ++i) {
        CHECK(tail[static_cast<size_t>(i)] == static_cast<float>(50 + i));
    }
}

auto test_float_double_overloads() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(16);

    // Advance cursors to 11 so subsequent I/O crosses wrap boundary
    std::vector<float> skip(11, 0.0f);
    CHECK(rb.put(skip.data(), 11) == 11);
    CHECK(rb.drop(11) == 11);

    std::vector<double> d_in(10);
    for (int i = 0; i < 10; ++i) {
        d_in[static_cast<size_t>(i)] = static_cast<double>(i) * 0.25 - 1.0;
    }
    CHECK(rb.put(d_in.data(), 10) == 10);

    std::vector<float> f_out(10, 0.0f);
    CHECK(rb.get(f_out.data(), 10) == 10);
    for (int i = 0; i < 10; ++i) {
        const auto idx = static_cast<size_t>(i);
        CHECK(std::abs(f_out[idx] - static_cast<float>(d_in[idx])) < 1e-6f);
    }

    std::vector<float> f_in(9);
    for (int i = 0; i < 9; ++i) {
        f_in[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.125f;
    }
    CHECK(rb.put(f_in.data(), 9) == 9);

    std::vector<double> d_out(12, -5.0);
    CHECK(rb.get(d_out.data(), 12) == 9);
    CHECK(rb.get_underrun_count() == 3);
    for (int i = 0; i < 9; ++i) {
        const auto idx = static_cast<size_t>(i);
        CHECK(std::abs(d_out[idx] - static_cast<double>(f_in[idx])) < 1e-6);
    }
    for (int i = 9; i < 12; ++i) {
        CHECK(d_out[static_cast<size_t>(i)] == 0.0);
    }
}

auto test_move_semantics() -> void {
    AudioRingBuffer src;
    src.init_with_capacity(32);

    std::vector<float> pad(20, -1.0f);
    CHECK(src.put(pad.data(), 20) == 20);
    CHECK(src.drop(20) == 20);

    std::vector<float> payload(18);
    std::iota(payload.begin(), payload.end(), 10.0f);
    CHECK(src.put(payload.data(), 18) == 18);

    AudioRingBuffer moved_ctor(std::move(src));
    CHECK(src.get_capacity() == 0);
    CHECK(src.get_available() == 0);
    CHECK(moved_ctor.get_capacity() == 32);
    CHECK(moved_ctor.get_available() == 18);

    AudioRingBuffer moved_assign;
    moved_assign = std::move(moved_ctor);
    CHECK(moved_ctor.get_capacity() == 0);
    CHECK(moved_ctor.get_available() == 0);
    CHECK(moved_assign.get_capacity() == 32);
    CHECK(moved_assign.get_available() == 18);

    std::vector<float> out(18, 0.0f);
    CHECK(moved_assign.get(out.data(), 18) == 18);
    for (int i = 0; i < 18; ++i) {
        CHECK(out[static_cast<size_t>(i)] == static_cast<float>(10 + i));
    }
}

auto test_concurrent_spsc_stress() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(1024);

    constexpr uint64_t k_total_samples = 2'000'000;
    std::atomic<bool> start{false};

    std::thread producer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::vector<float> chunk_buf(256);
        uint64_t written = 0;
        int chunk_len = 1;
        while (written < k_total_samples) {
            const int to_send = static_cast<int>(
                std::min<uint64_t>(static_cast<uint64_t>(chunk_len), k_total_samples - written));
            for (int i = 0; i < to_send; ++i) {
                chunk_buf[static_cast<size_t>(i)] =
                    static_cast<float>((written + static_cast<uint64_t>(i)) & 0xFFFFu);
            }

            int offset = 0;
            while (offset < to_send) {
                const int n = rb.put(chunk_buf.data() + offset, to_send - offset);
                offset += n;
                if (offset < to_send) {
                    std::this_thread::yield();
                }
            }

            written += static_cast<uint64_t>(to_send);
            chunk_len = (chunk_len % 197) + 1;
        }
    });

    std::thread consumer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::vector<float> chunk_buf(256);
        uint64_t read_total = 0;
        int chunk_len = 1;
        uint64_t iter = 0;
        while (read_total < k_total_samples) {
            const int want = static_cast<int>(
                std::min<uint64_t>(static_cast<uint64_t>(chunk_len), k_total_samples - read_total));
            int got = 0;
            if ((iter % 3u) == 0u) {
                got = rb.peek(chunk_buf.data(), want);
                if (got > 0) {
                    const int dropped = rb.drop(got);
                    CHECK(dropped == got);
                }
            } else {
                const int avail = rb.get_available();
                if (avail > 0) {
                    got = rb.get(chunk_buf.data(), std::min(want, avail));
                }
            }

            if (got == 0) {
                std::this_thread::yield();
                continue;
            }

            for (int i = 0; i < got; ++i) {
                const float expected =
                    static_cast<float>((read_total + static_cast<uint64_t>(i)) & 0xFFFFu);
                CHECK(chunk_buf[static_cast<size_t>(i)] == expected);
            }

            read_total += static_cast<uint64_t>(got);
            chunk_len = (chunk_len % 251) + 1;
            ++iter;
        }
    });

    start.store(true, std::memory_order_release);
    producer.join();
    consumer.join();

    CHECK(rb.get_available() == 0);
    CHECK(rb.get_underrun_count() == 0);
}

auto exercise_rt_empty_reads(AudioRingBuffer& rb, float* f_out, double* d_out,
                             int block_size) NETBENDER_NONBLOCKING -> void {
    (void)rb.get(f_out, block_size);
    (void)rb.get(d_out, block_size);
}

auto exercise_rt_methods(AudioRingBuffer& rb, const float* f_in, const double* d_in, float* f_out,
                         double* d_out, int block_size) NETBENDER_NONBLOCKING -> void {
    (void)rb.get_capacity();
    (void)rb.get_free_space();
    (void)rb.put(f_in, block_size);
    (void)rb.put(d_in, block_size);
    (void)rb.put(nullptr, block_size / 2);
    (void)rb.put_zeros(block_size / 2);
    (void)rb.get_available();
    (void)rb.peek(f_out, block_size);
    (void)rb.drop(block_size / 2);
    (void)rb.get(f_out, block_size);
    (void)rb.get(d_out, block_size);
    (void)rb.get_overrun_count();
    (void)rb.get_underrun_count();
    rb.reset_counters();
}

auto test_rtsan_nonblocking() -> void {
    AudioRingBuffer rb;
    rb.init_with_capacity(2048);

    constexpr int k_block = 128;
    std::vector<float> f_in(k_block, 0.25f);
    std::vector<double> d_in(k_block, -0.5);
    std::vector<float> f_out(k_block, 0.0f);
    std::vector<double> d_out(k_block, 0.0);

    exercise_rt_empty_reads(rb, f_out.data(), d_out.data(), k_block);
    for (int i = 0; i < 40; ++i) {
        exercise_rt_methods(rb, f_in.data(), d_in.data(), f_out.data(), d_out.data(), k_block);
    }
}

}  // namespace

auto main() -> int {
    std::cout << "=== NetBender AudioRingBuffer Unit Tests ===\n";
    std::cout << "RTSan instrumentation active: "
              << (netbender::rtsan::is_enabled() ? "YES (-fsanitize=realtime)" : "NO") << "\n";

    test_uninitialized_state();
    std::cout << "  [1/10] Uninitialized state: PASS\n";

    test_capacity_rounding();
    std::cout << "  [2/10] Capacity rounding: PASS\n";

    test_prime_chunk_wrap_around();
    std::cout << "  [3/10] Prime chunk wrap-around: PASS\n";

    test_peek_drop_and_zero_pad();
    std::cout << "  [4/10] Peek / drop / zero-pad: PASS\n";

    test_put_nullptr_and_zeros();
    std::cout << "  [5/10] put(nullptr) / put_zeros: PASS\n";

    test_overflow_non_corruption();
    std::cout << "  [6/10] Overflow non-corruption: PASS\n";

    test_float_double_overloads();
    std::cout << "  [7/10] Float / double overloads: PASS\n";

    test_move_semantics();
    std::cout << "  [8/10] Move semantics: PASS\n";

    test_concurrent_spsc_stress();
    std::cout << "  [9/10] Concurrent 2-thread 2M-sample SPSC stress: PASS\n";

    test_rtsan_nonblocking();
    std::cout << "  [10/10] RTSan nonblocking path: PASS\n";

    std::cout << "All AudioRingBuffer tests passed.\n";
    return 0;
}
