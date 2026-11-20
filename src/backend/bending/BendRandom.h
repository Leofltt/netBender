#pragma once

#include <cstdint>

namespace netbender::bending {

/**
 * Deterministic, zero-dependency, allocation-free PRNG used for stochastic weight drift.
 *
 * xoshiro128+ (Blackman & Vigna) seeded via splitmix64. Fast enough to generate one uniform
 * float per weight per inference block, fully reproducible for a given seed (unit tests,
 * preset recall), and safe to call from real-time threads.
 */
class Xoshiro128Plus {
public:
    static constexpr uint64_t k_default_seed = 0x4E424E4432303236ULL;  // "NBND2026"

    explicit Xoshiro128Plus(uint64_t seed = k_default_seed) noexcept { set_seed(seed); }

    auto set_seed(uint64_t seed) noexcept -> void {
        uint64_t sm = seed;
        for (auto& word : m_state) {
            word = static_cast<uint32_t>(splitmix64(sm) >> 32);
        }
        // xoshiro must never be seeded with an all-zero state
        if ((m_state[0] | m_state[1] | m_state[2] | m_state[3]) == 0U) {
            m_state[0] = 0x9E3779B9U;
        }
    }

    auto next_u32() noexcept -> uint32_t {
        const uint32_t result = m_state[0] + m_state[3];
        const uint32_t t = m_state[1] << 9;
        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];
        m_state[2] ^= t;
        m_state[3] = rotl(m_state[3], 11);
        return result;
    }

    // Uniform float in [0, 1) using the top 24 bits (exactly representable in float32)
    auto next_float() noexcept -> float {
        return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
    }

    // Uniform float in [-1, 1)
    auto next_bipolar() noexcept -> float { return next_float() * 2.0f - 1.0f; }

private:
    static constexpr auto rotl(uint32_t x, int k) noexcept -> uint32_t {
        return (x << k) | (x >> (32 - k));
    }

    static constexpr auto splitmix64(uint64_t& x) noexcept -> uint64_t {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    uint32_t m_state[4]{};
};

}  // namespace netbender::bending
