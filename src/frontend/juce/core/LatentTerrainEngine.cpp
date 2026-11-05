#include "LatentTerrainEngine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

auto LatentTerrainEngine::set_coords(float x, float y) -> void {
    m_x.store(std::clamp(x, -1.0f, 1.0f));
    m_y.store(std::clamp(y, -1.0f, 1.0f));
}

auto LatentTerrainEngine::set_depth(float depth) -> void {
    m_depth.store(std::clamp(depth, 0.0f, 2.0f));
}

auto LatentTerrainEngine::set_slew(float slew) -> void {
    m_slew.store(std::clamp(slew, 0.0f, 0.999f));
}

auto LatentTerrainEngine::reset() -> void {
    m_prev_latent_buffer.clear();
    m_rff_matrix.clear();
    m_cached_latent_dim = 0;
}

auto LatentTerrainEngine::apply_hook(IBackend::LatentBuffer& buf, float override_x, float override_y, float override_depth) -> void {
    if (buf.data == nullptr || buf.channels <= 0 || buf.length <= 0) {
        return;
    }

    float lx = (override_x > -900.0f) ? override_x : m_x.load();
    float ly = (override_y > -900.0f) ? override_y : m_y.load();
    float depth = (override_depth > -900.0f) ? override_depth : m_depth.load();
    auto mode = m_mode.load();

    int64_t latent_dim = buf.channels;
    int64_t length = buf.length;
    size_t total_elems = static_cast<size_t>(latent_dim * length);

    if (mode == LatentMode::FourierOrbit) {
        int64_t num_harmonics = std::max(static_cast<int64_t>(1), latent_dim / 2);
        if (m_rff_matrix.size() != static_cast<size_t>(num_harmonics * 2) || m_cached_latent_dim != static_cast<int>(latent_dim)) {
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 2.5f);
            m_rff_matrix.resize(static_cast<size_t>(num_harmonics * 2));
            for (auto& val : m_rff_matrix) {
                val = dist(rng);
            }
            m_cached_latent_dim = static_cast<int>(latent_dim);
        }

        float norm_depth = depth * (std::sqrt(16.0f) / std::sqrt(static_cast<float>(std::max(static_cast<int64_t>(1), latent_dim))));

        for (int64_t h = 0; h < num_harmonics; ++h) {
            float proj = m_rff_matrix[static_cast<size_t>(h * 2)] * lx + m_rff_matrix[static_cast<size_t>(h * 2 + 1)] * ly;
            float cos_offset = norm_depth * std::cos(proj);
            float sin_offset = norm_depth * std::sin(proj);

            if (h < latent_dim) {
                float* ch_ptr = buf.data + (h * length);
                for (int64_t t = 0; t < length; ++t) {
                    ch_ptr[t] += cos_offset;
                }
            }
            if ((num_harmonics + h) < latent_dim) {
                float* ch_ptr = buf.data + ((num_harmonics + h) * length);
                for (int64_t t = 0; t < length; ++t) {
                    ch_ptr[t] += sin_offset;
                }
            }
        }
    } else if (mode == LatentMode::LatentSlew) {
        float slew = m_slew.load();
        if (m_prev_latent_buffer.size() == total_elems) {
            for (size_t i = 0; i < total_elems; ++i) {
                buf.data[i] = (1.0f - slew) * buf.data[i] + slew * m_prev_latent_buffer[i];
            }
        } else {
            m_prev_latent_buffer.resize(total_elems);
        }
        std::memcpy(m_prev_latent_buffer.data(), buf.data, total_elems * sizeof(float));
    } else if (mode == LatentMode::Quantize) {
        float q_step = std::max(0.01f, depth * 0.5f);
        for (size_t i = 0; i < total_elems; ++i) {
            buf.data[i] = std::round(buf.data[i] / q_step) * q_step;
        }
    }
}

