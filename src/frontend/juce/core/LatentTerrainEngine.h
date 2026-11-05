#pragma once

#include "IBackend.h"
#include <atomic>
#include <vector>
#include <cstdint>

enum class LatentMode {
    FourierOrbit = 0, // 2D XY mapped via Random Fourier Features into multidimensional latent space
    LatentSlew,       // Temporal inertia / lowpass filter on latent activations
    Quantize          // Latent space bitcrushing / discretization
};

class LatentTerrainEngine {
public:
    LatentTerrainEngine() = default;

    auto set_mode(LatentMode mode) -> void { m_mode.store(mode); }
    [[nodiscard]] auto get_mode() const -> LatentMode { return m_mode.load(); }

    auto set_enabled(bool enabled) -> void { m_enabled.store(enabled); }
    [[nodiscard]] auto is_enabled() const -> bool { return m_enabled.load(); }

    auto set_coords(float x, float y) -> void;
    [[nodiscard]] auto get_x() const -> float { return m_x.load(); }
    [[nodiscard]] auto get_y() const -> float { return m_y.load(); }

    auto set_depth(float depth) -> void;
    [[nodiscard]] auto get_depth() const -> float { return m_depth.load(); }

    auto set_slew(float slew) -> void;
    [[nodiscard]] auto get_slew() const -> float { return m_slew.load(); }

    auto reset() -> void;

    // Applies Random Fourier Feature projection, slew filter, or quantization to latent tensor buffer
    auto apply_hook(IBackend::LatentBuffer& buf, float override_x = -999.0f, float override_y = -999.0f, float override_depth = -999.0f) -> void;

private:
    std::atomic<bool>       m_enabled { false };
    std::atomic<LatentMode> m_mode { LatentMode::FourierOrbit };
    std::atomic<float>      m_x { 0.0f };
    std::atomic<float>      m_y { 0.0f };
    std::atomic<float>      m_depth { 0.5f };
    std::atomic<float>      m_slew { 0.0f };

    std::vector<float>      m_prev_latent_buffer;
    std::vector<float>      m_rff_matrix;
    int                     m_cached_latent_dim { 0 };
};

