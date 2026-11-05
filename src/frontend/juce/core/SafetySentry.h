#pragma once

#include <atomic>
#include <functional>

class SafetySentry {
public:
    SafetySentry() = default;

    [[nodiscard]] auto is_fuse_blown() const -> bool { return m_blown_fuse.load(); }
    [[nodiscard]] auto isFuseBlown() const -> bool { return is_fuse_blown(); }

    auto trip_fuse() -> void { m_blown_fuse.store(true); }
    auto reset_fuse() -> void {
        m_blown_fuse.store(false);
        m_cooldown_blocks.store(0);
    }
    auto resetFuse() -> void { reset_fuse(); }

    auto set_auto_reset(bool enabled) -> void { m_auto_reset.store(enabled); }
    auto setAutoResetFuse(bool enabled) -> void { set_auto_reset(enabled); }

    [[nodiscard]] auto get_auto_reset() const -> bool { return m_auto_reset.load(); }
    [[nodiscard]] auto getAutoResetFuse() const -> bool { return get_auto_reset(); }

    // Safely blends wet model output with dry scratch buffer, enforcing soft headroom saturation,
    // NaN / Inf detection, and automatic output conversion clamp protection.
    auto process_channel(float* wet_dest, const float* dry_src, int num_samples, float wet_gain, float dry_gain) -> void;

    // Checks auto-recovery block timer and invokes on_reset if cooldown threshold is reached
    auto tick_recovery(const std::function<void()>& on_reset) -> void;

private:
    std::atomic<bool> m_blown_fuse { false };
    std::atomic<bool> m_auto_reset { false };
    std::atomic<int>  m_cooldown_blocks { 0 };
};

