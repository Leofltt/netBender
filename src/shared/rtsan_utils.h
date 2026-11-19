#pragma once

// Portable RealtimeSanitizer (RTSan) annotations for audio-thread real-time safety checks.
//
// When built with Clang 20+ and -DENABLE_RTSAN=ON (-fsanitize=realtime), functions annotated
// with NETBENDER_NONBLOCKING are dynamically checked at runtime for real-time violations
// (heap allocations, mutex locks, syscalls, I/O, etc.). On compilers without RTSan support,
// these macros expand cleanly to no-ops.

#if defined(__has_cpp_attribute)
  #if __has_cpp_attribute(clang::nonblocking)
    #define NETBENDER_NONBLOCKING [[clang::nonblocking]]
  #else
    #define NETBENDER_NONBLOCKING
  #endif
  #if __has_cpp_attribute(clang::blocking)
    #define NETBENDER_BLOCKING [[clang::blocking]]
  #else
    #define NETBENDER_BLOCKING
  #endif
#else
  #define NETBENDER_NONBLOCKING
  #define NETBENDER_BLOCKING
#endif

#if defined(__has_feature)
  #if __has_feature(realtime_sanitizer)
    #define NETBENDER_HAS_RTSAN 1
    #include <sanitizer/rtsan_interface.h>
  #else
    #define NETBENDER_HAS_RTSAN 0
  #endif
#else
  #define NETBENDER_HAS_RTSAN 0
#endif

namespace netbender::rtsan {

constexpr auto is_enabled() noexcept -> bool {
    return NETBENDER_HAS_RTSAN != 0;
}

#if NETBENDER_HAS_RTSAN
using ScopedDisabler = __rtsan::ScopedDisabler;
#else
class ScopedDisabler {
public:
    ScopedDisabler() noexcept = default;
    ~ScopedDisabler() noexcept = default;
    ScopedDisabler(const ScopedDisabler&) = delete;
    auto operator=(const ScopedDisabler&) -> ScopedDisabler& = delete;
    ScopedDisabler(ScopedDisabler&&) = delete;
    auto operator=(ScopedDisabler&&) -> ScopedDisabler& = delete;
};
#endif

} // namespace netbender::rtsan
