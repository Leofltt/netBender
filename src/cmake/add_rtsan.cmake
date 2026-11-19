option(ENABLE_RTSAN "Enable Clang RealtimeSanitizer (RTSan) for audio-thread real-time safety checks" OFF)

add_library(netbender_rtsan INTERFACE)

if (ENABLE_RTSAN)
    include(CheckCXXSourceCompiles)

    set(_rtsan_prev_flags "${CMAKE_REQUIRED_FLAGS}")
    set(_rtsan_prev_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=realtime")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=realtime")

    check_cxx_source_compiles(
        "void rt_check() [[clang::nonblocking]] {}
         int main() { rt_check(); return 0; }"
        NETBENDER_COMPILER_SUPPORTS_RTSAN
    )

    set(CMAKE_REQUIRED_FLAGS "${_rtsan_prev_flags}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_rtsan_prev_link_options}")

    if (NOT NETBENDER_COMPILER_SUPPORTS_RTSAN)
        message(FATAL_ERROR
            "ENABLE_RTSAN=ON requires a compiler and runtime that support -fsanitize=realtime (upstream LLVM/Clang 20+).\n"
            "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER}).\n"
            "Note: Apple Clang does not ship the RTSan runtime library.\n"
            "On macOS, install Homebrew LLVM and reconfigure in a separate build directory:\n"
            "  brew install llvm\n"
            "  cmake -S src -B build-rtsan \\\n"
            "    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \\\n"
            "    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \\\n"
            "    -DCMAKE_PREFIX_PATH=\"$(pwd)/libtorch\" \\\n"
            "    -DENABLE_RTSAN=ON \\\n"
            "    -DCMAKE_BUILD_TYPE=RelWithDebInfo"
        )
    endif()

    target_compile_options(netbender_rtsan INTERFACE
        -fsanitize=realtime
        -fno-omit-frame-pointer
    )
    target_link_options(netbender_rtsan INTERFACE
        -fsanitize=realtime
    )
    target_compile_definitions(netbender_rtsan INTERFACE
        NETBENDER_RTSAN_ENABLED=1
    )

    message(STATUS "RealtimeSanitizer (RTSan) ENABLED (-fsanitize=realtime)")
    message(STATUS "  Tip: Run with RTSAN_OPTIONS=\"halt_on_error=false\" to collect all violations in one pass.")
endif()
