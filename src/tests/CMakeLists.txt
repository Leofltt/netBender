cmake_minimum_required(VERSION 3.15)
project(nn_bending_tests VERSION ${CMAKE_PROJECT_VERSION})

add_executable(test_audio_ring_buffer
    test_audio_ring_buffer.cpp
)

target_compile_features(test_audio_ring_buffer PRIVATE cxx_std_20)

target_link_libraries(test_audio_ring_buffer PRIVATE
    NetBender::dsp
)

add_test(NAME test_audio_ring_buffer COMMAND test_audio_ring_buffer)

add_executable(test_dsp_core
    test_dsp_core.cpp
)

target_compile_features(test_dsp_core PRIVATE cxx_std_20)

target_link_libraries(test_dsp_core PRIVATE
    NetBender::dsp
)

add_test(NAME test_dsp_core COMMAND test_dsp_core)

add_executable(test_backend_core
    test_backend_core.cpp
)

target_include_directories(test_backend_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(test_backend_core PRIVATE cxx_std_20)

target_link_libraries(test_backend_core PRIVATE
    NetBender::backend
)

if (APPLE)
    set_target_properties(test_backend_core PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
    )
endif()

add_test(NAME test_backend_core COMMAND test_backend_core)

add_executable(test_resampling_spsc
    test_resampling_spsc.cpp
)

target_compile_features(test_resampling_spsc PRIVATE cxx_std_20)

target_link_libraries(test_resampling_spsc PRIVATE
    NetBender::dsp
)

add_test(NAME test_resampling_spsc COMMAND test_resampling_spsc)

add_executable(test_streaming_engine
    test_streaming_engine.cpp
)

target_include_directories(test_streaming_engine PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(test_streaming_engine PRIVATE cxx_std_20)

target_link_libraries(test_streaming_engine PRIVATE
    NetBender::backend
)

if (APPLE)
    set_target_properties(test_streaming_engine PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
    )
endif()

add_test(NAME test_streaming_engine COMMAND test_streaming_engine)

add_executable(test_bending
    test_bending.cpp
)

target_include_directories(test_bending PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(test_bending PRIVATE cxx_std_20)

target_link_libraries(test_bending PRIVATE
    NetBender::backend
)

if (APPLE)
    set_target_properties(test_bending PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
    )
endif()

add_test(NAME test_bending COMMAND test_bending)

add_executable(test_backends
    test_backends.cpp
)

target_include_directories(test_backends PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/backend"
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_compile_features(test_backends PRIVATE cxx_std_20)

target_link_libraries(test_backends PRIVATE
    netbender_rtsan
    backend
    "${TORCH_LIBRARIES}"
    onnxruntime
)

if (APPLE)
    set_target_properties(test_backends PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
    )
endif()

if (BUILD_JUCE_PLUGIN)
    juce_add_console_app(test_rtsan
        PRODUCT_NAME "NetBenderRTSanTest"
    )
    juce_generate_juce_header(test_rtsan)

    target_sources(test_rtsan PRIVATE
        test_rtsan.cpp
    )

    target_include_directories(test_rtsan PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/frontend/juce"
        "${CMAKE_SOURCE_DIR}/frontend/juce/ui"
        "${CMAKE_SOURCE_DIR}/backend"
        "${CMAKE_BINARY_DIR}/_deps"
    )

    target_compile_features(test_rtsan PRIVATE cxx_std_20)
    target_compile_definitions(test_rtsan PRIVATE
        JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
        NOMINMAX=1
    )

    target_link_libraries(test_rtsan PRIVATE
        netbender_rtsan
        nn_bending_plugin
        backend
        "${TORCH_LIBRARIES}"
        onnxruntime
        nlohmann_json::nlohmann_json
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_formats
        juce::juce_audio_devices
        juce::juce_audio_basics
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        juce::juce_data_structures
        juce::juce_events
        juce::juce_core
    )

    if (APPLE)
        set_target_properties(test_rtsan PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
        )
    endif()

    add_test(NAME test_rtsan COMMAND test_rtsan)

    juce_add_console_app(test_resampler_parity
        PRODUCT_NAME "NetBenderResamplerParityTest"
    )

    target_sources(test_resampler_parity PRIVATE
        test_resampler_parity.cpp
    )

    target_compile_features(test_resampler_parity PRIVATE cxx_std_20)
    target_compile_definitions(test_resampler_parity PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        NOMINMAX=1
    )

    target_link_libraries(test_resampler_parity PRIVATE
        NetBender::dsp
        juce::juce_audio_basics
        juce::juce_core
    )

    add_test(NAME test_resampler_parity COMMAND test_resampler_parity)
endif()

# Aggregate target building all model-free CTest unit tests (excludes test_backends)
set(_nb_unit_test_targets
    test_audio_ring_buffer
    test_dsp_core
    test_backend_core
    test_resampling_spsc
    test_streaming_engine
    test_bending
)
if (BUILD_JUCE_PLUGIN)
    list(APPEND _nb_unit_test_targets
        test_resampler_parity
        test_rtsan
    )
endif()
add_custom_target(unit_tests DEPENDS ${_nb_unit_test_targets})

if (BUILD_CSOUND AND NETBENDER_CSOUND_FOUND)
    add_executable(test_csound_adapter
        test_csound_adapter.cpp
        "${CMAKE_SOURCE_DIR}/frontend/csound/CsoundSessionAdapter.cpp"
    )

    target_include_directories(test_csound_adapter PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/frontend/csound"
    )

    target_compile_features(test_csound_adapter PRIVATE cxx_std_20)

    target_link_libraries(test_csound_adapter PRIVATE
        NetBender::backend
    )

    if (APPLE)
        set_target_properties(test_csound_adapter PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
        )
    endif()

    add_test(NAME test_csound_adapter COMMAND test_csound_adapter)
    add_custom_target(csound_tests DEPENDS test_csound_adapter)
endif()

# On Windows, copy ONNX Runtime DLLs next to test executables so they are loaded
# ahead of any older preinstalled C:\Windows\System32\onnxruntime.dll.
if (WIN32)
    file(GLOB _nb_ort_win_dlls "${ort_extracted_dir}/lib/*.dll")
    if (_nb_ort_win_dlls)
        set(_nb_ort_dll_targets
            test_backend_core
            test_streaming_engine
            test_bending
            test_backends
        )
        if (BUILD_JUCE_PLUGIN)
            list(APPEND _nb_ort_dll_targets test_rtsan)
        endif()
        if (BUILD_CSOUND AND NETBENDER_CSOUND_FOUND)
            list(APPEND _nb_ort_dll_targets test_csound_adapter)
        endif()
        foreach(_t IN LISTS _nb_ort_dll_targets)
            add_custom_command(
                TARGET ${_t} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_nb_ort_win_dlls}
                    "$<TARGET_FILE_DIR:${_t}>"
                COMMENT "Copying ONNX Runtime DLLs next to ${_t}"
            )
        endforeach()
    endif()
endif()

# On Windows, prepend LibTorch and ONNX Runtime DLL directories to PATH for CTest runs
if (WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.22")
    set(_nb_win_dll_tests
        test_backend_core
        test_streaming_engine
        test_bending
    )
    if (BUILD_JUCE_PLUGIN)
        list(APPEND _nb_win_dll_tests test_rtsan)
    endif()
    if (BUILD_CSOUND AND NETBENDER_CSOUND_FOUND)
        list(APPEND _nb_win_dll_tests test_csound_adapter)
    endif()
    set(_nb_win_dll_mods "")
    foreach(_dir IN ITEMS
        "${TORCH_INSTALL_PREFIX}/lib"
        "${torch_dir}/libtorch/lib"
        "${ort_extracted_dir}/lib"
    )
        if (_dir AND EXISTS "${_dir}")
            file(TO_NATIVE_PATH "${_dir}" _native_dir)
            list(APPEND _nb_win_dll_mods "PATH=path_list_prepend:${_native_dir}")
        endif()
    endforeach()
    if (_nb_win_dll_mods)
        set_tests_properties(${_nb_win_dll_tests} PROPERTIES
            ENVIRONMENT_MODIFICATION "${_nb_win_dll_mods}"
        )
    endif()
endif()
