cmake_minimum_required(VERSION 3.15)
project(nn_bending_tests VERSION ${CMAKE_PROJECT_VERSION})

add_executable(test_backends
    test_backends.cpp
    "${CMAKE_SOURCE_DIR}/frontend/juce/dsp/AudioRingBuffer.cpp"
    "${CMAKE_SOURCE_DIR}/frontend/juce/core/SafetySentry.cpp"
)

target_include_directories(test_backends PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/backend"
    "${CMAKE_SOURCE_DIR}/shared"
    "${CMAKE_SOURCE_DIR}/frontend/juce/dsp"
    "${CMAKE_SOURCE_DIR}/frontend/juce/core"
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
        "${CMAKE_SOURCE_DIR}/frontend/juce/core"
        "${CMAKE_SOURCE_DIR}/frontend/juce/dsp"
        "${CMAKE_SOURCE_DIR}/backend"
        "${CMAKE_SOURCE_DIR}/shared"
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
endif()
