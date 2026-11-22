cmake_minimum_required(VERSION 3.15)
project(netbender_csound VERSION ${CMAKE_PROJECT_VERSION})

set(NETBENDER_CSOUND_SOURCES
    CsoundSessionAdapter.cpp
    CsoundSessionAdapter.h
    CsoundRegistry.cpp
    CsoundRegistry.h
    NetBenderOpcodes.cpp
)

if(APPLE)
    add_library(netbender_csound SHARED ${NETBENDER_CSOUND_SOURCES})
else()
    add_library(netbender_csound MODULE ${NETBENDER_CSOUND_SOURCES})
endif()

set_target_properties(netbender_csound PROPERTIES
    OUTPUT_NAME "netbender"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(WIN32)
    set_target_properties(netbender_csound PROPERTIES PREFIX "")
endif()

target_include_directories(netbender_csound PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CSOUND_INCLUDE_DIRS}"
)

if(NOT VERSION)
    if(CMAKE_PROJECT_VERSION)
        set(VERSION "v${CMAKE_PROJECT_VERSION}")
    else()
        set(VERSION "v0.1.0")
    endif()
endif()

target_compile_definitions(netbender_csound PRIVATE
    NETBENDER_CSOUND_VERSION="${VERSION}"
)

if(CSOUND_USE_DOUBLE)
    target_compile_definitions(netbender_csound PRIVATE USE_DOUBLE)
else()
    target_compile_definitions(netbender_csound PRIVATE USE_FLOAT)
endif()

target_link_libraries(netbender_csound PRIVATE
    NetBender::backend
    nlohmann_json::nlohmann_json
)

if(APPLE)
    set(_nb_csound_rpaths
        "${TORCH_INSTALL_PREFIX}/lib"
        "${torch_dir}/libtorch/lib"
        "${ort_extracted_dir}/lib"
        "${CONDA_ENV_PATH}/lib"
        "/Library/Application Support/NetBender/lib"
        "@loader_path/netbender_libs"
    )
    if(EXECUTORCH_FOUND AND EXECUTORCH_LIBRARY_DIR)
        list(INSERT _nb_csound_rpaths 3 "${EXECUTORCH_LIBRARY_DIR}")
    endif()
    list(JOIN _nb_csound_rpaths ";" _nb_csound_rpath_str)

    set_target_properties(netbender_csound PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_nb_csound_rpath_str}"
    )

    if(BUNDLE_DEPENDENCIES)
        set(_nb_dylib_search_dirs "")
        foreach(_dir IN ITEMS
            "${torch_dir}/libtorch"
            "${CMAKE_SOURCE_DIR}/support"
            "${ort_extracted_dir}/lib"
            "${EXECUTORCH_LIBRARY_DIR}"
            "${CMAKE_BINARY_DIR}/_deps"
            "${CMAKE_SOURCE_DIR}/../env"
            "${HOMEBREW_PREFIX}"
        )
            if(_dir AND EXISTS "${_dir}")
                list(APPEND _nb_dylib_search_dirs "${_dir}")
            endif()
        endforeach()

        set(_nb_csound_bundle_dir "${CMAKE_CURRENT_BINARY_DIR}/netbender_libs")
        add_custom_command(
            TARGET netbender_csound
            POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_nb_csound_bundle_dir}"
            COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/../install/dylib_fix.py"
                -p "$<TARGET_FILE:netbender_csound>"
                -o "${_nb_csound_bundle_dir}"
                -l ${_nb_dylib_search_dirs}
                --sign_id "${SIGN_ID}"
            COMMENT "Bundling shared dependencies into netbender_libs and codesigning netbender_csound"
            VERBATIM
        )
        install(
            DIRECTORY "${_nb_csound_bundle_dir}"
            DESTINATION "${CSOUND_PLUGIN_INSTALL_DIR}"
            COMPONENT csound
            OPTIONAL
        )
    else()
        add_custom_command(
            TARGET netbender_csound
            POST_BUILD
            COMMAND xattr -cr "$<TARGET_FILE:netbender_csound>"
            COMMAND codesign --force --sign "${SIGN_ID}" "$<TARGET_FILE:netbender_csound>"
            COMMENT "De-quarantining and codesigning Csound 7 opcode plugin"
            VERBATIM
        )
    endif()
elseif(UNIX AND NOT APPLE)
    set_target_properties(netbender_csound PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH "$ORIGIN/netbender_libs"
    )
elseif(WIN32)
    if(BUNDLE_DEPENDENCIES)
        set(_nb_win_bundle_dlls "")
        if(EXISTS "${TORCH_INSTALL_PREFIX}/lib")
            file(GLOB _nb_torch_dlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
            list(APPEND _nb_win_bundle_dlls ${_nb_torch_dlls})
        endif()
        if(ONNXRUNTIME_DLL AND EXISTS "${ONNXRUNTIME_DLL}")
            list(APPEND _nb_win_bundle_dlls "${ONNXRUNTIME_DLL}")
        endif()
        if(EXECUTORCH_FOUND AND EXECUTORCH_LIBRARY_DIR AND EXISTS "${EXECUTORCH_LIBRARY_DIR}")
            file(GLOB _nb_et_dlls "${EXECUTORCH_LIBRARY_DIR}/*.dll")
            list(APPEND _nb_win_bundle_dlls ${_nb_et_dlls})
        endif()

        if(_nb_win_bundle_dlls)
            add_custom_command(
                TARGET netbender_csound
                POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:netbender_csound>/netbender_libs"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    ${_nb_win_bundle_dlls}
                    "$<TARGET_FILE_DIR:netbender_csound>/netbender_libs"
                COMMENT "Copying runtime DLLs into netbender_libs"
                VERBATIM
            )
            if(CMAKE_CONFIGURATION_TYPES)
                set(_nb_win_install_libs_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/netbender_libs")
            else()
                set(_nb_win_install_libs_dir "${CMAKE_CURRENT_BINARY_DIR}/netbender_libs")
            endif()
            install(
                DIRECTORY "${_nb_win_install_libs_dir}"
                DESTINATION "${CSOUND_PLUGIN_INSTALL_DIR}"
                COMPONENT csound
                OPTIONAL
            )
        endif()
    endif()
endif()

install(TARGETS netbender_csound
    LIBRARY DESTINATION "${CSOUND_PLUGIN_INSTALL_DIR}" COMPONENT csound
    RUNTIME DESTINATION "${CSOUND_PLUGIN_INSTALL_DIR}" COMPONENT csound
)
