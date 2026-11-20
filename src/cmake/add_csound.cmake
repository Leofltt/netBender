# cmake/add_csound.cmake
#
# Locates Csound 7 plugin SDK headers (preferring CsoundConfig.cmake shipped with
# Csound 7, falling back to src/cmake/FindCsound.cmake) and configures the default
# user plugin install directory. Skips cleanly when Csound 7 is not installed.

option(BUILD_CSOUND "Build Csound 7 opcode targets" ON)
option(CSOUND_USE_DOUBLE "Build Csound opcodes for 64-bit floats (plugins64)" ON)

set(NETBENDER_CSOUND_FOUND FALSE)

if(NOT BUILD_CSOUND)
    return()
endif()

set(_nb_csound_include_dir "")

find_package(Csound 7.0 CONFIG QUIET
    PATHS
        /Applications/Csound/CsoundLib64.framework/Versions/Current/Resources/cmake
        /Library/Frameworks/CsoundLib64.framework/Versions/Current/Resources/cmake
        "$ENV{HOME}/Library/Frameworks/CsoundLib64.framework/Versions/Current/Resources/cmake"
        "$ENV{ProgramFiles}/Csound7/cmake"
        /usr/local/share/cmake/Csound
        /usr/share/cmake/Csound
)

if(Csound_FOUND)
    if(TARGET Csound::CsoundLib64)
        get_target_property(_nb_csound_include_dir Csound::CsoundLib64 INTERFACE_INCLUDE_DIRECTORIES)
    elseif(TARGET Csound::CsoundLib)
        get_target_property(_nb_csound_include_dir Csound::CsoundLib INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    if(_nb_csound_include_dir AND EXISTS "${_nb_csound_include_dir}")
        get_filename_component(_nb_csound_include_dir "${_nb_csound_include_dir}" REALPATH)
    endif()
    set(CSOUND_INCLUDE_DIRS "${_nb_csound_include_dir}")
else()
    set(_nb_saved_module_path "${CMAKE_MODULE_PATH}")
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(Csound 7.0 QUIET MODULE)
    set(CMAKE_MODULE_PATH "${_nb_saved_module_path}")
    unset(_nb_saved_module_path)
endif()

if(CSOUND_INCLUDE_DIRS AND EXISTS "${CSOUND_INCLUDE_DIRS}/plugin.h")
    set(NETBENDER_CSOUND_FOUND TRUE)
    message(STATUS "Csound 7 headers found: ${CSOUND_INCLUDE_DIRS}")
else()
    set(NETBENDER_CSOUND_FOUND FALSE)
    message(STATUS "Csound 7 headers not found; skipping Csound opcodes. Pass -DCsound_DIR=/path/to/csound/cmake or -DCSOUND_INCLUDE_DIR_HINT=... to enable.")
    return()
endif()

set(_nb_csound_api_version "7.0")
if(CSOUND_USE_DOUBLE)
    set(_nb_csound_plug_subdir "plugins64")
else()
    set(_nb_csound_plug_subdir "plugins")
endif()

if(NOT DEFINED CSOUND_PLUGIN_INSTALL_DIR OR "${CSOUND_PLUGIN_INSTALL_DIR}" STREQUAL "")
    if(APPLE)
        set(_nb_default_cs_dir "$ENV{HOME}/Library/csound/${_nb_csound_api_version}/${_nb_csound_plug_subdir}")
    elseif(WIN32)
        file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}" _nb_win_localappdata)
        set(_nb_default_cs_dir "${_nb_win_localappdata}/csound/${_nb_csound_api_version}/${_nb_csound_plug_subdir}")
        unset(_nb_win_localappdata)
    else()
        set(_nb_default_cs_dir "$ENV{HOME}/.local/lib/csound/${_nb_csound_api_version}/${_nb_csound_plug_subdir}")
    endif()
    set(CSOUND_PLUGIN_INSTALL_DIR "${_nb_default_cs_dir}" CACHE PATH "Csound 7 plugin install directory")
    unset(_nb_default_cs_dir)
endif()

message(STATUS "Csound 7 plugin install directory: ${CSOUND_PLUGIN_INSTALL_DIR}")
unset(_nb_csound_include_dir)
unset(_nb_csound_api_version)
unset(_nb_csound_plug_subdir)
