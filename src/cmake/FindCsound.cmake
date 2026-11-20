# Try to find the Csound 7 headers and library.
# Once done this will define:
#  CSOUND_FOUND        - System has the Csound headers
#  CSOUND_INCLUDE_DIRS - The Csound include directories
#  CSOUND_LIBRARIES    - The Csound library (informational; plugin opcodes do not link it)
#  CSOUND_VERSION      - The version of the Csound headers found

if(APPLE)
    find_path(CSOUND_INCLUDE_DIR plugin.h
        HINTS
            ${CSOUND_INCLUDE_DIR_HINT}
            /Applications/Csound/CsoundLib64.framework/Versions/Current/Headers
            /Library/Frameworks/CsoundLib64.framework/Headers
            /Library/Frameworks/CsoundLib.framework/Headers
            "$ENV{HOME}/Library/Frameworks/CsoundLib64.framework/Headers"
            "$ENV{HOME}/Library/Frameworks/CsoundLib.framework/Headers"
    )
    find_path(CSOUND_FRAMEWORK CsoundLib64
        HINTS
            ${CSOUND_FRAMEWORK_DIR_HINT}
            /Applications/Csound/CsoundLib64.framework
            /Library/Frameworks/CsoundLib64.framework
            "$ENV{HOME}/Library/Frameworks/CsoundLib64.framework"
    )
    find_library(CSOUND_LIBRARY
        NAMES CsoundLib64 CsoundLib
        HINTS
            ${CSOUND_LIBRARY_DIR_HINT}
            /Applications/Csound
            /Library/Frameworks
            "$ENV{HOME}/Library/Frameworks"
    )
else()
    find_path(CSOUND_INCLUDE_DIR plugin.h
        PATH_SUFFIXES csound
        HINTS
            ${CSOUND_INCLUDE_DIR_HINT}
            /usr/local/include
            /usr/include
            "$ENV{ProgramFiles}/Csound7/include"
            "$ENV{ProgramW6432}/Csound7/include"
    )
    find_library(CSOUND_LIBRARY
        NAMES csound64 csound
        PATH_SUFFIXES lib lib64
        HINTS
            ${CSOUND_LIBRARY_DIR_HINT}
            /usr/local/lib
            /usr/lib
            "$ENV{ProgramFiles}/Csound7/lib"
            "$ENV{ProgramW6432}/Csound7/lib"
            "$ENV{ProgramFiles}/Csound7/bin"
            "$ENV{ProgramW6432}/Csound7/bin"
    )
endif()

if(CSOUND_INCLUDE_DIR AND EXISTS "${CSOUND_INCLUDE_DIR}/version.h")
    file(STRINGS "${CSOUND_INCLUDE_DIR}/version.h" _csound_version_major
        REGEX "^#define[ \t]+CS_VERSION[ \t]+\\(([0-9]+)\\)")
    file(STRINGS "${CSOUND_INCLUDE_DIR}/version.h" _csound_version_minor
        REGEX "^#define[ \t]+CS_SUBVER[ \t]+\\(([0-9]+)\\)")
    file(STRINGS "${CSOUND_INCLUDE_DIR}/version.h" _csound_version_patch
        REGEX "^#define[ \t]+CS_PATCHLEVEL[ \t]+\\(([0-9]+)\\)")

    if(_csound_version_major)
        string(REGEX REPLACE "^#define[ \t]+CS_VERSION[ \t]+\\(([0-9]+)\\)" "\\1"
            _csound_major "${_csound_version_major}")
    endif()
    if(_csound_version_minor)
        string(REGEX REPLACE "^#define[ \t]+CS_SUBVER[ \t]+\\(([0-9]+)\\)" "\\1"
            _csound_minor "${_csound_version_minor}")
    endif()
    if(_csound_version_patch)
        string(REGEX REPLACE "^#define[ \t]+CS_PATCHLEVEL[ \t]+\\(([0-9]+)\\)" "\\1"
            _csound_patch "${_csound_version_patch}")
    endif()

    if(DEFINED _csound_major AND NOT "${_csound_major}" STREQUAL ""
       AND DEFINED _csound_minor AND NOT "${_csound_minor}" STREQUAL ""
       AND DEFINED _csound_patch AND NOT "${_csound_patch}" STREQUAL "")
        set(CSOUND_VERSION "${_csound_major}.${_csound_minor}.${_csound_patch}")
    elseif(DEFINED _csound_major AND NOT "${_csound_major}" STREQUAL ""
           AND DEFINED _csound_minor AND NOT "${_csound_minor}" STREQUAL "")
        set(CSOUND_VERSION "${_csound_major}.${_csound_minor}")
    endif()

    unset(_csound_version_major)
    unset(_csound_version_minor)
    unset(_csound_version_patch)
    unset(_csound_major)
    unset(_csound_minor)
    unset(_csound_patch)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Csound
    REQUIRED_VARS CSOUND_INCLUDE_DIR
    VERSION_VAR CSOUND_VERSION
)
mark_as_advanced(CSOUND_INCLUDE_DIR CSOUND_LIBRARY CSOUND_FRAMEWORK)

if(CSOUND_FOUND)
    set(CSOUND_INCLUDE_DIRS "${CSOUND_INCLUDE_DIR}")
    if(CSOUND_LIBRARY)
        set(CSOUND_LIBRARIES "${CSOUND_LIBRARY}")
    else()
        set(CSOUND_LIBRARIES "")
    endif()
endif()
