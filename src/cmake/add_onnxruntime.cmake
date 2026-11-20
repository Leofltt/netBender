# cmake/add_onnxruntime.cmake
#
# Downloads and integrates the official Microsoft ONNX Runtime C++ release
# for macOS (Apple Silicon / Intel) and Windows.

set(ort_dir ${CMAKE_CURRENT_BINARY_DIR}/../onnxruntime)
set(ort_version "1.18.0")

if (APPLE)
  set(ort_archive_name "onnxruntime-osx-universal2-${ort_version}.tgz")
  set(ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_archive_name}")
  set(ort_extracted_dir "${ort_dir}/onnxruntime-osx-universal2-${ort_version}")
elseif (WIN32)
  set(ort_archive_name "onnxruntime-win-x64-${ort_version}.zip")
  set(ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_archive_name}")
  set(ort_extracted_dir "${ort_dir}/onnxruntime-win-x64-${ort_version}")
else() # Linux
  set(ort_archive_name "onnxruntime-linux-x64-${ort_version}.tgz")
  set(ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_archive_name}")
  set(ort_extracted_dir "${ort_dir}/onnxruntime-linux-x64-${ort_version}")
endif()

# Check if ONNX Runtime is already downloaded
if (NOT EXISTS "${ort_extracted_dir}/include/onnxruntime_cxx_api.h")
  message(STATUS "Fetching pre-built ONNX Runtime v${ort_version}...")
  file(MAKE_DIRECTORY "${ort_dir}")

  set(ort_archive_file "${ort_dir}/${ort_archive_name}")
  message(STATUS "Downloading ${ort_url} to ${ort_archive_file}...")
  file(DOWNLOAD "${ort_url}" "${ort_archive_file}" STATUS _ort_dl_status)
  list(GET _ort_dl_status 0 _ort_dl_code)
  if (_ort_dl_code EQUAL 0)
    message(STATUS "Extracting ${ort_archive_file}...")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar -xf "${ort_archive_file}"
      WORKING_DIRECTORY "${ort_dir}"
      RESULT_VARIABLE _ort_tar_res
    )
  endif()
  file(REMOVE "${ort_archive_file}")

  # Fallback to official Microsoft NuGet package if GitHub release is unavailable
  if (NOT EXISTS "${ort_extracted_dir}/include/onnxruntime_cxx_api.h")
    set(_ort_nuget_url "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ort_version}/microsoft.ml.onnxruntime.${ort_version}.nupkg")
    set(_ort_nuget_zip "${ort_dir}/ort_nuget.zip")
    set(_ort_nuget_tmp "${ort_dir}/_nuget_tmp")

    message(STATUS "GitHub ONNX Runtime release unavailable; falling back to official Microsoft NuGet package v${ort_version}...")
    file(DOWNLOAD "${_ort_nuget_url}" "${_ort_nuget_zip}" STATUS _ort_nuget_status)
    list(GET _ort_nuget_status 0 _ort_nuget_code)
    if (_ort_nuget_code EQUAL 0)
      file(REMOVE_RECURSE "${_ort_nuget_tmp}")
      file(MAKE_DIRECTORY "${_ort_nuget_tmp}")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar -xf "${_ort_nuget_zip}"
        WORKING_DIRECTORY "${_ort_nuget_tmp}"
        RESULT_VARIABLE _ort_nuget_tar_res
      )
      if (_ort_nuget_tar_res EQUAL 0 AND EXISTS "${_ort_nuget_tmp}/build/native/include/onnxruntime_cxx_api.h")
        file(MAKE_DIRECTORY "${ort_extracted_dir}/include" "${ort_extracted_dir}/lib")
        file(COPY "${_ort_nuget_tmp}/build/native/include/" DESTINATION "${ort_extracted_dir}/include")
        if (APPLE)
          set(_ort_arm64 "${_ort_nuget_tmp}/runtimes/osx-arm64/native/libonnxruntime.dylib")
          set(_ort_x64 "${_ort_nuget_tmp}/runtimes/osx-x64/native/libonnxruntime.dylib")
          set(_ort_ver_dylib "${ort_extracted_dir}/lib/libonnxruntime.${ort_version}.dylib")
          find_program(_lipo_bin lipo)
          if (_lipo_bin AND EXISTS "${_ort_arm64}" AND EXISTS "${_ort_x64}")
            execute_process(COMMAND "${_lipo_bin}" -create "${_ort_arm64}" "${_ort_x64}" -output "${_ort_ver_dylib}")
          endif()
          if (NOT EXISTS "${_ort_ver_dylib}")
            if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" AND EXISTS "${_ort_x64}")
              file(COPY "${_ort_x64}" DESTINATION "${ort_extracted_dir}/lib")
            else()
              file(COPY "${_ort_arm64}" DESTINATION "${ort_extracted_dir}/lib")
            endif()
            file(RENAME "${ort_extracted_dir}/lib/libonnxruntime.dylib" "${_ort_ver_dylib}")
          endif()
          execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "libonnxruntime.${ort_version}.dylib" "${ort_extracted_dir}/lib/libonnxruntime.dylib")
        elseif (WIN32)
          file(COPY "${_ort_nuget_tmp}/runtimes/win-x64/native/" DESTINATION "${ort_extracted_dir}/lib")
        else()
          file(COPY "${_ort_nuget_tmp}/runtimes/linux-x64/native/libonnxruntime.so" DESTINATION "${ort_extracted_dir}/lib")
          file(RENAME "${ort_extracted_dir}/lib/libonnxruntime.so" "${ort_extracted_dir}/lib/libonnxruntime.so.${ort_version}")
          execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "libonnxruntime.so.${ort_version}" "${ort_extracted_dir}/lib/libonnxruntime.so")
        endif()
      endif()
    endif()
    file(REMOVE_RECURSE "${_ort_nuget_tmp}")
    file(REMOVE "${_ort_nuget_zip}")
  endif()
endif()

if (EXISTS "${ort_extracted_dir}/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_FOUND TRUE)
  set(ONNXRUNTIME_INCLUDE_DIR "${ort_extracted_dir}/include")
  
  if (APPLE)
    set(ONNXRUNTIME_LIBRARY "${ort_extracted_dir}/lib/libonnxruntime.dylib")
  elseif (WIN32)
    set(ONNXRUNTIME_LIBRARY "${ort_extracted_dir}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${ort_extracted_dir}/lib/onnxruntime.dll")
  else()
    set(ONNXRUNTIME_LIBRARY "${ort_extracted_dir}/lib/libonnxruntime.so")
  endif()
  
  message(STATUS "ONNX Runtime located at: ${ort_extracted_dir}")
  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

  # Create imported target for clean linkage
  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  )
  if (WIN32)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    )
  endif()

  # Ensure ONNX Runtime library is mirrored in support/ alongside LibTorch for Max MSP / PureData externals
  if (EXISTS "${CMAKE_SOURCE_DIR}/support")
    if (APPLE)
      file(COPY "${ort_extracted_dir}/lib/libonnxruntime.1.18.0.dylib" DESTINATION "${CMAKE_SOURCE_DIR}/support")
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "libonnxruntime.1.18.0.dylib" "${CMAKE_SOURCE_DIR}/support/libonnxruntime.dylib")
    elseif (WIN32)
      file(COPY "${ort_extracted_dir}/lib/onnxruntime.dll" DESTINATION "${CMAKE_SOURCE_DIR}/support")
    else()
      file(COPY "${ort_extracted_dir}/lib/libonnxruntime.so.${ort_version}" DESTINATION "${CMAKE_SOURCE_DIR}/support")
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "libonnxruntime.so.${ort_version}" "${CMAKE_SOURCE_DIR}/support/libonnxruntime.so")
    endif()
  endif()
else()
  message(FATAL_ERROR "Failed to locate ONNX Runtime C++ headers in ${ort_extracted_dir}")
endif()
