set(torch_dir ${CMAKE_CURRENT_BINARY_DIR}/../torch)
set(torch_lib_name torch)

message("first looking for lib in : ${torch_dir}")
find_library(torch_lib
  NAMES ${torch_lib_name}
  PATHS ${torch_dir}/libtorch/lib
)

function (download_library url out)
  message(STATUS "Downloading ${url} to ${out}...")
  file(DOWNLOAD
      ${url}
      ${out}/torch_cc.zip
      STATUS _torch_dl_status
  )
  list(GET _torch_dl_status 0 _torch_dl_code)
  if (NOT _torch_dl_code EQUAL 0)
    file(REMOVE "${out}/torch_cc.zip")
    message(FATAL_ERROR "Failed to download LibTorch from ${url}: ${_torch_dl_status}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf torch_cc.zip
                  WORKING_DIRECTORY ${out}
                  RESULT_VARIABLE _torch_tar_res)
  file(REMOVE "${out}/torch_cc.zip")
  if (NOT _torch_tar_res EQUAL 0)
    message(FATAL_ERROR "Failed to extract LibTorch archive from ${url}")
  endif()
endfunction()

if (DEFINED torch_version)
  message("setting torch version : ${torch_version}")
else()
  set(torch_version "2.5.1")
  message("torch version : ${torch_version}")
endif()

if (NOT torch_lib)
  set(NEEDS_DL TRUE)
else()
  set(NEEDS_DL FALSE)
  if (UNIX AND NOT APPLE)
    if (torch_lib STREQUAL "/usr/lib/libtorch.so")
      set(NEEDS_DL TRUE)
    endif()
  endif()
endif()


if (NEEDS_DL)
  message(STATUS "Downloading torch C API pre-built")
  # Download
  if (UNIX AND NOT APPLE)  # Linux
    set(torch_url "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${torch_version}%2Bcpu.zip")
    download_library(${torch_url} ${torch_dir})
  elseif (UNIX AND APPLE)  # OSX
    if (NOT IS_DIRECTORY ${torch_dir})
      if (EXISTS ${CMAKE_SOURCE_DIR}/../install/torch_ub)
        execute_process(COMMAND cp -r ${CMAKE_SOURCE_DIR}/../install/torch_ub ${torch_dir})
      elseif(DEFINED TORCH_MAC_UB_URL)
        download_library(${TORCH_MAC_UB_URL} /tmp)
        execute_process(
          COMMAND mkdir -p ${torch_dir}
          COMMAND echo $(ls /tmp)
          COMMAND mv /tmp/torch ${torch_dir}/libtorch
        )
      else()
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${torch_dir})
        set(torch_url "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${torch_version}.zip")
        download_library(${torch_url} ${torch_dir})
      endif()
    endif()
  else()

    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${torch_dir})
    download_library("https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${torch_version}%2Bcpu.zip" ${torch_dir})
  endif()
endif()

# Find the libraries again
message("${torch_dir}")
find_library(torch_lib
  NAMES ${torch_lib_name}
  PATHS ${torch_dir}/libtorch/lib
)

if (NOT torch_lib)
  message(FATAL_ERROR "torch could not be included")
endif()
