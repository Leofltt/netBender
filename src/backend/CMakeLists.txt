cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(backend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Threads REQUIRED)

# Frontend-agnostic, zero-ML real-time audio DSP primitives
add_library(netbender_dsp STATIC
  dsp/AudioRingBuffer.cpp
  dsp/Resampler.cpp
  dsp/ResamplingRingBuffer.cpp
  dsp/SafetySentry.cpp
  dsp/WorkerThread.cpp
)
set_property(TARGET netbender_dsp PROPERTY CXX_STANDARD 20)
set_property(TARGET netbender_dsp PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET netbender_dsp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(netbender_dsp PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/dsp"
)
target_link_libraries(netbender_dsp PUBLIC netbender_rtsan Threads::Threads)
if (APPLE)
  target_link_libraries(netbender_dsp PUBLIC objc)
endif()
if (WIN32)
  target_compile_definitions(netbender_dsp PUBLIC NOMINMAX)
endif()
add_library(NetBender::dsp ALIAS netbender_dsp)

# Compile ExecuTorchBackend.cpp in an isolated OBJECT library so ExecuTorch's
# vendored c10 headers and macros never collide with LibTorch's c10 headers.
add_library(executorch_backend_obj OBJECT
  ExecuTorchBackend.cpp
  sampler/ExecuTorchSamplerBackend.cpp
)
set_property(TARGET executorch_backend_obj PROPERTY CXX_STANDARD 20)
set_property(TARGET executorch_backend_obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(executorch_backend_obj PRIVATE NETBENDER_NO_TORCH_HEADERS=1)
target_include_directories(executorch_backend_obj PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/sampler"
)
target_link_libraries(executorch_backend_obj PRIVATE nlohmann_json::nlohmann_json)

if (EXECUTORCH_FOUND)
  target_compile_definitions(executorch_backend_obj PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
  target_include_directories(executorch_backend_obj PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
else()
  target_compile_definitions(executorch_backend_obj PRIVATE NETBENDER_HAS_EXECUTORCH=0)
endif()

add_library(netbender_backend STATIC
  parsing_utils.cpp
  BackendFactory.cpp
  TorchBackend.cpp
  OnnxBackend.cpp
  engine/StreamingEngine.cpp
  engine/NetBenderSession.cpp
  bending/BendingManager.cpp
  bending/LatentTerrainEngine.cpp
  bending/PromptBendingEngine.cpp
  sampler/SamplerBundleLoader.cpp
  sampler/TextTokenizer.cpp
  sampler/DiffusionSchedule.cpp
  sampler/SamplerPrompter.cpp
  sampler/SampleAnalysis.cpp
  sampler/SamplerEngine.cpp
  $<TARGET_OBJECTS:executorch_backend_obj>
)
target_include_directories(netbender_backend
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/engine"
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/bending"
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/sampler"
  PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}"
)
target_link_libraries(netbender_backend PUBLIC
  netbender_dsp
  "${TORCH_LIBRARIES}"
  onnxruntime
  nlohmann_json::nlohmann_json
)

if (EXECUTORCH_FOUND)
  target_compile_definitions(netbender_backend PUBLIC NETBENDER_HAS_EXECUTORCH=1)
  target_link_libraries(netbender_backend PUBLIC ${EXECUTORCH_LIBRARIES})
else()
  target_compile_definitions(netbender_backend PUBLIC NETBENDER_HAS_EXECUTORCH=0)
endif()

set_property(TARGET netbender_backend PROPERTY CXX_STANDARD 20)
set_property(TARGET netbender_backend PROPERTY CXX_STANDARD_REQUIRED ON)

add_library(backend ALIAS netbender_backend)
add_library(NetBender::backend ALIAS netbender_backend)




