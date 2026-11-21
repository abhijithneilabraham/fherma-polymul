# CUDA build for the batched implementation. Use an explicit architecture for CPU-only CI.
#
# A CUDA build: the CUDA language is enabled and the answer is compiled by nvcc.
# It differs from the plain C++ CMakeLists in three ways — the CUDA language,
# the device-LTO flags, and linking NVIDIA cuPQC when the challenge image
# provides it. Everything else is the same, and the platform builds it with the
# same `cmake -B build && cmake --build build`.
cmake_minimum_required(VERSION 3.24)
project(fherma_solution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release unless told otherwise: an answer measured at -O0 is not a measurement
# of the answer.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build for the GPU visible on the official runner. CPU-only compile checks
# must pass an explicit architecture, since their host has no GPU.
# CMAKE_CUDA_ARCHITECTURES stays off because its -gencode flags conflict
# with device LTO; nvcc receives -arch= directly instead.
set(FHERMA_CUDA_ARCH native CACHE STRING "nvcc -arch value")

add_executable(solution main.cpp solve.cu wide.cu core/io-copy.cpp)
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_include_directories(solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(solution PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(solution PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(solution PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${FHERMA_CUDA_ARCH};-dlto>)
target_link_options(solution PRIVATE
  $<DEVICE_LINK:-arch=${FHERMA_CUDA_ARCH};-dlto>)

# NVIDIA cuPQC, when the challenge image provides it (CUPQC_SDK_DIR is set in
# fherma/cupqc): its device BigInt library and headers. BigInt is linked because
# the wide-integer kernels are what a CUDA answer here usually reaches for; add
# cupqc-ntt, cupqc-hash or cupqc-pk the same way if you use them.
if(DEFINED ENV{CUPQC_SDK_DIR})
  target_include_directories(solution PRIVATE
    $ENV{CUPQC_SDK_DIR}/include
    $ENV{CUPQC_SDK_DIR}/include/cupqc)
  target_link_libraries(solution PRIVATE
    $ENV{CUPQC_SDK_DIR}/lib/libcupqc-bigint.a
    $ENV{CUPQC_SDK_DIR}/lib/libcupqc-ntt.a)
endif()
