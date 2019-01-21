cmake_minimum_required(VERSION 2.8)

project(deviceQuery)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-I${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/")
cuda_add_executable(deviceQuery deviceQuery.cu)
