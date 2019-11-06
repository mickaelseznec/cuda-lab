cmake_minimum_required(VERSION 2.8)

project(convolution)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LIBTIFF_DIR ${ROOT_DIR}/external/libtiff)

find_package(CUDA REQUIRED)

find_library(LIBTIFF "tiff" PATHS ${LIBTIFF_DIR}/lib)
find_path(TIFF_H "tiff.h" PATHS ${LIBTIFF_DIR}/include)

if(NOT (LIBTIFF AND TIFF_H))
    message("Unable to find libtiff, let's build our own")
    execute_process(COMMAND ${CMAKE_COMMAND} .
        WORKING_DIRECTORY ${LIBTIFF_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY ${LIBTIFF_DIR})

    find_library(LIBTIFF "tiff" PATHS ${LIBTIFF_DIR}/lib)
    find_path(TIFF_H "tiff.h" PATHS ${LIBTIFF_DIR}/include)
endif()

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "-lineinfo;-I\"${CMAKE_CURRENT_SOURCE_DIR}/include\"")

add_executable(convolution src/main.cpp
    src/reference_calc.cpp
    src/compare.cpp
    ${ROOT_DIR}/common/src/tiffutil.cpp
    )
target_include_directories(convolution PUBLIC include)
target_include_directories(convolution PUBLIC ${ROOT_DIR}/common/include)
target_include_directories(convolution PUBLIC ${TIFF_H})

cuda_add_library(convolution_cuda   src/student_func.cu)

target_link_libraries(convolution PUBLIC convolution_cuda ${LIBTIFF})
