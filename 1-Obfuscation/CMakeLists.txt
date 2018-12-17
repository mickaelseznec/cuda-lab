cmake_minimum_required(VERSION 3.5)

project(obfuscation)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LIBTIFF_DIR ${ROOT_DIR}/external/libtiff)

add_executable(obfuscation  src/main.cpp
                            src/reference.cpp
                            )
target_include_directories(obfuscation PUBLIC include)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_library(LIBTIFF "tiff" PATHS ${LIBTIFF_DIR}/lib)
find_path(TIFF_H "tiff.h" PATHS ${LIBTIFF_DIR}/include)

if(NOT (LIBTIFF AND TIFF_H))
    message("Unable to find libtiff, let's build our own")
    execute_process(COMMAND ${CMAKE_COMMAND} .
        WORKING_DIRECTORY ${LIBTIFF_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY ${LIBTIFF_DIR})

    find_library(LIBTIFF "libtiff" "${LIBTIFF_DIR}/lib")
    find_path(TIFF_H "tiff.h" ${LIBTIFF_DIR}/include)
endif()

target_include_directories(obfuscation PUBLIC ${TIFF_H})
target_link_libraries(obfuscation PUBLIC ${LIBTIFF})
