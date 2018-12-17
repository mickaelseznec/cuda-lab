cmake_minimum_required(VERSION 3.5)

project(obfuscation)

add_executable(obfuscation  src/main.cpp
                            src/reference.cpp
                            )
target_include_directories(obfuscation PUBLIC include)
SET(CMAKE_CXX_FLAGS "-std=c++11")
SET(LIBTIFF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external/libtiff)

find_library(LIBTIFF "libtiff" ../external/libtiff/libtiff)
find_path(TIFF_H "tiff.h" ../external/libtiff/libtiff)

if(NOT (LIBTIFF AND TIFF_H))
    message("Unable to find libtiff, let's build our own")

    execute_process(COMMAND git submodule update --init -- ../external/libtiff
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(../external/libtiff ../../external/libtiff)

    find_library(LIBTIFF "libtiff" ../external/libtiff/libtiff)
    find_path(TIFF_H "tiff.h" ../external/libtiff/libtiff)
endif()

target_link_libraries(obfuscation PUBLIC ${LIBTIFF})
target_include_directories(obfuscation PUBLIC ${TIFF_H})
