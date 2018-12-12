cmake_minimum_required(VERSION 3.5)

project(grayscale)

add_executable(grayscale    src/main.cpp
                            src/reference.cpp
                            )
target_include_directories(grayscale PUBLIC include)
SET(CMAKE_CXX_FLAGS "-std=c++11")

find_library(LIBTIFF "libtiff.so")
message(${LIBTIFF})

if(LIBTIFF)
    target_link_libraries(grayscale PUBLIC tiff)
else()
    message("Unable to find libtiff, let's build our own")

    execute_process(COMMAND git submodule update --init -- ../external/libtiff
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/libtiff ${CMAKE_CURRENT_SOURCE_DIR}/../external/libtiff-local)
    add_dependencies(grayscale tiff)

    target_include_directories(grayscale PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/libtiff/libtiff/
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/libtiff-local/libtiff/)
    target_link_libraries(grayscale PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/libtiff-local/libtiff/libtiff.dylib)
endif()
