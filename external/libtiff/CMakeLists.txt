cmake_minimum_required(VERSION 3.5)

project(libtiff-wrapper NONE)

include(ExternalProject)
ExternalProject_Add(BUILD_LIBTIFF
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG v4.0.10
    PREFIX libtiff-build

    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}
)
