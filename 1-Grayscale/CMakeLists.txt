cmake_minimum_required(VERSION 3.8)

project(grayscale)
add_executable(grayscale    src/main.cpp
                            src/reference.cpp
                            )
target_include_directories(grayscale PUBLIC include)
target_compile_features(grayscale PUBLIC cxx_std_11)
target_link_libraries(grayscale PUBLIC tiff)
