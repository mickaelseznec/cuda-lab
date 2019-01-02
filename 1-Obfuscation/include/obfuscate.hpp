#pragma once

#include <stdint.h>
#include <cuda.h>

#define CUDA_CHECK(err) \
do {\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error here: " << __FILE__ << ":" << __LINE__ << " "\
            << cudaGetErrorString(err) << std::endl;\
    }\
} while (0)


void cuda_adder(uint8_t *in_buffer_1, uint8_t *in_buffer_2, uint32_t width, uint32_t height, uint8_t *out_buffer);
void reference_adder(uint8_t *in_buffer_1, uint8_t *in_buffer_2, uint32_t width, uint32_t height, uint8_t *out_buffer);
void compare_images(uint8_t *buffer_1, uint8_t *buffer_2, uint32_t width, uint32_t height);

static inline uint8_t get_R(uint32_t value) {return value & 0x000000FF;}
static inline uint8_t get_G(uint32_t value) {return (value & 0x0000FF00) >> 8;}
static inline uint8_t get_B(uint32_t value) {return (value & 0x00FF0000) >> 16;}
