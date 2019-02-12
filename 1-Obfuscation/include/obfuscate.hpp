#pragma once

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
do {\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error here: " << __FILE__ << ":" << __LINE__ << " "\
            << cudaGetErrorString(err) << std::endl;\
    }\
} while (0)


void cuda_exercise_1(uint8_t *in_buffer_1, uint8_t *in_buffer_2,
        uint32_t width, uint32_t height, uint8_t *out_buffer);
void reference_exercise_1(uint8_t *in_buffer_1, uint8_t *in_buffer_2,
        uint32_t width, uint32_t height, uint8_t *out_buffer);

void cuda_exercise_2(uint32_t *in_buffer_1, uint32_t *in_buffer_2,
        uint32_t width, uint32_t height, uint32_t *out_buffer);
void reference_exercise_2(uint32_t *in_buffer_1, uint32_t *in_buffer_2,
        uint32_t width, uint32_t height, uint32_t *out_buffer);

template <typename T>
void compare_images(T *buffer_1, T *buffer_2, uint32_t width, uint32_t height);

static inline __host__ __device__ uint8_t get_R(uint32_t value) {return value & 0x000000FF;}
static inline __host__ __device__ uint8_t get_G(uint32_t value) {return (value & 0x0000FF00) >> 8;}
static inline __host__ __device__ uint8_t get_B(uint32_t value) {return (value & 0x00FF0000) >> 16;}
static inline __host__ __device__ uint32_t make_RGB(uint8_t R, uint8_t G, uint8_t B) { return R + (G << 8) + (B << 16) + 0xFF000000;}

extern cudaEvent_t start_memory, start_kernel, start_copyback, end;
