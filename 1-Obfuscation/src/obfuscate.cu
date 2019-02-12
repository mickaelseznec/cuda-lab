#include <stdint.h>
#include <cuda.h>
#include <iostream>

#include "obfuscate.hpp"

/************************************************************************************
                                        EXERCISE 1
*************************************************************************************/

__global__
void k_cuda_exercise_1(uint8_t *d_in_buffer_1, uint8_t *d_in_buffer_2, uint32_t width, uint32_t height, uint8_t *d_out_buffer)
{
    //TODO 6: add an element coresponding to the thread index (hint: use threadIdx.x and threadIdx.y)
}

void cuda_exercise_1(uint8_t *in_buffer_1, uint8_t *in_buffer_2, uint32_t width, uint32_t height, uint8_t *out_buffer)
{
    uint8_t *d_in_buffer_1 = NULL, *d_in_buffer_2 = NULL, *d_out_buffer = NULL;
    const int size_bytes = width * height * sizeof(uint8_t);

    cudaEventRecord(start_memory);
    //TODO 1: allocate memory for d_in_buffer_2 and d_out_buffer
    CUDA_CHECK(cudaMalloc(&d_in_buffer_1, size_bytes));

    //TODO 2: copy in_buffer_2 to device
    cudaMemcpy(d_in_buffer_1, in_buffer_1, size_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start_kernel);
    //TODO 3: launch kernels with the right block dimensions
    dim3 blockSize(1, 1);
    k_cuda_exercise_1<<<1, blockSize>>>(d_in_buffer_1, d_in_buffer_2, width, height, d_out_buffer);

    cudaEventRecord(start_copyback);
    //TODO 4: copy d_out_buffer back to host

    cudaEventRecord(end);
    //TODO 5: don't forget to free all the device buffers
    cudaFree(d_in_buffer_1);

    cudaEventSynchronize(end);
}

/************************************************************************************
                                        EXERCISE 2
*************************************************************************************/

__global__
void k_cuda_exercise_2(uint32_t *d_in_buffer_1, uint32_t *d_in_buffer_2, uint32_t width, uint32_t height, uint32_t *d_out_buffer)
{
    //TODO 6: add an element coresponding to the thread index (hint: use blockIdx.x, blockIdx.y, threadIdx.x and threadIdx.y)
}

void cuda_exercise_2(uint32_t *in_buffer_1, uint32_t *in_buffer_2, uint32_t width, uint32_t height, uint32_t *out_buffer)
{
    uint32_t *d_in_buffer_1 = NULL, *d_in_buffer_2 = NULL, *d_out_buffer = NULL;
    const int size_bytes = width * height * sizeof(uint32_t);

    cudaEventRecord(start_memory);
    //TODO 1: allocate memory for d_in_buffer_2 and d_out_buffer
    CUDA_CHECK(cudaMalloc(&d_in_buffer_1, size_bytes));

    //TODO 2: copy in_buffer_2 to device
    cudaMemcpy(d_in_buffer_1, in_buffer_1, size_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start_kernel);
    //TODO 3: launch kernels with the right block and grid dimensions
    dim3 gridSize(1, 1);
    dim3 blockSize(1, 1);
    k_cuda_exercise_2<<<1, blockSize>>>(d_in_buffer_1, d_in_buffer_2, width, height, d_out_buffer);

    cudaEventRecord(start_copyback);
    //TODO 4: copy d_out_buffer back to host

    cudaEventRecord(end);
    //TODO 5: don't forget to free all the device buffers
    cudaFree(d_in_buffer_1);

    cudaEventSynchronize(end);
}

