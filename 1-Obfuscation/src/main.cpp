#include <iostream>
#include <stdint.h>
#include <string>

#include <tiffio.h>

#include "obfuscate.hpp"
#include "tiffutil.hpp"

void exercise_1(void);
void exercise_2(void);

cudaEvent_t start_memory, start_kernel, start_copyback, end;

int main(int argc, char *argv[])
{
    std::string help_string(std::string(argv[0]) + " (exercise_1 | exercise_2)");

    if (argc != 2) {
        std::cout << help_string << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaEventCreate(&start_memory); cudaEventCreate(&start_kernel);
    cudaEventCreate(&start_copyback); cudaEventCreate(&end);

    std::string exercise(argv[1]);
    if (exercise == "exercise_1") {
        exercise_1();
    } else if (exercise == "exercise_2") {
        exercise_2();
    } else {
        std::cout << help_string << std::endl;
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);

}

void exercise_1(void)
{
    const std::string fragment_1("../data/small_frag_1.tif");
    const std::string fragment_2("../data/small_frag_2.tif");
    const std::string sum("../data/small_sum.tif");

    uint8_t *fragment_buffer_1, *fragment_buffer_2;
    uint32_t image_width, image_height;

    read_tiff_grayscale(fragment_1, &fragment_buffer_1, &image_width, &image_height);
    read_tiff_grayscale(fragment_2, &fragment_buffer_2);

    uint8_t *out_buffer = (uint8_t *) _TIFFmalloc(image_width * image_height * sizeof(uint8_t));
    cuda_exercise_1(fragment_buffer_1, fragment_buffer_2, image_width, image_height, out_buffer);

    float memory_1, kernel, memory_2;
    cudaEventElapsedTime(&memory_1, start_memory, start_kernel);
    cudaEventElapsedTime(&kernel, start_kernel, start_copyback);
    cudaEventElapsedTime(&memory_2, start_copyback, end);

    float total = memory_1 + kernel + memory_2;

    std::cout << "GPU version took:\t" << memory_1 << "ms for allocation and copy.\n"
        "\t\t\t" << kernel << "ms for kernel execution and\n"
        "\t\t\t" << memory_2 << "ms to copy back the data.\n"
        "Total: " << total << "ms (" << 100 * (memory_1 + memory_2) / total << "% for memory management).\n\n";

    uint8_t *ref_sum_buffer = (uint8_t *) _TIFFmalloc(image_width * image_height * sizeof(uint8_t));
    cudaEventRecord(start_kernel);
    reference_exercise_1(fragment_buffer_1, fragment_buffer_2, image_width, image_height, ref_sum_buffer);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel, start_kernel, end);
    std::cout << "CPU version took:\t" << kernel << "ms.\n";

    compare_images(ref_sum_buffer, out_buffer, image_width, image_height);

    write_tiff_grayscale(sum, out_buffer, image_width, image_height);

    _TIFFfree(fragment_buffer_1);
    _TIFFfree(fragment_buffer_2);
    _TIFFfree(ref_sum_buffer);
    _TIFFfree(out_buffer);
}

void exercise_2(void)
{
    const std::string fragment_1("../data/big_frag_1.tif");
    const std::string fragment_2("../data/big_frag_2.tif");
    const std::string sum("../data/big_sum.tif");

    uint32_t image_width, image_height;
    uint32_t *fragment_buffer_1, *fragment_buffer_2;

    read_tiff_rgba(fragment_1, &fragment_buffer_1, &image_width, &image_height);
    read_tiff_rgba(fragment_2, &fragment_buffer_2);

    uint32_t *out_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    cuda_exercise_2(fragment_buffer_1, fragment_buffer_2, image_width, image_height, out_buffer);

    float memory_1, kernel, memory_2;
    cudaEventElapsedTime(&memory_1, start_memory, start_kernel);
    cudaEventElapsedTime(&kernel, start_kernel, start_copyback);
    cudaEventElapsedTime(&memory_2, start_copyback, end);

    float total = memory_1 + kernel + memory_2;

    std::cout << "GPU version took:\t" << memory_1 << "ms for allocation and copy.\n"
        "\t\t\t" << kernel << "ms for kernel execution and\n"
        "\t\t\t" << memory_2 << "ms to copy back the data.\n"
        "Total: " << total << "ms (" << 100 * (memory_1 + memory_2) / total << "% for memory management).\n\n";


    uint32_t *ref_sum_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    cudaEventRecord(start_kernel);
    reference_exercise_2(fragment_buffer_1, fragment_buffer_2, image_width, image_height, ref_sum_buffer);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel, start_kernel, end);
    std::cout << "CPU version took:\t" << kernel << "ms.\n";

    compare_images(ref_sum_buffer, out_buffer, image_width, image_height);

    write_tiff_rgba(sum, out_buffer, image_width, image_height);

    _TIFFfree(fragment_buffer_1);
    _TIFFfree(fragment_buffer_2);
    _TIFFfree(ref_sum_buffer);
    _TIFFfree(out_buffer);
}
