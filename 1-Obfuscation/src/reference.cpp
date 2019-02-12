#include <stdint.h>
#include <iostream>

#include "obfuscate.hpp"

template<typename T>
void compare_images(T *buffer_1, T *buffer_2, uint32_t width, uint32_t height)
{
    int errors = 0;
    bool first = true;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int idx = j * width + i;

            if (buffer_1[idx] != buffer_2[idx]) {
                if (first) {
                    first = false;
                    std::cout << "First error found on line " << j << ", column " << i <<
                        ". Expected " << (int) buffer_1[idx] << ", found " << (int) buffer_2[idx] << ".\n";
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        std::cout << "Congrats, the images match!\n";
    }
}

template void compare_images<uint8_t>(uint8_t *, uint8_t *, uint32_t, uint32_t);
template void compare_images<uint32_t>(uint32_t *, uint32_t *, uint32_t, uint32_t);

void reference_exercise_1(uint8_t *in_buffer_1, uint8_t *in_buffer_2, uint32_t width, uint32_t height, uint8_t *out_buffer)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            const int index = j * width + i;
            out_buffer[index] = in_buffer_1[index] + in_buffer_2[index];
        }
    }
}

void reference_exercise_2(uint32_t *in_buffer_1, uint32_t *in_buffer_2, uint32_t width, uint32_t height, uint32_t *out_buffer)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            const int index = j * width + i;
            uint32_t val_1 = in_buffer_1[index];
            uint32_t val_2 = in_buffer_2[index];
            out_buffer[index] = make_RGB(get_R(val_1) + get_R(val_2),
                    get_G(val_1) + get_G(val_2),
                    get_B(val_1) + get_B(val_2));
        }
    }
}
