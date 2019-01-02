#include <stdint.h>
#include <iostream>

#include "obfuscate.hpp"

void compare_images(uint8_t *buffer_1, uint8_t *buffer_2, uint32_t width, uint32_t height)
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

void reference_adder(uint8_t *in_buffer_1, uint8_t *in_buffer_2, uint32_t width, uint32_t height, uint8_t *out_buffer)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            const int index = j * width + i;
            out_buffer[index] = (in_buffer_1[index] + in_buffer_2[index]) % 255;
        }
    }
}
