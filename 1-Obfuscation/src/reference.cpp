#include <stdint.h>

#include "obfuscate.hpp"

void reference_adder(uint32_t *in_buffer_1, uint32_t *in_buffer_2, uint32_t width, uint32_t height, uint32_t *out_buffer)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            const int index = j * width + i;
            uint8_t value = (get_R(in_buffer_1[index]) + get_R(in_buffer_2[index])) % 255;
            out_buffer[index] = (0xFF << 24) + (value << 16) + (value << 8) + value;
        }
    }
}
