#include <stdint.h>

#include "grayscale.hpp"

void reference_grayscale(uint32_t *in_buffer, uint32_t width, uint32_t height, uint32_t *out_buffer)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            const int index = j * width + i;
            uint8_t value = get_R(in_buffer[index]) * 0.299 +
                get_G(in_buffer[index]) * .587 +
                get_B(in_buffer[index]) * .114;
            out_buffer[index] = (0xFF << 24) + (value << 16) + (value << 8) + value;
        }
    }
}
