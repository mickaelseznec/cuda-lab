#pragma once

#include <stdint.h>

void reference_grayscale(uint32_t *in_buffer, uint32_t width, uint32_t height, uint32_t *out_buffer);

static inline uint8_t get_R(uint32_t value) {return value & 0x000000FF;}
static inline uint8_t get_G(uint32_t value) {return (value & 0x0000FF00) >> 8;}
static inline uint8_t get_B(uint32_t value) {return (value & 0x00FF0000) >> 16;}
