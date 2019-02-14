#pragma once

#include <stdint.h>
#include <errno.h>
#include <string>

#include <tiffio.h>

int read_tiff_grayscale(const std::string &filename, uint8_t **image_buffer, uint32_t *image_width=NULL, uint32_t *image_height=NULL);
int write_tiff_grayscale(const std::string &filename, uint8_t *image_buffer, const uint32_t image_width, const uint32_t image_height);

int read_tiff_rgba(const std::string &filename, uint32_t **image_buffer, uint32_t *image_width=NULL, uint32_t *image_height=NULL);
int write_tiff_rgba(const std::string &filename, uint32_t *out_buffer, const uint32_t image_width, const uint32_t image_height);
