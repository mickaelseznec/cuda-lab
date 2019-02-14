#include <stdint.h>
#include <errno.h>
#include <string>
#include <stdlib.h>

#include <tiffio.h>

int read_tiff_grayscale(const std::string &filename, uint8_t **image_buffer, uint32_t *image_width, uint32_t *image_height)
{
    TIFF *tiff_file = TIFFOpen(filename.c_str(), "r");

    if (tiff_file == NULL) {
        return ENOENT;
    }

    if (image_width != NULL) {
        TIFFGetField(tiff_file, TIFFTAG_IMAGEWIDTH, image_width);
    }
    if (image_height != NULL) {
        TIFFGetField(tiff_file, TIFFTAG_IMAGELENGTH, image_height);
    }

    tsize_t tiff_ss = TIFFStripSize(tiff_file);
    *image_buffer = (uint8_t *) _TIFFmalloc(sizeof(uint8_t) * tiff_ss * TIFFNumberOfStrips(tiff_file));;

    if (*image_buffer == NULL) {
        TIFFClose(tiff_file);
        return ENOMEM;
    }

    for (int strip = 0; strip < TIFFNumberOfStrips(tiff_file); strip++) {
        TIFFReadEncodedStrip(tiff_file, strip, *image_buffer + strip * tiff_ss, tiff_ss);
    }

    TIFFClose(tiff_file);

    return 0;
}

int write_tiff_grayscale(const std::string &filename, uint8_t *out_buffer, const uint32_t image_width, const uint32_t image_height)
{
    TIFF *out_file = TIFFOpen(filename.c_str(), "w");
    TIFFSetField(out_file, TIFFTAG_IMAGEWIDTH, image_width);
    TIFFSetField(out_file, TIFFTAG_IMAGELENGTH, image_height);
    TIFFSetField(out_file, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(out_file, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(out_file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(out_file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out_file, image_width * sizeof(uint32_t)));

    for (uint32_t row = 0; row < image_height; row++) {
        TIFFWriteScanline(out_file, &out_buffer[row*image_width], row, 0);
    }

    TIFFClose(out_file);
}

int read_tiff_rgba(const std::string &filename, uint32_t **image_buffer, uint32_t *image_width, uint32_t *image_height)
{
    TIFF *tiff_file = TIFFOpen(filename.c_str(), "r");

    if (tiff_file == NULL) {
        return ENOENT;
    }

    uint32_t image_width__, image_height__;
    TIFFGetField(tiff_file, TIFFTAG_IMAGEWIDTH, &image_width__);
    TIFFGetField(tiff_file, TIFFTAG_IMAGELENGTH, &image_height__);

    if (image_width != NULL) {
        *image_width = image_width__;
    }
    if (image_height != NULL) {
        *image_height = image_height__;
    }

    *image_buffer = (uint32_t *) _TIFFmalloc(image_width__ * image_height__ * sizeof(uint32_t));


    if (*image_buffer == NULL) {
        return ENOMEM;
    }

    TIFFReadRGBAImage(tiff_file, image_width__, image_height__, *image_buffer, 0);
    TIFFClose(tiff_file);

    return 0;
}

int write_tiff_rgba(const std::string &filename, uint32_t *out_buffer, const uint32_t image_width, const uint32_t image_height)
{
    TIFF *out_file = TIFFOpen(filename.c_str(), "w");
    TIFFSetField(out_file, TIFFTAG_IMAGEWIDTH, image_width);
    TIFFSetField(out_file, TIFFTAG_IMAGELENGTH, image_height);
    TIFFSetField(out_file, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(out_file, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(out_file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(out_file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out_file, image_width * sizeof(uint32_t)));

    for (uint32_t row = 0; row < image_height; row++) {
        TIFFWriteScanline(out_file, &out_buffer[(image_height - 1 - row)*image_width], row, 0);
    }

    TIFFClose(out_file);

    return 0;
}
