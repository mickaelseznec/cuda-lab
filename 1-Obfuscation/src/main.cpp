#include <iostream>
#include <stdint.h>
#include <string>

#include <tiffio.h>

#include "obfuscate.hpp"

void exercise_1(void);
void exercise_2(void);

int main(int argc, char *argv[])
{
    std::string help_string(std::string(argv[0]) + " (exercise_1 | exercise_2)");

    if (argc != 2) {
        std::cout << help_string << std::endl;
        exit(EXIT_FAILURE);
    }

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

    TIFF *tiff_fragment_1 = TIFFOpen(fragment_1.c_str(), "r");
    TIFF *tiff_fragment_2 = TIFFOpen(fragment_2.c_str(), "r");
    if (tiff_fragment_1 == nullptr || tiff_fragment_2 == nullptr) {
        exit(EXIT_FAILURE);
    }

    uint32_t image_width, image_height;
    TIFFGetField(tiff_fragment_1, TIFFTAG_IMAGEWIDTH, &image_width);
    TIFFGetField(tiff_fragment_1, TIFFTAG_IMAGELENGTH, &image_height);

    tsize_t tiff_ss_1 = TIFFStripSize(tiff_fragment_1);
    tsize_t tiff_ss_2 = TIFFStripSize(tiff_fragment_2);

    uint8_t *fragment_buffer_1 = (uint8_t *) _TIFFmalloc(tiff_ss_1 * TIFFNumberOfStrips(tiff_fragment_1));
    uint8_t *fragment_buffer_2 = (uint8_t *) _TIFFmalloc(tiff_ss_2 * TIFFNumberOfStrips(tiff_fragment_2));

    if (fragment_buffer_1 == nullptr || fragment_buffer_2 == nullptr) {
        exit(EXIT_FAILURE);
    }
    for (int strip = 0; strip < TIFFNumberOfStrips(tiff_fragment_1); strip++) {
        TIFFReadEncodedStrip(tiff_fragment_1, strip, fragment_buffer_1 + strip * tiff_ss_1, tiff_ss_1);
    }
    for (int strip = 0; strip < TIFFNumberOfStrips(tiff_fragment_2); strip++) {
        TIFFReadEncodedStrip(tiff_fragment_2, strip, fragment_buffer_2 + strip * tiff_ss_2, tiff_ss_2);
    }

    uint8_t *out_buffer = (uint8_t *) _TIFFmalloc(image_width * image_height * sizeof(uint8_t));
    cuda_exercise_1(fragment_buffer_1, fragment_buffer_2, image_width, image_height, out_buffer);

    uint8_t *ref_sum_buffer = (uint8_t *) _TIFFmalloc(image_width * image_height * sizeof(uint8_t));
    reference_exercise_1(fragment_buffer_1, fragment_buffer_2, image_width, image_height, ref_sum_buffer);

    compare_images(ref_sum_buffer, out_buffer, image_width, image_height);

    TIFF *out_file = TIFFOpen(sum.c_str(), "w");
    TIFFSetField(out_file, TIFFTAG_IMAGEWIDTH, image_width);
    TIFFSetField(out_file, TIFFTAG_IMAGELENGTH, image_height);
    TIFFSetField(out_file, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(out_file, TIFFTAG_BITSPERSAMPLE, 8);
    /* TIFFSetField(out_file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT); */
    TIFFSetField(out_file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(out_file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out_file, image_width * sizeof(uint32_t)));

    for (uint32_t row = 0; row < image_height; row++) {
        TIFFWriteScanline(out_file, &out_buffer[row*image_width], row, 0);
    }

    _TIFFfree(fragment_buffer_1);
    _TIFFfree(fragment_buffer_2);
    _TIFFfree(ref_sum_buffer);
    _TIFFfree(out_buffer);
    TIFFClose(tiff_fragment_1);
    TIFFClose(tiff_fragment_2);
    TIFFClose(out_file);
}

void exercise_2(void)
{
    const std::string fragment_1("../data/big_frag_1.tif");
    const std::string fragment_2("../data/big_frag_2.tif");
    const std::string sum("../data/big_sum.tif");

    TIFF *tiff_fragment_1 = TIFFOpen(fragment_1.c_str(), "r");
    TIFF *tiff_fragment_2 = TIFFOpen(fragment_2.c_str(), "r");
    if (tiff_fragment_1 == nullptr || tiff_fragment_2 == nullptr) {
        exit(EXIT_FAILURE);
    }

    uint32_t image_width, image_height;
    TIFFGetField(tiff_fragment_1, TIFFTAG_IMAGEWIDTH, &image_width);
    TIFFGetField(tiff_fragment_1, TIFFTAG_IMAGELENGTH, &image_height);

    tsize_t tiff_ss_1 = TIFFStripSize(tiff_fragment_1);
    tsize_t tiff_ss_2 = TIFFStripSize(tiff_fragment_2);

    uint32_t *fragment_buffer_1 = (uint32_t *) _TIFFmalloc(tiff_ss_1 * TIFFNumberOfStrips(tiff_fragment_1));
    uint32_t *fragment_buffer_2 = (uint32_t *) _TIFFmalloc(tiff_ss_2 * TIFFNumberOfStrips(tiff_fragment_2));

    if (fragment_buffer_1 == nullptr || fragment_buffer_2 == nullptr) {
        exit(EXIT_FAILURE);
    }
    for (int strip = 0; strip < TIFFNumberOfStrips(tiff_fragment_1); strip++) {
        TIFFReadEncodedStrip(tiff_fragment_1, strip, fragment_buffer_1 + strip * tiff_ss_1, tiff_ss_1);
    }
    for (int strip = 0; strip < TIFFNumberOfStrips(tiff_fragment_2); strip++) {
        TIFFReadEncodedStrip(tiff_fragment_2, strip, fragment_buffer_2 + strip * tiff_ss_2, tiff_ss_2);
    }

    uint32_t *out_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    cuda_exercise_2(fragment_buffer_1, fragment_buffer_2, image_width, image_height, out_buffer);

    uint32_t *ref_sum_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    reference_exercise_2(fragment_buffer_1, fragment_buffer_2, image_width, image_height, ref_sum_buffer);

    compare_images(ref_sum_buffer, out_buffer, image_width, image_height);

    TIFF *out_file = TIFFOpen(sum.c_str(), "w");
    TIFFSetField(out_file, TIFFTAG_IMAGEWIDTH, image_width);
    TIFFSetField(out_file, TIFFTAG_IMAGELENGTH, image_height);
    TIFFSetField(out_file, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(out_file, TIFFTAG_BITSPERSAMPLE, 8);
    /* TIFFSetField(out_file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT); */
    TIFFSetField(out_file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(out_file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out_file, image_width * sizeof(uint32_t)));

    for (uint32_t row = 0; row < image_height; row++) {
        TIFFWriteScanline(out_file, &out_buffer[row*image_width], row, 0);
    }

    _TIFFfree(fragment_buffer_1);
    _TIFFfree(fragment_buffer_2);
    _TIFFfree(ref_sum_buffer);
    _TIFFfree(out_buffer);
    TIFFClose(tiff_fragment_1);
    TIFFClose(tiff_fragment_2);
    TIFFClose(out_file);
}
