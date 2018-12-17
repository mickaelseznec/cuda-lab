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

    uint32_t *fragment_buffer_1 = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    uint32_t *fragment_buffer_2 = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    if (fragment_buffer_1 == nullptr || fragment_buffer_2 == nullptr) {
        exit(EXIT_FAILURE);
    }
    if (!TIFFReadRGBAImage(tiff_fragment_1, image_width, image_height, fragment_buffer_1, 0) ||
            !TIFFReadRGBAImage(tiff_fragment_2, image_width, image_height, fragment_buffer_2, 0)) {
        exit(EXIT_FAILURE);
    }

    uint32_t *ref_sum_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    reference_adder(fragment_buffer_1, fragment_buffer_2, image_width, image_height, ref_sum_buffer);

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
        TIFFWriteScanline(out_file, &ref_sum_buffer[(image_height - 1 - row)*image_width], row, 0);
    }

    _TIFFfree(fragment_buffer_1);
    _TIFFfree(fragment_buffer_2);
    _TIFFfree(ref_sum_buffer);
    TIFFClose(tiff_fragment_1);
    TIFFClose(tiff_fragment_2);
    TIFFClose(out_file);
}

void exercise_2(void)
{
}

