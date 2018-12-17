#include <iostream>
#include <stdint.h>
#include <string>

#include <tiffio.h>

#include "grayscale.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "Usage:\n" << argv[0] << " in_image_color.tiff\n";
        return EXIT_FAILURE;
    }

    std::string filename(argv[1]);

    TIFF *tiff_file = TIFFOpen(filename.c_str(), "r");
    if (tiff_file == nullptr) {
        return EXIT_FAILURE;
    }

    uint32_t image_width, image_height;
    TIFFGetField(tiff_file, TIFFTAG_IMAGEWIDTH, &image_width);
    TIFFGetField(tiff_file, TIFFTAG_IMAGELENGTH, &image_height);

    uint32_t *image_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    if (image_buffer == nullptr) {
        return EXIT_FAILURE;
    }
    if (!TIFFReadRGBAImage(tiff_file, image_width, image_height, image_buffer, 0)) {
        return EXIT_FAILURE;
    }

    uint32_t *ref_grayscale_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));
    reference_grayscale(image_buffer, image_width, image_height, ref_grayscale_buffer);

    std::cout << "Openning " << filename << ": " << image_width << "x" << image_height << " pixels\n";

    std::string out_filename (filename);
    const std::string new_ext ("_gray.tiff");
    out_filename.replace(out_filename.length() - 5, new_ext.length(), new_ext);
    TIFF *out_file = TIFFOpen(out_filename.c_str(), "w");
    TIFFSetField(out_file, TIFFTAG_IMAGEWIDTH, image_width);
    TIFFSetField(out_file, TIFFTAG_IMAGELENGTH, image_height);
    TIFFSetField(out_file, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(out_file, TIFFTAG_BITSPERSAMPLE, 8);
    /* TIFFSetField(out_file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT); */
    TIFFSetField(out_file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    TIFFSetField(out_file, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out_file, image_width * 4));
    for (uint32_t row = 0; row < image_height; row++) {
        TIFFWriteScanline(out_file, &ref_grayscale_buffer[(image_height - 1 - row)*image_width], row, 0);
    }

    _TIFFfree(image_buffer);
    _TIFFfree(ref_grayscale_buffer);
    TIFFClose(tiff_file);
    TIFFClose(out_file);
    return EXIT_SUCCESS;
}
