#include "utils.h"
#include "tiffutil.hpp"

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
        double maxPerPixelError, double maxGlobalError)
{
    uint32_t *reference_buffer, *test_buffer;
    uint32_t image_width, image_height;
    int ret = 0;

    ret =  read_tiff_rgba(reference_filename, &reference_buffer, &image_width, &image_height);
    if (ret != 0)
        std::cerr << "Cannot read image " << reference_filename << std::endl;

    ret = read_tiff_rgba(test_filename, &test_buffer);
    if (ret != 0)
        std::cerr << "Cannot read image " << test_filename << std::endl;

    if (!useEpsCheck) {
        maxPerPixelError = 0;
    }

    uint32_t *diff_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));

    int globalError = 0;
    bool has_failed = false;
    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            const int pos = y * image_width + x;

            const uint8_t diff_R = abs(get_R(reference_buffer[pos]) - get_R(test_buffer[pos]));
            const uint8_t diff_G = abs(get_G(reference_buffer[pos]) - get_G(test_buffer[pos]));
            const uint8_t diff_B = abs(get_B(reference_buffer[pos]) - get_B(test_buffer[pos]));

            diff_buffer[pos] = make_RGB(diff_R, diff_G, diff_B);

            const int diff = diff_R + diff_G + diff_B;
            globalError += diff;

            if (!has_failed) {
                if (diff > maxPerPixelError) {
                    has_failed = true;
                    printf("%x\n", diff_buffer[pos]);
                    std::cerr << "ERROR: Difference at pixel (" << x << ", " << y
                        << ") of " << diff << std::endl;
                }
                if (globalError > maxGlobalError) {
                    has_failed = true;
                    std::cerr << "ERROR: Global difference over " << maxGlobalError << std::endl;
                }
            }
        }
    }
    if (!has_failed) {
        std::cout << "PASS: Images match, congrats!" << std::endl;
    }

    write_tiff_rgba(std::string("convolution_differenceImage.tiff"),
            diff_buffer, image_width, image_height);

    return;
}
