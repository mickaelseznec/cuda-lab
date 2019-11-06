#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath>

#include "tiffutil.hpp"

uint32_t image_width, image_height;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return image_height; }
size_t numCols() { return image_width; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
        uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
        unsigned char **d_redBlurred,
        unsigned char **d_greenBlurred,
        unsigned char **d_blueBlurred,
        float **h_filter, int *filterWidth,
        const std::string &filename) {

    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    uint32_t *image_buffer;

    int ret = read_tiff_rgba(filename, &image_buffer, &image_width, &image_height);

    if (ret != 0) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    uint32_t *out_buffer = (uint32_t *) _TIFFmalloc(image_width * image_height * sizeof(uint32_t));

    *h_inputImageRGBA  = (uchar4 *) image_buffer;
    *h_outputImageRGBA = (uchar4 *) out_buffer;

    const size_t numPixels = numRows() * numCols();
    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4)));
    //make sure no memory is left laying around

    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    d_inputImageRGBA__  = *d_inputImageRGBA;
    d_outputImageRGBA__ = *d_outputImageRGBA;

    //now create the filter that they will use
    const int blurKernelWidth = 9;
    const float blurKernelSigma = 2.;

    *filterWidth = FILTER_WIDTH;

    //create and fill the filter we will convolve with
    *h_filter = new float[blurKernelWidth * blurKernelWidth];
    h_filter__ = *h_filter;

    float filterSum = 0.f; //for normalization

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
            (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
            filterSum += filterValue;
        }
    }

    float normalizationFactor = 1.f / filterSum;

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
        }
    }

    //blurred
    checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file, uchar4* data_ptr) {
    write_tiff_rgba(output_file, reinterpret_cast<uint32_t *>(data_ptr), image_width, image_height);
}

void cleanUp(void)
{
    cudaFree(d_inputImageRGBA__);
    cudaFree(d_outputImageRGBA__);
    delete[] h_filter__;
}

