//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <cstdlib>

#include "reference_calc.h"
#include "compare.h"

//include the definitions of the above functions for this homework
#include "HW2.cpp"


/*******  DEFINED IN student_func.cu *********/

void your_gaussian_blur_version_0(const uchar4 * const h_inputImageRGBA,
        uchar4 * const d_inputImageRGBA,
        uchar4* const d_outputImageRGBA,
        const size_t numRows, const size_t numCols,
        unsigned char *d_redBlurred,
        unsigned char *d_greenBlurred,
        unsigned char *d_blueBlurred,
        const int filterWidth);
void your_gaussian_blur_version_1(const uchar4 * const h_inputImageRGBA,
        uchar4 * const d_inputImageRGBA,
        uchar4* const d_outputImageRGBA,
        const size_t numRows, const size_t numCols,
        unsigned char *d_redBlurred,
        unsigned char *d_greenBlurred,
        unsigned char *d_blueBlurred,
        const int filterWidth);
void your_gaussian_blur_version_2(const uchar4 * const h_inputImageRGBA,
        uchar4 * const d_inputImageRGBA,
        uchar4* const d_outputImageRGBA,
        const size_t numRows, const size_t numCols,
        unsigned char *d_redBlurred,
        unsigned char *d_greenBlurred,
        unsigned char *d_blueBlurred,
        const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
        const float* const h_filter, const size_t filterWidth);


/*******  Begin main *********/

int main(int argc, char **argv) {
    uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
    uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
    unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

    float *h_filter;
    int    filterWidth;

    std::string input_file;
    std::string output_file;
    std::string reference_file;
    double perPixelError = 0.0;
    double globalError   = 0.0;
    bool useEpsCheck = false;
    int version = 0;
    switch (argc)
    {
        case 3:
            version = atoi(argv[1]);
            input_file = std::string(argv[2]);
            break;
        case 5:
            useEpsCheck=true;
            version = atoi(argv[1]);
            input_file  = std::string(argv[2]);
            perPixelError = atof(argv[3]);
            globalError   = atof(argv[4]);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " version input_file [{perPixelError, globalError}]"
                << std::endl;
            exit(1);
    }
    output_file = "convolution_output.tiff";
    reference_file = "convolution_reference.tiff";

    //load the image and give us our input and output pointers
    preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
            &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
            &h_filter, &filterWidth, input_file);

    allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
    GpuTimer timer;
    timer.Start();
    //call the students' code
    switch (version)
    {
        case 2:
            your_gaussian_blur_version_2(h_inputImageRGBA,
                    d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                    d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
            break;
        case 1:
            your_gaussian_blur_version_1(h_inputImageRGBA,
                    d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                    d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
            break;
        case 0:
        default:
            your_gaussian_blur_version_0(h_inputImageRGBA,
                    d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                    d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
    }
    timer.Stop();
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
        //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

    //check results and output the blurred image

    size_t numPixels = numRows()*numCols();
    //copy the output back to the host
    checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

    if (numPixels < 1e6) {
        postProcess(output_file, h_outputImageRGBA);

        referenceCalculation(h_inputImageRGBA, h_outputImageRGBA,
                numRows(), numCols(),
                h_filter, filterWidth);

        postProcess(reference_file, h_outputImageRGBA);

        compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);
    } else {
        std::cout << "Skipping image verification, image too large." << std::endl;
    }

    checkCudaErrors(cudaFree(d_redBlurred));
    checkCudaErrors(cudaFree(d_greenBlurred));
    checkCudaErrors(cudaFree(d_blueBlurred));

    cleanUp();

    return 0;
}
