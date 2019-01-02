# Assignment Part One: CUDA basics

## Preliminary questions
### Knowing about your hardware

To get you to know what hardware is installed on your machine, we will use *deviceQuery*: a sample tool given by NVidia in all CUDA release. To compile it, use the following steps:

```bash
$ cd 0-deviceQuery
$ mkdir build && cd build
$ cmake ..
$ make
$ ./deviceQuery
```

Here are a few questions you will have to answer based on the ouput of *deviceQuery*:
1. How many compute units (ALU) does your GPU have?
2. Suppose that an ALU may proceed 3 floating-point operations per clock cycle. How many computations per second (expressed in FLOPS: Floating-Point Operations Per Second) can your GPU do?
3. What is the theoretical memory bandwidth of your GPU?
4. Compare your GPU frequency with your CPU frequency (you can use the `lscpu` command to find the information). Why is the GPU still interesting?

## Obfuscation
### Intro

Why make things simple when the can be complex? Obfuscation is the art making something hard to understand at the first sight. In this project, you will have to retrieve an image based on two fragments.

Go in the project *1-Obfuscation* and check the images in *data*. In Linux, you can open them from the terminal with `eog file.tif`.

You will have to sum the two images to get a resulting picture.

Here is the code organization :

* CMakeLists.txt                  -- the build system for the project
* data                            -- input images
 * big_frag_1.tif
 * big_frag_2.tif
 * small_frag_1.tif
 * small_frag_2.tif
* include
 * obfuscate.hpp               -- header file containing some usefull macros
* src
 * main.cpp
 * obfuscate.cu
 * obfuscate.cu.teacher
 * reference.cpp
