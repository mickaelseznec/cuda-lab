# Assignment Part One: CUDA basics

## Preliminary questions
### Knowing about your hardware

To get you to know what hardware is installed on your machine, we will use *deviceQuery*: a sample tool given by NVIDIA in every CUDA release. To compile it, use the following steps:

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

## Obfuscation (Part I)
### Intro

Why make things simple when the can be complex? Obfuscation is the art making something hard to understand at the first sight. In this project, someone divided an image into two parts to make it impossible to understand. You goal is to recombine them to find out what was the original image.

Go in the project *1-Obfuscation* and check the images in *data*. In Linux, you can open them from the terminal with `eog file.tif`.

You will have to sum the two images to get a resulting picture.

Please the subject entirely before beginning to write.

#### Code organization

* CMakeLists.txt                  -- the build system for the project
* data                            -- input images
  * big_frag_1.tif
  * big_frag_2.tif
  * small_frag_1.tif
  * small_frag_2.tif
* include
  * obfuscate.hpp               -- header file containing some usefull macros
* src
  * main.cpp                    -- basic structure of the program, you should not need to modify it
  * obfuscate.cu                -- **here is where you should work**
  * reference.cpp               -- contains reference implementation, comparison function...

#### Build the code

Just like *deviceQuery*, this is a cmake project. Once your are in *1-Obfuscation*, create and cd to a directory named *build*. Then `cmake ..` and `make`. Every time you do a change to your code, you will only need to run `make` and `./obfuscation exercise_1` to run the latest version.

### CUDA firsts steps

In obfuscate.cu, you will have to implement the basic framework of any CUDA program:
1. Allocate buffer on the device (GPU) memory.
2. Copy input data to the GPU
3. Launch your kernel
4. Copy output data back to the CPU
5. Free the memory you allocated

Some code is already there to help you. In this exercise, you are expected to run your kernel within a **single** block. Just make it the appropriate size for your data.

Here, the image is reprensented as a contiguous array of unsigned bytes (grayscale). You just need to sum the two input images and store the result in the output image.

Feel free to use *reference.cpp* as an example for your kernel implementation. If you need details about a particular CUDA API, some documentation in also included at the root of this project. Just run `tar xzf documentation.tar.gz` to decompress it.

### Questions

* How much time does your CUDA takes? What fraction of this time is used for memory management?
* Compare to the time taken by the CPU implementation.

## Obfuscation (Part II)

Just a few changes from part I:
* the image is now RGB, stored on 32 bit-integers. Use the macros in obfuscate.hpp for your convenience.
* the image is too large to fit into a single block. You will need a strategy to divide your problem into different blocks.

Now you need to run `./obfuscation exercise_2`

### Questions

Same questions as before:
* How much time does your CUDA takes? What fraction of this time is used for memory management?
* Compare to the time taken by the CPU implementation.

And a new one:
* What is the influence of the block/grid division of the problem?
