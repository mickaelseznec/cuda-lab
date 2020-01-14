# Assignment Part0: making it work
## Knowing about your hardware

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
4. Compare your GPU frequency with your CPU frequency (you can use the `lscpu` command to find the information). Why is the GPU still interesting for computing?


