# SparTen: Software for Sparse Tensor Decompositions

```
Sandia National Laboratories is a multimission laboratory managed
and operated by National Technology and Engineering Solutions of Sandia,
LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
U.S. Department of Energy's National Nuclear Security Administration under
contract DE-NA0003525.

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.
```

## Dependencies

- CMake v3.16 or later
- C++ compiler with C++14 support (GCC 5.3+, Clang 4.0+, etc.)
  - OpenMP support (for CPU multithreading support)
- CUDA (for GPU builds)

## Build Instructions

The instructions below assume a directory structure similar to the following.

```
top-level
| -- build
     | -- serial
     | -- openmp
     | -- gpu
| -- sparten
     | -- tpl
          | -- kokkos
          | -- googletest
```

This structure is not required, but modifying it will require
adjusting paths in the build instructions below.

### Serial Build (default)

1. Navigate to the ```build/serial``` directory.
2. Configure (using CMake): ```cmake -DKokkos_ENABLE_SERIAL=ON ../../sparten/.```
3. Build: ```make```
4. Test: ```make test```

### OpenMP Build

Additional dependencies: OpenMP

1. Navigate to the ```build/openmp``` directory.
2. Configure (using CMake): ```cmake -DKokkos_ENABLE_OPENMP=ON ../../sparten/.```
3. Build: ```make```
4. Test: ```make test```

### CUDA (GPU) Build

Additional dependencies: CUDA

1. Navigate to the ```build/gpu``` directory.
2. Set up compiler: ```export CXX=$PWD/../../sparten/tpl/kokkos/bin/nvcc_wrapper```
3. Configure (using CMake): ```cmake -DKokkos_ENABLE_CUDA=ON  -DKokkos_ARCH_AMPERE80=ON ../../sparten/.```
4. Build: ```make```
5. Test: ```make test```

NOTE: The above instructions assume you are using an NVIDIA A100 GPU (with 8.0 CUDA 
compute capability support). See the list of supported architectures in Kokkos and 
the corresponding ```Kokkos_ARCH_*``` parameters 
[here](https://github.com/kokkos/kokkos/wiki/Compiling#architecture-keywords) 

## Generating Reference Documentation

Additional dependencies: doxygen

1. Navigate to the ```build/serial``` directory.
2. Configure (using CMake): ```cmake ../../sparten/.```
3. Build: ```make```
4. Generate: ```make doc```

NOTE: This example builds the documentation for the serial build. You can run #4, 
`make doc` in any SparTen build directory.

