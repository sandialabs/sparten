# SparTen: Software for Sparse Tensor Decompositions

SparTen provides capabilities for computing reduced-dimension
representations of sparse multidimensional count value data. The
software consists of the data decompositions methods described in
published journal papers. These decomposition methods consist of
several numerical optimization methods (one based on a multiplicative
update iterative approach, one based on quasi-Newton optimization, and
one based on damped Newton optimization) for fitting the input data to
a reduced-dimension model of the data with the lowest amount of
error. The software also consists of generalized computation that
leverages the Kokkos hardware abstraction library to compute the
reduced-data representations on multiple computer architectures,
including multicore and GPU systems.

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
1. GCC 4.8.5 or later (requires C++11)
2. CMake
3. Kokkos 
4. Googletest (optional)
5. Doxygen (optional)
6. LCOV, GCOV (optional)


## Build Instructions
The instructions below assume a directory structure similar to the following.
For concreteness, assume we will building an optimized version of the code
using GNU compilers and OpenMP parallelism.

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

Of course that structure isn't required, but modifying it will require
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
3. Configure (using CMake): ```cmake -DKokkos_ENABLE_CUDA=ON  -DKokkos_ARCH_PASCAL61=ON ../../sparten/.```
4. Build: ```make```
5. Test: ```make test```

NOTE: See the list of supported architectures in Kokkos and the corresponding 
```Kokkos_ARCH_*``` parameters [here](https://github.com/kokkos/kokkos/wiki/Compiling#architecture-keywords) 

## Generating Reference Documentation

Additional dependencies: doxygen

1. Navigate to the ```build/serial``` directory.
2. Configure (using CMake): ```cmake ../../sparten/.```
3. Build: ```make```
4. Generate: ```make doc```
