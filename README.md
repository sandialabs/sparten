# SparTen: Software for Sparse Tensor Decompositions

SparTen is a set of C++ tools that provide capabilities for generating
sparse count tensor data and computing low-rank canonical polyadic
(CP) decompositions.

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

**Main point of contact:** Danny Dunlavy (dmdunla@sandia.gov)

## Downloading SparTen git submodules

SparTen includes git submodules that must be retrieved as follows 
before building SparTen:

```
git submodule update --init --recursive
```

## Building SparTen

See [BUILD.md](BUILD.md) for building serial, OpenMP, or NVIDIA GPU
versions.

## Running SparTen

Examples below assume you are running SparTen from a directory where
you built SparTen using the instructions above.

### Getting help

```
./bin/Sparten_main --help
```

### Using example test data

```
./bin/Sparten_main \
    --rank 3 \
    --input $PWD/test/data/cpapr_test_10x10x10_1e+06/tensor.txt \
    --output $PWD/cpapr_test_10x10x10.ktns
```

### Creating and using randomly generated tensor data

Create data:

```
./bin/Sparten_tensor_gen \
    --num-components 5 \
    --max-num-nonzeros 100 \
    --dim-sizes "10,20,30" \
    --sptensor-output-file $PWD/cpapr_10x20x30_100.tns \
    --ktensor-output-file $PWD/cpapr_10x20x30_100.gen.ktns
```

Run SparTen:

```
./bin/Sparten_main \
    --rank 3 \
    --input $PWD/cpapr_10x20x30_100.tns \
    --output $PWD/cpapr_10x20x30_100.gen.ktns
```

## Citing SparTen

If you use SparTen in your work, please cite the following:

Keita Teranishi, Daniel M. Dunlavy, Jeremy M. Myers and Richard F. Barrett, 
"SparTen: Leveraging Kokkos for On-node Parallelism in a Second-Order Method for Fitting Canonical Polyadic Tensor Models to Poisson Data," 
2020 IEEE High Performance Extreme Computing Conference (HPEC), 
Waltham, MA, USA, 2020, pp. 1-7, 
https://doi.org/10.1109/HPEC43674.2020.9286251.

```
@INPROCEEDINGS{SparTen,
  author={Teranishi, Keita and Dunlavy, Daniel M. and Myers, Jeremy M. and Barrett, Richard F.},
  booktitle={2020 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={SparTen: Leveraging Kokkos for On-node Parallelism in a Second-Order Method for Fitting Canonical Polyadic Tensor Models to Poisson Data}, 
  year={2020},
  pages={1-7},
  doi={10.1109/HPEC43674.2020.9286251}}
```
