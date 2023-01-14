## ************************************************************************
##     SparTen: Software for Sparse Tensor Decompositions
##     by Sandia National Laboratories
##
## Sandia National Laboratories is a multimission laboratory managed
## and operated by National Technology and Engineering Solutions of Sandia,
## LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
## U.S. Department of Energy's National Nuclear Security Administration under
## contract DE-NA0003525.
##
## Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
## (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
## Government retains certain rights in this software.
## ************************************************************************

===================================================
v1.1.0 (January 13, 2023)
===================================================
- Included Kokkos as a submodule (instead of static copy)
- Changed default ktensor type from float to double
- Add capability to use ktensor in file as initial guess
- Removed use of file to specify command line parameters
- Included timing and flop instrumentation
- Changed command line options to consistent formatting
- Fixed errors with logging
- Fixed testing issue when running in Docker container
- Fixed consistency issues with <iomanip>
- Fixed issues leading to warnings for CUDA builds

===================================================
v1.0.0 (June 17, 2020)
===================================================
