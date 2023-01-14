//@HEADER
// ************************************************************************
//     SparTen: Software for Sparse Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
// ************************************************************************
//@HEADER

/*!
 \file
 \brief
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

class SpartenEnvironment : public ::testing::Environment
{
private:
  int& argc;
  char** argv;

public:
  SpartenEnvironment(int& _argc, char** _argv) : argc(_argc), argv(_argv) {}

  virtual void SetUp()
  {
    Kokkos::initialize(argc, argv);
    Kokkos::print_configuration(std::cout);
  }

  virtual void TearDown()
  {
    Kokkos::finalize();
  }
};

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  SpartenEnvironment* spenv = new SpartenEnvironment(argc, argv);
  ::testing::Environment* const foo_env = ::testing::AddGlobalTestEnvironment(spenv);

  return RUN_ALL_TESTS();
}
