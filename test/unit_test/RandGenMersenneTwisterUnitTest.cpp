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

#include <sparten/Log.h>
#include <unit_test/RandGenMersenneTwisterUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void RandGenMersenneTwisterUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  //int32_t argcTest = 1;
  //const char *argvTest[] = { "RandGenMersenneTwisterUnitTest" };
  const int32_t nnSeed = 0; /// todo put a real value here...
  _testRandGen = new sparten::RandGenMersenneTwister(nnSeed);
  _iSeed = 1;
  _testRandGen->reset(_iSeed);
}

void RandGenMersenneTwisterUnitTest::TearDown()
{
  delete _testRandGen;
}

TEST_F(RandGenMersenneTwisterUnitTest, generate_rand_double)
{
  _testRandGen->reset(_iSeed);
  auto testValue1 = _testRandGen->generate_rand_double();
  ASSERT_DOUBLE_EQ(testValue1, 0.417021998437121511);

  _testRandGen->reset(_iSeed);
  auto testValue2 = _testRandGen->generate_rand_double_inclusive();
  ASSERT_DOUBLE_EQ(testValue2, 0.417021998534217009);
}

} // unitTest
