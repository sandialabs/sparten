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

#include <regression_test/RandGenMersenneTwisterRegressionTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>

namespace regressionTest
{

	void RandGenMersenneTwisterRegressionTest::SetUp()
	{
	  const int32_t nnSeed = 0;
	  _testRandGen = new sparten::RandGenMersenneTwister(nnSeed);
	  _iSeed = 1;
	  _testRandGen->reset(_iSeed);
	}

	void RandGenMersenneTwisterRegressionTest::TearDown()
	{
	  delete _testRandGen;
	}

	TEST_F(RandGenMersenneTwisterRegressionTest, DISABLED_matlab_equivalnce)
	{
	  double *original = new double[100];
	  double *diffs = new double[100];
	  std::ifstream input("./MatlabMT100_1.bin", std::ios::binary);
	  input.read((char *) original, sizeof(double) * 100);
	  input.close();

	  for (int32_t i = 0; i < 100; ++i)
	  {
	    diffs[i] = _testRandGen->generate_rand_double_matlab() - original[i];
	    ASSERT_EQ(diffs[i], 0);
	  }

	  delete[] original;
	  delete[] diffs;
	}

	TEST_F(RandGenMersenneTwisterRegressionTest, semiuniqueness )
	{
	  double *original = new double[100];
	  double *diffs = new double[100];

	  for (int32_t i = 0; i < 100; ++i )
	  {
	    _testRandGen->reset(i);
	    for (int32_t j = 0; j < 100; ++j)
	    {
	      original[j] = _testRandGen->generate_rand_double_matlab();
	    }
	    _testRandGen->reset(i+( rand()%100 +1 ));
	    for (int32_t j = 0; j < 100; ++j)
	    {
	      diffs[j] = _testRandGen->generate_rand_double_matlab() - original[j];
	    }
	    int32_t j=0;
	    bool result = false;
	    while( !result && j < 100 )
	    {
	      result = (diffs[j] != 0.0);
	      j++;
	    }
	    ASSERT_TRUE(result);
	  }

	  delete[] original;
	  delete[] diffs;
	}

	TEST_F(RandGenMersenneTwisterRegressionTest, reproducibility )
	{
	  double *original = new double[100];
	  double *diffs = new double[100];

	  for (int32_t i = 0; i < 100; ++i )
	  {
	    _testRandGen->reset(i);
	    for (int32_t j = 0; j < 100; ++j)
	    {
	      original[j] = _testRandGen->generate_rand_double_matlab();
	    }
	    _testRandGen->reset(i);
	    for (int32_t j = 0; j < 100; ++j)
	    {
	      diffs[j] = _testRandGen->generate_rand_double_matlab() - original[j];
	      ASSERT_EQ(diffs[j], 0);
	    }
	  }

	  delete[] original;
	  delete[] diffs;
	}
} // regressionTest
