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

#include <generator/DiscreteCumulativeDistribution.h>
#include <generator/RandGenMersenneTwister.h>
#include <regression_test/DiscreteCumulativeDistributionRegressionTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace regressionTest
{
	void DiscreteCumulativeDistributionRegressionTest::SetUp()
	{
	  _testCDF = new sparten::DiscreteCumulativeDistribution<double, int32_t>();
	}

	void DiscreteCumulativeDistributionRegressionTest::TearDown()
	{
	  delete _testCDF;
	}

	void DiscreteCumulativeDistributionRegressionTest::create_hist_vector(Kokkos::View<double*> &v, int32_t size)
	{
	  double dTotalWeight = 0.0;
	  for (int32_t i = 0; i < size; ++i)
	  {
	    dTotalWeight += v(i);
	  }
	  for (int32_t i = 0; i < size; ++i)
	  {
	    v(i) = v(i) / dTotalWeight;
	  }
	}

	void DiscreteCumulativeDistributionRegressionTest::create_hist_matrix(Kokkos::View<double**> &a, int32_t sizeM, int32_t sizeN)
	{
	  for (int32_t j = 0; j < sizeN; ++j)
	  {
	    double dTotalWeight = 0.0;
	    for (int32_t i = 0; i < sizeM; ++i)
	    {
	      dTotalWeight += a(i, j);
	    }
	    for (int32_t i = 0; i < sizeM; ++i)
	    {
	      a(i, j) = a(i, j) / dTotalWeight;
	    }
	  }
	}
} // regressionTest
