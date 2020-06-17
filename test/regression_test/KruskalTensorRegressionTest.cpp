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

#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/Log.h>
#include <regression_test/KruskalTensorRegressionTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace regressionTest
{
	KruskalTensorRegressionTest::KruskalTensorRegressionTest() :
	  _log(sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET))
	{
	}

	void KruskalTensorRegressionTest::SetUp()
	{
	  const sparten::type::SubIdx nComponent = 7;
	  _nComponent = nComponent;
	  const sparten::type::SubIdx nDim = 5;
	  _nDim = nDim;
	  sparten::type::SubIdx tmp[nDim] = { 6, 7, 8, 9, 10 };

	  _nPerMode = std::vector<sparten::type::SubIdx> (tmp, tmp + _nDim);
	  _testTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDim, _nComponent, _nPerMode);
	}

	void KruskalTensorRegressionTest::TearDown()
	{
	  delete _testTensor;
	}

	TEST_F(KruskalTensorRegressionTest, DISABLED_show)
	{
	  ASSERT_EQ(true, false);
	}
} // regressionTest
