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

#include <sparten/Log.h>
#include <regression_test/DimensionRegressionTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace regressionTest
{
	void DimensionRegressionTest::SetUp()
	{
	  _nDim = 7;
	}

	void DimensionRegressionTest::TearDown()
	{
	}
} // regressionTest
