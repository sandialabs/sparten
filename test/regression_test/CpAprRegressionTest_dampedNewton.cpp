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
#include <regression_test/CpAprRegressionTest_dampedNewton.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace regressionTest
{
	void CpAprRegressionTest_dampedNewton::SetUp()
	{
	  _cpApr = nullptr;

	  _tolerance        = 1e-3;
	  _maxOuterIter     = 10;
	  _maxInnerIter     = 5;
	  _offset           = 1e-2;
	  _offsetTolerance  = 1e-2;
	  _progressInterval = 2;
	  _eps              = 1e-10;

	  _cpApr = new sparten::CpAprDampedNewton<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
	}

	void CpAprRegressionTest_dampedNewton::TearDown()
	{
	  if (_cpApr != nullptr)
	  {
	    delete _cpApr;
	  }
	}
} // regressionTest