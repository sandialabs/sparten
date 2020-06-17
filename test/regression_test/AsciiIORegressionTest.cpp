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

#include <regression_test/AsciiIORegressionTest.h>
#include <sparten/Log.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace regressionTest
{
	void AsciiIORegressionTest::SetUp()
	{
	  _filename = "test/data/cpapr_test_10x10x10_1e+06/tensor.txt";
	  _nDim = 3;
	  _nElement = 892;
	  _nIndices.push_back(10);
	  _nIndices.push_back(10);
	  _nIndices.push_back(10);
	  _maxIndex = 10;
	  _precision = 16;
	  _testHeader = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(_filename, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
	  _testHeader->readHeaderData();

	  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);
	}

	void AsciiIORegressionTest::TearDown()
	{
	  delete _testHeader;
	}
} // regressionTest
