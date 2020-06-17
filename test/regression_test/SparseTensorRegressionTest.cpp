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

#include <sparten/AsciiIO.h>
#include <regression_test/SparseTensorRegressionTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace regressionTest
{
	void SparseTensorRegressionTest::SetUp()
	{
	  _nDim = 5;
	  _nElement = 1000;
	  _sizeDim.push_back(2);
	  _sizeDim.push_back(3);
	  _sizeDim.push_back(4);
	  _sizeDim.push_back(5);
	  _sizeDim.push_back(6);
	  _testTensor = new sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx>(_nDim, _nElement, _sizeDim);
	}

	void SparseTensorRegressionTest::TearDown()
	{
	  delete _testTensor;
	}

	TEST_F(SparseTensorRegressionTest, DISABLED_test_write)
	{
	  sparten::type::SubIdx *ids = new sparten::type::SubIdx[_nDim];
	  double *diffs = new double[_nDim];

	  // Write initial file.
	  sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *asciiIO_init =
	      new sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("testSparseAscii.txt",
	          sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, 8);
	  asciiIO_init->write(*_testTensor);

	  // Read it back in.
	  sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *asciiIO = new sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(
	      "testSparseAscii.txt", sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, 8);
	  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *readTensor = NULL;
	  readTensor = asciiIO->read();

	  // Write it out again.
	  sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *asciiIO_rewrite = new sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx,
	      sparten::type::SubIdx>("testSparseAscii_output.txt", sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, 8);
	  asciiIO_rewrite->write(*readTensor);
	}
} // regressionTest
