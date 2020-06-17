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

/* Class declaration for CP-APR Small Problem testing */

#pragma once

#include <sparten/spartenUtils.h>
#include <sparten/CpAprDampedNewton.h>
#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/CpAprQuasiNewton.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

using CpAprSparseValue = sparten::type::SubIdx;
using CpAprKruskalValue = double;
using CpAprElemIdx = sparten::type::SubIdx;
using CpAprSubIdx = sparten::type::SubIdx;

namespace regressionTest
{
	/* CP-APR Test Fixture */
	class CpAprRegressionTest_smallProblems : public ::testing::Test
	{
		public:
		  virtual void SetUp();
		  virtual void TearDown();

		protected:
		  double _tolerance;
		  sparten::type::SubIdx _maxOuterIter;
		  sparten::type::SubIdx _maxInnerIter;
		  double _offset;
		  double _offsetTolerance;
		  sparten::type::SubIdx _progressInterval;
		  double _eps;
		  int32_t _precision;

		  sparten::CpAprMultiplicativeUpdate<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpAprMultiplicativeUpdate;
		  sparten::CpAprDampedNewton<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpAprDampedNewton;
		  sparten::CpAprQuasiNewton<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpAprQuasiNewton;

		  sparten::SparseTensor<CpAprSparseValue, CpAprElemIdx, CpAprSubIdx> *loadSparseData(std::string const &sparseInputFilename);
	};

	template<typename T, typename TP>
	void storeKruskalData(T &kruskalTensor, std::string const &kruskalOutputFilename, TP precision);
} // regressionTest
