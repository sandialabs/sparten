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

/* Class declaration for Factor Test Set Generator testing */

#pragma once

#include <generator/FactTestSetGenerator.h>
#include <sparten/AsciiIO.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <Sparten_Configure.h>

namespace regressionTest
{
	/* Factor Matrices Test Set Generator Test Fixture */
	class FactTestSetGeneratorRegressionTest : public ::testing::Test
	{
		public:
		  void SetUp();
		  void TearDown();

		protected:
		  sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> *_testGen;
	};
} // regressionTest
