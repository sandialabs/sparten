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

#pragma once

#include <sparten/spartenUtils.h>
#include <unit_test/CpAprUnitTestOnly.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

// Temporary method for changing testing (Todo: test all of these)
using CpAprSparseValue = sparten::type::SubIdx;
using CpAprKruskalValue = double;
using CpAprElemIdx = sparten::type::SubIdx;
using CpAprSubIdx = sparten::type::SubIdx;

//! Namespace for testing
namespace unitTest
{

//! CP-APR Test Fixture
class CpAprUnitTest_setGet : public ::testing::Test
{
public:
  virtual void SetUp();       //!<
  virtual void TearDown();    //!<

protected:
  double _tolerance;                        //!<
  sparten::type::SubIdx _maxOuterIter;      //!<
  sparten::type::SubIdx _maxInnerIter;      //!<
  double _offset;                           //!<
  double _offsetTolerance;                  //!<
  sparten::type::SubIdx _progressInterval;  //!<
  double _eps;                              //!<

  unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpAprMultiplicativeUpdate;  //!<
  unitTest::CpAprDampedNewtonUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpAprDampedNewton;  //!<
  unitTest::CpAprQuasiNewtonUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpAprQuasiNewton;  //!<
};

} // unitTest
