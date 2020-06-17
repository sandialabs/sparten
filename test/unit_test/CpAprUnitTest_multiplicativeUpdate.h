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

#include <sparten/Dimension.h>
#include <sparten/spartenUtils.h>
#include <sparten/CpAprMultiplicativeUpdate.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

// Temporary method for changing testing (Todo: test all of these)
using CpAprSparseValue = sparten::type::SubIdx;
using CpAprKruskalValue = double;
using CpAprElemIdx = sparten::type::ElemIdx;
using CpAprSubIdx = sparten::type::SubIdx;

//! Namespace for testing
namespace unitTest
{

//! CP-APR Test Fixture
class CpAprUnitTest_multiplicativeUpdate : public ::testing::Test
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

  sparten::type::SubIdx _nDimSmall;                                                                //!<
  sparten::type::SubIdx _nComponentSmall;                                                                //!<
  sparten::type::ElemIdx _nElementSmall;                                                           //!<
  std::vector<sparten::type::SubIdx> _sizeDimSmall;                                                //!<
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *_smallTensor;  //!<

  unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *_cpApr;  //!<
};

} // unitTest
