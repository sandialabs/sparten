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

/*!
 \file
 \brief Class declaration for sparse tensor testing
 */

#pragma once

#include <sparten/SparseTensor.h>
#include <sparten/spartenUtils.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

namespace unitTest
{

//! Sparse Tensor test fixture
class SparseTensorUnitTest : public ::testing::Test
{
public:
  virtual void SetUp();       //!<
  virtual void TearDown();    //!<

protected:
  sparten::type::SubIdx _nDim;                                                                //!<
  sparten::type::ElemIdx _nElement;                                                           //!<
  std::vector<sparten::type::SubIdx> _sizeDim;                                                //!<
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *_testTensor;  //!<

  sparten::type::SubIdx _nDimSmall;                                                                //!<
  sparten::type::ElemIdx _nElementSmall;                                                           //!<
  std::vector<sparten::type::SubIdx> _sizeDimSmall;                                                //!<
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *_smallTensor;  //!<
};

} // unitTest
