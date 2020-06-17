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
 \brief Class declaration for Kurskal tensor testing
 */

#pragma once

#include <sparten/KruskalTensor.h>
#include <sparten/Log.h>
#include <sparten/spartenUtils.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

//! Kruskal Tensor Test Fixture
class KruskalTensorUnitTest : public ::testing::Test
{
public:
  KruskalTensorUnitTest();

  virtual void SetUp();       //!<
  virtual void TearDown();    //!<

protected:
  sparten::type::SubIdx _nDim;                                            //!<
  sparten::type::SubIdx _maxModeSize;                                     //!<
  sparten::type::SubIdx _nComponent;                                      //!<
  std::vector<sparten::type::SubIdx> _nPerMode;                           //!<
  sparten::KruskalTensor<double, sparten::type::SubIdx> *_testTensor;     //!<
  sparten::Log &_log;
};

} // unitTest
