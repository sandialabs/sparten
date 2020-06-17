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
 \brief Class declaration for Mersenne Twister random number generator testing
 */

#pragma once

#include <generator/RandGenMersenneTwister.h>
#include <sparten/spartenUtils.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

namespace unitTest
{

//! Random number generator (Mersenne Twister) test fixture
class RandGenMersenneTwisterUnitTest : public ::testing::Test
{
public:
  void SetUp();       //!<
  void TearDown();    //!<

protected:
  int32_t _iSeed;                                       //!<
  int32_t _iNumTests;                                   //!<
  sparten::RandGenMersenneTwister *_testRandGen;        //!<
};

} // unitTest
