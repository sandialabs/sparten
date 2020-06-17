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
 \brief Class declaration discrete cumulative distribution function
 */

#pragma once

#include <generator/DiscreteCumulativeDistribution.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

namespace unitTest
{

//! Discrete Cumulative Distribution Test Fixture
class DiscreteCumulativeDistributionUnitTest : public ::testing::Test
{
public:
  void SetUp();     //!<
  void TearDown();  //!<

  void create_hist_vector(Kokkos::View<double*> &v, int32_t size);                //!<
  void create_hist_matrix(Kokkos::View<double**> &a, int32_t sizeM, int32_t sizeN);   //!<

protected:
  sparten::DiscreteCumulativeDistribution<double, int32_t> *_testCDF;             //!<
};

} // unitTest
