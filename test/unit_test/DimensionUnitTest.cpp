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
 \brief
 */

#include <sparten/Log.h>
#include <unit_test/DimensionUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void DimensionUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  _nDim = 7;
}

void DimensionUnitTest::TearDown()
{
}

// Tests that all dimensions are iterated over, but no dereferencing used in this test
TEST_F(DimensionUnitTest, iterateOverAllDims)
{
  const sparten::type::SubIdx nDim = 7;
  sparten::type::SubIdx dimOutput[nDim];

  memset(dimOutput, 0, nDim * sizeof(sparten::type::SubIdx));

  sparten::Dimension dim(nDim);

  sparten::type::SubIdx externalCounter = 0;
  for (auto iDim : dim)
  {
    dimOutput[externalCounter] = externalCounter;
    ++externalCounter;
  }

  for (sparten::type::SubIdx iDim = 0; iDim < nDim; ++iDim)
  {
    ASSERT_EQ(dimOutput[iDim], iDim);
  }
}

// Tests that all dimensions are iterated over, with dereferencing this time
TEST_F(DimensionUnitTest, iterateOverAllDimsDereference)
{
  const sparten::type::SubIdx nDim = 9;
  sparten::type::SubIdx dimOutput[nDim];

  memset(dimOutput, 0, nDim * sizeof(sparten::type::SubIdx));

  sparten::Dimension dim(nDim);

  for (auto iDim : dim)
  {
    dimOutput[iDim] = iDim;
  }

  for (sparten::type::SubIdx iDim = 0; iDim < nDim; ++iDim)
  {
    ASSERT_EQ(dimOutput[iDim], iDim);
  }
}

// Makes sure getter for nDim works
TEST_F(DimensionUnitTest, get_nDim)
{
  const sparten::type::SubIdx nDim = 9;

  sparten::Dimension dim(nDim);

  ASSERT_EQ(dim.get_nDim(), nDim);
}

///\todo Add test for each Dimension::Iterator public member here

} // unitTest
