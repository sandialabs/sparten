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
#include <unit_test/SpartenUtilsUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void SpartenUtilsUnitTest::SetUp()
{
}

void SpartenUtilsUnitTest::TearDown()
{
}

// Tests that all dimensions are iterated over, but no dereferencing used in this test
TEST_F(SpartenUtilsUnitTest, min)
{
  const int32_t a_int32_t = -212;
  const int32_t b_int32_t = -32;

  const uint32_t a_uint32_t = 212;
  const uint32_t b_uint32_t = 32;

  const int64_t a_int64_t = -(1LL << 45);
  const int64_t b_int64_t = -(1LL << 43);

  const uint64_t a_uint64_t = 1LL << 45;
  const uint64_t b_uint64_t = 1LL << 43;

  const float a_float = -212;
  const float b_float = -32;

  const double a_double = -212;
  const double b_double = -32;

  ASSERT_EQ(sparten::min(a_int32_t, b_int32_t), a_int32_t);
  ASSERT_EQ(sparten::min(b_int32_t, a_int32_t), a_int32_t);

  ASSERT_EQ(sparten::min(a_uint32_t, b_uint32_t), b_uint32_t);
  ASSERT_EQ(sparten::min(b_uint32_t, a_uint32_t), b_uint32_t);

  ASSERT_EQ(sparten::min(a_int64_t, b_int64_t), a_int64_t);
  ASSERT_EQ(sparten::min(b_int64_t, a_int64_t), a_int64_t);

  ASSERT_EQ(sparten::min(a_uint64_t, b_uint64_t), b_uint64_t);
  ASSERT_EQ(sparten::min(b_uint64_t, a_uint64_t), b_uint64_t);

  ASSERT_EQ(sparten::min(a_float, b_float), a_float);
  ASSERT_EQ(sparten::min(b_float, a_float), a_float);

  ASSERT_EQ(sparten::min(a_double, b_double), a_double);
  ASSERT_EQ(sparten::min(b_double, a_double), a_double);
}

TEST_F(SpartenUtilsUnitTest, max)
{
  const int32_t a_int32_t = -212;
  const int32_t b_int32_t = -32;

  const uint32_t a_uint32_t = 212;
  const uint32_t b_uint32_t = 32;

  const int64_t a_int64_t = -(1LL << 45);
  const int64_t b_int64_t = -(1LL << 43);

  const uint64_t a_uint64_t = 1LL << 45;
  const uint64_t b_uint64_t = 1LL << 43;

  const float a_float = -212;
  const float b_float = -32;

  const double a_double = -212;
  const double b_double = -32;

  ASSERT_EQ(sparten::max(a_int32_t, b_int32_t), b_int32_t);
  ASSERT_EQ(sparten::max(b_int32_t, a_int32_t), b_int32_t);

  ASSERT_EQ(sparten::max(a_uint32_t, b_uint32_t), a_uint32_t);
  ASSERT_EQ(sparten::max(b_uint32_t, a_uint32_t), a_uint32_t);

  ASSERT_EQ(sparten::max(a_int64_t, b_int64_t), b_int64_t);
  ASSERT_EQ(sparten::max(b_int64_t, a_int64_t), b_int64_t);

  ASSERT_EQ(sparten::max(a_uint64_t, b_uint64_t), a_uint64_t);
  ASSERT_EQ(sparten::max(b_uint64_t, a_uint64_t), a_uint64_t);

  ASSERT_EQ(sparten::max(a_float, b_float), b_float);
  ASSERT_EQ(sparten::max(b_float, a_float), b_float);

  ASSERT_EQ(sparten::max(a_double, b_double), b_double);
  ASSERT_EQ(sparten::max(b_double, a_double), b_double);
}

TEST_F(SpartenUtilsUnitTest, abs)
{
  const int32_t a_int32_t = -212;
  const int32_t b_int32_t = 32;

  const uint32_t a_uint32_t = 212;
  const uint32_t b_uint32_t = 32;

  const int64_t a_int64_t = -(1LL << 45);
  const int64_t b_int64_t = 1LL << 43;

  const uint64_t a_uint64_t = 1LL << 45;
  const uint64_t b_uint64_t = 1LL << 43;

  const float a_float = -212;
  const float b_float = 32;

  const double a_double = -212;
  const double b_double = 32;

  ASSERT_EQ(sparten::abs(a_int32_t), -a_int32_t);
  ASSERT_EQ(sparten::abs(b_int32_t), b_int32_t);

  ASSERT_EQ(sparten::abs(a_uint32_t), a_uint32_t);
  ASSERT_EQ(sparten::abs(b_uint32_t), b_uint32_t);

  ASSERT_EQ(sparten::abs(a_int64_t), -a_int64_t);
  ASSERT_EQ(sparten::abs(b_int64_t), b_int64_t);

  ASSERT_EQ(sparten::abs(a_uint64_t), a_uint64_t);
  ASSERT_EQ(sparten::abs(b_uint64_t), b_uint64_t);

  ASSERT_EQ(sparten::abs(a_float), -a_float);
  ASSERT_EQ(sparten::abs(b_float), b_float);

  ASSERT_EQ(sparten::abs(a_double), -a_double);
  ASSERT_EQ(sparten::abs(b_double), b_double);
}

} // unitTest
