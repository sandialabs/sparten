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

#include <sparten/AsciiIO.h>
#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/KruskalTensor.h>
#include <sparten/Log.h>
#include <sparten/SparseTensor.h>
#include <unit_test/CpAprUnitTest_setGet.h>
#include <generator/FactTestSetGenerator.h>
#include <generator/RandGenMersenneTwister.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace unitTest
{

void CpAprUnitTest_setGet::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  _cpAprMultiplicativeUpdate = nullptr;
  _cpAprDampedNewton = nullptr;
  _cpAprQuasiNewton = nullptr;

  _tolerance = 1.1235e-4;
  _maxOuterIter = 10543;
  _maxInnerIter = 34;
  _offset = 0.02;
  _offsetTolerance = 2.3456e-6;
  _progressInterval = 3456;
  _eps = 4.2345e-9;

  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
}

void CpAprUnitTest_setGet::TearDown()
{
  if (_cpAprMultiplicativeUpdate != nullptr)
  {
    delete _cpAprMultiplicativeUpdate;
  }

  if (_cpAprDampedNewton != nullptr)
  {
    delete _cpAprDampedNewton;
  }

  if (_cpAprQuasiNewton!= nullptr)
  {
    delete _cpAprQuasiNewton;
  }
}

TEST_F(CpAprUnitTest_setGet, get_tolerance)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_NEAR(_cpAprMultiplicativeUpdate->get_tolerance(), _tolerance, std::numeric_limits<CpAprKruskalValue>::epsilon());
}

TEST_F(CpAprUnitTest_setGet, get_max_outer_iter)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_max_outer_iter(), _maxOuterIter);
}

TEST_F(CpAprUnitTest_setGet, get_max_inner_iter)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_max_inner_iter(), _maxInnerIter);
}

TEST_F(CpAprUnitTest_setGet, get_offset)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_NEAR(_cpAprMultiplicativeUpdate->get_offset(), _offset, std::numeric_limits<CpAprKruskalValue>::epsilon());
}

TEST_F(CpAprUnitTest_setGet, offset_tolerance)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_NEAR(_cpAprMultiplicativeUpdate->get_offset_tolerance(), _offsetTolerance, std::numeric_limits<CpAprKruskalValue>::epsilon());
}

TEST_F(CpAprUnitTest_setGet, get_progress_interval)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_progress_interval(), _progressInterval);
}

TEST_F(CpAprUnitTest_setGet, get_eps)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  ASSERT_NEAR(_cpAprMultiplicativeUpdate->get_eps(), _eps, std::numeric_limits<CpAprKruskalValue>::epsilon());
}

TEST_F(CpAprUnitTest_setGet, set_and_get_tolerance)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const CpAprKruskalValue tolerance = 1e-4;
  _cpAprMultiplicativeUpdate->set_tolerance(tolerance);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_tolerance(), tolerance);
}

TEST_F(CpAprUnitTest_setGet, set_and_get_max_outer_iter)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const sparten::type::SubIdx maxOuterIter = 20000;
  _cpAprMultiplicativeUpdate->set_max_outer_iter(maxOuterIter);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_max_outer_iter(), maxOuterIter);
}

TEST_F(CpAprUnitTest_setGet, set_and_get_max_inner_iter)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const sparten::type::SubIdx maxInnerIter = 100;
  _cpAprMultiplicativeUpdate->set_max_inner_iter(maxInnerIter);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_max_inner_iter(), maxInnerIter);
}

TEST_F(CpAprUnitTest_setGet, set_and_get_offset)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const sparten::type::SubIdx offset = 5;
  _cpAprMultiplicativeUpdate->set_offset(offset);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_offset(), offset);
}

TEST_F(CpAprUnitTest_setGet, set_and_get_offset_tolerance)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const CpAprKruskalValue offsetTolerance = 1e-5;
  _cpAprMultiplicativeUpdate->set_offset_tolerance(offsetTolerance);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_offset_tolerance(), offsetTolerance);
}

TEST_F(CpAprUnitTest_setGet, set_and_get_progress_interval)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const sparten::type::SubIdx progressInterval = 50;
  _cpAprMultiplicativeUpdate->set_progress_interval(progressInterval);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_progress_interval(), progressInterval);
}

TEST_F(CpAprUnitTest_setGet, set_and_get_eps)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const CpAprKruskalValue eps = 1e-9;
  _cpAprMultiplicativeUpdate->set_eps(eps);
  ASSERT_EQ(_cpAprMultiplicativeUpdate->get_eps(), eps);
}

// Update this test to use a double type SparseTensor. (Can't get negatives with ElemIdx or SubIdx types.)
TEST_F(CpAprUnitTest_setGet, sparse_is_non_negative)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const sparten::type::SubIdx nDim = 3;
  const sparten::type::ElemIdx nElement = 30;
  std::vector<sparten::type::SubIdx> sizeDim;
  sizeDim.push_back(2);
  sizeDim.push_back(3);
  sizeDim.push_back(4);
  sparten::SparseTensor<CpAprSparseValue, CpAprElemIdx, CpAprSubIdx> spten(nDim, nElement, sizeDim);

  const CpAprElemIdx someIndex = 5;

  const CpAprSparseValue somePositiveValue = 3.14159;
  spten.set_element_data(someIndex, somePositiveValue); // Eliminate negative value.
  ASSERT_TRUE(_cpAprMultiplicativeUpdate->test_only_is_non_negative(spten)); // There is no negative values, and this should be true.
}

TEST_F(CpAprUnitTest_setGet, kruskal_is_non_negative)
{
  _cpAprMultiplicativeUpdate = new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
  const sparten::type::SubIdx nDim = 3;
  const sparten::type::SubIdx nComponent = 7;
  sparten::type::SubIdx tmp[nDim] = { 2, 3, 4 };
  std::vector<sparten::type::SubIdx> nPerMode(tmp, tmp + nDim);
  sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx> kten(nDim, nComponent, nPerMode);

  const CpAprKruskalValue negativeValue = -1.0;
  const sparten::type::ElemIdx someRowIndex = 2;
  const sparten::type::ElemIdx someColIndex = 1;
  const sparten::type::ElemIdx someComponentIndex = 2;
  kten.set_factor_matrix_element(negativeValue, someComponentIndex, someRowIndex, someColIndex);
  ASSERT_FALSE(_cpAprMultiplicativeUpdate->test_only_is_non_negative(kten)); // There is a negative value, and this should be false.
  const CpAprKruskalValue somePositiveValue = 3.14159;
  kten.set_factor_matrix_element(somePositiveValue, someComponentIndex, someRowIndex, someColIndex); // Eliminate negative value.

  ASSERT_TRUE(_cpAprMultiplicativeUpdate->test_only_is_non_negative(kten)); // There is no negative values, and this should be true.
}

} // unitTest


