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
#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/SparseTensor.h>
#include <unit_test/CpAprUnitTest_multiplicativeUpdate.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void CpAprUnitTest_multiplicativeUpdate::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  _cpApr = nullptr;

  _tolerance = 1e-8;
  _maxOuterIter = 20;
  _maxInnerIter = 10;
  _offset = 1e-2;
  _offsetTolerance = 1e-2;
  _progressInterval = 1;
  _eps = 1e-10;

  _cpApr= new unitTest::CpAprMultiplicativeUpdateUnitTestOnly<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);

  // Small sparse tensor in coordinate format. Size: 2 x 3 x 4
  // 0 1 0 0
  // 0 1 1 1
  // 0 1 2 2
  // 0 1 3 3
  _nDimSmall = 3;
  _nComponentSmall = 2;
  _nElementSmall = 4;
  _sizeDimSmall.push_back(2);
  _sizeDimSmall.push_back(3);
  _sizeDimSmall.push_back(4);
  _smallTensor =  new sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx>(_nDimSmall, _nElementSmall, _sizeDimSmall);

  sparten::type::SubIdx *ids = new sparten::type::SubIdx[_nDimSmall];
  ids[0] = 0;
  ids[1] = 1;
  for (auto i : *_smallTensor)
    {
      ids[2] = i;
      std::vector<sparten::type::SubIdx> tmp(ids, ids + _nDimSmall);
      _smallTensor->set_element( i, tmp, i);
    }
}

void CpAprUnitTest_multiplicativeUpdate::TearDown()
{
  if (_cpApr != nullptr)
  {
    delete _cpApr;
  }
}

TEST_F(CpAprUnitTest_multiplicativeUpdate, compute)
{
  // KruskalTensor
  auto *kTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDimSmall, _nComponentSmall, _sizeDimSmall);
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  kruskalInit.set_random_seed(1);
  kruskalInit.initialize(*kTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);

  ASSERT_EQ(_cpApr->get_isConverged(), false);
  ASSERT_EQ(_cpApr->get_nIter(), 0);
  // this should converge in two outer iterations
  _cpApr->compute(*kTensor, *_smallTensor);

  ASSERT_EQ(_cpApr->get_isConverged(), true);
  ASSERT_EQ(_cpApr->get_nIter(), 2);

  //TODO: Test the output
}

TEST_F(CpAprUnitTest_multiplicativeUpdate, protected_methods)
{
  // KruskalTensor
  auto *kTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDimSmall, _nComponentSmall, _sizeDimSmall);
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  kruskalInit.set_random_seed(1);
  kruskalInit.initialize(*kTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);

  ASSERT_EQ(_cpApr->test_only_is_non_negative(*_smallTensor), true);
  ASSERT_EQ(_cpApr->test_only_is_non_negative(*kTensor), true);
  _cpApr->compute(*kTensor, *_smallTensor);
  ASSERT_EQ(_cpApr->test_only_does_violate_karush_kuhn_tucker_condition(*kTensor), true);  
}

TEST_F(CpAprUnitTest_multiplicativeUpdate, DISABLED_check_requirements)
{
  // KruskalTensor with different nDim
  auto *kTensor_bad_nDim = new sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDimSmall+1, _nComponentSmall, _sizeDimSmall);
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  kruskalInit.set_random_seed(1);
  kruskalInit.initialize(*kTensor_bad_nDim, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);
  ASSERT_ANY_THROW(_cpApr->compute(*kTensor_bad_nDim, *_smallTensor));

  // KruskalTensor with negative value
  auto *kTensor_negative = new sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDimSmall, _nComponentSmall, _sizeDimSmall);
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit2;
  kruskalInit2.set_random_seed(1);
  kruskalInit2.initialize(*kTensor_negative, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);
  kTensor_negative->set_factor_matrix_element(-1.0, 0, 0, 0);
  ASSERT_ANY_THROW(_cpApr->compute(*kTensor_negative, *_smallTensor));

}

} // unitTest
