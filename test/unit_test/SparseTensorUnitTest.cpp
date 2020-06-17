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

#include <sparten/AsciiIO.h> /// todo Remove this include.
#include <sparten/Log.h>
#include <unit_test/SparseTensorUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void SparseTensorUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  _nDim = 5;
  _nElement = 1000;
  _sizeDim.push_back(2);
  _sizeDim.push_back(3);
  _sizeDim.push_back(4);
  _sizeDim.push_back(5);
  _sizeDim.push_back(6);
  _testTensor = new sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx>(_nDim, _nElement, _sizeDim);


  // Small sparse tensor in coordinate format. Size: 2 x 3 x 4
  // 0 1 0 0
  // 0 1 1 1
  // 0 1 2 2
  // 0 1 3 3
  _nDimSmall = 3;
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

void SparseTensorUnitTest::TearDown()
{
  delete _testTensor;
  delete _smallTensor;
}

TEST_F(SparseTensorUnitTest, get_dimension)
{
  ASSERT_EQ(_testTensor->get_nDim(), _nDim);
}

TEST_F(SparseTensorUnitTest, get_nElement)
{
  ASSERT_EQ(_testTensor->get_nElement(), _nElement);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(SparseTensorUnitTest, DISABLED_set_and_get)
#else
TEST_F(SparseTensorUnitTest, set_and_get)
#endif
{
  const sparten::type::SubIdx multiplier = 8;
  sparten::type::SubIdx *ids = new sparten::type::SubIdx[_nDim];
  sparten::type::SubIdx *diffs = new sparten::type::SubIdx[_nDim];

//  for (sparten::type::ElemIdx i = 0; i < _testTensor->get_nElement(); ++i)
  for (auto i : *_testTensor)
  {
    sparten::type::SubIdx value = i * multiplier;

    ids[0] = 1 * i;
    ids[1] = 2 * i;
    ids[2] = 3 * i;
    ids[3] = 4 * i;
    ids[4] = 5 * i;
    std::vector<sparten::type::SubIdx> tmp(ids, ids + _nDim);
    _testTensor->set_element(i, tmp, value);
  }

  sparten::type::SubIdx diffValue = 0;
  memset(diffs, 0, _nDim * sizeof(sparten::type::SubIdx));

  for (auto i : *_testTensor)
  {
    diffValue += sparten::abs(i * multiplier - _testTensor->get_element_data(i));

    for (sparten::type::SubIdx j = 0; j < _testTensor->get_nDim(); ++j)
    {
      diffs[j] += sparten::abs(((j + 1) * i) - _testTensor->get_element_index(i, 0));
    }
  }

  ASSERT_EQ(diffValue, 0);

  for (auto j : _testTensor->get_dim())
  {
    ASSERT_EQ(diffs[0], 0);
  }

  delete [] ids;
  delete [] diffs;
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(SparseTensorUnitTest, DISABLED_set_element_data_to_zero)
#else
TEST_F(SparseTensorUnitTest, set_element_data_to_zero)
#endif
{
  const sparten::type::ElemIdx someIndex = 123;
  const double someNonZeroValue = 1.11;

  _testTensor->set_element_data_to_zero();

  _testTensor->set_element_data(someIndex, someNonZeroValue);

  _testTensor->set_element_data_to_zero();

  double diffValue = 0.0;

  for (auto i : *_testTensor)
  {
    diffValue += _testTensor->get_element_data(i);
  }

  ASSERT_EQ(diffValue, 0.0);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(SparseTensorUnitTest, DISABLED_deep_copy_constructor)
#else
TEST_F(SparseTensorUnitTest, deep_copy_constructor)
#endif
{
  const sparten::type::SubIdx multiplier = 10;
  sparten::type::SubIdx *ids = new sparten::type::SubIdx[_nDim];
  sparten::type::SubIdx *diffs = new sparten::type::SubIdx[_nDim];

  for (auto i : *_testTensor)
  {
    sparten::type::SubIdx value = i * multiplier;

    ids[0] = 1 * i;
    ids[1] = 2 * i;
    ids[2] = 3 * i;
    ids[3] = 4 * i;
    ids[4] = 5 * i;
    std::vector<sparten::type::SubIdx> tmp(ids, ids + _nDim);
    _testTensor->set_element( i, tmp, value);
  }

  // Deep copy.
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx> tensorCopy(*_testTensor);

  // Verify copy.
  sparten::type::SubIdx diffValue = 0;
  memset(diffs, 0, _nDim * sizeof(sparten::type::SubIdx));

  for (auto i : tensorCopy)
  {
    diffValue += std::fabs(i * multiplier - tensorCopy.get_element_data(i));

    for (auto j : tensorCopy.get_dim())
    {
      diffs[j] += sparten::abs(((j + 1) * i) - tensorCopy.get_element_index(i, 0));
    }
  }

  ASSERT_EQ(diffValue, 0.0);

  for (auto j : tensorCopy.get_dim())
  {
    ASSERT_EQ(diffs[0], 0);
  }

  delete [] ids;
  delete [] diffs;
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(SparseTensorUnitTest, DISABLED_show)
#else
TEST_F(SparseTensorUnitTest, show)
#endif
{
  std::string tensorShow = "0\t1\t0\t0\n0\t1\t1\t1\n0\t1\t2\t2\n0\t1\t3\t3\n";
  
  ASSERT_EQ(tensorShow, _smallTensor->show());
}

TEST_F(SparseTensorUnitTest, get_nrow)
{
  for (int i=0; i < _nDimSmall; i++)
    ASSERT_EQ(_smallTensor->get_nRow(i), _sizeDimSmall[i]);
}

#if defined(KOKKOS_ENABLE_CUDA)
  TEST_F(SparseTensorUnitTest, DISABLED_copy_to_device_host)
#else
TEST_F(SparseTensorUnitTest, copy_to_device_host)
#endif
{
  // Nothing to do when testing on host
  _smallTensor->copy_to_host();
  _smallTensor->copy_to_device();
}

//==============================================================================

template <
  class SparseTensorType,
  class ElemIdx
>
int _do_get_data_view(
  SparseTensorType& smallTensor,
  ElemIdx nElementSmall
)
{
  int failure_count = 0;
  auto data = smallTensor.get_data_view();
  const auto data_const = static_cast<const SparseTensorType*>(&smallTensor)->get_data_view();

  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
    KOKKOS_LAMBDA(int, int& num_failures) {
      for (int i=0; i < nElementSmall; i++) {
        if(data[i] != i) ++num_failures;
      }

      for (int i=0; i < nElementSmall; i++) {
        if(data_const[i] != i) ++num_failures;
      }
    },
    failure_count
  );

  Kokkos::fence();

  return failure_count;
}

TEST_F(SparseTensorUnitTest, get_data_view)
{
  _smallTensor->copy_to_device();

  auto failure_count = _do_get_data_view(*_smallTensor, _nElementSmall);
  ASSERT_EQ(failure_count, 0);

  auto data_host = _smallTensor->get_data_host_view();
  for (int i=0; i < _nElementSmall; i++)
    ASSERT_EQ(data_host[i], i);

  const auto data_host_const = static_cast<const sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx>*>(_smallTensor)->get_data_host_view();
  for (int i=0; i < _nElementSmall; i++)
    ASSERT_EQ(data_host_const[i], i);
}

//==============================================================================

template <
  class SparseTensorType,
  class ElemIdx
>
int _do_get_indices_view(
  SparseTensorType& smallTensor,
  ElemIdx nElementSmall
)
{
  int passing_test_count = 0;
  auto idsView = smallTensor.get_indices_view();
  const auto idsView_const = static_cast<const SparseTensorType*>(&smallTensor)->get_indices_view();

  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
    KOKKOS_LAMBDA(int, int& passing_tests) {
      for (int i=0; i < nElementSmall; i++)
      {
        if(0 == idsView(0,i)) ++passing_tests;
        if(1 == idsView(1,i)) ++passing_tests;
        if(i == idsView(2,i)) ++passing_tests;
      }

      //sparten::type::SubIdx *ids = new sparten::type::SubIdx[_nDimSmall];
      for (int i=0; i < nElementSmall; i++)
      {
        if(0 == idsView_const(0,i)) ++passing_tests;
        if(1 == idsView_const(1,i)) ++passing_tests;
        if(i == idsView_const(2,i)) ++passing_tests;
      }
    },
    passing_test_count
  );

  Kokkos::fence();

  return passing_test_count;

}

TEST_F(SparseTensorUnitTest, get_indices_view)
{
  _smallTensor->copy_to_device();

  auto passing_test_count = _do_get_indices_view(*_smallTensor, _nElementSmall);

  ASSERT_EQ(passing_test_count, _nElementSmall * 6);

  auto idsView_host = _smallTensor->get_indices_host_view();
  for (int i=0; i < _nElementSmall; i++)
    {
      ASSERT_EQ(0, idsView_host(0,i));
      ASSERT_EQ(1, idsView_host(1,i));
      ASSERT_EQ(i, idsView_host(2,i));
    }

  const auto idsView_host_const = static_cast<const sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx>*>(_smallTensor)->get_indices_host_view();
  for (int i=0; i < _nElementSmall; i++)
    {
      ASSERT_EQ(0, idsView_host_const(0,i));
      ASSERT_EQ(1, idsView_host_const(1,i));
      ASSERT_EQ(i, idsView_host_const(2,i));
    }
}

//==============================================================================

TEST_F(SparseTensorUnitTest, DISABLED_test_write)
{
  //const double multiplier = 0.1;
  sparten::type::SubIdx *ids = new sparten::type::SubIdx[_nDim];
  double *diffs = new double[_nDim];

  /// todo Specify execution space (i.e., not just the default).  Will be needed later for GPU support.
//  Kokkos::parallel_for((_testTensor->get_nElement()), KOKKOS_LAMBDA (sparten::type::ElemIdx i)
//  {
//    double value = i * multiplier;
//
//    ids[0] = 1 * i;
//    ids[1] = 2 * i;
//    ids[2] = 3 * i;
//    ids[3] = 4 * i;
//    ids[4] = 5 * i;
//    std::vector<sparten::type::SubIdx> tmp(ids, ids + _nDim);
//    _testTensor->set_element(i, tmp, value );
//  });

  // Write initial file.
  sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *asciiIO_init =
      new sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("testSparseAscii.txt",
          sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, 8);
  asciiIO_init->write(*_testTensor);

  // Read it back in.
  sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *asciiIO = new sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(
      "testSparseAscii.txt", sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, 8);
  //asciiIO->open_file();
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *readTensor = NULL;
  readTensor = asciiIO->read();
  //asciiIO->close_file();

  // Write it out again.
  sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *asciiIO_rewrite = new sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx,
      sparten::type::SubIdx>("testSparseAscii_output.txt", sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, 8);
  asciiIO_rewrite->write(*readTensor);
  delete asciiIO_init;
  delete asciiIO;
  delete asciiIO_rewrite;
  delete readTensor;
  delete [] ids;
  delete [] diffs;
}

} // unitTest
