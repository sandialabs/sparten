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

#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/Log.h>
#include <unit_test/KruskalTensorUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

KruskalTensorUnitTest::KruskalTensorUnitTest() :
  _log(sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET))
{
}

void KruskalTensorUnitTest::SetUp()
{
  const sparten::type::SubIdx nComponent = 4;
  _nComponent = nComponent;
  const sparten::type::SubIdx nDim = 5;
  _nDim = nDim;
  sparten::type::SubIdx tmp[nDim] = { 6, 7, 8, 9, 10 };
  // max of tmp:
  _maxModeSize = 10;
  
  _nPerMode = std::vector<sparten::type::SubIdx> (tmp, tmp + _nDim);
  _testTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDim, _nComponent, _nPerMode);
}

void KruskalTensorUnitTest::TearDown()
{
  delete _testTensor;
}

TEST_F(KruskalTensorUnitTest, get_nComponent)
{
  ASSERT_EQ(_testTensor->get_nComponent(), _nComponent);
}

TEST_F(KruskalTensorUnitTest, get_maxModeSize)
{
  ASSERT_EQ(_testTensor->get_max_mode_size(), _maxModeSize);
}

TEST_F(KruskalTensorUnitTest, get_nDim)
{
  ASSERT_EQ(_testTensor->get_nDim(), _nDim);
}

TEST_F(KruskalTensorUnitTest, get_nPerMode)
{
  for (sparten::type::SubIdx iDim = 0; iDim < _nDim; ++iDim)
  {
    ASSERT_EQ(_testTensor->get_nPerMode(iDim), _nPerMode[iDim]);
  }

  const auto nPerMode = _testTensor->get_nPerMode();

  for (sparten::type::SubIdx iDim = 0; iDim < _nDim; ++iDim)
  {
    ASSERT_EQ(nPerMode[iDim], _nPerMode[iDim]);
  }
}

TEST_F(KruskalTensorUnitTest, get_factor_matrix_nRow)
{
  for (sparten::type::SubIdx iDim = 0; iDim < _nDim; ++iDim)
  {
    ASSERT_EQ(_testTensor->get_factor_matrix_nRow(iDim), _nPerMode[iDim]);
  }

  // host version
  for (sparten::type::SubIdx iDim = 0; iDim < _nDim; ++iDim)
  {
    ASSERT_EQ(_testTensor->get_factor_matrix_nRow_host(iDim), _nPerMode[iDim]);
  }
}

TEST_F(KruskalTensorUnitTest, get_factor_matrix_nColumn)
{
  for (sparten::type::SubIdx iDim = 0; iDim < _nDim; ++iDim)
  {
    ASSERT_EQ(_testTensor->get_factor_matrix_nColumn(iDim), _nComponent);
  }
}

TEST_F(KruskalTensorUnitTest, get_factor_matrix_buffer)
{
  const auto nDim = _testTensor->get_nDim();
  const auto nComponent = _testTensor->get_nComponent();

  for (sparten::type::SubIdx iDim = 0; iDim < nDim; ++iDim)
  {
    auto factorMatrixBuffer = _testTensor->get_factor_matrix_buffer(iDim);
    ASSERT_EQ(factorMatrixBuffer.extent(0), _nPerMode[iDim]);
    ASSERT_EQ(factorMatrixBuffer.extent(1), _nComponent);
  }
}

#if defined(KOKKOS_ENABLE_CUDA)
  TEST_F(KruskalTensorUnitTest, DISABLED_move_to_device_host)
#else
TEST_F(KruskalTensorUnitTest, move_to_device_host)
#endif
{
  // Nothing to do when testing on host
  _testTensor->move_to_host();
  _testTensor->move_to_device();

  // const version
  static_cast<const sparten::KruskalTensor<double, sparten::type::SubIdx>*>(_testTensor)->move_to_host();
  static_cast<const sparten::KruskalTensor<double, sparten::type::SubIdx>*>(_testTensor)->move_to_device();
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_set_and_get_factor_matrix_element)
#else
TEST_F(KruskalTensorUnitTest, set_and_get_factor_matrix_element)
#endif
{
  const double value = 5.314159;
  const sparten::type::SubIdx dim = 2;
  const sparten::type::SubIdx comp = 1;
  const sparten::type::SubIdx column = 1;

  _testTensor->set_factor_matrix_element(value, dim, comp, column);

  ASSERT_EQ(_testTensor->get_factor_matrix_element(dim, comp, column), value);
  // host version
  ASSERT_EQ(_testTensor->get_factor_matrix_element_host(dim, comp, column), value);
}


#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_set_factor_matrix_row_and_column)
#else
TEST_F(KruskalTensorUnitTest, set_factor_matrix_row_and_column)
#endif
{
  
  double rowValue = 1.234;
  double colValue = 3.456;
  const auto nRow = _testTensor->get_factor_matrix_nRow(0);
  const auto nCol = _testTensor->get_factor_matrix_nColumn(0);
  // there are nCol values in each row and nRow values in each column
  std::vector<double> factorMatrixRowValues(nCol, rowValue);
  std::vector<double> factorMatrixColumnValues(nRow, colValue);

  _testTensor->set_factor_matrix_row(factorMatrixRowValues, 0, 0);
  //std::cout << _testTensor->show();
  for (sparten::type::SubIdx iCol= 0; iCol < nCol; ++iCol)
    ASSERT_EQ(_testTensor->get_factor_matrix_element(0,0,iCol), rowValue);
  
  _testTensor->set_factor_matrix_column(factorMatrixColumnValues, 0, 0);
  for (sparten::type::SubIdx iRow= 0; iRow < nRow; ++iRow)
    ASSERT_EQ(_testTensor->get_factor_matrix_element(0,iRow,0), colValue);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_set_and_get_weight_element)
#else
TEST_F(KruskalTensorUnitTest, set_and_get_weight_element)
#endif
{
  const double value = 1.314159;
  const sparten::type::SubIdx index = 1;
  auto weights = _testTensor->get_weight_vector();
 
  weights(index) = value;

  ASSERT_EQ(weights(index), value);
  ASSERT_EQ(_testTensor->get_weight_element(index), value);
  // host version
  ASSERT_EQ(_testTensor->get_weight_element_host(index), value);

  _testTensor->set_weight_element(2 * value, index);
  ASSERT_EQ(weights(index), 2 * value);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_set_and_get_weight_vector)
#else
TEST_F(KruskalTensorUnitTest, set_and_get_weight_vector)
#endif
{
  const auto nComponent = _testTensor->get_nComponent();

  std::vector<double> setTestWeights(nComponent);


  for (sparten::type::SubIdx i = 0; i < nComponent; ++i)
  {
    setTestWeights[i] = i * 3; // index times an arbitrary multiplier
  }

  _testTensor->set_weight_vector(setTestWeights);

  for (sparten::type::SubIdx i = 0; i < nComponent; ++i)
  {
    ASSERT_EQ(_testTensor->get_weight_element(i), setTestWeights[i]);
  }

  const auto getTestWeightsConst = static_cast<const sparten::KruskalTensor<double, sparten::type::SubIdx>*>(_testTensor)->get_weight_vector();
  auto getTestWeights = _testTensor->get_weight_vector();

  for (sparten::type::SubIdx i = 0; i < nComponent; ++i)
  {
    ASSERT_EQ(getTestWeightsConst(i), setTestWeights[i]);
    ASSERT_EQ(getTestWeights(i), setTestWeights[i]);
  }

  // should result in a runtime error
  std::vector<double> setTestWeights_fail(nComponent+1);
  for (sparten::type::SubIdx i = 0; i < nComponent+1; ++i)
  {
    setTestWeights_fail[i] = 0; // index times an arbitrary multiplier
  }

  ASSERT_ANY_THROW(_testTensor->set_weight_vector(setTestWeights_fail));
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_distribute_weights_to_factor_matrices)
#else
TEST_F(KruskalTensorUnitTest, distribute_weights_to_factor_matrices)
#endif
{
  const auto nDim = _testTensor->get_nDim();
  const auto nComponent = _testTensor->get_nComponent();

  for (sparten::type::SubIdx iDim = 0; iDim < nDim; ++iDim)
  {
    const auto nRow = _testTensor->get_factor_matrix_nRow(iDim);
    const auto nCol = _testTensor->get_factor_matrix_nColumn(iDim);

    for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow)
    {
      for (sparten::type::SubIdx iCol = 0; iCol < nCol; ++iCol)
      {
        _testTensor->set_factor_matrix_element(1, iDim, iRow, iCol);
      }
    }
  }

  for (sparten::type::SubIdx iComp = 0; iComp < nComponent; ++iComp)
  {
    _testTensor->set_weight_element((iComp + 1) *  0.1, iComp);
  }

  _testTensor->distribute_weights_to_factor_matrices();

  for (sparten::type::SubIdx iDim = 0; iDim < nDim; ++iDim)
  {
    const auto nRow = _testTensor->get_factor_matrix_nRow(iDim);
    const auto nCol = _testTensor->get_factor_matrix_nColumn(iDim);

    for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow)
    {
      for (sparten::type::SubIdx iCol = 0; iCol < nCol; ++iCol)
      {
        ASSERT_EQ(_testTensor->get_factor_matrix_element(iDim, iRow, iCol), (iCol + 1) * 0.1);
      }
    }
  }

  for (sparten::type::SubIdx iComp = 0; iComp < nComponent; ++iComp)
  {
    ASSERT_EQ(_testTensor->get_weight_element(iComp), 1.0);
  }
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_distribute_weights_to_factor_matrices_provide_iDim)
#else
TEST_F(KruskalTensorUnitTest, distribute_weights_to_factor_matrices_provide_iDim)
#endif

{
  auto dim = _testTensor->get_dim();
  const auto nDim = _testTensor->get_nDim();
  const auto nComponent = _testTensor->get_nComponent();

  for (auto iDim : dim)
  {
    const auto nRow = _testTensor->get_factor_matrix_nRow(iDim);
    const auto nCol = _testTensor->get_factor_matrix_nColumn(iDim);

    for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow)
    {
      for (sparten::type::SubIdx iCol = 0; iCol < nCol; ++iCol)
      {
        _testTensor->set_factor_matrix_element(1, iDim, iRow, iCol);
      }
    }
  }

  for (sparten::type::SubIdx iComp = 0; iComp < nComponent; ++iComp)
  {
    _testTensor->set_weight_element((iComp + 1) *  0.1, iComp);
  }

  for (auto iDim : dim)
  {
    _testTensor->distribute_weights_to_factor_matrices(iDim, false);
  }

  for (auto iDim : dim)
  {
    const auto nRow = _testTensor->get_factor_matrix_nRow(iDim);
    const auto nCol = _testTensor->get_factor_matrix_nColumn(iDim);

    for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow)
    {
      for (sparten::type::SubIdx iCol = 0; iCol < nCol; ++iCol)
      {
        ASSERT_EQ(_testTensor->get_factor_matrix_element(iDim, iRow, iCol), (iCol + 1) * 0.1) <<
            "iDim = " << iDim << ", " <<
            "iRow = " << iRow << ", " <<
            "iCol = " << iCol << std::endl;
      }
    }
  }

  for (sparten::type::SubIdx iComp = 0; iComp < nComponent; ++iComp)
  {
    ASSERT_EQ(_testTensor->get_weight_element(iComp), 1.0);
  }
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_normalize)
#else
TEST_F(KruskalTensorUnitTest, normalize)
#endif
{
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  auto *kTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);
  auto dim = kTensor->get_dim();
  const auto nDim = kTensor->get_nDim();

  kruskalInit.initialize(*kTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);
  kTensor->normalize(); 

 
  for (auto iDim : dim)
  {
    const auto nRow = kTensor->get_factor_matrix_nRow(iDim);
    const auto nCol = kTensor->get_factor_matrix_nColumn(iDim);
    
    for (sparten::type::SubIdx iCol = 0; iCol < nCol; ++iCol)
    {
      double colNorm = 0.0;
      for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow)
      {
	colNorm += sparten::abs(kTensor->get_factor_matrix_element(iDim, iRow, iCol));
      }

      ASSERT_DOUBLE_EQ(colNorm, 1.0);
    }
  }

  delete(kTensor);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_normalize_provide_iDim)
#else
TEST_F(KruskalTensorUnitTest, normalize_provide_iDim)
#endif
{
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  auto *kTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);
  auto dim = kTensor->get_dim();
  const auto nDim = kTensor->get_nDim();

  // Normalize to get the weights
  kruskalInit.initialize(*kTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);
  //std::cout << kTensor->show();
  kTensor->normalize();
  //std::cout << kTensor->show();
  const auto nComponent = _testTensor->get_nComponent();
  auto weights = kTensor->get_weight_vector();
  
  // Normalize, pushing all weight on to dim 0 factor matrix, then normalizing on that factor only
  kTensor->distribute_weights_to_factor_matrices(0);
  //std::cout << kTensor->show();
  kTensor->normalize(0);
  //std::cout << kTensor->show();

  for (auto iDim : dim)
  {
    const auto nRow = kTensor->get_factor_matrix_nRow(iDim);
    const auto nCol = kTensor->get_factor_matrix_nColumn(iDim);

    for (sparten::type::SubIdx iCol = 0; iCol < nCol; ++iCol)
    {
      double colNorm = 0.0;
      for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow)
      {
	colNorm += sparten::abs(kTensor->get_factor_matrix_element(iDim, iRow, iCol));
      }
      
      ASSERT_DOUBLE_EQ(colNorm, 1.0);
    }
  }
  delete(kTensor);
}

//==============================================================================

template <class KruskalTensorType>
void _do_set_nan(KruskalTensorType& kTensor) {
  auto factor_matrix = kTensor.get_factor_matrix(0);
  Kokkos::parallel_for(1,
    KOKKOS_LAMBDA(int _unused) {
      // __builtin_nanf may not exist on the device, I guess
      uint64_t x = 0x7ff8000000000000; // NAN
      factor_matrix(0, 0) = *reinterpret_cast<double*>(&x);
    }
  );
}

TEST_F(KruskalTensorUnitTest, check_for_nans)
{
  auto *kTensor = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);
  kTensor->move_to_device();
  ASSERT_EQ(kTensor->check_for_nans(),false);
  _do_set_nan(*kTensor);
  ASSERT_EQ(kTensor->check_for_nans(),true);
  Kokkos::fence();
  delete(kTensor);
}

//==============================================================================

// Note: This also verifies the KruskalTensor copy constructor (deep copy)
#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, compare)
#else
TEST_F(KruskalTensorUnitTest, compare)
#endif
{
  const double tolerance = 1e-5;

  // Kruskal tensor #1, randomly generated
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  kruskalInit.initialize(*_testTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);

  // Kruskal tensor #2, copy #1
  auto kruskalTensor2 = sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);

  // Check to see if deep copy is the same (it should be)
  bool isSame = _testTensor->compare(kruskalTensor2, tolerance);
  ASSERT_TRUE(isSame) << "failed on comparing deep copy of Kruskal Tensor.";
  // Modify Kruskal Tensor #2
  kruskalTensor2.set_factor_matrix_element(3.14159 * 100, 1, 2, 3);
  kruskalTensor2.move_to_device();

 
  // Compare again (it should be different)
  isSame = _testTensor->compare(kruskalTensor2, tolerance);
  ASSERT_FALSE(isSame) << "Failed to detect change in copied Kruskal Tensor.";

  // Failure to match nDim
  _nPerMode.push_back(1);
  auto kTensor_bad_nDim = sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDim + 1, _nComponent, _nPerMode);
  isSame = _testTensor->compare(kTensor_bad_nDim, tolerance);
  ASSERT_FALSE(isSame) << "Failed to detect change in different Kruskal Tensor.";

  // Failure to match nComponent
  auto kTensor_bad_nComponent = sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDim, _nComponent+1, _nPerMode);
  isSame = _testTensor->compare(kTensor_bad_nComponent, tolerance);
  ASSERT_FALSE(isSame) << "Failed to detect change in different Kruskal Tensor.";

  //// Failure to match nPerMode
  auto nPerMode_bad = std::vector<sparten::type::SubIdx> (_nPerMode);
  nPerMode_bad[0] = 1;
  auto kTensor_bad_nPerMode = sparten::KruskalTensor<double, sparten::type::SubIdx>(_nDim, _nComponent, nPerMode_bad);
  isSame = _testTensor->compare(kTensor_bad_nPerMode, tolerance);
  ASSERT_FALSE(isSame) << "Failed to detect change in different Kruskal Tensor.";
  Kokkos::fence();
}

TEST_F(KruskalTensorUnitTest, show)
{
  // Kruskal tensor #1
  std::string s1 = _testTensor->show();
  
  // Kruskal tensor #2, copy #1
  auto *kruskalTensor2 = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);
  std::string s2 = kruskalTensor2->show();

  ASSERT_STREQ(s1.c_str(),s2.c_str());
  delete( kruskalTensor2 );
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(KruskalTensorUnitTest, DISABLED_show_per_mode)
#else
TEST_F(KruskalTensorUnitTest, show_per_mode)
#endif
{
  auto dim = _testTensor->get_dim();
  for (auto iDim : dim)
  {
    // Kruskal tensor #1
    std::string s1 = _testTensor->show_by_mode(iDim);
  
    // Kruskal tensor #2, copy #1
    auto *kruskalTensor2 = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);
    std::string s2 = kruskalTensor2->show_by_mode(iDim);
    
    ASSERT_STREQ(s1.c_str(),s2.c_str());
    delete( kruskalTensor2 );
  }
}

//==============================================================================

// Because NVCC can't handle lambdas in private member functions...
template <
  class KruskalTensorType
>
int _do_permute_factor_matrix_columns(
  KruskalTensorType& testTensor
) {
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  kruskalInit.initialize(testTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::UNITY);

  auto dim = testTensor.get_dim();
  auto nDim = testTensor.get_nDim();
  auto nComponent = testTensor.get_nComponent();

  testTensor.move_to_device();

  for (auto iDim: dim)
  {
    auto factorMatrix_iDim = testTensor.get_factor_matrix(iDim);
    const auto nCol = testTensor.get_factor_matrix_nColumn(iDim);
    Kokkos::parallel_for(
      nCol,
      KOKKOS_LAMBDA(int iCol) {
        factorMatrix_iDim(0, iCol) = (double)iCol;
      }
    );
  }

  testTensor.normalize();

  auto kTensor = sparten::KruskalTensor<double, sparten::type::SubIdx>(testTensor);

  auto weights = kTensor.get_weight_vector();

  testTensor.permute_factor_matrix_columns();

  auto weights2 = testTensor.get_weight_vector();

  int failure_count = 0;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
    KOKKOS_LAMBDA(int, int& num_failures) {

      for (sparten::type::SubIdx iComp = 0; iComp < nComponent; iComp++)
      {
        // check the permuted weights
        //ASSERT_EQ(weights(iComp), weights2(nComponent-iComp-1));
        if(weights(iComp) != weights2(nComponent-iComp-1)) ++num_failures;
      }
    },
    failure_count
  );

  for (sparten::type::SubIdx d = 0; d < nDim; d++)
  {
    auto test_factor_matrix = testTensor.get_factor_matrix(d);
    auto k_factor_matrix = kTensor.get_factor_matrix(d);
    int ifailure_count = 0;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nComponent),
      KOKKOS_LAMBDA(int iComp, int& num_failures) {
        // Just check the values we changed
        // ASSERT_EQ(kTensor->get_factor_matrix_element(d,0,iComp),_testTensor->get_factor_matrix_element(d,0,nComponent-iComp-1));
        if(k_factor_matrix(0, iComp) != test_factor_matrix(0, nComponent-iComp-1)) ++num_failures;
      },
      ifailure_count
    );
    failure_count += ifailure_count;
  }

  Kokkos::fence();

  return failure_count;
}

TEST_F(KruskalTensorUnitTest, permute_factor_matrix_columns)
{
  auto num_failures = _do_permute_factor_matrix_columns(*_testTensor);
  ASSERT_EQ(num_failures, 0);
}

//==============================================================================

TEST_F(KruskalTensorUnitTest, initialize)
{
  const double tolerance = 1e-5;

  // Kruskal tensor #1, randomly generated
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  kruskalInit.initialize(*_testTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::RANDOM);

  // Kruskal tensor #2, copy #1
  auto *kruskalTensor2 = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);

  // Check to see if deep copy is the same (it should be)
  bool isSame = _testTensor->compare(*kruskalTensor2, tolerance);
  ASSERT_TRUE(isSame) << "failed on comparing deep copy of Kruskal Tensor.";

  // Kruskal tensor #1, all ones
  kruskalInit.initialize(*_testTensor, sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::UNITY);

  // Kruskal tensor #3, copy #1
  auto *kruskalTensor3 = new sparten::KruskalTensor<double, sparten::type::SubIdx>(*_testTensor);

  // Check to see if deep copy is the same (it should be)
  isSame = _testTensor->compare(*kruskalTensor3, tolerance);
  ASSERT_TRUE(isSame) << "failed on comparing deep copy of Kruskal Tensor.";

  // Kruskal tensor #1, incorrect initialization requested
  ASSERT_ANY_THROW(kruskalInit.initialize(*_testTensor, (sparten::KruskalTensorInitializer<double, sparten::type::SubIdx>::Method)(-1)));
  
  delete( kruskalTensor2 );
  delete( kruskalTensor3 );
}

TEST_F(KruskalTensorUnitTest, initialize_set_and_get_random_seed)
{
  // Kruskal tensor #1, randomly generated
  sparten::KruskalTensorInitializer<double, sparten::type::SubIdx> kruskalInit;
  int32_t seed = 12345;
  kruskalInit.set_random_seed(seed);
  ASSERT_EQ(kruskalInit.get_random_seed(), seed);
}
  
} // unitTest





















