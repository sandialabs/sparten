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

#include <generator/DiscreteCumulativeDistribution.h>
#include <generator/RandGenMersenneTwister.h>
#include <sparten/Log.h>
#include <unit_test/DiscreteCumulativeDistributionUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void DiscreteCumulativeDistributionUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  //const signed long int32_t nnSeed = 0; /// todo put a real value here...
  _testCDF = new sparten::DiscreteCumulativeDistribution<double, int32_t>(); /// todo Verify this, update types
}

void DiscreteCumulativeDistributionUnitTest::TearDown()
{
  delete _testCDF;
}

void DiscreteCumulativeDistributionUnitTest::create_hist_vector(Kokkos::View<double*> &v, int32_t size)
{

  double dTotalWeight = 0.0;
  for (int32_t i = 0; i < size; ++i)
  {
    dTotalWeight += v(i);
  }
  for (int32_t i = 0; i < size; ++i)
  {
    v(i) = v(i) / dTotalWeight;
    //    std::cout << "V(" << i << ") = " << v(i) << std::endl;
  }

}

void DiscreteCumulativeDistributionUnitTest::create_hist_matrix(Kokkos::View<double**> &a, int32_t sizeM, int32_t sizeN)
{

  for (int32_t j = 0; j < sizeN; ++j)
  {
    double dTotalWeight = 0.0;
    for (int32_t i = 0; i < sizeM; ++i)
    {
      dTotalWeight += a(i, j);
    }

    for (int32_t i = 0; i < sizeM; ++i)
    {
      a(i, j) = a(i, j) / dTotalWeight;
    }
  }
}


#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(DiscreteCumulativeDistributionUnitTest, DISABLED_load_Vector)
#else
TEST_F(DiscreteCumulativeDistributionUnitTest, load_Vector)
#endif
{
  // 1. Testing that load(const VectorConst<KruskalValue> &cPDF) generates something
  // 2. Testing that the results are repeatable

  // First sample
  sparten::RandGenMersenneTwister myRand(1);
  typedef Kokkos::View<double*> ViewVectorType;
  ViewVectorType y("y", 20);

  for (int32_t i = 0; i < 20; i++)
  {
    y(i) = myRand.generate_rand_double_matlab();
  }
  create_hist_vector(y, 20);

  _testCDF->load(y);
  std::vector<int32_t> x(100);

  for (int32_t i = 0; i < 100; ++i)
  {
    x[i] = _testCDF->get_random_sample(myRand.generate_rand_double_matlab());
  }

  //Second sample
  sparten::RandGenMersenneTwister myRand2(1);
  typedef Kokkos::View<double*> ViewVectorType;
  ViewVectorType y2("y2", 20);

  for (int32_t i = 0; i < 20; i++)
  {
    y2(i) = myRand.generate_rand_double_matlab();
  }
  create_hist_vector(y2, 20);

  _testCDF->load(y2);
  std::vector<int32_t> x2(100);

  for (int32_t i = 0; i < 100; ++i)
  {
    x2[i] = _testCDF->get_random_sample(myRand.generate_rand_double_matlab());
  }

  // Repeatability, given the same RNG seed
  for (int32_t i = 0; i < 100; ++i)
    ASSERT_EQ(x[i],x2[i]);

  // PDF length of 1
  sparten::RandGenMersenneTwister myRand3(1);
  typedef Kokkos::View<double*> ViewVectorType;
  ViewVectorType y3("y3", 1);
  y3(0) = myRand.generate_rand_double_matlab();
  create_hist_vector(y3, 1);
  ASSERT_EQ(_testCDF->load(y3), true);

  // value in PDF is invalid
  sparten::RandGenMersenneTwister myRand4(1);
  typedef Kokkos::View<double*> ViewVectorType;
  ViewVectorType y4("y4", 2);
  y4(0) = myRand.generate_rand_double_matlab();
  y4(1) = myRand.generate_rand_double_matlab();
  create_hist_vector(y4, 2);
  y4(0) = 2.0;
  ASSERT_EQ(_testCDF->load(y4), false);
  y4(0) = -2.0;
  ASSERT_EQ(_testCDF->load(y4), false);

  // values in PDF do not sum to 1.0
  typedef Kokkos::View<double*> ViewVectorType;
  ViewVectorType y5("y5", 2);
  y5(0) = 0.75;
  y5(1) = 0.75;
  ASSERT_EQ(_testCDF->load(y5), false);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(DiscreteCumulativeDistributionUnitTest, DISABLED_load_Array2D)
#else
TEST_F(DiscreteCumulativeDistributionUnitTest, load_Array2D)
#endif
{

  // First sample
  sparten::RandGenMersenneTwister myRand(1);
  typedef Kokkos::View<double**> ViewMatrixType;
  ViewMatrixType y("y", 20, 4);

  for (int32_t j = 0; j < 4; ++j)
  {
    for (int32_t i = 0; i < 20; ++i)
    {
      y(i, j) = myRand.generate_rand_double_matlab();
    }
  }
  create_hist_matrix(y, 20, 4);
  std::vector<std::vector<int32_t> > x(4);
  for (int32_t j = 0; j < 4; ++j)
  {
    _testCDF->load(y, j);
    x[j].resize(100);
    for (int32_t i = 0; i < 100; ++i)
    {
      x[j][i] = _testCDF->get_random_sample(myRand.generate_rand_double_matlab());
    }
  }

  // Second sample
  sparten::RandGenMersenneTwister myRand2(1);
  typedef Kokkos::View<double**> ViewMatrixType;
  ViewMatrixType y2("y2", 20, 4);

  for (int32_t j = 0; j < 4; ++j)
  {
    for (int32_t i = 0; i < 20; ++i)
    {
      y2(i, j) = myRand.generate_rand_double_matlab();
    }
  }
  create_hist_matrix(y2, 20, 4);
  std::vector<std::vector<int32_t> > x2(4);
  for (int32_t j = 0; j < 4; ++j)
  {
    _testCDF->load(y2, j);
    x2[j].resize(100);
    for (int32_t i = 0; i < 100; ++i)
    {
      x2[j][i] = _testCDF->get_random_sample(myRand.generate_rand_double_matlab());
    }
  }

  // Testing repeatability
  for (int32_t j = 0; j < 4; ++j)
    for (int32_t i = 0; i < 100; ++i)
      ASSERT_EQ(x[j][i], x2[j][i]);

  // PDF with invalid value
  typedef Kokkos::View<double**> ViewMatrixType;
  ViewMatrixType y3("y3", 3, 2);
  y3(0,0) = 2.0;
  y3(0,1) = 2.0;
  y3(1,0) = 2.0;
  y3(1,1) = 2.0;
  y3(2,0) = 2.0;
  y3(2,1) = 2.0;
  ASSERT_EQ(_testCDF->load(y3,0), false);
  y3(0,0) = -2.0;
  y3(0,1) = -2.0;
  y3(1,0) = -2.0;
  y3(1,1) = -2.0;
  y3(2,0) = -2.0;
  y3(2,1) = -2.0;
  // values do not sum to 1
  ASSERT_EQ(_testCDF->load(y3,0), false);
  y3(0,0) = 0.99;
  y3(0,1) = 0.99;
  y3(1,0) = 0.99;
  y3(1,1) = 0.99;
  y3(2,0) = 0.99;
  y3(2,1) = 0.99;
  ASSERT_EQ(_testCDF->load(y3,0), false);

}


#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(DiscreteCumulativeDistributionUnitTest, DISABLED_get_random_sample)
#else
TEST_F(DiscreteCumulativeDistributionUnitTest, get_random_sample)
#endif
{

  sparten::RandGenMersenneTwister myRand(1);
  typedef Kokkos::View<double*> ViewVectorType;
  ViewVectorType y("y", 10);
  
  for (int32_t i = 0; i < 10; i++)
  {
    y(i) = myRand.generate_rand_double_matlab();
  }
  create_hist_vector(y, 10);

  // should walk through CDF without search 
  _testCDF->load(y);
  _testCDF->get_random_sample(0.5);
  _testCDF->get_random_sample(2.0);
  

}

} // unitTest
