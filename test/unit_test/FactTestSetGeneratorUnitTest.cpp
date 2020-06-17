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
#include <sparten/Log.h>
#include <unit_test/FactTestSetGeneratorUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

namespace unitTest
{

void FactTestSetGeneratorUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  _testGen = new sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>();
}

void FactTestSetGeneratorUnitTest::TearDown()
{
  delete _testGen;
}

TEST_F(FactTestSetGeneratorUnitTest, constructors)
{
  sparten::type::SubIdx nNumComponents = 2;
  sparten::type::ElemIdx nMaxNonz = 10;
  uint64_t nSeed = 123;
  double dFracBoosted = 0.1;
  double dMaxValue = 0.2;
  double dSmallValue = 0.0001;

  auto testGen = new sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>(nNumComponents,nMaxNonz,nSeed,dFracBoosted,dMaxValue,dSmallValue);

  ASSERT_EQ(testGen->get_NumComponents(),nNumComponents);
  ASSERT_EQ(testGen->get_MaxNnz(),nMaxNonz);
  ASSERT_EQ(testGen->get_RandomSeed(),nSeed);
  ASSERT_DOUBLE_EQ(testGen->get_FracBoosted(),dFracBoosted);
  ASSERT_DOUBLE_EQ(testGen->get_MaxValue(),dMaxValue);
  ASSERT_DOUBLE_EQ(testGen->get_SmallValue(),dSmallValue); 
  delete testGen;
}

TEST_F(FactTestSetGeneratorUnitTest, set_get_methods)
{
  // define parameters
  sparten::type::SubIdx nNumComponents = 2;
  sparten::type::ElemIdx nMaxNonz = 10;
  uint64_t nSeed = 123;
  double dFracBoosted = 0.1;
  double dMaxValue = 0.2;
  double dSmallValue = 0.0001;

  // create generator instance
  auto testGen = new sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>();

  // set parameters
  testGen->set_NumComponents(nNumComponents);
  testGen->set_MaxNnz(nMaxNonz);
  testGen->set_RandomSeed(nSeed);
  testGen->set_FracBoosted(dFracBoosted);
  testGen->set_MaxValue(dMaxValue);
  testGen->set_SmallValue(dSmallValue);
  std::vector<sparten::type::SubIdx> v = { 10 };
  testGen->set_FactoredTensorDims(v);

  // test returned parameters
  ASSERT_EQ(testGen->get_NumComponents(),nNumComponents);
  ASSERT_EQ(testGen->get_MaxNnz(),nMaxNonz);
  ASSERT_EQ(testGen->get_RandomSeed(),nSeed);
  ASSERT_DOUBLE_EQ(testGen->get_FracBoosted(),dFracBoosted);
  ASSERT_DOUBLE_EQ(testGen->get_MaxValue(),dMaxValue);
  ASSERT_DOUBLE_EQ(testGen->get_SmallValue(),dSmallValue); 
  auto dims =  testGen->get_FactoredTensorDims();
  ASSERT_EQ(dims[0],10);

  // set parameters in single call
  testGen->set_Parameters(nNumComponents+1,nMaxNonz+1,nSeed+1,dFracBoosted+0.01,dMaxValue+0.01,dSmallValue+0.01);

  /*
  TODO: the following does not compile
  // define second set of parameters to store returned parameters
  sparten::type::SubIdx nNumComponents2 = 0;
  sparten::type::ElemIdx nMaxNonz2 = 0;
  uint64_t nSeed2 = 0;
  double dFracBoosted2 = 0.0;
  double dMaxValue2 = 0.0;
  double dSmallValue2 = 0.0;

  // get second set of parameters
  testGen->get_Parameters(nNumComponents2,nMaxNonz2,nSeed2,dFracBoosted2,dMaxValue2,dSmallValue2);
  
  // test returned parameters
  ASSERT_EQ(nNumComponents2,nNumComponents+1);
  ASSERT_EQ(nMaxNonz2,nMaxNonz+1);
  ASSERT_EQ(nSeed2,nSeed+1);
  ASSERT_DOUBLE_EQ(dFracBoosted2,dFracBoosted+0.01);
  ASSERT_DOUBLE_EQ(dMaxValue2,dMaxValue+0.01);
  ASSERT_DOUBLE_EQ(dSmallValue2,dSmallValue+0.01); 
  */
  delete testGen;
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(FactTestSetGeneratorUnitTest, DISABLED_genNonNegativeKruscalTensor)
#else
TEST_F(FactTestSetGeneratorUnitTest, genNonNegativeKruscalTensor)
#endif
{
  sparten::type::SubIdx nNumComponents = 2;
  sparten::type::ElemIdx nMaxNonz = 10;
  uint64_t nSeed = 123;
  double dFracBoosted = 0.1;
  double dMaxValue = 1.0;
  double dSmallValue = 0.0001;

  auto testGen = new sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>(nNumComponents,nMaxNonz,nSeed,dFracBoosted,dMaxValue,dSmallValue);
  std::vector<sparten::type::SubIdx> v = { 2,3,4 };
  testGen->set_FactoredTensorDims(v);

  testGen->set_RandomSeed(1);
  auto kTensor = testGen->genNoisedNonNegativeKruscalTensor(0.2, 0.05);
  //std::cout << kTensor->show();
  
  // should match if seed is the same
  testGen->set_RandomSeed(1);
  auto kTensor2 = testGen->genNoisedNonNegativeKruscalTensor(0.2, 0.05);
  //std::cout << kTensor2->show();
  ASSERT_TRUE(kTensor->compare(*kTensor2, 1e-5));
  
  // should not match if seed is not the same
  testGen->set_RandomSeed(2);
  auto kTensor3 = testGen->genNoisedNonNegativeKruscalTensor(0.2, 0.05);
  //std::cout << kTensor3->show();
  ASSERT_FALSE(kTensor->compare(*kTensor3, 1e-5));

  // should not match if threshold and noise is not the same
  testGen->set_RandomSeed(1);
  auto kTensor4 = testGen->genNoisedNonNegativeKruscalTensor(0.1, 0.1);
  //std::cout << kTensor4->show();
  ASSERT_FALSE(kTensor->compare(*kTensor4, 1e-5));
  delete( testGen );
  delete( kTensor );
  delete( kTensor2 );
  delete( kTensor3 );
  delete( kTensor4 );
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST_F(FactTestSetGeneratorUnitTest, DISABLED_test_with_io_sparsetensor_integer)
#else
TEST_F(FactTestSetGeneratorUnitTest, test_with_io_sparsetensor_integer)
#endif
{
  typedef sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> SpartenIO;
  sparten::KruskalTensor<double, sparten::type::SubIdx> *testFactors = nullptr;
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::SubIdx, sparten::type::SubIdx> *spTensor = nullptr;
  std::vector<sparten::type::SubIdx> v = { 10, 10, 10, 10 };

  _testGen->set_FactoredTensorDims(v);
  _testGen->set_NumComponents(4);
  _testGen->set_MaxNnz(10000);
  _testGen->set_RandomSeed(1);
  _testGen->set_FracBoosted(0.2); //  May not be necessary
  _testGen->set_MaxValue(100.0);  //  May not be necesasry

  testFactors = _testGen->genNoisedNonNegativeKruscalTensor(0.2, 0.05);
  spTensor = _testGen->genSparseTensor( testFactors );

  SpartenIO myIO1("test/ktensor.txt", SpartenIO::OVERWRITE, SpartenIO::SCIENTIFIC, 16, SpartenIO::INDEX_1);
  SpartenIO myIO2("test/sptensor.txt", SpartenIO::OVERWRITE, SpartenIO::SCIENTIFIC, 16, SpartenIO::INDEX_1);
  myIO1.write(*testFactors);
  myIO2.write(*spTensor);
  myIO1.close_file();
  myIO2.close_file();

  sparten::KruskalTensor<double, sparten::type::SubIdx> *testFactorsFromFile = nullptr;
  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::SubIdx, sparten::type::SubIdx> *spTensorFromFile = nullptr;
  SpartenIO myIO3("test/sptensor.txt", SpartenIO::READ, SpartenIO::SCIENTIFIC, 16,SpartenIO::INDEX_1 );
  SpartenIO myIO4("test/ktensor.txt", SpartenIO::READ, SpartenIO::SCIENTIFIC, 16, SpartenIO::INDEX_1);
  spTensorFromFile = myIO3.read();
  testFactorsFromFile = myIO4.read_ktensor();
  myIO3.close_file();
  myIO4.close_file();

  // Verify the value of tensors
  ASSERT_EQ( testFactors->get_nDim(), testFactorsFromFile->get_nDim());
  for ( int32_t i = 0; i < testFactors->get_nDim(); ++i )
  {
    ASSERT_EQ( testFactors->get_factor_matrix_nRow(i), testFactorsFromFile->get_factor_matrix_nRow(i) );
  }

  ASSERT_EQ( testFactors->get_nComponent(), testFactorsFromFile->get_nComponent());
  for (int32_t r = 0 ; r <  _testGen->get_NumComponents(); ++r )
  {
    bool result = true;
    if( std::fabs( (double)testFactorsFromFile->get_weight_element( r ) - (double) testFactors->get_weight_element( r ) ) / (double) testFactors->get_weight_element( r )  > 1e-16 )
    {
       result = false;
    }
    ASSERT_EQ(result, true);
  }
  
  for (int32_t i = 0; i < testFactors->get_nDim(); ++i ) {
    for ( int32_t k = 0; k < testFactors->get_factor_matrix_nColumn( i); ++k ) {
      for ( int32_t j = 0; j < testFactors->get_factor_matrix_nRow( i); ++j ) {
        bool result = true;
        if( std::fabs( (double)testFactorsFromFile->get_factor_matrix_element( i,j, k ) - (double) testFactors->get_factor_matrix_element( i,j, k ) ) / (double) testFactors->get_factor_matrix_element( i,j, k ) > 1e-16 )
        {
          result = false;

        }
        ASSERT_EQ(result, true);
      }
    }
  }

  ASSERT_EQ(spTensor->get_nDim(), spTensorFromFile->get_nDim());
  ASSERT_EQ(spTensor->get_nElement(), spTensorFromFile->get_nElement());
  std::vector<sparten::type::SubIdx> dims_1 = spTensor->get_size_dim();
  std::vector<sparten::type::SubIdx> dims_2 = spTensorFromFile->get_size_dim();
  for ( int32_t i = 0 ; i < spTensor->get_nDim(); ++i )
  {
    ASSERT_EQ(dims_1[i],dims_2[i]);
  }

  auto spValues_1 = spTensor->get_data_view();
  auto spValues_2 = spTensorFromFile->get_data_view();

  auto spIndices_1 = spTensor->get_indices_view();
  auto spIndices_2 = spTensorFromFile->get_indices_view();

  for( int32_t j = 0; j < spTensor->get_nDim(); ++j  )
  {
    auto spRows_1 = spTensor->get_nRow(j);
    auto spRows_2 = spTensorFromFile->get_nRow(j);
    ASSERT_EQ(spRows_1, spRows_2);
  }

  for (int32_t i = 0; i < spTensor->get_nElement() ; ++i)
  {
    for (int32_t j = 0; j < spTensor->get_nDim(); ++j)
    {
      ASSERT_EQ(spIndices_1(j,i),spIndices_2(j,i));
    }
    bool result = std::fabs( (double)spValues_1(i) - (double)spValues_2(i) ) / std::fabs( (double)spValues_1(i)) <=  1e-16 ;
    ASSERT_EQ(result,true);
  }

  SpartenIO myIO5("test/ktensor_verify.txt", SpartenIO::OVERWRITE, SpartenIO::SCIENTIFIC, 16, SpartenIO::INDEX_1);
  SpartenIO myIO6("test/sptensor_verify.txt", SpartenIO::OVERWRITE, SpartenIO::SCIENTIFIC, 16, SpartenIO::INDEX_1);
  myIO5.write(*testFactorsFromFile);
  myIO6.write(*spTensorFromFile);
  myIO5.close_file();
  myIO6.close_file();

  delete( spTensor );
  delete( testFactors );
  delete( spTensorFromFile );
  delete( testFactorsFromFile );



}

} // unitTest
