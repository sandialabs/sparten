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

#include <sparten/AsciiIO.h>
#include <sparten/Log.h>
#include <regression_test/FactTestSetGeneratorRegressionTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

namespace regressionTest
{
	void FactTestSetGeneratorRegressionTest::SetUp()
	{
	  _testGen = new sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>();
	}

	void FactTestSetGeneratorRegressionTest::TearDown()
	{
	  delete _testGen;
	}

    #if defined(KOKKOS_ENABLE_CUDA)
	TEST_F(FactTestSetGeneratorRegressionTest, DISABLED_test_with_io)
    #else
	TEST_F(FactTestSetGeneratorRegressionTest, test_with_io)
    #endif
	{
	  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);
	  typedef sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> SpartenIO;
	  sparten::KruskalTensor<double, sparten::type::SubIdx> *testFactors = nullptr;
	  sparten::SparseTensor<sparten::type::SubIdx, sparten::type::SubIdx, sparten::type::SubIdx> *spTensor = nullptr;
	  std::vector<sparten::type::SubIdx> v = { 10, 10, 10 };

	  _testGen->set_FactoredTensorDims(v);
	  _testGen->set_NumComponents(5);
	  _testGen->set_MaxNnz(1000000);
	  _testGen->set_RandomSeed(1);
	  _testGen->set_FracBoosted(0.2);
	  _testGen->set_MaxValue(100.0);

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

	  for (int32_t i = 0; i < spTensor->get_nElement() ; ++i )
	  {
	    for( int32_t j = 0; j < spTensor->get_nDim(); ++j  )
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
} // regressionTest
