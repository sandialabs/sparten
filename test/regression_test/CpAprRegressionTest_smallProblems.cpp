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
#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/Log.h>
#include <sparten/SparseTensor.h>
#include <regression_test/CpAprRegressionTest_smallProblems.h>
#include <generator/FactTestSetGenerator.h>
#include <generator/RandGenMersenneTwister.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace regressionTest
{
	void CpAprRegressionTest_smallProblems::SetUp()
	{
	  _cpAprMultiplicativeUpdate  = nullptr;
	  _cpAprDampedNewton          = nullptr;
	  _cpAprQuasiNewton           = nullptr;

	  _tolerance        = 1e-4;
	  _maxOuterIter     = 400;
	  _maxInnerIter     = 10;
	  _offset           = 0.01;
	  _offsetTolerance  = 1e-10;
	  _progressInterval = 10;
	  _eps              = 1e-10;
	  _precision        = 16;

	  _cpAprMultiplicativeUpdate = new sparten::CpAprMultiplicativeUpdate<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
	}

	void CpAprRegressionTest_smallProblems::TearDown()
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

	sparten::SparseTensor<CpAprSparseValue, CpAprElemIdx, CpAprSubIdx> *CpAprRegressionTest_smallProblems::loadSparseData(std::string const &sparseInputFilename)
	{
	  auto *asciiInput = new sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(sparseInputFilename + ".txt", sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>::READ, sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>::FIXED_PRECISION, _precision, sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>::INDEX_1);
	  sparten::SparseTensor<CpAprSparseValue, CpAprElemIdx, CpAprSubIdx> *sparseTensor = nullptr;
	  sparseTensor = asciiInput->read();
	  delete asciiInput;

	  return sparseTensor;
	}

	template<typename T, typename TP>
	void storeKruskalData(T &kruskalTensor, std::string const &kruskalOutputFilename, TP precision)
	{
	  sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx> *kruskalOutput = new sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(kruskalOutputFilename + ".txt", sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>::OVERWRITE, sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>::SCIENTIFIC, precision, sparten::AsciiIO<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>::INDEX_1);
	  kruskalOutput->write(kruskalTensor);
	  kruskalOutput->close_file();
	  delete kruskalOutput;
	}

	template <typename cpAprType, typename kruskalType, typename sparseType>
	static void cpapminitestproblems_10x10x10(cpAprType &cpAprMethod, kruskalType &kruskalTensor, sparseType &sparseTensor, int32_t nComponent, double tolerance, double maxOuterIter, double maxInnerIter, double offset, double offsetTolerance, int32_t progressInterval, double eps, int32_t precision, std::string const &sparseInputFilename, std::string const &kruskalOutputFilename)
	{
	  // Init kruskal tensor
	  sparten::KruskalTensorInitializer<CpAprKruskalValue, CpAprSubIdx> kruskalInit;
	  kruskalInit.initialize(kruskalTensor, sparten::KruskalTensorInitializer<CpAprKruskalValue, CpAprSubIdx>::RANDOM);

	#ifdef SPARTEN_CHECK_NANS
	  std::cout << "NaNs present in kruskal tensor: " << (kruskalTensor.check_for_nans() ? "true" : "false") << std::endl;
	#endif

	  Kokkos::Timer timer;
	  cpAprMethod.compute(kruskalTensor, sparseTensor);

	#ifdef SPARTEN_CHECK_NANS
	  std::cout << "NaNs present in kruskal tensor: " << (kruskalTensor.check_for_nans() ? "true" : "false") << std::endl;
	#endif

	  storeKruskalData(kruskalTensor, kruskalOutputFilename, precision);
	}

	// First test of entire algorithm (MU).
	TEST_F(CpAprRegressionTest_smallProblems, 10x10x10_multiplicativeUpdate)
	{
	  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded.txt", sparten::Log::QUIET);
	  const std::string sparseInputFilename = "test/data/cpapr_test_10x10x10_1e+06/tensor";
	  const sparten::type::SubIdx nComponent = 5;
	  auto sparseTensor = loadSparseData(sparseInputFilename);

	  _cpAprMultiplicativeUpdate = new sparten::CpAprMultiplicativeUpdate<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
	  auto *kruskalTensor = new sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx>(sparseTensor->get_nDim(), nComponent, sparseTensor->get_size_dim());
	  cpapminitestproblems_10x10x10(*_cpAprMultiplicativeUpdate, *kruskalTensor, *sparseTensor, nComponent, _tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps, _precision, "test/data/cpapr_test_10x10x10_1e+06/tensor", "kruskalTestOutputMultiplicativeUpdate");
	  log.finalize();

	  ASSERT_TRUE(true);
	}

	// First test of entire algorithm (Damped Newton).
	TEST_F(CpAprRegressionTest_smallProblems, 10x10x10_dampedNewton)
	{
	  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);
	  const std::string sparseInputFilename = "test/data/cpapr_test_10x10x10_1e+06/tensor";
	  const sparten::type::SubIdx nComponent = 5;
	  auto sparseTensor = loadSparseData(sparseInputFilename);

	  _cpAprDampedNewton = new sparten::CpAprDampedNewton<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
	  auto *kruskalTensor = new sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx>(sparseTensor->get_nDim(), nComponent, sparseTensor->get_size_dim());
	  cpapminitestproblems_10x10x10(*_cpAprDampedNewton, *kruskalTensor, *sparseTensor, nComponent, _tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps, _precision, "test/data/cpapr_test_10x10x10_1e+06/tensor", "kruskalTestOutputDampedNewton");
	  log.finalize();

	  ASSERT_TRUE(true);
	}

	// First test of entire algorithm (Quasi Newton).
	TEST_F(CpAprRegressionTest_smallProblems, 10x10x10_quasiNewton)
	{
	  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);
	  const std::string sparseInputFilename = "test/data/cpapr_test_10x10x10_1e+06/tensor";
	  const sparten::type::SubIdx nComponent = 5;
	  auto sparseTensor = loadSparseData(sparseInputFilename);

	  _cpAprQuasiNewton = new sparten::CpAprQuasiNewton<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);
	  auto *kruskalTensor = new sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx>(sparseTensor->get_nDim(), nComponent, sparseTensor->get_size_dim());
	  cpapminitestproblems_10x10x10(*_cpAprQuasiNewton, *kruskalTensor, *sparseTensor, nComponent, _tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps, _precision, "test/data/cpapr_test_10x10x10_1e+06/tensor", "kruskalTestOutputQuasiNewton");
	  log.finalize();

	  ASSERT_TRUE(true);
	}

	// First test of entire algorithm.
	TEST_F(CpAprRegressionTest_smallProblems, DISABLED_withrandomgen)
	{
	  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

	  auto *_testGen = new sparten::FactTestSetGenerator<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>();
	  _cpAprMultiplicativeUpdate = new sparten::CpAprMultiplicativeUpdate<CpAprSparseValue, CpAprKruskalValue, CpAprElemIdx, CpAprSubIdx>(_tolerance, _maxOuterIter, _maxInnerIter, _offset, _offsetTolerance, _progressInterval, _eps);

	  sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx> *testTensor = nullptr;
	  sparten::SparseTensor<CpAprSparseValue, CpAprElemIdx, CpAprSubIdx> *sparseTensor = nullptr;
	  const sparten::type::SubIdx nComponent = 10;
	  std::vector<sparten::type::SubIdx> v = { 3000, 4000, 5000 };

	  _testGen->set_FactoredTensorDims(v);
	  _testGen->set_NumComponents(nComponent);
	  _testGen->set_MaxNnz(10000000);
	  _testGen->set_RandomSeed(1);

	  testTensor = _testGen->genNoisedNonNegativeKruscalTensor(0.2, 0.05);
	  sparseTensor = _testGen->genSparseTensor(testTensor);
	  sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx> *kruskalTensor = new sparten::KruskalTensor<CpAprKruskalValue, CpAprSubIdx>(sparseTensor->get_nDim(), nComponent, sparseTensor->get_size_dim());

	  // Making Initial Guess.
	  std::vector<CpAprKruskalValue> lambda(nComponent);
	  for (auto lambdaVal : lambda)
	  {
	    lambdaVal = 1.0;
	  }

	  kruskalTensor->set_weight_vector(lambda);
	  sparten::RandGenMersenneTwister cRGen(1);
	  for (auto iDim : kruskalTensor->get_dim())
	  {
	    sparten::FactorMatrix<CpAprKruskalValue> kData = kruskalTensor->get_factor_matrix(iDim);
	    for(sparten::type::SubIdx iCol = 0; iCol < kruskalTensor->get_factor_matrix_nColumn(iDim); ++iCol)
	    {
	      for(sparten::type::SubIdx iRow = 0; iRow < kruskalTensor->get_factor_matrix_nRow(iDim); ++iRow)
	      {
	        kData(iRow, iCol) = static_cast<CpAprKruskalValue>(cRGen.generate_rand_double_matlab());
	      }
	    }
	  }

	  std::cout << "Actual nnz " << sparseTensor->get_nElement() << std::endl;
	  Kokkos::Timer timer;
	  _cpAprMultiplicativeUpdate->compute(*kruskalTensor, *sparseTensor);
	  std::cout << "This took " << timer.seconds() << " seconds." << std::endl;

	  kruskalTensor->normalize();

	  delete sparseTensor;
	  delete testTensor;
	  log.finalize();

	  ASSERT_TRUE(true);
	}
} // regressionTest


