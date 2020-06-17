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

#pragma once

#include <SimpleInput.h>
#include <sparten/AsciiIO.h>
#include <sparten/CpAprBase.h>
#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/PrecisionTraits.h>
#include <sparten/DampedNewtonCpApr.h>
#include <sparten/QuasiNewtonCpApr.h>
#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/Log.h>
#include <sparten/SparseTensor.h>
#include <string>
#include <sstream>
#include <vector>

template<typename SparseValue, typename KruskalValue, typename ElemIdx, typename SubIdx>
void cpApr_compute(
  CommandLineOptions const &clo,
  sparten::KruskalTensor<KruskalValue, SubIdx>& kruskalOutput,
  sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const& sparseInput
)
{
  using numeric_traits = sparten::PrecisionTraits<SparseValue, KruskalValue, ElemIdx, SubIdx>;
	auto config = sparten::CpAprConfiguration<numeric_traits>{
		/* max_outer_iterations = */ SubIdx(clo.maxOuterIter),
		/* max_inner_iterations = */ SubIdx(clo.maxInnerIter),
		/* offset_value = */         KruskalValue(clo.offset),
		/* offset_tolerance = */     KruskalValue(clo.offsetTolerance),
		/* eps = */                  KruskalValue(clo.eps)
	};

  auto output_config = sparten::CpAprOutputOptions{
      /* progress_print_interval = */ clo.progressInterval
  };

  if (clo.scheduling == "static")
  {
		config.scheduling = 0;
  }
  else if (clo.scheduling == "dynamic")
  {
  	config.scheduling = 1;
  }

  if (clo.solver == "Multiplicative-Update")
  {
      auto cpApr = sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo.tolerance, clo.maxOuterIter, clo.maxInnerIter, clo.offset, clo.offsetTolerance, clo.progressInterval, clo.eps);
      cpApr.compute(kruskalOutput, sparseInput);
  }
  else if (clo.solver == "Damped-Newton")
  {
  	auto config_PDNR = sparten::DampedNewtonSolverConfiguration<numeric_traits>(config);
	  config_PDNR.eps_div_zero_grad = KruskalValue(clo.eps_div_zero_grad);
	  config_PDNR.log_zero_safeguard = KruskalValue(clo.log_zero_safeguard);
	  config_PDNR.row_tolerance = KruskalValue(clo.tolerance);
	  config_PDNR.eps_active_set = KruskalValue(clo.eps_active_set);
	  config_PDNR.max_backtrack_steps = SubIdx(clo.max_backtrack_steps);
	  config_PDNR.min_variable_nonzero_tolerance = KruskalValue(clo.min_variable_nonzero_tolerance);
	  config_PDNR.step_reduction_factor = KruskalValue(clo.step_reduction_factor);
	  config_PDNR.suff_decrease_tolerance = KruskalValue(clo.suff_decrease_tolerance);
	  config_PDNR.mu_initial = KruskalValue(clo.mu_initial);
	  config_PDNR.damping_increase_factor = KruskalValue(clo.damping_increase_factor);
	  config_PDNR.damping_decrease_factor = KruskalValue(clo.damping_decrease_factor);
	  config_PDNR.damping_increase_tolerance = KruskalValue(clo.damping_increase_tolerance);
	  config_PDNR.damping_decrease_tolerance = KruskalValue(clo.damping_decrease_tolerance);
	  config_PDNR.should_scale_hessian = clo.should_scale_hessian;
	  config_PDNR.solver_option =  clo.solver_option_pdnr;
	  config_PDNR.threshold_nonz_per_row =  clo.threshold_nonz_per_row_pdnr;
//	  config_PDNR.scheduling = std::string(clo.scheduling);

	  auto cpApr = sparten::DampedNewtonCpApr<numeric_traits>(config_PDNR, output_config);
	  cpApr.compute(kruskalOutput, sparseInput);
  }
  else if (clo.solver == "Quasi-Newton")
  {
	  auto config_PQNR = sparten::QuasiNewtonSolverConfiguration<numeric_traits>(config);
	  config_PQNR.eps_div_zero_grad = KruskalValue(clo.eps_div_zero_grad);
	  config_PQNR.log_zero_safeguard = KruskalValue(clo.log_zero_safeguard);
	  config_PQNR.row_tolerance = KruskalValue(clo.tolerance);
	  config_PQNR.eps_active_set = KruskalValue(clo.eps_active_set);
	  config_PQNR.max_backtrack_steps = SubIdx(clo.max_backtrack_steps);
	  config_PQNR.min_variable_nonzero_tolerance = KruskalValue(clo.min_variable_nonzero_tolerance);
	  config_PQNR.step_reduction_factor = KruskalValue(clo.step_reduction_factor);
	  config_PQNR.suff_decrease_tolerance = KruskalValue(clo.suff_decrease_tolerance);
	  config_PQNR.size_LBFGS = SubIdx(clo.size_LBFGS);
//	  config_PQNR.scheduling = std::string(clo.scheduling);

	  auto cpApr = sparten::QuasiNewtonCpApr<numeric_traits>(config_PQNR, output_config);
	  cpApr.compute(kruskalOutput, sparseInput);
  }
  else
  {
    throw std::runtime_error("Unknown CP-APR solver method.");
  }

}

class MainWorkflow
{
public:
  MainWorkflow(CommandLineOptions clo)
  {
    if (clo.sparseValueType.compare("int32") == 0 && clo.kruskalValueType.compare("float") == 0 && clo.globalOrdinalType.compare("int32") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint32_t, float, uint32_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int32") == 0 && clo.kruskalValueType.compare("double") == 0 && clo.globalOrdinalType.compare("int32") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint32_t, double, uint32_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int64") == 0 && clo.kruskalValueType.compare("float") == 0 && clo.globalOrdinalType.compare("int32") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint64_t, float, uint32_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int64") == 0 && clo.kruskalValueType.compare("double") == 0 && clo.globalOrdinalType.compare("int32") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint64_t, double, uint32_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int32") == 0 && clo.kruskalValueType.compare("float") == 0 && clo.globalOrdinalType.compare("int64") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint32_t, float, uint64_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int32") == 0 && clo.kruskalValueType.compare("double") == 0 && clo.globalOrdinalType.compare("int64") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint32_t, double, uint64_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int64") == 0 && clo.kruskalValueType.compare("float") == 0 && clo.globalOrdinalType.compare("int64") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint64_t, float, uint64_t, uint32_t>(clo);
    }
    else if (clo.sparseValueType.compare("int64") == 0 && clo.kruskalValueType.compare("double") == 0 && clo.globalOrdinalType.compare("int64") == 0 && clo.localOrdinalType.compare("int32") == 0)
    {
      mainRoutine<uint64_t, double, uint64_t, uint32_t>(clo);
    }

    else
    {
      throw std::runtime_error("Unsupported option");
    }
  }

private:
  template<typename SparseValue, typename KruskalValue, typename ElemIdx, typename SubIdx>
  void mainRoutine(CommandLineOptions clo)
  {
    sparten::Log &log = sparten::Log::new_log();

//    SimpleInput input(clo.inputOptions, SimpleInput::READ);
//    InputOptions inputOptions = input.read();

    sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> *sparseTensor = nullptr;
    // Timer
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    start = std::chrono::system_clock::now();
    log.print("Loading input file ...", sparten::Log::DEBUG_0);
    std::cout << "Loading sparse Tensor\n";
    auto *asciiInput = new sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo.workingPath + "/" + clo.inputFile, sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::READ, sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::FIXED_PRECISION, clo.precision, sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::INDEX_1);
    sparseTensor = asciiInput->read();
    delete asciiInput;
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    log.print("Loading input file ... completed (" + std::to_string(elapsed_seconds.count()) + ")", sparten::Log::DEBUG_0);
    std::cout << "Creating Kruskal Tensor\n";
    start = std::chrono::system_clock::now();
    log.print("Initializing solution ...", sparten::Log::DEBUG_0);
    auto *kruskalTensor = new sparten::KruskalTensor<KruskalValue, SubIdx>(sparseTensor->get_nDim(), clo.nComponent, sparseTensor->get_size_dim());
    sparten::KruskalTensorInitializer<KruskalValue, SubIdx> kruskalInit;
    if(!clo.randomSeed.empty())
    {
      kruskalInit.set_random_seed(std::stoi(clo.randomSeed));
    }
    kruskalInit.initialize(*kruskalTensor, sparten::KruskalTensorInitializer<KruskalValue, SubIdx>::RANDOM);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    log.print("Initializing solution ... completed (" + std::to_string(elapsed_seconds.count()) + ")", sparten::Log::DEBUG_0);
    log.print("nDim: " + std::to_string(sparseTensor->get_nDim()), sparten::Log::RELEASE);
    log.print("NNZ: " + std::to_string(sparseTensor->get_nElement()), sparten::Log::RELEASE);
    for(auto dim : sparseTensor->get_dim())
    {
      log.print("Size Mode " + std::to_string(static_cast<uint32_t>(dim)) + ": " + std::to_string(sparseTensor->get_size_dim()[dim]), sparten::Log::RELEASE);
    }

    log.print("-----Kruskal Tensor-----", sparten::Log::DEBUG_3);
    log.print(kruskalTensor->show(), sparten::Log::DEBUG_3);
    log.print("-----Kruskal Tensor-----", sparten::Log::DEBUG_3);
    log.print("", sparten::Log::DEBUG_3);
#if 0
    log.print("-----Sparse Tensor-----", sparten::Log::DEBUG_3);
    log.print(clo.logVerbosity.compare("DEBUG_3") == 0 ? sparseTensor->show() : "", sparten::Log::DEBUG_3);
    log.print("-----Sparse Tensor-----", sparten::Log::DEBUG_3);
#endif

    start = std::chrono::system_clock::now();
    log.print("Iterating improved solution ...", sparten::Log::DEBUG_0);
    cpApr_compute<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo, *kruskalTensor, *sparseTensor);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    log.print("Iterating improved solution ... completed (" + std::to_string(elapsed_seconds.count()) + ")", sparten::Log::DEBUG_0);

    start = std::chrono::system_clock::now();
    log.print("Writing results to file ...", sparten::Log::DEBUG_0);
    auto *asciiOutput = new sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo.workingPath + "/" + clo.outputFile, sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::OVERWRITE, sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::SCIENTIFIC, clo.precision, sparten::AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::INDEX_1);
    asciiOutput->write(*kruskalTensor);
    asciiOutput->close_file();
    delete asciiOutput;
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    log.print("Writing results to file ... completed (" + std::to_string(elapsed_seconds.count()) + ")", sparten::Log::DEBUG_0);
  }


  template<typename SparseValue, typename KruskalValue, typename ElemIdx, typename SubIdx>
  sparten::CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx> *cpApr_select(CommandLineOptions const &clo)
  {
    if (clo.solver.compare("Multiplicative-Update") == 0)
    {
      return new sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo.tolerance, clo.maxOuterIter, clo.maxInnerIter, clo.offset, clo.offsetTolerance, clo.progressInterval, clo.eps);
    }
    else if (clo.solver.compare("Damped-Newton") == 0)
    {
#if !defined(KOKKOS_ENABLE_CUDA)
   //   return new sparten::CpAprDampedNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo.tolerance, clo.maxOuterIter, clo.maxInnerIter, clo.offset, clo.offsetTolerance, clo.progressInterval, clo.eps);
#else
      throw std::runtime_error("CP-APR Damped-Newton solver method not working with CUDA yet.");
#endif
    }
    else if (clo.solver.compare("Quasi-Newton") == 0)
    {
#if !defined(KOKKOS_ENABLE_CUDA)
   //   return new sparten::CpAprQuasiNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>(clo.tolerance, clo.maxOuterIter, clo.maxInnerIter, clo.offset, clo.offsetTolerance, clo.progressInterval, clo.eps);
#else
      throw std::runtime_error("CP-APR Damped-Newton solver method not working with CUDA yet.");
#endif
    }
    else
    {
      throw std::runtime_error("Unknown CP-APR solver method.");
    }
  }
};
