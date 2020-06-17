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

#include <CommandLineOptions.h>
#include <MainWorkflow.h>
#include <SimpleInput.h>
#include <Sparten_Configure.h>
#include <sparten/AsciiIO.h>
#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/Log.h>
#include <sparten/SparseTensor.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

//! Convenience function to get std::string with args
std::string get_argument(int argc, char const **argv, int i)
{
  std::string tmp(argv[i]);
  return tmp;
}

//! Convenience function to get std::string with args
std::string get_argument(int argc, char **argv, std::string const &arg)
{
  bool found = false;
  std::string value;

  CommandLineOptions clo;

  if (argc % 2 == 0)
  {
    clo.print_help(argc, argv);
    throw std::runtime_error("Invalid number of command line options.");
  }

  for (int i = 1; i < argc; i += 2)
  {
    if (arg.compare(argv[i]) == 0)
    {
      found = true;
      value = argv[i + 1];
    }
  }

  if (found)
  {
    return value;
  } else
  {
    clo.print_help(argc, argv);
    throw std::runtime_error(arg + " not found.");
  }
}

//! Convenience function to get std::string with optional args
std::string get_argument_optional(int argc, char **argv, std::string const &arg)
{
  bool found = false;
  std::string value;
  CommandLineOptions clo;

  if (argc % 2 == 0)
  {
    clo.print_help(argc, argv);
    throw std::runtime_error("Invalid number of command line options.");
  }

  for (int i = 1; i < argc; i += 2)
  {
    if (arg.compare(argv[i]) == 0)
    {
      found = true;
      value = argv[i + 1];
    }
  }

  if (found)
  {
    return value;
  }
  else
  {
    return "";
  }
}

//! Convenience function to get command line options
CommandLineOptions get_command_line_options(int argc, char **argv)
{
  CommandLineOptions clo;

  if (argc < 2)
  {
    clo.print_help(argc, argv);
    throw std::runtime_error("No arguments provided");
  }

	std::string inputOptionsFile = get_argument_optional(argc, argv, "--input-options-file");
	std::string inputFile = get_argument_optional(argc, argv, "--input-file");
	std::string outputFile = get_argument_optional(argc, argv, "--output-file");
	std::string workingPath = get_argument_optional(argc, argv, "--working-path");
	std::string scheduling = get_argument_optional(argc, argv, "--scheduling");
	std::string localOrdinalType = get_argument_optional(argc, argv, "--local-ordinal-type");
	std::string globalOrdinalType = get_argument_optional(argc, argv, "--global-ordinal-type");
	std::string sparseValueType = get_argument_optional(argc, argv, "--sparse-value-type");
	std::string kruskalValueType = get_argument_optional(argc, argv, "--kruskal-value-type");
	std::string logFile = get_argument_optional(argc, argv, "--log-file");
	std::string logVerbosity = get_argument_optional(argc, argv, "--log-verbosity");
	std::string indexStyle = get_argument_optional(argc, argv, "--index-style");
	std::string solver = get_argument_optional(argc, argv, "--solver");
	std::string randomSeed = get_argument_optional(argc, argv, "--randomSeed");

	clo.inputOptionsFile = inputOptionsFile.empty() ? clo.inputOptionsFile : inputOptionsFile;
	clo.inputFile = inputFile.empty() ? clo.inputFile : inputFile;
	clo.outputFile = outputFile.empty() ? clo.outputFile : outputFile;
	clo.workingPath = workingPath.empty() ? clo.workingPath : workingPath;
	clo.scheduling = scheduling.empty() ? clo.scheduling : scheduling;
	clo.localOrdinalType = localOrdinalType.empty() ? clo.localOrdinalType : localOrdinalType;
	clo.globalOrdinalType = globalOrdinalType.empty() ? clo.globalOrdinalType : globalOrdinalType;
	clo.sparseValueType = sparseValueType.empty() ? clo.sparseValueType : sparseValueType;
	clo.kruskalValueType = kruskalValueType.empty() ? clo.kruskalValueType : kruskalValueType;
	clo.logFile = logFile.empty() ? clo.logFile : logFile;
	clo.logVerbosity = logVerbosity.empty() ? clo.logVerbosity : logVerbosity;
	clo.indexStyle = indexStyle.empty() ? clo.indexStyle : indexStyle;
	clo.solver = solver.empty() ? clo.solver : solver;
	clo.randomSeed = randomSeed.empty() ? clo.randomSeed : randomSeed;

	std::string toleranceStr = get_argument_optional(argc, argv, "--tolerance");
	std::string maxOuterIterStr = get_argument_optional(argc, argv, "--maxOuterIter");
	std::string maxInnerIterStr = get_argument_optional(argc, argv, "--maxInnerIter");
	std::string offsetStr = get_argument_optional(argc, argv, "--offset");
	std::string offsetToleranceStr = get_argument_optional(argc, argv, "--offsetTolerance");
	std::string progressIntervalStr = get_argument_optional(argc, argv, "--progressInterval");
	std::string epsStr = get_argument_optional(argc, argv, "--eps");
	std::string precisionStr = get_argument_optional(argc, argv, "--precision");
	std::string nComponentStr = get_argument_optional(argc, argv, "--nComponent");
	std::string eps_div_zero_gradStr = get_argument_optional(argc, argv, "--eps_div_zero_grad");
	std::string log_zero_safeguardStr = get_argument_optional(argc, argv, "--log_zero_safeguard");
	std::string eps_active_setStr = get_argument_optional(argc, argv, "--eps_active_set");
	std::string max_backtrack_stepsStr = get_argument_optional(argc, argv, "--max_backtrack_steps");
	std::string min_variable_nonzero_toleranceStr = get_argument_optional(argc, argv, "--min_variable_nonzero_tolerance");
	std::string step_reduction_factorStr = get_argument_optional(argc, argv, "--step_reduction_factor");
	std::string suff_decrease_toleranceStr = get_argument_optional(argc, argv, "--suff_decrease_tolerance");
	std::string mu_initialStr = get_argument_optional(argc, argv, "--mu_initial");
	std::string damping_increase_factorStr = get_argument_optional(argc, argv, "--damping_increase_factor");
	std::string damping_decrease_factorStr = get_argument_optional(argc, argv, "--damping_decrease_factor");
	std::string damping_increase_toleranceStr = get_argument_optional(argc, argv, "--damping_increase_tolerance");
	std::string damping_decrease_toleranceStr = get_argument_optional(argc, argv, "--damping_decrease_tolerance");
	std::string should_scale_hessianStr = get_argument_optional(argc, argv, "--should_scale_hessian");
	std::string size_LBFGSStr = get_argument_optional(argc, argv, "--size_LBFGS");
	std::string solver_option_pdnr_str = get_argument_optional(argc, argv, "--PDNR_solver_option");
	std::string threshold_nonz_per_row_pdnr_str = get_argument_optional(argc, argv, "--PDNR_nonz_per_row_for_switch");

	std::istringstream toleranceIss(toleranceStr); toleranceIss >> clo.tolerance;
	std::istringstream maxOuterIterIss(maxOuterIterStr); maxOuterIterIss >> clo.maxOuterIter;
	std::istringstream maxInnerIterIss(maxInnerIterStr); maxInnerIterIss >> clo.maxInnerIter;
	std::istringstream offsetIss(offsetStr); offsetIss >> clo.offset;
	std::istringstream offsetToleranceIss(offsetToleranceStr); offsetToleranceIss >> clo.offsetTolerance;
	std::istringstream progressIntervalIss(progressIntervalStr); progressIntervalIss >> clo.progressInterval;
	std::istringstream epsIss(epsStr); epsIss >> clo.eps;
	std::istringstream precisionIss(precisionStr); precisionIss >> clo.precision;
	std::istringstream nComponentIss(nComponentStr); nComponentIss >> clo.nComponent;
	std::istringstream eps_div_zero_gradIss(eps_div_zero_gradStr); eps_div_zero_gradIss >> clo.eps_div_zero_grad;
	std::istringstream log_zero_safeguardIss(log_zero_safeguardStr); log_zero_safeguardIss >> clo.log_zero_safeguard;
	std::istringstream eps_active_setIss(eps_active_setStr); eps_active_setIss >> clo.eps_active_set;
	std::istringstream max_backtrack_stepsIss(max_backtrack_stepsStr); max_backtrack_stepsIss >> clo.max_backtrack_steps;
	std::istringstream min_variable_nonzero_toleranceIss(min_variable_nonzero_toleranceStr); min_variable_nonzero_toleranceIss >> clo.min_variable_nonzero_tolerance;
	std::istringstream step_reduction_factorIss(step_reduction_factorStr); step_reduction_factorIss >> clo.step_reduction_factor;
	std::istringstream suff_decrease_toleranceIss(suff_decrease_toleranceStr); suff_decrease_toleranceIss >> clo.suff_decrease_tolerance;
	std::istringstream mu_initialIss(mu_initialStr); mu_initialIss >> clo.mu_initial;
	std::istringstream damping_increase_factorIss(damping_increase_factorStr); damping_increase_factorIss >> clo.damping_increase_factor;
	std::istringstream damping_decrease_factorIss(damping_decrease_factorStr); damping_decrease_factorIss >> clo.damping_decrease_factor;
	std::istringstream damping_increase_toleranceIss(damping_increase_toleranceStr); damping_increase_toleranceIss >> clo.damping_increase_tolerance;
	std::istringstream damping_decrease_toleranceIss(damping_decrease_toleranceStr); damping_decrease_toleranceIss >> clo.damping_decrease_tolerance;
	std::istringstream should_scale_hessianIss(should_scale_hessianStr); should_scale_hessianIss >> clo.should_scale_hessian;
	std::istringstream size_LBFGSIss(size_LBFGSStr); size_LBFGSIss >> clo.size_LBFGS;
	std::istringstream solver_option_pdnr_ss(solver_option_pdnr_str); solver_option_pdnr_ss >> clo.solver_option_pdnr;
  std::istringstream threshold_nonz_per_row_pdnr_ss(threshold_nonz_per_row_pdnr_str); threshold_nonz_per_row_pdnr_ss >> clo.threshold_nonz_per_row_pdnr;

  if (!inputOptionsFile.empty())
  {
    SimpleInput input(clo.inputOptionsFile, SimpleInput::READ);
    InputOptions inputOptions = input.read();

	  clo.tolerance = !toleranceStr.empty() ? clo.tolerance : inputOptions.tolerance;
	  clo.maxOuterIter = !maxOuterIterStr.empty() ? clo.maxOuterIter : inputOptions.maxOuterIter;
	  clo.maxInnerIter = !maxInnerIterStr.empty() ? clo.maxInnerIter : inputOptions.maxInnerIter;
	  clo.offset = !offsetStr.empty() ? clo.offset : inputOptions.offset;
	  clo.offsetTolerance = !offsetToleranceStr.empty() ? clo.offsetTolerance : inputOptions.offsetTolerance;
	  clo.progressInterval = !progressIntervalStr.empty() ? clo.progressInterval : inputOptions.progressInterval;
	  clo.eps = !epsStr.empty() ? clo.eps : inputOptions.eps;
	  clo.precision = !precisionStr.empty() ? clo.precision : inputOptions.precision;
	  clo.nComponent = !nComponentStr.empty() ? clo.nComponent : inputOptions.nComponent;
	  clo.solver = !clo.solver.empty() ? inputOptions.solver : clo.solver;
	  clo.randomSeed = !clo.randomSeed.empty() ? clo.randomSeed : inputOptions.randomSeed;

	  clo.eps_div_zero_grad = !eps_div_zero_gradStr.empty() ? clo.eps_div_zero_grad : inputOptions.eps_div_zero_grad;
	  clo.log_zero_safeguard = !log_zero_safeguardStr.empty() ? clo.log_zero_safeguard : inputOptions.log_zero_safeguard;
	  clo.eps_active_set = !eps_active_setStr.empty() ? clo.eps_active_set : inputOptions.eps_active_set;
	  clo.max_backtrack_steps = !max_backtrack_stepsStr.empty() ? clo.max_backtrack_steps : inputOptions.max_backtrack_steps;
	  clo.min_variable_nonzero_tolerance = !min_variable_nonzero_toleranceStr.empty() ? clo.min_variable_nonzero_tolerance : inputOptions.min_variable_nonzero_tolerance;
	  clo.step_reduction_factor = !step_reduction_factorStr.empty() ? clo.step_reduction_factor : inputOptions.step_reduction_factor;
	  clo.suff_decrease_tolerance = !suff_decrease_toleranceStr.empty() ? clo.suff_decrease_tolerance : inputOptions.suff_decrease_tolerance;
	  clo.mu_initial = !mu_initialStr.empty() ? clo.mu_initial : inputOptions.mu_initial;
	  clo.damping_increase_factor = !damping_increase_factorStr.empty() ? clo.damping_increase_factor : inputOptions.damping_increase_factor;
	  clo.damping_decrease_factor = !damping_decrease_factorStr.empty() ? clo.damping_decrease_factor : inputOptions.damping_decrease_factor;
	  clo.damping_increase_tolerance = !damping_increase_toleranceStr.empty() ? clo.damping_increase_tolerance : inputOptions.damping_increase_tolerance;
	  clo.damping_decrease_tolerance = !damping_decrease_toleranceStr.empty() ? clo.damping_decrease_tolerance : inputOptions.damping_decrease_tolerance;
	  clo.should_scale_hessian = !should_scale_hessianStr.empty() ? clo.should_scale_hessian : inputOptions.should_scale_hessian;
	  clo.size_LBFGS = !size_LBFGSStr.empty() ? clo.size_LBFGS : inputOptions.size_LBFGS;
	  clo.solver_option_pdnr = !solver_option_pdnr_str.empty() ? clo.solver_option_pdnr : inputOptions.solver_option_pdnr;
	  clo.threshold_nonz_per_row_pdnr = !threshold_nonz_per_row_pdnr_str.empty() ?  clo.threshold_nonz_per_row_pdnr : inputOptions.threshold_nonz_per_row_pdnr;

	  clo.inputFile = inputOptions.inputFile.empty() ? clo.inputFile : inputOptions.inputFile;
	  clo.outputFile = inputOptions.outputFile.empty() ? clo.outputFile : inputOptions.outputFile;
	  clo.workingPath = inputOptions.workingPath.empty() ? clo.workingPath : inputOptions.workingPath;
	  clo.scheduling = inputOptions.scheduling.empty() ? clo.scheduling : inputOptions.scheduling;
	  clo.localOrdinalType = inputOptions.localOrdinalType.empty() ? clo.localOrdinalType : inputOptions.localOrdinalType;
	  clo.globalOrdinalType = inputOptions.globalOrdinalType.empty() ? clo.globalOrdinalType : inputOptions.globalOrdinalType;
	  clo.sparseValueType = inputOptions.sparseValueType.empty() ? clo.sparseValueType : inputOptions.sparseValueType;
	  clo.kruskalValueType = inputOptions.kruskalValueType.empty() ? clo.kruskalValueType : inputOptions.kruskalValueType;
	  clo.logFile = inputOptions.logFile.empty() ? clo.logFile : inputOptions.logFile;
	  clo.logVerbosity = inputOptions.logVerbosity.empty() ? clo.logVerbosity : inputOptions.logVerbosity;
	  clo.indexStyle = inputOptions.indexStyle.empty() ? clo.indexStyle : inputOptions.indexStyle;
  }

  return clo;
}

//! Convenience function to get log type (screen, file, ...)
sparten::Log::LogType get_log_type(CommandLineOptions const &clo)
{
  if (clo.logFile.empty())
  {
    return sparten::Log::TO_SCREEN;
  } else
  {
    return sparten::Log::TO_FILE;
  }
}

//! Convenience function to get log verbosity
sparten::Log::Verbosity get_log_verbosity(CommandLineOptions const &clo)
{
  if (clo.logVerbosity.compare("quiet") == 0)
  {
    return sparten::Log::QUIET;
  }
  else if (clo.logVerbosity.compare("release") == 0)
  {
    return sparten::Log::RELEASE;
  }
  else if (clo.logVerbosity.compare("debug_0") == 0)
  {
    return sparten::Log::DEBUG_0;
  }
  else if (clo.logVerbosity.compare("debug_1") == 0)
  {
    return sparten::Log::DEBUG_1;
  }
  else if (clo.logVerbosity.compare("debug_2") == 0)
  {
    return sparten::Log::DEBUG_2;
  }
  else if (clo.logVerbosity.compare("debug_3") == 0)
  {
    return sparten::Log::DEBUG_3;
  }
  else
  {
    if (!clo.logFile.empty())
    {
      throw std::runtime_error("Invalid log verbosity level provided.");
    }
    else
    {
      return sparten::Log::RELEASE;;
    }
  }
}

int main(int argc, char **argv)
{
  //// Initialization ////
  Kokkos::initialize(argc, argv);
  // TODO: do we need to print the following? Maybe this should be a command line argument
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  try
  {
    //// Main program ////
    // Get command line options
    CommandLineOptions clo = get_command_line_options(argc, argv);

    // Setup logging
    sparten::Log &log = sparten::Log::new_log(get_log_type(clo), clo.workingPath + "/" + clo.logFile, get_log_verbosity(clo));

    log.print("Sparten Logfile -- Version " + std::string(SPARTEN_VERSION) + "\n", sparten::Log::RELEASE);

    clo.print_options(clo.solver);

    // Date + Current Time + Timer
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::stringstream ss_start;
    ss_start << "Started computation at " << std::ctime(&start_time);
    log.print(ss_start.str(), sparten::Log::RELEASE);

    // Do stuff here
    MainWorkflow mainWorkflow(clo);

    //// Finalization ////
    end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::stringstream ss_finish;
    ss_finish << "Finished computation at " << std::ctime(&end_time);
    log.print(ss_finish.str(), sparten::Log::RELEASE);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    log.print("Elapsed time: " + std::to_string(elapsed_seconds.count()), sparten::Log::RELEASE);

    log.finalize();
  }
  catch (std::exception &e)
  {
    std::cout << e.what() << std::endl;
  }

  //// Finalization ////
  Kokkos::finalize();

  return 0;
}
