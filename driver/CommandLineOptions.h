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

#include <iostream>
#include <string>
#include <Sparten_Configure.h>
#include <sparten/Log.h>
#include <sstream>

class CommandLineOptions
{
public:
  CommandLineOptions() :

  /* Required Arguments */
	  nComponent(5),
	  solver("Multiplicative-Update"),
	  randomSeed(""),
	  inputFile("tensor.tns"),
	  outputFile("tensor.out"),
	  scheduling("static"),

	  /* Convergence Arguments */
	  tolerance(1e-4),
	  maxOuterIter(20),
	  maxInnerIter(10),
	  eps(1e-10),
	  offset(1e-2),
	  offsetTolerance(1e-10),

	  /* Row Subproblem Arguments */
	  eps_div_zero_grad(1e-10),
	  log_zero_safeguard(1e-16),
	  eps_active_set(1e-3),
	  max_backtrack_steps(10),
	  min_variable_nonzero_tolerance(1e-7),
	  step_reduction_factor(0.5),
	  suff_decrease_tolerance(1e-4),
	  mu_initial(1e-5),
	  damping_increase_factor(3.5),
	  damping_decrease_factor(0.2857142857142857), // 2.0/7.0
	  damping_increase_tolerance(0.25),
	  damping_decrease_tolerance(0.75),
	  should_scale_hessian(false),
	  size_LBFGS(3),

	  /* PDNR Options */
     solver_option_pdnr(1),
     threshold_nonz_per_row_pdnr(8192),

	  /* Data Types */
	  localOrdinalType("int32"),
	  globalOrdinalType("int32"),
	  sparseValueType("int32"),
	  kruskalValueType("float"),

	  /* Data Input */
	  indexStyle("1"),

	  /* Data Output */
	  precision(16),

	  /* Logging */
	  progressInterval(20),
	  logFile(""),
	  logVerbosity("release"),

	  /* System */
	  mode(""),
	  workingPath("."),
	  asciiFormat(""),

	  /* Option File */
	  inputOptionsFile("inputOptions.txt")

  {;}

  void print_help(int argvc, char **argv)
  {
    std::cout << "SparTen v" << SPARTEN_VERSION << std::endl;
    std::cout << "CXX Compiler: " << SPARTEN_CXX_COMPILER << std::endl;
    std::cout << "C Compiler: " << SPARTEN_CXX_COMPILER << std::endl;
    std::cout << std::endl;

    std::cout << "SparTen usage: " << argv[0] << " [arg1 value1 arg2 value2 ...]" <<
    std::endl << std::endl <<
    "NOTE: All arguments are optional, using defeault values if none are provided on the command line." << std::endl << std::endl <<
    "COMMON ARGUMENTS:" << std::endl <<
    "     --nComponent                      INT          Rank of factorization to compute [Default: 5] " << std::endl <<
    "     --solver                          STRING       Name of solver method to use [Default: Multiplicative-Update] " << std::endl <<
    "           Multiplicative-Update:  slower convergence, less computation per outer iteration " << std::endl <<
    "           Quasi-Newton:           faster convergence, uses first order derivatives, more computation per outer iteration " << std::endl <<
    "           Damped-Newton:          faster convergence, uses second order derivatives, more computation per outer iteration " << std::endl <<
    "     --input-file                      FILENAME     Name of file containing tensor data [Default: tensor.tns]" << std::endl <<
    "     --output-file                     FILENAME     Name of file where tensor factorization will be stored [Default: tensor.out]" << std::endl <<
    "     --working-path                    PATHNAME     Name of path where tensor input and output files will be found [Default: ./]" << std::endl <<
    "     --randomSeed                      STRING       Value of seed to use in random generator [Default: 131] " << std::endl <<
    "     --scheduling                      STRING       Specify Kokkos scheduling policy for work items {Static, Dynamic} [Default: Static] " << std::endl <<
    std::endl <<
    "ADVANCED ARGUMENTS:" << std::endl <<
    "   CONVERGENCE CRITERIA: " << std::endl <<
    "     --tolerance                       FLOAT        Stopping tolerance [Default: 1e-4] " << std::endl <<
    "     --maxOuterIter                    INT          Maximum number of outer iterations to perform [Default: 20] " << std::endl <<
    "     --maxInnerIter                    INT          Maximum number of inner iterations to perform [Default: 10] " << std::endl <<
    "     --offset                          FLOAT        Offset for complementary slackness [Default: 1e-2] " << std::endl <<
    "     --offsetTolerance                 FLOAT        Tolerance for complementary slackness [Default: 1e-10] " << std::endl <<
    "     --eps                             FLOAT        Safeguard against divide by zero [Default: 1e-10] " << std::endl <<
    "   ROW SUBPROBLEM CONFIGURATIONS: " << std::endl <<
    "    LINE SEARCH " << std::endl <<
    "     --eps_div_zero_grad               FLOAT        Safeguard against divide by zero [Default: 1e-10] " << std::endl <<
    "     --log_zero_safeguard              FLOAT        Defines 0log0 = 0 [Default: 1e-16 ] " << std::endl <<
    "     --eps_active_set                  FLOAT        Tolerance defining active (nonzero) variables [Default: 1e-3] " << std::endl <<
    "     --max_backtrack_steps             INT          Maximum number of backtracking steps in line search [Default: 10] " << std::endl <<
    "     --min_variable_nonzero_tolerance  FLOAT        Tolerance for nonzero step length [Default: 1e-7] " << std::endl <<
    "     --step_reduction_factor           FLOAT        Factor to reduce line search step size [Default: 0.5] " << std::endl <<
    "     --suff_decrease_tolerance         FLOAT        Tolerance to ensure the next iterate decreases objective [Default: 1e-4] " << std::endl <<
    "    PDN-R " << std::endl <<
    "     --mu_initial                      FLOAT        Initial guess of damping parameter [Default: 1e-5] " << std::endl <<
    "     --damping_increase_factor         FLOAT        Factor to increase damping parameter [Default: 3.5] " << std::endl <<
    "     --damping_decrease_factor         FLOAT        Factor to decrease damping parameter [Default: 0.2857142857142857] " << std::endl <<
    "     --damping_increase_tolerance      FLOAT        Tolerance to increase damping parameter [Default: 0.25] " << std::endl <<
    "     --damping_decrease_tolerance      FLOAT        Tolerance to decrease damping parameter [Default: 0.75] " << std::endl <<
    "     --should_scale_hessian            BOOL         Scale the Hessian matrix [Default: 0] " << std::endl <<
    "     --PDNR_solver_option              INT          PDNR internal algorithm options 1: No All-kernel, 2: Hybrid" << std::endl <<
    "     --PDNR_nonz_per_row_for_switch    INT          The least nuumber of nonzero per row to schdule All-GPU/cores Kernel" << std::endl <<
    "    PQN-R" << std::endl <<
    "     --size_LBFGS                      INT          Number of limited-memory update pairs [Default: 3] " << std::endl <<
    "   DATA TYPES: " << std::endl <<
    "     --global-ordinal-type             TYPE         Type of ordinal used for global indexing {int32, int64} [Default: int32] " << std::endl <<
    "     --local-ordinal-type              TYPE         Type of ordinal used for local indexing {int32, int64} [Default: int32] " << std::endl <<
    "     --sparse-value-type               TYPE         Type of value used for input tensor data {int32, int64} [Default: int32] " << std::endl <<
    "     --kruskal-value-type              TYPE         Type of value used for output tensor data {float, double} [Default: float] " << std::endl <<
    "   DATA INPUT: " << std::endl <<
    "     --index-style                     STYLE        Index starting value for tensor data {1,0} [Default: 1]" << std::endl <<
    "   DATA OUTPUT: " << std::endl <<
    "     --precision                       INT          Number of significant digits to store in output file [Default: 16] " << std::endl <<
    "   LOGGING: " << std::endl <<
    "     --progressInterval                INT          Number of outer iterations performed before displaying status [Default: 20] " << std::endl <<
    "     --log-file                        FILENAME     Store log information in FILENAME [Default: send to STDOUT]" << std::endl <<
    "     --log-verbosity                   VERBOSITY    Level of verbosity to store in log file: [Default: release]" << std::endl <<
    "           quiet:    no output" << std::endl <<
    "           release:  information regarding the SparTen factorization" << std::endl <<
    "           debug_0:  some debug information (mainly used by developers)" << std::endl <<
    "           debug_1:  more debug information (mainly used by developers)" << std::endl <<
    "           debug_2:  all debug information (mainly used by developers)" << std::endl <<
    "           debug_3:  all (extreme level) debug information (mainly used by developers)" << std::endl <<
    "   OPTION FILE: " << std::endl <<
    "     --input-options-file              FILENAME     Name of file containing solver options [Default: inputOptions.txt]" << std::endl <<
    "           EXAMPLE: (text file containing space-separated argument-value pairs per line)" << std::endl <<
    "              tolerance 1e-6" << std::endl <<
    "              maxOuterIter 100" << std::endl <<
    "              maxInnerIter 20" << std::endl <<
    "           NOTE: arguments specified in the file will overwrite those specified on the command line" << std::endl <<
    std::endl;
  }

  void print_options(std::string config) const
  {
    sparten::Log &log = sparten::Log::new_log();

    // These options may have values below the 6 digit cutoff of std::to_string so should be printed another way
	  std::ostringstream toleranceOss;
	  std::ostringstream epsOss;
	  std::ostringstream offsetOss;
	  std::ostringstream offsetToleranceOss;
	  std::ostringstream eps_div_zero_gradOss;
	  std::ostringstream log_zero_safeguardOss;
	  std::ostringstream eps_active_setOss;
	  std::ostringstream min_variable_nonzero_toleranceOss;
	  std::ostringstream suff_decrease_toleranceOss;
	  std::ostringstream mu_initialOss;

	  toleranceOss << tolerance;
	  epsOss << eps;
	  offsetOss << offset;
	  offsetToleranceOss << offsetTolerance;
	  eps_div_zero_gradOss << eps_div_zero_grad;
	  log_zero_safeguardOss << log_zero_safeguard;
	  eps_active_setOss << eps_active_set;
	  min_variable_nonzero_toleranceOss << min_variable_nonzero_tolerance;
	  suff_decrease_toleranceOss << suff_decrease_tolerance;
	  mu_initialOss << mu_initial;

	  std::string toleranceStr = toleranceOss.str();
	  std::string epsStr = epsOss.str();
	  std::string offsetStr = offsetOss.str();
	  std::string offsetToleranceStr = offsetToleranceOss.str();
	  std::string eps_div_zero_gradStr = eps_div_zero_gradOss.str();
	  std::string log_zero_safeguardStr = log_zero_safeguardOss.str();
	  std::string eps_active_setStr = eps_active_setOss.str();
	  std::string min_variable_nonzero_toleranceStr = min_variable_nonzero_toleranceOss.str();
	  std::string suff_decrease_toleranceStr = suff_decrease_toleranceOss.str();
	  std::string mu_initialStr = mu_initialOss.str();

	  // Random seed's default value is hardcoded later on, so print_options() doesn't know a priori
	  std::string randomSeedStr = !randomSeed.empty() ? randomSeed : "131";

	  log.print("Configuration:", sparten::Log::RELEASE);
	  log.print("  Input File: " + inputFile, sparten::Log::RELEASE);
	  log.print("  Output File: " + outputFile, sparten::Log::RELEASE);
	  log.print("  Working Path: " + workingPath, sparten::Log::RELEASE);
	  log.print("  Scheduling Policy: " + scheduling, sparten::Log::RELEASE);
	  log.print("  Local Ordinal Type: " + localOrdinalType, sparten::Log::RELEASE);
	  log.print("  Global Ordinal Type: " + globalOrdinalType, sparten::Log::RELEASE);
	  log.print("  Sparse Value Type: " + sparseValueType, sparten::Log::RELEASE);
	  log.print("  Kruskal Value Type: " + kruskalValueType, sparten::Log::RELEASE);
	  log.print("  Log File: " + logFile, sparten::Log::RELEASE);
	  log.print("  Log Verbosity: " + (logVerbosity.empty() ? "release" : logVerbosity), sparten::Log::RELEASE);
	  log.print("  progressInterval: " + std::to_string(progressInterval), sparten::Log::RELEASE);
	  log.print("  precision: " + std::to_string(precision), sparten::Log::RELEASE);
	  log.print("  randomSeed: " + randomSeedStr, sparten::Log::RELEASE);
	  log.print("  nComponent: " + std::to_string(nComponent), sparten::Log::RELEASE);
	  log.print("  solver: " + solver, sparten::Log::RELEASE);
	  log.print("  tolerance: " + toleranceStr, sparten::Log::RELEASE);
	  log.print("  maxOuterIter: " + std::to_string(maxOuterIter), sparten::Log::RELEASE);
	  log.print("  maxInnerIter: " + std::to_string(maxInnerIter), sparten::Log::RELEASE);
	  log.print("  eps: " + epsStr, sparten::Log::RELEASE);
	  log.print("  offset: " + offsetStr, sparten::Log::RELEASE);
	  log.print("  offsetTolerance: " + offsetToleranceStr, sparten::Log::RELEASE);
	  if (solver == "Damped-Newton" || solver == "Quasi-Newton") {
		  log.print("  eps_div_zero_grad: " + eps_div_zero_gradStr, sparten::Log::RELEASE);
		  log.print("  log_zero_safeguard: " + log_zero_safeguardStr, sparten::Log::RELEASE);
		  log.print("  eps_active_set: " + eps_active_setStr, sparten::Log::RELEASE);
		  log.print("  max_backtrack_steps: " + std::to_string(max_backtrack_steps), sparten::Log::RELEASE);
		  log.print("  min_variable_nonzero_tolerance: " + min_variable_nonzero_toleranceStr, sparten::Log::RELEASE);
		  log.print("  step_reduction_factor: " + std::to_string(step_reduction_factor), sparten::Log::RELEASE);
		  log.print("  suff_decrease_tolerance: " + suff_decrease_toleranceStr, sparten::Log::RELEASE);
		  if (solver == "Damped-Newton") {
			  log.print("  mu_initial: " + mu_initialStr, sparten::Log::RELEASE);
			  log.print("  damping_increase_factor: " + std::to_string(damping_increase_factor), sparten::Log::RELEASE);
			  log.print("  damping_decrease_factor: " + std::to_string(damping_decrease_factor), sparten::Log::RELEASE);
			  log.print("  damping_increase_tolerance: " + std::to_string(damping_increase_tolerance), sparten::Log::RELEASE);
			  log.print("  damping_decrease_tolerance: " + std::to_string(damping_decrease_tolerance), sparten::Log::RELEASE);
			  log.print("  should_scale_hessian: " + std::to_string(should_scale_hessian), sparten::Log::RELEASE);
           log.print("  solver_option: " + std::to_string(solver_option_pdnr), sparten::Log::RELEASE);
           log.print("  threshold_nonz_per_row: " + std::to_string(threshold_nonz_per_row_pdnr), sparten::Log::RELEASE);
		  }
		  else {
			  log.print("  size_LBFGS: " + std::to_string(size_LBFGS), sparten::Log::RELEASE);
		  }
    }
    log.print("", sparten::Log::RELEASE);
  }

	uint64_t nComponent;
	std::string solver;
	std::string randomSeed;
	std::string inputFile;
	std::string outputFile;
	std::string scheduling;

	double tolerance;
	uint64_t maxOuterIter;
	uint64_t maxInnerIter;
	double eps;
	double offset;
	double offsetTolerance;
	double eps_div_zero_grad;
	double log_zero_safeguard;
	double eps_active_set;
	uint64_t max_backtrack_steps;
	double min_variable_nonzero_tolerance;
	double step_reduction_factor;
	double suff_decrease_tolerance;
	double mu_initial;
	double damping_increase_factor;
	double damping_decrease_factor;
	double damping_increase_tolerance;
	double damping_decrease_tolerance;
	bool should_scale_hessian;
	uint64_t size_LBFGS;

	uint64_t solver_option_pdnr;
	uint64_t threshold_nonz_per_row_pdnr;

	std::string localOrdinalType;
	std::string globalOrdinalType;
	std::string sparseValueType;
	std::string kruskalValueType;
	std::string indexStyle;
	uint64_t precision;
	uint64_t progressInterval;
	std::string logFile;
	std::string logVerbosity;
	std::string mode;
	std::string workingPath;
	std::string asciiFormat;
	std::string inputOptionsFile;
};
