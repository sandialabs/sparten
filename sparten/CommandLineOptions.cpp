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

#include <iostream>
#include <string>
#include <Sparten_Configure.h>
#include <sparten/Log.h>
#include <sstream>
#include <unistd.h>
#include <sparten/CommandLineOptions.h>

sparten::CommandLineOptions::CommandLineOptions() :
    nComponent(5),
    solver("Multiplicative-Update"),
    randomSeed(131),
    inputFile(""),
    outputFile(""),
    scheduling("static"),
    tolerance(1e-4),
    maxOuterIter(20),
    maxInnerIter(10),
    eps(1e-10),
    offset(1e-2),
    offsetTolerance(1e-10),
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
    solver_option_pdnr(false),
    threshold_nonz_per_row_pdnr(8192),
    localOrdinalType("int32"),
    globalOrdinalType("int32"),
    sparseValueType("int32"),
    kruskalValueType("double"),
    indexStyle("1"),
    precision(16),
    progressInterval(1),
    debug(false),
    logFile(""),
    print_level(1),
    mode(""),
    asciiFormat(""),
    // inputOptionsFile("inputOptions.txt"),
    init_file("")
{}

void sparten::CommandLineOptions::print_help(std::ostream &out) {
    out << "Generic options: " << std::endl;
    out << "  --method         <string>  CPAPR solver: Multiplicative-Update, Quasi-Newton, Damped-Newton" << std::endl;
    out << "  --rank           <int>     Rank of factorization to compute" << std::endl;
    out << "  --seed           <int>     Seed for random number generator used in initial guess" << std::endl;
    out << "  --maxiters       <int>     Maximum number of outer iterations to perform" << std::endl;
    out << "  --maxinneriters  <int>     Maximum number of inner iterations to perform" << std::endl;
    out << "  --tol            <float>   Stopping tolerance" << std::endl;
    out << "  --printitn       <int>     Print every <int>th outer iterations" << std::endl;
    out << std::endl;
    out << "MU options: " << std::endl;
    out << "  --mu-kappa     <float>  Offset for complementary slackness" << std::endl;
    out << "  --mu-kappatol  <float>  Tolerance on complementary slackness" << std::endl;
    out << "  --mu-eps       <float>  Safeguard against divide by zero" << std::endl;
    out << std::endl;
    out << "Row subproblem options: " << std::endl;
    out << "  --row-eps-zero-guard           <float>  Guard against divide by zero" << std::endl;
    out << "  --row-log-zero-guard           <float>  Guard against undefined 0*log(0)" << std::endl;
    out << "  --row-eps-active-set           <float>  Tolerance defining active (nonzero) variables" << std::endl;
    out << "  --row-max-backtrack-steps      <float>  Maximum number of backtracking steps in line search" << std::endl;
    out << "  --row-min-step-size            <float>  Tolerance for nonzero step length" << std::endl;
    out << "  --row-step-reduction-factor    <float>  Factor to reduce line search step size" << std::endl;
    out << "  --row-suff-decrease-tolerance  <float>  Tolerance to ensure the next iterate decreases objective" << std::endl;
    out << std::endl;
    out << "PDNR-only options: " << std::endl;
    out << "  --pdnr-mu                        <float>  Initial damping parameter" << std::endl;
    out << "  --pdnr-increase-factor           <float>  Factor to increase damping parameter" << std::endl;
    out << "  --pdnr-decrease-factor           <float>  Factor to decrease damping parameter" << std::endl;
    out << "  --pdnr-increase-tol              <float>  Tolerance to increase damping parameter" << std::endl;
    out << "  --pdnr-decrease-tol              <float>  Tolerance to decrease damping parameter" << std::endl;
    out << "  --pdnr-scale-hessian             <bool>   Scale the Hessian matrix (not currently implemented)" << std::endl;
    out << "  --pdnr-kernel                    <int>    PDNR solver kernel options (1: Single-kernel, 2: Hybrid multi-kernel)" << std::endl;
    out << "  --pdnr-multikernel-switch-value  <int>    Switch threshold value used in the multi-kernel approach" << std::endl;
    out << std::endl;
    out << "PQNR-only options: " << std::endl;
    out << "  --pqnr-lbfgs  <int>  Number of vector pairs to store for L-BFGS" << std::endl;
    out << std::endl;
    out << "Logging options: " << std::endl;
    out << "  --log-level  <int>     0 (quiet) -- 2 (verbose)" << std::endl;
    out << "  --log-file   <string>  Store log information in a file" << std::endl;
    out << "  --debug      <bool>    Report debug info with log info" << std::endl;
    /*
    out << std::endl;
    out << "Additional options: " << std::endl;
    out << "  --solver-options <string> Name of file containing solver options" << std::endl;
    out << "    NOTE: arguments given in this file will overwrite those provided at the command line" << std::endl;
    out << "    EXAMPLE: (text file containing one space-separated argument-value pair per line)" << std::endl;
    out << "      tol 1e-6 " << std::endl;
    out << "      maxiters 100 " << std::endl;
     */
    out << std::endl;
    out << "Advanced options:" << std::endl;
    out << "  --working-path         <string>  Name of path where tensor input and output files will be found" << std::endl;
    out << "  --scheduling           <string>  Specify Kokkos scheduling policy for work items (static or dynamic)" << std::endl;
    out << "  --precision            <int>     Number of significant digits to store in output file" << std::endl;
    out << "  --global-index-type    <type>    Integer type for global indexing" << std::endl;
    out << "  --local-index-type     <type>    Integer type for for local indexing" << std::endl;
    out << "  --sptensor-type        <type>    Integer type for for input tensor data" << std::endl;
    out << "  --ktensor-type         <type>    Type of value used for output tensor data" << std::endl;
    out << std::endl;
    out << "Run sparten with --help to see this message again." << std::endl;
}

void sparten::CommandLineOptions::print_options(std::ostringstream &out) const {
    Log &log = Log::new_log();

    out << "Driver options:" << "\n";
    out << "  input = " << inputFile << "\n";
    out << "  output = " << outputFile << "\n";
    out << "  init = " << init_file << "\n";
    out << "Generic options:" << "\n";
    out << "  method = " << solver << "\n";
    out << "  rank = " << nComponent << "\n";
    out << "  seed = " << randomSeed << "\n";
    out << "  maxiters = " << maxInnerIter << "\n";
    out << "  maxinneriters = " << maxInnerIter << "\n";
    out << "  printitn = " << progressInterval << "\n";
    out << std::setprecision(1) << std::scientific << "  tol = " << tolerance << "\n";
    if (solver.compare("Multiplicative-Update")) {
        out << "MU options:" << "\n";
        out << "  mu-kappa = " << offset << "\n";
        out << "  mu-kappatol = " << offsetTolerance << "\n";
        out << "  mu-eps = " << eps << "\n";
    }
    else
    {
        out << "Row subproblem options:" << "\n";
        out << "  row-eps-zero-guard = " << eps_div_zero_grad << "\n";
        out << "  row-log-zero-guard = " << log_zero_safeguard << "\n";
        out << "  row-eps-active-set = " << eps_active_set << "\n";
        out << "  row-max-backtrack-steps = " << max_backtrack_steps << "\n";
        out << "  row-min-step-size = " << min_variable_nonzero_tolerance << "\n";
        out << "  row-step-reduction-factor = " << step_reduction_factor << "\n";
        out << "  row-suff-decrease-tolerance = " << suff_decrease_tolerance << "\n";
        out << "PDNR-only options:" << "\n";
        out << "  pdnr-mu = " << mu_initial << "\n";
        out << std::setprecision(3) << std::fixed;
        out << "  pdnr-increase-factor = " << damping_increase_factor << "\n";
        out << "  pdnr-decrease-factor = " << damping_decrease_factor << "\n";
        out << "  pdnr-increase-tol = " << damping_increase_tolerance << "\n";
        out << "  pdnr-decrease-tol = " << damping_decrease_tolerance << "\n";
        out << "  pdnr-scale-hessian = " << ( should_scale_hessian ? "true" : "false" ) << "\n";
        out << "  pdnr-kernel = " <<  solver_option_pdnr << "\n";
        out << "  pdnr-multikernel-switch-value = " << threshold_nonz_per_row_pdnr << "\n";
        out << "PQNR-only options:" << "\n";
        out << "  pqnr-lbfgs = " << size_LBFGS << "\n";
    }
    out << "Logging options:" << "\n";
    out << "  log-type = " << (debug ? "debug" : "release") << "\n";
    out << "  log-level = " << print_level << "\n";
    out << "  log-file = " << ( logFile.empty() ? "none provided (written to STDOUT)" : logFile) << "\n";
    out << "Advanced options:" << "\n";
    out << "  scheduling = " << scheduling << "\n";
    out << "  precision = " << precision << "\n";
    out << "  global-index-type = " << globalOrdinalType << "\n";
    out << "  local-index-type = " << localOrdinalType << "\n";
    out << "  sptensor-type = " << sparseValueType << "\n";
    out << "  ktensor-type = " << kruskalValueType;

    log.print(out.str(),Log::VERBOSE);
}

//! Convenience function to get command line options
void sparten::CommandLineOptions::get_command_line_options(std::vector<std::string>& args)
{
    /* Read command line arguments */
    // Driver options
//    inputFile                      = parse_string(args, "--input", "");
    init_file                      = parse_string(args, "--init", init_file.c_str());
    outputFile                     = parse_string(args, "--output", outputFile.c_str());
    indexStyle                     = parse_uint64_t(args, "--index-base", 1);
    // Generic options
    solver                         = parse_string(args, "--method", solver.c_str());
    nComponent                     = parse_uint64_t(args, "--rank", 2, 1, UINT64_MAX);
    randomSeed                     = parse_uint64_t(args, "--seed", 131, 0, UINT64_MAX);
    maxOuterIter                   = parse_uint64_t(args, "--maxiters", 20, 1, UINT64_MAX);
    maxInnerIter                   = parse_uint64_t(args, "--maxinneriters", 10, 1, UINT64_MAX);
    tolerance                      = parse_double(args, "--tol", 1e-04, 0.0, 1.0);
    progressInterval               = parse_uint64_t(args, "--printitn", 1, 0, UINT64_MAX);
    // MU options
    offset                         = parse_double(args, "--mu-kappa", 1e-10, 0, 1.0);
    offsetTolerance                = parse_double(args, "--mu-kappatol", 1e-10, 0, 1.0);
    eps                            = parse_double(args, "--mu-eps", 1e-10, 0, 1.0);
    // Row subproblem options
    eps_div_zero_grad              = parse_double(args, "--row-eps-zero-guard", 1e-16, 0, 1.0);
    log_zero_safeguard             = parse_double(args, "--row-log-zero-guard", 1e-16, 0, 1.0);
    eps_active_set                 = parse_double(args, "--row-eps-active-set", 1e-10, 0, 1.0);
    max_backtrack_steps            = parse_uint64_t(args, "--row-max-backtrack-steps", 10, 1, UINT64_MAX);
    min_variable_nonzero_tolerance = parse_double(args, "--row-min-step-size", 1e-8, 0, 1.0);
    step_reduction_factor          = parse_double(args, "--row-step-reduction-factor", 0.5, 0, 1.0);
    suff_decrease_tolerance        = parse_double(args, "--row-suff-decrease-tolerance", 1e-4, 0, 1.0);
    // PDNR-only options
    mu_initial                     = parse_double(args, "--pdnr-mu", 1e-5, 0, 1.0);
    damping_increase_factor        = parse_double(args, "--pdnr-increase-factor", 3.5, 0.0, 100.0);
    damping_decrease_factor        = parse_double(args, "--pdnr-decrease-factor", 2.0/7.0, 0.0, 100.0);
    damping_increase_tolerance     = parse_double(args, "--pdnr-increase-tolerance", 0.25, 0.0, 10.0);
    damping_decrease_tolerance     = parse_double(args, "--pdnr-decrease-tolerance", 0.75, 0.0, 10.0);
    should_scale_hessian           = parse_bool(args, "--pdnr-no-scale-hessian", "--pdnr-scale-hessian", false);
    solver_option_pdnr             = parse_uint64_t(args, "--pdnr-kernel-option", 1, 1, 2);
    threshold_nonz_per_row_pdnr    = parse_uint64_t(args, "--pdnr-multi-kernel-size", 4096, 1, UINT64_MAX);
    // PQNR-only options
    size_LBFGS                     = parse_uint64_t(args, "--pqnr-lbfgs", 3, 1, UINT64_MAX);
    // Logging options
    debug                          = parse_bool(args,"--debug","--release",false);
    logFile                        = parse_string(args, "--log-file", "");
    print_level                    = parse_uint64_t(args, "--log-level", 1,0,2);
    // Additional options
    // inputOptionsFile               = parse_string(args, "--solver-options-file", "inputOptions.txt");

    // Advanced options
    scheduling                     = parse_string(args, "--scheduling", "static");
    precision                      = parse_uint64_t(args, "--precision", 16, 0, 16);
    globalOrdinalType              = parse_string(args, "--global-ordinal-type", "int32");
    localOrdinalType               = parse_string(args, "--local-ordinal-type", "int32");
    sparseValueType                = parse_string(args, "--sparse-value-type", "int32");
    kruskalValueType               = parse_string(args, "--kruskal-value-type", "double");

    if (sparten::check_and_print_unused_args(args, std::cout))
    {
        print_help (std::cout);
        throw std::runtime_error("sparten::get_command_line_options::check_and_print_unused_args::runtime_error: Invalid command line options.");
    }

    /* Set arguments */
    /*
	clo.inputOptionsFile  = inputOptionsFile.empty() ? clo.inputOptionsFile : inputOptionsFile;
	clo.inputFile         = inputFile.empty() ? clo.inputFile : inputFile;
	clo.outputFile        = outputFile.empty() ? clo.outputFile : outputFile;
	clo.workingPath       = workingPath.empty() ? clo.workingPath : workingPath;
	clo.scheduling        = scheduling.empty() ? clo.scheduling : scheduling;
	clo.localOrdinalType  = localOrdinalType.empty() ? clo.localOrdinalType : localOrdinalType;
	clo.globalOrdinalType = globalOrdinalType.empty() ? clo.globalOrdinalType : globalOrdinalType;
	clo.sparseValueType   = sparseValueType.empty() ? clo.sparseValueType : sparseValueType;
	clo.kruskalValueType  = kruskalValueType.empty() ? clo.kruskalValueType : kruskalValueType;
	clo.logFile           = logFile.empty() ? clo.logFile : logFile;
	clo.print_level       = print_level.empty() ? clo.print_level : print_level;
	clo.indexStyle        = indexStyle.empty() ? clo.indexStyle : indexStyle;
	clo.solver            = solver.empty() ? clo.solver : solver;
//	clo.randomSeed        = randomSeed.empty() ? clo.randomSeed : randomSeed;
	clo.init_file         = init_file.empty() ? clo.init_file : init_file;

     Process remaining arguments
	std::istringstream toleranceIss(toleranceStr);
	std::istringstream maxOuterIterIss(maxOuterIterStr);
	std::istringstream maxInnerIterIss(maxInnerIterStr);
	std::istringstream offsetIss(offsetStr);
	std::istringstream offsetToleranceIss(offsetToleranceStr);
	std::istringstream progressIntervalIss(progressIntervalStr);
	std::istringstream epsIss(epsStr);
	std::istringstream precisionIss(precisionStr);
	std::istringstream nComponentIss(nComponentStr);
	std::istringstream eps_div_zero_gradIss(eps_div_zero_gradStr);
	std::istringstream log_zero_safeguardIss(log_zero_safeguardStr);
	std::istringstream eps_active_setIss(eps_active_setStr);
	std::istringstream max_backtrack_stepsIss(max_backtrack_stepsStr);
	std::istringstream min_variable_nonzero_toleranceIss(min_variable_nonzero_toleranceStr);
	std::istringstream step_reduction_factorIss(step_reduction_factorStr);
	std::istringstream suff_decrease_toleranceIss(suff_decrease_toleranceStr);
	std::istringstream mu_initialIss(mu_initialStr);
	std::istringstream damping_increase_factorIss(damping_increase_factorStr);
	std::istringstream damping_decrease_factorIss(damping_decrease_factorStr);
	std::istringstream damping_increase_toleranceIss(damping_increase_toleranceStr);
	std::istringstream damping_decrease_toleranceIss(damping_decrease_toleranceStr);
//	std::istringstream should_scale_hessianIss(should_scale_hessianStr);
	std::istringstream size_LBFGSIss(size_LBFGSStr);
	std::istringstream solver_option_pdnr_ss(solver_option_pdnr_str);
    std::istringstream threshold_nonz_per_row_pdnr_ss(threshold_nonz_per_row_pdnr_str);

	toleranceIss                      >> clo.tolerance;
	maxOuterIterIss                   >> clo.maxOuterIter;
	maxInnerIterIss                   >> clo.maxInnerIter;
	offsetIss                         >> clo.offset;
	offsetToleranceIss                >> clo.offsetTolerance;
	progressIntervalIss               >> clo.progressInterval;
	epsIss                            >> clo.eps;
	precisionIss                      >> clo.precision;
	nComponentIss                     >> clo.nComponent;
	eps_div_zero_gradIss              >> clo.eps_div_zero_grad;
	log_zero_safeguardIss             >> clo.log_zero_safeguard;
	eps_active_setIss                 >> clo.eps_active_set;
	max_backtrack_stepsIss            >> clo.max_backtrack_steps;
	min_variable_nonzero_toleranceIss >> clo.min_variable_nonzero_tolerance;
	step_reduction_factorIss          >> clo.step_reduction_factor;
	suff_decrease_toleranceIss        >> clo.suff_decrease_tolerance;
	mu_initialIss                     >> clo.mu_initial;
	damping_increase_factorIss        >> clo.damping_increase_factor;
	damping_decrease_factorIss        >> clo.damping_decrease_factor;
	damping_increase_toleranceIss     >> clo.damping_increase_tolerance;
	damping_decrease_toleranceIss     >> clo.damping_decrease_tolerance;
//	should_scale_hessianIss           >> clo.should_scale_hessian;
	size_LBFGSIss                     >> clo.size_LBFGS;
	solver_option_pdnr_ss             >> clo.solver_option_pdnr;
	threshold_nonz_per_row_pdnr_ss    >> clo.threshold_nonz_per_row_pdnr;

	if (!inputOptionsFile.empty())
    {

    SimpleInput input(clo.inputOptionsFile, SimpleInput::READ);
    InputOptions inputOptions = input.read();

	  clo.inputFile                      = inputOptions.inputFile.empty() ? clo.inputFile : inputOptions.inputFile;
	  clo.outputFile                     = inputOptions.outputFile.empty() ? clo.outputFile : inputOptions.outputFile;
	  clo.workingPath                    = inputOptions.workingPath.empty() ? clo.workingPath : inputOptions.workingPath;
	  clo.scheduling                     = inputOptions.scheduling.empty() ? clo.scheduling : inputOptions.scheduling;
	  clo.localOrdinalType               = inputOptions.localOrdinalType.empty() ? clo.localOrdinalType : inputOptions.localOrdinalType;
	  clo.globalOrdinalType              = inputOptions.globalOrdinalType.empty() ? clo.globalOrdinalType : inputOptions.globalOrdinalType;
	  clo.sparseValueType                = inputOptions.sparseValueType.empty() ? clo.sparseValueType : inputOptions.sparseValueType;
	  clo.kruskalValueType               = inputOptions.kruskalValueType.empty() ? clo.kruskalValueType : inputOptions.kruskalValueType;
	  clo.logFile                        = inputOptions.logFile.empty() ? clo.logFile : inputOptions.logFile;
	  clo.print_level                    = inputOptions.print_level.empty() ? clo.print_level : inputOptions.print_level;
	  clo.indexStyle                     = inputOptions.indexStyle.empty() ? clo.indexStyle : inputOptions.indexStyle;

	  clo.tolerance                      = !toleranceStr.empty() ? clo.tolerance : inputOptions.tolerance;
	  clo.maxOuterIter                   = !maxOuterIterStr.empty() ? clo.maxOuterIter : inputOptions.maxOuterIter;
	  clo.maxInnerIter                   = !maxInnerIterStr.empty() ? clo.maxInnerIter : inputOptions.maxInnerIter;
	  clo.offset                         = !offsetStr.empty() ? clo.offset : inputOptions.offset;
	  clo.offsetTolerance                = !offsetToleranceStr.empty() ? clo.offsetTolerance : inputOptions.offsetTolerance;
	  clo.progressInterval               = !progressIntervalStr.empty() ? clo.progressInterval : inputOptions.progressInterval;
	  clo.eps                            = !epsStr.empty() ? clo.eps : inputOptions.eps;
	  clo.precision                      = !precisionStr.empty() ? clo.precision : inputOptions.precision;
	  clo.nComponent                     = !nComponentStr.empty() ? clo.nComponent : inputOptions.nComponent;
	  clo.solver                         = !clo.solver.empty() ? inputOptions.solver : clo.solver;
	  clo.randomSeed                     = !clo.randomSeed.empty() ? clo.randomSeed : inputOptions.randomSeed;
	  clo.eps_div_zero_grad              = !eps_div_zero_gradStr.empty() ? clo.eps_div_zero_grad : inputOptions.eps_div_zero_grad;
	  clo.log_zero_safeguard             = !log_zero_safeguardStr.empty() ? clo.log_zero_safeguard : inputOptions.log_zero_safeguard;
	  clo.eps_active_set                 = !eps_active_setStr.empty() ? clo.eps_active_set : inputOptions.eps_active_set;
	  clo.max_backtrack_steps            = !max_backtrack_stepsStr.empty() ? clo.max_backtrack_steps : inputOptions.max_backtrack_steps;
	  clo.min_variable_nonzero_tolerance = !min_variable_nonzero_toleranceStr.empty() ? clo.min_variable_nonzero_tolerance : inputOptions.min_variable_nonzero_tolerance;
	  clo.step_reduction_factor          = !step_reduction_factorStr.empty() ? clo.step_reduction_factor : inputOptions.step_reduction_factor;
	  clo.suff_decrease_tolerance        = !suff_decrease_toleranceStr.empty() ? clo.suff_decrease_tolerance : inputOptions.suff_decrease_tolerance;
	  clo.mu_initial                     = !mu_initialStr.empty() ? clo.mu_initial : inputOptions.mu_initial;
	  clo.damping_increase_factor        = !damping_increase_factorStr.empty() ? clo.damping_increase_factor : inputOptions.damping_increase_factor;
	  clo.damping_decrease_factor        = !damping_decrease_factorStr.empty() ? clo.damping_decrease_factor : inputOptions.damping_decrease_factor;
	  clo.damping_increase_tolerance     = !damping_increase_toleranceStr.empty() ? clo.damping_increase_tolerance : inputOptions.damping_increase_tolerance;
	  clo.damping_decrease_tolerance     = !damping_decrease_toleranceStr.empty() ? clo.damping_decrease_tolerance : inputOptions.damping_decrease_tolerance;
//	  clo.should_scale_hessian           = !should_scale_hessianStr.empty() ? clo.should_scale_hessian : inputOptions.should_scale_hessian;
	  clo.size_LBFGS                     = !size_LBFGSStr.empty() ? clo.size_LBFGS : inputOptions.size_LBFGS;
	  clo.solver_option_pdnr             = !solver_option_pdnr_str.empty() ? clo.solver_option_pdnr : inputOptions.solver_option_pdnr;
	  clo.threshold_nonz_per_row_pdnr    = !threshold_nonz_per_row_pdnr_str.empty() ?  clo.threshold_nonz_per_row_pdnr : inputOptions.threshold_nonz_per_row_pdnr;

  }

     */

}

bool
sparten::check_and_print_unused_args(const std::vector<std::string>& args, std::ostream& out)
{
    if (args.size() == 0)
        return false;

    out << std::endl << "Error!! Unknown command line arguments: ";
    for (auto arg : args)
        out << arg << " ";
    out << std::endl << std::endl;

    return true;
}

std::string
sparten::parse_string(std::vector<std::string>& args, const std::string& cl_arg, const std::string& default_val)
{
    std::string tmp = default_val;
    auto it = std::find(args.begin(), args.end(), cl_arg);

    // If not found, try removing the '--'
    if ((it == args.end()) && (cl_arg.size() > 2) &&
        (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
        it = std::find(args.begin(), args.end(), cl_arg.substr(2));
    }

    if (it != args.end()) {
        auto arg_it = it;
        // get next cl_arg
        ++it;
        if (it == args.end()) {
            args.erase(arg_it);
            return tmp;
        }
        // get argument
        tmp = *it;
        // Remove argument from list
        args.erase(arg_it, ++it);
    }
    return tmp;
}

uint64_t
sparten::parse_uint64_t(std::vector<std::string>& args, const std::string& cl_arg, uint64_t default_val, uint64_t min, uint64_t max)
{
    uint64_t tmp = default_val;
    auto it = std::find(args.begin(), args.end(), cl_arg);

    // If not found, try removing the '--'
    if ((it == args.end()) && (cl_arg.size() > 2) &&
        (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
        it = std::find(args.begin(), args.end(), cl_arg.substr(2));
    }

    if (it != args.end()) {
        auto arg_it = it;
        // get next cl_arg
        ++it;
        if (it == args.end()) {
            args.erase(arg_it);
            return tmp;
        }
        // convert to uint64_t
        char *cend = 0;
        tmp = std::strtoull(it->c_str(), &cend, 10);

        if (tmp >= LLONG_MAX)
        {
            std::ostringstream error_string;
            error_string << "Unparseable input: " << cl_arg << " " << *it
                         << ", value out of the range of representable values"
                         << " (try checking the sign of " << *it << ")" << std::endl;
            throw std::runtime_error(error_string.str());
        }

        // check if cl_arg is actually uint64_t
        if (it->c_str() == cend) {
            std::ostringstream error_string;
            error_string << "Unparseable input: " << cl_arg << " " << *it
                         << ", must be an integer" << std::endl;
            throw std::runtime_error(error_string.str());
        }

        // Remove argument from list
        args.erase(arg_it, ++it);
    }

    // check if arg is within bounds
    if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << tmp
                     << ",  must be in the range (" << min << ", " << max
                     << ")" << std::endl;
        throw std::runtime_error(error_string.str());
    }

    return tmp;
}

double
sparten::parse_double(std::vector<std::string>& args, const std::string& cl_arg, double default_val, double min, double max)
{
    double tmp = default_val;
    auto it = std::find(args.begin(), args.end(), cl_arg);

    // If not found, try removing the '--'
    if ((it == args.end()) && (cl_arg.size() > 2) &&
        (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
        it = std::find(args.begin(), args.end(), cl_arg.substr(2));
    }

    if (it != args.end()) {
        auto arg_it = it;
        // get next cl_arg
        ++it;
        if (it == args.end()) {
            args.erase(arg_it);
            return tmp;
        }
        // convert to double
        char *cend = 0;
        tmp = std::strtod(it->c_str(),&cend);

        // check if cl_arg is actually double
        if (it->c_str() == cend) {
            std::ostringstream error_string;
            error_string << "Unparseable input: " << cl_arg << " " << *it
                         << ", must be a float" << std::endl;
            throw std::runtime_error(error_string.str());
        }

        // Remove argument from list
        args.erase(arg_it, ++it);
    }

    // check if arg is within bounds
    if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << tmp
                     << ",  must be in the range (" << min << ", " << max
                     << ")" << std::endl;
        throw std::runtime_error(error_string.str());
    }
    return tmp;
}

bool
sparten::parse_bool(std::vector<std::string>& args, const std::string& cl_arg_on, const std::string& cl_arg_off, bool default_value)
{
    // return true if arg_on is found
    auto it = std::find(args.begin(), args.end(), cl_arg_on);
    // If not found, try removing the '--'
    if ((it == args.end()) && (cl_arg_on.size() > 2) &&
        (cl_arg_on[0] == '-') && (cl_arg_on[1] == '-')) {
        it = std::find(args.begin(), args.end(), cl_arg_on.substr(2));
    }
    if (it != args.end()) {
        args.erase(it);
        return true;
    }

    // return false if arg_off is found
    it = std::find(args.begin(), args.end(), cl_arg_off);
    // If not found, try removing the '--'
    if ((it == args.end()) && (cl_arg_off.size() > 2) &&
        (cl_arg_off[0] == '-') && (cl_arg_off[1] == '-')) {
        it = std::find(args.begin(), args.end(), cl_arg_off.substr(2));
    }
    if (it != args.end()) {
        args.erase(it);
        return false;
    }

    // return default value if not specified on command line
    return default_value;
}

std::vector<std::string>
sparten::get_arg_list(int argc, char** argv)
{
    std::vector<std::string> args(argc-1);
    for (int i = 1; i < argc; ++i)
        args[i-1] = argv[i];
    return args;
}
