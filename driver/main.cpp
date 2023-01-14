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

#include <sparten/CommandLineOptions.h>
#include <MainWorkflow.h>
#include <sparten/SimpleInput.h>
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

//bool check_and_print_unused_args(const std::vector<std::string>& args, std::ostream& out)
//{
//    if (args.size() == 0)
//        return false;
//
//    out << std::endl << "Error!! Unknown command line arguments: ";
//    for (auto arg : args)
//        out << arg << " ";
//    out << std::endl << std::endl;
//
//    return true;
//}
//
//std::string parse_string(std::vector<std::string>& args, const std::string& cl_arg, const std::string& default_val)
//{
//    std::string tmp = default_val;
//    auto it = std::find(args.begin(), args.end(), cl_arg);
//
//    // If not found, try removing the '--'
//    if ((it == args.end()) && (cl_arg.size() > 2) &&
//        (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
//        it = std::find(args.begin(), args.end(), cl_arg.substr(2));
//    }
//
//    if (it != args.end()) {
//        auto arg_it = it;
//        // get next cl_arg
//        ++it;
//        if (it == args.end()) {
//            args.erase(arg_it);
//            return tmp;
//        }
//        // get argument
//        tmp = *it;
//        // Remove argument from list
//        args.erase(arg_it, ++it);
//    }
//    return tmp;
//}
//
//uint64_t parse_uint64_t(std::vector<std::string>& args, const std::string& cl_arg, uint64_t default_val, uint64_t min=0, uint64_t max=100)
//{
//    uint64_t tmp = default_val;
//    auto it = std::find(args.begin(), args.end(), cl_arg);
//
//    // If not found, try removing the '--'
//    if ((it == args.end()) && (cl_arg.size() > 2) &&
//        (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
//        it = std::find(args.begin(), args.end(), cl_arg.substr(2));
//    }
//
//    if (it != args.end()) {
//        auto arg_it = it;
//        // get next cl_arg
//        ++it;
//        if (it == args.end()) {
//            args.erase(arg_it);
//            return tmp;
//        }
//        // convert to uint64_t
//        char *cend = 0;
//        tmp = std::strtoull(it->c_str(),&cend,10);
//
//        // check if cl_arg is actually uint64_t
//        if (it->c_str() == cend) {
//            std::ostringstream error_string;
//            error_string << "Unparseable input: " << cl_arg << " " << *it
//                         << ", must be an integer" << std::endl;
//            throw std::runtime_error(error_string.str());
//        }
//
//        // Remove argument from list
//        args.erase(arg_it, ++it);
//    }
//
//    // check if arg is within bounds
//    if (tmp < min || tmp > max) {
//        std::ostringstream error_string;
//        error_string << "Bad input: " << cl_arg << " " << tmp
//                     << ",  must be in the range (" << min << ", " << max
//                     << ")" << std::endl;
//        throw std::runtime_error(error_string.str());
//    }
//    return tmp;
//}
//
//double parse_double(std::vector<std::string>& args, const std::string& cl_arg, double default_val, double min=0.0, double max=1.0)
//{
//    double tmp = default_val;
//    auto it = std::find(args.begin(), args.end(), cl_arg);
//
//    // If not found, try removing the '--'
//    if ((it == args.end()) && (cl_arg.size() > 2) &&
//        (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
//        it = std::find(args.begin(), args.end(), cl_arg.substr(2));
//    }
//
//    if (it != args.end()) {
//        auto arg_it = it;
//        // get next cl_arg
//        ++it;
//        if (it == args.end()) {
//            args.erase(arg_it);
//            return tmp;
//        }
//        // convert to double
//        char *cend = 0;
//        tmp = std::strtod(it->c_str(),&cend);
//
//        // check if cl_arg is actually double
//        if (it->c_str() == cend) {
//            std::ostringstream error_string;
//            error_string << "Unparseable input: " << cl_arg << " " << *it
//                         << ", must be a float" << std::endl;
//            throw std::runtime_error(error_string.str());
//        }
//
//        // Remove argument from list
//        args.erase(arg_it, ++it);
//    }
//
//    // check if arg is within bounds
//    if (tmp < min || tmp > max) {
//        std::ostringstream error_string;
//        error_string << "Bad input: " << cl_arg << " " << tmp
//                     << ",  must be in the range (" << min << ", " << max
//                     << ")" << std::endl;
//        throw std::runtime_error(error_string.str());
//    }
//    return tmp;
//}
//
//bool parse_bool(std::vector<std::string>& args, const std::string& cl_arg_on, const std::string& cl_arg_off, bool default_val)
//{
//    // return true if arg_on is found
//    auto it = std::find(args.begin(), args.end(), cl_arg_on);
//
//    // If not found, try removing the '--'
//    if ((it == args.end()) && (cl_arg_on.size() > 2) &&
//        (cl_arg_on[0] == '-') && (cl_arg_on[1] == '-')) {
//        it = std::find(args.begin(), args.end(), cl_arg_on.substr(2));
//    }
//
//    if (it != args.end()) {
//        args.erase(it);
//        return true;
//    }
//
//    // return false if arg_off is found
//    it = std::find(args.begin(), args.end(), cl_arg_off);
//
//    // If not found, try removing the '--'
//    if ((it == args.end()) && (cl_arg_off.size() > 2) &&
//        (cl_arg_off[0] == '-') && (cl_arg_off[1] == '-')) {
//        it = std::find(args.begin(), args.end(), cl_arg_off.substr(2));
//    }
//
//    if (it != args.end()) {
//        args.erase(it);
//        return false;
//    }
//
//    // return default value if not specified on command line
//    return default_val;
//}

////! Convenience function to get std::string with args
//std::string get_argument(int argc, char const **argv, int i)
//{
//  std::string tmp(argv[i]);
//  return tmp;
//}
//
////! Convenience function to get std::string with args
//std::string get_argument(int argc, char **argv, std::string const &arg)
//{
//  bool found = false;
//  std::string value;
//
//  sparten::CommandLineOptions clo;
//
//  if (argc % 2 == 0)
//  {
//    clo.print_help(argv,std::cout);
//    throw std::runtime_error("Invalid number of command line options.");
//  }
//
//  for (int i = 1; i < argc; i += 2)
//  {
//    if (arg.compare(argv[i]) == 0)
//    {
//      found = true;
//      value = argv[i + 1];
//    }
//  }
//
//  if (found)
//  {
//    return value;
//  } else
//  {
//    clo.print_help(argv,std::cout);
//    throw std::runtime_error(arg + " not found.");
//  }
//}
//
////! Convenience function to get std::string with optional args
//std::string get_argument_optional(int argc, char **argv, std::string const &arg)
//{
//  bool found = false;
//  std::string value;
////  CommandLineOptions clo;
//
////  if (argc % 2 == 0)
////  {
////    clo.print_help(argv,std::cout);
////    throw std::runtime_error("Invalid number of command line options.");
////  }
//
//  for (int i = 1; i < argc; i += 2)
//  {
//    if (arg.compare(argv[i]) == 0)
//    {
//      found = true;
//      value = argv[i + 1];
//    }
//  }
//
//  if (found)
//  {
//    return value;
//  }
//  else
//  {
//    return "";
//  }
//}

////! Convenience function to get command line options
//sparten::CommandLineOptions get_command_line_options(int argc, char **argv) {
//
//    sparten::CommandLineOptions clo;
//
//    std::vector<std::string> args(argc - 1);
//    int nargs = 0;
//    for (int i = 1; i < argc; ++i)
//    {
//        args[i-1] = argv[i];
//        ++nargs;
//    }
////    std::cout << " num args: " << nargs << std::endl;
////    for (int i = 0; i < nargs; ++i )
////        std::cout << " " << args[i] << std::endl;
//
//
//  if (nargs < 2)
//  {
//    clo.print_help(argv,std::cout);
//    throw std::runtime_error("No arguments provided");
//  }
//  else if (nargs % 2 != 0)
//  {
//      clo.print_help(argv, std::cout);
//      throw std::runtime_error("Invalid number of command line options.");
//  }
//
////    std::string inputFile = parse_string(args, "--input", "tensor.tns");
////    bool should_scale_hessianStr = parse_bool(args, "--pdnr-scale-hessian", "--pdnr-no-scale-hessian", false);
////    uint64_t randomSeed = parse_int(args, "--seed", 131, 0, UINT64_MAX);
////    clo.mu_initial = parse_double(args, "--pdnr-mu", 1e-05, 0.0, 1.0);
////    std::cout << "in " << inputFile << std::endl;
////    std::cout << "hess " << should_scale_hessianStr << std::endl;
////    std::cout << "seed " << randomSeed << std::endl;
////    std::cout << "mu " << clo.mu_initial << std::endl;
//
//  /* Read command line arguments */
//    // Driver options
//    clo.inputFile                      = sparten::parse_string(args, "--input", "tensor.txt");
//    clo.init_file                      = sparten::parse_string(args, "--init", "init.txt");
//    clo.outputFile                     = sparten::parse_string(args, "--output", "ktensor.txt");
//    clo.indexStyle                     = sparten::parse_uint64_t(args, "--index-base", 0);
//    // Generic options
//    clo.solver                         = sparten::parse_string(args, "--method", "Multiplicative-Update");
//    clo.nComponent                     = sparten::parse_uint64_t(args, "--rank", 2, 1, UINT64_MAX);
//    clo.randomSeed                     = sparten::parse_uint64_t(args, "--seed", 131, 0, UINT64_MAX);
//    clo.maxOuterIter                   = sparten::parse_uint64_t(args, "--maxiters", 20, 1, UINT64_MAX);
//    clo.maxInnerIter                   = sparten::parse_uint64_t(args, "--maxinneriters", 10, 1, UINT64_MAX);
//    clo.tolerance                      = sparten::parse_double(args, "--tol", 1e-04, 0.0, 1.0);
//    clo.progressInterval               = sparten::parse_uint64_t(args, "--printitn", 1, 0, UINT64_MAX);
//    // MU options
//    clo.offset                         = sparten::parse_double(args, "--mu-kappa", 1e-10, 0, 1.0);
//    clo.offsetTolerance                = sparten::parse_double(args, "--mu-kappatol", 1e-10, 0, 1.0);
//    clo.eps                            = sparten::parse_double(args, "--mu-eps", 1e-10, 0, 1.0);
//    // Row subproblem options
//    clo.eps_div_zero_grad              = sparten::parse_double(args, "--row-eps-zero-guard", 1e-16, 0, 1.0);
//    clo.log_zero_safeguard             = sparten::parse_double(args, "--row-log-zero-guard", 1e-16, 0, 1.0);
//    clo.eps_active_set                 = sparten::parse_double(args, "--row-eps-active-set", 1e-10, 0, 1.0);
//    clo.max_backtrack_steps            = sparten::parse_uint64_t(args, "--row-max-backtrack-steps", 10, 1, UINT64_MAX);
//    clo.min_variable_nonzero_tolerance = sparten::parse_double(args, "--row-min-step-size", 1e-8, 0, 1.0);
//    clo.step_reduction_factor          = sparten::parse_double(args, "--row-step-reduction-factor", 1e-4, 0, 1.0);
//    clo.suff_decrease_tolerance        = sparten::parse_double(args, "--row-suff-decrease-tolerance", 1e-4, 0, 1.0);
//    // PDNR-only options
//    clo.mu_initial                     = sparten::parse_double(args, "--pdnr-mu", 1e-5, 0, 1.0);
//    clo.damping_increase_factor        = sparten::parse_double(args, "--pdnr-increase-factor", 0.25, 0.0, 1.0);
//    clo.damping_decrease_factor        = sparten::parse_double(args, "--pdnr-decrease-factor", 0.75, 0.0, 1.0);
//    clo.damping_increase_tolerance     = sparten::parse_double(args, "--pdnr-increase-tolerance", 7 / 2, 0.0, 10000);
//    clo.damping_decrease_tolerance     = sparten::parse_double(args, "--pdnr-decrease-tolerance", 2 / 7, 0.0, 10000);
//    clo.should_scale_hessian           = sparten::parse_bool(args, "--pdnr-scale-hessian", "--pdnr-no-scale-hessian", false);
//    clo.solver_option_pdnr             = sparten::parse_uint64_t(args, "--pdnr-kernel-option", 1, 0, 1);
//    clo.threshold_nonz_per_row_pdnr    = sparten::parse_uint64_t(args, "--pdnr-multikernel-switch-value", 4096, 1, UINT64_MAX);
//    // PQNR-only options
//    clo.size_LBFGS                     = sparten::parse_uint64_t(args, "--pqnr-lbfgs", 3, 1, UINT64_MAX);
//    // Logging options
//    clo.logFile                        = sparten::parse_string(args, "--log-file", "sparten.log");
//    clo.print_level                    = sparten::parse_uint64_t(args, "--log-level", 1, 0, 5);
//    // Additional options
//    clo.inputOptionsFile               = sparten::parse_string(args, "--solver-options", "inputOptions.txt");
//    // Advanced options
//    clo.workingPath                    = sparten::parse_string(args, "--working-path", ".");
//    clo.scheduling                     = sparten::parse_string(args, "--scheduling", "static");
//    clo.precision                      = sparten::parse_uint64_t(args, "--precision", 16, 0, 16);
//    clo.globalOrdinalType              = sparten::parse_string(args, "--global-ordinal-type", "int64");
//    clo.localOrdinalType               = sparten::parse_string(args, "--local-ordinal-type", "int64");
//    clo.sparseValueType                = sparten::parse_string(args, "--sparse-value-type", "int64");
//    clo.kruskalValueType               = sparten::parse_string(args, "--kruskal-value-type", "int64");
//
//
//    if (sparten::check_and_print_unused_args(args, std::cout))
//    {
//        throw std::runtime_error("Invalid command line options.");
//    }
//    /* Set arguments */
////	clo.inputOptionsFile  = inputOptionsFile.empty() ? clo.inputOptionsFile : inputOptionsFile;
////	clo.inputFile         = inputFile.empty() ? clo.inputFile : inputFile;
////	clo.outputFile        = outputFile.empty() ? clo.outputFile : outputFile;
////	clo.workingPath       = workingPath.empty() ? clo.workingPath : workingPath;
////	clo.scheduling        = scheduling.empty() ? clo.scheduling : scheduling;
////	clo.localOrdinalType  = localOrdinalType.empty() ? clo.localOrdinalType : localOrdinalType;
////	clo.globalOrdinalType = globalOrdinalType.empty() ? clo.globalOrdinalType : globalOrdinalType;
////	clo.sparseValueType   = sparseValueType.empty() ? clo.sparseValueType : sparseValueType;
////	clo.kruskalValueType  = kruskalValueType.empty() ? clo.kruskalValueType : kruskalValueType;
////	clo.logFile           = logFile.empty() ? clo.logFile : logFile;
////	clo.print_level       = print_level.empty() ? clo.print_level : print_level;
////	clo.indexStyle        = indexStyle.empty() ? clo.indexStyle : indexStyle;
////	clo.solver            = solver.empty() ? clo.solver : solver;
//////	clo.randomSeed        = randomSeed.empty() ? clo.randomSeed : randomSeed;
////	clo.init_file         = init_file.empty() ? clo.init_file : init_file;
//
//	// Process remaining arguments
////	std::istringstream toleranceIss(toleranceStr);
////	std::istringstream maxOuterIterIss(maxOuterIterStr);
////	std::istringstream maxInnerIterIss(maxInnerIterStr);
////	std::istringstream offsetIss(offsetStr);
////	std::istringstream offsetToleranceIss(offsetToleranceStr);
////	std::istringstream progressIntervalIss(progressIntervalStr);
////	std::istringstream epsIss(epsStr);
////	std::istringstream precisionIss(precisionStr);
////	std::istringstream nComponentIss(nComponentStr);
////	std::istringstream eps_div_zero_gradIss(eps_div_zero_gradStr);
////	std::istringstream log_zero_safeguardIss(log_zero_safeguardStr);
////	std::istringstream eps_active_setIss(eps_active_setStr);
////	std::istringstream max_backtrack_stepsIss(max_backtrack_stepsStr);
////	std::istringstream min_variable_nonzero_toleranceIss(min_variable_nonzero_toleranceStr);
////	std::istringstream step_reduction_factorIss(step_reduction_factorStr);
////	std::istringstream suff_decrease_toleranceIss(suff_decrease_toleranceStr);
////	std::istringstream mu_initialIss(mu_initialStr);
////	std::istringstream damping_increase_factorIss(damping_increase_factorStr);
////	std::istringstream damping_decrease_factorIss(damping_decrease_factorStr);
////	std::istringstream damping_increase_toleranceIss(damping_increase_toleranceStr);
////	std::istringstream damping_decrease_toleranceIss(damping_decrease_toleranceStr);
//////	std::istringstream should_scale_hessianIss(should_scale_hessianStr);
////	std::istringstream size_LBFGSIss(size_LBFGSStr);
////	std::istringstream solver_option_pdnr_ss(solver_option_pdnr_str);
////    std::istringstream threshold_nonz_per_row_pdnr_ss(threshold_nonz_per_row_pdnr_str);
////
////	toleranceIss                      >> clo.tolerance;
////	maxOuterIterIss                   >> clo.maxOuterIter;
////	maxInnerIterIss                   >> clo.maxInnerIter;
////	offsetIss                         >> clo.offset;
////	offsetToleranceIss                >> clo.offsetTolerance;
////	progressIntervalIss               >> clo.progressInterval;
////	epsIss                            >> clo.eps;
////	precisionIss                      >> clo.precision;
////	nComponentIss                     >> clo.nComponent;
////	eps_div_zero_gradIss              >> clo.eps_div_zero_grad;
////	log_zero_safeguardIss             >> clo.log_zero_safeguard;
////	eps_active_setIss                 >> clo.eps_active_set;
////	max_backtrack_stepsIss            >> clo.max_backtrack_steps;
////	min_variable_nonzero_toleranceIss >> clo.min_variable_nonzero_tolerance;
////	step_reduction_factorIss          >> clo.step_reduction_factor;
////	suff_decrease_toleranceIss        >> clo.suff_decrease_tolerance;
////	mu_initialIss                     >> clo.mu_initial;
////	damping_increase_factorIss        >> clo.damping_increase_factor;
////	damping_decrease_factorIss        >> clo.damping_decrease_factor;
////	damping_increase_toleranceIss     >> clo.damping_increase_tolerance;
////	damping_decrease_toleranceIss     >> clo.damping_decrease_tolerance;
//////	should_scale_hessianIss           >> clo.should_scale_hessian;
////	size_LBFGSIss                     >> clo.size_LBFGS;
////	solver_option_pdnr_ss             >> clo.solver_option_pdnr;
////	threshold_nonz_per_row_pdnr_ss    >> clo.threshold_nonz_per_row_pdnr;
//
////	if (!inputOptionsFile.empty())
////    {
////
////    SimpleInput input(clo.inputOptionsFile, SimpleInput::READ);
////    InputOptions inputOptions = input.read();
////
////	  clo.inputFile                      = inputOptions.inputFile.empty() ? clo.inputFile : inputOptions.inputFile;
////	  clo.outputFile                     = inputOptions.outputFile.empty() ? clo.outputFile : inputOptions.outputFile;
////	  clo.workingPath                    = inputOptions.workingPath.empty() ? clo.workingPath : inputOptions.workingPath;
////	  clo.scheduling                     = inputOptions.scheduling.empty() ? clo.scheduling : inputOptions.scheduling;
////	  clo.localOrdinalType               = inputOptions.localOrdinalType.empty() ? clo.localOrdinalType : inputOptions.localOrdinalType;
////	  clo.globalOrdinalType              = inputOptions.globalOrdinalType.empty() ? clo.globalOrdinalType : inputOptions.globalOrdinalType;
////	  clo.sparseValueType                = inputOptions.sparseValueType.empty() ? clo.sparseValueType : inputOptions.sparseValueType;
////	  clo.kruskalValueType               = inputOptions.kruskalValueType.empty() ? clo.kruskalValueType : inputOptions.kruskalValueType;
////	  clo.logFile                        = inputOptions.logFile.empty() ? clo.logFile : inputOptions.logFile;
////	  clo.print_level                    = inputOptions.print_level.empty() ? clo.print_level : inputOptions.print_level;
////	  clo.indexStyle                     = inputOptions.indexStyle.empty() ? clo.indexStyle : inputOptions.indexStyle;
////
////	  clo.tolerance                      = !toleranceStr.empty() ? clo.tolerance : inputOptions.tolerance;
////	  clo.maxOuterIter                   = !maxOuterIterStr.empty() ? clo.maxOuterIter : inputOptions.maxOuterIter;
////	  clo.maxInnerIter                   = !maxInnerIterStr.empty() ? clo.maxInnerIter : inputOptions.maxInnerIter;
////	  clo.offset                         = !offsetStr.empty() ? clo.offset : inputOptions.offset;
////	  clo.offsetTolerance                = !offsetToleranceStr.empty() ? clo.offsetTolerance : inputOptions.offsetTolerance;
////	  clo.progressInterval               = !progressIntervalStr.empty() ? clo.progressInterval : inputOptions.progressInterval;
////	  clo.eps                            = !epsStr.empty() ? clo.eps : inputOptions.eps;
////	  clo.precision                      = !precisionStr.empty() ? clo.precision : inputOptions.precision;
////	  clo.nComponent                     = !nComponentStr.empty() ? clo.nComponent : inputOptions.nComponent;
////	  clo.solver                         = !clo.solver.empty() ? inputOptions.solver : clo.solver;
////	  clo.randomSeed                     = !clo.randomSeed.empty() ? clo.randomSeed : inputOptions.randomSeed;
////	  clo.eps_div_zero_grad              = !eps_div_zero_gradStr.empty() ? clo.eps_div_zero_grad : inputOptions.eps_div_zero_grad;
////	  clo.log_zero_safeguard             = !log_zero_safeguardStr.empty() ? clo.log_zero_safeguard : inputOptions.log_zero_safeguard;
////	  clo.eps_active_set                 = !eps_active_setStr.empty() ? clo.eps_active_set : inputOptions.eps_active_set;
////	  clo.max_backtrack_steps            = !max_backtrack_stepsStr.empty() ? clo.max_backtrack_steps : inputOptions.max_backtrack_steps;
////	  clo.min_variable_nonzero_tolerance = !min_variable_nonzero_toleranceStr.empty() ? clo.min_variable_nonzero_tolerance : inputOptions.min_variable_nonzero_tolerance;
////	  clo.step_reduction_factor          = !step_reduction_factorStr.empty() ? clo.step_reduction_factor : inputOptions.step_reduction_factor;
////	  clo.suff_decrease_tolerance        = !suff_decrease_toleranceStr.empty() ? clo.suff_decrease_tolerance : inputOptions.suff_decrease_tolerance;
////	  clo.mu_initial                     = !mu_initialStr.empty() ? clo.mu_initial : inputOptions.mu_initial;
////	  clo.damping_increase_factor        = !damping_increase_factorStr.empty() ? clo.damping_increase_factor : inputOptions.damping_increase_factor;
////	  clo.damping_decrease_factor        = !damping_decrease_factorStr.empty() ? clo.damping_decrease_factor : inputOptions.damping_decrease_factor;
////	  clo.damping_increase_tolerance     = !damping_increase_toleranceStr.empty() ? clo.damping_increase_tolerance : inputOptions.damping_increase_tolerance;
////	  clo.damping_decrease_tolerance     = !damping_decrease_toleranceStr.empty() ? clo.damping_decrease_tolerance : inputOptions.damping_decrease_tolerance;
//////	  clo.should_scale_hessian           = !should_scale_hessianStr.empty() ? clo.should_scale_hessian : inputOptions.should_scale_hessian;
////	  clo.size_LBFGS                     = !size_LBFGSStr.empty() ? clo.size_LBFGS : inputOptions.size_LBFGS;
////	  clo.solver_option_pdnr             = !solver_option_pdnr_str.empty() ? clo.solver_option_pdnr : inputOptions.solver_option_pdnr;
////	  clo.threshold_nonz_per_row_pdnr    = !threshold_nonz_per_row_pdnr_str.empty() ?  clo.threshold_nonz_per_row_pdnr : inputOptions.threshold_nonz_per_row_pdnr;
////
////  }
//
//  return clo;
//
//}

//! Convenience function to get log type (screen, file, ...)
sparten::Log::LogType get_log_type(sparten::CommandLineOptions const &clo)
{
  if (clo.logFile.empty())
  {
    return sparten::Log::TO_SCREEN;
  } else
  {
    return sparten::Log::TO_FILE;
  }
}

//! Convenience function to get log print level
sparten::Log::Verbosity get_print_level(sparten::CommandLineOptions const &clo)
{
    sparten::Log::Verbosity level;
    if (clo.debug)
    {
        switch (clo.print_level)
        {
            case 0:
                level = sparten::Log::DEBUG_1;
                break;
            case 1:
                level = sparten::Log::DEBUG_2;
                break;
            case 2:
                level = sparten::Log::DEBUG_3;
                break;
            default:
                level = sparten::Log::RELEASE;
                break;
        }
    }
    else
    {
        switch (clo.print_level)
        {
            case 0:
                level = sparten::Log::QUIET;
                break;
            case 1:
                level = sparten::Log::RELEASE;
                break;
            case 2:
                level = sparten::Log::VERBOSE;
                break;
            default:
                level = sparten::Log::RELEASE;
                break;
        }
    }
    return level;
    /*
	if (clo.print_level.compare("0") == 0)
	{
		return sparten::Log::QUIET;
	}
	else if (clo.print_level.compare("1") == 0)
	{
		return sparten::Log::RELEASE;
	}
	else if (clo.print_level.compare("2") == 0)
	{
		return sparten::Log::VERBOSE;
	}
	else if (clo.print_level.compare("3") == 0)
	{
		return sparten::Log::DEBUG_1;
	}
	else if (clo.print_level.compare("4") == 0)
	{
		return sparten::Log::DEBUG_2;
	}
	else if (clo.print_level.compare("5") == 0)
	{
		return sparten::Log::DEBUG_3;
	} // Remaining options will be deprecated in a future version.
  else if (clo.print_level.compare("quiet") == 0)
  {
    return sparten::Log::QUIET;
  }
  else if (clo.print_level.compare("release") == 0)
  {
    return sparten::Log::RELEASE;
  }
  else if (clo.print_level.compare("debug_0") == 0)
  {
    return sparten::Log::VERBOSE;
  }
  else if (clo.print_level.compare("debug_1") == 0)
  {
    return sparten::Log::DEBUG_1;
  }
  else if (clo.print_level.compare("debug_2") == 0)
  {
    return sparten::Log::DEBUG_2;
  }
  else if (clo.print_level.compare("debug_3") == 0)
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
      return sparten::Log::RELEASE;
    }
  }
     */
}

void usage(char** argv)
{
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "--------------------------------";
    std::cout << " SparTen v " << SPARTEN_VERSION << " ";
    std::cout << "-------------------------------\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Usage: " << argv[0] << " [options]\n";
    std::cout << "Driver options: \n";
    std::cout << "  --input       <string>  File containing tensor data\n";
    std::cout << "  --output      <string>  File where ktensor factorization will be stored\n";
    std::cout << "  --init        <string>  File containing ktensor initial guess (leave blank for random initial guess)\n";
    std::cout << "  --index-base  <int>     Starting index for tensor data {0,1}\n";
    //\TODO: Include options if calling SparTen as problem generator
    std::cout << std::endl;
    sparten::CommandLineOptions::print_help (std::cout);
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
}

int main(int argc, char **argv)
{
  //// Initialization ////
  Kokkos::initialize(argc, argv);

  try
  {
    //// Main program ////
    // Get command line options
    auto args = sparten::get_arg_list(argc, argv);

    auto help = sparten::parse_bool ( args, "--help", "--no-help", false);
    if ((argc) < 2 || (help))
    {
        usage(argv);
        Kokkos::finalize();
        return 0;
    }

    std::string input_file_name = sparten::parse_string ( args, "--input", "");
    if (input_file_name.empty())
    {
        throw std::runtime_error("No input file provided. Exiting.");
    }

    sparten::CommandLineOptions clo;
    clo.inputFile = input_file_name;
    clo.get_command_line_options(args);

    /* TODO Process options file & return args to clo; will require a copy constructor from SimpleInput down to
    auto options_file = sparten::parse_string(args, "--options-file", "");
    if (!options_file.empty())
    {
        std::cout << "Input options file: " << options_file << std::endl;

        //  CommandLineOption
        // sparten::SimpleInput input_from_file(options_file,sparten::SimpleInput::READ);
        // auto clo = input_from_file.read();
    }
    else
    {
    }
     */

    // Setup logging
    sparten::Log &log = sparten::Log::new_log(get_log_type(clo),clo.logFile,get_print_level(clo));

    std::ostringstream kokkos_config_msg;
    Kokkos::print_configuration(kokkos_config_msg);
    log.print(kokkos_config_msg.str(), sparten::Log::VERBOSE);
    
    std::ostringstream welcome_msg;
    welcome_msg << "------------------------------------------------------------------------------------------\n";
    welcome_msg << "------------------------------------- Sparten v " << std::string(SPARTEN_VERSION);
    welcome_msg << " ------------------------------------\n";
    welcome_msg << "------------------------------------------------------------------------------------------";
    log.print (welcome_msg.str(),sparten::Log::VERBOSE);
    welcome_msg.str("");
    clo.print_options(welcome_msg);

    // Date + Current Time + Timer
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::ostringstream ss_start;
    ss_start << "\nStarted: " << std::ctime(&start_time);
    log.print(ss_start.str(), sparten::Log::VERBOSE);

    // Do stuff here
    MainWorkflow mainWorkflow(clo);

    //// Finalization ////
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    log.print("Elapsed time: " + std::to_string(elapsed_seconds.count()), sparten::Log::VERBOSE);

    end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::ostringstream ss_finish;
    ss_finish << "Finished: " << std::ctime(&end_time)
              << "------------------------------------------------------------------------------------------"
              << "\n------------------------------------------------------------------------------------------"
              << "\n------------------------------------------------------------------------------------------";
    log.print(ss_finish.str(), sparten::Log::VERBOSE);
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
