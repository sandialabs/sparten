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
#include <sparten/Log.h>
#include <Sparten_Configure.h>
#include <sstream>
#include <unistd.h>

namespace sparten {

    // Struct for handling Sparten parameters
    struct CommandLineOptions {
        /* Generic options */
        uint64_t nComponent;
        std::string solver;
        uint64_t randomSeed;
        std::string inputFile;
        std::string outputFile;
        std::string scheduling;
        double tolerance;
        uint64_t maxOuterIter;
        uint64_t maxInnerIter;

        /* MU options */
        double eps;
        double offset;
        double offsetTolerance;

        /* PDNR & PQNR common options */
        double eps_div_zero_grad;
        double log_zero_safeguard;
        double eps_active_set;

        /* PDNR-only options */
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
        uint64_t solver_option_pdnr;
        uint64_t threshold_nonz_per_row_pdnr;

        /* PQNR-only options */
        uint64_t size_LBFGS;

        /* System, logging, I/O options */
        std::string localOrdinalType;
        std::string globalOrdinalType;
        std::string sparseValueType;
        std::string kruskalValueType;
        std::string indexStyle;
        uint64_t precision;
        uint64_t progressInterval;
        bool debug;
        std::string logFile;
        uint64_t print_level;
        std::string mode;
        std::string asciiFormat;
        // std::string inputOptionsFile;
        std::string init_file;

        // Constructor for defaults
        CommandLineOptions();

        // Wrapper function to parse command line options
        void get_command_line_options(std::vector<std::string>& args);

        // Print help screen
        static void print_help(std::ostream &out);

        // Print options
        void print_options(std::ostringstream &out) const;
    };

    bool check_and_print_unused_args(const std::vector<std::string>& args, std::ostream& out);

    std::string parse_string(std::vector<std::string>& args, const std::string& cl_arg, const std::string& default_val);

    uint64_t parse_uint64_t(std::vector<std::string>& args, const std::string& cl_arg, uint64_t default_val, uint64_t min=0, uint64_t max=100);

    double parse_double(std::vector<std::string>& args, const std::string& cl_arg, double default_val, double min=0.0, double max=1.0);

    bool parse_bool(std::vector<std::string>& args, const std::string& cl_arg_on, const std::string& cl_arg_off, bool default_value);

    std::vector<std::string> get_arg_list(int argc, char** argv);

} // end namespace sparten