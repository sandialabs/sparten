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
 \brief Class declaration for simple text input file for SparTen main driver options
 */

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sparten/CommandLineOptions.h>

namespace sparten {
    class InputOptions : public CommandLineOptions {
      public:
        InputOptions() : CommandLineOptions() { std::cout << "class InputOptions : public CommandLineOptions" << std::endl; };
        virtual ~InputOptions() = default;
        virtual void read() {};


//	/* Required Arguments */
//	uint64_t    nComponent;
//	std::string solver;
//	std::string randomSeed;
//	std::string inputFile;
//	std::string outputFile;
//	std::string scheduling;
//
//	/* Convergence Criteria */
//	double   tolerance;
//	uint64_t maxOuterIter;
//	uint64_t maxInnerIter;
//	double   eps;
//	double   offset;
//	double   offsetTolerance;
//
//	/* Row Subproblem Configurations */
//	double   eps_div_zero_grad;
//	double   log_zero_safeguard;
//	double   eps_active_set;
//	uint64_t max_backtrack_steps;
//	double   min_variable_nonzero_tolerance;
//	double   step_reduction_factor;
//	double   suff_decrease_tolerance;
//	double   mu_initial;
//	double   damping_increase_factor;
//	double   damping_decrease_factor;
//	double   damping_increase_tolerance;
//	double   damping_decrease_tolerance;
//	bool     should_scale_hessian;
//	uint64_t size_LBFGS;
//	uint64_t solver_option_pdnr;
//	uint64_t threshold_nonz_per_row_pdnr;
//
//	/* Data Types */
//	std::string localOrdinalType;
//	std::string globalOrdinalType;
//	std::string sparseValueType;
//	std::string kruskalValueType;
//
//	/* Data Input */
//	std::string indexStyle;
//
//	/* Data Output */
//	uint64_t precision;
//
//	/* Logging */
//	uint64_t    progressInterval;
//	std::string logFile;
//	std::string print_level;
//
//	/* System */
//	std::string mode;
//	std::string workingPath;
//	std::string asciiFormat;
//
//	/* Ktensor initial guess file */
//	std::string init_file;
    };

//! File I/O for ASCII files
    class SimpleInput {
      public:
        //! File I/O mode
        enum Access {
            READ,           //!< Read-only
            WRITE_APPEND,   //!< Write, append to end of file
            OVERWRITE,      //!< Write, erase any previous data in file
            READ_WRITE      //!< Read and write access to same file
        };

        //! ASCII number formatting
        enum Format {
            FIXED_PRECISION,  //!< Prints numbers in ASCII file using fixed precision (user-specified precision)
            SCIENTIFIC        //!< Prints numbers in ASCII file using scientific notation
        };

        SimpleInput( std::string const &filename, Access mode );                                    //!< Constructor
        virtual ~SimpleInput();                                                                   //!< Destructor

        virtual void open_file();     //!< Opens the specified file
        virtual void close_file();    //!< Closes the specified file

        virtual InputOptions read();

      protected:
        bool _isFileOpen;       //!< File open status
        Access _mode;           //!< File access mode
        Format _format;         //!< Number format
        std::string _filename;  //!< Filename
        std::fstream _file;    //!< File
        std::string _fileType;  //!< Format type ("sptensor", etc.)

        void get_line_error_check( std::fstream const &input );
    };

    /*
    SimpleInput::SimpleInput( std::string const &filename, Access mode ) :
            _filename ( filename ),
            _mode ( mode ),
            _isFileOpen ( false ),
            _format ( SCIENTIFIC ),
            _fileType ( "" ) {

        std::cout << "_filename = " << _filename << std::endl;
        std::cout << "_mode = " << _mode << std::endl;
        std::cout << "_isFileOpen = " << _isFileOpen << std::endl;
        std::cout << "_format = " << _format << std::endl;
        std::cout << "_fileType = " << _fileType << std::endl;

        _file.exceptions(std::fstream::failbit | std::fstream::badbit);

        switch (mode) {
            case READ:
                try {
                    _file.open ( filename.c_str (), std::fstream::in );
                    _isFileOpen = true;
                }
                catch (std::system_error& err)
                {
                    std::cerr << "Error: " << strerror(errno) << std::endl;
                }
                break;
            case WRITE_APPEND:
                _file.open ( filename.c_str (), std::fstream::out | std::fstream::app );
                _isFileOpen = true;
                break;
            case OVERWRITE:
                _file.open ( filename.c_str (), std::fstream::out );
                _isFileOpen = true;
                break;
            case READ_WRITE:
                _file.open ( filename.c_str (), std::fstream::in | std::fstream::out );
                _isFileOpen = true;
                break;
            default:
                throw std::runtime_error ( "Invalid file I/O mode selected." );
        }

        if (!_file.is_open ()) {
            throw std::runtime_error ( "hi hi 2: Input options file failed to open." );
        }
    }

    SimpleInput::~SimpleInput() {
        if (_isFileOpen) {
            _file.close ();
        }
    }

    void SimpleInput::open_file() {
        _file.open ( _filename.c_str ());

        if (!_file.is_open ()) {
            throw std::runtime_error ( "hi hi: Input options file failed to open." );
        } else {
            _isFileOpen = true;
        }
    }

    void SimpleInput::close_file() {
        if (_isFileOpen) {
            _file.close ();
        } else {
            throw std::runtime_error ( "Input options file is already closed." );
        }
    }

    void SimpleInput::get_line_error_check( std::fstream const &input ) {
        if ((input.fail ()) != 0) {
            throw std::runtime_error ( "Logical I/O failure." );
        }

        if ((input.bad ()) != 0) {
            throw std::runtime_error ( "I/O operation error." );
        }
    }

    InputOptions SimpleInput::read() {
        InputOptions inputOptions;

        if (_isFileOpen) {
            std::string line;

            while (std::getline ( _file, line )) {
                get_line_error_check ( _file );
                std::istringstream iss ( line );

                std::string param;
                iss >> param;

                if (param.find ( "#" ) == 0)
                    continue;

                if (param.compare ( "tolerance" ) == 0) {
                    iss >> inputOptions.tolerance;
                } else if (param.compare ( "max-outer-iter" ) == 0) {
                    iss >> inputOptions.maxOuterIter;
                } else if (param.compare ( "max-inner-iter" ) == 0) {
                    iss >> inputOptions.maxInnerIter;
                } else if (param.compare ( "offset" ) == 0) {
                    iss >> inputOptions.offset;
                } else if (param.compare ( "offset-tolerance" ) == 0) {
                    iss >> inputOptions.offsetTolerance;
                } else if (param.compare ( "progress-interval" ) == 0) {
                    iss >> inputOptions.progressInterval;
                } else if (param.compare ( "eps" ) == 0) {
                    iss >> inputOptions.eps;
                } else if (param.compare ( "precision" ) == 0) {
                    iss >> inputOptions.precision;
                } else if (param.compare ( "num-components" ) == 0) {
                    iss >> inputOptions.nComponent;
                } else if (param.compare ( "eps-div-zero-grad" ) == 0) {
                    iss >> inputOptions.eps_div_zero_grad;
                } else if (param.compare ( "log-zero-safeguard" ) == 0) {
                    iss >> inputOptions.log_zero_safeguard;
                } else if (param.compare ( "eps-active-set" ) == 0) {
                    iss >> inputOptions.eps_active_set;
                } else if (param.compare ( "max-backtrack-steps" ) == 0) {
                    iss >> inputOptions.max_backtrack_steps;
                } else if (param.compare ( "min-step-size" ) == 0) {
                    iss >> inputOptions.min_variable_nonzero_tolerance;
                } else if (param.compare ( "step-reduction-factor" ) == 0) {
                    iss >> inputOptions.step_reduction_factor;
                } else if (param.compare ( "suff-decrease-tolerance" ) == 0) {
                    iss >> inputOptions.suff_decrease_tolerance;
                } else if (param.compare ( "mu-initial" ) == 0) {
                    iss >> inputOptions.mu_initial;
                } else if (param.compare ( "damping-increase-factor" ) == 0) {
                    iss >> inputOptions.damping_increase_factor;
                } else if (param.compare ( "damping-decrease-factor" ) == 0) {
                    iss >> inputOptions.damping_decrease_factor;
                } else if (param.compare ( "damping-increase-tolerance" ) == 0) {
                    iss >> inputOptions.damping_increase_tolerance;
                } else if (param.compare ( "damping-decrease-tolerance" ) == 0) {
                    iss >> inputOptions.damping_decrease_tolerance;
                } else if (param.compare ( "should-scale-hessian" ) == 0) {
                    iss >> inputOptions.should_scale_hessian;
                } else if (param.compare ( "size-lbfgs" ) == 0) {
                    iss >> inputOptions.size_LBFGS;
                } else if (param.compare ( "solver" ) == 0) {
                    iss >> inputOptions.solver;
                } else if (param.compare ( "seed" ) == 0) {
                    iss >> inputOptions.randomSeed;
                } else if (param.compare ( "mode" ) == 0) {
                    iss >> inputOptions.mode;
                } else if (param.compare ( "input-file" ) == 0) {
                    iss >> inputOptions.inputFile;
                } else if (param.compare ( "output-file" ) == 0) {
                    iss >> inputOptions.outputFile;
                } else if (param.compare ( "working-path" ) == 0) {
                    iss >> inputOptions.workingPath;
                } else if (param.compare ( "local-ordinal-type" ) == 0) {
                    iss >> inputOptions.localOrdinalType;
                } else if (param.compare ( "global-ordinal-type" ) == 0) {
                    iss >> inputOptions.globalOrdinalType;
                } else if (param.compare ( "sparse-value-type" ) == 0) {
                    iss >> inputOptions.sparseValueType;
                } else if (param.compare ( "kruskal-value-type" ) == 0) {
                    iss >> inputOptions.kruskalValueType;
                } else if (param.compare ( "log-file" ) == 0) {
                    iss >> inputOptions.logFile;
                } else if (param.compare ( "print-level" ) == 0) {
                    iss >> inputOptions.print_level;
                } else if (param.compare ( "ascii-format" ) == 0) {
                    iss >> inputOptions.asciiFormat;
                } else if (param.compare ( "index-style" ) == 0) {
                    iss >> inputOptions.indexStyle;
                } else if (param.compare ( "init" ) == 0) {
                    iss >> inputOptions.init_file;
                } else {
                    throw std::runtime_error ( "Unknown parameter in input options file: \"" + param + "\"" );
                }
            }
        } else {
            throw std::runtime_error ( "Input option file needs to be open before reading." );
        }

        return inputOptions;
    }
    */

} // namespace sparten