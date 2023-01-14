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

//#include <iostream>
#include <string>
#include <Sparten_Configure.h>
#include <sparten/Log.h>
#include <sstream>
#include <unistd.h>
#include <sparten/CommandLineOptions.h>
#include <fstream>
#include <sparten/SimpleInput.h>

sparten::SimpleInput::SimpleInput( std::string const &filename, Access mode ) :
        _filename ( filename ),
        _mode ( mode ),
        _isFileOpen ( false ),
        _format ( SCIENTIFIC ),
        _fileType ( "" ) {

    // TODO File fails to open on MacOSX with clang

    _file.exceptions(std::fstream::failbit | std::fstream::badbit);

    switch (mode) {
        case READ:
            try {
                _file.open ( filename, std::fstream::in );
                _isFileOpen = true;
            }
            catch (std::fstream::failure& err)
            {
                std::cerr << "std::fstream_file.good() = " << _file.good() << std::endl;
                std::cerr << "std::fstream_file.eof() = " << _file.eof() << std::endl;
                std::cerr << "std::fstream_file.fail() = " << _file.fail() << std::endl;
                std::cerr << "std::fstream _file.bad() = " << _file.bad() << std::endl;
                std::cerr << "strerror(errno) =  " << strerror(errno) << std::endl;
                std::cerr << "std::ifstream::fail& err.code(): " << err.code() << std::endl;
                std::cerr << "std::ifstream::fail& err.what(): " << err.what() << std::endl;
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
        throw std::runtime_error ( "Input options file failed to open." );
    }
}

sparten::SimpleInput::~SimpleInput() {
    if (_isFileOpen) {
        _file.close ();
    }
}

void sparten::SimpleInput::open_file() {
    _file.open ( _filename.c_str ());

    if (!_file.is_open ()) {
        throw std::runtime_error ( "Input options file failed to open." );
    } else {
        _isFileOpen = true;
    }
}

void sparten::SimpleInput::close_file() {
    if (_isFileOpen) {
        _file.close ();
    } else {
        throw std::runtime_error ( "Input options file is already closed." );
    }
}

void sparten::SimpleInput::get_line_error_check( std::fstream const &input ) {
    if ((input.fail ()) != 0) {
        throw std::runtime_error ( "Logical I/O failure." );
    }

    if ((input.bad ()) != 0) {
        throw std::runtime_error ( "I/O operation error." );
    }
}

sparten::InputOptions sparten::SimpleInput::read() {
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