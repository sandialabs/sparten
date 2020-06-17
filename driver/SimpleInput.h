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

class InputOptions
{
public:

	/* Required Arguments */
	uint64_t    nComponent;
	std::string solver;
	std::string randomSeed;
	std::string inputFile;
	std::string outputFile;
	std::string scheduling;

	/* Convergence Criteria */
	double   tolerance;
	uint64_t maxOuterIter;
	uint64_t maxInnerIter;
	double   eps;
	double   offset;
	double   offsetTolerance;

	/* Row Subproblem Configurations */
	double   eps_div_zero_grad;
	double   log_zero_safeguard;
	double   eps_active_set;
	uint64_t max_backtrack_steps;
	double   min_variable_nonzero_tolerance;
	double   step_reduction_factor;
	double   suff_decrease_tolerance;
	double   mu_initial;
	double   damping_increase_factor;
	double   damping_decrease_factor;
	double   damping_increase_tolerance;
	double   damping_decrease_tolerance;
	bool     should_scale_hessian;
	uint64_t size_LBFGS;
	uint64_t solver_option_pdnr;
	uint64_t threshold_nonz_per_row_pdnr;

	/* Data Types */
	std::string localOrdinalType;
	std::string globalOrdinalType;
	std::string sparseValueType;
	std::string kruskalValueType;

	/* Data Input */
	std::string indexStyle;

	/* Data Output */
	uint64_t precision;

	/* Logging */
	uint64_t    progressInterval;
	std::string logFile;
	std::string logVerbosity;

	/* System */
	std::string mode;
	std::string workingPath;
	std::string asciiFormat;
};

//! File I/O for ASCII files
class SimpleInput
{
public:
  //! File I/O mode
  enum Access
  {
    READ,           //!< Read-only
    WRITE_APPEND,   //!< Write, append to end of file
    OVERWRITE,      //!< Write, erase any previous data in file
    READ_WRITE      //!< Read and write access to same file
  };

  //! ASCII number formatting
  enum Format
  {
    FIXED_PRECISION,  //!< Prints numbers in ASCII file using fixed precision (user-specified precision)
    SCIENTIFIC        //!< Prints numbers in ASCII file using scientific notation
  };

  SimpleInput(std::string const &filename, Access mode);                                    //!< Constructor
  virtual ~SimpleInput();                                                                   //!< Destructor

  virtual void open_file();     //!< Opens the specified file
  virtual void close_file();    //!< Closes the specified file

  virtual InputOptions read();

protected:
  bool _isFileOpen;       //!< File open status
  Access _mode;           //!< File access mode
  Format _format;         //!< Number format
  std::string _filename;  //!< Filename
  std::fstream _file;     //!< File
  std::string _fileType;  //!< Format type ("sptensor", etc.)

  void get_line_error_check(std::fstream const &input);
};

SimpleInput::SimpleInput(std::string const &filename, Access mode) :
    _filename(filename),
    _mode(mode),
    _isFileOpen(false),
    _format(SCIENTIFIC),
    _fileType("")
{
  switch (mode)
  {
  case READ:
    _file.open(filename.c_str(), std::fstream::in);
    _isFileOpen = true;
    break;
  case WRITE_APPEND:
    _file.open(filename.c_str(), std::fstream::out | std::fstream::app);
    _isFileOpen = true;
    break;
  case OVERWRITE:
    _file.open(filename.c_str(), std::fstream::out);
    _isFileOpen = true;
    break;
  case READ_WRITE:
    _file.open(filename.c_str(), std::fstream::in | std::fstream::out);
    _isFileOpen = true;
    break;
  default:
    throw std::runtime_error("Invalid file I/O mode selected.");
  }

  if (!_file.is_open())
  {
    throw std::runtime_error("Input options file failed to open.");
  }
}

SimpleInput::~SimpleInput()
{
  if (_isFileOpen)
  {
    _file.close();
  }
}

void SimpleInput::open_file()
{
  _file.open(_filename.c_str());

  if (!_file.is_open())
  {
    throw std::runtime_error("Input options file failed to open.");
  }
  else
  {
    _isFileOpen = true;
  }
}

void SimpleInput::close_file()
{
  if (_isFileOpen)
  {
    _file.close();
  }
  else
  {
    throw std::runtime_error("Input options file is already closed.");
  }
}

void SimpleInput::get_line_error_check(std::fstream const &input)
{
  if ((input.fail()) != 0)
  {
    throw std::runtime_error("Logical I/O failure.");
  }

  if ((input.bad()) != 0)
  {
    throw std::runtime_error("I/O operation error.");
  }
}

InputOptions SimpleInput::read()
{
  InputOptions inputOptions;

  if (_isFileOpen)
  {
    std::string line;

    while (std::getline(_file, line))
    {
      get_line_error_check(_file);
      std::istringstream iss(line);

      std::string param;
      iss >> param;

      if (param.find("#") == 0)
	continue;

	    if (param.compare("tolerance") == 0)
	    {
		    iss >> inputOptions.tolerance;
	    }
	    else if (param.compare("maxOuterIter") == 0)
	    {
		    iss >> inputOptions.maxOuterIter;
	    }
	    else if (param.compare("maxInnerIter") == 0)
	    {
		    iss >> inputOptions.maxInnerIter;
	    }
	    else if (param.compare("offset") == 0)
	    {
		    iss >> inputOptions.offset;
	    }
	    else if (param.compare("offsetTolerance") == 0)
	    {
		    iss >> inputOptions.offsetTolerance;
	    }
	    else if (param.compare("progressInterval") == 0)
	    {
		    iss >> inputOptions.progressInterval;
	    }
	    else if (param.compare("eps") == 0)
	    {
		    iss >> inputOptions.eps;
	    }
	    else if (param.compare("precision") == 0)
	    {
		    iss >> inputOptions.precision;
	    }
	    else if (param.compare("nComponent") == 0)
	    {
		    iss >> inputOptions.nComponent;
	    }
	    else if (param.compare("eps_div_zero_grad") == 0)
	    {
		    iss >> inputOptions.eps_div_zero_grad;
	    }
	    else if (param.compare("log_zero_safeguard") == 0)
	    {
		    iss >> inputOptions.log_zero_safeguard;
	    }
	    else if (param.compare("eps_active_set") == 0)
	    {
		    iss >> inputOptions.eps_active_set;
	    }
	    else if (param.compare("max_backtrack_steps") == 0)
	    {
		    iss >> inputOptions.max_backtrack_steps;
	    }
	    else if (param.compare("min_variable_nonzero_tolerance") == 0)
	    {
		    iss >> inputOptions.min_variable_nonzero_tolerance;
	    }
	    else if (param.compare("step_reduction_factor") == 0)
	    {
		    iss >> inputOptions.step_reduction_factor;
	    }
	    else if (param.compare("suff_decrease_tolerance") == 0)
	    {
		    iss >> inputOptions.suff_decrease_tolerance;
	    }
	    else if (param.compare("mu_initial") == 0)
	    {
		    iss >> inputOptions.mu_initial;
	    }
	    else if (param.compare("damping_increase_factor") == 0)
	    {
		    iss >> inputOptions.damping_increase_factor;
	    }
	    else if (param.compare("damping_decrease_factor") == 0)
	    {
		    iss >> inputOptions.damping_decrease_factor;
	    }
	    else if (param.compare("damping_increase_tolerance") == 0)
	    {
		    iss >> inputOptions.damping_increase_tolerance;
	    }
	    else if (param.compare("damping_decrease_tolerance") == 0)
	    {
		    iss >> inputOptions.damping_decrease_tolerance;
	    }
	    else if (param.compare("should_scale_hessian") == 0)
	    {
		    iss >> inputOptions.should_scale_hessian;
	    }
	    else if (param.compare("size_LBFGS") == 0)
	    {
		    iss >> inputOptions.size_LBFGS;
	    }
	    else if (param.compare("solver") == 0)
	    {
		    iss >> inputOptions.solver;
	    }
	    else if (param.compare("randomSeed") == 0)
	    {
		    iss >> inputOptions.randomSeed;
	    }
	    else if (param.compare("mode") == 0)
	    {
		    iss >> inputOptions.mode;
	    }
	    else if (param.compare("input-file") == 0)
	    {
		    iss >> inputOptions.inputFile;
	    }
	    else if (param.compare("output-file") == 0)
	    {
		    iss >> inputOptions.outputFile;
	    }
	    else if (param.compare("working-path") == 0)
	    {
		    iss >> inputOptions.workingPath;
	    }
	    else if (param.compare("local-ordinal-type") == 0)
	    {
		    iss >> inputOptions.localOrdinalType;
	    }
	    else if (param.compare("global-ordinal-type") == 0)
	    {
		    iss >> inputOptions.globalOrdinalType;
	    }
	    else if (param.compare("sparse-value-type") == 0)
	    {
		    iss >> inputOptions.sparseValueType;
	    }
	    else if (param.compare("kruskal-value-type") == 0)
	    {
		    iss >> inputOptions.kruskalValueType;
	    }
	    else if (param.compare("log-file") == 0)
	    {
		    iss >> inputOptions.logFile;
	    }
	    else if (param.compare("log-verbosity") == 0)
	    {
		    iss >> inputOptions.logVerbosity;
	    }
	    else if (param.compare("ascii-format") == 0)
	    {
		    iss >> inputOptions.asciiFormat;
	    }
	    else if (param.compare("index-style") == 0)
	    {
		    iss >> inputOptions.indexStyle;
	    }
      else
      {
        throw std::runtime_error("Unknown parameter in input options file: \"" + param + "\"");
      }
    }
  }
  else
  {
    throw std::runtime_error("Input option file needs to be open before reading.");
  }

  return inputOptions;
}

