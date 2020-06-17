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
  uint64_t numComponents;
  uint64_t maxNumNonzeros;
  uint64_t numDimensions;
  double ktensorThreshhold;
  double ktensorNoise;
  std::string sptensorOutputFile;
  std::string ktensorOutputFile;
  std::string indexStyle;
  std::string randomSeed;
  std::vector<std::string> sizeDim;
  std::vector<sparten::type::SubIdx> cDims;
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

      if (param.compare("num-components") == 0)
      {
        iss >> inputOptions.numComponents;
      }
      else if (param.compare("max-num-nonzeros") == 0)
      {
        iss >> inputOptions.maxNumNonzeros;
      }
      else if (param.compare("num-dimensions") == 0)
      {
        iss >> inputOptions.numDimensions;
      }
      else if (param.compare("ktensor-threshhold") == 0)
      {
        iss >> inputOptions.ktensorThreshhold;
      }
      else if (param.compare("ktensor-noise") == 0)
      {
        iss >> inputOptions.ktensorNoise;
      } 
      else if (param.compare("sptensor-output-file") == 0)
      {
        iss >> inputOptions.sptensorOutputFile;
      }
      else if (param.compare("ktensor-output-file") == 0)
      {
        iss >> inputOptions.ktensorOutputFile;
      }
      else if (param.compare("random-seed") == 0)
      {
        iss >> inputOptions.randomSeed;
      }
      else if (param.compare("index-style") == 0)
      {
        iss >> inputOptions.indexStyle;
      }
      else
      {
        throw std::runtime_error("Unknown parameter in input options file.");
      }
    }
  }
  else
  {
    throw std::runtime_error("Input option file needs to be open before reading.");
  }

  return inputOptions;
}

