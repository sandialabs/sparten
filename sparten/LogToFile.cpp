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
 \brief Class definition for runtime-selectable logging system (to file)
 */

#include <sparten/LogToFile.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sparten
{

LogToFile::LogToFile(std::string const &filename, Verbosity level) :
        Log(level)
{
  _file.open(filename.c_str());

  if (!_file.is_open())
  {
    throw std::runtime_error("Log file failed to open.");
  }
}

LogToFile::~LogToFile()
{
  this->finalize();
  _file.close();
}

void LogToFile::print(std::string const &message, Verbosity level)
{
  if (_level >= level && _level != QUIET)
  {
    _loggedData << message << std::endl;
  }
}

void LogToFile::finalize()
{
  _file << _loggedData.str() << std::endl;
}

} // sparten
