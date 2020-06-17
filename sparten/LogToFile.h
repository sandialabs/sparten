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
 \brief Class declaration for runtime-selectable logging system (to file)
 */

#pragma once

#include <sparten/Log.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sparten
{

//! Logger with runtime verbosity selection (This overwrites an existing file. (i.e., does not append)
class LogToFile : public Log
{
public:
  LogToFile(std::string const &filename, Verbosity level);                   //!< Constructor
  virtual ~LogToFile();                                                      //!< Destructor

  virtual void print(std::string const &message, Verbosity level) override;  //!< Prints message to file
  virtual void finalize() override;                                          //!< Writes deferred output to file

protected:
  std::ofstream _file;                //!< File to be written.
  std::stringstream _loggedData;      //!< Stringstream that accumulates output to be written at program shutdown (Deferred file i/o for improved performance.)
};

} // sparten
