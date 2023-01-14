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
 \brief Class declaration for runtime-selectable logging system
 */

#pragma once

#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>

namespace sparten
{

//! Logger with runtime verbosity selection (Has performance implications, but can be changed at command line.)
//! This is a singleton with a twist. The new_log factory with arguments needs to be called first somewhere near the beginning of the program.
//! Every other instance of accessing the log, within local scope, should call new_log with no arguments
//! For example, if currently TO_SCREEN, new_log with arguments can be called again to destroy the current instance and create a new one (e.g., with TO_FILE instead) ... or vice versa
class Log
{
public:
  //! Specifies destination location of the logging output
  enum LogType
  {
    TO_SCREEN,
    TO_FILE
  };

  //! Verbosity level of logging output
  enum Verbosity
  {
	QUIET   = 0,	//!< Absolutely no output
    RELEASE = 1,	//!< Standard progress output
    VERBOSE = 2,	//!< Wordy progress output
    DEBUG_1 = 3,	//!< High-level debug info (e.g. driver, I/O, etc.)
    DEBUG_2 = 4,	//!< Coarse grain debug info (e.g. solver level)
    DEBUG_3 = 5,    //!< Fine grain debug info (e.g. row subproblem level)
  };

  Log(Verbosity level) : _level(level) {;}  //!< Constructor
  virtual ~Log() {;}                        //!< Destructor

  static Log &new_log(LogType type, std::string const &filename, Verbosity level);  //!< Factory method, singleton (This object retains ownership for the life of the program instance.)
  static Log &new_log();                                                            //!< Factory method, can only be called after first instance is created

  virtual void print(std::string const &message, Verbosity level) = 0;              //!< Prints message to selected destination
  virtual void finalize() = 0;                                                      //!< Finalizes output. This needs to be called before the destructor.

  virtual Verbosity get_verbosity() const { return _level; }                        //!< Returns the verbosity level

protected:
  Verbosity _level;   //!< Verbosity level selection
};

} // sparten
