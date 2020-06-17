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
 \brief Class declaration for runtime-selectable logging system (to screen)
 */

#pragma once

#include <sparten/Log.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace sparten
{

//! Logger with runtime verbosity selection
class LogToScreen : public Log
{
public:
  LogToScreen(Verbosity level);     //!< Constructor
  virtual ~LogToScreen();           //!< Destructor

  virtual void print(std::string const &message, Verbosity level) override;    //!< Prints message to screen
  virtual void finalize() override;                                            //!< Does nothing when printing to screen
};

} // sparten
