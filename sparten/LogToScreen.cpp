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
 \brief Class definition for runtime-selectable logging system (to screen)
 */

#include <sparten/LogToScreen.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <stdexcept>

namespace sparten
{

LogToScreen::LogToScreen(Verbosity level) :
        Log(level)
{
}

LogToScreen::~LogToScreen()
{
  this->finalize();
}

void LogToScreen::print(std::string const &message, Verbosity level)
{
  if (_level >= level && _level != QUIET)
  {
    std::cout << message << std::endl;
  }
}

void LogToScreen::finalize()
{
}

} // sparten
