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
 \brief Class definition for runtime-selectable logging system
 */

#include <sparten/Log.h>
#include <sparten/LogToFile.h>
#include <sparten/LogToScreen.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <stdexcept>

namespace sparten
{

static Log *LOG_GLOBAL = nullptr;

Log &Log::new_log(LogType type, std::string const &filename, Verbosity level)
{
  if (LOG_GLOBAL == nullptr)
  {
    switch (type)
    {
    case Log::TO_SCREEN:
      LOG_GLOBAL = new LogToScreen(level);
      break;
    case Log::TO_FILE:
      LOG_GLOBAL = new LogToFile(filename, level);
      break;
    default:
      throw std::runtime_error("Incorrect logging destination selected.");
    };
  }
  else
  {
    delete LOG_GLOBAL;
    LOG_GLOBAL = nullptr;
    new_log(type, filename, level);
  }

  return *LOG_GLOBAL;
}

Log &Log::new_log()
{
  if (LOG_GLOBAL == nullptr)
  {
    throw std::runtime_error("First instance of Log needs to be create first.");
  }

  return *LOG_GLOBAL;
}

} // sparten
