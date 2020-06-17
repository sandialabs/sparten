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
 \brief
 */

#include <unit_test/LogUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void LogUnitTest::SetUp()
{
}

void LogUnitTest::TearDown()
{
}

TEST_F(LogUnitTest, to_screen)
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded.txt", sparten::Log::DEBUG_0);
  log.print("LogUnitTest to screen", sparten::Log::RELEASE);
}

TEST_F(LogUnitTest, to_file)
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_FILE, "test/data/output_LogTest_to_file.txt", sparten::Log::DEBUG_0);
  log.print("LogUnitTest to file", sparten::Log::RELEASE);
  
}

TEST_F(LogUnitTest, get_verbosity)
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_FILE, "test/data/output_LogTest_to_file.txt", sparten::Log::RELEASE);
  ASSERT_EQ(log.get_verbosity(), sparten::Log::RELEASE);
  
}

TEST_F(LogUnitTest, fail_on_open_file)
{
  ASSERT_ANY_THROW(sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_FILE, "/etc/shouldntbeabletoopen", sparten::Log::DEBUG_0));
}

TEST_F(LogUnitTest, fail_on_log_type)
{
  ASSERT_ANY_THROW(sparten::Log &log = sparten::Log::new_log((sparten::Log::LogType)(-1), "noFileNeeded.txt", sparten::Log::RELEASE));
}

TEST_F(LogUnitTest, fail_on_new_log)
{
  ASSERT_ANY_THROW(sparten::Log &log = sparten::Log::new_log());
}

} // unitTest
