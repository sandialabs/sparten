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
// Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
// ************************************************************************
//@HEADER

#pragma once

#include <sparten/spartenUtils.h>
#include <gtest/gtest.h>
#include <sparten/Log.h>
#include <sparten/CommandLineOptions.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest {

class CommandLineOptionsUnitTest : public ::testing::Test
{
public:
    virtual void SetUp();
    virtual void TearDown();

protected:
    sparten::CommandLineOptions _clo;
    std::vector<std::string> _default_args;
    std::vector<std::string> _empty_args;
    std::vector<std::string> _test_args;
    uint64_t val_uint64_t;
    uint64_t min_uint64_t;
    uint64_t max_uint64_t;
    double val_double;
    double min_double;
    double max_double;
};

} // end namespace unitTest