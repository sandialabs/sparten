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

#pragma once

#include <sparten/spartenUtils.h>
#include <gtest/gtest.h>
#include <sparten/AsciiIO.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>

//! Namespace for testing
namespace unitTest
{

//! AsciiIOHeader Test Fixture
class AsciiIOUnitTest : public ::testing::Test
{
public:
  virtual void SetUp();       //!<
  virtual void TearDown();    //!<

protected:
  std::string _sptensorFilename;                          //!<
  sparten::type::SubIdx _nDim;                    //!<
  std::vector<sparten::type::SubIdx> _nIndices;   //!<
  sparten::type::SubIdx _maxIndex;                //!<
  sparten::type::ElemIdx _nElement;               //!<
  sparten::type::SubIdx _precision;               //!<

  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *_sptensorHeader;
};

} // unitTest
