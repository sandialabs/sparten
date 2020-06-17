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

#pragma once

#include <iostream>
#include <string>
#include <Sparten_Configure.h>
#include <sparten/Log.h>

class CommandLineOptions
{
public:
  CommandLineOptions() :
    inputOptionsFile("inputOptions_problemgen.txt"),
    sptensorOutputFile("sptensor.tns"),
    ktensorOutputFile("ktensor.tns"),
    indexStyle("1"),
    randomSeed("12345"),
    numComponents(),
    maxNumNonzeros(10000000),
    ktensorThreshhold(0.2),
    ktensorNoise(0.05),
    numDimensions(),
    sizeDim(),
    cDims()
    {;}

  void print_help(int argvc, char **argv)
  {
    std::cout << "SparTen v" << SPARTEN_VERSION << std::endl;
    std::cout << "CXX Compiler: " << SPARTEN_CXX_COMPILER << std::endl;
    std::cout << "C Compiler: " << SPARTEN_CXX_COMPILER << std::endl;
    std::cout << std::endl;

    std::cout << "SparTen_tensor_gen usage: " << argv[0] << " [arg1 value1 arg2 value2 ...]" <<
      std::endl << std::endl <<
      "REQUIRED ARGUMENTS:" << std::endl <<
      "     --num-components        INT          Number of components in ktensor " << std::endl <<
      "     --max-num-nonzeros      INT          Maximum number of nonzero elements to generate in the sparse tensor " << std::endl <<
      "     --dim-sizes             STRING       Comma-delited string (with no spaces) containing the sizes of each dimension " << std::endl <<
      std::endl <<
      "OPTIONAL ARGUMENTS:" << std::endl <<
      "     --sptensor-output-file  FILENAME     Name of output file containing sparse tensor data [Default: sptensor.tns]" << std::endl <<
      "     --ktensor-output-file   FILENAME     Name of output file containing kruskal tensor data [Default: ktensor.tns]" << std::endl <<
      "     --ktensor-threshhold    FLOAT        Threshhold used in generating ktensor [Default: 0.2] " << std::endl <<
      "     --ktensor-noise         FLOAT        Amount of noise to add to ktensor [Default: 0.05] " << std::endl <<
      "     --index-style           INT          Index starting value for tensor data {1,0} [Default: 1]" << std::endl <<
      "     --random-seed           STRING       Value of seed to use in random generator [Default: 12345] " << std::endl <<
      std::endl;
  }

  void print_options() const
  {
    sparten::Log &log = sparten::Log::new_log();
  }

  //REQUIRED
  sparten::type::SubIdx numComponents;
  sparten::type::SubIdx maxNumNonzeros;
  sparten::type::SubIdx numDimensions;
  std::vector<std::string> sizeDim;
  std::vector<sparten::type::SubIdx> cDims;

  // OPTIONAL
  std::string inputOptionsFile;
  std::string sptensorOutputFile;
  std::string ktensorOutputFile;
  std::string indexStyle;
  std::string randomSeed;
  double ktensorThreshhold;
  double ktensorNoise;
};
