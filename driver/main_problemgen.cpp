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

#include <CommandLineOptions_problemgen.h>
// Input options file not currently working
//#include <SimpleInput_problemgen.h> 
#include <Sparten_Configure.h>
#include <sparten/AsciiIO.h>
#include <sparten/KruskalTensor.h>
#include <sparten/Log.h>
#include <sparten/SparseTensor.h>
#include <generator/FactTestSetGenerator.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <ctime>
#include <iostream>
#include <istream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>


//! Convenience function to get std::string with args
std::string get_argument(int argc, char const **argv, int i)
{
  std::string tmp(argv[i]);
  return tmp;
}

//! Convenience function to get std::string with args
std::string get_argument(int argc, char **argv, std::string const &arg)
{
  bool found = false;
  std::string value;

  CommandLineOptions clo;

  if (argc % 2 == 0)
  {
    clo.print_help(argc, argv);
    throw std::runtime_error("Invalid number of command line options.");
  }

  for (int i = 1; i < argc; i += 2)
  {
    if (arg.compare(argv[i]) == 0)
    {
      found = true;
      value = argv[i + 1];
    }
  }

  if (found)
  {
    return value;
  } else
  {
    clo.print_help(argc, argv);
    throw std::runtime_error(arg + " not found.");
  }
}

//! Convenience function to get std::string with optional args
std::string get_argument_optional(int argc, char **argv, std::string const &arg)
{
  bool found = false;
  std::string value;
  CommandLineOptions clo;

  if (argc % 2 == 0)
  {
    clo.print_help(argc, argv);
    throw std::runtime_error("Invalid number of command line options.");
  }

  for (int i = 1; i < argc; i += 2)
  {
    if (arg.compare(argv[i]) == 0)
    {
      found = true;
      value = argv[i + 1];
    }
  }

  if (found)
  {
    return value;
  }
  else
  {
    return "";
  }
}

//! Convenience function to get command line options
CommandLineOptions  get_command_line_options(int argc, char **argv)
{
  CommandLineOptions clo;
  std::string::size_type sz;
   
  if (argc < 4)
  {
    clo.print_help(argc, argv);
    throw std::runtime_error("No arguments provided");
  }

  // REQUIRED
  std::string numComponentsStr = get_argument(argc, argv, "--num-components");
  std::string maxNumNonzerosStr = get_argument(argc, argv, "--max-num-nonzeros");
  std::string dimSizesStr = get_argument(argc, argv, "--dim-sizes");

  std::istringstream numComponentsIss(numComponentsStr); numComponentsIss >> clo.numComponents;
  std::istringstream maxNumNonzerosIss(maxNumNonzerosStr); maxNumNonzerosIss >> clo.maxNumNonzeros;

  std::istringstream dimSizesIss(dimSizesStr);
  //std::vector<std::string> dimSizes;
  std::string token;
  while (std::getline(dimSizesIss, token, ','))
  {
    clo.sizeDim.push_back(token);
  }
  clo.numDimensions = clo.sizeDim.size();

  // Variable number of dimensions
  clo.cDims.resize(clo.numDimensions);
  for( int i = 0; i < clo.numDimensions ; ++i )
  {
    clo.cDims[i] = std::stoi(clo.sizeDim[i] ,&sz);
  }

  // OPTIONAL
  // Input options file not currently working
  //std::string inputOptionsFile = get_argument_optional(argc, argv, "--input-options-file");
  std::string sptensorOutputFile = get_argument_optional(argc, argv, "--sptensor-output-file");
  std::string ktensorOutputFile = get_argument_optional(argc, argv, "--ktensor-output-file");
  std::string indexStyle = get_argument_optional(argc, argv, "--index-style");
  std::string randomSeed = get_argument_optional(argc, argv, "--random-seed");

  // Input options file not currently working
  //clo.inputOptionsFile = inputOptionsFile.empty() ? clo.inputOptionsFile : inputOptionsFile;
  clo.sptensorOutputFile = sptensorOutputFile.empty() ? clo.sptensorOutputFile : sptensorOutputFile;
  clo.ktensorOutputFile = ktensorOutputFile.empty() ? clo.ktensorOutputFile : ktensorOutputFile;
  clo.indexStyle = indexStyle.empty() ? clo.indexStyle : indexStyle;
  clo.randomSeed = randomSeed.empty() ? clo.randomSeed : randomSeed;

  std::string ktensorThreshholdStr = get_argument_optional(argc, argv, "--ktensor-threshhold");
  std::string ktensorNoiseStr = get_argument_optional(argc, argv, "--ktensor-noise");

  std::istringstream ktensorThreshholdIss(ktensorThreshholdStr); ktensorThreshholdIss >> clo.ktensorThreshhold;
  std::istringstream ktensorNoiseIss(ktensorNoiseStr); ktensorNoiseIss >> clo.ktensorNoise;

  // Input options file not currently working
  /*
  if (!inputOptionsFile.empty())
  {
    SimpleInput input(clo.inputOptionsFile, SimpleInput::READ);
    InputOptions inputOptions = input.read();

    clo.numComponents = !numComponentsStr.empty() ? clo.numComponents : inputOptions.numComponents;
    clo.maxNumNonzeros = !maxNumNonzerosStr.empty() ? clo.maxNumNonzeros : inputOptions.maxNumNonzeros;
    clo.numDimensions = !numDimensionsStr.empty() ? clo.numDimensions : inputOptions.numDimensions;
    clo.ktensorThreshhold = !ktensorThreshholdStr.empty() ? clo.ktensorThreshhold : inputOptions.ktensorThreshhold;
    clo.ktensorNoise = !ktensorNoiseStr.empty() ? clo.ktensorNoise : inputOptions.ktensorNoise;

    clo.sptensorOutputFile = inputOptions.sptensorOutputFile.empty() ? clo.sptensorOutputFile : inputOptions.sptensorOutputFile;
    clo.ktensorOutputFile = inputOptions.ktensorOutputFile.empty() ? clo.ktensorOutputFile : inputOptions.ktensorOutputFile;
    clo.randomSeed = inputOptions.randomSeed.empty() ? clo.randomSeed : inputOptions.randomSeed;
    clo.indexStyle = inputOptions.indexStyle.empty() ? clo.indexStyle : inputOptions.indexStyle;
  }
  */

  return clo;
}

int main(int argc, char **argv)
{
  //// Initialization ////
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  try
  {
    //// Main program ////
    // Get command line options
    CommandLineOptions clo = get_command_line_options(argc, argv);

    // Setup logging
    sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN,  "./" , sparten::Log::RELEASE);
 
    log.print("Sparten Logfile -- Version " + std::string(SPARTEN_VERSION) + "\n", sparten::Log::RELEASE);
 
    clo.print_options();

    typedef sparten::AsciiIO<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> SpartenIO;
    sparten::KruskalTensor<double, sparten::type::SubIdx> *testFactors = NULL;
    // TODO: determine type for indices (could be above 4B!)
    sparten::SparseTensor<sparten::type::SubIdx, sparten::type::SubIdx, sparten::type::SubIdx> *spTensor = NULL;

    sparten::FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> testGen;
    testGen.set_FactoredTensorDims(clo.cDims);
    
    testGen.set_NumComponents(clo.numComponents);
    testGen.set_MaxNnz(clo.maxNumNonzeros);
    testGen.set_RandomSeed(std::stoi(clo.randomSeed));

    // Generate Kruskal and Sparse Tensors
    testFactors = testGen.genNoisedNonNegativeKruscalTensor(clo.ktensorThreshhold, clo.ktensorNoise);
    spTensor = testGen.genSparseTensor(testFactors);

    std::cout << "Number of dimensions: " << clo.numDimensions << std::endl;
    sparten::type::ElemIdx maxNnz = 1;
    for( int i = 0; i < clo.numDimensions ; ++i ) 
    {
      std::cout << "\tDimension " << i+1 << ": " << clo.cDims[i] << std::endl;
      maxNnz *= clo.cDims[i];
    }
    std::cout << "Number of components: " << clo.numComponents << std::endl;
    std::cout << "Number of nonzeros in sparse tensor: " << spTensor->get_nElement()  << std::endl;
    std::cout << "Density of sparse tensor: " << 1.0*spTensor->get_nElement()/maxNnz << std::endl;

    // File IO
    SpartenIO myIO1(clo.ktensorOutputFile, SpartenIO::OVERWRITE, SpartenIO::SCIENTIFIC, 16, (SpartenIO::IndexStyle) std::stoi(clo.indexStyle));
    SpartenIO myIO2(clo.sptensorOutputFile, SpartenIO::OVERWRITE, SpartenIO::SCIENTIFIC, 16, (SpartenIO::IndexStyle) std::stoi(clo.indexStyle));

    myIO1.write(*testFactors);
    myIO2.write(*spTensor);
    myIO1.close_file();
    myIO2.close_file();


    delete(testFactors);
    delete(spTensor);
    log.finalize();
  } catch (std::exception &e)
  {
    std::cout << e.what() << std::endl;
  }

  //// Finalization ////
  Kokkos::finalize();

  return 0;
}
