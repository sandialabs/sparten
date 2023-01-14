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
 \brief Class declaration for a Mersenne Twister random number generator.
 */

#pragma once

#include <cstdint>

namespace sparten
{

//! Mersenne Twister random number generator
class RandGenMersenneTwister
{
public:
  RandGenMersenneTwister(const uint32_t nnSeed);                 	      //!< Constructor
  ~RandGenMersenneTwister(void);                                      	//!< Destructor

  int32_t reset(int32_t nnSeed);                                        //!< Reset the sequence
  uint32_t generate_rand_int32();                            	          //!< Return a uniform random number on the interval [0,0xffffffff]
  double generate_rand_double();                                      	//!< Return a uniform random number on the interval [0,1)
  double generate_rand_double_inclusive();                            	//!< Return a uniform random number on the interval [0,1]
  double generate_rand_double_matlab();                               	//!< Return a uniform random number on the interval [0,
private:
  RandGenMersenneTwister(RandGenMersenneTwister const &);               //!< Disable copy constructor
  RandGenMersenneTwister &operator=(RandGenMersenneTwister const &);    //!< Disable assignment operator
};

} // end namespace sparten
