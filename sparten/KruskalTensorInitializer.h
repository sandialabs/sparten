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
 \brief Class declaration for Kruskal tensor initialization
 */

#pragma once

#include <sparten/Dimension.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>

namespace sparten
{

template<class KruskalValue, class SubIdx>
class KruskalTensorInitializer;

//! Kruskal Tensor Initializer Class
template<class KruskalValue, class SubIdx>
class KruskalTensorInitializer
{
public:
  //!
  enum Method
  {
    UNITY,    //!< Populates with all ones
    RANDOM    //!< Populates with the random generator
  };

  KruskalTensorInitializer();                                                             //!< Constructor
  virtual ~KruskalTensorInitializer();                                                    //!< Destructor (RAII)

  void initialize(KruskalTensor<KruskalValue, SubIdx> &kruskalTensor, Method method);     //!< Populates weights of kruskal tensor
  void set_random_seed(int32_t iSeed);                                                    //!< Set the value of the seed for random generation
  int32_t get_random_seed();                                                              //!< Get the value of the seed for random generation

private:
  int32_t _iSeed;                                                                         //!< Private storage for seed value
};

} // sparten
