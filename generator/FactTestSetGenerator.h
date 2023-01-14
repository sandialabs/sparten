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
 \brief Class declaration for a test set generator.
 */

#pragma once

#include <generator/RandGenMersenneTwister.h>
#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <iostream>
#include <utility>
#include <vector>

namespace sparten
{

template<class SubIdx>
bool compare_int_arrays(std::vector<SubIdx> const &parg1, std::vector<SubIdx> const &parg2);

template<class KruskalValue, class SubIdx>
using SortablePair = std::pair<KruskalValue,SubIdx>;

template<class KruskalValue, class SubIdx>
bool sortablePair_compare(SortablePair<KruskalValue, SubIdx> const &a, SortablePair<KruskalValue, SubIdx> const &b);

//! Factor matrice test set generator
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class FactTestSetGenerator
{
public:
  FactTestSetGenerator();                                                                                                                                                 //!< Constructor
  FactTestSetGenerator(SubIdx nNumComponents, ElemIdx nMaxNonz, uint64_t nSeed, KruskalValue dFracBoosted, KruskalValue dMaxValue, KruskalValue dSmallValue);   //!< Constructor
  ~FactTestSetGenerator();                                                                                                                                                //!< Destructor

  bool set_FactoredTensorDims(std::vector<SubIdx> v);       //!< Set dimensions of the factored tensor
  int32_t set_NumComponents(SubIdx nNumComponents);         //!< Set number of components in factorization
  int32_t set_MaxNnz(ElemIdx nMaxNonz);                     //!< Set maximum number of nonzeros
  int32_t set_RandomSeed(uint64_t nSeed);                   //!< Set the value of the seed for the random generator
  int32_t set_FracBoosted(KruskalValue dFracBoosted);       //!< Set the
  int32_t set_MaxValue(KruskalValue dMaxValue);             //!< Set the maximum value to be generated
  int32_t set_SmallValue(KruskalValue dSmallValue);         //!< Set the minimum value to be generated

  std::vector<SubIdx> get_FactoredTensorDims();             //!< Get the dimensions of the factored tensor
  SubIdx get_NumComponents();                               //!< Get the number of components in the factorization
  ElemIdx get_MaxNnz();                                     //!< Get the maximum number of nonzeros
  uint64_t get_RandomSeed();                                //!< Get the value of the seed used for the random generator
  KruskalValue get_FracBoosted();                           //!< Get the
  KruskalValue get_MaxValue();                              //!< Get the maximum value to be generated
  KruskalValue get_SmallValue();                            //!< Get the minimum value to be generated

  int32_t set_Parameters(SubIdx nNumComponents, ElemIdx nMaxNonz, uint64_t nSeed, KruskalValue dFracBoosted, KruskalValue dMaxValue, KruskalValue dSmallValue);	        //!< Set the
  int32_t get_Parameters(SubIdx &nNumComponents, ElemIdx &nMaxNonz, uint64_t &nSeed, KruskalValue &dFracBoosted, KruskalValue &dMaxValue, KruskalValue &dSmallValue);	  //!< Get the

  KruskalTensor<KruskalValue, SubIdx> *genNoisedNonNegativeKruscalTensor(KruskalValue thres, KruskalValue noise);         //!< Generate a random KruskalTensor
  SparseTensor<SparseValue, ElemIdx, SubIdx> *genSparseTensor(KruskalTensor<KruskalValue, SubIdx>  *cExpectedFactor);     //!< Generate sparse tensor data from a random factored KruskalTensor
  
private:
  SubIdx nKEY_LENGTH = 1;                                                           //!<
  std::vector<SubIdx> _cDims;                                                       //!< Sizes of each dimension/mode
  SubIdx _nNumComponents;                                                           //!< Number of components in the factorization
  ElemIdx _nMaxNnz;                                                                 //!< Maximum number of nonzeros
  uint64_t _nSeed;                                                                  //!< Seed value used in random generation
  KruskalValue _dFracBoosted;                                                       //!<
  bool _bBoosted;                                                                   //!<

  RandGenMersenneTwister _cRNG;                                                     //!< Random number generator

  KruskalValue _dMaxValue;                                                          /// \todo This member may not be necessary
  KruskalValue _dSmallValue;                                                        /// \todo This member may not be necessary

  SparseTensor<SparseValue, ElemIdx, SubIdx> *_cSpTensor;                           //!<

  bool drawNonzeroElements(KruskalTensor<KruskalValue, SubIdx> *cExpectedFactors);  //!<

  FactTestSetGenerator(FactTestSetGenerator const &);                               //!< Blocking use of copy constructor
  FactTestSetGenerator &operator=(FactTestSetGenerator const &);                    //!< Blocking use of assignment operator
};

} // end namespace sparten

