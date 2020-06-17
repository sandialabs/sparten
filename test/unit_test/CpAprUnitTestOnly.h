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
 \brief Class declaration for Canonical Polyadic Alternating Poisson Regression (CP-APR)
 */

#pragma once

#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/CpAprDampedNewton.h>
#include <sparten/CpAprQuasiNewton.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace unitTest
{

// Forward Declarations
template<class SparseValue, class ElemIdx, class SubIdx>
class SparseTensor;

template<class KruskalValue, class SubIdx>
class KruskalTensor;

//! Canonical Polyadic Alternating Poisson Regression Class, Multiplicative Update (Testing only, exposes protected/private members)
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class CpAprMultiplicativeUpdateUnitTestOnly : public sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>
{
public:
    CpAprMultiplicativeUpdateUnitTestOnly(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps);  //!< Constructor
  virtual ~CpAprMultiplicativeUpdateUnitTestOnly();                                                                                                                                                 //!< Destructor (RAII)

  virtual bool test_only_does_violate_karush_kuhn_tucker_condition(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput);     //!< Performs check to ensure Karush-Kuhn-Tucker (KKT) condition is met for each mode
  virtual bool test_only_is_non_negative(sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const;                   //!< Performs non-negativity check on a sparse tensor
  virtual bool test_only_is_non_negative(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const;                         //!< Performs non-negativity check on a Kruskal tensor
  virtual bool test_only_multiplicative_update(sparten::KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDim);        //!< Performs the multiplicative update
  virtual void test_only_progress() const;                                                                                                //!< Prints the progress information
};

//! Canonical Polyadic Alternating Poisson Regression Class, Damped Newton (Testing only, exposes protected/private members)
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class CpAprDampedNewtonUnitTestOnly : public sparten::CpAprDampedNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>
{
public:
    CpAprDampedNewtonUnitTestOnly(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps);  //!< Constructor
  virtual ~CpAprDampedNewtonUnitTestOnly();                                                                                                                                                 //!< Destructor (RAII)

  virtual bool test_only_is_non_negative(sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const;                   //!< Performs non-negativity check on a sparse tensor
  virtual bool test_only_is_non_negative(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const;                         //!< Performs non-negativity check on a Kruskal tensor
  virtual void test_only_progress() const;                                                                                                //!< Prints the progress information
};

//! Canonical Polyadic Alternating Poisson Regression Class, Quasi Newton (Testing only, exposes protected/private members)
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class CpAprQuasiNewtonUnitTestOnly : public sparten::CpAprQuasiNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>
{
public:
    CpAprQuasiNewtonUnitTestOnly(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps);  //!< Constructor
  virtual ~CpAprQuasiNewtonUnitTestOnly();                                                                                                                                                 //!< Destructor (RAII)

  virtual bool test_only_is_non_negative(sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const;                   //!< Performs non-negativity check on a sparse tensor
  virtual bool test_only_is_non_negative(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const;                         //!< Performs non-negativity check on a Kruskal tensor
  virtual void test_only_progress() const;                                                                                                //!< Prints the progress information
};

} // unitTest
