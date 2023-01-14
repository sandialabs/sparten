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

#include <sparten/CpAprBase.h>
#include <sparten/Dimension.h>
#include <sparten/KruskalTensor.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace sparten
{

// Forward Declarations
template<class ValueType, class ElemIdx, class SubIdx>
class SparseTensor;

template<class ValueType, class SubIdx>
class KruskalTensor;

//! Canonical Polyadic Alternating Poisson Regression Class
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class CpAprMultiplicativeUpdate : public CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>
{
public:
  CpAprMultiplicativeUpdate(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps);  //!< Constructor
  virtual ~CpAprMultiplicativeUpdate();                                                                                                                                                       //!< Destructor (RAII)

  virtual void compute(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) override;  //!< Performs Canonical Polyadic Alternating Poisson Regression
  //virtual void compute(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) override;        //!< Performs Canonical Polyadic Alternating Poisson Regression

  virtual void progress() const override;                           //!< Prints the progress information

protected:
  FactorMatrix<KruskalValue> _pi;                                   //!< Buffer for Pi (Nnz * nComponents)
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::View<KruskalValue *,  Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>> _dKktVal;
#else
  Kokkos::View<KruskalValue *,  Kokkos::HostSpace> _dKktVal;
#endif

  bool does_violate_karush_kuhn_tucker_condition(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput );                                                    //!< Performs check to ensure Karush-Kuhn-Tucker (KKT) condition is met for each mode
  bool does_violate_karush_kuhn_tucker_condition(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput, SubIdx const iDim );                                 //!< Performs check to ensure Karush-Kuhn-Tucker (KKT) condition is met for each mode
  bool multiplicative_update(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDim);  //!< Performs the multiplicative update
  void compute_pi(KruskalTensor<KruskalValue, SubIdx> const &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDimSkip);   //!< Performs the computation of Pi (See introduction of the SparTen User Guide for more details)
  void compute_phi(KruskalTensor<KruskalValue, SubIdx> const &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDim);      //!< Performs the computation of Phi (See introduction of the SparTen User Guide for more details)

 KruskalValue compute_obj( KruskalTensor<KruskalValue, SubIdx> const &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDim);
};

} // sparten
