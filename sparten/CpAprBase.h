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
 \brief Class declaration for Canonical Polyadic Alternating Poisson Regression (CP-APR) base class
 */

#pragma once

#include <sparten/Dimension.h>
#include <sparten/KruskalTensor.h> /// \todo Refactor to not need this to get access to typedefs
#include <sparten/SparseTensor.h> /// \todo Refactor to not need this to get access to typedefs
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

//! Canonical Polyadic Alternating Poisson Regression Base Class
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class CpAprBase
{
public:
  CpAprBase(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps);  //!< Constructor
  virtual ~CpAprBase();                                                                                                                                                       //!< Destructor (RAII)

  virtual void compute(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) = 0;  //!< Performs Canonical Polyadic Alternating Poisson Regression

  virtual KruskalValue get_tolerance() const;                       //!< Returns stop tolerance for convergence criteria
  virtual SubIdx get_max_outer_iter() const;                        //!< Returns maximum number of outer iterations for stop criteria
  virtual SubIdx get_max_inner_iter() const;                        //!< Returns maximum nubmer of inner iterations for stop criteria
  virtual KruskalValue get_offset() const;                          //!< Returns offset to fix "complementary slackness"
  virtual KruskalValue get_offset_tolerance() const;                //!< Returns tolerance for values that may be offset
  virtual SubIdx get_progress_interval() const;                     //!< Returns progress interval for output
  virtual KruskalValue get_eps() const;                             //!< Returns minimum value for protection against division-by-zero
  virtual bool get_isConverged() const;                             //!< Returns whether or not convergence has been reached
  virtual SubIdx get_nIter() const;                                 //!< Returns the number of outer iterations completed

  virtual void set_tolerance(KruskalValue tolerance);               //!< Sets stop tolerance for convergence criteria
  virtual void set_max_outer_iter(SubIdx maxOuterIter);             //!< Sets maximum number of outer iterations for stop criteria
  virtual void set_max_inner_iter(SubIdx maxInnerIter);             //!< Sets maximum nubmer of inner iterations for stop criteria
  virtual void set_offset(KruskalValue offsetValue);                //!< Sets offset to fix "complementary slackness"
  virtual void set_offset_tolerance(KruskalValue offsetTolerance);  //!< Sets tolerance for values that may be offset
  virtual void set_progress_interval(SubIdx progressInterval);      //!< Sets progress interval for output
  virtual void set_eps(KruskalValue eps);                           //!< Sets minimum value for protection against division-by-zero
  virtual void progress() const;                                    //!< Prints the progress information

  class Iterator {
  public:
    Iterator(SubIdx position) {;}                                   // STUB, do not include in coverage: LCOV_EXCL_LINE

  private:
    SubIdx _position;
  };

protected:
  bool _isConverged;                                                //!< Status flag indicating convergence
  bool _isOffset;                                                   //!< One or more entries has been _offset by _offsetTolerance
  bool _bufferAllocated;                                            //!< Keeps track of allocation of temp space on first call (performed only once)
  SubIdx _maxOuterIter;                                             //!< Maximum number of outer iterations for stop criteria
  SubIdx _maxInnerIter;                                             //!< Maximum number of inner iterations for stop criteria
  SubIdx _progressInterval;                                         //!< Specified how often CpApr will output progress information (i.e., number of iterations between output)
  SubIdx _nOuterIter;                                               //!< Number of outer iterations performed
  SubIdx _nInnerIter;                                               //!< Number of inner iterations performed, total
  SubIdx _nInnerIterDim;                                            //!< Number of inner iterations performed, per dim, per outer iteration
  SubIdx _nFlops;                                                   //!< Number of FLOPS performed, total
  KruskalValue _tolerance;                                          //!< Stop tolerance for convergence criteria
  KruskalValue _offsetTolerance;                                    //!< Values less than this tolerance may be offset /// \todo again, what is slackness?
  KruskalValue _offsetValue;                                        //!< Values to offset inaddmissible zeros (or nearly zeros)
  KruskalValue _eps;                                                //!< Minimum value for division to protect against division-by-zero
  KruskalValue _errorNorm;                                          //!< KKT error (Can be used for outer loop convergence determination. Although, it wasn't actually used in TTB_cpp.)
  KruskalValue _objective_func_value;                               //!< Objective Function Value
  KruskalValue _outerLoopTimerSeconds;                              //!< Outer loop timer
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::View<KruskalValue *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>> _eps_dev;
#else
  Kokkos::View<KruskalValue *, Kokkos::HostSpace> _eps_host;
#endif


  std::vector<KruskalValue> _kktErrorPerDim;                             //!< Stores kkt error per dimension so that it can be used with different norms later in convergence criteria
  NonZeroLocations<SubIdx> _nonzLocIdx;                             //!< Index of nonzero location
  NonZeroLocations<ElemIdx> _nonzLoc;                               //!< Location of nonzero
#if defined(KOKKOS_ENABLE_CUDA) 
  NonZeroLocationsHost<SubIdx> _nonzLocIdx_host;                             //!< Index of nonzero location
  NonZeroLocationsHost<ElemIdx> _nonzLoc_host;                               //!< Location of nonzero
#endif
  SubIdx _numNonzeroViolations;                                     //!< Number of zeros (or close-to-zeros)

  KruskalTensor<KruskalValue, SubIdx> *_phiBuffer;       //!< Double-buffer to allow parallel operation (e.g., task parallelism to simultaneously perform all compute_phi's)


  virtual bool is_non_negative(SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const;                                                   //!< Performs non-negativity check on a sparse tensor
  virtual bool is_non_negative(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const;                                                         //!< Performs non-negativity check on a Kruskal tensor
  virtual void offset(KruskalTensor<KruskalValue, SubIdx> &kruskalInput, SubIdx iDim);                                                                 //!< Performs offset of zero or close-to-zero values to make them larger
  virtual void check_requirements(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput);  //!< Performs check
  virtual void reorder(int32_t iDim, int32_t nRow, int32_t nNonz, SparseTensor<SparseValue, ElemIdx, SubIdx> const &inputTensor);                                                        //!< Reorders the data
  virtual void update_karush_kuhn_tucker_condition_error_norm();                                                                                       //!< Update KKT error norm calculation
};

} // sparten
