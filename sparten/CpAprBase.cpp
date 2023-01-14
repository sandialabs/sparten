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
 \brief Class definition for Canonical Polyadic Alternating Poisson Regression (CP-APR)
 */

#include <sparten/CpAprBase.h>
#include <sparten/Log.h>
#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace sparten
{

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::CpAprBase(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offsetValue, KruskalValue offsetTolerance,
    SubIdx progressInterval, KruskalValue eps) :
        _tolerance(tolerance),
        _maxOuterIter(maxOuterIter),
        _maxInnerIter(maxInnerIter),
        _offsetValue(offsetValue),
        _offsetTolerance(offsetTolerance),
        _progressInterval(progressInterval),
        _isConverged(false),
        _nOuterIter(0),
        _isOffset(false),
        _nonzLoc("_nonzLoc", 1),
        _nonzLocIdx("_nonzLocIdx", 1),
        _bufferAllocated(false),
        _errorNorm(0.0),
        _phiBuffer(nullptr)
{
    
    _eps = eps;
#if 0
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
    std::cout << "Initalize the solver" <<std::endl;
    typename Kokkos::View<KruskalValue *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>::HostMirror myEps("myeps",1);
    Kokkos::resize( _eps_dev, 1);
    myEps(0) = eps;
    Kokkos::deep_copy( _eps_dev, myEps );
#else
    Kokkos::resize( _eps_host, 1);
    _eps_host(0) = eps; 
#endif
#endif

}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::~CpAprBase()
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_tolerance() const
{
  return _tolerance;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_max_outer_iter() const
{
  return _maxOuterIter;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_max_inner_iter() const
{
  return _maxInnerIter;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_offset() const
{
  return _offsetValue;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_offset_tolerance() const
{
  return _offsetTolerance;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_progress_interval() const
{
  return _progressInterval;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_eps() const
{
  return _eps;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_isConverged() const
{
  return _isConverged;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_nIter() const
{
  return _nOuterIter;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_tolerance(KruskalValue tolerance)
{
  _tolerance = tolerance;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_max_outer_iter(SubIdx maxOuterIter)
{
  _maxOuterIter = maxOuterIter;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_max_inner_iter(SubIdx maxInnerIter)
{
  _maxInnerIter = maxInnerIter;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_offset(KruskalValue offsetValue)
{
  _offsetValue = offsetValue;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_offset_tolerance(KruskalValue offsetTolerance)
{
  _offsetTolerance = offsetTolerance;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_progress_interval(SubIdx progressInterval)
{
  _progressInterval = progressInterval;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_eps(KruskalValue eps)
{
  _eps = eps;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
ElemIdx CpAprBaseCaller_is_non_negative_sparse_tensor(
    ElemIdx sparse_nElement,
    const SparseData<SparseValue> sparseData)
{

  ElemIdx nNegative = 0;
  #if 0
  Kokkos::parallel_reduce (sparse_nElement, KOKKOS_LAMBDA (ElemIdx elemId, ElemIdx &lnNegative)
  {
    lnNegative += static_cast<ElemIdx>(sparseData(elemId) < 0); ///\todo: pointless comparison since sparseData is of type SparseValue, which is and unsigned int
  }, nNegative);
#endif
  return nNegative;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const
{
  const auto sparseData = sparseInput.get_data_view();

  ElemIdx nNegative = CpAprBaseCaller_is_non_negative_sparse_tensor<SparseValue, KruskalValue, ElemIdx, SubIdx>(sparseInput.get_nElement(), sparseData);

  return nNegative == 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBaseCaller_is_non_negative_kruskal_tensor(
    ElemIdx &nNegative,
    SubIdx dim,
    SubIdx kruskal_nRow,
    SubIdx kruskal_nColumn,
    const FactorMatrix<KruskalValue> factorMatrix)
{
  const auto teamSize = Kokkos::AUTO;
  Kokkos::parallel_reduce (Kokkos::TeamPolicy<>(kruskal_nRow, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type const &thread, ElemIdx &lnNegative)
  {
    const auto iMode = thread.league_rank();

    ElemIdx tnNegative = 0;
    Kokkos::parallel_reduce (Kokkos::TeamThreadRange(thread, kruskal_nColumn), [=] (SubIdx iComp, ElemIdx &llnNegative)
    {
      llnNegative += static_cast<ElemIdx>(factorMatrix(iMode, iComp) < 0);
    }, tnNegative);

      lnNegative += tnNegative;
  }, nNegative);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const
{
  ElemIdx nNegative = 0;

  for (auto iDim : kruskalInput.get_dim())
  {
    const auto factorMatrices = kruskalInput.get_factor_matrices();

    CpAprBaseCaller_is_non_negative_kruskal_tensor<SparseValue, KruskalValue, ElemIdx, SubIdx>(nNegative, iDim, kruskalInput.get_factor_matrix_nRow(iDim), kruskalInput.get_factor_matrix_nColumn(iDim), factorMatrices[iDim]);
  }

  return nNegative == 0;
}


template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBaseCaller_offset_parallelRegion1(
    ElemIdx &offsetCount,
    SubIdx kruskal_nRow,
    SubIdx kruskal_nColumn,
    const FactorMatrix<KruskalValue> tmpFactorMatrixRead,
    const FactorMatrix<KruskalValue> tmpPhi,
    KruskalValue offsetValue,
    KruskalValue offsetTolerance)
{
  const auto teamSize = Kokkos::AUTO;
  Kokkos::parallel_reduce (Kokkos::TeamPolicy<>(kruskal_nRow, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type thread, ElemIdx &lisOffset)
  {
    const auto iMode = thread.league_rank();
    ElemIdx tisOffset = 0;
    Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(thread, kruskal_nColumn), [=] (SubIdx iComp, ElemIdx &llisOffset)
    {
      llisOffset += ((tmpPhi(iMode, iComp) > 1.0) && (tmpFactorMatrixRead(iMode, iComp) < offsetTolerance));
    }, tisOffset);

    lisOffset += tisOffset;
  }, offsetCount);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBaseCaller_offset_parallelRegion2(
    ElemIdx &offsetCount,
    SubIdx kruskal_nRow,
    SubIdx kruskal_nColumn,
    FactorMatrix<KruskalValue> tmpFactorMatrix,
    const FactorMatrix<KruskalValue> tmpFactorMatrixRead,
    const FactorMatrix<KruskalValue> tmpPhi,
    KruskalValue offsetValue,
    KruskalValue offsetTolerance)
{
  const auto teamSize = Kokkos::AUTO;
  Kokkos::parallel_for (Kokkos::TeamPolicy<>(kruskal_nRow, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type thread)
  {
    const auto iMode = thread.league_rank();
    Kokkos::parallel_for (Kokkos::ThreadVectorRange(thread, kruskal_nColumn), [=] (SubIdx iComp)
    {
      if ((tmpPhi(iMode, iComp) > 1.0) &&  (tmpFactorMatrixRead(iMode, iComp) < offsetTolerance))
      {
        tmpFactorMatrix(iMode, iComp) += offsetValue;
      }
    });
  });

  // // Get the number of elements that were scooched
  // Kokkos::parallel_reduce("Sparten::offset::get_nScooched",
  //                         Kokkos::TeamPolicy<>(kruskal_nRow, teamSize), 
  //                         KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type thread, SubIdx &cout) {
                            
  //                           const auto iMode = thread.league_rank();
  //                           SubIdx cin = 0;
  //                           Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, kruskal_nColumn),
  //                           [&](SubIdx iComp, SubIdx &lc) {
  //                             if ((tmpPhi(iMode,iComp > 1.0) && (tmpFactorMatrixRead(iMode, iComp) < offsetTolerance))) {
  //                               lc += 1;
  //                             }
  //                           }, cin);
  //                           cout += cin;
  //                         }, offsetCount);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::offset(KruskalTensor<KruskalValue, SubIdx> &kruskalInput, SubIdx iDim)
{
  auto tmpFactorMatrix = kruskalInput.get_factor_matrix(iDim);
  const auto tmpFactorMatrixRead = kruskalInput.get_factor_matrix(iDim);
  const auto tmpPhi =  _phiBuffer->get_factor_matrix(iDim);
  _isOffset = false;

  const KruskalValue offsetValue = _offsetValue;
  const KruskalValue offsetTolerance = _offsetTolerance;

  ElemIdx offsetCount = 0;
  CpAprBaseCaller_offset_parallelRegion1<SparseValue, KruskalValue, ElemIdx, SubIdx>(offsetCount, kruskalInput.get_factor_matrix_nRow(iDim), kruskalInput.get_factor_matrix_nColumn(iDim), tmpFactorMatrixRead, tmpPhi, offsetValue, offsetTolerance);
  this->_nFlops += offsetCount;

  _isOffset = offsetCount != 0;

  CpAprBaseCaller_offset_parallelRegion2<SparseValue, KruskalValue, ElemIdx, SubIdx>(offsetCount, kruskalInput.get_factor_matrix_nRow(iDim), kruskalInput.get_factor_matrix_nColumn(iDim), tmpFactorMatrix, tmpFactorMatrixRead, tmpPhi, offsetValue, offsetTolerance);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::check_requirements(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput,
     SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput)
{
  // Perform consistency checks
  if (kruskalOutput.get_nDim() != sparseInput.get_nDim())
  {
    throw std::runtime_error("CpAprBase.compute(...) - Inconsistent Kruskal (output) and Sparse (input) tensor dimensions are inconsistent.");
  }

  // Perform non-negativity check on input tensor
  if (!is_non_negative(sparseInput))
  {
    throw std::runtime_error("Sparse (input) fails non-negativity test");
  }

  // Perform non-negativity check on initial guess in output tensor
  if (!is_non_negative(kruskalOutput))
  {
    throw std::runtime_error("Kruskal (output) fails non-negativity test");
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::reorder(int32_t iDim, int32_t nRow, int32_t nNonz, 
     SparseTensor<SparseValue, ElemIdx, SubIdx>  const &inputTensor )
//SparseIndices<SubIdx> spidx )
{
  // Needs to be done at the HostSpace
  ElemIdx nz_count = 0;
  typename SparseIndices<SubIdx>::HostMirror  spidx = inputTensor.get_indices_host_view(); 
  std::vector<std::vector<ElemIdx>> mystack(nRow);

  for (ElemIdx i = 0; i < nNonz; ++i)
  {
    SubIdx iRow =  spidx(iDim,i);
    mystack[iRow].push_back(i);
  }
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  const auto nZloc = _nonzLoc_host(iDim);
  const auto nZlocIdx = _nonzLocIdx_host(iDim);
#else
  const auto nZloc = _nonzLoc(iDim);
  const auto nZlocIdx = _nonzLocIdx(iDim);
#endif
  nZlocIdx(0) = 0;
  for (SubIdx iRow = 0; iRow < nRow; ++iRow)
  {
     nZlocIdx(iRow + 1) = nZlocIdx(iRow) + mystack[iRow].size();
     for(SubIdx i = 0; i < mystack[iRow].size(); ++i)
     {
       nZloc(nz_count) = mystack[iRow][i];
       ++nz_count;
     }
  }

#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy(_nonzLoc(iDim), nZloc);
  Kokkos::deep_copy(_nonzLocIdx(iDim), nZlocIdx);
#endif

}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::update_karush_kuhn_tucker_condition_error_norm()
{
  KruskalValue tmp = 0;

 // for (SubIdx iDim = 0; iDim < _kktErrorPerDim.dimension(0); ++iDim)
  for (SubIdx iDim = 0; iDim < _kktErrorPerDim.size(); ++iDim)
  {
  //  tmp = (KruskalValue)std::max(static_cast<double>(tmp), std::fabs(static_cast<double>(_kktErrorPerDim(iDim))));
    tmp = (KruskalValue)std::max(static_cast<double>(tmp), std::fabs(static_cast<double>(_kktErrorPerDim[iDim])));
  }
  _errorNorm = tmp;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::progress() const
{
  Log &log = Log::new_log();

  std::stringstream message;
  message << "\tNumber of iterations completed = " << _nOuterIter << std::endl;
  message << "\tSolution Converged = " << (_isConverged ? "true" : "false") << std::endl;
  message << "\tKKT Violation " << std::scientific << _errorNorm << std::endl;
  message << "\tNumber of nonzero violations " << _numNonzeroViolations << std::endl;
  message << "\tTolerance = " << _tolerance << std::endl;
  message << "\tMaximum number of outer iterations = " << _maxOuterIter << std::endl;
  message << "\tMaximum number of inner iterations = " << _maxInnerIter << std::endl;
  message << "\tOffset = " << _offsetValue << std::endl;
  message << "\tOffset tolerance= " << _offsetTolerance << std::endl;
  message << "\tProgress Interval = " << _progressInterval << std::endl;
  message << "\teps = " << _eps << std::endl;
  log.print(message.str(), Log::DEBUG_1);
}

// Explicit instantiation
template class CpAprBase<type::SubIdx, float, type::SubIdx, type::SubIdx>;
template class CpAprBase<type::SubIdx, float, type::ElemIdx, type::SubIdx>;
template class CpAprBase<type::SubIdx, float, type::ElemIdx, type::ElemIdx>;

template class CpAprBase<type::ElemIdx, float, type::SubIdx, type::SubIdx>;
template class CpAprBase<type::ElemIdx, float, type::ElemIdx, type::SubIdx>;
template class CpAprBase<type::ElemIdx, float, type::ElemIdx, type::ElemIdx>;

template class CpAprBase<type::SubIdx, double, type::SubIdx, type::SubIdx>;
template class CpAprBase<type::SubIdx, double, type::ElemIdx, type::SubIdx>;
template class CpAprBase<type::SubIdx, double, type::ElemIdx, type::ElemIdx>;

template class CpAprBase<type::ElemIdx, double, type::SubIdx, type::SubIdx>;
template class CpAprBase<type::ElemIdx, double, type::ElemIdx, type::SubIdx>;
template class CpAprBase<type::ElemIdx, double, type::ElemIdx, type::ElemIdx>;

} // sparten
