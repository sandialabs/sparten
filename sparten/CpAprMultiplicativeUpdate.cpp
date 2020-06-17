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

#include <sparten/CpAprMultiplicativeUpdate.h>
#include <sparten/Log.h>
#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace sparten
{

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::CpAprMultiplicativeUpdate(
    KruskalValue tolerance,
    SubIdx maxOuterIter,
    SubIdx maxInnerIter,
    KruskalValue offsetValue,
    KruskalValue offsetTolerance,
    SubIdx progressInterval,
    KruskalValue eps
    ) : CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>(
        tolerance,
        maxOuterIter,
        maxInnerIter,
        offsetValue,
        offsetTolerance,
        progressInterval,
        eps
        ),
          _pi("Pi", 1, 1)
{
  Kokkos::resize(_dKktVal,1);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::~CpAprMultiplicativeUpdate()
{
  if (this->_phiBuffer != nullptr)
  {
    delete this->_phiBuffer;
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::compute(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput)
{
  Log &log = Log::new_log();
  log.print("Entering CpAprMultiplicativeUpdate", Log::DEBUG_3);

  Kokkos::Timer cpAprTimer;

  // Only does this only once on the first compute call (essentially the tail end of program startup)
  // This way the user doesn't have to know anything about temporary work space allocation
  if (!this->_bufferAllocated)
  {
    log.print("CpAprMultiplicativeUpdate: _buffAllocated = false. Allocating now.", Log::DEBUG_3);
    this->_phiBuffer = new KruskalTensor<KruskalValue, SubIdx>(kruskalOutput);
    this->_kktErrorPerDim.resize( sparseInput.get_nDim());
    //Kokkos::resize(this->_kktErrorPerDim, sparseInput.get_nDim());

    Kokkos::resize(this->_nonzLoc, sparseInput.get_nDim());
    Kokkos::resize(this->_nonzLocIdx, sparseInput.get_nDim());
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::resize(this->_nonzLoc_host, sparseInput.get_nDim());
    Kokkos::resize(this->_nonzLocIdx_host, sparseInput.get_nDim());
#endif
    this->_isConverged = false;
    this->_bufferAllocated = true;
  }

  Kokkos::Timer checkRequirementsTimer;
  this->check_requirements(kruskalOutput, sparseInput);
  log.print("CpAprMultiplicativeUpdate::check_requirements: " + std::to_string(checkRequirementsTimer.seconds()) + " s", Log::RELEASE);

  Kokkos::resize(_pi, sparseInput.get_nElement(), kruskalOutput.get_nComponent());
#if !defined(KOKKOS_ENABLE_CUDA)
  Kokkos::resize(this->_nonzLoc, sparseInput.get_nDim()); // Verify this can be moved up to only happen once
  Kokkos::resize(this->_nonzLocIdx, sparseInput.get_nDim());
#else
  // Current version does not let GPU to choose non-atomic code
  #if 0
  Kokkos::resize(this->_nonzLoc_host, sparseInput.get_nDim()); // Verify this can be moved up to only happen once
  Kokkos::resize(this->_nonzLocIdx_host, sparseInput.get_nDim());
  #endif
#endif

  log.print("CpAprMultiplicativeUpdate: Resize and reorder.", Log::DEBUG_3);

#if !defined(KOKKOS_ENABLE_CUDA)
  for (ElemIdx i = 0; i < this->_nonzLoc.size(); ++i)
  {
    Kokkos::resize(this->_nonzLoc(i), static_cast<int32_t>(sparseInput.get_nElement()));
    Kokkos::resize(this->_nonzLocIdx(i),static_cast<int32_t>(kruskalOutput.get_factor_matrix_nRow(i) + 1));
    this->reorder(i,static_cast<int32_t>(kruskalOutput.get_factor_matrix_nRow(static_cast<SubIdx>(i))), static_cast<int32_t>(sparseInput.get_nElement()), sparseInput);
  }
#else 
  // Current version does not let GPU to choose non-atomic code
  #if 0
  for (ElemIdx i = 0; i < this->_nonzLoc.size(); ++i)
  {
    Kokkos::resize(this->_nonzLoc_host(i), static_cast<int32_t>(sparseInput.get_nElement()));
    Kokkos::resize(this->_nonzLocIdx_host(i),static_cast<int32_t>(kruskalOutput.get_factor_matrix_nRow(i) + 1));
    this->reorder(i,static_cast<int32_t>(kruskalOutput.get_factor_matrix_nRow(static_cast<SubIdx>(i))), static_cast<int32_t>(sparseInput.get_nElement()), sparseInput);
  }
  #endif
#endif

  Kokkos::Timer offsetTimer;
  Kokkos::Timer distributeWeightsTimer;
  Kokkos::Timer computePiTimer;
  Kokkos::Timer multiplicativeUpdateTimer;
  Kokkos::Timer normalizeTimer;
  Kokkos::Timer kktErrorNormTimer;
  Kokkos::Timer permuteTimer;

  double offsetTimerSeconds = 0.0;
  double distributeWeightsTimerSeconds = 0.0;
  double computePiTimerSeconds = 0.0;
  double multiplicativeUpdateTimerSeconds = 0.0;
  double normalizeTimerSeconds = 0.0;
  double kktErrorNormTimerSeconds = 0.0;
  double permuteTimerSeconds = 0.0;
  double tmpTime;
  std::vector<double> computeTimePerMode(sparseInput.get_nDim());
  for ( auto iDim : sparseInput.get_dim() ) 
  {
    computeTimePerMode[iDim] = 0;
  }
 
  this->_nOuterIter = 0;
  this->_numNonzeroViolations = 0;
  // Outer loop
  while (this->_nOuterIter < this->_maxOuterIter && !this->_isConverged)
  {
    log.print("nOuterIter: " + std::to_string(this->_nOuterIter), Log::DEBUG_1);
    this->_numNonzeroViolations = 0;
    this->_nInnerIter = 0;
    this->_isConverged = true;
    for (auto iDim : sparseInput.get_dim())
    {
      log.print("\tiDim: " + std::to_string(iDim), Log::DEBUG_1);
      if(this->_nOuterIter > 0) {
        offsetTimer.reset();
        this->offset(kruskalOutput,iDim);
        tmpTime =  offsetTimer.seconds();
        offsetTimerSeconds += tmpTime;
        computeTimePerMode[iDim]+=tmpTime;

        if (this->_isOffset)
        {
          ++this->_numNonzeroViolations;
        }
      }


      distributeWeightsTimer.reset();
      kruskalOutput.distribute_weights_to_factor_matrices(iDim);
      tmpTime = distributeWeightsTimer.seconds();
      distributeWeightsTimerSeconds += tmpTime;
      computeTimePerMode[iDim] += tmpTime;


      computePiTimer.reset();
      compute_pi(kruskalOutput, sparseInput, iDim);
      tmpTime = computePiTimer.seconds();

      computePiTimerSeconds += tmpTime;
      computeTimePerMode[iDim] += tmpTime;

#ifdef SPARTEN_CHECK_NANS
      std::cout << "NaNs present in kruskal tensor after computePi (MU method): " << (kruskalOutput.check_for_nans() ? "true" : "false") << std::endl;
#endif

      // Inner loop
      SubIdx innerLoopCount = 0;
      bool localConverged = false;
      while (innerLoopCount < this->_maxInnerIter && !localConverged)
      {
	log.print("\t\tinnerLoopCount: " + std::to_string(innerLoopCount), Log::DEBUG_1);

        // Includes computing timeComputePhi
        multiplicativeUpdateTimer.reset();
        localConverged = multiplicative_update(kruskalOutput, sparseInput, iDim);
        tmpTime = multiplicativeUpdateTimer.seconds();
        multiplicativeUpdateTimerSeconds += tmpTime;
        computeTimePerMode[iDim] += tmpTime;
        ++innerLoopCount;
#ifdef SPARTEN_CHECK_NANS
         std::cout << "NaNs present in kruskal tensor after multiplicative update: " << (kruskalOutput.check_for_nans() ? "true" : "false") << std::endl;
#endif
      }
      this->_nInnerIter += innerLoopCount;
      if(localConverged == false)
      {
        this->_isConverged = false;
      }

      normalizeTimer.reset();
      kruskalOutput.normalize(iDim); // using NormOne  Bring Lambda back
      tmpTime = normalizeTimer.seconds();
      normalizeTimerSeconds += tmpTime;
      computeTimePerMode[iDim] += tmpTime;

#ifdef SPARTEN_CHECK_NANS
      std::cout << "NaNs present in kruskal tensor after normalize (MU method): " << (kruskalOutput.check_for_nans() ? "true" : "false") << std::endl;
#endif
    }
    this->update_karush_kuhn_tucker_condition_error_norm(); // using NormInf
    ++this->_nOuterIter;  // Update iteration counter

    if (this->_nOuterIter % this->_progressInterval == 0)
    {
      this->progress();
    }
  }

  normalizeTimer.reset();
  kruskalOutput.normalize(); // using NormOne
  normalizeTimerSeconds += normalizeTimer.seconds();

  permuteTimer.reset();
  kruskalOutput.permute_factor_matrix_columns();
  permuteTimerSeconds += permuteTimer.seconds();
  // Output progress after final step
  this->progress();
  // Output timing information
  log.print("CpAprMultiplicativeUpdate Execute: " + std::to_string(cpAprTimer.seconds()) + " s", Log::RELEASE);
  log.print("\tOffset: " + std::to_string(offsetTimerSeconds) + " s", Log::RELEASE);
  log.print("\tDistribute Weights: " + std::to_string(distributeWeightsTimerSeconds) + " s", Log::RELEASE);
  log.print("\tCompute Pi: " + std::to_string(computePiTimerSeconds) + " s", Log::RELEASE);
  log.print("\tMultiplicative Update: " + std::to_string(multiplicativeUpdateTimerSeconds) + " s", Log::RELEASE);
  log.print("\tNormalize: " + std::to_string(normalizeTimerSeconds) + " s", Log::RELEASE);
  log.print("\tKKT Error Norm: " + std::to_string(kktErrorNormTimerSeconds) + " s", Log::RELEASE);
  log.print("\tPermute: " + std::to_string(permuteTimerSeconds) + " s", Log::RELEASE);

  for ( auto iDim : sparseInput.get_dim() )
  {
    std::string msg = "\tComputing Mode " + std::to_string(iDim) + ": " + std::to_string(computeTimePerMode[iDim]) + " s";
    log.print(msg, Log::RELEASE);
  }
}


template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateCaller_does_violate_karush_kuhn_tucker_condition_kruskal_tensor(
    KruskalValue &dMyMax,
    SubIdx kruskal_nRow,
    SubIdx kruskal_nColumn,
    const FactorMatrixConst<const KruskalValue> &dPhi,
    const FactorMatrixConst<const KruskalValue> &dKdata)
{
  const auto teamSize = Kokkos::AUTO;
  Kokkos::parallel_reduce (Kokkos::TeamPolicy<>(kruskal_nRow, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type thread, KruskalValue &ldMyMax)
  {
    const auto iMode = thread.league_rank();
    KruskalValue tldMyMax = 0;
    Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(thread, kruskal_nColumn), [=] (SubIdx iComp, KruskalValue &lldMyMax)
    {
      lldMyMax = static_cast<double>(max(static_cast<double>(lldMyMax), abs(min(static_cast<double>(dKdata(iMode,iComp)), 1.0 - static_cast<double>(dPhi(iMode,iComp))))));
    },Kokkos::Max<KruskalValue>(tldMyMax));
    
    ldMyMax = static_cast<double>(max(static_cast<double>(ldMyMax), static_cast<double>(tldMyMax)));
  }, Kokkos::Max<KruskalValue>(dMyMax));
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::does_violate_karush_kuhn_tucker_condition(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput, SubIdx const iDim )
{
  bool retval = false;

  const FactorMatrixConst<const KruskalValue> dPhi = this->_phiBuffer->get_factor_matrix(iDim);
  const FactorMatrixConst<const KruskalValue> dKdata = kruskalInput.get_factor_matrix(iDim);

  KruskalValue dMyMax = 0;

  CpAprMultiplicativeUpdateCaller_does_violate_karush_kuhn_tucker_condition_kruskal_tensor<SparseValue, KruskalValue, ElemIdx, SubIdx>(dMyMax, kruskalInput.get_factor_matrix_nRow(iDim), kruskalInput.get_factor_matrix_nColumn(iDim), dPhi, dKdata);

  this->_kktErrorPerDim[iDim] = dMyMax;
  if (dMyMax < this->_tolerance)
  {
    retval = true;
  }

  return retval;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::does_violate_karush_kuhn_tucker_condition(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput )
{
  bool retval = false;
  for (auto iDim : kruskalInput.get_dim()) {
    if (does_violate_karush_kuhn_tucker_condition(kruskalInput, iDim) == true) {
      retval = true;
      break;
    }
  }
  return retval; /// \todo update this...
}
  
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateCaller_multiplicative_update(
    SubIdx kruskal_nRow,
    SubIdx kruskal_nColumn,
    const FactorMatrixConst<const KruskalValue> &dPhi,
    Kokkos::View<KruskalValue **> &dKdata)
{
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> Policy;
#if defined(KOKKOS_ENABLE_CUDA)
  const SubIdx VectorSize = kruskal_nColumn == 1 ? 1 : std::min(16,2 << int(std::log2(kruskal_nColumn))-1);
  const SubIdx TeamSize = 128/VectorSize;
  const SubIdx LeagueSize = (kruskal_nRow+TeamSize-1)/TeamSize;
#else
  const SubIdx VectorSize = kruskal_nColumn == 1 ? 1 : std::min(8,2 << (int(std::log2(kruskal_nColumn))-1));
  const SubIdx TeamSize = 1;
  const SubIdx LeagueSize = kruskal_nRow;
#endif
  Policy policy(LeagueSize,TeamSize,VectorSize);
  Kokkos::parallel_for (policy, KOKKOS_LAMBDA (Policy::member_type team)
  {
    const auto team_index = team.team_rank();
    const auto iMode = team.league_rank()*TeamSize+team_index;
    if( iMode >= kruskal_nRow ) return;
    Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nColumn), [&] (SubIdx iComp)
    {
      dKdata(iMode, iComp) *= dPhi(iMode, iComp);
    });
  });
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::multiplicative_update(KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDim)
{
  Log &log = Log::new_log();
  log.print("----- CpAprMultiplicativeUpdate::multiplicative_update() -----", Log::DEBUG_2);

  bool kktViolation = false;

  Kokkos::Timer computePhiTimer;
  compute_phi(kruskalOutput, sparseInput, iDim);
  log.print("\tCpApr::multiplicative_update() compute_phi(): " + std::to_string(computePhiTimer.seconds()) + " s", Log::DEBUG_2);

#ifdef SPARTEN_CHECK_NANS
  std::cout << "NaNs present in kruskal tensor after computePhi (MU method): " << (kruskalOutput.check_for_nans() ? "true" : "false") << std::endl;
#endif

  Kokkos::Timer kktVioliationTimer;
  kktViolation = this->does_violate_karush_kuhn_tucker_condition(kruskalOutput, iDim);
  log.print("\tCpApr::multiplicative_update() does_violate_karush_kuhn_tucker_condition(): " + std::to_string(kktVioliationTimer.seconds()) + " s", Log::DEBUG_2);

  Kokkos::Timer multiplicativeUpdateCoreTimer;
  if (!kktViolation)
  {
    const FactorMatrixConst<const KruskalValue> dPhi = this->_phiBuffer->get_factor_matrix(iDim);
    auto dKdata = kruskalOutput.get_factor_matrix(iDim);

    CpAprMultiplicativeUpdateCaller_multiplicative_update<SparseValue, KruskalValue, ElemIdx, SubIdx>(kruskalOutput.get_factor_matrix_nRow(iDim), kruskalOutput.get_factor_matrix_nColumn(iDim), dPhi, dKdata);
  }
  log.print("\tCpApr::multiplicative_update() Core: " + std::to_string(multiplicativeUpdateCoreTimer.seconds()) + " s", Log::DEBUG_2);

  return kktViolation;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateCaller_compute_pi_region1(
    ElemIdx sparse_nElement,
    SubIdx kruskal_nComponent,
    FactorMatrix<KruskalValue> &dPi)
{
  Kokkos::deep_copy( dPi, 1.0 );
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateCaller_compute_pi_region2(
    ElemIdx sparse_nElement,
    SubIdx sparse_nDim,
    SubIdx iDimSkip,
    SubIdx kruskal_nComponent,
    FactorMatrix<KruskalValue> &dPi,
    const SparseIndicesConst<SubIdx> &indices,
    const std::vector<Kokkos::View<KruskalValue **>> &factorMatrices)
{
#if defined(KOKKOS_ENABLE_CUDA) 
  const SubIdx VectorSize = kruskal_nComponent == 1 ? 1 : std::min(16,2 << int(std::log2(kruskal_nComponent))-1);
  const SubIdx TeamSize  = 128/VectorSize;
  const SubIdx LeagueSize = (sparse_nElement+TeamSize-1)/TeamSize;
#else
  const SubIdx VectorSize = 16;
  const SubIdx TeamSize = 1;
  const SubIdx LeagueSize = (sparse_nElement+(SubIdx)TeamSize-1)/(SubIdx)TeamSize;
#endif

// Option 0
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> Policy;
  Policy policy(LeagueSize,TeamSize,VectorSize);

  for (SubIdx iDim = 0; iDim < sparse_nDim; ++iDim)
  {
    if (iDim != iDimSkip)
    {
      const auto kData = factorMatrices[iDim];
      Kokkos::parallel_for (policy, KOKKOS_LAMBDA (Policy::member_type team)
      {
        const auto team_index = team.team_rank();
        const auto iNonz = team.league_rank()*TeamSize+team_index;
        if( iNonz >= sparse_nElement ) return;
        const auto index = indices(iDim,iNonz);
        Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [=] (SubIdx iComp)
        {
          dPi(iNonz, iComp) *= kData(index, iComp);
        });
      });
    }
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::compute_pi(
    KruskalTensor<KruskalValue, SubIdx> const &kruskalOutput,
    SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput,
    SubIdx iDimSkip)
{
  CpAprMultiplicativeUpdateCaller_compute_pi_region1<SparseValue, KruskalValue, ElemIdx, SubIdx>(sparseInput.get_nElement(), kruskalOutput.get_nComponent(), _pi);

  // Grab the index storage
  const auto indices = sparseInput.get_indices_view();
  const auto factorMatrices = kruskalOutput.get_factor_matrices();

  CpAprMultiplicativeUpdateCaller_compute_pi_region2<SparseValue, KruskalValue, ElemIdx, SubIdx>(sparseInput.get_nElement(), sparseInput.get_nDim(), iDimSkip, kruskalOutput.get_nComponent(), _pi, indices, factorMatrices);

}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateCaller_compute_phi_region1(
    SubIdx kruskal_nRow,
    SubIdx kruskal_nColumn,
    FactorMatrix<KruskalValue> &phi)
{
  Log &log = Log::new_log();
  log.print("Entering CpAprMultiplicativeUpdateCaller_compute_phi_region1", Log::DEBUG_3);
  Kokkos::deep_copy( phi, 0.0 );
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateCaller_compute_phi_region2(
    SubIdx kruskal_nRow,
    SubIdx kruskal_nComponent,
    FactorMatrix<KruskalValue> &phi,
    NonZeroLocations<ElemIdx> &nonzLoc,
    NonZeroLocations<SubIdx> &nonzLocIdx,
    SubIdx iDim,
    KruskalValue eps,
    const FactorMatrixConst<KruskalValue> &pi,
    const SparseDataConst<SparseValue> &spData,
    const SubIdx sparse_nElement,
    const SparseIndices<SubIdx> &indices,
    const FactorMatrixConst<KruskalValue> &kData)
{
  Log &log = Log::new_log();
  log.print("Entering CpAprMultiplicativeUpdateCaller_compute_phi_region2", Log::DEBUG_3);
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> Policy;

#if defined(KOKKOS_ENABLE_CUDA)
  //
  // For GPUs (with atomics)
  //
  const SubIdx VectorSize = kruskal_nComponent == 1 ? 1 : std::min(16,2 << int(std::log2(kruskal_nComponent))-1);
  const SubIdx TeamSize = 128/VectorSize;
  const SubIdx LeagueSize = (sparse_nElement+TeamSize-1)/TeamSize;
  Policy policy(LeagueSize,TeamSize,VectorSize);

  Kokkos::parallel_for (policy, KOKKOS_LAMBDA (Policy::member_type team)
  {
    const auto iNonz_start = team.league_rank()*TeamSize;
    const auto iNonz_end = iNonz_start + TeamSize < sparse_nElement ? iNonz_start + TeamSize : sparse_nElement;

    Kokkos::parallel_for (Kokkos::TeamThreadRange(team, iNonz_start, iNonz_end), [&] (ElemIdx iNnz)
    {
      const auto index = indices(iDim,iNnz);
      KruskalValue dVal = 0;
      Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [=] (SubIdx iComp, KruskalValue &ldVal)
      {
        ldVal += kData(index, iComp) *pi(iNnz, iComp);
      }, dVal);
      
      dVal = static_cast<KruskalValue>(spData(iNnz)) /
        static_cast<KruskalValue>(max(static_cast<KruskalValue>(eps), static_cast<KruskalValue>(dVal)));
      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (SubIdx iComp)
      {
        Kokkos::atomic_add(&phi(index, iComp), dVal * pi(iNnz, iComp ));
      });
    });
  });
#elif 1
#if defined(KOKKOS_ENABLE_CUDA)
  // This is not used in the current CUDA version
  // For GPUs/CPUs (with least atomics)
  //
  const int RowBlockSize = 32;
  const SubIdx VectorSize = kruskal_nComponent == 1 ? 1 : std::min(32,2 << int(std::log2(kruskal_nComponent))-1);
  const SubIdx TeamSize = 128/VectorSize;
  const SubIdx RowsPerTeam = TeamSize*RowBlockSize;
  const SubIdx LeagueSize = (sparse_nElement+RowsPerTeam-1)/RowsPerTeam;
#else
   //
   // For CPUs (with least atomics)
   //
   const int RowBlockSize = 32;
   const SubIdx VectorSize = 1;
   const SubIdx TeamSize = 1;
   const SubIdx RowsPerTeam = TeamSize*RowBlockSize;
   const SubIdx LeagueSize = (sparse_nElement+RowsPerTeam-1)/RowsPerTeam;
#endif

   Vector<ElemIdx>  myNonzLoc    = nonzLoc(iDim);
   Vector<SubIdx>   myNonzLocIdx = nonzLocIdx(iDim);
   Policy policy(LeagueSize,TeamSize,VectorSize);

  typedef Kokkos::View< KruskalValue *, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
          ScratchViewType;

   int scratch_size = ScratchViewType::shmem_size( kruskal_nComponent );
   // int scratch_size = ScratchViewType::shmem_size( kruskal_nComponent*sizeof(KruskalValue) );

   // Need temporary space per thread
   Kokkos::parallel_for (policy.set_scratch_size( 0, Kokkos::PerThread( scratch_size )), KOKKOS_LAMBDA (Policy::member_type team)
   {
      const ElemIdx iNonz_start = (team.league_rank()*TeamSize + team.team_rank())*RowBlockSize;
      //const ElemIdx iNonz_end = iNonz_start + RowBlockSize;

      ScratchViewType mytmp( team.team_scratch( 0 ), kruskal_nComponent ); // temporary STORAGE
      ElemIdx firstRow;
      ElemIdx iPrev;
      ElemIdx iRow;
      for( ElemIdx ii =0 ; ii < RowBlockSize; ++ii )
      {
         ElemIdx iNnz = iNonz_start + ii;
         if( iNnz >= sparse_nElement )
            continue;

         const ElemIdx index = myNonzLoc(iNnz); // Location in spData and Pi
         iRow  = indices(iDim,index);
         KruskalValue dVal = 0;
         Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [=] (SubIdx iComp, KruskalValue &ldVal)
         {
            ldVal += kData(iRow, iComp) *pi(index, iComp);
         }, dVal);

         KruskalValue dValNew = static_cast<KruskalValue>(spData(index)) /
                                static_cast<KruskalValue>(max(static_cast<KruskalValue>(eps), static_cast<KruskalValue>(dVal)));

         if( ii == 0   )  // First time
         {
            firstRow = iRow;
            Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (SubIdx iComp)
            {
               mytmp[iComp] = dValNew * pi(index, iComp);
            });

         } else if(  iPrev != iRow ) { // New Row
            if ( iPrev == firstRow )
            {
               Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&](SubIdx iComp) {
                  Kokkos::atomic_add(&phi(iPrev, iComp), mytmp[iComp]); // Add previous accumulator
                  mytmp[iComp] = dValNew * pi(index, iComp);
               });
            } else {
               Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&](SubIdx iComp) {
                  // Add previous accumulator. This is not working with hyperthreading. 
                  // May need to disable ThreadVectorRange for CPUs (and it is OK from the performance viewpoint).
                  phi(iPrev, iComp) += mytmp[iComp] ; 
                 // Kokkos::atomic_add(&phi(iPrev, iComp), mytmp[iComp]); // Add previous accumulator
                  mytmp[iComp] = dValNew * pi(index, iComp);
               });
            }
         } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&](SubIdx iComp) {
               mytmp[iComp] += dValNew * pi(index, iComp);
            });
         }
         iPrev = iRow;
      }

      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (SubIdx iComp)
      {
         Kokkos::atomic_add( &phi(iRow, iComp), mytmp[iComp]);
      });
   });
   Kokkos::fence();
#else

  //
  // For CPUs (no atomics)
  //
  const SubIdx VectorSize = kruskal_nComponent == 1 ? 1 : std::min(16,2 << (int(std::log2(kruskal_nComponent))-1));
  const SubIdx TeamSize = 1;
  const SubIdx LeagueSize = (kruskal_nRow+TeamSize-1)/TeamSize;
  Policy policy(LeagueSize,TeamSize,VectorSize);
  Vector<ElemIdx>  myNonzLoc    = nonzLoc(iDim);
  Vector<SubIdx>   myNonzLocIdx = nonzLocIdx(iDim);
  Kokkos::parallel_for  (policy, KOKKOS_LAMBDA (Policy::member_type team)
  {
    const SubIdx iRow = team.league_rank()*TeamSize + team.team_rank();
    if( iRow >= kruskal_nRow ) return;
    for (ElemIdx i = myNonzLocIdx(iRow); i < myNonzLocIdx(iRow+1); ++i)
    {
      KruskalValue dVal = 0;
      const ElemIdx myind = myNonzLoc(i);
      Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (SubIdx iComp, KruskalValue &ldVal)
      {
        ldVal += pi(myind, iComp) * kData(iRow, iComp);
      },dVal);
      dVal = static_cast<KruskalValue>(spData(myind)) / static_cast<KruskalValue>(max(static_cast<double>(eps), static_cast<double>(dVal)));
      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (SubIdx iComp)
      {
        phi(iRow, iComp) +=  dVal * pi(myind, iComp );
      });
    }
  });
#endif
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::compute_phi(
    KruskalTensor<KruskalValue, SubIdx> const &kruskalOutput,
    SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput,
    SubIdx iDim)
{
  Log &log = Log::new_log();

  // Get Factored Matrix
  auto  kData = kruskalOutput.get_factor_matrix(iDim);
  const auto  spData = sparseInput.get_data_view();
  const auto  indices = sparseInput.get_indices_view();

  // Zero-out Phi
  FactorMatrix<KruskalValue> phi = this->_phiBuffer->get_factor_matrix(iDim);

  deep_copy(this->_phiBuffer->get_factor_matrix(iDim), 0.0);

  auto nonzLoc = this->_nonzLoc;
  auto nonzLocIdx = this->_nonzLocIdx;
 //  FactorMatrixAtomic<KruskalValue> phiAtomic = this->_phiBuffer->get_factor_matrix(iDim);

  CpAprMultiplicativeUpdateCaller_compute_phi_region2<SparseValue, KruskalValue, ElemIdx, SubIdx>(kruskalOutput.get_factor_matrix_nRow(iDim), kruskalOutput.get_factor_matrix_nColumn(iDim), phi, nonzLoc, nonzLocIdx, iDim, this->_eps, _pi, spData, sparseInput.get_nElement(), indices,  kData);


}


template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::progress() const
{
  Log &log = Log::new_log();

  std::stringstream message_release;
  message_release << " Iter " << std::setfill(' ') << std::setw(4) << this->_nOuterIter ;
  message_release << ": Inner Its = " << std::setfill(' ') << std::setw(2) << this->_nInnerIter ;
  message_release << ", KKT Violation = " << std::scientific << this->_errorNorm;
  message_release << ", nViolations = " << std::setfill(' ') << std::setw(2) << this->_numNonzeroViolations;
  log.print(message_release.str(), Log::RELEASE);

  CpAprBase<SparseValue, KruskalValue, ElemIdx, SubIdx>::progress();
}

// Explicit instantiation
template class CpAprMultiplicativeUpdate<type::SubIdx, float, type::SubIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::SubIdx, float, type::ElemIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::SubIdx, float, type::ElemIdx, type::ElemIdx>;

template class CpAprMultiplicativeUpdate<type::ElemIdx, float, type::SubIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::ElemIdx, float, type::ElemIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::ElemIdx, float, type::ElemIdx, type::ElemIdx>;

template class CpAprMultiplicativeUpdate<type::SubIdx, double, type::SubIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::SubIdx, double, type::ElemIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::SubIdx, double, type::ElemIdx, type::ElemIdx>;

template class CpAprMultiplicativeUpdate<type::ElemIdx, double, type::SubIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::ElemIdx, double, type::ElemIdx, type::SubIdx>;
template class CpAprMultiplicativeUpdate<type::ElemIdx, double, type::ElemIdx, type::ElemIdx>;

} // sparten
