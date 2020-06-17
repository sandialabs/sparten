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

#include <sparten/CpAprCommon.h>

namespace sparten {

//==============================================================================

template <class NumericTypes>
bool
CpAprCommon<NumericTypes>::is_non_negative(
  kruskal_tensor_t const& kruskal_tensor
)
{
  for (auto iDim : kruskal_tensor.get_dim())
  {
    element_index_t nNegative = 0;
    auto factorMatrix = kruskal_tensor.get_factor_matrix(iDim);
    auto nRow = kruskal_tensor.get_factor_matrix_nRow_host(iDim);
    auto nColumn = kruskal_tensor.get_factor_matrix_nColumn_host(iDim);
    auto policy = Kokkos::TeamPolicy<>(nRow, Kokkos::AUTO);
    using policy_type = decltype(policy);
    using member_type = typename policy_type::member_type;
    Kokkos::parallel_reduce("check_ktensor_nonnegative",
      std::move(policy),
      KOKKOS_LAMBDA (
        member_type const& member,
        element_index_t& lnNegative
      ) {
        const auto iRow = member.league_rank();

        element_index_t tnNegative = 0;
        Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, nColumn),
          [&](sub_index_t iComp, element_index_t& llnNegative)
          {
            llnNegative += static_cast<element_index_t>(factorMatrix(iRow, iComp) < 0);
          },
          tnNegative
        );
        lnNegative += tnNegative;
      }, nNegative
    );
    Kokkos::fence();
    if(nNegative > 0) return false;
  }
  return true;
}

//==============================================================================

template <class NumericTypes>
bool
CpAprCommon<NumericTypes>::is_non_negative(
  sparse_tensor_t const& sparse_tensor
)
{
  using el_idx_t = element_index_t; // CUDA LAMBDA workaround
  el_idx_t nNegative = 0;

  auto sparseData = sparse_tensor.get_data_view();
  Kokkos::parallel_reduce("check_sptensor_nonnegative",
    sparse_tensor.get_nElement(),
    KOKKOS_LAMBDA(el_idx_t elemId, el_idx_t& lnNegative)
    {
      lnNegative += static_cast<el_idx_t>( static_cast<double>(sparseData(elemId)) < 0); ///\todo: pointless comparison since sparseData is of type SparseValue, which is and unsigned int
    },
    Kokkos::Sum<el_idx_t> (nNegative)
  );

  return nNegative == 0;
}

//==============================================================================

template <class NumericTypes>
void
CpAprCommon<NumericTypes>::check_requirements(
  kruskal_tensor_t const& kruskalOutput,
  sparse_tensor_t const& sparseInput
)
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

//==============================================================================

template <class NumericTypes>
template <
  class SparseIndicesHostView,
  class NonzeroLocsHostView,
  class NonzeroLocIdxsHostView
>
void CpAprCommon<NumericTypes>::reorder_nonzeros_host(
  sub_index_t nRow,
  sub_index_t nNonz,
  SparseIndicesHostView spidx,
  NonzeroLocsHostView nonzLocs,
  NonzeroLocIdxsHostView nonzLocIdxs
)
{
  auto mystack = sparten::make_unique_array<std::vector<element_index_t>>(nRow);
  for(element_index_t iElem = 0; iElem < nNonz; ++iElem) {
    auto iRow = spidx[iElem];
    mystack[iRow].push_back(iElem);
  }

  element_index_t nz_count = 0;
  nonzLocIdxs[0] = 0;
  for(sub_index_t iRow = 0; iRow < nRow; ++iRow)
  {
    nonzLocIdxs[iRow + 1] = nonzLocIdxs[iRow] + mystack[iRow].size();
    for(sub_index_t iElem = 0; iElem < mystack[iRow].size(); ++iElem)
    {
      nonzLocs[nz_count] = mystack[iRow][iElem];
      ++nz_count;
    }
  }

}

#if 0
template <class NumericTypes,class Derived>
void
CpAprCommon<NumericTypes,Derived>::compute(
  kruskal_tensor_t& kruskalOutput,
  sparse_tensor_t const& sparseInput
)
{
  using host_memory_space = typename Kokkos::DefaultHostExecutionSpace::memory_space;
  using device_memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
  using team_policy_t = Kokkos::TeamPolicy<>;

  Log &log = Log::new_log();

  // readability aliases:
  auto nElement = sparseInput.get_nElement();
  auto nComp = kruskalOutput.get_nComponent();
  auto nDim = sparseInput.get_nDim();

  Kokkos::Timer cpAprTimer;

  auto kktErrorPerDim = sparten::make_unique_array<kruskal_value_t>(sparseInput.get_nDim());

  // Check requirements
  {
    Kokkos::Timer checkRequirementsTimer;
    base_t::check_requirements(kruskalOutput, sparseInput);
    log.print(self().className + ": check_requirements: " + std::to_string(checkRequirementsTimer.seconds()) + " s", Log::RELEASE);
  }

  // allocate pi
  auto pi = Kokkos::View<kruskal_value_t**, device_memory_space>(
    "pi matrix temporary",
    sparseInput.get_nElement(),
    kruskalOutput.get_nComponent()
  );

  // Allocate and reorder nonzero locations and indices

  // Conditionals
  if ( self().reorder == true )
  {
    using nonzero_locs_view_t = Kokkos::View<element_index_t*, device_memory_space>;
    using nonzero_locs_idx_view_t = Kokkos::View<sub_index_t*, device_memory_space>;
    auto nonzLocs = sparten::make_unique_array<nonzero_locs_view_t>(nDim);
    auto nonzLocsIdx = sparten::make_unique_array<nonzero_locs_idx_view_t>(nDim);
    for(sub_index_t iDim = 0; iDim < nDim; ++iDim)
    {

      auto nRow = kruskalOutput.get_factor_matrix_nRow_host(iDim);

      nonzLocs[iDim] = nonzero_locs_view_t{Kokkos::view_alloc(device_memory_space{}), nElement};
      nonzLocsIdx[iDim] = nonzero_locs_idx_view_t{Kokkos::view_alloc(device_memory_space{}), nRow + 1};

      auto host_nonzLocs = Kokkos::create_mirror_view(nonzLocs[iDim]);
      auto host_nonzLocsIdx = Kokkos::create_mirror_view(nonzLocsIdx[iDim]);
      base_t::reorder_nonzeros_host(
        nRow, nElement,
        Kokkos::subview(sparseInput.get_indices_host_view(), iDim, Kokkos::ALL),
        host_nonzLocs,
        host_nonzLocsIdx
      );
      Kokkos::deep_copy(/*dst=*/nonzLocs[iDim], /*src=*/host_nonzLocs);
      Kokkos::deep_copy(/*dst=*/nonzLocsIdx[iDim], /*src=*/host_nonzLocsIdx);
    }
  }
  auto sparse_tensor_info =
   SparseTensorInfo<NumericTypes>{ 0 };

  if ( self().reorder == true ) {
    sparse_tensor_info =
      SparseTensorInfo<NumericTypes>{
        /* max_nonzeros_per_row = */
          compute_max_nonzero<sub_index_t>(
           nDim, nonzLocsIdx, sparseInput.get_size_dim()
          )
      };
  }

  auto kruskal_info =
    KruskalTensorInfo<NumericTypes>{
      /* num_components = */ kruskalOutput.get_nComponent()
    };

  auto kktViolationPerMode = sparten::make_unique_array<kruskal_value_t>(nDim);

  auto outer_iter = 0;
  bool converged = false;
  int32_t funcEvals = 0;
  int32_t innerIters = 0;
  kruskal_value_t errorNorm = std::numeric_limits<kruskal_value_t>::max();

  auto timingData = self().DampedNewtonTimingData{};

  auto scratch_per_team = RowSubProblemDampedNewton<team_policy_t, NumericTypes>::required_scratch_size(
   _config, kruskal_info, sparse_tensor_info);

  while(outer_iter < _config.max_outer_iterations && not converged)
  {
    auto numNonzeroViolations = 0;
    auto anyRowVarsModified = false;

    for(auto iDim : sparseInput.get_dim()) {

      // Distribute weights
      {
        // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
        //auto _ = ScopedTimerContribution(timingData.distributeWeights);
        // TODO make this work on the device
        kruskalOutput.distribute_weights_to_factor_matrices(iDim);
      }

      // Compute pi
      {
        // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
        //auto _ = ScopedTimerContribution(timingData.computePi);
        compute_pi(pi, kruskalOutput, sparseInput, iDim);
      }

      // The task parallel part
      {
        ScopedTimerContribution _t(timingData.taskParallel);
        kruskal_value_t maxKktViolationForMode = 0.0;
        bool rowsModifiedForMode = false;

        // TODO function evaluation counts, iteration counts
        // TODO dynamic scheduling policy

        // Parallel Computaiton Starts here
        // Loop over rows
        using reducer_type = CpAprCustomReducer<kruskal_value_t>;
        using reducer_value_t = typename reducer_type::value_type;
        auto modeValues = reducer_value_t{};

        auto team_policy = team_policy_t(kruskalOutput.get_nPerMode(iDim), Kokkos::AUTO);

        Kokkos::parallel_reduce(
          team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_per_team)),
          DampedNewtonKernel<team_policy_t, NumericTypes>{
            /* pi = */ pi,
            /* dKdata = */ kruskalOutput.get_factor_matrix(iDim),
            /* nonz_loc = */ nonzLocs[iDim],
            /* nonz_loc_idx = */ nonzLocsIdx[iDim],
            /* sparse_data = */ sparseInput.get_data_view(),
            /* configuration = */ _config,
            /* input_info = */ sparse_tensor_info,
            /* kruskal_info = */ kruskal_info
          },
          reducer_type{modeValues}
        );

        // TODO put in the work to remove this fence?
        Kokkos::fence(); // to get reduction data

        kktViolationPerMode[iDim] = modeValues.max_kkt_violation;
        anyRowVarsModified = anyRowVarsModified || modeValues.row_vars_modified;
        funcEvals += modeValues.func_evals;
        innerIters += modeValues.iteration_count;
      }

      {
        // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
        //auto _ = ScopedTimerContribution(timingData.normalize);
        kruskalOutput.normalize(iDim);
      }

    }

    errorNorm = *std::max_element(
      kktErrorPerDim.get(), kktErrorPerDim.get() + nDim
    );

    if(not anyRowVarsModified) converged = true;

    ++outer_iter;

    // TODO objective function value
    if(outer_iter % _output_config.progress_print_interval == 0)
    {
      _log_progress(outer_iter, innerIters, errorNorm, 0.0);
    }

  }

  {
    // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
    //auto _ = ScopedTimerContribution(timingData.normalize);
    kruskalOutput.normalize();
  }

  {
    // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
    //auto _ = ScopedTimerContribution(timingData.permute);
    kruskalOutput.permute_factor_matrix_columns();
  }

  Kokkos::fence();

  // TODO display timing output and progress output
  _log_progress(outer_iter, innerIters, errorNorm, 0.0);
}
#endif
//==============================================================================

} // end namespace sparten
