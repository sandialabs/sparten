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

#include <sparten/CpAprCommon.timpl.h>
#include <sparten/QuasiNewtonCpApr.h>
#include <sparten/QuasiNewtonKernel.h>
#include <sparten/spartenUtils.h>
#include <sparten/Log.h>
#include <sparten/PrecisionTraits.h>
#include <Kokkos_Core.hpp>
#include <sstream>

namespace sparten {

//=================== ===========================================================

template<class NumericTypes,class team_policy_t, class dkView,class spView,class piView, class locView, class locindexView, class statsView,
        class solver_conf_t, class kruskal_conf_t, class mode_conf_t, class scratch_space_t>
void _quasi_newton_driver(
        mode_conf_t  sparse_tensor_info,
        solver_conf_t config,
        kruskal_conf_t kruskal_info,
        piView pi,
        dkView dkdata,
        locView nonzLocs,
        locindexView nonzLocsIdx,
        spView spdata,
        statsView &row_stats,
        team_policy_t team_policy,
        scratch_space_t scratch_per_team

  )
{

   Kokkos::parallel_for( team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_per_team)), KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team )
   {
      const auto iRow = team.league_rank();

      // Construct the row subproblem Solver
      RowSubProblemQuasiNewton<team_policy_t, NumericTypes> row_solver =  RowSubProblemQuasiNewton<team_policy_t, NumericTypes>(
              /* iRow = */        iRow,
              /* team_member = */ team,
              /* config = */      config,
              /* input_info = */  sparse_tensor_info,
              /* kruskal_info = */ kruskal_info,
              /* pi = */ pi,
              /* dKdata = */ dkdata,
              /* nonz_loc = */ nonzLocs,
              /* nonz_loc_idx = */ nonzLocsIdx,
              /* sparse_data = */  spdata
      );


      if( not row_solver.has_nonzero_entries())
      {
         sparten::deep_copy(team, Kokkos::subview( dkdata, iRow, Kokkos::ALL), 0);
         Kokkos::single( Kokkos::PerTeam( team ), [&] () {
              // Manually set output and stats for rows that are skipped
              row_stats[iRow].row_modified = false;
              row_stats[iRow].t_elapsed = 0.0;
              row_stats[iRow].t_compute_phi = 0.0;
              row_stats[iRow].t_search_direction = 0.0;
              row_stats[iRow].t_line_search = 0.0;
              row_stats[iRow].n_func_evals = 0;
              row_stats[iRow].n_iters = 0;
              row_stats[iRow].obj = 0.0;
         });
      } else {
              auto result = row_solver._solve();
         Kokkos::single( Kokkos::PerTeam( team ), [&] () {
              row_stats[iRow] = row_solver.get_stats();

              row_stats[iRow].KKT = result.second;
              row_stats[iRow].row_modified = result.first;
         });
              // TODO Ideally, timers and counters will be contained in result; this will be unnecessary

      }

   });
}

template <class NumericTypes>
void
QuasiNewtonCpApr<NumericTypes>::compute(
	kruskal_tensor_t& kruskalOutput,
	sparse_tensor_t const& sparseInput
)
{
	if (_config.scheduling == 0)
	{
		using team_policy_t = Kokkos::TeamPolicy<>;
		_compute_with_policy<team_policy_t>(kruskalOutput, sparseInput);
	}
	else if (_config.scheduling == 1)
	{
		using team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
		_compute_with_policy<team_policy_t>(kruskalOutput, sparseInput);
	}
}

template <class NumericTypes>
template <class TeamPolicyType>
void
QuasiNewtonCpApr<NumericTypes>::_compute_with_policy(
  kruskal_tensor_t& kruskalOutput,
  sparse_tensor_t const& sparseInput
)
{
  using host_memory_space = typename Kokkos::DefaultHostExecutionSpace::memory_space;
  using device_memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
  using team_policy_t = TeamPolicyType;

  Log &log = Log::new_log();

  std::cout << std::setfill('-');
  std::cout << std::setw(70);
  std::cout << std::left;
  if (log.get_verbosity() > 1)
      std::cout << std::setw(90);
  std::cout << "------------------------- CP-APR Quasi-Newton " << std::endl;

  auto nElement = sparseInput.get_nElement();
  auto nComp = kruskalOutput.get_nComponent();
  auto nDim = sparseInput.get_nDim();
  kruskal_value_t piTime=0, computeTime=0;
  Kokkos::Timer cpAprTimer;

  auto kktErrorPerDim = sparten::make_unique_array<kruskal_value_t>(sparseInput.get_nDim());

  // Check requirements
  {
    Kokkos::Timer checkRequirementsTimer;
    base_t::check_requirements(kruskalOutput, sparseInput);
    log.print("CpAprQuasiNewton::check_requirements: " + std::to_string(checkRequirementsTimer.seconds()) + " s",
              Log::DEBUG_3);
  }

  // allocate pi
  auto pi = Kokkos::View<kruskal_value_t**, device_memory_space>(
    "pi matrix temporary",
    sparseInput.get_nElement(),
    kruskalOutput.get_nComponent()
  );

  // Allocate and reorder nonzero locations and indices
  using nonzero_locs_view_t = Kokkos::View<element_index_t*, device_memory_space>;
  using nonzero_locs_idx_view_t = Kokkos::View<sub_index_t*, device_memory_space>;
  auto nonzLocs = sparten::make_unique_array<nonzero_locs_view_t>(nDim);
  auto nonzLocsIdx = sparten::make_unique_array<nonzero_locs_idx_view_t>(nDim);

	// Want to get statistics about the number of nonzero entries per row per mode

	std::vector<sub_index_t > max_nonz_per_row(nDim);
	std::vector<sub_index_t > min_nonz_per_row(nDim);
	std::vector<kruskal_value_t > mean_nonz_per_row(nDim);
	std::vector<kruskal_value_t > stdev_nonz_per_row(nDim);
	std::ostringstream message_release;
	message_release << "\nSparse Tensor Nonzeros Per Row";
   for(sub_index_t iDim = 0; iDim < nDim; ++iDim) {

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

	  // Use Welford's algorithm to compute sample stats for each mode in a single pass
	  sub_index_t iRow = 0;
	  sub_index_t nnz = 0;
	  sub_index_t max = sparten::numeric_limits<sub_index_t >::min;
	  sub_index_t min = sparten::numeric_limits<sub_index_t >::max;
	  kruskal_value_t tmp = 0.0;
	  kruskal_value_t sum = 0.0;
	  kruskal_value_t mean = 0.0;
	  message_release << "\nMode: " << iDim;
	  while (iRow < nRow)
	  {
		  message_release << "\n" << nnz;
		  nnz = host_nonzLocsIdx[iRow+1] - host_nonzLocsIdx[iRow];
		  ++iRow;
		  tmp = nnz - mean;
		  sum += tmp * tmp * (iRow-1) / iRow;
		  mean += tmp / iRow;
		  if (nnz > max)
		  {
			  max = nnz;
		  }
		  else if (nnz < min) {
			  min = nnz;
		  }
	  }
	  message_release << "\n";
	  max_nonz_per_row[iDim] = max;
	  min_nonz_per_row[iDim] = min;
	  mean_nonz_per_row[iDim] = mean;
	  stdev_nonz_per_row[iDim] = std::sqrt(sum / iRow);
  }
	message_release << "\n";
	log.print(message_release.str(), Log::DEBUG_2);

  auto sparse_tensor_info =
    SparseTensorInfo<NumericTypes>{
       0
    };

  auto kruskal_info =
    KruskalTensorInfo<NumericTypes>{
      /* num_components = */ kruskalOutput.get_nComponent()
    };

  //auto kktViolationPerMode = sparten::make_unique_array<kruskal_value_t>(nDim);
  std::vector<kruskal_value_t> kktViolationPerMode(nDim);
  auto outer_iter              = 0;
  bool converged               = false;
  element_index_t func_evals   = 0;
  element_index_t inner_iters  = 0;
  kruskal_value_t obj          = 0.0;
  kruskal_value_t errorNorm    = sparten::numeric_limits<kruskal_value_t>::max;
#ifndef KOKKOS_ENABLE_CUDA
  sub_index_t num_threads      = Kokkos::DefaultExecutionSpace::impl_thread_pool_size();
#else
  sub_index_t num_threads      = 1;
#endif

	auto timing_data = QuasiNewtonTimingData{};

  auto scratch_per_team = RowSubProblemQuasiNewton<team_policy_t, NumericTypes>::required_scratch_size(
   _config, kruskal_info, sparse_tensor_info.max_nonzeros_per_row
  );

  std::ostringstream msg;
  msg << "  i";
  msg << "         kkt-violation";
  msg << "          -log-likelihood";
  msg << "      time (s)";
  if (log.get_verbosity() > 1)
  {
      msg << "  inner iters";
      msg << "   func evals";
  }
  log.print(msg.str(), Log::RELEASE);

  kruskal_value_t outerLoopTime = 0;
  while(outer_iter < _config.max_outer_iterations && not converged)
  {
  	Kokkos::Timer outerLoopTimer;
    auto anyRowVarsModified = false;

    for(auto iDim : sparseInput.get_dim()) {
       obj = 0;
      // Distribute weights
      {
        // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
        //auto _ = ScopedTimerContribution(timing_data.distributeWeights);
        // TODO make this work on the device
        kruskalOutput.distribute_weights_to_factor_matrices(iDim);
      }

      // Compute pi
      {
        // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
        //auto _ = ScopedTimerContribution(timing_data.computePi);
        Kokkos::Timer piTimer;
        compute_pi(pi, kruskalOutput, sparseInput, iDim);
        piTime+=piTimer.seconds();
      }

      // The task parallel part
      {
        ScopedTimerContribution _t(timing_data.taskParallel);
        kruskal_value_t maxKktViolationForMode = 0.0;
        //bool rowsModifiedForMode = false;

        // TODO dynamic scheduling policy

        // Parallel Computation Starts here
        // Loop over rows
        using reducer_type = CpAprCustomReducer<kruskal_value_t>;
        using reducer_value_t = typename reducer_type::value_type;
        //auto modeValues = reducer_value_t{};

        auto team_policy = sparten::get_team_policy<team_policy_t>(kruskalOutput.get_nPerMode(iDim));

        // TODO this is a workaround that will be much cleaner when Kokkos has a combined reducers feature (e.g. tuple)
        // TODO see: Kokkos issue #2013
        // For now, we pull out what we want by passing vector, row_stats, with a structure of timers and counters for each row
        // Then the results are coalesced below

        Kokkos::View<QuasiNewtonRowStats<element_index_t,kruskal_value_t> *> row_stats("Stats", kruskalOutput.get_nPerMode(iDim));
        typename Kokkos::View<QuasiNewtonRowStats<element_index_t ,kruskal_value_t> *>::HostMirror row_stats_host = Kokkos::create_mirror_view(row_stats);
#if 0
        Kokkos::parallel_reduce(
          team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_per_team)),
          QuasiNewtonKernel<team_policy_t, NumericTypes>{
            /* pi = */ pi,
            /* dKdata = */ kruskalOutput.get_factor_matrix(iDim),
            /* nonz_loc = */ nonzLocs[iDim],
            /* nonz_loc_idx = */ nonzLocsIdx[iDim],
            /* sparse_data = */ sparseInput.get_data_view(),
            /* configuration = */ _config,
            /* input_info = */ sparse_tensor_info,
            /* kruskal_info = */ kruskal_info,
            /* row_stats = */ row_stats,
          },
          reducer_type{modeValues}
        );
#else

        _quasi_newton_driver<NumericTypes,team_policy_t>( sparse_tensor_info, _config, kruskal_info, pi, kruskalOutput.get_factor_matrix(iDim), nonzLocs[iDim],
                 nonzLocsIdx[iDim], sparseInput.get_data_view(), row_stats, team_policy, scratch_per_team );
#endif
        // TODO put in the work to remove this fence?
        Kokkos::fence(); // to get reduction data
        deep_copy(row_stats_host,row_stats);
      //  kktViolationPerMode[iDim] = modeValues.max_kkt_violation;
       // anyRowVarsModified = anyRowVarsModified || modeValues.row_vars_modified;

        // Aggregate stats from each row per outer iteration
        for (int j = 0; j < kruskalOutput.get_nPerMode(iDim); j++) {
           timing_data.row.time_elapsed += row_stats_host[j].t_elapsed/num_threads;
           timing_data.row.compute_phi += row_stats_host[j].t_compute_phi;
           timing_data.row.search_direction  += row_stats_host[j].t_search_direction/num_threads;
           timing_data.row.line_search += row_stats_host[j].t_line_search;
           timing_data.row.failed_search += row_stats_host[j].t_failed_search;
           func_evals += row_stats_host[j].n_func_evals;
           inner_iters += row_stats_host[j].n_iters;
           obj += row_stats_host[j].obj;
           maxKktViolationForMode = std::max<kruskal_value_t>(maxKktViolationForMode,row_stats_host[j].KKT);
           anyRowVarsModified = anyRowVarsModified || row_stats_host[j].row_modified;
        }
        kktViolationPerMode[iDim] = maxKktViolationForMode;
      }

      {
        // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
        //auto _ = ScopedTimerContribution(timing_data.normalize);

        kruskalOutput.normalize(iDim);

      }

    }

   // errorNorm = *std::max_element(
   //   kktErrorPerDim.get(), kktErrorPerDim.get() + nDim
   // );
    errorNorm= * std::max_element( kktViolationPerMode.begin(), kktViolationPerMode.end() );
    if(not anyRowVarsModified) converged = true;

    ++outer_iter;

    // TODO objective function value
    outerLoopTime += outerLoopTimer.seconds();
    if(outer_iter % _output_config.progress_print_interval == 0)
    {
	    _log_progress(outer_iter, inner_iters, func_evals, errorNorm, obj, outerLoopTime);
    }
  }
  {
    // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
    //auto _ = ScopedTimerContribution(timing_data.normalize);
    kruskalOutput.normalize();
  }

  {
    // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
    //auto _ = ScopedTimerContribution(timing_data.permute);
    kruskalOutput.permute_factor_matrix_columns();
  }

  computeTime+=cpAprTimer.seconds();

  // Final logging
  _log_summary(outer_iter, inner_iters, func_evals, errorNorm, obj, timing_data.row.time_elapsed,
	             timing_data.row.compute_phi, timing_data.row.search_direction,
	             timing_data.row.line_search, piTime,computeTime,
	             max_nonz_per_row, min_nonz_per_row, mean_nonz_per_row, stdev_nonz_per_row);
  std::cout << "----------------------------------------------------------------------";
  if (log.get_verbosity() > 1)
        std::cout << "--------------------";
  std::cout << std::endl;
}

//==============================================================================

template <class NumericalTypes>
void QuasiNewtonCpApr<NumericalTypes>::_log_progress(
  sub_index_t outer_iter,
  element_index_t inner_iter,
  element_index_t func_evals,
  kruskal_value_t error_norm,
  kruskal_value_t obj,
  kruskal_value_t loopTimer
) const
{
  Log &log = Log::new_log();

  std::ostringstream message_release;
  message_release << std::setfill(' ')
                  << std::right
                  << std::setw(5) << outer_iter
                  << "  " << std::scientific << std::setprecision(16) << error_norm
                  << "  " << std::scientific << std::setprecision(16) << obj
                  << "  " << std::scientific << std::setprecision(3) << loopTimer;
  if (log.get_verbosity() > 1)
  {
      message_release << " " << std::setfill(' ') << std::setw(12) << inner_iter;
      message_release << " " << std::setfill(' ') << std::setw(12) << func_evals;
  }
  log.print(message_release.str(), Log::RELEASE);
}

//==============================================================================

template <class NumericalTypes>
void QuasiNewtonCpApr<NumericalTypes>::_log_history(
	sub_index_t outer_iter,
	element_index_t inner_iter,
	element_index_t func_evals,
	kruskal_value_t error_norm,
	kruskal_value_t obj,
	kruskal_value_t loopTimer
) const {
	Log &log = Log::new_log();

	std::ostringstream message_release;
	message_release << outer_iter;
	message_release << "," << inner_iter;
	message_release << "," << func_evals;
	message_release << "," << std::scientific << std::setprecision(16) << error_norm;
	message_release << "," << std::scientific << std::setprecision(16) << obj;
	message_release << "," << std::fixed << std::setprecision(3) << loopTimer;
	log.print(message_release.str(), Log::RELEASE);
}

//==============================================================================

template <class NumericalTypes>
void QuasiNewtonCpApr<NumericalTypes>::_log_summary(
	sub_index_t outer_iter,
	element_index_t inner_iter,
	element_index_t func_evals,
	kruskal_value_t error_norm,
	kruskal_value_t obj,
	kruskal_value_t elapsed_time,
	kruskal_value_t compute_phi,
	kruskal_value_t search_direction,
	kruskal_value_t line_search,
	kruskal_value_t piTime,
	kruskal_value_t computeTime,
	std::vector<sub_index_t > max_nonz_per_row,
	std::vector<sub_index_t > min_nonz_per_row,
	std::vector<kruskal_value_t > mean_nonz_per_row,
	std::vector<kruskal_value_t > stdev_nonz_per_row
) const
{
	Log &log = Log::new_log();

	std::ostringstream message_release;
	message_release << "\nAdvanced stats";
	message_release << "\n--------------";
	message_release << "\nFinal Log Likelihood: " << std::setw(16) << std::setprecision(9) << obj;
	message_release << "\nFinal KKT Violation: " << std::setw(16) << std::setprecision(9) << error_norm;
	message_release << "\nTotal Outer Iterations: " << std::setw(16) << outer_iter;
	message_release << "\nTotal Inner Iterations: " << std::setw(16) << inner_iter;
	message_release << "\nTotal Function Evaluations: " << std::setw(16) << func_evals;
	message_release << "\nTotal Time PDNR.compute() (s): " << std::setw(16) << std::setprecision(9) << computeTime;
	message_release << "\nTotal Time Pi (s): " << std::setw(16) << std::setprecision(9) << piTime;
	message_release << "\nTotal Time Phi (s): " << std::setw(16) << std::setprecision(9) << compute_phi;
	message_release << "\nTotal Time Perform Line Search (s): " << std::setw(16) << std::setprecision(9) << line_search;
#ifndef KOKKOS_ENABLE_CUDA
	message_release << "\nAverage Time per Thread Row Subproblem (s): " << std::setw(16) << std::setprecision(9) << elapsed_time;
	message_release << "\nAverage Time per Thread Compute Search Directions (s): " << std::setw(16) << std::setprecision(9) << search_direction;
#else
	message_release << "\nTotal Time Row Subproblem (s) " << std::setw(16) << std::setprecision(9) << elapsed_time;
	message_release << "\nTotal Time Compute Search Directions (s) " << std::setw(16) << std::setprecision(9) << search_direction;
#endif
	message_release << std::endl;
	message_release << "\nNonzeros Per Row";
	message_release << "\n----------------\n";
	message_release << std::left << std::setw(6) << "Mode";
	message_release << std::right << std::setw(8) << "Max";
	message_release << std::right << std::setw(8) << "Min";
	message_release << std::right << std::setw(25) << "Mean";
	message_release << std::right << std::setw(25) << "Stdev";
	for (int i = 0; i < max_nonz_per_row.size(); ++i)
	{
		message_release << std::fixed;
		message_release << std::left << "\n" << std::setw(6) << i;
		message_release << std::right << std::setfill(' ') << std::setw(8) << max_nonz_per_row[i];
		message_release << std::right << std::setfill(' ') << std::setw(8) << min_nonz_per_row[i];
		message_release << std::right << std::setfill(' ') << std::setw(25) << mean_nonz_per_row[i];
		message_release << std::right << std::setfill(' ') << std::setw(25) << stdev_nonz_per_row[i];
	}
	message_release << std::endl;

	log.print(message_release.str(), Log::DEBUG_1);
}
//==============================================================================

} // end namespace sparten
