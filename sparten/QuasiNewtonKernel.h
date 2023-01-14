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

#pragma once

#include <sparten/PrecisionTraits.h>
#include <sparten/CpAprCustomReducer.h>
#include <sparten/SparseTensor.h>
#include <sparten/RowSubProblemQuasiNewton.h>
#include <sparten/spartenAlgorithms.h>
#include <sparten/SolverConfiguration.h>

namespace sparten {


// TODO move this to a forward declarations file
template <class TeamPolicy, class NumericalTypes>
class RowSubProblemQuasiNewton;

template <
  class TeamPolicy,
  class NumericalTypes
>
struct QuasiNewtonKernel
{
  // Kokkos-related member types:

  using team_policy_t = TeamPolicy;
  // When Kokkos sees an execution_space member type, it will override the
  // default execution space and run on that one.  This likely doesn't change
  // anything here, but just so that we don't get any unexpected behavior:
  using execution_space = typename team_policy_t::execution_space;
  using device_memory_space = typename execution_space::memory_space;
  using team_member_t = typename team_policy_t::member_type;


  // SparTen-specific member types

  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sparse_value_t = typename NumericalTypes::sparse_value_t;
  using element_index_t = typename NumericalTypes::element_index_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;

  using reducer_type = CpAprCustomReducer<kruskal_value_t>;
  using reducer_value_t = typename reducer_type::value_type;

  Kokkos::View<kruskal_value_t const**, device_memory_space> pi;
  Kokkos::View<kruskal_value_t**, device_memory_space> dKdata;
  Kokkos::View<element_index_t const*, device_memory_space> nonz_loc;
  Kokkos::View<sub_index_t const*, device_memory_space> nonz_loc_idx;
  Kokkos::View<sparse_value_t const*, device_memory_space> sparse_data;

  QuasiNewtonSolverConfiguration<NumericalTypes> configuration;
  SparseTensorInfo<NumericalTypes> input_info;
  KruskalTensorInfo<NumericalTypes> kruskal_info;
  Kokkos::View<QuasiNewtonRowStats<element_index_t, kruskal_value_t> *> row_stats;


  KOKKOS_INLINE_FUNCTION
  void operator()(
    team_member_t const& team_member,
    reducer_value_t& output
  ) const
  {

    // Readability aliases:
    auto iRow = team_member.league_rank();

    // Construct the row subproblem Solver
    auto row_solver = RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>{
      /* iRow = */ sub_index_t(iRow),
      /* kernel = */ *this,
      /* team_member = */ team_member
    };

    if(not row_solver.has_nonzero_entries())
    {
	    sparten::deep_copy(team_member, Kokkos::subview( dKdata, iRow, Kokkos::ALL), 0);

	    // Manually set output and stats for rows that are skipped
	    output.row_vars_modified = false;
	    row_stats[iRow].t_elapsed = 0.0;
	    row_stats[iRow].t_compute_phi = 0.0;
	    row_stats[iRow].t_search_direction = 0.0;
	    row_stats[iRow].t_line_search = 0.0;
	    row_stats[iRow].n_func_evals = 0;
	    row_stats[iRow].n_iters = 0;
	    row_stats[iRow].obj = 0.0;
    }
    else
    {
      auto result = row_solver._solve();

      output.row_vars_modified = output.row_vars_modified || result.first;
      if(result.second > output.max_kkt_violation) {
        output.max_kkt_violation = result.second;
      }

	    // TODO Ideally, timers and counters will be contained in result; this will be unnecessary
#ifndef KOKKOS_ENABLE_CUDA
	    row_stats[iRow] = row_solver.get_stats();
#endif
       row_stats[iRow].KKT = result.second;
       row_stats[iRow].row_modified = result.first;
    }

  }

};

} // end namespace sparten
