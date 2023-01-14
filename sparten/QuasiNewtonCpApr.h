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

#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <sparten/spartenUtils.h>
#include <sparten/spartenTiming.h>
#include <sparten/SolverConfiguration.h>
#include <sparten/CpAprCommon.h>


#include <vector>

namespace sparten {

//==============================================================================

template <class NumericalTypes>
class QuasiNewtonCpApr : private CpAprCommon<NumericalTypes>
{
private:
  using base_t = CpAprCommon<NumericalTypes>;
  using kruskal_value_t = typename base_t::kruskal_value_t;
  using element_index_t = typename base_t::element_index_t;
  using sub_index_t = typename base_t::sub_index_t;
  using kruskal_tensor_t = typename base_t::kruskal_tensor_t;
  using sparse_tensor_t = typename base_t::sparse_tensor_t;

  QuasiNewtonSolverConfiguration<NumericalTypes> _config;
  CpAprOutputOptions _output_config;

public:

  QuasiNewtonCpApr(
   QuasiNewtonSolverConfiguration<NumericalTypes> config,
   CpAprOutputOptions output_config
  ) : _config(config),
      _output_config(output_config)
  { }

  void compute(kruskal_tensor_t& kruskalOutput, sparse_tensor_t const& sparseInput);

  template<class TeamPolicyType>
  void _compute_with_policy(kruskal_tensor_t& kruskalOutput, sparse_tensor_t const& sparseInput);

private:

	void _log_progress(
		sub_index_t outer_iter,
		element_index_t inner_iter,
		element_index_t func_evals,
		kruskal_value_t error_norm,
		kruskal_value_t obj,
		kruskal_value_t loopTimer
	) const;

	void _log_history(
		sub_index_t outer_iter,
		element_index_t inner_iter,
		element_index_t func_evals,
		kruskal_value_t error_norm,
		kruskal_value_t obj,
		kruskal_value_t loopTimer
	) const;

	void _log_summary(
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
	) const;

};

} // end namespace sparten
