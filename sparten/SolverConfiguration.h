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

namespace sparten {

struct CpAprOutputOptions {
  size_t progress_print_interval;
};

template <
  class NumericalTypes
>
struct CpAprConfiguration {
  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;

    sub_index_t max_outer_iterations;
    sub_index_t max_inner_iterations;

    kruskal_value_t offset_value;
    kruskal_value_t offset_tolerance;
    kruskal_value_t eps;

    int scheduling;
};


template <
  class NumericalTypes
>
struct DampedNewtonSolverConfiguration
  : CpAprConfiguration<NumericalTypes>
{
  using base_t = CpAprConfiguration<NumericalTypes>;
  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;


  DampedNewtonSolverConfiguration(
    CpAprConfiguration<NumericalTypes> cfg
  ) noexcept
    : base_t(cfg)
    { }

    kruskal_value_t eps_div_zero_grad;
    kruskal_value_t log_zero_safeguard;
    kruskal_value_t row_tolerance;
    kruskal_value_t eps_active_set;
    sub_index_t     max_backtrack_steps;
    kruskal_value_t min_variable_nonzero_tolerance;
    kruskal_value_t step_reduction_factor;
    kruskal_value_t suff_decrease_tolerance;
    kruskal_value_t mu_initial;
    kruskal_value_t damping_increase_factor;
    kruskal_value_t damping_decrease_factor;
    kruskal_value_t damping_increase_tolerance;
    kruskal_value_t damping_decrease_tolerance;
    sub_index_t     solver_option;
    sub_index_t     threshold_nonz_per_row;
    bool should_scale_hessian;
};


template <
  class NumericalTypes
>
struct QuasiNewtonSolverConfiguration
  : CpAprConfiguration<NumericalTypes>
{
  using base_t = CpAprConfiguration<NumericalTypes>;
  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;

  QuasiNewtonSolverConfiguration(
    CpAprConfiguration<NumericalTypes> cfg
  ) noexcept
    : base_t(cfg)
    { }

    kruskal_value_t eps_div_zero_grad;
    kruskal_value_t log_zero_safeguard;
    kruskal_value_t row_tolerance;
    kruskal_value_t eps_active_set;
    sub_index_t max_backtrack_steps;
    kruskal_value_t min_variable_nonzero_tolerance;
    kruskal_value_t step_reduction_factor;
    kruskal_value_t suff_decrease_tolerance;
    sub_index_t size_LBFGS;
};


template <class NumericalTypes>
struct KruskalTensorInfo {
using sub_index_t = typename NumericalTypes::sub_index_t;

sub_index_t num_components;
};

template <class NumericalTypes>
struct SparseTensorInfo
{
using sub_index_t = typename NumericalTypes::sub_index_t;

sub_index_t max_nonzeros_per_row;
//sub_index_t max_n_rows;

};

} // end namespace sparten
