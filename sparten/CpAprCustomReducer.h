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
#include <Kokkos_Core.hpp>


namespace sparten {

template <class KruskalValue>
struct CpAprCustomReducer
{
public:

  using reducer = CpAprCustomReducer;
  struct value_type {
    bool row_vars_modified = false; // join is logical or
    KruskalValue max_kkt_violation = 0.0; // join is max
    int64_t func_evals = 0; // join is sum
    int64_t iteration_count = 0; // join is sum

    //--------------------------------------------------------------------------
    // UGLY KOKKOS BUG WORKAROUND :-( !!!
    void operator+=(value_type const volatile& src) volatile {
      row_vars_modified = row_vars_modified || src.row_vars_modified;
      max_kkt_violation = sparten::max(max_kkt_violation, src.max_kkt_violation);
      func_evals += src.func_evals;
      iteration_count += src.iteration_count;
    }
    // END UGLY KOKKOS BUG WORKAROUND
    //--------------------------------------------------------------------------

  };
  using result_view_type =
    Kokkos::View<
      value_type,
      Kokkos::DefaultHostExecutionSpace,
      Kokkos::MemoryUnmanaged
    >;

private:
  value_type* value;

public:

  KOKKOS_INLINE_FUNCTION
  explicit CpAprCustomReducer(value_type& val) : value(&val) { }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src) const {
    dst.row_vars_modified = dst.row_vars_modified || src.row_vars_modified;

    dst.max_kkt_violation = sparten::max(dst.max_kkt_violation, src.max_kkt_violation);
    dst.func_evals += src.func_evals;
    dst.iteration_count += src.iteration_count;

  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dst, const volatile value_type& src) const {

    dst.row_vars_modified = dst.row_vars_modified || src.row_vars_modified;
    dst.max_kkt_violation = sparten::max(dst.max_kkt_violation, src.max_kkt_violation);
    dst.func_evals += src.func_evals;
    dst.iteration_count += src.iteration_count;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = value_type{};
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const {
    return result_view_type(value);
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value; }

};
}
