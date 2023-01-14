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

#include <sparten/RowSubProblem.h>
#include <sparten/spartenUtils.h>
#include <sparten/spartenBlas.h>

namespace sparten {

//==============================================================================
// change to constructor that takes number of rows of particular problem and makes a kokkos view
template <
 class IntType,
 class ScalarType
>
struct DampedNewtonRowStats {
   IntType n_iters               = 0;
   IntType n_func_evals          = 0;
   ScalarType t_compute_phi      = 0.0;
   ScalarType t_search_direction = 0.0;
   ScalarType t_line_search      = 0.0;
   ScalarType t_failed_search    = 0.0;
   ScalarType t_elapsed          = 0.0;

	/* return the objective function value for each row -- Temporary fix */
	ScalarType obj            = 0.0;

	// Temporary solution as the custom reducer is not working
	ScalarType KKT = 0.0;
	bool row_modified = false;
};

template <
  class TeamPolicy,
  class NumericalTypes
>
class RowSubProblemDampedNewton
  : RowSubProblemCommon<RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>>
{
private:

  using common_base_t = RowSubProblemCommon<RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>>;
  template <class>
  friend class RowSubProblemCommon;

public:

  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sparse_value_t = typename NumericalTypes::sparse_value_t;
  using element_index_t = typename NumericalTypes::element_index_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;

protected:

  using team_policy_t = TeamPolicy;
  using execution_space = typename team_policy_t::execution_space;
  using team_member_t = typename team_policy_t::member_type;
  using scratch_memory_space = typename execution_space::scratch_memory_space;
  using device_memory_space = typename execution_space::memory_space;
  using size_type = typename device_memory_space::size_type;

  template <class T>
  using scratch_view_1d =
    Kokkos::View<
      T*,
      /* memory space = */ scratch_memory_space,
      /* memory traits = */ Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>
    >;

  template <class T>
  using scratch_view_2d =
    Kokkos::View<
      T**,
      /* memory space = */ scratch_memory_space,
      /* memory traits = */ Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>
    >;

  //----------------------------------------------------------------------------

  team_member_t const& _team_member;

  Kokkos::View<FactorMatrix<kruskal_value_t> *> const &_kTensor;
  DampedNewtonSolverConfiguration<NumericalTypes> const _config;
  KruskalTensorInfo<NumericalTypes> const _kruskal_info;
  DampedNewtonRowStats<element_index_t ,kruskal_value_t> stats;

  Kokkos::View<
    kruskal_value_t const**, device_memory_space,
    /* memory traits = */ Kokkos::MemoryTraits</*Kokkos::RandomAccess |*/ Kokkos::Restrict>
  > _pi;

  Kokkos::View<
    kruskal_value_t**, device_memory_space,
    /* memory traits = */ Kokkos::MemoryTraits<Kokkos::Restrict>
  > _dKdata;

  // TODO Re-enable RandomAccess in some outer scope
  Kokkos::View<
    sparse_value_t const*, device_memory_space,
    /* memory traits = */ Kokkos::MemoryTraits</*Kokkos::RandomAccess |*/  Kokkos::Restrict>
  > _sparseData;

   Kokkos::View<
        sub_index_t const**, device_memory_space,
         /* memory traits = */ Kokkos::MemoryTraits</*Kokkos::RandomAccess |*/  Kokkos::Restrict>
   > _sparseIndex;

   Kokkos::View<
    element_index_t const*, device_memory_space,
    /* memory traits = */ Kokkos::MemoryTraits<Kokkos::Restrict>
  > _nonzLoc;


  // These are temporaries that can be recomputed, but they're used so much
  // we'll just store it here for now
  sub_index_t _iRow;
  sub_index_t _nNonz;
  sub_index_t _numModes;
  sub_index_t _iModeSkip;

  // For counting the number of function evaluations & inner iterations
//  sub_index_t _nFnEvaluations = 0;
//  sub_index_t _nIters = 0;


  // Temporaries
  scratch_view_1d<kruskal_value_t> _daVars;
  scratch_view_1d<kruskal_value_t> _daVarsOld;
  scratch_view_1d<kruskal_value_t> _daWorkObj;
  scratch_view_1d<kruskal_value_t> _daPhi;
  scratch_view_1d<kruskal_value_t> _daGrad;
  scratch_view_1d<kruskal_value_t> _daWork1; // Workspace for UpsRow and Solution of Hessian system
  scratch_view_1d<kruskal_value_t> _daWork2; // Hessian for PDNR
  scratch_view_1d<kruskal_value_t> _daScaleD;
  scratch_view_1d<kruskal_value_t> _daSearchDirection;
  scratch_view_1d<kruskal_value_t> _daFreeSearch;
  scratch_view_1d<kruskal_value_t> _daS;
  scratch_view_2d<kruskal_value_t> _localPi;  // Never used

  //----------------------------------------------------------------------------
  scratch_view_1d<sub_index_t> _naFreeIndices;

  //----------------------------------------------------------------------------
private:

  /**
   *  Here, we're doing a bit of a dance to avoid putting the sizes of
   *  all of the temporary scratch views in multiple places.  We need to use
   *  the sizes in both the binding of the scratch and the computation of the
   *  scratch size, and it's **very** easy to accidentally make those two
   *  mismatch.  While this structure introduces a bit of extra complexity for
   *  the reader, in this case it's worth it for the maintainability and
   *  decrease in brittleness.
   *
   *  In the size computation case, Operation will be a function object that computes
   *  the aligned size in scratch and all of the other types will be references
   *  to size types that get summed in the caller to compute the scratch size.
   *
   *  In the binding to scratch case, Operation will create a view bound to
   *  scratch and the rest of the types will be the types of the scratch views
   *  themselves.
   *
   *  (I recognize that this pattern has other problems that introduce other
   *  places where it's easy to make mistakes, like the number of arguments, etc.
   *  I decided on this because it focuses on removing the most brittleness from
   *  the part that's most likely to get messed up: the sizes.)
   */




  template <
    class Operation,
    class WorkObjT,
    class Work1T,
    class Work2T,
    class FreeSearchT,
    class VarsT,
    class VarsOldT,
    class PhiT,
    class GradT,
    class SearchDirectionT,
    class ScaleDT
  >
  KOKKOS_INLINE_FUNCTION
  static void _handle_temporaries(
    DampedNewtonSolverConfiguration<NumericalTypes> const& config,
    KruskalTensorInfo<NumericalTypes> const& kruskal_info,
    // SparseTensorInfo<NumericalTypes> const& sparse_info,
    sub_index_t max_nonzeros_per_row,
    Operation&& op,
    WorkObjT& workObj,
    Work1T& work1,
    Work2T& work2,
    FreeSearchT& freeSearch,
    VarsT& vars,
    VarsOldT& varsOld,
    PhiT& phi,
    GradT& grad,
    SearchDirectionT& searchDirection,
    ScaleDT& scaleD
  ) {
    auto nComps = kruskal_info.num_components;
//    auto maxNzPerRow = max_nonzeros_per_row;
     //  Fix for Issue #216
    auto maxNzPerRow = nComps;
    workObj = op(workObj,maxNzPerRow); // Optional
    work1 = op(work1, nComps*nComps);  // Optional
    work2 = op(work2, nComps*nComps);  // Hessian
    freeSearch = op(freeSearch, nComps);
    vars = op(vars, nComps);
    varsOld = op(varsOld, nComps);
    phi = op(phi, nComps);
    grad = op(grad, nComps);
    searchDirection = op(searchDirection, nComps);
    scaleD = op(scaleD, nComps);

  }


   // See comment _handle_temporaries() for a discussion of why this extra
  // complexity is necessary
  struct _bindToScratch {
    scratch_memory_space const& _scratch_space;
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    ViewType operator()(ViewType const&, size_type size) const {
      return ViewType(_scratch_space, size);
    }
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    ViewType operator()(ViewType const&, size_type size1, size_type size2) const {
      return ViewType(_scratch_space, size1, size2);
    }
  };

  // TODO fix this once we need to bind temporaries to some type other than kruskal_value_t
  struct _computeTemporarySize{
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    size_type operator()(ViewType const&, size_type size) const {
      return scratch_memory_space::align(size * sizeof(kruskal_value_t));
    }
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    size_type operator()(ViewType const&, size_type size1, size_type size2) const {
      // TODO allow for padding (and add Kokkos::AllowPadding in the bind functor)
      return scratch_memory_space::align(size1 * size2 * sizeof(kruskal_value_t));
    }

  };

  KOKKOS_INLINE_FUNCTION
  void _bindTemporariesToScratchSpace(
    sub_index_t const& input_info
  ) {

    // We're going to do the actual binding to scratch here
    auto& scratch = _team_member.team_scratch(1);

    // See comment _handle_temporaries() for a discussion of why this extra
    // complexity is necessary
    _handle_temporaries(
      /* config = */ _config,
      /* kruskal_info = */ _kruskal_info,
      /* sparse_info = */ input_info,
      /* op = */ _bindToScratch{scratch},
      /* workObj = */ _daWorkObj,
      /* work1 = */ _daWork1,
      /* work2 = */ _daWork2,
      /* freeSearch = */ _daFreeSearch,
      /* vars = */ _daVars,
      /* varsOld = */ _daVarsOld,
      /* timeComputePhi = */ _daPhi,
      /* grad = */ _daGrad,
      /* timeComputeSearchDirection = */ _daSearchDirection,
      /* scaleD = */ _daScaleD
    );
     _naFreeIndices = scratch_view_1d<sub_index_t>(scratch, _kruskal_info.num_components);
  }


  //----------------------------------------------------------------------------
//protected:
public:
  template <
    class PiView,
    class KDataView,
    class NonzLocView,
    class NonzLocIdxView,
    class SparseDataView,
          class SparseIndexView
  >
  KOKKOS_INLINE_FUNCTION
  RowSubProblemDampedNewton(
    sub_index_t iRow,
    sub_index_t iMode,
    sub_index_t nMode,
    typename TeamPolicy::member_type const& team_member,
    DampedNewtonSolverConfiguration<NumericalTypes> config,
    SparseTensorInfo<NumericalTypes>  const& input_info,
    KruskalTensorInfo<NumericalTypes> const& kruskal_info,
    PiView const& pi,
    KDataView const& kdata,
    NonzLocView const& nonzLoc,
    NonzLocIdxView const& nonzLocIdx,
    SparseDataView const& sparseData,
    SparseIndexView const& index,
    Kokkos::View<FactorMatrix<kruskal_value_t> *> const& kTensor

  )
    : _iRow(iRow),
      _iModeSkip(iMode),
      _numModes(nMode),
      _team_member(team_member),
      _config(config),
      _kruskal_info(kruskal_info),
      _pi(pi),
      _dKdata(kdata),
      _sparseData(sparseData),
      _kTensor(kTensor),
      _sparseIndex(index)
  {
    // We only need the locations of the non-zeros in this row
    auto nonzBegin = nonzLocIdx(iRow);
    auto nonzEnd = nonzLocIdx(iRow + 1);
    _nNonz = nonzEnd - nonzBegin;
    _nonzLoc = Kokkos::subview(nonzLoc, Kokkos::make_pair(nonzBegin, nonzEnd));
    _bindTemporariesToScratchSpace(input_info.max_nonzeros_per_row);

  }

public:

  // KOKKOS_INLINE_FUNCTION
  static size_t
  required_scratch_size(
    DampedNewtonSolverConfiguration<NumericalTypes> const& config,
    KruskalTensorInfo<NumericalTypes> const& kruskal_info,
    sub_index_t const& input_info
  )
  {
    size_type workObjSize = 0;
    size_type work1Size = 0;
    size_type work2Size = 0;
    size_type varsSize = 0;
    size_type varsOldSize = 0;
    size_type phiSize = 0;
    size_type gradSize = 0;
    size_type searchDirectionSize = 0;
    size_type freeSearchSize = 0;
    size_type scaleDSize = 0;


    // See comment _handle_temporaries() for a discussion of why this extra
    // complexity is necessary
    _handle_temporaries(
      /* config = */ config,
      /* kruskal_info = */ kruskal_info,
      /* sparse_info = */ input_info,
      /* op = */  _computeTemporarySize{},
      /* workObj = */ workObjSize,
      /* work1 = */ work1Size,
      /* work2 = */ work2Size,
      /* freeSearch = */ freeSearchSize,
      /* vars = */  varsSize,
      /* varsOld = */ varsOldSize,
      /* timeComputePhi = */  phiSize,
      /* grad = */ gradSize,
      /* timeComputeSearchDirection = */ searchDirectionSize,
      /* scaleD = */ scaleDSize
    );


    return workObjSize
     + work1Size
     + work2Size
     + freeSearchSize
     + varsSize
     + varsOldSize
     + phiSize
     + gradSize
     + searchDirectionSize
     + scaleDSize
     /* _naFreeIndices */
     + scratch_view_1d<sub_index_t>::required_allocation_size(kruskal_info.num_components)
     ;
  }

  template <class DampedNewtonKernelType>
  // KOKKOS_INLINE_FUNCTION
  RowSubProblemDampedNewton(
    sub_index_t iRow,
    DampedNewtonKernelType const& kernel,
    typename DampedNewtonKernelType::team_member_t const& team_member
  ) : RowSubProblemDampedNewton(
        /* iRow = */ iRow,
        /* team_member = */ team_member,
        /* config = */ kernel.configuration,
        /* input_info = */ kernel.input_info,
        /* kruskal_info = */ kernel.kruskal_info,
        /* pi = */ kernel.pi,
        /* kdata = */ kernel.dKdata,
        /* nonzLoc = */ kernel.nonz_loc,
        /* nonzLocIdx = */ kernel.nonz_loc_idx,
        /* sparseData = */ kernel.sparse_data
      )
  { /* forwarding constructor, must be empty */ }

  KOKKOS_INLINE_FUNCTION
  bool has_nonzero_entries() const
  {
    return _nNonz != 0;
  }

  //----------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<bool, kruskal_value_t>
  _solve();



  //----------------------------------------------------------------------------

	KOKKOS_INLINE_FUNCTION            // TODO this will be unnecessary in the future
	DampedNewtonRowStats<element_index_t ,kruskal_value_t> get_stats()
	{
	return stats;
	}

protected:


  /**
   *
   * @param dMuDamping
   * @post  dPredictedReduction
   */
  KOKKOS_INLINE_FUNCTION
  int32_t
  _compute_search_dir(
    kruskal_value_t  dMuDamping,
    kruskal_value_t  &dPredictedReduction,
    sub_index_t  numFree
  );




  KOKKOS_INLINE_FUNCTION
  void
  setup_local_pi() ;

  /**
   *
   * TODO explain what phi is and what this does
   *
   *
   * @pre `_sparseData` contains the data from the sparse input // TODO this is just a class precondition
   * @pre `_localPi` is a view (2D, # of nonzeros in the row by # of components0
   *      of the values of pi for the row.
   * @pre `_nonzLoc` is a view (1D, # of nonzeros in the row) of the indices in
   *       `sparseData_` of the non-zeros for this row
   * @pre `_daVars` is a view of the row variables
   * @pre `_daWork1` is a view with at least (# of nonzeros for this row) entries.
   * @pre `_daWork2` is a view with at least (# of nonzeros for this row) entries.
   *
   * @post `_daPhi` is a view of Phi (???)
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  compute_phi() const;

#if 1
  template <class KDataView>
  KOKKOS_INLINE_FUNCTION
  void
  _set_daVars( KDataView const &inKdata )
  {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(_team_member,_kruskal_info.num_components),
    [&] (sub_index_t iFixed) {

      _daVars(iFixed) = inKdata(iFixed);
    });

  }


  template <class KDataView>
  KOKKOS_INLINE_FUNCTION
  void
  _get_daPhi( KDataView &outPhi )
  {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(_team_member,_kruskal_info.num_components),
    [&] (sub_index_t iFixed) {

      outPhi(iFixed) = _daPhi(iFixed);
    });

  }
#endif



  /**
   *
   * Computes the value of the objective function (???) for the row variables
   * in the parameter
   *
   * @tparam KruskalView1D A view (1D, # of components, kruskal_value_t)
   * @param[in] daRowVars The variables to evaluate the function on.
   * @return value (???) of the objective function for the row variables given
   *
   * @pre `_sparseData` contains the data from the sparse input
   * @pre `_nonzLoc` is a view (1D, # of nonzeros in the row) of the indices in
   *       `sparseData_` of the non-zeros for this row
   * @pre all of the indices in `_nonzLoc` are valid entries in `_sparseData`
   * @pre `_localPi` is a view (2D, # of nonzeros in the row by # of components0
   *      of the values of pi for the row.
   * @pre `_team_member` is the `TeamMember` instance in charge of this row
   *      subproblem
   * @pre `_daWorkObj` is a view with at least # of nonzeros for this row entries.
   *
   * @post `daRowVars` is unchanged
   * @post `_nFnEvaluations` is incremented unless the function evaluation failed
   *
   */
  template <class KruskalView1D>
  KOKKOS_INLINE_FUNCTION
  //kruskal_value_t
  typename  KruskalView1D::value_type
  obj_likelihood  (
    KruskalView1D &daRowVars
  ) ;

  /**
   *
   * Computes the new row variables by stepping `daRowVars` along
   * `_daSearchDirection` until (???)
   *
   * @pre `_daSearchDirection` must be a view (1D, # of components) of the
   *      search direction to step in from `daRowVars` to produce `daNewRowVars`.
   * @pre `_daPhi` must be a view (1D, # of components) of phi (i.e., output of
   *      `_compute_phi` under ??? conditions).  Note that `_daPhi` is only
   *      used when the line search fails (i.e., `nRetCode == ????`)
   * @pre `_team_member` is the `TeamMember` instance in charge of this row
   *      subproblem
   * @pre It must be valid to call `_obj_likelihood` with `daRowVars` and `daNewRowVars`
   *
   * @post `_daSearchDirection` is unmodified
   * @post `_daPhi` is unmodified
   *
   *
   * @todo These (and others) are part of the contract of `NewtonSolverConfiguration`
   * @pre `_config.max_backtrack_steps` must be positive.
   * @pre `_config.step_reduction_factor` must be a real number in the range (0.0, 1.0].
   * @pre `_config.min_variable_nonzero_tolerance` must be a non-negative real number
   *
   *
   * @param[in] daRowVars The input row variables view (1D, # of components)
   * @param[in] daRowGrad A view (1D, # of components) of the gradient of the input row variables
   * @param[out] daNewRowVars The output row variables view (1D, # of components)
   * @param[out] dObjNew The value of the objective function after taking the step
   * @param[out] dUnitStepAred ???
   * @param[in,out] nRetCode ???
   */
  template <class RowVarsView, class RowGradView, class NewRowVarsView>
  KOKKOS_INLINE_FUNCTION
  void
  line_search(
    RowVarsView const& daRowVars,
    RowGradView const& daRowGrad,
    NewRowVarsView & daNewRowVars,
    kruskal_value_t& dObjNew,
    kruskal_value_t& dUnitStepAred,
    int32_t& nRetCode
  );


};

} // end namespace sparten
#include <sparten/RowSubProblemDampedNewton.timpl.h>
