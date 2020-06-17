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

namespace sparten {

//==============================================================================

template <
  class SubIdx,
  class NonzLocIdxViewArray,
  class DimSizes
>
SubIdx
compute_max_nonzero(
  SubIdx nDim,
  NonzLocIdxViewArray const& nonzLocIdxs,
  DimSizes const& dimSizes
)
{
  SubIdx return_val = 0;
  for(SubIdx iDim = 0; iDim < nDim; ++iDim)
  {
    SubIdx dimMax = 0;
    auto nonzLocIdxDim = nonzLocIdxs[iDim];
    Kokkos::parallel_reduce(
      dimSizes[iDim],
      KOKKOS_LAMBDA (int iRow, SubIdx& output) {
        output = sparten::max(output, nonzLocIdxDim[iRow+1] - nonzLocIdxDim[iRow]);
      },
      Kokkos::Max<SubIdx>(dimMax)
    );
    Kokkos::fence();
    return_val = sparten::max(return_val, dimMax);
  }
  return return_val;
}

template <class SubIdx, class NonzLocIdxViewArray, class DimSizes, class sizeView>
void compute_max_nonzero_per_mode(
           SubIdx nDim,
           sizeView &result,
           NonzLocIdxViewArray const& nonzLocIdxs,
           DimSizes const& dimSizes
   )
   {
      SubIdx return_val = 0;
      for(SubIdx iDim = 0; iDim < nDim; ++iDim)
      {
         SubIdx dimMax = 0;
         auto nonzLocIdxDim = nonzLocIdxs[iDim];
         Kokkos::parallel_reduce(
                 dimSizes[iDim],
                 KOKKOS_LAMBDA (int iRow, SubIdx& output) {
                    output = sparten::max(output, nonzLocIdxDim[iRow+1] - nonzLocIdxDim[iRow]);
                 },
                 Kokkos::Max<SubIdx>(dimMax)
         );
         Kokkos::fence();
         result[iDim]=dimMax;
      }
      return ;
   }

//==============================================================================

//----------------------------------------------------------------------
// NVCC 9.2 bug workaround
template <class PiView, class KDataT, class IndicesT, class SubIdx>
struct _compute_pi_workaround {
  PiView pi;
  KDataT kData;
  IndicesT indices;
  SubIdx nComponent;
  SubIdx iDim;
  KOKKOS_INLINE_FUNCTION
  void operator()(int iNonz) const {
    auto index = indices(iDim, iNonz);
    for(SubIdx iComp = 0; iComp < nComponent; ++iComp)
    {
      pi(iNonz, iComp) *= kData(index, iComp);
    }
  }
};
// end NVCC 9.2 bug workaround
//----------------------------------------------------------------------


template<class ElemIdx, class SubIdx, class SubIdx2, class PiView, class factMT, class IndicesT>
void  _sparten_compute_pi_internal(
    ElemIdx sparse_nElement,
    SubIdx sparse_nDim,
    SubIdx2 iDimSkip,
    SubIdx kruskal_nComponent,
    PiView &pi,
    factMT &factorMatrices,
    IndicesT &indices)
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
        Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (SubIdx iComp)
        {
          pi(iNonz, iComp) *= kData(index, iComp);
        });
      });
    }
  }
}

template <
  class... PiViewProperties,
  class KruskalTensorT,
  class SparseTensorT,
  class SubIdx
>
static void
compute_pi(
  Kokkos::View<PiViewProperties...> const& pi,
  KruskalTensorT& kruskalOutput,
  SparseTensorT const& sparseInput,
  SubIdx iDimSkip
)
{
  // reset Pi
  Kokkos::deep_copy(pi, 1.0);

  // grab the index storage
  auto indices = sparseInput.get_indices_view();
  auto factorMatrices = kruskalOutput.get_factor_matrices();
  auto nDim  = sparseInput.get_nDim();
  auto nElement = sparseInput.get_nElement();
  auto nComponent = kruskalOutput.get_nComponent();
#if 1
  // NVCC 9.2 compiler bug workaround
  _sparten_compute_pi_internal( 
     nElement,nDim, 
    iDimSkip,nComponent,
    pi, factorMatrices, indices);
  // end NVCC 9.2 compiler bug workaround
#else // David's workaround triggers illigal memory access. 


  for (SubIdx  iDim =0; iDim < nDim; ++iDim ) 
  {
    if (iDim != iDimSkip)
    {
      const auto kData = factorMatrices[iDim];
      // This version needs to take team policy

      Kokkos::parallel_for(
        nElement,
        //----------------------------------------------------------------------
        // NVCC 9.2 compiler bug workaround
        _compute_pi_workaround<
          decltype(pi), decltype(kData), decltype(indices), SubIdx
        >{pi, kData, indices, nComponent, iDim}
        // end NVCC 9.2 compiler bug workaround
        //----------------------------------------------------------------------
        // Without the workaround, this code would be:
        //KOKKOS_LAMBDA (int iNonz) {
        //  auto index = indices(iDim, iNonz);
        //  for(SubIdx iComp = 0; iComp < nComponent; ++iComp)
        //  {
        //    pi(iNonz, iComp) *= kData(index, iComp);
        //  }
        //}
        //----------------------------------------------------------------------
       );
       Kokkos::fence();
    }
  }
#endif
}

//==============================================================================


template <class NumericTypes>
class CpAprCommon {
public:

  using kruskal_value_t = typename NumericTypes::kruskal_value_t;
  using sparse_value_t = typename NumericTypes::sparse_value_t;
  using element_index_t = typename NumericTypes::element_index_t;
  using sub_index_t = typename NumericTypes::sub_index_t;
  using kruskal_tensor_t = KruskalTensor<kruskal_value_t, sub_index_t>;
  using sparse_tensor_t = SparseTensor<sparse_value_t, element_index_t, sub_index_t>;

protected:

  void _compute(
    kruskal_tensor_t& kruskalOutput,
    sparse_tensor_t const& sparseInput
  );

  static void
  check_requirements(
    kruskal_tensor_t const& kruskalOutput,
    sparse_tensor_t const& sparseInput
  );

  // TODO stop being lazy and write out the actual types here?

  template <
    class SparseIndicesHostView,
    class NonzeroLocsHostView,
    class NonzeroLocIdxsHostView
  >
  static void
  reorder_nonzeros_host(
    sub_index_t nRow, sub_index_t nNonz,
    SparseIndicesHostView spidx,
    NonzeroLocsHostView nonzLocs,
    NonzeroLocIdxsHostView nonzLocIdxs
  );


public: // CUDA 9.2 workaround :-(

  static bool
  is_non_negative(kruskal_tensor_t const& kruskal_tensor);

  static bool
  is_non_negative(sparse_tensor_t const& sparse_tensor);


};


} // end namespace sparten
