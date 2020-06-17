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

template <
  class SparseValue,
  class KruskalValue,
  class ElemIdx,
  class SubIdx
>
struct PrecisionTraits
{
  using sparse_value_t = SparseValue;
  using kruskal_value_t = KruskalValue;
  using element_index_t = ElemIdx;
  using sub_index_t = SubIdx;
};
} // end namespace sparten
