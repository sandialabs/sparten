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

#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <unit_test/CpAprUnitTestOnly.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace unitTest
{
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::CpAprMultiplicativeUpdateUnitTestOnly(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps) :
        sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>(tolerance, maxOuterIter, maxInnerIter, offset, offsetTolerance, progressInterval, eps)
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::~CpAprMultiplicativeUpdateUnitTestOnly()
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_does_violate_karush_kuhn_tucker_condition(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput)
{
  return sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::does_violate_karush_kuhn_tucker_condition(kruskalInput);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_is_non_negative(sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const
{
  return sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(sparseInput);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_is_non_negative(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const
{
  return sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(kruskalInput);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_multiplicative_update(sparten::KruskalTensor<KruskalValue, SubIdx> &kruskalOutput, sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput, SubIdx iDim)
{
  return sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::multiplicative_update(kruskalOutput, sparseInput, iDim);
}
  
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprMultiplicativeUpdateUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_progress() const
{
  return sparten::CpAprMultiplicativeUpdate<SparseValue, KruskalValue, ElemIdx, SubIdx>::progress();
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprDampedNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::CpAprDampedNewtonUnitTestOnly(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps) :
        sparten::CpAprDampedNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>(tolerance, maxOuterIter, maxInnerIter, offset, offsetTolerance, progressInterval, eps)
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprDampedNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::~CpAprDampedNewtonUnitTestOnly()
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprDampedNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_is_non_negative(sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const
{
  return sparten::CpAprDampedNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(sparseInput);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprDampedNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_is_non_negative(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const
{
  return sparten::CpAprDampedNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(kruskalInput);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprDampedNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_progress() const
{
  return sparten::CpAprDampedNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>::progress();
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprQuasiNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::CpAprQuasiNewtonUnitTestOnly(KruskalValue tolerance, SubIdx maxOuterIter, SubIdx maxInnerIter, KruskalValue offset, KruskalValue offsetTolerance, SubIdx progressInterval, KruskalValue eps) :
        sparten::CpAprQuasiNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>(tolerance, maxOuterIter, maxInnerIter, offset, offsetTolerance, progressInterval, eps)
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
CpAprQuasiNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::~CpAprQuasiNewtonUnitTestOnly()
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprQuasiNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_is_non_negative(sparten::SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput) const
{
  return sparten::CpAprQuasiNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(sparseInput);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool CpAprQuasiNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_is_non_negative(sparten::KruskalTensor<KruskalValue, SubIdx> const &kruskalInput) const
{
  return sparten::CpAprQuasiNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>::is_non_negative(kruskalInput);
}
  
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void CpAprQuasiNewtonUnitTestOnly<SparseValue, KruskalValue, ElemIdx, SubIdx>::test_only_progress() const
{
  return sparten::CpAprQuasiNewton<SparseValue, KruskalValue, ElemIdx, SubIdx>::progress();
}

// Explicit instantiation, Multiplicative Update
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::SubIdx, float, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprMultiplicativeUpdateUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx>;

// Explicit instantiation, Damped Newton
template class CpAprDampedNewtonUnitTestOnly<sparten::type::SubIdx, float, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprDampedNewtonUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprDampedNewtonUnitTestOnly<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprDampedNewtonUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprDampedNewtonUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx>;

#define SPARTEN_INSTANTIATE_UNIT_TESTS_ONLY 0
// Explicit instantiation, Quasi Newton
#ifndef SPARTEN_INSTANTIATE_UNIT_TESTS_ONLY
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::SubIdx, float, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprQuasiNewtonUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprQuasiNewtonUnitTestOnly<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>;
#endif
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>;
#ifndef SPARTEN_INSTANTIATE_UNIT_TESTS_ONLY
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx>;

template class CpAprQuasiNewtonUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::SubIdx, sparten::type::SubIdx>;
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>;
template class CpAprQuasiNewtonUnitTestOnly<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx>;
#endif

} // unitTest
