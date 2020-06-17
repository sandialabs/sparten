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
 \brief Class definition for Kruskal tensor storage
 */

#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/Log.h>
#include <generator/FactTestSetGenerator.h>
#include <generator/RandGenMersenneTwister.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace sparten
{

template<class KruskalValue, class SubIdx>
KruskalTensorInitializer<KruskalValue, SubIdx>::KruskalTensorInitializer()
{
  _iSeed = 131;
}

template<class KruskalValue, class SubIdx>
KruskalTensorInitializer<KruskalValue, SubIdx>::~KruskalTensorInitializer()
{

}

template<class KruskalValue, class SubIdx>
void KruskalTensorInitializer<KruskalValue, SubIdx>::initialize(KruskalTensor<KruskalValue, SubIdx> &kruskalTensor, Method method)
{
  const auto nComponent = kruskalTensor.get_nComponent();
  std::vector<KruskalValue> lambda(nComponent, 1.0);

  kruskalTensor.set_weight_vector(lambda);

  switch(method)
  {
  case UNITY:
    for (auto iDim : kruskalTensor.get_dim())
    {
      const auto nRow = kruskalTensor.get_factor_matrix_nRow(iDim);
      const auto nCol = kruskalTensor.get_factor_matrix_nColumn(iDim);

      for (type::SubIdx iCol = 0; iCol < nCol; ++iCol)
      {
        for (type::SubIdx iRow = 0; iRow < nRow; ++iRow)
        {
          kruskalTensor.set_factor_matrix_element(1, iDim, iRow, iCol);
        }
      }
    }
    break;
  case RANDOM:
    {
      RandGenMersenneTwister cRGen(_iSeed);
      for (auto iDim : kruskalTensor.get_dim())
      {
        const auto nRow = kruskalTensor.get_factor_matrix_nRow(iDim);
        const auto nCol = kruskalTensor.get_factor_matrix_nColumn(iDim);

        for (type::SubIdx iCol = 0; iCol < nCol; ++iCol)
        {
          for (type::SubIdx iRow = 0; iRow < nRow; ++iRow)
          {
            kruskalTensor.set_factor_matrix_element(static_cast<KruskalValue>(cRGen.generate_rand_double_matlab()), iDim, iRow, iCol);
          }
        }
      }
    }
    break;
  default:
    throw std::runtime_error("Unknown kruskal tensor initialization method.");
  }
  kruskalTensor.move_to_device();
  kruskalTensor.normalize();
}

template<class KruskalValue, class SubIdx>
void KruskalTensorInitializer<KruskalValue, SubIdx>::set_random_seed( int32_t iSeed )
{
  _iSeed = iSeed;
}

template<class KruskalValue, class SubIdx>
int32_t KruskalTensorInitializer<KruskalValue, SubIdx>::get_random_seed( )
{
  return _iSeed;
}

// Explicit instantiation
template class KruskalTensorInitializer<float, type::SubIdx>;
template class KruskalTensorInitializer<float, type::ElemIdx>;

template class KruskalTensorInitializer<double, type::SubIdx>;
template class KruskalTensorInitializer<double, type::ElemIdx>;

} // sparten
