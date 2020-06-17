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
 \brief Class definition for sparse tensor data storage
 */

#include <sparten/SparseTensor.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <string>
#include <sstream>

namespace sparten
{

template<class SparseValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx>::SparseTensor()

{

}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx>::SparseTensor(SubIdx nDim, ElemIdx nElement, std::vector<SubIdx> const &sizeDim) :
        _data("_data", nElement),
        _nDim(nDim),
        _dim(new Dimension(nDim)),
        _nElement(nElement),
        _indices("_indices",  nDim, nElement),
        _sizeDim(sizeDim)
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
   _indices_host=Kokkos::create_mirror(_indices);
   _data_host=Kokkos::create_mirror(_data);
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx>::SparseTensor(SparseTensor<SparseValue, ElemIdx, SubIdx> const &source) :
        _data("_data", source.get_nElement()),
        _nDim(source.get_nDim()),
        _dim(new Dimension(source.get_nDim())),
        _nElement(source.get_nElement()),
        _indices("_indices", source.get_nDim(), source.get_nElement()),
        _sizeDim(source.get_size_dim())

{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
   _indices_host=Kokkos::create_mirror(_indices);
   _data_host=Kokkos::create_mirror(_data);
#endif
  Kokkos::deep_copy(_data, source.get_data_view());
  Kokkos::deep_copy(_indices, source.get_indices_view());

}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx>::~SparseTensor()
{
  delete _dim;
}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseValue SparseTensor<SparseValue, ElemIdx, SubIdx>::get_element_data(ElemIdx elemId) const
{
  return _data(elemId);
}

template<class SparseValue, class ElemIdx, class SubIdx>
ElemIdx SparseTensor<SparseValue, ElemIdx, SubIdx>::get_element_index(ElemIdx elemId, SubIdx indiceId) const
{
  return _indices(indiceId, elemId);
}

template<class SparseValue, class ElemIdx, class SubIdx>
void SparseTensor<SparseValue, ElemIdx, SubIdx>::set_element_data(ElemIdx elemId, SparseValue value )
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  _data_host(elemId) = value;
#else
  _data(elemId) = value;
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
void SparseTensor<SparseValue, ElemIdx, SubIdx>::set_element_index(ElemIdx elemId, SubIdx indiceId, SubIdx iVal)
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  _indices_host(indiceId, elemId) = iVal;
#else
  _indices(indiceId, elemId) = iVal;
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
void SparseTensor<SparseValue, ElemIdx, SubIdx>::set_element(ElemIdx elemId, std::vector<SubIdx> const &indices, SparseValue value)
{
  set_element_data(elemId, value);

  for (ElemIdx indiceId = 0; indiceId < indices.size(); ++indiceId)
  {
    set_element_index(elemId, indiceId, indices[indiceId]);
  }
}

template<class SparseValue, class ElemIdx, class SubIdx>
void SparseTensor<SparseValue, ElemIdx, SubIdx>::set_element_data_to_zero()
{
  const auto nElement = _nElement;
  auto data = _data;

  Kokkos::parallel_for(nElement, KOKKOS_LAMBDA (ElemIdx elemId)
  {
    data(elemId) = 0;
  });
}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseData<SparseValue> SparseTensor<SparseValue, ElemIdx, SubIdx>::get_data_view()
{
  return _data;
}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseData<SparseValue> SparseTensor<SparseValue, ElemIdx, SubIdx>::get_data_view() const
{
  return _data;
}

template<class SparseValue, class ElemIdx, class SubIdx>
typename SparseData<SparseValue>::HostMirror SparseTensor<SparseValue, ElemIdx, SubIdx>::get_data_host_view()
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy( _data_host, _data);
  return _data_host;
#else
  return _data;
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
typename SparseData<SparseValue>::HostMirror SparseTensor<SparseValue, ElemIdx, SubIdx>::get_data_host_view() const
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy( _data_host, _data);
  return _data_host;
#else
  return _data;
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseIndices<SubIdx> SparseTensor<SparseValue, ElemIdx, SubIdx>::get_indices_view()
{
  return _indices;
}

template<class SparseValue, class ElemIdx, class SubIdx>
SparseIndices<SubIdx> SparseTensor<SparseValue, ElemIdx, SubIdx>::get_indices_view() const
{
  return _indices;
}

template<class SparseValue, class ElemIdx, class SubIdx>
typename SparseIndices<SubIdx>::HostMirror SparseTensor<SparseValue, ElemIdx, SubIdx>::get_indices_host_view() 
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy( _indices_host, _indices);
  return _indices_host;
#else
  return _indices;
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
typename SparseIndices<SubIdx>::HostMirror SparseTensor<SparseValue, ElemIdx, SubIdx>::get_indices_host_view() const
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy( _indices_host, _indices);
  return _indices_host;
#else
  return _indices;
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
SubIdx SparseTensor<SparseValue, ElemIdx, SubIdx>::get_nRow(SubIdx iDim) const
{
  return _sizeDim[iDim];
}


template<class SparseValue, class ElemIdx, class SubIdx>
SubIdx SparseTensor<SparseValue, ElemIdx, SubIdx>::get_nDim() const
{
  return _nDim;
}

template<class SparseValue, class ElemIdx, class SubIdx>
Dimension SparseTensor<SparseValue, ElemIdx, SubIdx>::get_dim() const
{
  return *_dim;
}

template<class SparseValue, class ElemIdx, class SubIdx>
std::vector<SubIdx> const &SparseTensor<SparseValue, ElemIdx, SubIdx>::get_size_dim() const
{
  return _sizeDim;
}

template<class SparseValue, class ElemIdx, class SubIdx>
ElemIdx SparseTensor<SparseValue, ElemIdx, SubIdx>::get_nElement() const
{
  return _nElement;
}

template<class SparseValue, class ElemIdx, class SubIdx>
void SparseTensor<SparseValue, ElemIdx, SubIdx>::copy_to_device()
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy( _indices, _indices_host);
  Kokkos::deep_copy( _data,    _data_host);
#endif
}

template<class SparseValue, class ElemIdx, class SubIdx>
void SparseTensor<SparseValue, ElemIdx, SubIdx>::copy_to_host()
{
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  Kokkos::deep_copy( _indices_host, _indices);
  Kokkos::deep_copy( _data_host,    _data);
#endif
}


template<class SparseValue, class ElemIdx, class SubIdx>
std::string SparseTensor<SparseValue, ElemIdx, SubIdx>::show()
{
  std::stringstream ss;

  for (ElemIdx i = 0; i < _nElement; ++i)
  {
    for(auto iDim : *_dim)
    {
      ss << _indices(iDim, i) << "\t";
    }

    ss << _data(i) << std::endl;
  }

  return ss.str();
}

// Explicit instantiation
template class SparseTensor<type::SubIdx, type::SubIdx, type::SubIdx>;
template class SparseTensor<type::SubIdx, type::ElemIdx, type::SubIdx>;
template class SparseTensor<type::SubIdx, type::ElemIdx, type::ElemIdx>;

template class SparseTensor<type::ElemIdx, type::SubIdx, type::SubIdx>;
template class SparseTensor<type::ElemIdx, type::ElemIdx, type::SubIdx>;
template class SparseTensor<type::ElemIdx, type::ElemIdx, type::ElemIdx>;

} // sparten
