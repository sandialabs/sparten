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
 \brief Class declaration for sparse tensor data storage
 */

#pragma once

#include <sparten/Dimension.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>

namespace sparten
{

//! Data layouts for more efficient sparse tensor operations
enum DataLayout
{
  DATALAYOUT_ASCENDING,  //!< Mode 0 is fastest varying, Mode N is slowest varying in memory
  DATALAYOUT_DESCENDING  //!< Mode N is fastest varying, Mode 0 is slowest varying in memory
};

template<class SparseValue, class ElemIdx, class SubIdx>
class SparseTensor;

//! Sparse Tensor Class
template<class SparseValue, class ElemIdx, class SubIdx>
class SparseTensor
{
public:

  class Iterator {
  public:
    KOKKOS_FUNCTION Iterator(type::ElemIdx position) : _position(position) {;}

    KOKKOS_FUNCTION Iterator operator++() { ++_position; return *this; }
    KOKKOS_FUNCTION type::ElemIdx operator*() { return _position; }
    KOKKOS_FUNCTION bool operator!=(Iterator const &other) { return _position != other._position; }
    // 2018-11-28: Do we need the following operators?
    //KOKKOS_FUNCTION Iterator operator++(int32_t /*not used*/) { ++_position; return *this; }
    //KOKKOS_FUNCTION bool operator==(Iterator const &other) { return _position == other._position; }

  private:
    type::ElemIdx _position;
  };

  SparseTensor(SubIdx nDim, ElemIdx nElement, std::vector<SubIdx> const &sizeDim);                  //!< Constructor, force user to specify N-ways and fixed allocation size (RAII)
  SparseTensor(SparseTensor<SparseValue, ElemIdx, SubIdx> const &source);                           //!< Copy constructor, performs deep copy
  SparseTensor();
  //SparseTensor() = delete;                                                                          //!< Disallow default constructor (insufficient information provided by user)
  virtual ~SparseTensor();                                                                          //!< Destructor (RAII)

  virtual SparseValue get_element_data(ElemIdx elemId) const;                                       //!< Get data directly by array elemId
  virtual ElemIdx get_element_index(ElemIdx elemId, SubIdx indiceId) const;                         //!< Get indices directly by array elemId

  virtual void set_element_data(ElemIdx elemId, SparseValue value);                                 //!< Set data value directly by array elemId
  virtual void set_element_index(ElemIdx elemId, SubIdx indiceId, SubIdx iVal);                     //!< Set indices directly through the array elemId

  virtual void set_element(ElemIdx elemId, std::vector<SubIdx> const &indices, SparseValue value);  //!< Set data and indices directly though the array elemId

  virtual void set_element_data_to_zero();                                                          //!< Reset entire data array (Kokkos::View) to zeros

  virtual SparseData<SparseValue> get_data_view();                                                  //!< Returns shallow copy of data view
  virtual SparseData<SparseValue> get_data_view() const;                                            //!< Returns shallow copy of data view

  virtual typename SparseData<SparseValue>::HostMirror get_data_host_view();                                                  //!< Returns shallow copy of data view
  virtual typename SparseData<SparseValue>::HostMirror get_data_host_view() const;                                            //!< Returns shallow copy of data view

  virtual SparseIndices<SubIdx> get_indices_view();                                                 //!< Returns shallow copy of indices view
  virtual SparseIndices<SubIdx> get_indices_view() const;                                           //!< Returns shallow copy of indices view
  virtual typename SparseIndices<SubIdx>::HostMirror get_indices_host_view();                       //!< Returns shallow copy of indices view
  virtual typename SparseIndices<SubIdx>::HostMirror get_indices_host_view() const;                 //!< Returns shallow copy of indices view

  virtual SubIdx get_nRow(SubIdx iDim) const;                                                       //!< Retuns the number of rows
  virtual SubIdx get_nDim() const;                                                                  //!< Returns the dimension (order) of the sparse tensor
  virtual Dimension get_dim() const;                                                                //!< Returns the number of dimensions
  virtual ElemIdx get_nElement() const;                                                             //!< Returns the number of elements in the tensor
  virtual std::vector<SubIdx> const &get_size_dim() const;                                          //!< Returns a std::vector containing the size of each dimension

  virtual std::string show();
  virtual void copy_to_device();                          //!< Copy the host data into device when CUDA is enabled
  virtual void copy_to_host();                            //!< Copy the device data back to the host 
  Iterator begin() { return Iterator(0); }
  Iterator end() { return Iterator(_nElement); }

private:
  SubIdx _nDim;                                             //!< Number of dimensions (indiceId.e., this is an _nDim-way or N-way tensor)
  Dimension *_dim;
  ElemIdx _nElement;                                        //!< Maximum number to allocate for non-zero entries in sparse tensor
  std::vector<SubIdx> const _sizeDim;                       //!< Stores the size of each dimension
  SparseIndices<SubIdx> _indices;                           //!< Indices object to store elemId sets for non-zero elements in an N-way sparse tensor (_nDim number of indices in this object)
  SparseData<SparseValue> _data;                            //!< Non-zero values are stored in this array

#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
 // Kokkos::View<SubIdx **, Kokkos::HostSpace>  _indices_host; //!< GPU version needs a storage in hostspace.
  typename SparseIndices<SubIdx>::HostMirror  _indices_host; //!< GPU version needs a storage in hostspace.
  typename SparseData<SparseValue>::HostMirror  _data_host;  //!< GPU version needs a storage in hostspace.
 // Kokkos::View<SparseValue *, Kokkos::HostSpace>  _data_host;  //!< GPU version needs a storage in hostspace.
#endif

};

} // sparten
