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
 \brief Class declaration for Kruskal tensor storage
 */

#pragma once

#include <sparten/Dimension.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>

namespace sparten
{

template<class KruskalValue, class SubIdx>
class KruskalTensor;

//! Kruskal Tensor Class
template<class KruskalValue, class SubIdx>
class KruskalTensor
{
public:
  KruskalTensor() = delete;                                                                                         //!< Disallow default constructor (insufficient information provided by user)
  KruskalTensor(SubIdx nDim, SubIdx nComponent, std::vector<SubIdx> const &nPerMode);                               //!< Constructor, force user to specify N-ways and fixed allocation size (RAII)
  KruskalTensor(KruskalTensor<KruskalValue, SubIdx> const &src);                                                    //!< Copy constructor, deep copy
  ~KruskalTensor();                                                                                         //!< Destructor (RAII)

  Kokkos::View<KruskalValue **> get_factor_matrix(SubIdx dim);                                                 //!< Returns Kokkos::View with a shallow copy of the specified factor matrix (read + write)
  Kokkos::View<KruskalValue **> get_factor_matrix(SubIdx dim)  const;                                      //!< Returns Kokkos::View with a shallow copy of the specified factor matrix (read only)
  Kokkos::View<KruskalValue **> get_factor_matrix_buffer(SubIdx dim);                                                 //!< Returns Kokkos::View with a shallow copy of the specified factor matrix (read + write)

  std::vector<Kokkos::View<KruskalValue **>>   get_factor_matrices() const;                                            //!< Return Kokkos::View of factor matrices

  Kokkos::View<KruskalValue *> get_weight_vector();                                                         //!< Return Kokkos::View with a shallow copy of the weights (read + write)
  Kokkos::View<KruskalValue *> get_weight_vector() const;                                              //!< Return Kokkos::View with a shallow copy of the weights (read only)

  void set_weight_vector(std::vector<KruskalValue> const &src);                                             //!< Sets weights from std::vector that contains the weights

  KruskalValue get_factor_matrix_element(SubIdx dim, SubIdx iMode, SubIdx iComp) const;                     //!< Returns a single element of the specified factor matrix (read only)
  KruskalValue get_factor_matrix_element_host(SubIdx dim, SubIdx iMode, SubIdx iComp) const;                     //!< Returns a single element of the specified factor matrix (read only)
  void set_factor_matrix_element(KruskalValue value, SubIdx dim, SubIdx iMode, SubIdx iComp);               //!< Sets a single elemetn of the specified factor matrix

  KruskalValue get_weight_element(SubIdx elemId) const;                                                     //!< Returns a single element of the weight vector
  KruskalValue get_weight_element_host(SubIdx elemId) const;                                                     //!< Returns a single element of the weight vector
  void set_weight_element(KruskalValue weight, SubIdx elemId);                                              //!< Sets a single element of the weight vector

  void set_factor_matrix_row(std::vector<KruskalValue> const &rowValues, SubIdx dim, SubIdx iMode);         //!< Populates factor matrix row from std::vector
  void set_factor_matrix_column(std::vector<KruskalValue> const &columnValues, SubIdx dim, SubIdx iComp);   //!< Populates factor matrix column from std::vector

  void distribute_weights_to_factor_matrices();                                                             //!< Distributes (multiplies) weight values to each factor matrix
  void distribute_weights_to_factor_matrices(SubIdx iDim, bool resetWeightstOnEachDim = true);              //!< Distributes (multiplies) weight values to each factor matrix

  void permute_factor_matrix_columns();                                                                     //!< Permute order of factor matrix columns

  bool compare(KruskalTensor<KruskalValue, SubIdx> const &other, KruskalValue tolerance) const;             //!< Performs comparison with another Kruskal tensor and returns true if they are identical (within a tolerance)
  void normalize();                                                                                         //!< Normalizes the data
  void normalize(SubIdx iDim);                                                                              //!< Normalizes the data
  SubIdx get_nComponent() const;                                                                            //!< Returns the number of components
  SubIdx get_nDim() const;
  Dimension get_dim() const;                                                                                //!< Returns the number of dimensions
  SubIdx  get_max_mode_size() const;                                                                                //!< Returns the number of dimensions
  std::vector<SubIdx> const &get_nPerMode() const;                                                          //!< Returns a std::vector containing n elements per mode (size = nDim)
  SubIdx get_nPerMode(SubIdx mode) const;                                                                   //!< Returns the length of a mode
  SubIdx get_factor_matrix_nRow(SubIdx dim) const;                                                          //!< Returns the number of rows in the specified factor matrix
  SubIdx get_factor_matrix_nRow_host(SubIdx dim) const;                                                          //!< Returns the number of rows in the specified factor matrix
  SubIdx get_factor_matrix_nColumn(SubIdx dim) const;                                                       //!< Return the number of columns in the specified factor matrix
  SubIdx get_factor_matrix_nColumn_host(SubIdx dim) const;                                                       //!< Return the number of columns in the specified factor matrix
  bool check_for_nans() const;                                                                              //!< Checks for NaNs and logs locations of NaNs

  std::string show();                                                                                       //!< Outputs contents of KruskalTensor to string for logger
  std::string show_by_mode( SubIdx i );                                                                                       //!< Outputs contents of KruskalTensor to string for logger
  void move_to_device();
  void move_to_host();
  void move_to_device() const;
  void move_to_host() const;
private:
  Dimension *_dim;                                                      //!< Number of dimensions (i.e., this is an _nDim-way or N-way tensor)
  SubIdx _nDim;                                                         //!< Number of dimensions (i.e., this is an _nDim-way or N-way tensor)
  SubIdx _nComponent;                                                   //!< Number of components in the decomposition
  const std::vector<SubIdx> _nPerMode;                                  //!< Length of each mode

  //std::unique_ptr<Kokkos::View<KruskalValue *>> _factorMatrices;

  std::vector<Kokkos::View<KruskalValue **>> _factorMatrices;

 // Kokkos::View<SubIdx *> _numRows;
 // Kokkos::View<

  std::vector<typename  Kokkos::View<KruskalValue **>::HostMirror> _factorMatrices_host;
  std::vector<Kokkos::View<KruskalValue **>> _factorMatricesBuffer;
  std::vector<typename  Kokkos::View<KruskalValue **>::HostMirror> _factorMatricesBuffer_host;
  Kokkos::View<KruskalValue *> _weights;
  Kokkos::View<KruskalValue *> _weightsBuffer;
  typename Kokkos::View<KruskalValue *>::HostMirror _weights_host;
  typename Kokkos::View<KruskalValue *>::HostMirror _weightsBuffer_host;

  Kokkos::View<KruskalValue *> _norms;
  typename Kokkos::View<KruskalValue *>::HostMirror _norms_host;
  Kokkos::View<SubIdx *> _permutation;
  typename Kokkos::View<SubIdx *>::HostMirror  _permutation_host;

  Kokkos::View<SubIdx *> _atomicCount;
#if 0
  FactorMatrices<KruskalValue> _factorMatrices;                         //!< Factor matrices
  FactorMatrices<KruskalValue> _factorMatricesBuffer;                   //!< Factor matrices buffer
  FactorMatricesHost<KruskalValue>  _factorMatrices_host; //!< GPU version needs a storage in hostspace.
  FactorMatricesHost<KruskalValue>  _factorMatricesBuffer_host;  //!< GPU version needs a storage in hostspace.

  KruskalWeights<KruskalValue> _weights;                                //!< This the weight for each component (lambda vector)
  KruskalWeights<KruskalValue> _weightsBuffer;                          //!< Temporary copy of the weights for parallel permutation
  Vector<SubIdx> _permutation;                                          //!< Temporary storage to hold permutation
  Vector<KruskalValue> _norms;                                          //!< Temporary work space for norm calculations

  VectorAtomic<SubIdx> _atomicCount;                                    //!< This is for a temporary Kokkos bug workaround in KruskalTensor::compare (remove when bug is fixed)


  typename KruskalWeights<KruskalValue>::HostMirror  _weights_host;
  typename KruskalWeights<KruskalValue>::HostMirror  _weightsBuffer_host;
  typename Vector<SubIdx>::HostMirror                _permutation_host;
  typename Vector<KruskalValue>::HostMirror          _norms_host;
#endif
};

} // sparten
