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
 \brief Class declaration for Dimension
 */

#pragma once

#include <sparten/spartenUtils.h>

namespace sparten {

//! Dimension object to contain number of dimensions and provide and iterator
class Dimension {
public:
  //! Iterator for Dimension
  class Iterator {
  public:
    KOKKOS_FUNCTION Iterator(type::SubIdx position) : _position(position) {;}                         //!< Iterator constructor

    KOKKOS_FUNCTION Iterator operator++() { ++_position; return *this; }                              //!< Increment operator
    KOKKOS_FUNCTION Iterator operator++(int32_t /*not used*/) { ++_position; return *this; }          //!< Increment operator
    KOKKOS_FUNCTION type::SubIdx operator*() { return _position; }                                    //!< Dereferencing converts Iterator to integer
    KOKKOS_FUNCTION bool operator==(Iterator const &other) { return _position == other._position; }   //!< Check for equality
    KOKKOS_FUNCTION bool operator!=(Iterator const &other) { return _position != other._position; }   //!< Check for inequality

  private:
    type::SubIdx _position;                           //!< The current position (index)
  };

  Dimension(type::SubIdx nDim) : _nDim(nDim) {;}      //!< Dimension constructor

  Iterator begin() { return Iterator(0); }            //!< Returns the beginning position
  Iterator end() { return Iterator(_nDim); }          //!< Returns the end position

  type::SubIdx get_nDim() const { return _nDim; }     //!< Returns the number of dimensions

private:
  type::SubIdx _nDim;                                 //!< Private storage for the number of dimensions
};

} // sparten
