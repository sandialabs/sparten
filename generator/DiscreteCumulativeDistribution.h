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
 \brief Class declaration for the CDF of a discrete random variable.
 */

#pragma once 

#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>

namespace sparten
{

//! Computes random samples from a Discrete Cumulative Distribution Function.
/*!
 *  Assemble the probability histogram of a discrete valued random variable,
 *  and provide methods for efficient random sampling.
 */
template<class KruskalValue, class SubIdx>
class DiscreteCumulativeDistribution
{
public:
  DiscreteCumulativeDistribution();                                                     //!< Constructor
  ~DiscreteCumulativeDistribution();                                                    //!< Destructor

  bool load(const VectorConst<KruskalValue> &cPDF);                                     //!< Load with a PDF
  bool load(const Array2DConst<KruskalValue> &cPDF, const int32_t nColumn);             //!< Load with a PDF
  SubIdx get_random_sample(KruskalValue dRandomNumber);                                 //!< Return a value of the discrete random variable

private:
  DiscreteCumulativeDistribution(DiscreteCumulativeDistribution const &);               //!< Disable copy constructor
  DiscreteCumulativeDistribution &operator=(DiscreteCumulativeDistribution const &);    //!< Disable assignment operator

  Vector<KruskalValue> _cCDF;                                                           //!< Cumulative distribution function
};

}
