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
 \brief Class definition for the CDF of a discrete random variable
 */

#include <generator/DiscreteCumulativeDistribution.h>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>

namespace sparten
{

template<class KruskalValue, class SubIdx>
DiscreteCumulativeDistribution<KruskalValue, SubIdx>::DiscreteCumulativeDistribution()
{
}

template<class KruskalValue, class SubIdx>
DiscreteCumulativeDistribution<KruskalValue, SubIdx>::~DiscreteCumulativeDistribution()
{
}

/*!
 *  @param[in] cPDF     The vector is a probability density function
 *                      histogram: all values are in [0,1) and sum to 1.
 *  @return  False if an argument is illegal.
 */
template<class KruskalValue, class SubIdx>
bool DiscreteCumulativeDistribution<KruskalValue, SubIdx>::load(const VectorConst<KruskalValue> &cPDF)
{
  // Imports the lambda entries (Kokkos::View) of KruskalTensor
  // _cCDF.resize(cPDF.extent(0));
  Kokkos::resize(_cCDF, cPDF.extent(0));

  if (cPDF.extent(0) == 1)
  {
    _cCDF(0) = 1.0;
    return true;
  }

  for (SubIdx i = 0; i < cPDF.extent(0); i++)
  {
    KruskalValue dNext = cPDF(i);
    if ((dNext < 0.0) || (dNext > 1.0))
    {
      //std::cout << "*** Bad input to DiscreteCumulativeDistribution.load:  (" << i << ") = " << dNext << "\n";
      return false;
    }
    if (i == 0)
    {
      _cCDF(i) = dNext;
    } else
    {
      _cCDF(i) = dNext + _cCDF(i - 1);
    }
        //  std::cout << "V(" << i << ") = " << _cCDF(i) << std::endl;
  }

  KruskalValue dTotal = _cCDF(_cCDF.extent(0) - 1);
  double dErr =  (double)fabs((double)dTotal - 1.0);

  if ( dErr > 1000 * std::numeric_limits<KruskalValue>::epsilon())
  {
    //std::cout << "*** Bad input to DiscreteCumulativeDistribution.load: " << " sums to " << std::setw(24) << std::setprecision(16) << dTotal << " instead of 1 (error "
    //    << dErr << " )." << " Tolerance = " << 1000 * std::numeric_limits<KruskalValue>::epsilon() << std::endl;
    return false;
  }

  return true;
}

/*!
 *  @param[in] cPDF     Each column is a probability density function
 *                      histogram: all values are in [0,1) and sum to 1.
 *  @param[in] nColumn  Column index.
 *  @return  False if an argument is illegal.
 */
template<class KruskalValue, class SubIdx>
bool DiscreteCumulativeDistribution<KruskalValue, SubIdx>::load(const Array2DConst<KruskalValue> &cPDF, const int32_t nColumnIndex)
{
  Kokkos::resize(_cCDF, cPDF.extent(0));  // Dimension_0 = Number of rows = Size of the column

  for (SubIdx r = 0; r < cPDF.extent(0); ++r)
  {
    double dNext = cPDF(r, nColumnIndex);
    if ((dNext < 0.0) || (dNext > 1.0))
    {
      //std::cout << "*** Bad input to DiscreteCumulativeDistribution.load:  (" << r << "," << nColumnIndex << ") = " << dNext << "\n";
      return false;
    }
    if (r == 0)
      _cCDF(r) = dNext;
    else
      _cCDF(r) = dNext + _cCDF(r - 1);

    //std::cout << "V(" << r << ") = " << _cCDF(r) << std::endl;
  }

  double dTotal = _cCDF(_cCDF.extent(0) - 1);
  if (fabs(dTotal - 1.0) > 1000 * std::numeric_limits<KruskalValue>::epsilon())
  {
    //std::cout << "*** Bad input to DiscreteCumulativeDistribution.load: sums to " << std::setw(24) << std::setprecision(16) << dTotal << " instead of 1 (error " << std::fabs(dTotal - 1.0) << ")" << " Tolerance = " << 1000 * std::numeric_limits<KruskalValue>::epsilon() << std::endl;
    return false;
  }

  return true;
}

/*!
 *  Use the random number to select a bin of the CDF, and return the
 *  index of the bin, starting from zero.  The method encapsulates
 *  a search algorithm with worst case log N retrieval time.
 *
 *  The random number is passed so the caller can control random number
 *  generation.  Generally, it is from a uniform distribution.
 *
 *  @param[in] dRandomNumber  Number in [0,1).
 *  @return  Value of the discrete random variable, starting with 0.
 */
template<class KruskalValue, class SubIdx>
SubIdx DiscreteCumulativeDistribution<KruskalValue, SubIdx>::get_random_sample(KruskalValue dRandomNumber)
{
  const SubIdx nMAXLEN_FULLSEARCH = 16;

  // If the histogram is short, then just walk thru it to find the index.
  if (_cCDF.extent(0) < nMAXLEN_FULLSEARCH)
  {
    for (SubIdx i = 0; i < _cCDF.extent(0); i++)
    {
      if (dRandomNumber < _cCDF(i))
        return (i);
    }
    return (_cCDF.extent(0) - 1);
  }

  // For a longer histogram, use a binary search.
  SubIdx nStart = 0;
  SubIdx nEnd = _cCDF.extent(0) - 1;
  while (true)
  {
    if ((nEnd - nStart) <= 1)
    {
      if (dRandomNumber < _cCDF(nStart))
        return (nStart);
      else
        return (nEnd);
    }
    SubIdx nMiddle = (nStart + nEnd) / 2;
    if (dRandomNumber < _cCDF(nMiddle))
      nEnd = nMiddle;
    else
      nStart = nMiddle;
  }
}

// Explicit instantiation
template class DiscreteCumulativeDistribution<double, int32_t> ;
template class DiscreteCumulativeDistribution<double, type::SubIdx> ;
template class DiscreteCumulativeDistribution<double, type::ElemIdx> ;
template class DiscreteCumulativeDistribution<float, type::ElemIdx> ;
template class DiscreteCumulativeDistribution<float, type::SubIdx> ;
}
