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
 \brief Class definition for a test set generator
 */

#include <generator/DiscreteCumulativeDistribution.h>
#include <generator/FactTestSetGenerator.h>
#include <generator/RandGenMersenneTwister.h>
#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <algorithm>
#include <map>
#include <sstream>
#include <string>

namespace sparten
{
template<class SubIdx>
bool compare_int_arrays(std::vector<SubIdx> const &parg1, std::vector<SubIdx> const &parg2)
{
  if (parg1.size() != parg2.size())
  {
    return false;
  }

  for (SubIdx i = 0; i < parg1.size(); ++i)
  {
    if (parg1[i] < parg2[i])
    {
      return true;
    } else if (parg1[i] > parg2[i])
    {
      return false;
    }
  }

  return false;
}

template<class KruskalValue, class SubIdx>
bool sortablePair_compare(SortablePair<KruskalValue, SubIdx> const &a, SortablePair<KruskalValue, SubIdx> const &b)
{
  return a.first < b.first;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::FactTestSetGenerator() :
        _cDims(1),
        _nNumComponents(0),
        _nMaxNnz(0),
        _nSeed(1),
        _dFracBoosted(1.0),
        _bBoosted(false),
        _dMaxValue(1.0),
        _dSmallValue(1.0),
        _cRNG(1)
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::FactTestSetGenerator(SubIdx nNumComponents, ElemIdx nMaxNonz, uint64_t nSeed, KruskalValue dFracBoosted,
    KruskalValue dMaxValue, KruskalValue dSmallValue) :
        _nNumComponents(nNumComponents),
        _nMaxNnz(nMaxNonz),
        _nSeed(nSeed),
        _dFracBoosted(dFracBoosted),
        _bBoosted(true),
        _dMaxValue(dMaxValue),
        _dSmallValue(dSmallValue),
        _cRNG(1)
{
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::~FactTestSetGenerator()
{
  return;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_FactoredTensorDims(std::vector<SubIdx> v)
{
  _cDims = v;
  nKEY_LENGTH = _cDims.size();
  return true;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_NumComponents(SubIdx nNumComponents)
{
  _nNumComponents = nNumComponents;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_MaxNnz(ElemIdx nMaxNonz)
{
  _nMaxNnz = nMaxNonz;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_RandomSeed(uint64_t nSeed)
{
  _nSeed = nSeed;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_FracBoosted(KruskalValue dFracBoosted)
{
  _dFracBoosted = dFracBoosted;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_MaxValue(KruskalValue dMaxValue)
{
  _dMaxValue = dMaxValue;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_SmallValue(KruskalValue dSmallValue)
{
  _dSmallValue = dSmallValue;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
std::vector<SubIdx> FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_FactoredTensorDims()
{
  return _cDims;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_NumComponents()
{
  return _nNumComponents;

}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
ElemIdx FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_MaxNnz()
{
  return _nMaxNnz;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
uint64_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_RandomSeed()
{
  return _nSeed;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_FracBoosted()
{
  return _dFracBoosted;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_MaxValue()
{
  return _dMaxValue;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalValue FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_SmallValue()
{
  return _dSmallValue;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::set_Parameters(SubIdx nNumComponents, ElemIdx nMaxNnz, uint64_t nSeed, KruskalValue dFracBoosted,
    KruskalValue dMaxValue, KruskalValue dSmallValue)
{
  _nNumComponents = nNumComponents;
  _nMaxNnz = nMaxNnz;
  _nSeed = nSeed;
  _dFracBoosted = dFracBoosted;
  _bBoosted = true;
  _dMaxValue = dMaxValue;
  _dSmallValue = dSmallValue;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
int32_t FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_Parameters(SubIdx &nNumComponents, ElemIdx &nMaxNnz, uint64_t &nSeed,
    KruskalValue &dFracBoosted, KruskalValue &dMaxValue, KruskalValue &dSmallValue)
{
  nNumComponents = _nNumComponents;
  nMaxNnz = _nMaxNnz;
  nSeed = _nSeed;
  dFracBoosted = _dFracBoosted;
  // bBoosted =   _bBoosted ;
  dMaxValue = _dMaxValue;
  dSmallValue = _dSmallValue;
  return 0;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::drawNonzeroElements(KruskalTensor<KruskalValue, SubIdx> *cExpectedFactors)
{
  // Use probabilities in the stochastic factors to choose nMaxNnz sparse
  // tensor elements.  Matlab code first decides how many are in each
  // component, based on the weights as probabilities.  Then for each
  // component, the Matlab code chooses elements in each dimension.
  // Duplicates are counted.

  // Reset random number genrator
  _cRNG.reset(_nSeed);

  // Build a histogram for component probabilities.
  DiscreteCumulativeDistribution<KruskalValue, SubIdx> cCompProbs;
  if (cCompProbs.load(cExpectedFactors->get_weight_vector()) == false)
  {
    std::cout << "*** Failed to load CDF for weights" << std::endl;
    return false;
  }

  std::vector<SubIdx> naCompCounts(_nNumComponents);

  for (SubIdx c = 0; c < _nNumComponents; ++c)
  {
    naCompCounts[c] = 0;
  }

  // Sample and total the number of nonzeroes to draw from each component.
  for (SubIdx i = 0; i < _nMaxNnz; ++i)
  {
    KruskalValue dNextRan = _cRNG.generate_rand_double_matlab();
    ++naCompCounts[cCompProbs.get_random_sample(dNextRan)];
  }

  int32_t nLargestCompCount = 0;
  for (SubIdx c = 0; c < _nNumComponents; ++c)
  {
    if (naCompCounts[c] > nLargestCompCount)
      nLargestCompCount = naCompCounts[c];
  }

  // Build a CDF for each factor vector, then sample it for nonzero indices.
  // Count the nonzeroes in a set, incrementing when duplicates are found.
  std::vector<DiscreteCumulativeDistribution<KruskalValue, SubIdx>> pCDFs(_cDims.size());
  nKEY_LENGTH = _cDims.size();

  std::map<std::vector<SubIdx>, SubIdx, bool (*)(std::vector<SubIdx> const &, std::vector<SubIdx> const &)> cIndexCounts(compare_int_arrays<SubIdx>);
  typedef typename std::map<std::vector<SubIdx>, SubIdx, bool (*)(std::vector<SubIdx> const &, std::vector<SubIdx> const &)>::iterator MyIterator;
  MyIterator it;

  std::vector<std::vector<SubIdx> > pTempIndices(_cDims.size(), std::vector<SubIdx> (nLargestCompCount));

  // Loop ordering matches Matlab generate_data_sparse.
  for (SubIdx c = 0; c < _nNumComponents; ++c)
  {
    for (SubIdx n = 0; n < _cDims.size(); ++n)
    {
      if (pCDFs[n].load(cExpectedFactors->get_factor_matrix(n), c) == false)
      {
        std::cout << "*** Failed to load CDF for component " << c << ", dim " << n << std::endl;
        return false;
      }
    }

    // Could avoid storing all the indices by reordering these two
    // loops, but have to match the Matlab code since random numbers
    // must be generated in the same sequence.
    for (SubIdx n = 0; n < _cDims.size(); ++n)
    {
      for (int32_t i = 0; i < naCompCounts[c]; ++i)
      {
        KruskalValue dNextRan = _cRNG.generate_rand_double_matlab();
        pTempIndices[n][i] = static_cast<SubIdx>(pCDFs[n].get_random_sample(dNextRan));
      }
    }

    // Store the new nonzeroes, counting duplicates.
    // I suspect this is what makes execution slow for large factors.
    // Making the key into a string is definitely slower.
    for (int32_t i = 0; i < naCompCounts[c]; ++i)
    {
      std::vector<SubIdx> naNextKey(nKEY_LENGTH);
      for (SubIdx n = 0; n < _cDims.size(); ++n)
      {
        naNextKey[n] = pTempIndices[n][i];
      }

      it = cIndexCounts.find(naNextKey);

      if (it == cIndexCounts.end())
      {
        cIndexCounts.insert(std::pair<std::vector<SubIdx>, int32_t>(naNextKey, 1));
      } else  // Found the same index set
      {
        (*it).second += 1;
      }
    }
  }

  // Form the sparse data tensor from the index counts.
  _cSpTensor = new SparseTensor<SparseValue, ElemIdx, SubIdx>(static_cast<SubIdx>(_cDims.size()), static_cast<ElemIdx>(cIndexCounts.size()), _cDims);
  SubIdx nNextIndex = 0;
  for (it = cIndexCounts.begin(); it != cIndexCounts.end(); ++it)
  {
    _cSpTensor->set_element(nNextIndex, (*it).first,  (*it).second);
    ++nNextIndex;
  }
  return true;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalTensor<KruskalValue, SubIdx>* FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::genNoisedNonNegativeKruscalTensor( KruskalValue dThres, KruskalValue dNoise )
{

  // Reset random number genrator
  _cRNG.reset(_nSeed);

  KruskalTensor<KruskalValue, SubIdx> *myFactor = new KruskalTensor<KruskalValue, SubIdx>(_cDims.size(), _nNumComponents, _cDims);
  std::vector<KruskalValue> weights(_nNumComponents);

  for (SubIdx n = 0; n < _cDims.size(); ++n)
  {
    // Create the initial random matrix except the last component
    for (SubIdx r = 0; r < _nNumComponents-1; ++r)
    {
      for (SubIdx i = 0; i < _cDims[n]; ++i)
      {
        myFactor->set_factor_matrix_element( _cRNG.generate_rand_double_matlab(), n, i, r);
    //    std::cout << "VAl = " <<  myFactor->get_factor_matrix_element( n, i, r) << std::endl;
      }
    }

    // Sparsify the matrix
    KruskalValue dThresFinal;
    for (SubIdx r = 0; r < _nNumComponents-1; ++r)
    {
      KruskalValue dMin = 1.0;
      for (SubIdx i = 0; i < _cDims[n]; ++i)
      {
        if( myFactor->get_factor_matrix_element( n, i, r) < dMin )
        {
          dMin =  myFactor->get_factor_matrix_element( n, i, r);
        }
      }
    //  std::cout << "dMin  "  << dMin << std::endl;
      if (dMin > dThres )
      {
        dThresFinal = dMin;
      } else
      {
        dThresFinal = dThres;
      }
     // std::cout << "dThresFinal  "  << dThresFinal<< std::endl;
      for (SubIdx i = 0; i < _cDims[n]; ++i)
      {
        if( dThresFinal >= myFactor->get_factor_matrix_element( n, i, r)  )
        {
          myFactor->set_factor_matrix_element( 1, n, i, r );
        } else {
          myFactor->set_factor_matrix_element( 0, n, i, r );
        }
      }
    }

    // Insert unifrom noise to the last component
    for (SubIdx i = 0; i < _cDims[n]; ++i)
    {
      myFactor->set_factor_matrix_element( dNoise, n, i, _nNumComponents-1 ) ;
   //   std::cout << "VAl ( " << n << " , " << i << " ,  " << _nNumComponents-1 << " ) =" << myFactor->get_factor_matrix_element( n, i, _nNumComponents-1 ) << std::endl;
    }
  }

  // Choose random component weights and normalize to make the factor matrix completely stochastic.
  for (SubIdx r = 0; r < _nNumComponents; ++r)
  {
    weights[r] = 1;
  }

  myFactor->set_weight_vector(weights);
  myFactor->normalize();          // Use One Norm
  for (SubIdx r = 0 ; r < _nNumComponents; r++ ) {
    weights[r] = myFactor->get_weight_element( r );
  //  std::cout << "Labmda " << weights[r] <<std::endl;
  }
  KruskalValue dTotalWeight = 0.0;
  for (SubIdx r = 0; r < _nNumComponents; ++r)
  {
    dTotalWeight += weights[r];
  }
  for (SubIdx r = 0; r < _nNumComponents; ++r)
  {
    weights[r] = weights[r] / dTotalWeight;
 //   std::cout << "Labmda after eta" << weights[r] <<std::endl;
  }
  myFactor->set_weight_vector(weights);


  return myFactor;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx>* FactTestSetGenerator<SparseValue, KruskalValue, ElemIdx, SubIdx>::genSparseTensor( KruskalTensor<KruskalValue, SubIdx> *myFactor )
{
  std::vector<KruskalValue> weights(_nNumComponents);
  for (SubIdx r = 0 ; r < _nNumComponents; r++ ) {
    weights[r] = myFactor->get_weight_element( r );
  }

  if (drawNonzeroElements(myFactor) == false)
  {
    return nullptr;
  }

  // Rescale the weights so the expected factors sum to the target number of samples.
  for (SubIdx r = 0; r < _nNumComponents; ++r)
  {
    weights[r] *= static_cast<KruskalValue>(_nMaxNnz);

  }
  myFactor->set_weight_vector(weights);

  return _cSpTensor;
}

// Explicit instantiation
template class FactTestSetGenerator<sparten::type::SubIdx, float, sparten::type::SubIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::SubIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx> ;

template class FactTestSetGenerator<sparten::type::ElemIdx, float, sparten::type::SubIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::ElemIdx, float, sparten::type::ElemIdx, sparten::type::ElemIdx> ;

template class FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::SubIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx> ;

template class FactTestSetGenerator<sparten::type::ElemIdx, double, sparten::type::SubIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> ;
template class FactTestSetGenerator<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::ElemIdx> ;

} // end namespace sparten