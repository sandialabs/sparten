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
#include <sparten/Log.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <type_traits> // std::is_floating_point

namespace sparten
{

/** Check for NaNs the old-fashioned way.
 *
 * @note Like the standard library analog, this function
 *       cannot be marked `noexcept` because operations are
 *       performed that would cause signaling NaNs to throw
 *       exceptions.
 *
 * @todo Move this to some sort of utility header file
 *
 * @tparam Floating A floating point type (requires
 *                  `std::is_floating_point_v<Floating>`)
 * @param f Number to check
 * @return true if `f` is a (quiet) NaN
 */
template <class Floating>
KOKKOS_FORCEINLINE_FUNCTION
bool is_nan(Floating const& f) {
  static_assert(
    std::is_floating_point<Floating>::value,
    "sparten::is_nan only works on floating point types"
  );
  // The following reference claims this should always work:
  //   https://en.cppreference.com/w/cpp/numeric/math/isnan
  // (This is unusal enough that we need to disable the linter here)
  return f != f; // NOLINT(misc-redundant-expression)
}

template<class KruskalValue, class SubIdx>
KruskalTensor<KruskalValue, SubIdx>::KruskalTensor(SubIdx nDim, SubIdx nComponents, std::vector<SubIdx> const &nPerMode) :
    _nDim(nDim),
    _dim(new Dimension(nDim)),
    _nComponent(nComponents),
    _nPerMode(nPerMode),
    _weights("_weights", nComponents),
    _weightsBuffer("_weightsBuffer", nComponents),
    _factorMatrices(nDim),
    _factorMatricesBuffer(nDim),
    _factorMatrices_host(nDim),
    _factorMatricesBuffer_host(nDim),
    _permutation("_factorMatricesWorkSpace", nComponents),
    _norms("_norms", nComponents),
    _atomicCount("atomicCounter", 1)
{

  // Non-device code or UVM do nothing for host Views
  for (auto iDim : *_dim)
  {
    // Initialize _factorMatrices
    Kokkos::resize(_factorMatrices[iDim],_nPerMode[iDim], _nComponent);
    _factorMatrices_host[iDim] = Kokkos::create_mirror_view(_factorMatrices[iDim]);
    Kokkos::resize(_factorMatricesBuffer[iDim],_nPerMode[iDim], _nComponent);
    _factorMatricesBuffer_host[iDim] = Kokkos::create_mirror_view(_factorMatricesBuffer[iDim]);
  }
  _weights_host = Kokkos::create_mirror_view( _weights );
  _weightsBuffer_host = Kokkos::create_mirror_view( _weightsBuffer );
  _permutation_host = Kokkos::create_mirror_view( _permutation );
  _norms_host = Kokkos::create_mirror_view( _norms );
}

template<class KruskalValue, class SubIdx>
KruskalTensor<KruskalValue, SubIdx>::KruskalTensor(KruskalTensor<KruskalValue, SubIdx> const &src) :
        _nDim(src.get_nDim()),
        _dim(new Dimension(src.get_nDim())),
        _nComponent(src.get_nComponent()),
        _nPerMode(src.get_nPerMode()),
        _weights("_weights", src.get_nComponent()),
        _weightsBuffer("_weightsBuffer", src.get_nComponent()),
        _factorMatrices( src.get_nDim()),
        _factorMatricesBuffer(src.get_nDim()),
        _factorMatrices_host(src.get_nDim()),
        _factorMatricesBuffer_host(src.get_nDim()),
        _permutation("_factorMatricesWorkSpace", src.get_nComponent()),
        _norms("_norms", src.get_nComponent()),
        _atomicCount("atomicCounter", 1)
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor:::KruskalTensor(KruskalTensor<KruskalValue, SubIdx> const &src)", Log::DEBUG_3);


  for (auto iDim : *_dim)
  {

    Kokkos::resize(_factorMatrices[iDim],_nPerMode[iDim], _nComponent);
    Kokkos::deep_copy ( _factorMatrices[iDim], src.get_factor_matrix(iDim) );
    _factorMatrices_host[iDim] = Kokkos::create_mirror_view(_factorMatrices[iDim]);


    Kokkos::resize(_factorMatricesBuffer[iDim],_nPerMode[iDim], _nComponent);
    _factorMatricesBuffer_host[iDim] = Kokkos::create_mirror_view(_factorMatricesBuffer[iDim]);

  }

  Kokkos::deep_copy(_weights, src.get_weight_vector());
  _weights_host = Kokkos::create_mirror_view( _weights );
  _weightsBuffer_host = Kokkos::create_mirror_view( _weightsBuffer );
  _permutation_host = Kokkos::create_mirror_view( _permutation );
  _norms_host = Kokkos::create_mirror_view( _norms );
  Kokkos::fence();
}

template<class KruskalValue, class SubIdx>
KruskalTensor<KruskalValue, SubIdx>::~KruskalTensor()
{
  delete _dim;
}

template<class KruskalValue, class SubIdx>
Kokkos::View<KruskalValue **>  KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix(SubIdx dim)
{
  return _factorMatrices[dim];
}


template<class KruskalValue, class SubIdx>
Kokkos::View<KruskalValue **> KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix(SubIdx dim)  const
{
   return _factorMatrices[dim];
}


template<class KruskalValue, class SubIdx>
Kokkos::View<KruskalValue **>  KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_buffer(SubIdx dim)
{
  return _factorMatricesBuffer[dim];
}

template<class KruskalValue, class SubIdx>
std::vector<Kokkos::View<KruskalValue **>>  KruskalTensor<KruskalValue, SubIdx>::get_factor_matrices() const
{
  return _factorMatrices;
}

template<class KruskalValue, class SubIdx>
Kokkos::View<KruskalValue *>  KruskalTensor<KruskalValue, SubIdx>::get_weight_vector()
{
  return _weights;
}

template<class KruskalValue, class SubIdx>Kokkos::View<KruskalValue *> KruskalTensor<KruskalValue, SubIdx>::get_weight_vector() const
{
  return _weights;
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::set_weight_vector(std::vector<KruskalValue> const &src)
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor::set_weight_vector", Log::DEBUG_3);

  if (_nComponent != src.size())
  {
    throw std::runtime_error("Mismatch between nComponents in std::vector and KruskalTensor");
  }

  for (SubIdx iComp = 0; iComp < _nComponent; ++iComp)
  {
    _weights_host(iComp) = src[iComp];
  }
  Kokkos::deep_copy(_weights,_weights_host);
}

template<class KruskalValue, class SubIdx>
KruskalValue KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_element(SubIdx dim, SubIdx iMode, SubIdx iComp) const
{
  return _factorMatrices[dim](iMode, iComp);
}

template<class KruskalValue, class SubIdx>
KruskalValue KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_element_host(SubIdx dim, SubIdx iMode, SubIdx iComp) const
{

  return _factorMatrices_host[dim](iMode, iComp);

}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::set_factor_matrix_element(KruskalValue value, SubIdx dim, SubIdx iMode, SubIdx iComp)
{

  _factorMatrices_host[dim](iMode, iComp) = value;

//  _factorMatrices[dim](iMode, iComp) = value;

}

template<class KruskalValue, class SubIdx>
KruskalValue KruskalTensor<KruskalValue, SubIdx>::get_weight_element(SubIdx elemId) const
{
  return _weights(elemId);
}

template<class KruskalValue, class SubIdx>
KruskalValue KruskalTensor<KruskalValue, SubIdx>::get_weight_element_host(SubIdx elemId) const
{

  return _weights_host(elemId);
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::set_weight_element(KruskalValue weight, SubIdx elemId)
{
  _weights(elemId) = weight;
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::set_factor_matrix_row(std::vector<KruskalValue> const &rowValues, SubIdx dim, SubIdx iMode)
{
  auto factorMatrices = _factorMatrices;
  for (SubIdx iComp = 0; iComp < get_factor_matrix_nColumn(dim); ++iComp)
  {
    const auto tmpRowVal = rowValues[iComp];
    factorMatrices[dim](iMode, iComp) = tmpRowVal;
  }
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::set_factor_matrix_column(std::vector<KruskalValue> const &columnValues, SubIdx dim, SubIdx iComp)
{
  auto factorMatrices = _factorMatrices;
  for (SubIdx iMode = 0; iMode < get_factor_matrix_nRow(dim); ++iMode)
  {
    const auto tmpColVal = columnValues[iMode];
    factorMatrices[dim](iMode, iComp) = tmpColVal;
  }
}

template<class KruskalValue, class SubIdx>
bool KruskalTensor<KruskalValue, SubIdx>::compare(KruskalTensor<KruskalValue, SubIdx> const &other, KruskalValue tolerance) const
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor::compare", Log::DEBUG_3);

  
  bool nComponentsEqual = true;
  if (_nComponent != other.get_nComponent())
  {
    return false;
  }

  bool nDimEqual = true;
  if (_nDim != other.get_nDim())
  {
    return false;
  }

  bool nPerModeEqual = true;
  for (SubIdx iDim = 0; iDim < _nDim; ++iDim)
  {
    if (_nPerMode[iDim] != other.get_nPerMode(iDim))
    {
      nPerModeEqual = false;
    }
  }

  if ( !nPerModeEqual )
  {
    return false;
  }
  // Compare weights
  const auto teamSize = Kokkos::AUTO;
  auto weights = _weights;
  const auto dTestWeights = other.get_weight_vector();
  SubIdx dataNotEqual = 0;

  Kokkos::parallel_reduce("compare_ktensor_weights", Kokkos::RangePolicy<>( 0, _nComponent),
      KOKKOS_LAMBDA(SubIdx iComp, SubIdx& lcount) {
      lcount += static_cast<SubIdx>( fabs( weights(iComp) - dTestWeights(iComp) ) > tolerance);   
  }, dataNotEqual );

  Kokkos::fence();

  if( dataNotEqual != 0 ) 
  {
    return false;
  } 
  // Compare Values in FactorMatrices
  bool dataEqual = true;
  SubIdx dataUnequalCount = 0;


  for (auto iDim : *_dim)
  {
    SubIdx dataUnequalCountDim = 0;
    const auto nRows = get_factor_matrix_nRow(iDim);
    const auto nCols = get_factor_matrix_nColumn(iDim);
    const auto otherFactorMatrix = other.get_factor_matrices()[iDim];
    const auto thisFactorMatrix = this->get_factor_matrices()[iDim];
    Kokkos::parallel_reduce ("compare_ktensor_factors",Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type const &team, SubIdx &lcount)
    {
      const auto iMode = team.league_rank()*team.team_size()+team.team_rank();
      if( iMode >= nRows) return;
      SubIdx tcount = 0;
      Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(team, nCols), [&] (SubIdx iComp, SubIdx &llcount)
      {
        llcount += static_cast<SubIdx>(fabs(otherFactorMatrix(iMode, iComp) - thisFactorMatrix(iMode, iComp)) > tolerance);
      }, tcount);
      lcount += tcount;


    }, dataUnequalCountDim);

    if( dataUnequalCountDim != 0 ) 
    {
      dataUnequalCount++;
    }
  }
   

#if 0
  auto atomicCount = _atomicCount;
  atomicCount(0) = 0;

  for (auto iDim : *_dim)
  {
    const auto nRows = get_factor_matrix_nRow(iDim);
    const auto nCols = get_factor_matrix_nColumn(iDim);
    const auto otherFactorMatrix = other.get_factor_matrices()(iDim);
    const auto thisFactorMatrix = this->get_factor_matrices()(iDim);

    Kokkos::parallel_for (Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
    {
      const auto iMode = team.league_rank()*team.team_size()+team.team_rank();
      if( iMode >= nRows) return;
      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, nCols), [=] (SubIdx iComp)
      {
        atomicCount(0) += static_cast<SubIdx>(fabs(otherFactorMatrix(iMode, iComp) - thisFactorMatrix(iMode, iComp)) > tolerance);
      });
    });

    Kokkos::fence();

    dataUnequalCount = atomicCount(0);
  }
#endif

  dataEqual = dataUnequalCount == 0;
  return nComponentsEqual && nDimEqual && nPerModeEqual && dataEqual;
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::normalize()
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor::normalize", Log::DEBUG_3);

  auto norms = _norms;
  auto nComponent = _nComponent;
  auto weights = _weights;

  // We could have _cComponents x _nDim space for keeping 1-norm for every single column
  for (auto iDim : *_dim)
  {
    auto factorMatrix = get_factor_matrix(iDim);
    auto nRows = get_factor_matrix_nRow(iDim);
    auto nCols = get_factor_matrix_nColumn(iDim);
    Kokkos::parallel_for("normalize_ktensor_init",_nComponent, KOKKOS_LAMBDA (SubIdx iComp)
    {
      norms(iComp) = 0;
    });

    // Compute norm
    const auto teamSize = Kokkos::AUTO;  // TeamPolicy configuration needs special tuning.
                                         // For small nCols, set large teamSize (especially for GPUs)
    Kokkos::parallel_for ("normalize_ktensor_norms", Kokkos::TeamPolicy<>(nCols, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
    {
      const auto iComp = team.league_rank();

      KruskalValue tnorm = 0;
      Kokkos::parallel_reduce (Kokkos::TeamThreadRange(team, nRows), [&] (SubIdx iMode, KruskalValue &llnorm)
      {
        llnorm += factorMatrix(iMode, iComp);
      }, tnorm);
      Kokkos::single(Kokkos::PerTeam( team ), [&] ()
      {
          norms(iComp) = tnorm;
      });
    });

    Kokkos::parallel_for("normalize_ktensor_remove_zero_norms", nComponent, KOKKOS_LAMBDA (SubIdx iComp)
    {
      if (norms(iComp) == 0)
      {
        norms(iComp) = 1.0;
      }
    });

    // Scale
    Kokkos::parallel_for ("normalize_ktensor_scale_factors", Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
    {
      const auto iMode = team.league_rank()*team.team_size()+team.team_rank();
      if( iMode >= nRows ) return;
      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, nCols), [&] (SubIdx iComp)
      {
        factorMatrix(iMode, iComp) = factorMatrix(iMode, iComp) / norms(iComp);
      });
    });

    // Scale weight accordingly
    Kokkos::parallel_for ("normalize_ktensor_scale_weights", nComponent, KOKKOS_LAMBDA (SubIdx iComp)
    {
      weights(iComp) *= norms(iComp);
    });
  }
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::normalize(SubIdx iDim)
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor::normalize(SubIdx iDim)", Log::DEBUG_3);

  auto norms = _norms;
  auto weights = _weights;
  auto factorMatrices = _factorMatrices;

  // We could have _cComponents x _nDim space for keeping 1-norm for every single column
  Kokkos::parallel_for ("normalize_ktensor_single_mode_init",_nComponent, KOKKOS_LAMBDA (SubIdx iComp)
  {
    norms(iComp) = 0;
  });

  // Compute norm
  const auto nRows = get_factor_matrix_nRow(iDim);
  const auto nCols = get_factor_matrix_nColumn(iDim);
  const auto teamSize = Kokkos::AUTO;  // TeamPolicy configuration needs special tuning.
                                       // For small nCols, set large teamSize (especially for GPUs)
  auto kData = factorMatrices[iDim];

  Kokkos::parallel_for ("normalize_ktensor_single_mode_norms",Kokkos::TeamPolicy<>(nCols, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
  {
    const auto iComp = team.league_rank();
    KruskalValue tnorm = 0;
    Kokkos::parallel_reduce (Kokkos::TeamThreadRange(team, nRows), [&] (SubIdx iMode, KruskalValue &llnorm)
    {
      llnorm += kData(iMode, iComp);
    }, tnorm);
    Kokkos::single(Kokkos::PerTeam( team ), [&] ()
    {
       norms(iComp) = tnorm;
    });
  });

  Kokkos::parallel_for ("normalize_ktensor_single_mode_remove_zero_norms",_nComponent, KOKKOS_LAMBDA (SubIdx iComp)
  {
    if (norms(iComp) == 0)
    {
      norms(iComp) = 1.0;
    }
  });

  // Scale
  Kokkos::parallel_for ("normalize_ktensor_single_mode_scale_factors",Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
  {
    const auto iMode = team.league_rank()*team.team_size() + team.team_rank();
    if( iMode >= nRows ) return;
    Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, nCols), [&] (SubIdx iComp)
    {
      kData(iMode, iComp) = kData(iMode, iComp) / norms(iComp);
    });
  });

  // Scale weight accordingly
  Kokkos::parallel_for ("normalize_ktensor_single_mode_scale_weights",_nComponent, KOKKOS_LAMBDA (SubIdx iComp)
  {
    weights(iComp) *= norms(iComp);
  });
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_nComponent() const
{
  return _nComponent;
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_nDim() const
{
  return _nDim;
}

template<class KruskalValue, class SubIdx>
Dimension KruskalTensor<KruskalValue, SubIdx>::get_dim() const
{
  return *_dim;
}

template<class KruskalValue, class SubIdx>
std::vector<SubIdx> const &KruskalTensor<KruskalValue, SubIdx>::get_nPerMode() const
{
  return _nPerMode;
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_nPerMode(SubIdx mode) const
{
  return _nPerMode[mode];
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_nRow(SubIdx dim) const
{
  const auto factorMatrices = _factorMatrices;

  return factorMatrices[dim].extent(0);
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_nRow_host(SubIdx dim) const
{
  const auto factorMatrices = _factorMatrices_host;
  return factorMatrices[dim].extent(0);
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_max_mode_size() const
{
  SubIdx myMax = 0;
  for ( SubIdx i = 0;  i < _nDim; ++i ) 
  {
    myMax = (myMax < _nPerMode[i] ) ? _nPerMode[i] : myMax;
  }
  return myMax;
}

/// \todo Parallelize this
template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::distribute_weights_to_factor_matrices()
{
  auto weights = _weights;
  auto factorMatrices = _factorMatrices;
  for (auto iDim : *_dim)
  {
    const auto factorMatrixnColumn = get_factor_matrix_nColumn(iDim);
    auto kData = factorMatrices[iDim];
    const auto teamSize = Kokkos::AUTO;
    const auto nRows = get_factor_matrix_nRow(iDim);
    Kokkos::parallel_for ("distribute_ktensor_weights", Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
    {
      const auto iMode = team.team_size()*team.league_rank()+team.team_rank();
      if ( iMode >= nRows ) return;
      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, factorMatrixnColumn), [=] (SubIdx iComp)
      {
        kData(iMode, iComp) *= weights(iComp);
      });
    });
  }

  Kokkos::deep_copy(weights,1.0);
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::distribute_weights_to_factor_matrices(SubIdx iDim, bool resetWeightstOnEachDim)
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor::distribute_weights_to_factor_matrices(SubIdx iDim)", Log::DEBUG_3);

  const auto factorMatrixnColumn = get_factor_matrix_nColumn(iDim);

  auto weights = _weights;
  auto factorMatrices = _factorMatrices;

  const auto teamSize = Kokkos::AUTO;
  auto kData = factorMatrices[iDim];
  const auto nRows = get_factor_matrix_nRow(iDim);
  Kokkos::parallel_for ("distribute_ktensor_weights_single_mode",Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
  {
     const auto iMode = team.team_size()*team.league_rank()+team.team_rank();
     if ( iMode >= nRows ) return;
    Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, factorMatrixnColumn), [=] (SubIdx iComp)
    {
      kData(iMode, iComp) *= weights(iComp);
    });
  });

  if (iDim == _nDim - 1 || resetWeightstOnEachDim)
  {
    Kokkos::deep_copy(weights,1.0);
  }
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::permute_factor_matrix_columns()
{
  Log &log = Log::new_log();
  log.print("Entering KruskalTensor::permute_factor_matrix_columns()", Log::DEBUG_3);
  auto permutation = _permutation;
  auto weights = _weights;
  auto weightsBuffer = _weightsBuffer;
  auto factorMatrices = _factorMatrices;
  auto factorMatricesBuffer = _factorMatricesBuffer;

  Kokkos::parallel_for ("permute_ktensor_factor_columns_init",_nComponent, KOKKOS_LAMBDA (SubIdx iComp)
  {
    permutation(iComp) = iComp;
  });
  log.print("\tInitialized permutation list.", Log::DEBUG_3);

  // Permutation will work on CPU/KNL/GPU
  //
  Kokkos::deep_copy( _permutation_host, permutation ); 
  class Permutation
  {
  public:
    KOKKOS_INLINE_FUNCTION Permutation(typename Kokkos::View<KruskalValue *>::HostMirror weights) : _weights(weights) {}
    KOKKOS_INLINE_FUNCTION bool operator()(SubIdx i, SubIdx j) const { return _weights(i) > _weights(j); }
  protected:
      typename Kokkos::View<KruskalValue *>::HostMirror _weights;
  };
  
  // Sort Weights
  Kokkos::deep_copy( _weights_host, weights ); 
#if 0
  std::cout << "Sorting now \n";
  for( int iComp = 0; iComp < _nComponent; ++iComp )
  {
     std::cout << _permutation_host(iComp) << "  " << _weights_host(iComp) << std::endl;
  }
#endif
  std::sort( _permutation_host.data(), _permutation_host.data() + _nComponent, Permutation(_weights_host));

  // Do permutation on weights, can be parallelized
  for ( int iComp = 0; iComp < _nComponent; ++iComp ) 
  {
    _weightsBuffer_host(iComp) = _weights_host(_permutation_host(iComp));
  }

  log.print("\tPerformed sort.", Log::DEBUG_3);
  
  log.print("\tPerformed permutation on weights.", Log::DEBUG_3);

  // Swap weights (These are shallow copies only.)
  auto weightsSwap = _weights_host;
  _weights_host = _weightsBuffer_host;
  _weightsBuffer_host = weightsSwap;

  Kokkos::deep_copy( weights, _weights_host ); 
  Kokkos::deep_copy( _permutation, _permutation_host ); 
  // Permute columns
  for (auto iDim : *_dim)
  {
    const auto nCols = get_factor_matrix_nColumn(iDim);
    const auto teamSize = Kokkos::AUTO;
    const auto nRows = get_factor_matrix_nRow(iDim);
    auto factorMatrix = get_factor_matrix(iDim);
    auto factorMatrixBuff =  get_factor_matrix_buffer(iDim);
   
    Kokkos::parallel_for ("permute_ktensor_factor_columns", Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
    {
      const auto iMode = team.team_size()*team.league_rank()+team.team_rank();
      if ( iMode >= nRows ) return;
      Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, nCols), [=] (SubIdx iComp)
      {
        factorMatrixBuff(iMode, iComp) = factorMatrix(iMode, permutation(iComp));
      });
    });
    Kokkos::fence();
  }
  log.print("\tPerformed permutation on columns.", Log::DEBUG_3);

  // Swap factor matrices
  for( auto iDim: *_dim ) {
     auto matrixswap = _factorMatricesBuffer[iDim];
     _factorMatricesBuffer[iDim] = _factorMatrices[iDim];
     _factorMatrices[iDim] = matrixswap;
  }
}


template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_nColumn(SubIdx dim) const
{
  const auto factorMatrices = _factorMatrices;

  return factorMatrices[dim].extent(1);
}

template<class KruskalValue, class SubIdx>
SubIdx KruskalTensor<KruskalValue, SubIdx>::get_factor_matrix_nColumn_host(SubIdx dim) const
{

  const auto factorMatrices = _factorMatrices_host;
  return factorMatrices[dim].extent(1);
}


template<class KruskalValue, class SubIdx>
bool KruskalTensor<KruskalValue, SubIdx>::check_for_nans() const {
  Log &log = Log::new_log();
  log.print("Start NaN Check:", Log::DEBUG_3);

  SubIdx nanFoundInDim = 0;
  SubIdx nanFoundTotal = 0;

  const auto teamSize = Kokkos::AUTO;
  for (auto iDim : *_dim)
  {
    const auto nRows = get_factor_matrix_nRow(iDim);
    const auto nCols = get_factor_matrix_nColumn(iDim);
    const auto thisFactorMatrix = this->get_factor_matrix(iDim);
    Kokkos::parallel_reduce (Kokkos::TeamPolicy<>(nRows, teamSize), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team, SubIdx &lcount)
    {
      const auto iMode = team.league_rank()*team.team_size()+team.team_rank();
      if ( iMode >= nRows ) return;
      SubIdx tlcount = 0;
      Kokkos::parallel_reduce (Kokkos::ThreadVectorRange(team, nCols), [=] (SubIdx iComp, SubIdx &llcount)
      {
	llcount += sparten::is_nan(thisFactorMatrix(iMode, iComp)) ? 1 : 0;
        //llcount = static_cast<SubIdx>(thisFactorMatrix(iMode, iComp) != thisFactorMatrix(iMode, iComp)); ///\TODO: There are more rigorous ways to check for this...
      }, tlcount);
      lcount += tlcount;
    }, nanFoundInDim);
    nanFoundTotal += nanFoundInDim;
  }

  Kokkos::fence();

  log.print("End NaN Check:", Log::DEBUG_3);

  return nanFoundTotal != 0;
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::move_to_device()
{

  Kokkos::deep_copy( _weights, _weights_host);
  for ( SubIdx i = 0; i <  _nDim; ++i )
  {
    Kokkos::deep_copy( _factorMatrices[i], _factorMatrices_host[i]);
  }

}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::move_to_device() const
{

  Kokkos::deep_copy( _weights, _weights_host);
  for ( SubIdx i = 0; i <  _nDim; ++i )
  {
    Kokkos::deep_copy( _factorMatrices[i], _factorMatrices_host[i]);
  }
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::move_to_host()
{

  Kokkos::deep_copy( _weights_host, _weights);
  for ( SubIdx i = 0; i <  _nDim; ++i )
  {
    Kokkos::deep_copy( _factorMatrices_host[i], _factorMatrices[i]);
  }
}

template<class KruskalValue, class SubIdx>
void KruskalTensor<KruskalValue, SubIdx>::move_to_host() const
{

  Kokkos::deep_copy( _weights_host, _weights);
  for ( SubIdx i = 0; i <  _nDim; ++i )
  {
    Kokkos::deep_copy( _factorMatrices_host[i], _factorMatrices[i]);
  }

}

template<class KruskalValue, class SubIdx>
std::string KruskalTensor<KruskalValue, SubIdx>::show()
{
  std::stringstream ss;

  move_to_host();
  ss << "Weights = ";
  for (SubIdx iComp = 0; iComp < _nComponent; ++iComp )
  {
    ss << std::scientific << std::setprecision(16) << _weights_host(iComp) << " ";
  }
  ss << std::endl;

  for (SubIdx iMode = 0; iMode < _nDim; ++iMode )
  {
    ss << "Mode " << iMode << std::endl;
    for (SubIdx iRow = 0; iRow < _nPerMode[iMode]; ++iRow)
    {
      //ss << "Mode: " << iMode << " Row: " << iRow;
      for (SubIdx iComp = 0; iComp < _nComponent; ++iComp)
      {
        ss <<  " " <<  std::scientific << std::setprecision(16) << _factorMatrices_host[iMode](iRow, iComp);
      }
      ss << std::endl;
    }
  }


  return ss.str();
}

template<class KruskalValue, class SubIdx>
std::string KruskalTensor<KruskalValue, SubIdx>::show_by_mode( SubIdx iMode )
{
  std::stringstream ss;

  move_to_host();
  ss << "Mode " << iMode << std::endl;
  for (SubIdx iRow = 0; iRow < _nPerMode[iMode]; ++iRow)
  {
    //ss << "Mode: " << iMode << " Row: " << iRow;
    for (SubIdx iComp = 0; iComp < _nComponent; ++iComp)
    {
      ss <<  " " <<  std::scientific << std::setprecision(16) << _factorMatrices_host[iMode](iRow, iComp);
    }
    ss << std::endl;
  }


  return ss.str();
}

// Explicit instantiation
template class KruskalTensor<float, type::SubIdx>;
template class KruskalTensor<float, type::ElemIdx>;

template class KruskalTensor<double, type::SubIdx>;
template class KruskalTensor<double, type::ElemIdx>;

} // sparten
