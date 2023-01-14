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
 \brief Sparten utilities
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <memory>

//! Namespace for Sparten project
namespace sparten
{

//! Namespace for Sparten datatypes
namespace type
{

using ElemIdx = uint64_t;         //!< This is a 64-bit unsigned integer
using SubIdx = uint32_t;          //!< This is a 32-bit unsigned integer

}

// Typedef's to assist compilation for host, GPU+UVM, and GPU (no UVM, explicit transfers)
// If building with CUDA (no UVM)
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  template <typename T> using Vector = Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>;
  template <typename T> using VectorAtomic = Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using VectorConst = Kokkos::View<T const *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using Array2D = Kokkos::View<T **,  Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>;
  template <typename T> using Array2DAtomic = Kokkos::View<T **, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using Array2DConst = Kokkos::View<T const **, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using RaggedArray2D = Kokkos::View< Vector<T> *, Kokkos::HostSpace >;
  template <typename T> using RaggedArray2DHost = Kokkos::View< typename Vector<T>::HostMirror *, Kokkos::HostSpace >;
  template <typename T> using RaggedArray2DAtomic = Kokkos::View<Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>> *, Kokkos::Device<Kokkos::Cuda,Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using RaggedArray2DConst = Kokkos::View<Kokkos::View<T const *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>> const *, Kokkos::Device<Kokkos::Cuda,Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using Array3D = Kokkos::View<T ***,  Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>;
  template <typename T> using Array3DAtomic = Kokkos::View<T ***, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using Array3DConst = Kokkos::View<T const ***, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

#endif


// If building with CUDA UVM
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_USE_CUDA_UVM)
  template <typename T> using Vector = Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>>;
  template <typename T> using VectorAtomic = Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using VectorConst = Kokkos::View<T const *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using Array2D = Kokkos::View<T **, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>>;
  template <typename T> using Array2DAtomic = Kokkos::View<T **, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using Array2DConst = Kokkos::View<T const **, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using RaggedArray2D = Kokkos::View<Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>> *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>>;
  template <typename T> using RaggedArray2DAtomic = Kokkos::View<Kokkos::View<T *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>> *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using RaggedArray2DConst = Kokkos::View<Kokkos::View<T const *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>> const *, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;


  template <typename T> using Array3D = Kokkos::View<T ***, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>>;
  template <typename T> using Array3DAtomic = Kokkos::View<T ***, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using Array3DConst = Kokkos::View<T const ***, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#endif

// If building for host
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  template <typename T> using Vector = Kokkos::View<T *, Kokkos::HostSpace>;
  template <typename T> using VectorAtomic = Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using VectorConst = Kokkos::View<T const *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using Array2D = Kokkos::View<T **, Kokkos::HostSpace>;
  template <typename T> using Array2DAtomic = Kokkos::View<T **, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using Array2DConst = Kokkos::View<T const **, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using RaggedArray2D = Kokkos::View<Kokkos::View<T *, Kokkos::HostSpace> *, Kokkos::HostSpace>;
  template <typename T> using RaggedArray2DAtomic = Kokkos::View<Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using RaggedArray2DConst = Kokkos::View<Kokkos::View<T const *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>> const *, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <typename T> using Array3D = Kokkos::View<T ***, Kokkos::HostSpace>;
  template <typename T> using Array3DAtomic = Kokkos::View<T ***, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  template <typename T> using Array3DConst = Kokkos::View<T const ***, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#endif

  // "Derived Types" - Built from primitives to reduce code bloat
  template <typename T> using SparseIndices = Array2D<T>;
  template <typename T> using SparseIndicesAtomic = Array2DAtomic<T>;
  template <typename T> using SparseIndicesConst = Array2DConst<T>;

  template <typename T> using SparseData = Vector<T>;
  template <typename T> using SparseDataAtomic = VectorAtomic<T>;
  template <typename T> using SparseDataConst = VectorConst<T>;

  template <typename T> using FactorMatrix = Array2D<T>;
  template <typename T> using FactorMatrixAtomic = Array2DAtomic<T>;
  template <typename T> using FactorMatrixConst = Array2DConst<T>;

  template <typename T> using FactorMatrices = Kokkos::View<Array2D<T> *, Kokkos::HostSpace>;
  template <typename T> using FactorMatricesHost = Kokkos::View<typename Array2D<T>::HostMirror *, Kokkos::HostSpace>;
  template <typename T> using FactorMatricesAtomic = Kokkos::View<Array2DAtomic<T>, Kokkos::HostSpace>;
  template <typename T> using FactorMatricesConst =  Kokkos::View<Array2D<T> *, Kokkos::HostSpace>;  ///\todo make this const again (needs work from kokkos dev team)

  template <typename T> using KruskalWeights = Vector<T>;
  template <typename T> using KruskalWeightsAtomic = VectorAtomic<T>;
  template <typename T> using KruskalWeightsConst = VectorConst<T>;

  template <typename T> using NonZeroLocations = RaggedArray2D<T>;

#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  template <typename T> using NonZeroLocationsHost = RaggedArray2DHost<T>;
#endif
  template <typename T> using NonZeroLocationsAtomic = RaggedArray2DAtomic<T>;
  template <typename T> using NonZeroLocationsConst = RaggedArray2DConst<T>;

// Functions to assist working on host/device
template<typename T, typename U>
KOKKOS_FORCEINLINE_FUNCTION
typename std::common_type<T, U>::type
min(T const &a, U const &b)
{
  return a < b ? a : b;
}

template<typename T, typename U>
KOKKOS_FORCEINLINE_FUNCTION
typename std::common_type<T, U>::type
max(T const &a, U const &b)
{
  return a > b ? a : b;
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION T abs(T const &a)
{
  return a > 0 ? a : -a;
}

// isolate uses of `new` to this one method, when possible
template <class T>
std::unique_ptr<T[]>
make_unique_array(size_t n) {
  return std::unique_ptr<T[]>(new T[n]);
}

template <class TeamPolicy, class ExecSpace>
TeamPolicy get_team_policy(ExecSpace const&, size_t n_iter) {
  return TeamPolicy(n_iter,  Kokkos::AUTO);
}

#ifdef KOKKOS_ENABLE_CUDA
template <class TeamPolicy>
TeamPolicy get_team_policy(Kokkos::Cuda const&, size_t n_iter) {
  return TeamPolicy(n_iter,1,32 );
 // Kokkos::AUTO);
}
#endif


template <class TeamPolicy>
TeamPolicy get_team_policy(size_t n_iter) {
  return
    sparten::get_team_policy<TeamPolicy>(
      typename TeamPolicy::execution_space{}, n_iter
    );
}

template <class TeamPolicy>
TeamPolicy get_team_policy_bigrow(size_t n_iter) {
   return TeamPolicy(n_iter, Kokkos::AUTO);
}

#ifdef KOKKOS_ENABLE_CUDA
   template <class TeamPolicy>
TeamPolicy get_team_policy_bigrow(Kokkos::Cuda const&, size_t n_iter) {
   return TeamPolicy(n_iter,Kokkos::AUTO);
}
#endif

KOKKOS_INLINE_FUNCTION double DLog( double x  )
{
#ifdef KOKKOS_ENABLE_CUDA
   return log(x);
#else
   return std::log(x);
#endif
}

template <class T>
  struct numeric_limits {
    static constexpr auto max = std::numeric_limits<T>::max();
    static constexpr auto min = std::numeric_limits<T>::min();
    static constexpr auto infinity = std::numeric_limits<T>::infinity();
    static constexpr auto epsilon = std::numeric_limits<T>::epsilon();
  };
} // sparten
