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

#pragma once

#include <Kokkos_Core.hpp>


namespace sparten {

// TODO use Kokkos for this so that we get `memcpy`s and `memset`s when we can

template <
  class TeamMember,
  class DstView1D,
  class Value
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<DstView1D>::value
    && not Kokkos::is_view<Value>::value
    && DstView1D::Rank == 1
>::type
deep_copy(
  TeamMember& member,
  DstView1D const& dst,
  Value const& src_val
)
{
  Kokkos::parallel_for(
    Kokkos::ThreadVectorRange(member, dst.extent(0)),
    [&] (int i)
    {
      dst[i] = src_val;
    }
  );
}

template <
  class TeamMember,
  class DstView1D,
  class SrcView1D
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<DstView1D>::value
    && Kokkos::is_view<SrcView1D>::value
    && DstView1D::Rank == 1
>::type
deep_copy(
  TeamMember& member,
  DstView1D const& dst,
  SrcView1D const& src
)
{
  Kokkos::parallel_for(
    Kokkos::ThreadVectorRange(member, dst.extent(0)),
    [&] (int i)
    {
      dst[i] = src[i];
    }
  );
}

template <
  class TeamMember,
  class DstView2D,
  class Value
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<DstView2D>::value
    && not Kokkos::is_view<Value>::value
    && DstView2D::Rank == 2
    && std::is_same<typename DstView2D::array_layout, Kokkos::LayoutLeft>::value
>::type
deep_copy(
  TeamMember& member,
  DstView2D const& dst,
  Value const& src_val
)
{
  for(auto j = 0; j < dst.extent_int(1); ++j) {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(member, dst.extent(0)),
      [&] (int i)
      {
        dst(i, j) = src_val;
      }
    );
  }
}

template <
  class TeamMember,
  class DstView2D,
  class Value
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<DstView2D>::value
    && not Kokkos::is_view<Value>::value
    && DstView2D::Rank == 2
    && not std::is_same<typename DstView2D::array_layout, Kokkos::LayoutLeft>::value
>::type
deep_copy(
  TeamMember& member,
  DstView2D const& dst,
  Value const& src_val
)
{
  for(auto i = 0; i < dst.extent_int(0); ++i) {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(member, dst.extent(1)),
      [&] (int j)
      {
        dst(i, j) = src_val;
      }
    );
  }
}


template <
  class LeftView1D,
  class RightView1D
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<LeftView1D>::value
    && Kokkos::is_view<RightView1D>::value,
  decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
>::type
dot_product(
  LeftView1D const& left,
  RightView1D const& right
) {
  using return_type =
    decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
  using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
  return_type rv = 0;
  // TODO assert matching sizes

  for(index_type i = 0; i < left.extent(0); ++i) {
    rv += left[i] * right[i];
  }

  return rv;
}

template <
  class LeftView1D,
  class RightView1D,
    class IndexView1D,
  class IndexType
>
typename std::enable_if<
  Kokkos::is_view<LeftView1D>::value
    && Kokkos::is_view<RightView1D>::value,
  decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
>::type
dot_product_top(
  LeftView1D const& left,
  RightView1D const& right,
  IndexView1D const& idx,
  IndexType const num
) {
   using return_type = decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
   using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
   return_type rv = 0;
#if 0

  for(index_type i = 0; i <num; ++i) {
    rv += left[idx[i]] * right[idx[i]];
  }
#else
   Kokkos::parallel_reduce( num, KOKKOS_LAMBDA( index_type i, return_type &local_rv ) {
      local_rv += left[idx[i]] * right[idx[i]];
   },rv);
#endif
  return rv;
}


template <
  class TeamMember,
  class LeftView1D,
  class RightView1D
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<LeftView1D>::value
    && Kokkos::is_view<RightView1D>::value,
  decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
>::type
dot_product(
  TeamMember& member,
  LeftView1D const& left,
  RightView1D const& right
) {
  using return_type =
    decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
  using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
  return_type rv = 0;
  // TODO assert matching sizes
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(member, left.extent(0)),
    [&] (index_type i, return_type &reduce_val) {
      reduce_val += left[i] * right[i];
    },
    Kokkos::Sum<return_type>(rv)
  );
  return rv;
}

template <
  class TeamMember,
  class LeftView1D,
  class RightView1D,
  class IdxType
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<LeftView1D>::value
         && Kokkos::is_view<RightView1D>::value,
   decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
   >::type
   dot_product(
         TeamMember& member,
         LeftView1D const& left,
         RightView1D const& right,
         IdxType const num
)
{
  using return_type =
    decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
  using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
  return_type rv = 0;

  // TODO assert matching sizes
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(member, num),
    [&] (index_type i, return_type &reduce_val) {
      reduce_val += left[i] * right[i];
    },
    Kokkos::Sum<return_type>(rv)
  );
  return rv;
}

template <
  class TeamMember,
  class LeftView1D,
  class RightView1D,
  class IndexView1D,
  class IdxType
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<LeftView1D>::value
    && Kokkos::is_view<RightView1D>::value,
  decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
>::type
dot_product(
  TeamMember& member,
  LeftView1D const& left,
  RightView1D const& right,
  IndexView1D const& idx,
  IdxType const num
) {
  using return_type =
    decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
  using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
  return_type rv = 0;
  // TODO assert matching sizes
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(member, num),
    [&] (index_type i, return_type &reduce_val) {
      reduce_val += left[idx[i]] * right[idx[i]];
    },
    Kokkos::Sum<return_type>(rv)
  );
  return rv;
}

template <
         class LeftView1D,
         class RightView1D,
         class IndexView1D,
         class IdxType>
typename std::enable_if<
         Kokkos::is_view<LeftView1D>::value
         && Kokkos::is_view<RightView1D>::value,
         decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
   >::type
dot_product_left_with_index(
   LeftView1D const& left,
   RightView1D const& right,
   IndexView1D const& idx,
   IdxType const num )
{
   using return_type = decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
   using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
   return_type rv = 0;

   // TODO assert matching sizes
   Kokkos::parallel_reduce( num, KOKKOS_LAMBDA (index_type i, return_type &reduce_val) {
      reduce_val += left[idx[i]] * right[i];
   }, Kokkos::Sum<return_type>(rv) );
   return rv;
}

template <
  class TeamMember,
  class LeftView1D,
  class RightView1D,
  class IndexView1D,
  class IdxType
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<LeftView1D>::value
    && Kokkos::is_view<RightView1D>::value,
  decltype(typename LeftView1D::value_type() * typename RightView1D::value_type())
>::type
dot_product_left_with_index(
  TeamMember& member,
  LeftView1D const& left,
  RightView1D const& right,
  IndexView1D const& idx,
  IdxType const num
) {
  using return_type =
    decltype(typename LeftView1D::value_type() * typename RightView1D::value_type());
  using index_type = typename std::make_unsigned<typename LeftView1D::size_type>::type;
  return_type rv = 0;
  // TODO assert matching sizes
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(member, num),
    [&] (index_type i, return_type &reduce_val) {
      reduce_val += left[idx[i]] * right[i];
    },
    Kokkos::Sum<return_type>(rv)
  );
  return rv;
}

template <
  class TeamMember,
  class View1D
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<View1D>::value
    && View1D::Rank == 1,
  typename View1D::value_type
>::type
sum_of_elements(
  TeamMember& member,
  View1D const& data
)
{
  using return_type = typename View1D::value_type;
  return_type rv = return_type();
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(member, data.extent(0)),
    [&] (int i, return_type& total)
    {
      total += data[i];
    },
    rv
  );
  return rv;
}


template <
  class TeamMember,
  class View1D,
  class UnaryPredicate
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<View1D>::value
    && View1D::Rank == 1,
  bool
>::type
any_of(
  TeamMember& member,
  View1D const& data,
  UnaryPredicate&& predicate
)
{
  // TODO change this to use bool once Kokkos has reductions over booleans
  int result = int(false);
  // TODO This should really be a short-circuiting reduction, which Kokkos doesn't have
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(member, data.extent(0)),
    [&] (int i, int& result) {
      result = result || predicate(data[i]);
    },
    Kokkos::LOr<int>(result)
  );
  return bool(result);
}


template <
  class TeamMember,
  class View1D,
  class Value
>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Kokkos::is_view<View1D>::value
    && not Kokkos::is_view<Value>::value
    && View1D::Rank == 1,
  void
>::type
scale(
  TeamMember& member,
  View1D const& data,
  Value value
)
{
  Kokkos::parallel_for(
    Kokkos::ThreadVectorRange(member, data.extent(0)),
    [&] (int i) {
      data[i] *= value;
    }
  );
}

} // end namespace sparten
