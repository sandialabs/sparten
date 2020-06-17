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
 \brief Collection of functions for a SparTen reference blas executing on the host/KNL/GPU architectures
 */

#pragma once

#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdexcept>


namespace sparten
{

namespace spartenBlas
{

  // This is a column major Cholesky factorization applied to the lower triangular part.
  // The leading dimension size must be equalt to the column size. (LDA=N)
  template<class SubIdx, class MatView>
  KOKKOS_INLINE_FUNCTION
  bool posv (SubIdx n, MatView a, MatView x)
  {
    // Cholesky factorziation on lower triangular matrix
    int N = (int)n;
    for (int k = 0; k < N; ++k)
    {
      for (int j = 0; j < k; ++j)
      {
        for (int i = k; i < N; ++i)
        {
          a(k * N + i) = a(k * N + i) - a(j * N + k) * a(j * N + i);
        }
      }



      const typename MatView::value_type lkk = sqrt(a(k * N + k));
      if (lkk <= 0 )
      {
        return false;
      }
      a(k * N + k) = lkk;

      for(int j = k + 1; j < n; ++j)
      {
        a(k * N + j) = a(k * N + j) / lkk;
      }
    }

    // Triangular solve (Lower)
    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < i; ++j)
      {
        x(i) = x(i) - a(j * N + i) * x(j);
      }

      x(i) = x(i) / a(i * N + i);
    }

    // Triangular solve (Upper)
    for (int i = N - 1; i >= 0; --i)
    {
      for (int j = i + 1; j < N; ++j)
      {

        x(i) = x(i) - a(i * N + j) * x(j);
      }

      x(i) = x(i) / a(i * N + i);
    }
    return true;
  }

  template<class SubIdx, class SpartenScalar, class MatView>
  KOKKOS_INLINE_FUNCTION
  void gemv( char trans, SubIdx m, SubIdx n, SpartenScalar alpha,
             MatView a, SubIdx lda, MatView x,
             SubIdx incx, SpartenScalar beta, MatView y,
             SubIdx incy)
  {

    // Need to improve the parallelism (ThreadVectorRange)
        if (trans == 'N')
        {
                for (SubIdx i = 0; i < m; ++i )
                {
                        SpartenScalar tmp = 0;
                        for (SubIdx j = 0; j < n; ++j )
                        {
                                tmp += a(i + (j * lda)) * x(j * incx);
                        }
                        y(i * incy) = alpha * tmp + beta * y(i * incy);
                }

        }
        else if (trans == 'T')
        {
                for (SubIdx i = 0; i < n; ++i )
                {
                        SpartenScalar tmp = 0;
                        for (SubIdx j = 0; j < m; ++j )
                        {
                                tmp += a(j + (i * lda)) * x(j * incx);
                        }
                        y(i * incy) = alpha * tmp + beta * y(i * incy);
                }
        }
  }

  template<class SubIdx, class SpartenScalar, class MatView>
  KOKKOS_INLINE_FUNCTION
  void gemm(char transa, char transb, SubIdx m, SubIdx n, SubIdx k,
            SpartenScalar alpha, MatView a, SubIdx lda,
            MatView b, SubIdx ldb, SpartenScalar beta,
            MatView c, SubIdx ldc)
  {


  //    if ((transa == 'N') && (transb == 'N'))
  //    {
      // Not implemented as SparTen uses T-N case only
  //    }
  //    else if ((transa == 'T') && (transb == 'N'))
  //    {
      // Form  C := alpha*A^T*B + beta*C
      for (SubIdx j = 0; j < n; ++j )
      {
        for( SubIdx i = 0; i < m; ++i )
        {
          SpartenScalar temp = 0.0;
          for( SubIdx l = 0; l < k; ++l )
          {
            temp += a(l+i*lda) * b(l+j*ldb);
          }
          c(i+j*ldc) = alpha*temp + beta*c(i+j*ldc);
        }
      }
  //    }
  //    else
  //    {
                // Not implemented
  //    }
  }

  template<class SubIdx, class MatView>
  KOKKOS_INLINE_FUNCTION
  void trmv( char uplo, char trans, char diag, SubIdx n, MatView a,
             SubIdx lda, MatView x, SubIdx incx )
  {

      int ldap1 = (int)lda + 1;
      int N = (int)n;

      if(uplo == 'L' && trans == 'T' && diag == 'N' )
      {
          for(int j = 0, jaj = 0, jx  = 0; j < N; j++, jaj += ldap1, jx += incx)
          {
              typename MatView::value_type t0 = x(jx);
              t0 *= a(jaj);
              for(SubIdx i = j+1, iaij = jaj+1, ix = jx + (int)incx; i < n; i++, iaij += 1, ix += (int)incx)
              {
                  t0 += a(iaij) * x(ix);
              }
              x(jx) = t0;
          }
      }
      else if(uplo == 'L' && trans == 'N' && diag == 'N' )
      {
          for(int j = N-1, jaj = (N-1)*(ldap1), jx = (N-1)*(int)incx; j >= 0; j--, jaj -= ldap1, jx -= (int)incx )
          {
              typename MatView::value_type t0 = x(jx);
              x(jx) *= a(jaj);
              for(int i = j+1, iaij = jaj+1, ix = jx + (int)incx; i < n; i++, iaij += 1, ix += (int)incx)
              {
                  x(ix) += a(iaij) * t0;
              }
          }
      }
      else
      {
                // Not implemented as SparTen does not need this

      }
  }

} // spartenBlas

} // sparten

