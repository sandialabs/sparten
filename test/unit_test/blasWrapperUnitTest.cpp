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
 \brief
 */

#include <sparten/spartenBlas.h>
#include <unit_test/blasWrapperUnitTest.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace unitTest
{

void blasWrapperUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);
}

void blasWrapperUnitTest::TearDown()
{
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, trmv_double)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    A = [ 1.1234567890123456789 0 0 ]
        [ 1 2 0 ]
        [ 1 2 3 ]
    x = [ 1 2 3 ]'

    trmv => x = A*x  ('N' in second parameter [i.e., NO TRANSPOSE])
    trmv => x = A'*x ('T' in second parameter [i.e., TRANSPOSE])
  */
  sparten::type::SubIdx idxOne = 1;
  const sparten::type::SubIdx n = 3;

  sparten::Vector<double> A("A", n*n);
  sparten::Vector<double> x("x", n);

  typename sparten::Vector<double>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<double>::HostMirror xHost = Kokkos::create_mirror_view(x);

  // initialize A
  for(int32_t i = 0; i < n*n; ++i)
  {
    AHost(i) = 0.0;
  }

  AHost(0) = 1.1234567890123456789;
  AHost(1) = 1.0;
  AHost(2) = 1.0;
  AHost(4) = 2.0;
  AHost(5) = 2.0;
  AHost(8) = 3.0;

  // initialize x
  for(int32_t i = 0; i < n; ++i)
  {
    xHost(i) = 1.0 + i;
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(x,xHost);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::trmv('L','N','N',n,A,n,x,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(xHost,x);

  ASSERT_DOUBLE_EQ(xHost(0), 1.1234567890123456);
  ASSERT_DOUBLE_EQ(xHost(1), 5);
  ASSERT_DOUBLE_EQ(xHost(2), 14);

  // second call, first matrix is transposed
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::trmv('L','T','N',n,A,n,x,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(xHost,x);

  ASSERT_DOUBLE_EQ(xHost(0), 20.262155156777929);
  ASSERT_DOUBLE_EQ(xHost(1), 38);
  ASSERT_DOUBLE_EQ(xHost(2), 42);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, trmv_float)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    A = [ 1.1234567890123456789 0 0 ]
        [ 1 2 0 ]
        [ 1 2 3 ]
    x = [ 1 2 3 ]'

    trmv => x = A*x  ('N' in second parameter [i.e., NO TRANSPOSE])
    trmv => x = A'*x ('T' in second parameter [i.e., TRANSPOSE])
  */
  sparten::type::SubIdx idxOne = 1;
  const sparten::type::SubIdx n = 3;

  sparten::Vector<float> A("A", n*n);
  sparten::Vector<float> x("x", n);

  typename sparten::Vector<float>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<float>::HostMirror xHost = Kokkos::create_mirror_view(x);

  // initialize A
  for(int32_t i = 0; i < n*n; ++i)
  {
    AHost(i) = 0.0;
  }

  AHost(0) = 1.1234567890123456789;
  AHost(1) = 1.0;
  AHost(2) = 1.0;
  AHost(4) = 2.0;
  AHost(5) = 2.0;
  AHost(8) = 3.0;

  // initialize x
  for(int32_t i = 0; i < n; ++i)
  {
    xHost(i) = 1.0 + i;
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(x,xHost);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::trmv('L','N','N',n,A,n,x,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(xHost,x);

  ASSERT_FLOAT_EQ(xHost(0), 1.12345678);
  ASSERT_FLOAT_EQ(xHost(1), 5);
  ASSERT_FLOAT_EQ(xHost(2), 14);

  // second call, first matrix is transposed
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::trmv('L','T','N',n,A,n,x,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(xHost,x);

  ASSERT_FLOAT_EQ(xHost(0), 20.262155156777929);
  ASSERT_FLOAT_EQ(xHost(1), 38);
  ASSERT_FLOAT_EQ(xHost(2), 42);
}


#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, gemv_double)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    A = [ 1.1234567890123456789 0 0 ]
        [ 1 2 0 ]
        [ 1 2 3 ]
    x = [ 1 2 3 ]'
    y = [ 1 1 1 ]'

    gemv => y = A*x + y  ('N' in second parameter [i.e., NO TRANSPOSE])
    gemv => y = A'*x + y ('T' in second parameter [i.e., TRANSPOSE])
  */  
  double tempOne = 1.0;
  sparten::type::SubIdx idxOne = 1;
  const sparten::type::SubIdx n = 3;

  sparten::Vector<double> A("A", n*n);
  sparten::Vector<double> x("x", n);
  sparten::Vector<double> y("y", n);

  typename sparten::Vector<double>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<double>::HostMirror xHost = Kokkos::create_mirror_view(x);
  typename sparten::Vector<double>::HostMirror yHost = Kokkos::create_mirror_view(y);

  // initialize A
  for(int32_t i = 0; i < n*n; ++i)
  {
    AHost(i) = 0.0;
  }
  AHost(0) = 1.1234567890123456789;
  AHost(1) = 1.0;
  AHost(2) = 1.0;
  AHost(4) = 2.0;
  AHost(5) = 2.0;
  AHost(8) = 3.0;

  // initialize x and y
  for(int32_t i = 0; i < n; ++i)
  {
    xHost(i) = 1.0 + i;
    yHost(i) = 1.0;   
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(x,xHost);
  Kokkos::deep_copy(y,yHost);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::gemv('N',n,n,tempOne,A,n,x,idxOne,tempOne,y,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(yHost,y);

  ASSERT_DOUBLE_EQ(yHost(0), 2.1234567890123456);
  ASSERT_DOUBLE_EQ(yHost(1), 6);
  ASSERT_DOUBLE_EQ(yHost(2), 15);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::gemv('T',n,n,tempOne,A,n,x,idxOne,tempOne,y,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(yHost,y);

  ASSERT_DOUBLE_EQ(yHost(0), 8.246913578024692);
  ASSERT_DOUBLE_EQ(yHost(1), 16);
  ASSERT_DOUBLE_EQ(yHost(2), 24);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, gemv_float)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    A = [ 1.1234567890123456789 0 0 ]
        [ 1 2 0 ]
        [ 1 2 3 ]
    x = [ 1 2 3 ]'
    y = [ 1 1 1 ]'

    gemv => y = A*x + y  ('N' in second parameter [i.e., NO TRANSPOSE])
    gemv => y = A'*x + y ('T' in second parameter [i.e., TRANSPOSE])
  */  
  float tempOne = 1.0;
  sparten::type::SubIdx idxOne = 1;
  const sparten::type::SubIdx n = 3;

  sparten::Vector<float> A("A", n*n);
  sparten::Vector<float> x("x", n);
  sparten::Vector<float> y("y", n);

  typename sparten::Vector<float>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<float>::HostMirror xHost = Kokkos::create_mirror_view(x);
  typename sparten::Vector<float>::HostMirror yHost = Kokkos::create_mirror_view(y);

  // initialize A
  for(int32_t i = 0; i < n*n; ++i)
  {
    AHost(i) = 0.0;
  }
  AHost(0) = 1.1234567890123456789;
  AHost(1) = 1.0;
  AHost(2) = 1.0;
  AHost(4) = 2.0;
  AHost(5) = 2.0;
  AHost(8) = 3.0;

  // initialize x and y
  for(int32_t i = 0; i < n; ++i)
  {
    xHost(i) = 1.0 + i;
    yHost(i) = 1.0;   
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(x,xHost);
  Kokkos::deep_copy(y,yHost);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::gemv('N',n,n,tempOne,A,n,x,idxOne,tempOne,y,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(yHost,y);

  ASSERT_FLOAT_EQ(yHost(0), 2.1234567890123456);
  ASSERT_FLOAT_EQ(yHost(1), 6);
  ASSERT_FLOAT_EQ(yHost(2), 15);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::gemv('T',n,n,tempOne,A,n,x,idxOne,tempOne,y,idxOne);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(yHost,y);

  ASSERT_FLOAT_EQ(yHost(0), 8.246913578024692);
  ASSERT_FLOAT_EQ(yHost(1), 16);
  ASSERT_FLOAT_EQ(yHost(2), 24);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, gemm_double)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    A = [ 1 3 5 ]
        [ 2 4 6 ]
    B = [ 2 4 6 ]
        [ 3 5 7.1234567890123456789 ]
    C = [ 1  7 13 ]
        [ 3  9 15 ]
        [ 5 11 17 ]

    gemm => C = A*B + C 
  */
  double tempOne = 1.0;
  const uint32_t m = 3;
  const uint32_t n = 3;
  const uint32_t k = 2;

  sparten::Vector<double> A("A", k*m);
  sparten::Vector<double> B("B", k*n);
  sparten::Vector<double> C("C", m*n);

  typename sparten::Vector<double>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<double>::HostMirror BHost = Kokkos::create_mirror_view(B);
  typename sparten::Vector<double>::HostMirror CHost = Kokkos::create_mirror_view(C);

  // 'N' for TRANSA flag
  for(int32_t i = 0; i < k*m; ++i)
  {
    AHost(i) = 1.0 * i + 1.0;
  }

  for(int32_t i = 0; i < k*n; ++i)
  {
    BHost(i) = 1.0 * i + 2.0;
  }
  BHost(k*n-1) = 7.1234567890123456789;

  for(int32_t i = 0; i < m*n; ++i)
  {
    CHost(i) = 2.0 * i + 1.0;
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(B,BHost);
  Kokkos::deep_copy(C,CHost);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::gemm('T','N',m,n,k,tempOne,A,k,B,k,tempOne,C,m);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(CHost,C);

  ASSERT_DOUBLE_EQ(CHost(0), 9);
  ASSERT_DOUBLE_EQ(CHost(1), 21);
  ASSERT_DOUBLE_EQ(CHost(2), 33);
  ASSERT_DOUBLE_EQ(CHost(3), 21);
  ASSERT_DOUBLE_EQ(CHost(4), 41);
  ASSERT_DOUBLE_EQ(CHost(5), 61);
  ASSERT_DOUBLE_EQ(CHost(6), 33.246913578024689);
  ASSERT_DOUBLE_EQ(CHost(7), 61.493827156049377);
  ASSERT_DOUBLE_EQ(CHost(8), 89.740740734074080);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, gemm_float)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    A = [ 1 3 5 ]
        [ 2 4 6 ]
    B = [ 2 4 6 ]
        [ 3 5 7.1234567890123456789 ]
    C = [ 1  7 13 ]
        [ 3  9 15 ]
        [ 5 11 17 ]

    gemm => C = A*B + C 
  */
  float tempOne = 1.0;
  const uint32_t m = 3;
  const uint32_t n = 3;
  const uint32_t k = 2;

  sparten::Vector<float> A("A", k*m);
  sparten::Vector<float> B("B", k*n);
  sparten::Vector<float> C("C", m*n);

  typename sparten::Vector<float>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<float>::HostMirror BHost = Kokkos::create_mirror_view(B);
  typename sparten::Vector<float>::HostMirror CHost = Kokkos::create_mirror_view(C);

  // 'N' for TRANSA flag
  for(int32_t i = 0; i < k*m; ++i)
  {
    AHost(i) = 1.0 * i + 1.0;
  }

  for(int32_t i = 0; i < k*n; ++i)
  {
    BHost(i) = 1.0 * i + 2.0;
  }
  BHost(k*n-1) = 7.1234567890123456789;

  for(int32_t i = 0; i < m*n; ++i)
  {
    CHost(i) = 2.0 * i + 1.0;
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(B,BHost);
  Kokkos::deep_copy(C,CHost);

  // first call, no transposed matrices
  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::gemm('T','N',m,n,k,tempOne,A,k,B,k,tempOne,C,m);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(CHost,C);

  ASSERT_FLOAT_EQ(CHost(0), 9);
  ASSERT_FLOAT_EQ(CHost(1), 21);
  ASSERT_FLOAT_EQ(CHost(2), 33);
  ASSERT_FLOAT_EQ(CHost(3), 21);
  ASSERT_FLOAT_EQ(CHost(4), 41);
  ASSERT_FLOAT_EQ(CHost(5), 61);
  ASSERT_FLOAT_EQ(CHost(6), 33.246913578024689);
  ASSERT_FLOAT_EQ(CHost(7), 61.493827156049377);
  ASSERT_FLOAT_EQ(CHost(8), 89.740740734074080);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, posv_double)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    U'= [ sqrt(1/3)  sqrt(1/3)  sqrt(1/3) ]
        [ sqrt(1/2)  sqrt(1/2)  0         ]
        [ 0         -sqrt(1/2) -sqrt(1/2) ]

    A = U*diag([10,5,1])*U'
        [ 5.833333333333333  5.833333333333333   3.333333333333333 ]
        [ 5.833333333333333  6.333333333333333   3.833333333333333 ]
        [ 3.333333333333333  3.833333333333333   3.833333333333333 ]

    x = [ 1 1 1 ]'

    posv => x = A*x = L*L*'x (where L*L' is the Cholesky factorization of A)

    Solution:
    x = [ 0.3 -0.3 0.3 ]
  */
  const uint32_t n = 3;
  sparten::Vector<double> A("A", n*n);
  sparten::Vector<double> x("x", n);

  typename sparten::Vector<double>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<double>::HostMirror xHost = Kokkos::create_mirror_view(x);

  AHost(0) = 5.0 + 5.0/6.0;
  AHost(1) = 5.0 + 5.0/6.0;
  AHost(2) = 3.0 + 1.0/3.0;
  AHost(3) = 5.0 + 5.0/6.0;
  AHost(4) = 6.0 + 1.0/3.0;
  AHost(5) = 3.0 + 5.0/6.0;
  AHost(6) = 3.0 + 1.0/3.0;
  AHost(7) = 3.0 + 5.0/6.0;
  AHost(8) = 3.0 + 5.0/6.0;

  for(int32_t i = 0; i < n; ++i)
  {
    xHost(i) = 1.0;
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(x,xHost);

  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::posv(n,A,x);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(AHost,A);
  Kokkos::deep_copy(xHost,x);

  double tol = 1e-10;
  // Only check lower triangular contains Cholesky factorization
  ASSERT_NEAR(AHost(0), 2.415229457698240, tol);
  ASSERT_NEAR(AHost(1), 2.415229457698240, tol);
  ASSERT_NEAR(AHost(2), 1.380131118684708, tol);
  ASSERT_NEAR(AHost(4), 0.707106781186548, tol);
  ASSERT_NEAR(AHost(5), 0.707106781186547, tol);
  ASSERT_NEAR(AHost(8), 1.195228609334394, tol);

  ASSERT_NEAR(xHost(0),  0.3, tol);
  ASSERT_NEAR(xHost(1), -0.3, tol);
  ASSERT_NEAR(xHost(2),  0.3, tol);
}

  #ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(blasWrapperUnitTest, posv_float)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
  /*
    U'= [ sqrt(1/3)  sqrt(1/3)  sqrt(1/3) ]
        [ sqrt(1/2)  sqrt(1/2)  0         ]
        [ 0         -sqrt(1/2) -sqrt(1/2) ]

    A = U*diag([10,5,1])*U'
        [ 5.833333333333333  5.833333333333333   3.333333333333333 ]
        [ 5.833333333333333  6.333333333333333   3.833333333333333 ]
        [ 3.333333333333333  3.833333333333333   3.833333333333333 ]

    x = [ 1 1 1 ]'

    posv => x = A*x = L*L*'x (where L*L' is the Cholesky factorization of A)

    Solution:
    x = [ 0.3 -0.3 0.3 ]
  */
  const uint32_t n = 3;
  sparten::Vector<float> A("A", n*n);
  sparten::Vector<float> x("x", n);

  typename sparten::Vector<float>::HostMirror AHost = Kokkos::create_mirror_view(A);
  typename sparten::Vector<float>::HostMirror xHost = Kokkos::create_mirror_view(x);

  AHost(0) = 5.0 + 5.0/6.0;
  AHost(1) = 5.0 + 5.0/6.0;
  AHost(2) = 3.0 + 1.0/3.0;
  AHost(3) = 5.0 + 5.0/6.0;
  AHost(4) = 6.0 + 1.0/3.0;
  AHost(5) = 3.0 + 5.0/6.0;
  AHost(6) = 3.0 + 1.0/3.0;
  AHost(7) = 3.0 + 5.0/6.0;
  AHost(8) = 3.0 + 5.0/6.0;

  for(int32_t i = 0; i < n; ++i)
  {
    xHost(i) = 1.0;
  }

  // copy data to HostMirror
  Kokkos::deep_copy(A,AHost);
  Kokkos::deep_copy(x,xHost);

  Kokkos::parallel_for(1, KOKKOS_LAMBDA (const int i) {
      sparten::spartenBlas::posv(n,A,x);
    });
  
  // copy result from HostMirror
  Kokkos::deep_copy(AHost,A);
  Kokkos::deep_copy(xHost,x);

  double tol = 1e-5;
  // Only check lower triangular contains Cholesky factorization
  ASSERT_NEAR(AHost(0), 2.415229457698240, tol);
  ASSERT_NEAR(AHost(1), 2.415229457698240, tol);
  ASSERT_NEAR(AHost(2), 1.380131118684708, tol);
  ASSERT_NEAR(AHost(4), 0.707106781186548, tol);
  ASSERT_NEAR(AHost(5), 0.707106781186547, tol);
  ASSERT_NEAR(AHost(8), 1.195228609334394, tol);

  ASSERT_NEAR(xHost(0),  0.3, tol);
  ASSERT_NEAR(xHost(1), -0.3, tol);
  ASSERT_NEAR(xHost(2),  0.3, tol);
}

} // unitTest

