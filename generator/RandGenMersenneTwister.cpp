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
 \brief Class definition for a Mersenne Twister random number generator
 */

#include <generator/RandGenMersenneTwister.h>

static const int32_t N = 624;
static const int32_t M = 397;

static const uint32_t MATRIX_A = 0x9908b0dfUL;     // constant vector a
static const uint32_t UPPER_MASK = 0x80000000UL;   // most significant w-r bits
static const uint32_t LOWER_MASK = 0x7fffffffUL;   // least significant r bits

static uint32_t _mt[N];    // State vector
static int32_t _mti;

namespace sparten
{

RandGenMersenneTwister::RandGenMersenneTwister(const uint32_t nnSeed)
{
  _mti = N + 1;

  _mt[0] = nnSeed & 0xffffffffUL;
  for (_mti = 1; _mti < N; _mti++)
  {
    _mt[_mti] = (1812433253UL * (_mt[_mti - 1] ^ (_mt[_mti - 1] >> 30)) + _mti);
    _mt[_mti] &= 0xffffffffUL;
  }
}

RandGenMersenneTwister::~RandGenMersenneTwister()
{
}

int32_t RandGenMersenneTwister::reset(int32_t nnSeed)
{
  _mti = N + 1;

  _mt[0] = nnSeed & 0xffffffffUL;
  for (_mti = 1; _mti < N; _mti++)
  {
    _mt[_mti] = (1812433253UL * (_mt[_mti - 1] ^ (_mt[_mti - 1] >> 30)) + _mti);
    _mt[_mti] &= 0xffffffffUL;
  }

  return 0;  /// \todo Why return anything if it is hard-coded?
}

uint32_t RandGenMersenneTwister::generate_rand_int32()
{
  static uint32_t mag01[2] = { 0x0UL, MATRIX_A };
  uint32_t y;

  if (_mti >= N)
  { /* generate N words at one time */
    int32_t kk;

    for (kk = 0; kk < N - M; kk++)
    {
      y = (_mt[kk] & UPPER_MASK) | (_mt[kk + 1] & LOWER_MASK);
      _mt[kk] = _mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < N - 1; kk++)
    {
      y = (_mt[kk] & UPPER_MASK) | (_mt[kk + 1] & LOWER_MASK);
      _mt[kk] = _mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (_mt[N - 1] & UPPER_MASK) | (_mt[0] & LOWER_MASK);
    _mt[N - 1] = _mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    _mti = 0;
  }

  y = _mt[_mti++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

double RandGenMersenneTwister::generate_rand_double()
{
  //---- DIVIDE BY 2^32 TO CONFORM WITH [0,1).
  return generate_rand_int32() * (1.0 / 4294967296.0);
}

double RandGenMersenneTwister::generate_rand_double_inclusive()
{
  //---- DIVIDE BY 2^32 - 1 TO CONFORM WITH [0,1].
  return generate_rand_int32() * (1.0 / 4294967295.0);
}

/*!
 *  Calling this method generates the same stream of random samples as
 *  the following Matlab code:
 *    > rstrm = RandStream('mt19937ar', 'Seed', 1);
 *    > rand (rstrm, 1, 10);
 *
 *  Note that the C++ instance must be constructed with the same seed as
 *  Matlab, and the seed must be positive.
 *
 *  Matlab RandStream behavior was checked in versions 7.10 (2010a)
 *  and 8.1 (R2013a), and found to be in agreement to 16 or more digits.
 *  Unofficial documentation indicates Matlab uses the genrand_res53()
 *  method of Matsumoto and Nishimura.
 */
double RandGenMersenneTwister::generate_rand_double_matlab()
{
  int32_t nn1 = generate_rand_int32() >> 5;
  int32_t nn2 = generate_rand_int32() >> 6;

  //---- USE 2^26 = 67,108,864 AND 2^53 = 9,007,199,254,740,992.
  double dResult = ((static_cast<double>(nn1)) * 67108864.0 + (static_cast<double>(nn2))) * (1.0 / 9007199254740992.0);

  return dResult;
}

} // end namespace sparten
