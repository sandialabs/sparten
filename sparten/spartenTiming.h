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

#include <Kokkos_Timer.hpp>

#pragma once

namespace sparten {

struct RowProblemTimingData
{
	double compute_phi       = 0.0;
	double search_direction  = 0.0;
	double line_search       = 0.0;
	double failed_search     = 0.0;
	double time_elapsed      = 0.0;
};

struct QuasiNewtonTimingData
{
	double distributeWeights = 0.0;
	double computePi         = 0.0;
	double taskParallel      = 0.0;
	double normalize         = 0.0;
	double permute           = 0.0;

	RowProblemTimingData row;
};

struct DampedNewtonTimingData
{
	double distributeWeights = 0.0;
	double computePi         = 0.0;
	double taskParallel      = 0.0;
	double normalize         = 0.0;
	double permute           = 0.0;

	RowProblemTimingData row;
};


class ScopedTimerContribution
{
private:
  double& _contribute_to;
  Kokkos::Timer _timer;
public:
  ScopedTimerContribution(ScopedTimerContribution const&) = delete;
  ScopedTimerContribution(ScopedTimerContribution&&) = delete;
  explicit
  ScopedTimerContribution(double& contrib)
    : _contribute_to(contrib)
  { }
  ~ScopedTimerContribution() { _contribute_to += _timer.seconds(); }
};

} // end namespace sparten

