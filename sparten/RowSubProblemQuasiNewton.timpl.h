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

#pragma once

#include <sparten/RowSubProblemQuasiNewton.h>
#include <sparten/spartenUtils.h>
#include <sparten/spartenBlas.h>
#include <fstream>
#include <sstream>

// TODO The cleanest way to get rid of these macros??
#ifndef KOKKOS_ENABLE_CUDA
#define reset_timer(timer) timer.reset()
#define stop_timer(timer) timer.seconds()
#else
#define reset_timer(timer)
#define stop_timer(timer) 0
#endif

namespace sparten {

//==============================================================================

template <
  class TeamPolicy,
  class NumericalTypes
>
KOKKOS_INLINE_FUNCTION
Kokkos::pair<bool, typename NumericalTypes::kruskal_value_t>
RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>::_solve()
{
  // Readability aliases
  // Each team is responsible for one row:
  auto iRow = _team_member.league_rank();
  auto nComps = _nComponents;
  auto nSizeLBFGS = _config.size_LBFGS;

  kruskal_value_t dInitialKktError = 0.0;
  kruskal_value_t dKktError = 0; //std::numeric_limits<kruskal_value_t>::max();  //  DOUBLE_MAX;
  kruskal_value_t dObj;// = -std::numeric_limits<kruskal_value_t>::max(); // DOUBLE_MAX;

  // Get current values of row subproblem
  // Kokkos is getting this, but for now, call a hand-rolled version:
  sparten::deep_copy(_team_member, _daVars, Kokkos::subview(_dKdata, iRow, Kokkos::ALL));

  // Initialize
  sparten::deep_copy(_team_member, _daRho, 0);
  sparten::deep_copy(_team_member, _daDeltaM, 0);
  sparten::deep_copy(_team_member, _daDeltaG, 0);

  // Extract local Pi
  _team_member.team_barrier();

  sub_index_t nlbfgsPos = 0;
  int32_t nLineSearchCode = 0;

	// Collect algorithm timing data
#ifndef KOKKOS_ENABLE_CUDA
	Kokkos::Timer timer;
#else
	//	double timer = 0.0;
#endif
	kruskal_value_t elapsed_time = 0.0;

  // Perform iterations to solve the row subproblem.
  for (sub_index_t iIts = 0; iIts < _config.max_inner_iterations; ++iIts)
  {
    // Gradient is 1 - ComputePhi, where
    // phi_j = sum i=1:nnz X[i] / (sum r=1:R m[r] Pi[i,r]), for j=1:R
    _compute_phi();
    _team_member.team_barrier();

    // Compute the gradient and maximum KKT error for the row subproblem.
    dKktError = 0.0;
    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(_team_member, nComps),
      [&] (sub_index_t iComp, kruskal_value_t& dKktErrorLocal) {
        _daGrad(iComp) = 1.0 - _daPhi(iComp);
        kruskal_value_t d = min(_daGrad(iComp), _daVars(iComp));
        dKktErrorLocal = max(abs(d), dKktError);
      },
      Kokkos::Max<kruskal_value_t>{dKktError}
    );

    // Original MATLAB code does a gradient step to prime the L-BFGS approximation.
    // However, it means a row subproblem that has already converged
    // wastes time doing a gradient step before checking KKT conditions.
    // This fixes that.
    if (iIts == 0)
    {

      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(_team_member, nComps),
        [&] (sub_index_t iComp) {
          _daGradOld[iComp] = _daGrad[iComp];
          _daSearchDirection[iComp] = -_daGrad[iComp];
          _daVarsOld[iComp] = _daVars[iComp];
        }
      );
      _team_member.team_barrier(); // TODO make sure this is necessary

      // Get new daVars
      kruskal_value_t dAred;
      this->_line_search(
        /* daRowVars = */ _daVarsOld,
        /* daRowGrad = */ _daGrad,
        /* dNewRowVars = */ _daVars,
        /* dObjNew = */ dObj,
        /* dUnitStepAred = */ dAred,
        /* nRetCode = */ nLineSearchCode
      );
      _team_member.team_barrier();

      /* Compute New Gradient */
      // Compute this timeComputePhi
      _compute_phi();
      _team_member.team_barrier();
      dKktError = 0.0;
      Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(_team_member, nComps),
        [&] (sub_index_t iComp, kruskal_value_t& dKktErrorLocal) {
          _daGrad(iComp) = 1.0 - _daPhi(iComp);
          kruskal_value_t d = min(_daGrad(iComp), _daVars(iComp));
          dKktErrorLocal = max(abs(d), dKktError);
        },
        Kokkos::Max<kruskal_value_t>{dKktError}
      );

      // set the initial error
      dInitialKktError = dKktError;
    }

    if (dKktError < _config.row_tolerance)
    {
	    // Return since the stop tolerance is met.
      if (iIts > 0)
      {
        // Variables were modified to achieve the stop tolerance.
        sparten::deep_copy( _team_member, Kokkos::subview(_dKdata, iRow, Kokkos::ALL), _daVars );
        elapsed_time = stop_timer(timer);


        Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
           stats.t_elapsed = elapsed_time;
        });
        _team_member.team_barrier();
        dObj = this->_obj_likelihood(_daVars);
        Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
            stats.obj = dObj;
        });

        _team_member.team_barrier();
        return { true, dInitialKktError };
      }
      else
      {
         // Variables were not modified.
         // Compute the objective since there was no line search.
	      elapsed_time = stop_timer(timer);
	      Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
	         stats.t_elapsed = elapsed_time;
	      });

	      dObj = this->_obj_likelihood(_daVars);
	      Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
	         stats.obj = dObj;
	      });
         _team_member.team_barrier();
         return { false, dInitialKktError };
      }
    }

    // Compute a search direction based on the active and free variable
    // using a Quasi Newton step.
    // Return the predicted reduction of the objective.
    kruskal_value_t dDeltaProduct = 0.0;

    _team_member.team_barrier();
    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(_team_member, nComps),
      [&] (const sub_index_t iComp, kruskal_value_t& deltaProduct) {
        _daTmpDeltaM(iComp) = _daVars(iComp) - _daVarsOld(iComp);
        _daTmpDeltaG(iComp) = _daGrad(iComp) - _daGradOld(iComp);
        deltaProduct += _daTmpDeltaM(iComp) * _daTmpDeltaG(iComp);
      },
      Kokkos::Sum<kruskal_value_t>(dDeltaProduct)
    );

    kruskal_value_t dTmpRho = static_cast<kruskal_value_t>(1.0) / dDeltaProduct;

    // Update the L-BFGS approximation.
    if (dTmpRho < sparten::numeric_limits<kruskal_value_t>::max && dTmpRho > 0.0)
    {
      sparten::deep_copy(_team_member,
        Kokkos::subview(_daDeltaM, nlbfgsPos, Kokkos::ALL),
        _daTmpDeltaM
      );
      sparten::deep_copy(_team_member,
        Kokkos::subview(_daDeltaG, nlbfgsPos, Kokkos::ALL),
        _daTmpDeltaG
      );
      _daRho(nlbfgsPos) = dTmpRho;
    }
    else
    {
      // Rho is required to be positive; if not, then skip
      // the L-BFGS update pair.  The recommended safeguard for
      // full BFGS is Powell damping, but not clear how to damp
      // in 2-loop L-BFGS.

      // Roll back lbfgsPos since it will be incremented later.
      if (nlbfgsPos == 0)
      {
        if(_daRho(nSizeLBFGS-1) > 0)
        {
          nlbfgsPos = nSizeLBFGS-1;
        }
        else
        {
	        elapsed_time = stop_timer(timer);
	        Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
	           stats.t_elapsed = elapsed_time;
	        });

	        dObj = this->_obj_likelihood(_daVars);
           _team_member.team_barrier();
	        Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
	           stats.obj = dObj;
	        });

	        return { true, dInitialKktError };
        }
      }
      else
      {
        nlbfgsPos = nlbfgsPos - 1;
      }
    }

    nLineSearchCode = _compute_search_dir(nlbfgsPos, iIts);


    // Increment L-BFGS Position
    nlbfgsPos = (nlbfgsPos+1) % nSizeLBFGS;

    _team_member.team_barrier();
    sparten::deep_copy(_team_member, _daVarsOld, _daVars);
    sparten::deep_copy(_team_member, _daGradOld, _daGrad);

    _team_member.team_barrier();

    // Perform a line search.
    kruskal_value_t dAred; // is this unused?!??
    this->_line_search(
      /* daRowVars = */ _daVarsOld,
      /* daRowGrad = */ _daGradOld,
      /* dNewRowVars = */ _daVars,
      /* dObjNew = */ dObj,
      /* dUnitStepAred = */ dAred,
      /* nRetCode = */ nLineSearchCode
    );

    Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
	     ++stats.n_iters;
    });
  }

  _team_member.team_barrier();
  sparten::deep_copy( _team_member, Kokkos::subview(_dKdata, iRow, Kokkos::ALL), _daVars );

  elapsed_time = stop_timer(timer);
  Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
     stats.t_elapsed = elapsed_time;
  });

  Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
     stats.obj = -dObj;
  });
  return { true, dInitialKktError };
}

//==============================================================================


template <class TeamPolicy, class NumericalTypes>
void RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>::_compute_phi()
{

#ifndef KOKKOS_ENABLE_CUDA // TODO Get rid of these ugly macros
   Kokkos::Timer timer;
#else
   //double timer = 0.0;
#endif
   kruskal_value_t elapsed_time = 0.0;


   Kokkos::parallel_for( Kokkos::ThreadVectorRange( _team_member, _nComponents), [&] ( const sub_index_t  &iComp )
   {
      _daPhi(iComp)= 0;
   });

   _team_member.team_barrier(); // Make sure _daPhi is initialized
#if 1
   Kokkos::parallel_for( Kokkos::ThreadVectorRange(_team_member, _nNonz), [&] (const sub_index_t &iNonz) {
      auto mySum = sparten::dot_product(Kokkos::subview(_pi, _nonzLoc(iNonz), Kokkos::ALL),_daVars);
      auto valforPhi = _sparseData(_nonzLoc(iNonz)) / max( mySum,_config.eps_div_zero_grad);
      for( sub_index_t iComp = 0; iComp < _nComponents; ++iComp) {
         Kokkos::atomic_add(&_daPhi(iComp), _pi(_nonzLoc(iNonz),iComp) * valforPhi);
      }
   });
#else
   for( sub_index_t iComp = 0; iComp < _nComponents; ++iComp) {
      Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(_team_member, _nNonz), [&] (const sub_index_t &iNonz, kruskal_value_t &modifier) {
         auto mySum = sparten::dot_product(Kokkos::subview(_pi, _nonzLoc(iNonz), Kokkos::ALL),_daVars);
         mySum = _sparseData(_nonzLoc(iNonz)) / max( mySum,_config.eps_div_zero_grad);
         modifier+= _pi(_nonzLoc(iNonz),iComp) * mySum;
      },_daPhi(iComp));
   }
#endif
   elapsed_time = stop_timer(timer);


   Kokkos::single( Kokkos::PerTeam( _team_member), [&] () {
      const_cast<kruskal_value_t &>(stats.t_compute_phi) = elapsed_time;
   });

}


template <class TeamPolicy, class NumericalTypes>
int32_t
RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>::_compute_search_dir(
  sub_index_t nlbfgsPos,
  sub_index_t iters
)
{
  // Readability alias
  auto& daAlpha = _daAlpha;
  int32_t nRetCode = 0;

#ifndef KOKKOS_ENABLE_CUDA // TODO replace this macro
	Kokkos::Timer timer;
#else
	//	double timer = 0.0;
#endif
	kruskal_value_t elapsed_time = 0.0;

  // Determine active and free variables.
  // If dEpsActiveSet is zero, then the following works:
  //   fixedVars = find((m_row == 0) & (grad' > 0));
  // For the general case this works but is less clear and assumes m_row > 0:
  //  fixedVars = find((grad' > 0) & (m_row <= min(epsActSet,grad')));

  // Calculate the Bertsekas active set threshold.
  // projGradStep = (m_row - grad') .* (m_row - grad' > 0);
  kruskal_value_t dSumSq = 0.0;
  Kokkos::parallel_reduce(
    Kokkos::ThreadVectorRange(_team_member, _nComponents),
    [&] (sub_index_t iComp, kruskal_value_t& sumSq) {
      kruskal_value_t d = _daVars(iComp) - _daGrad(iComp);

       if (d > 0.0)
      {
        sumSq += (_daVars(iComp) - d) * (_daVars(iComp) - d);
      }
    },
    Kokkos::Sum<kruskal_value_t >(dSumSq)
  );


  // Get 2-norm
  kruskal_value_t dEps = sqrt(static_cast<double>(dSumSq));
  dEps = sparten::min(_config.eps_active_set, dEps);

  // Determine free variables, and set the fixed variables.
  sub_index_t nNumFixed = 0;
  //Kokkos::parallel_reduce(
  //  Kokkos::ThreadVectorRange(_team_member, _nComponents),
  //  [&] (sub_index_t iComp, sub_index_t& numFixed) {
  // TODO vectorize this?
  for(sub_index_t iComp = 0; iComp < _nComponents; ++iComp) {
    if ((_daGrad(iComp) > 0.0) && (_daVars(iComp) < dEps))
    {
      // Active (fixed) variable.
      _daSearchDirection(iComp) = 0.0;
      _naFreeIndices(nNumFixed) = iComp;
      ++nNumFixed;
    }
    else
    {
      // Free variables
      _daSearchDirection(iComp) = -_daGrad(iComp);
    }
  }
  //  },
  //  nNumFixed
  //);



  kruskal_value_t dDeltaNorm = sparten::dot_product(
    _team_member,
    Kokkos::subview(_daDeltaM, nlbfgsPos, Kokkos::ALL),
    Kokkos::subview(_daDeltaG, nlbfgsPos, Kokkos::ALL)
  );

  kruskal_value_t dDeltaGNorm = sparten::dot_product(
    _team_member,
    Kokkos::subview(_daDeltaG, nlbfgsPos, Kokkos::ALL),
    Kokkos::subview(_daDeltaG, nlbfgsPos, Kokkos::ALL)
  );

  if (dDeltaNorm == 0.0)
  {
    // DMD: this does not actually use the gradient in this case, we should return here
    // Cannot proceed with this L-BFGS data; most likely the iteration
    printf("WARNING: L-BFGS update is orthogonal, using gradient\n");
  }

  sparten::deep_copy(_team_member, daAlpha, 1.0);
  _team_member.team_barrier();

  sub_index_t iK = nlbfgsPos;
  sub_index_t iEnd = min(iters + 1, _config.size_LBFGS);

  // Perform an L-BFGS two-loop recursion to compute the search direction.
  // note that we don't need any more barriers until we zero out the fixed positions,
  // since each vector lane is responsible for the same component until then,
  // and that is the only thing that's being written

  for (sub_index_t iCount = 0; iCount < iEnd; ++iCount)
  {

    // alpha(k) = rho(k) * delta_m(:, k)' * d;
    daAlpha(iK) = _daRho(iK) * dot_product(
      _team_member,
      Kokkos::subview(_daDeltaM, iK, Kokkos::ALL),
      _daSearchDirection
    );

    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(_team_member, _nComponents),
      [&] (sub_index_t iComp) {
        // d = d - alpha(k) * delta_g(:, k);
        _daSearchDirection(iComp) -= daAlpha(iK) * _daDeltaG(iK, iComp);

      }
    );
    // k = lbfgsSize - mod(1 - k, lbfgsSize);
    // MATLAB is 1-based, sparten is 0-based, so slightly different mod calculation
    iK = (_config.size_LBFGS + iK - 1) % _config.size_LBFGS; // Declement by 1
  }

  // coef = 1 / rho(lbfgs_pos) / (delta_g(:, lbfgs_pos)' * delta_g(:, lbfgs_pos));
  kruskal_value_t coef = 1.0 / _daRho(nlbfgsPos) / dDeltaGNorm;
  // d = coef * d;
  sparten::scale(_team_member, _daSearchDirection, coef);

  for (sub_index_t iCount = 0; iCount < iEnd; ++iCount)
  {
    // k = mod(k, lbfgsSize) + 1;
    iK = ((iK + 1) % _config.size_LBFGS); // Increment by 1

    // b = rho(k) * delta_g(:, k)' * d;

    kruskal_value_t beta = _daRho(iK) * sparten::dot_product(
      _team_member,
      Kokkos::subview(_daDeltaG, iK, Kokkos::ALL),
      _daSearchDirection
    );

    //  d = d + (alpha(k) - b) * delta_m(:, k);
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(_team_member, _nComponents),
      [&] (sub_index_t iComp) {
        _daSearchDirection(iComp) += (daAlpha(iK) - beta) * _daDeltaM(iK, iComp);
      }
    );
  }

  _team_member.team_barrier();

  // 0-out fixed positions
  // d(fixedVars) = 0;
  Kokkos::parallel_for(
    Kokkos::ThreadVectorRange(_team_member, nNumFixed),
    [&] (sub_index_t iFixed)
    {
      _daSearchDirection(_naFreeIndices(iFixed)) = 0;
    }
  );

	elapsed_time = stop_timer(timer);
	Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
		stats.t_search_direction = elapsed_time;
	});

  return nRetCode;
}

//==============================================================================


template <class TeamPolicy, class NumericalTypes>
template <class KruskalView1D>
//typename NumericalTypes::kruskal_value_t
typename  KruskalView1D::value_type
RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>::obj_likelihood (
  KruskalView1D &daRowVars
)
{
  return this->_obj_likelihood( daRowVars );
}


template <class TeamPolicy, class NumericalTypes>
template <class RowVarsView, class RowGradView, class NewRowVarsView>
void
RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>::line_search(
  RowVarsView const& daRowVars,
  RowGradView const& daRowGrad,
  NewRowVarsView &daNewRowVars,
  kruskal_value_t& dObjNew,
  kruskal_value_t& dUnitStepAred,
  int32_t& nRetCode
)
{
  this->_line_search( daRowVars, daRowGrad, daNewRowVars, dObjNew, dUnitStepAred, nRetCode );
}

template <class TeamPolicy, class NumericalTypes>
void RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>::setup_local_pi()
{
  this->_setup_local_pi();
}
//==============================================================================

//==============================================================================

} // end namespace sparten

