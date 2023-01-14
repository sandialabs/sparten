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

#include <sparten/RowSubProblemDampedNewton.h>
#include <sparten/spartenUtils.h>
#include <sparten/spartenBlas.h>

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
            class TeamMember,
            class View2D,
            class View1D
      >
      void compute_scale_D(
            TeamMember& team_member,
            int nNumFree,
            View2D freeHessian,
            View1D output
      )
      {
         // Fixed value here now, since this is a property of the machine precision
         using value_type = typename View1D::value_type;
         constexpr auto free_hessian_scale_cutoff = sparten::numeric_limits<value_type>::epsilon;

         // Barrett this is where your code goes!
         Kokkos::parallel_for(
               Kokkos::ThreadVectorRange(team_member, nNumFree),
               [&](int iFree) {
                  // TODO change this to 2D array
                  auto& diag = freeHessian[iFree * nNumFree + iFree];
                  if(diag > free_hessian_scale_cutoff) {
                     output[iFree] = 1.0 / sqrt(diag);
                  }
                  else {
                     output[iFree] = 1.0;
                  }
               }
         );
         // TODO figure out if this is necessary
         team_member.team_barrier();

      }


//==============================================================================


//==============================================================================

template <
           class TeamPolicy,
           class NumericalTypes
   >
KOKKOS_INLINE_FUNCTION
Kokkos::pair<bool, typename NumericalTypes::kruskal_value_t> RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>::_solve()
{
   // Readability aliases
   // Each team is responsible for one row. This is subject to change
   auto iRow = _iRow;
   auto nComps = _kruskal_info.num_components;

   kruskal_value_t dInitialKktError = 0.0;
   kruskal_value_t dKktError = sparten::numeric_limits<kruskal_value_t>::max;
   kruskal_value_t dObj ;
   kruskal_value_t dPredictedReduction;
   kruskal_value_t dMuDamping = _config.mu_initial;

   // Get current values of row subproblem
   // Kokkos is getting this, but for now, call a hand-rolled version:
   sparten::deep_copy(_team_member, _daVars, Kokkos::subview(_dKdata, iRow, Kokkos::ALL));

   int32_t nLineSearchCode = 0;
   const int32_t threadLocalSize=128;  // Put maximum temporary array size per thread.

   // Collect algorithm timing data
#ifndef KOKKOS_ENABLE_CUDA
   Kokkos::Timer timer;
#else
   //      double timer = 0.0;
#endif

   kruskal_value_t elapsed_time = 0.0;


   // Perform iterations to solve the row subproblem.
   for (sub_index_t iIts = 0; iIts < _config.max_inner_iterations; ++iIts)
   {
      // Gradient is 1 - timeComputePhi, where
      // phi_j = sum i=1:nnz X[i] / (sum r=1:R m[r] Pi[i,r]), for j=1:R
      // Save timeComputePhi to use later in computing the Hessian.

      // Initialize Phi values.  Phi is not used for Hessian, but it might be used when lineSearch is not successful.
     Kokkos::parallel_for(Kokkos::ThreadVectorRange(_team_member, nComps), [&] (const sub_index_t &iComp) {
         _daPhi(iComp) = 0;
      });
      //
      // Initializing Hessian Matrix
      // It's OK to change reduce for 2D Kokkos::View
      auto& daFreeHessian_full = _daWork1;
      auto& daFreeHessian_active = _daWork2;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(_team_member, nComps*nComps), [&]( const sub_index_t iComp ) {
         daFreeHessian_full(iComp)=0;
      });

      _team_member.team_barrier();

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(_team_member, _nNonz), [&] (const sub_index_t &iNonz) {
         // i is cov = m_row * Pi
         kruskal_value_t UpsRow = 0;
         kruskal_value_t mypi[threadLocalSize];

         // Initialize mypi to 1
         for (sub_index_t iComp = 0; iComp < nComps; ++iComp)
         {
            mypi[iComp] = 1;
         }


         // Compute Partial Derivative and Local Pi
         for (sub_index_t imode = 0; imode < _numModes; ++imode)
         {
            if (imode != _iModeSkip) {
               const auto myKdata = _kTensor(imode);
               const auto index = _sparseIndex(imode, _nonzLoc(iNonz));

               for (sub_index_t iComp=0; iComp< nComps; ++iComp) {
                  mypi[iComp] *= myKdata(index, iComp);
               }
            }
         }

         for (sub_index_t iComp=0; iComp< nComps; ++iComp) {
            UpsRow += _daVars(iComp) * mypi[iComp];
         }



         // Eliminate small values. Future version handles this issue in a better way.
         if (UpsRow < _config.eps_div_zero_grad)
         {
            UpsRow = _config.eps_div_zero_grad;
         }

         auto ValForPhi = _sparseData( _nonzLoc(iNonz) ) / UpsRow;


         // CPU version needs thread-local _daPhi
         for(sub_index_t iComp = 0; iComp < nComps; ++iComp )
         {
#ifndef KOKKOS_ENABLE_CUDA
            _daPhi(iComp) += mypi[iComp] * ValForPhi;
#else
            Kokkos::atomic_add(&_daPhi(iComp), mypi[iComp] * ValForPhi);
#endif
         }



         auto sqVal = UpsRow*UpsRow;
         UpsRow = _sparseData(_nonzLoc(iNonz)) / max(sqVal, _config.eps_div_zero_grad);
#if 0
         for(sub_index_t iComp = 0; iComp < nComps; ++iComp) {
            for(sub_index_t jComp = 0; jComp < nComps; ++jComp) {
               auto localVal = mypi[iComp] * UpsRow *  daTmpPi[jComp];
               Kokkos::atomic_add(&daFreeHessian_full(iComp*nComps+jComp), localVal);

            }
         }
#else
         for(sub_index_t iComp = 0; iComp < nComps*nComps; ++iComp )
         {
            auto localVal = mypi[ iComp / nComps] * UpsRow * mypi[ iComp % nComps];
            // One core/thread is assigned to Kokkos::team
#ifndef KOKKOS_ENABLE_CUDA
            daFreeHessian_full(iComp) += localVal;
#else
            Kokkos::atomic_add(&daFreeHessian_full(iComp), localVal);
#endif
         }
#endif
      });

      _team_member.team_barrier();

      // Compute the gradient and maximum KKT error for the row subproblem.
      // phi is kept for remedy of a failed line search.
      dKktError = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(_team_member, nComps), [&] (sub_index_t iComp, kruskal_value_t& dKktErrorLocal) {

         _daGrad(iComp) = 1.0 - _daPhi(iComp);
         kruskal_value_t d = min(_daGrad(iComp), _daVars(iComp));
         dKktErrorLocal = max(abs(d), dKktError);

      },Kokkos::Max<kruskal_value_t>{dKktError});

      if ( iIts == 0 )
      {
         dInitialKktError = dKktError;
      }

      if (dKktError < _config.row_tolerance)
      {
         // Return since the stop tolerance is met.
         if (iIts > 0)
         {
            // Variables were modified to achieve the stop tolerance.
            sparten::deep_copy( _team_member, Kokkos::subview(_dKdata, iRow, Kokkos::ALL), _daVars );
            _team_member.team_barrier();

            elapsed_time = stop_timer(timer);
            Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
               stats.t_elapsed = elapsed_time;
            });

            dObj = this->_obj_likelihood2(_daVars);  // Use the special version of obj_liklihood
            Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {

               stats.obj = dObj;

            });

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

            dObj = this->_obj_likelihood2(_daVars); // Use the special version of obj_liklihood

            Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {

               stats.obj = dObj;

            });

            return { false, dInitialKktError };
         }
      }

      //
      // This part is used to compute active set
      //
      // Start Norm Calculation
      //

      kruskal_value_t dSumSq = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(_team_member, nComps), [&] (const sub_index_t iComp, kruskal_value_t& sumSq) {
         kruskal_value_t d = _daVars(iComp) - _daGrad(iComp);
         if (d > 0.0)
         {
            sumSq += (_daVars(iComp) - d) * (_daVars(iComp) - d);
         }
      }, Kokkos::Sum<kruskal_value_t>(dSumSq) );


      // Get 2-norm
      kruskal_value_t dEps = static_cast<kruskal_value_t >(sqrt(static_cast<double>(dSumSq)));
      dEps = sparten::min(_config.eps_active_set, dEps);

      // Computing Active Set
      sub_index_t nNumFree = 0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(_team_member, nComps), [&] (const sub_index_t iComp, sub_index_t& lnNumFree ) {
         if ((_daGrad(iComp) > 0.0) && (_daVars(iComp) == 0.0)) {
            _daSearchDirection(iComp) = 0.0;
            _naFreeIndices(iComp) = 0;
         }
         else if( (_daGrad(iComp) > 0.0) && (_daVars(iComp) <= dEps) )
         {
            _daSearchDirection(iComp) = - _daGrad(iComp);
            _naFreeIndices(iComp) = 0;
         } else {
            ++lnNumFree;
            _naFreeIndices(iComp) = 1;

         }
      }, Kokkos::Sum<sub_index_t >(nNumFree) );


      // Update free indices by one thread per team.
      Kokkos::single( Kokkos::PerTeam(_team_member), [&] () {
         int nUpd = 0;
         for (sub_index_t iComp=0; iComp < nComps; ++iComp ){
            if (_naFreeIndices(iComp) == 1)
            {
               _naFreeIndices(nUpd) = iComp;
               nUpd++;
            }
         }
      });
      _team_member.team_barrier();

      //
      // Copy full to active set
      //
      // Alternatively, it is possible to develop parallel Chokesky with active set indices.
      //
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(_team_member, nNumFree), [&] (const sub_index_t indCol) {
         const sub_index_t indColFull = _naFreeIndices(indCol);
         for( sub_index_t indRow = 0; indRow < nNumFree; ++indRow )
         {
            const sub_index_t indRowFull = _naFreeIndices(indRow);
            daFreeHessian_active(indCol*nNumFree+indRow) = daFreeHessian_full(indColFull*nComps + indRowFull );
         }
      });

      _team_member.team_barrier();

      // Compute a search direction based active and free variables,
      // using a damped Newton step for the free variables.

      _compute_search_dir( dMuDamping, dPredictedReduction, nNumFree  );
      _team_member.team_barrier();


      // Perform a line search.
      kruskal_value_t dAred;

      // Copying daVar
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(_team_member, nComps), [&] (sub_index_t iComp) {
         _daVarsOld(iComp) = _daVars(iComp);
      });


      this->_line_search2(
                 /* daRowVars = */ _daVarsOld,
                 /* daRowGrad = */ _daGrad, //
                 /* dNewRowVars = */ _daVars,
                 /* dObjNew = */ dObj,
                 /* dUnitStepAred = */ dAred,
                 /* nRetCode = */ nLineSearchCode);

      if (dPredictedReduction == 0.0)
      {
         dMuDamping = 10.0 * dMuDamping; // Need to re-initialized for the next row
      } else {
         kruskal_value_t dRho = dAred / -dPredictedReduction;
         if (dRho < _config.damping_increase_tolerance  ) // Poor agreement, increase the damping parameter.
         {
            dMuDamping = _config.damping_increase_factor * dMuDamping; // Need to re-initialized for the next row
            }
            else if (dRho > _config.damping_decrease_tolerance ) // Good agreement, reduce the damping parameter.
            {
               dMuDamping = _config.damping_decrease_factor * dMuDamping; // Need to re-initialized for the next row
            }
         }
         Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {

            ++stats.n_iters;

         });

   }

   sparten::deep_copy(_team_member, Kokkos::subview(_dKdata, iRow, Kokkos::ALL), _daVars);
   _team_member.team_barrier();
   elapsed_time = stop_timer(timer);
   Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
       stats.t_elapsed = elapsed_time;
   });

   Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
      stats.obj = -dObj;
   });
   return { true, dInitialKktError };
}




template <class TeamPolicy, class NumericalTypes>
int32_t RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>::_compute_search_dir(
           kruskal_value_t  dMuDamping,
           kruskal_value_t  &dPredictedReduction,
           sub_index_t nNumFree )
{
   // Readability alias
   auto& daFreeHessian = _daWork2;
   int32_t nRetCode = 0;

#ifndef KOKKOS_ENABLE_CUDA // TODO replace this macro
      Kokkos::Timer timer;
#else
      //      double timer = 0.0;
#endif
   kruskal_value_t  elapsed_time = 0.0;

   // Shift the diagonal values
   auto& daFreeSearch  = _daWork1;
   Kokkos::parallel_for (Kokkos::TeamThreadRange( _team_member, nNumFree ), [&]  (sub_index_t iNum  ) {
      daFreeHessian(iNum * nNumFree + iNum) +=  dMuDamping;
      daFreeSearch(iNum) = -_daGrad(_naFreeIndices(iNum));
   });
   _team_member.team_barrier();


#if 0
   if(_config.should_scale_hessian) {
    // (This is your cue Barrett)
    // TODO!!! Compute scaleD here!
    compute_scale_D(_team_member, nNumFree, daFreeHessian, _daScaleD);
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(_team_member, nNumFree),
      [&] (int i) {
        // TODO @minor think about moving this loop out?
        for(int j = 0; j < nNumFree; ++j) {
          // TODO use 2D
          daFreeHessian(i*nNumFree + j) *= _daScaleD[i] * _daScaleD[j];
        }
        daFreeSearch[i] *= _daScaleD[i];
      }
    );
  }
#endif

   bool  posvSucceeded = 1;
   Kokkos::single( Kokkos::PerThread(_team_member), [&] () {
      posvSucceeded = spartenBlas::posv(nNumFree, daFreeHessian, daFreeSearch);
   });


   if(!posvSucceeded)
   {
     Kokkos::parallel_for (Kokkos::ThreadVectorRange( _team_member, nNumFree), [&]  (sub_index_t iNum )
      {
         _daSearchDirection(_naFreeIndices(iNum)) = -_daGrad(_naFreeIndices(iNum));
      });
      nRetCode = -2;
   }


#if 0
      else if(_config.should_scale_hessian) {
	Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(_team_member, nNumFree),
      [&] (int i) {
        daFreeSearch[i] *= _daScaleD[i];
      }
    );
  }
#endif

   Kokkos::parallel_for (Kokkos::ThreadVectorRange( _team_member, nNumFree), [&]  (sub_index_t iNum )
   {
         _daSearchDirection(_naFreeIndices(iNum)) = daFreeSearch(iNum);
   });
   auto& daSolution  = _daWork1; // daSolution

   _team_member.team_barrier();
   Kokkos::single( Kokkos::PerThread(_team_member), [&] () {
      spartenBlas::trmv('L', 'T', 'N', nNumFree, daFreeHessian, nNumFree, daSolution, static_cast<sub_index_t>(1));
      spartenBlas::trmv('L', 'N', 'N', nNumFree, daFreeHessian, nNumFree, daSolution, static_cast<sub_index_t>(1));
   });
   _team_member.team_barrier();
   kruskal_value_t Q0, Q1;
   Q0 = sparten::dot_product_left_with_index( _team_member, _daSearchDirection, daSolution, _naFreeIndices, nNumFree );
   Q1 = sparten::dot_product( _team_member, _daSearchDirection, _daGrad, _naFreeIndices, nNumFree );

   dPredictedReduction = (0.5 * Q0) + Q1;

   elapsed_time += stop_timer(timer);
   Kokkos::single( Kokkos::PerTeam( _team_member ), [&] () {
      stats.t_search_direction = elapsed_time;
   });
   return nRetCode;
}


template <class TeamPolicy, class NumericalTypes>
void RowSubProblemDampedNewton<
  TeamPolicy,
  NumericalTypes
>::compute_phi() const
{
  this->_compute_phi();
}

template <class TeamPolicy, class NumericalTypes>
template <class KruskalView1D>
//typename NumericalTypes::kruskal_value_t
typename  KruskalView1D::value_type
RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>::obj_likelihood (
  KruskalView1D &daRowVars
)
{
  return this->_obj_likelihood( daRowVars );
}

template <class TeamPolicy, class NumericalTypes>
template <class RowVarsView, class RowGradView, class NewRowVarsView>
void
RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>::line_search(
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
void RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>::setup_local_pi()
{
  this->_setup_local_pi();
}

} // end namespace sparten
