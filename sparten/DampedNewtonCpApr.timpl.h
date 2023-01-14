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

#include <sparten/CpAprCommon.timpl.h>
#include <sparten/DampedNewtonCpApr.h>
#include <sparten/DampedNewtonKernel.h>
#include <sparten/spartenUtils.h>
#include <sparten/Log.h>
#include <sparten/PrecisionTraits.h>
#include <Kokkos_Core.hpp>
#include <sstream>
#include <tpl/kokkos/containers/src/Kokkos_ScatterView.hpp>

namespace sparten {
//==============================================================================

template<class NumericTypes,class team_policy_t, class dkView,class spView,class piView,class locView, class locindexView, class statsView,
           class solver_conf_t, class kruskal_conf_t, class mode_conf_t, class scratch_space_t, class mode_ind_t, class num_mode_t, class krTensor, class spIndexView>
void _damped_newton_driver_1(
           mode_conf_t  sparse_tensor_info,
           solver_conf_t config,
           kruskal_conf_t kruskal_info,
           mode_ind_t iMode,
           num_mode_t numMode,
           piView pi,
           dkView dkdata,
           locView nonzLocs,
           locindexView nonzLocsIdx,
           krTensor kOutput,
           spView spdata,
           spIndexView spindex,
           statsView &row_stats,
           team_policy_t team_policy,
           scratch_space_t scratch_per_team )
{
  Kokkos::parallel_for("damped_newton_driver_1", team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_per_team)), KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team) {
     const auto iRow = team.league_rank()*team.team_size()+team.team_rank();

     // Construct the row subproblem Solver
     RowSubProblemDampedNewton<team_policy_t, NumericTypes> row_solver = RowSubProblemDampedNewton<team_policy_t, NumericTypes>(
                 /* iRow = */        iRow,
                 /* iMode = */       iMode,
                 /* numMode = */     numMode,
                 /* team_member = */ team,
                 /* config = */      config,
                 /* input_info = */  sparse_tensor_info,
                 /* kruskal_info = */ kruskal_info,
                 /* pi = */ pi,
                 /* dKdata = */ dkdata,
                 /* nonz_loc = */ nonzLocs,
                 /* nonz_loc_idx = */ nonzLocsIdx,
                 /* sparse_data = */  spdata,
                 /* sparse_index */   spindex,
                 /*  KruskalTensor = */ kOutput );

      if (not row_solver.has_nonzero_entries()) {
         sparten::deep_copy(team, Kokkos::subview(dkdata, iRow, Kokkos::ALL), 0);
            // Manually set output and stats for rows that are skipped

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
               row_stats[iRow].row_modified = false;
               row_stats[iRow].t_elapsed = 0.0;
               row_stats[iRow].t_compute_phi = 0.0;
               row_stats[iRow].t_search_direction = 0.0;
               row_stats[iRow].t_line_search = 0.0;
               row_stats[iRow].n_func_evals = 0;
               row_stats[iRow].n_iters = 0;
               row_stats[iRow].obj = 0.0;
          });
      } else {
         auto result = row_solver._solve();
         Kokkos::single(Kokkos::PerTeam(team), [&]() {
               row_stats[iRow] = row_solver.get_stats();
               row_stats[iRow].KKT = result.second;
               row_stats[iRow].row_modified = result.first;
         });
            // TODO Ideally, timers and counters will be contained in result; this will be unnecessary}
      }
  });

}


template<class NumericTypes,class team_policy_t, class dkView,class spView,class piView,class locView, class locindexView, class statsView,
         class solver_conf_t, class kruskal_conf_t, class mode_conf_t, class scratch_space_t, class row_offset_t, class row_size_t, class  scheduleView, class mode_ind_t, class num_mode_t, class krTensor, class spIndexView>

void _damped_newton_driver_2(
         mode_conf_t  sparse_tensor_info,
         solver_conf_t config,
         kruskal_conf_t kruskal_info,
         row_offset_t  iRowOffset,
         row_size_t   nRows,
         scheduleView iScheduler,
         mode_ind_t iMode,
         num_mode_t numMode,
         piView pi,
         dkView dkdata,
         locView nonzLocs,
         locindexView nonzLocsIdx,
         krTensor kOutput,
         spView spdata,
         spIndexView spindex,
         statsView &row_stats,
         team_policy_t team_policy,
         scratch_space_t scratch_per_team )
{


   // Need to set team_policy here or there?
   Kokkos::parallel_for("damped_newton_driver_2", team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_per_team)), KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team) {
      // const auto iRow = team.league_rank()*team.team_size()+team.team_rank();
      const auto index = team.league_rank()*team.team_size()+team.team_rank();
      if( ( index + iRowOffset ) > nRows ) return;
      const auto iRow = iScheduler[index+iRowOffset].second;

      // Construct the row subproblem Solver
      RowSubProblemDampedNewton<team_policy_t, NumericTypes> row_solver = RowSubProblemDampedNewton<team_policy_t, NumericTypes>(
            /* iRow = */        iRow,
            /* iMode = */       iMode,
            /* numMode = */     numMode,
            /* team_member = */ team,
            /* config = */      config,
            /* input_info = */  sparse_tensor_info,
            /* kruskal_info = */ kruskal_info,
            /* pi = */ pi,
            /* dKdata = */ dkdata,
            /* nonz_loc = */ nonzLocs,
            /* nonz_loc_idx = */ nonzLocsIdx,
            /* sparse_data = */  spdata,
            /* sparse_index */   spindex,
            /*  KruskalTensor = */ kOutput );

      if (not row_solver.has_nonzero_entries()) {
         sparten::deep_copy(team, Kokkos::subview(dkdata, iRow, Kokkos::ALL), 0);
         // Manually set output and stats for rows that are skipped

         Kokkos::single(Kokkos::PerTeam(team), [&]() {
            row_stats[iRow].row_modified = false;
            row_stats[iRow].t_elapsed = 0.0;
            row_stats[iRow].t_compute_phi = 0.0;
            row_stats[iRow].t_search_direction = 0.0;
            row_stats[iRow].t_line_search = 0.0;
            row_stats[iRow].n_func_evals = 0;
            row_stats[iRow].n_iters = 0;
            row_stats[iRow].obj = 0.0;
         });
      } else {
         auto result = row_solver._solve();
         Kokkos::single(Kokkos::PerTeam(team), [&]() {
            row_stats[iRow] = row_solver.get_stats();
            row_stats[iRow].KKT = result.second;
            row_stats[iRow].row_modified = result.first;
         });
         // TODO Ideally, timers and counters will be contained in result; this will be unnecessary}
      }
   });
}

template<
         class NumericTypes,
         class comps_t,
         class nonz_t,
         class nmode_t,
         class imode_skip_t,
         class KruskalView1D,
         class KtensorView,
         class SpDataView,
         class SpIndexView,
         class LocIndexView,
         class scalar_t>
typename KruskalView1D::value_type _obj_likelihood_big(
      KruskalView1D &daRowVars,
      comps_t       nComps,
      nonz_t        nNonz,
      nmode_t       nModes,
      imode_skip_t  iModeSkip,
      KtensorView   &kTensor,
      SpDataView    &spData,
      SpIndexView   &spIndex,
      LocIndexView  &locIndex,
      scalar_t      log_zero_safeguard,
      bool          &func_flag
      )
{
   using kruskal_value_t = typename NumericTypes::kruskal_value_t;
   using sparse_value_t = typename NumericTypes::sparse_value_t;
   using element_index_t = typename NumericTypes::element_index_t;
   using sub_index_t = typename NumericTypes::sub_index_t;
   using team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;

   typedef Kokkos::View< kruskal_value_t *, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
         ScratchViewType;
   int scratch_size = ScratchViewType::shmem_size( nComps );


   kruskal_value_t f = 0;

  // success = true;
  // Kokkos::View<bool *> flag("myflag",1);

   Kokkos::parallel_reduce( nComps ,KOKKOS_LAMBDA( const sub_index_t &iComp, kruskal_value_t& local_f )
   {
      local_f -= daRowVars(iComp);
   }, Kokkos::Sum<kruskal_value_t>(f) );

   kruskal_value_t f_part = 0;
   kruskal_value_t dSafeGuard = log_zero_safeguard;

   auto team_policy = sparten::get_team_policy<team_policy_t>(nNonz);

   // The value in _daWorkObj explodes in a random manner.
   Kokkos::parallel_reduce(team_policy.set_scratch_size( 0, Kokkos::PerThread( scratch_size )), KOKKOS_LAMBDA(team_policy_t::member_type team, kruskal_value_t &f_inner)
   {
      // Need to compute UpsRow
      // i is cov = m_row * Pi
#if 1
      auto iNonz =  team.league_rank()*team.team_size()+team.team_rank();
      if ( iNonz >= nNonz ) return;
#else
      iNnz = team.league_rank();
#endif
      ScratchViewType mypi( team.team_scratch( 0 ), nComps );

      kruskal_value_t UpsRow=0;

      // Initialize mypi to 1
      // Kokkos::parallel_for(Kokkos::TeamThreadRange( team, nComps ), [&]( const sub_index_t iComp )
      Kokkos::parallel_for(Kokkos::ThreadVectorRange( team, nComps ), [&]( const sub_index_t iComp )
      {
         mypi(iComp) = 1;
      });

      // Compute Partial Derivative and Local Pi
      for( sub_index_t imode = 0 ; imode < nModes; ++imode )
      {
         if( imode != iModeSkip )
         {
            const auto myKdata = kTensor(imode);
            const auto index = spIndex(imode, locIndex(iNonz));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange( team, nComps ), [&]( const sub_index_t iComp )
            {
               mypi(iComp) *= myKdata(index,iComp);
            });
         }
      }

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange( team, nComps ), [&]( const sub_index_t iComp, kruskal_value_t &local_UpsRow )
      {
         local_UpsRow += daRowVars(iComp)* mypi[iComp];
      },UpsRow);

      auto mySum = UpsRow;
      auto d = static_cast<kruskal_value_t>(mySum);

      if( d < dSafeGuard ) {
         d = dSafeGuard;
         f_inner = sparten::numeric_limits<kruskal_value_t>::infinity;
      } else {
         // Need double log function
         f_inner += spData(locIndex(iNonz)) * static_cast<kruskal_value_t>(log(d));
      }

   },Kokkos::Sum<kruskal_value_t >(f_part) );

   f+=f_part;
   if( f ==  sparten::numeric_limits<kruskal_value_t>::infinity ) func_flag = false;
      // Increment function evaluation
      // ++(self().stats.n_func_evals);
   // Returns infinity if it cannot evaluate the function
   return f;
}

template<class NumericTypes,
         class dkView,
         class spView,
         class locView,
         class locindexView,
         class statsView,
         class solver_conf_t,
         class kruskal_conf_t,
         class mode_conf_t,
         class row_ind_t,
         class nnz_begin_t,
         class nnz_end_t,
         class mode_ind_t,
         class num_mode_t,
         class krTensor,
         class spIndexView,
         class FreeIndView,
         class DVarView,
         class DNVarView,
         class DPhiView,
         class DGradView,
         class DSearchView,
         class DHessianFullView,
         class DHessianAView>

void _damped_newton_driver_big(
        mode_conf_t  sparse_tensor_info,
        solver_conf_t config,
        kruskal_conf_t kruskal_info,
        row_ind_t iRow,
        nnz_begin_t nonzBegin,
        nnz_end_t nonzEnd,
        mode_ind_t iThisMode,
        num_mode_t numMode,
        dkView dkdata,
        locView nonzLoc,
        locindexView nonzLocsIdx,
        krTensor kTensor,
        spView spdata,
        spIndexView spindex,
        FreeIndView _naFreeIndices,
        DVarView _daVar,
        DNVarView _daNewVar,
        DPhiView _daPhi,
        DGradView _daGrad,
        DSearchView _daSearchDirection,
        DHessianFullView _daFreeHessian_full,
        DHessianAView     _daFreeHessian_active,
        statsView &stats )
{

   using kruskal_value_t = typename NumericTypes::kruskal_value_t;
   using sparse_value_t = typename NumericTypes::sparse_value_t;
   using element_index_t = typename NumericTypes::element_index_t;
   using sub_index_t = typename NumericTypes::sub_index_t;
   using team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;

   // Use the entire device to solve single row
   auto nComps =  kruskal_info.num_components;
   auto nNonz = nonzEnd-nonzBegin;

/*
#ifndef KOKKOS_ENABLE_CUDA
   int blk_size = 32;
#else
   int blk_size =1;
#endif
*/
   // Don't compute if the row is empty
   if( nNonz == 0 )
   {
      stats(iRow).row_modified  = false;
      stats(iRow).t_elapsed = 0.0;
      stats(iRow).KKT = 0.0;
      stats(iRow).t_compute_phi = 0.0;
      stats(iRow).t_search_direction = 0.0;
      stats(iRow).t_line_search = 0.0;
      stats(iRow).n_func_evals = 0;
      stats(iRow).n_iters = 0;
      stats(iRow).obj = 0.0;
      return;
   }

   // Use local variables fot temporary date
   Kokkos::View< element_index_t const*, /* memory traits = */ Kokkos::MemoryTraits<Kokkos::Restrict> > _nonzLoc;
   _nonzLoc = Kokkos::subview(nonzLoc, Kokkos::make_pair(nonzBegin, nonzEnd));

   // Copy the row entries
   Kokkos::deep_copy(_daVar, Kokkos::subview(dkdata, iRow, Kokkos::ALL));

   kruskal_value_t dInitialKktError = 0.0;
   kruskal_value_t dKktError = sparten::numeric_limits<kruskal_value_t>::max;
   kruskal_value_t dObj ;
   kruskal_value_t dPredictedReduction;
   kruskal_value_t dMuDamping = config.mu_initial;

   typedef Kokkos::View< kruskal_value_t *, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
         ScratchViewType;

   for (sub_index_t iIts = 0; iIts < config.max_inner_iterations; ++iIts)
   {
      // Initialize phi
      Kokkos::parallel_for(nComps, KOKKOS_LAMBDA(sub_index_t iComp)
      {
         _daPhi(iComp) = 0;
      });

      // Initialize the space for Hessian matrix
      Kokkos::parallel_for( nComps*nComps, KOKKOS_LAMBDA(sub_index_t iComp)
      {
         _daFreeHessian_full(iComp) =  0;
      });

      // New team policy to iterative over sparse tensor entries
      auto team_policy = sparten::get_team_policy<team_policy_t>(nNonz);

      Kokkos::parallel_reduce(nComps, KOKKOS_LAMBDA(const sub_index_t iComp, kruskal_value_t &local_f )
      {
         local_f -= _daVar(iComp);
      },dObj);

      kruskal_value_t f_part;

      Kokkos::Experimental::ScatterView<kruskal_value_t *> scatter_phi(_daPhi);
      Kokkos::Experimental::ScatterView<kruskal_value_t *> scatter_hessian(_daFreeHessian_full);
      int scratch_size = ScratchViewType::shmem_size( nComps*sizeof(kruskal_value_t) );


      Kokkos::parallel_reduce(team_policy.set_scratch_size( 0, Kokkos::PerThread( scratch_size )), KOKKOS_LAMBDA(team_policy_t::member_type team, kruskal_value_t &f_inner)
      {
       //  auto iNonz_start = (team.league_rank()*team.team_size()+team.team_rank()) * 32;
       //  auto iNonz_end = iNonz_start+blk_size;

         // Thread local data storage
         ScratchViewType mypi( team.team_scratch( 0 ), nComps );

         auto access_phi = scatter_phi.access();
         auto access_hessian = scatter_hessian.access();

       //  for( auto iNonz = iNonz_start; iNonz < iNonz_end; ++iNonz ) {
         auto iNonz = (team.league_rank()*team.team_size()+team.team_rank());
            if (iNonz >= nNonz) return;

            auto spval = spdata(_nonzLoc(iNonz));
            kruskal_value_t UpsRow = 0;

            // Initialize the space for pi
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nComps), [&](const sub_index_t iComp) {
               mypi[iComp] = 1;
            });

            // Compute Partial Derivative and Local Pi
            for (sub_index_t imode = 0; imode < numMode; ++imode) {
               if (imode != iThisMode) {
                  const auto myKdata = kTensor(imode);
                  const auto index = spindex(imode, _nonzLoc(iNonz));
                  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nComps),
                                       [&](const sub_index_t iComp) {
                                          mypi[iComp] *= myKdata(index, iComp);
                                       });
               }
            }

            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, nComps),
                                    [&](const sub_index_t iComp, kruskal_value_t &localUpsRow) {
                                       localUpsRow += _daVar(iComp) * mypi[iComp];
                                    }, UpsRow);


            // Computing objective function
            auto d = static_cast<kruskal_value_t>(UpsRow);
            if (d < config.log_zero_safeguard) {
               d = config.log_zero_safeguard;
            }

            // Eliminate small values. Future version handles this issue in a better way.
            if (UpsRow < config.eps_div_zero_grad) {
               UpsRow = config.eps_div_zero_grad;
            }

            // For Objective likelihood
            f_inner += spval * static_cast<kruskal_value_t>(log(d));
            // For Phi
            auto ValForPhi = spval / UpsRow;

            // Need atomic avoiding techniques for CPUs (CPU should handle multiple of nonzero entries in a single parallel_for)

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nComps), [&](const sub_index_t iComp) {
               // Kokkos::atomic_add(&_daPhi(iComp), mypi[iComp] * ValForPhi);
               access_phi(iComp) += mypi[iComp] * ValForPhi;

            });

            auto sqVal = UpsRow * UpsRow;
            UpsRow = spval / max(sqVal, config.eps_div_zero_grad);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nComps * nComps), [&](const sub_index_t iComp) {
               auto localVal = mypi[iComp / nComps] * UpsRow * mypi[iComp % nComps];
               access_hessian(iComp) += localVal;
               // Kokkos::atomic_add(&daFreeHessian_full(iComp), localVal);
            });
       //  }
      },f_part);

      dObj += f_part;

      // Increment function evaluation
      ++(stats[iRow].n_func_evals);

      // Build phi and Hessian
      Kokkos::Experimental::contribute(_daPhi,scatter_phi);
      Kokkos::Experimental::contribute(_daFreeHessian_full,scatter_hessian);

      //
      //  The rest of the code does not require much computation. Could be done faster on single core CPU
      //

      dKktError = 0.0;
      team_policy = sparten::get_team_policy<team_policy_t>(nComps);
      // Create a team policy for KKTerror
      Kokkos::parallel_reduce(nComps, KOKKOS_LAMBDA(const sub_index_t  iComp,  kruskal_value_t &dKktErrorLocal ) {


         _daGrad(iComp) = static_cast<kruskal_value_t >(1.0) - _daPhi(iComp);

         kruskal_value_t d = min(_daGrad(iComp), _daVar(iComp));
         dKktErrorLocal= max( abs(d), dKktErrorLocal);

      }, Kokkos::Max<kruskal_value_t>{dKktError} );  // Why is it not working?

      if ( iIts == 0 )
      {
         dInitialKktError = dKktError;
      }

      if (dKktError < config.row_tolerance)
      {
         stats[iRow].KKT = dInitialKktError;
         // Return since the stop tolerance is met.
         if (iIts > 0)
         {
            // Variables were modified to achieve the stop tolerance.
            Kokkos::deep_copy( Kokkos::subview( dkdata, iRow, Kokkos::ALL), _daVar );
            stats[iRow].row_modified  =true;
            stats[iRow].obj = dObj;

            return;
         } else {
            stats[iRow].row_modified  = false;
            stats[iRow].obj = dObj;
            return ;
         }
      }

      //
      // This part is used to compute active set
      //
      // Start Norm Calculation
      //

      // Initialize the Timer
#ifndef KOKKOS_ENABLE_CUDA
      Kokkos::Timer timer;
#else
      //      double timer = 0.0;
#endif
      kruskal_value_t search_dir_time = 0.0;

      kruskal_value_t dSumSq = 0.0;
      team_policy = sparten::get_team_policy<team_policy_t>(nComps);
      // Create a team policy for KKTerror
      Kokkos::parallel_reduce(nComps, [=] KOKKOS_FUNCTION(const sub_index_t iComp,  kruskal_value_t &sumSq ) {
         kruskal_value_t d = _daVar(iComp) - _daGrad(iComp);
         if (d > 0.0)
         {
            sumSq += (_daVar(iComp) - d) * (_daVar(iComp) - d);
         }
      }, Kokkos::Sum<kruskal_value_t>(dSumSq) );

      kruskal_value_t dEps = static_cast<kruskal_value_t >(sqrt(static_cast<double>(dSumSq)));

      dEps = sparten::min(config.eps_active_set, dEps);

      // Computing Active Set
      // Create Team Policy for this

      sub_index_t nNumFree = 0;
      Kokkos::parallel_reduce(nComps, KOKKOS_LAMBDA(const sub_index_t iComp, sub_index_t& lnNumFree ) {


         if ((_daGrad(iComp) > 0.0) && (_daVar(iComp) == 0.0)) {
             _daSearchDirection(iComp) = 0.0;
             _naFreeIndices(iComp) = 0;
         } else if( (_daGrad(iComp) > 0.0) && (_daVar(iComp) <= dEps) ) {
            _daSearchDirection(iComp) = - _daGrad(iComp);
            _naFreeIndices(iComp) = 0;
         } else {
            lnNumFree ++;
            _naFreeIndices(iComp) = 1;

         }
      }, Kokkos::Sum<sub_index_t >(nNumFree) );


      // Replaced by parallel scan
      Kokkos::parallel_for(1, KOKKOS_LAMBDA(const sub_index_t ii)
      {
         int nUpd = 0;
         for (sub_index_t iComp=0; iComp < nComps; ++iComp ){
            if (_naFreeIndices(iComp) == 1) {
               _naFreeIndices(nUpd) = iComp;
               nUpd++;
            }
         }
      });

      //
      // Copy full to active set
      //
      // Alternatively, it is possible to develop parallel Chokesky with active set indices.
      //

      Kokkos::parallel_for(nNumFree, KOKKOS_LAMBDA(const sub_index_t indCol) {

         const sub_index_t indColFull = _naFreeIndices(indCol);

         for (sub_index_t indRow = 0; indRow < nNumFree; ++indRow) {
            const sub_index_t indRowFull = _naFreeIndices(indRow);
            _daFreeHessian_active(indCol * nNumFree + indRow) = _daFreeHessian_full(indColFull * nComps + indRowFull);
         }
      });

      // Compute a search direction based active and free variables,
      // using a damped Newton step for the free variables.
      //auto compute_search_dir_bigrow  = [&](auto nNumFree, , auto nj, auto Nj)
      {
         // Shift the diagonal values
         auto& daFreeSearch  = _daFreeHessian_full;
         Kokkos::parallel_for (  nNumFree, KOKKOS_LAMBDA (sub_index_t iNum  )
         {
            _daFreeHessian_active(iNum * nNumFree + iNum) +=  dMuDamping;
            daFreeSearch(iNum) = -_daGrad(_naFreeIndices(iNum));
         });

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
         int posvSucceeded = 0;
         Kokkos::parallel_reduce (  1, KOKKOS_LAMBDA (sub_index_t iNum, int &local_flag  )
         {
            local_flag += spartenBlas::posv(nNumFree,_daFreeHessian_active,daFreeSearch);
         }, posvSucceeded);

         if( posvSucceeded == 0)
         {
            Kokkos::parallel_for (  nNumFree, KOKKOS_LAMBDA (sub_index_t iNum )
            {
               _daSearchDirection(_naFreeIndices(iNum)) = -_daGrad(_naFreeIndices(iNum));
            });
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

         Kokkos::parallel_for (  nNumFree, KOKKOS_LAMBDA(sub_index_t iNum )
         {
            _daSearchDirection(_naFreeIndices(iNum)) = daFreeSearch(iNum);
         });

         auto& daSolution  = _daFreeHessian_full; // Use the space of Hessian_full

         // We need KokkosKernel TRMV (Dense Triangular Solve) for "all" and "team" versions
         Kokkos::parallel_for (  1, KOKKOS_LAMBDA (sub_index_t i ) {
            spartenBlas::trmv('L', 'T', 'N', nNumFree, _daFreeHessian_active, nNumFree, daSolution, static_cast<sub_index_t>(1));
            spartenBlas::trmv('L', 'N', 'N', nNumFree, _daFreeHessian_active, nNumFree, daSolution, static_cast<sub_index_t>(1));
         });


         kruskal_value_t Q0, Q1;
         Q0 = sparten::dot_product_left_with_index(  _daSearchDirection, daSolution, _naFreeIndices, nNumFree );
         Q1 = sparten::dot_product_top(  _daSearchDirection, _daGrad, _naFreeIndices, nNumFree );

         dPredictedReduction = (0.5 * Q0) + Q1;

         // stats.t_search_direction = elapsed_time;
      }
      // MDRangePolicy<Kokkos::Rank<2>>({0,0},{Ni,Nj}), Kokkos_LAMBDA( int i, int j) ... from 0 to Ni and ) to Nj
      search_dir_time += stop_timer(timer);
      stats[iRow].t_search_direction += search_dir_time;

      // Perform a line search.
      kruskal_value_t dAred;

      {
         bool flag = true;
         kruskal_value_t dStepLength = 1.0;
         sub_index_t nNumSteps = 0;
         kruskal_value_t dNewL1Norm;
         kruskal_value_t dObj1;
         kruskal_value_t  dObjNew;

#ifndef KOKKOS_ENABLE_CUDA // TODO Get rid of these ugly macros
         Kokkos::Timer line_search_timer;
#else
	 //         double timer = 0.0;
#endif

         double elapsed_time = 0.0;
         // Negative objective reduction means the unit step failed.
         kruskal_value_t dUnitStepAred = -1.0;

         // f_old = -1 * tt_loglikelihood_row(isSparse, x_row, m_old, Pi);
         // kruskal_value_t

         auto dObjOld = - _obj_likelihood_big<NumericTypes>(
               _daVar,
               nComps,
               nNonz,
               numMode,
               iThisMode,
               kTensor,
               spdata,
               spindex,
               _nonzLoc,
               config.log_zero_safeguard,
               flag);

         if (flag == false ) dObjOld = sparten::numeric_limits<kruskal_value_t>::max;
         ++(stats[iRow].n_func_evals);
 //        reset_timer(timer); // don't measure time in _obj_likelihood

         while (nNumSteps < config.max_backtrack_steps)
         {
            flag = true;
            // Compute a new projected iterate at alpha times the step length.
            dNewL1Norm = 0.0;

            Kokkos::parallel_reduce( nComps , KOKKOS_LAMBDA( sub_index_t iComp, kruskal_value_t &newL1Norm )
            {
               _daNewVar(iComp) = _daVar(iComp) + (dStepLength * _daSearchDirection(iComp));
#if 0
               if (_daNewVar(iComp) < 0.0)
               {
                  _daNewVar(iComp) = 0.0;
               } else {
                  newL1Norm += _daNewVar(iComp);
               }
#endif
               _daNewVar(iComp) *= (_daNewVar(iComp) >= 0.0);
               newL1Norm += _daNewVar(iComp);

            }, Kokkos::Sum<kruskal_value_t>(dNewL1Norm) );


            // TODO this could be part of the previous loop
            // Compute the gradient times the new step.
            kruskal_value_t dGradDotNew = 0.0;
            Kokkos::parallel_reduce( nComps ,KOKKOS_LAMBDA(sub_index_t iComp, kruskal_value_t& gradDotNew) {
               gradDotNew += _daGrad(iComp) * (_daNewVar(iComp) - _daVar(iComp));
            }, Kokkos::Sum<kruskal_value_t>(dGradDotNew) );

            // Check that the step is a descent direction.
            // Also reject the step if its length is nearly zero, because the
            // objective value is unreliable due to logarithm functions.
            if ((dGradDotNew > 0.0) || (dNewL1Norm < config.min_variable_nonzero_tolerance))
            {
               // No descent.  Treat this as a failed step.
               dObjNew = sparten::numeric_limits<kruskal_value_t>::max;
               if (nNumSteps == 0) {
                  dObj1 = dObjNew;
               }
            }
            else
            {
             //  elapsed_time += stop_timer(timer);
               // Compute the new value for the objective function

               dObjNew = -_obj_likelihood_big<NumericTypes>(_daNewVar,nComps,
                                              nNonz,
                                              numMode,
                                              iThisMode,
                                              kTensor,
                                              spdata,
                                              spindex,
                                              _nonzLoc,
                                              config.log_zero_safeguard,
                                              flag);
             //  reset_timer(timer);

               if (nNumSteps == 0)
               {
                  // Only update the dUnitStepAred the first time???
                  dObj1 = dObjNew;

               }
               // Break out if there is sufficient decrease.
               if ( (flag == true) &&
               (dObjNew <= dObjOld + dGradDotNew * config.suff_decrease_tolerance))
               {
                  break;
               }

            }

            // Backtrack the step length.
            dStepLength = dStepLength * config.step_reduction_factor;

            // Increment the number of steps
            ++nNumSteps;
         }

         if ( fabs(dObj1) > sparten::numeric_limits<kruskal_value_t>::max)
         {
            dObj1 = dObjOld;
         }
         /*
         if (nRetCode != -2)
         {
            nRetCode = nNumSteps;
         }
          */
         // Test if the line search failed.
         if (((nNumSteps >= config.max_backtrack_steps) && (dObjNew > dObjOld)) ||
             (dNewL1Norm < config.min_variable_nonzero_tolerance))
         {
            // Fall back on a multiplicative update step (scaled steepest descent).
            // Experiments indicate it works better than a unit step in the direction
            // of steepest descent.
            // One way of computing the multiplicative update is to use
            // timeComputePhi = 1 - grad; however, it suffers from round-off error when timeComputePhi
            // is tiny.  The more accurate way computes directly from timeComputePhi.
            Kokkos::parallel_for( nComps, KOKKOS_LAMBDA(sub_index_t iComp)
            {
               _daNewVar(iComp) = _daVar(iComp) * _daPhi(iComp);
                     // std::cout << "DaNewRowVars2( " << iComp << " ) " << daNewRowVars(iComp) << std::endl;
            });

           // elapsed_time += stop_timer(timer);
            dObjNew = _obj_likelihood_big<NumericTypes>(_daVar,nComps,
                                          nNonz,
                                          numMode,
                                          iThisMode,
                                          kTensor,
                                          spdata,
                                          spindex,
                                          _nonzLoc,
                                          config.log_zero_safeguard,
                                          flag);

            ++(stats[iRow].n_func_evals);

#if 0
            Kokkos::single( Kokkos::PerTeam( team_member ), [&] () {
      });
#endif
            // Negative objective reduction means the unit step failed.
            dUnitStepAred = -1.0;
         } else {
            dUnitStepAred = dObjOld-dObj1;
         }
         dAred = dUnitStepAred;
         elapsed_time += stop_timer(line_search_timer);
         stats[iRow].t_line_search += elapsed_time;
      }

      // Copy the new data back to the original row
      Kokkos::parallel_for( nComps, KOKKOS_LAMBDA (sub_index_t iComp) {
         _daVar(iComp) = _daNewVar(iComp);
      });

      if (dPredictedReduction == 0.0)
      {
         dMuDamping = 10.0 * dMuDamping; // Need to re-initialized for the next row
      } else {
         kruskal_value_t dRho = dAred / -dPredictedReduction;

         if (dRho < config.damping_increase_tolerance  ) {
            dMuDamping = config.damping_increase_factor * dMuDamping;
         } else if (dRho > config.damping_decrease_tolerance ) {
           dMuDamping = config.damping_decrease_factor * dMuDamping;
         }
      }
      // Update the iteration counts
      ++(stats[iRow].n_iters);
   } // End of the iteration

   // Finished iterations
   Kokkos::deep_copy( Kokkos::subview(dkdata, iRow, Kokkos::ALL), _daVar);

#if 0
   elapsed_time = stop_timer(timer);
   stats.t_elapsed = elapsed_time;
#endif
   stats[iRow].KKT = dInitialKktError;
   stats[iRow].obj = dObj;
   stats[iRow].row_modified = true;
   return ;


#if 0
      if (not row_solver.has_nonzero_entries()) {
         sparten::deep_copy(team, Kokkos::subview(dkdata, iRow, Kokkos::ALL), 0);
         // Manually set output and stats for rows that are skipped

         Kokkos::single(Kokkos::PerTeam(team), [&]() {
               row_stats[iRow].row_modified = false;
               row_stats[iRow].t_elapsed = 0.0;
               row_stats[iRow].t_compute_phi = 0.0;
               row_stats[iRow].t_search_direction = 0.0;
               row_stats[iRow].t_line_search = 0.0;
               row_stats[iRow].n_func_evals = 0;
               row_stats[iRow].n_iters = 0;
               row_stats[iRow].obj = 0.0;
         });

      } else {

         auto result = row_solver._solve();
         Kokkos::single(Kokkos::PerTeam(team), [&]() {
               row_stats[iRow] = row_solver.get_stats();
               row_stats[iRow].KKT = result.second;
               row_stats[iRow].row_modified = result.first;
         });

         // TODO Ideally, timers and counters will be contained in result; this will be unnecessary}
      }
#endif

}



template <class NumericTypes>
void
DampedNewtonCpApr<NumericTypes>::compute(
	kruskal_tensor_t& kruskalOutput,
	sparse_tensor_t const& sparseInput
)
{
	if (_config.scheduling == 0)
	{
		using team_policy_t = Kokkos::TeamPolicy<>;
		_compute_with_policy<team_policy_t>(kruskalOutput, sparseInput);
	}
	else if (_config.scheduling == 1)
	{
		using team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
		_compute_with_policy<team_policy_t>(kruskalOutput, sparseInput);
	}
}




template <class NumericTypes>
template <class TeamPolicyType>
void
DampedNewtonCpApr<NumericTypes>::_compute_with_policy(
	kruskal_tensor_t& kruskalOutput,
	sparse_tensor_t const& sparseInput
	) {

   using host_memory_space = typename Kokkos::DefaultHostExecutionSpace::memory_space;
   using device_memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
   using team_policy_t2 = TeamPolicyType;
   using team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;

   typedef Kokkos::pair<sub_index_t, sub_index_t> mypair;

   struct {
      bool operator()(const mypair &l, const mypair &r) const { return l.first > r.first; }
   } customMore;

   if( _config.solver_option != 1 && _config.solver_option != 2) {
     std::cout << "PDNR Solver Option " << _config.solver_option << "not supported" << std::endl;;
   }

   Log &log = Log::new_log();

   std::cout << std::setfill('-');
   std::cout << std::setw(70);
   std::cout << std::left;
   if (log.get_verbosity() > 1)
       std::cout << std::setw(90);
   std::cout << "------------------------- CP-APR Damped-Newton " << std::endl;

   // Readability aliases:
   auto nElement = sparseInput.get_nElement();
   auto nComp = kruskalOutput.get_nComponent();
   auto nDim = sparseInput.get_nDim();

   typedef struct {
      Kokkos::View<sub_index_t *> FreeIndice;
      Kokkos::View<kruskal_value_t *> Var;
      Kokkos::View<kruskal_value_t *> NewVar;
      Kokkos::View<kruskal_value_t *> Phi;
      Kokkos::View<kruskal_value_t *> Grad;
      Kokkos::View<kruskal_value_t *> Search;
      Kokkos::View<kruskal_value_t *> FreeHessianFull;
      Kokkos::View<kruskal_value_t *> FreeHessianActive;
   } TmpViewsPDNR;

   TmpViewsPDNR bigTemp;
   Kokkos::View<sub_index_t *> _naFreeIndices("FeeIndex", nComp);
   Kokkos::View<kruskal_value_t *> _daVar("daVar", nComp);
   Kokkos::View<kruskal_value_t *> _daNewVar("daVarOld", nComp);
   Kokkos::View<kruskal_value_t *> _daPhi("daPhi", nComp);
   Kokkos::View<kruskal_value_t *> _daGrad("daGrad", nComp);
   Kokkos::View<kruskal_value_t *> _daSearchDirection("daSearch", nComp);
   Kokkos::View<kruskal_value_t *> _daFreeHessian_full("HessianFull", nComp * nComp);
   Kokkos::View<kruskal_value_t *> _daFreeHessian_active("HessianActive", nComp * nComp);
   bigTemp.FreeIndice = _naFreeIndices;
   bigTemp.Var = _daVar;
   bigTemp.NewVar = _daNewVar;
   bigTemp.Phi = _daPhi;
   bigTemp.Grad = _daGrad;
   bigTemp.Search = _daSearchDirection;
   bigTemp.FreeHessianFull = _daFreeHessian_full;
   bigTemp.FreeHessianActive = _daFreeHessian_active;

   Kokkos::Timer cpAprTimer;
   kruskal_value_t computeTime = 0, piTime = 0, sparseTime = 0, denseTime = 0;

   auto kktErrorPerDim = sparten::make_unique_array<kruskal_value_t>(sparseInput.get_nDim());

   // Check requirements
   {
      Kokkos::Timer checkRequirementsTimer;
      base_t::check_requirements(kruskalOutput, sparseInput);
      log.print("CpAprDampedNewton::check_requirements: " + std::to_string(checkRequirementsTimer.seconds()) + " s",
                Log::DEBUG_2);
   }

   // Allocate pi, which will be removed soon
   Kokkos::View<kruskal_value_t **, device_memory_space> pi("pi matrix temporary", 1, 1);

   // Allocate and reorder nonzero locations and indices
   using nonzero_locs_view_t = Kokkos::View<element_index_t *, device_memory_space>;
   using nonzero_locs_idx_view_t = Kokkos::View<sub_index_t *, device_memory_space>;
   auto nonzLocs = sparten::make_unique_array<nonzero_locs_view_t>(nDim);
   auto nonzLocsIdx = sparten::make_unique_array<nonzero_locs_idx_view_t>(nDim);

   // Want to get statistics about the number of nonzero entries per row per mode
   std::vector<sub_index_t> max_nonz_per_row(nDim);
   std::vector<sub_index_t> min_nonz_per_row(nDim);
   std::vector<kruskal_value_t> mean_nonz_per_row(nDim);
   std::vector<kruskal_value_t> stdev_nonz_per_row(nDim);
   std::stringstream message_release;
   message_release << "\nSparse Tensor Nonzeros Per Row";


   std::vector<Kokkos::View<mypair *>> scheduler(nDim);
   std::vector<std::vector<mypair>> scheduler_offset(nDim);
   std::vector<sub_index_t > nLargeRows(nDim);
   std::vector<std::vector<sub_index_t >> locOffset(nDim);
   {
      if( _config.solver_option == 1 )
      {
         for (sub_index_t iDim = 0; iDim < nDim; ++iDim) {
            auto nRow = kruskalOutput.get_factor_matrix_nRow_host(iDim);
            locOffset[iDim].resize(nRow + 1);
            Kokkos::resize(scheduler[iDim], nRow);
            typename Kokkos::View<mypair *>::HostMirror scheduler_host = Kokkos::create_mirror_view(scheduler[iDim]);
            nonzLocs[iDim] = nonzero_locs_view_t{Kokkos::view_alloc(device_memory_space{}), nElement};
            nonzLocsIdx[iDim] = nonzero_locs_idx_view_t{Kokkos::view_alloc(device_memory_space{}), nRow + 1};

            auto host_nonzLocs = Kokkos::create_mirror_view(nonzLocs[iDim]);
            auto host_nonzLocsIdx = Kokkos::create_mirror_view(nonzLocsIdx[iDim]);
            base_t::reorder_nonzeros_host(
               nRow, nElement, Kokkos::subview(sparseInput.get_indices_host_view(), iDim, Kokkos::ALL), host_nonzLocs, host_nonzLocsIdx
            );
            for (sub_index_t i = 0; i <= nRow; ++i) {
               locOffset[iDim][i] = host_nonzLocsIdx[i];
            }

            Kokkos::deep_copy(/*dst=*/nonzLocs[iDim], /*src=*/host_nonzLocs);
            Kokkos::deep_copy(/*dst=*/nonzLocsIdx[iDim], /*src=*/host_nonzLocsIdx);

            // Use Welford's algorithm to compute sample stats for each mode in a single pass
            sub_index_t iRow = 0;
            sub_index_t nnz = 0;
            sub_index_t max = sparten::numeric_limits<sub_index_t>::min;
            sub_index_t min = sparten::numeric_limits<sub_index_t>::max;
            kruskal_value_t tmp = 0.0;
            kruskal_value_t sum = 0.0;
            kruskal_value_t mean = 0.0;
            message_release << "\nMode: " << iDim;

            while (iRow < nRow) {
               message_release << "\n" << nnz;
               nnz = host_nonzLocsIdx[iRow+1] - host_nonzLocsIdx[iRow];
               ++iRow;
               tmp = nnz - mean;
               sum += tmp * tmp * (iRow - 1) / iRow;
               mean += tmp / iRow;
               if (nnz > max) {
                  max = nnz;
               } else if (nnz < min) {
                  min = nnz;
               }
            }
            message_release << "\n";
            max_nonz_per_row[iDim] = max;
            min_nonz_per_row[iDim] = min;
            mean_nonz_per_row[iDim] = mean;
            stdev_nonz_per_row[iDim] = std::sqrt(sum / iRow);
         }
      } else if ( _config.solver_option == 2 ) {
         std::vector<std::vector<mypair>> bucket(nDim);
         for (sub_index_t iDim = 0; iDim < nDim; ++iDim) {
            auto nRow = kruskalOutput.get_factor_matrix_nRow_host(iDim);

            bucket[iDim].resize(nRow);
            locOffset[iDim].resize(nRow + 1);
            Kokkos::resize(scheduler[iDim], nRow);
            typename Kokkos::View<mypair *>::HostMirror scheduler_host = Kokkos::create_mirror_view(scheduler[iDim]);
            nonzLocs[iDim] = nonzero_locs_view_t{Kokkos::view_alloc(device_memory_space{}), nElement};
            nonzLocsIdx[iDim] = nonzero_locs_idx_view_t{Kokkos::view_alloc(device_memory_space{}), nRow + 1};

            auto host_nonzLocs = Kokkos::create_mirror_view(nonzLocs[iDim]);
            auto host_nonzLocsIdx = Kokkos::create_mirror_view(nonzLocsIdx[iDim]);
            base_t::reorder_nonzeros_host(
               nRow, nElement, Kokkos::subview(sparseInput.get_indices_host_view(), iDim, Kokkos::ALL), host_nonzLocs, host_nonzLocsIdx
            );
            for (sub_index_t i = 0; i <= nRow; ++i) {
               locOffset[iDim][i] = host_nonzLocsIdx[i];
            }

            Kokkos::deep_copy(/*dst=*/nonzLocs[iDim], /*src=*/host_nonzLocs);
            Kokkos::deep_copy(/*dst=*/nonzLocsIdx[iDim], /*src=*/host_nonzLocsIdx);

            // Use Welford's algorithm to compute sample stats for each mode in a single pass
            sub_index_t iRow = 0;
            sub_index_t nnz = 0;
            sub_index_t max = sparten::numeric_limits<sub_index_t>::min;
            sub_index_t min = sparten::numeric_limits<sub_index_t>::max;
            kruskal_value_t tmp = 0.0;
            kruskal_value_t sum = 0.0;
            kruskal_value_t mean = 0.0;
            message_release << "\nMode: " << iDim;

            while (iRow < nRow) {
               message_release << "\n" << nnz;
               nnz = host_nonzLocsIdx[iRow+1] - host_nonzLocsIdx[iRow];
               bucket[iDim][iRow] = {nnz, iRow};
               ++iRow;
               tmp = nnz - mean;
               sum += tmp * tmp * (iRow-1) / iRow;
               mean += tmp / iRow;
               if (nnz > max) {
                  max = nnz;
               } else if (nnz < min) {
                  min = nnz;
               }
            }
            message_release << "\n";
            max_nonz_per_row[iDim] = max;
            min_nonz_per_row[iDim] = min;
            mean_nonz_per_row[iDim] = mean;
            stdev_nonz_per_row[iDim] = std::sqrt(sum / iRow);


            // Create rows sorted by the number of nonzero entries
            std::sort(bucket[iDim].begin(), bucket[iDim].end(), customMore);

         //
         // Create scheduler, which is reserved for future branch
         //
#if 0
            int buffSize=0;
            int mymax_nonz_row = bucket[iDim][0].first;
            for ( int i = 0; i < nRow; ++i )
            {
               if( buffSize > 1048576)  // 16777216
               {
                  scheduler_offset[iDim].push_back( {i,mymax_nonz_row} );
                  buffSize = 0;
                  mymax_nonz_row =  bucket[iDim][i].first;
               }
               scheduler_host[i] = bucket[iDim][i];
               buffSize+= bucket[iDim][i].first;
            }
            scheduler_offset[iDim].push_back({nRow,mymax_nonz_row});
            Kokkos::deep_copy(scheduler[iDim],scheduler_host);
#endif
         //
            sub_index_t largeRowCount=0;
            sub_index_t total_nonz_dense=0;
            for (int i = 0; i < nRow; ++i) {

               if( bucket[iDim][i].first > _config.threshold_nonz_per_row )
               {
                  scheduler_offset[iDim].push_back(bucket[iDim][i]);
                  largeRowCount++;
                  total_nonz_dense += bucket[iDim][i].first;
               }
               scheduler_host[i] = bucket[iDim][i];
            }
            nLargeRows[iDim] = largeRowCount;;
            Kokkos::deep_copy(scheduler[iDim], scheduler_host);
         }
      }
   }
   message_release << "\n";
   log.print(message_release.str(), Log::DEBUG_2);

   // May not be used in future
   auto sparse_tensor_info =
      SparseTensorInfo<NumericTypes>{
         0
   };

   // compute_max_nonzero_per_mode<sub_index_t >( nDim, max_nonz_per_row,  nonzLocsIdx, sparseInput.get_size_dim());
   auto kruskal_info =
           KruskalTensorInfo<NumericTypes>{
      /* num_components = */ kruskalOutput.get_nComponent()
   };

   std::vector<kruskal_value_t> kktViolationPerMode(nDim);
   auto outer_iter              = 0;
   bool converged               = false;
   element_index_t func_evals   = 0;
   element_index_t inner_iters  = 0;
   kruskal_value_t obj          = 0.0;
   kruskal_value_t errorNorm    = sparten::numeric_limits<kruskal_value_t>::max;
#ifndef KOKKOS_ENABLE_CUDA
	 sub_index_t num_threads      = Kokkos::DefaultExecutionSpace::impl_thread_pool_size();
#else
	 sub_index_t num_threads      = 1;
#endif
   auto timing_data = DampedNewtonTimingData{};

   // Set up Factor Matrices accessible from Accelerators
   auto numMode = kruskalOutput.get_nDim();
   Kokkos::View<FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp", numMode);
   auto kt_host = Kokkos::create_mirror_view(ktensor);
   for (sub_index_t i = 0; i < numMode; ++i) {
      kt_host(i) = kruskalOutput.get_factor_matrix(i);
   }
   Kokkos::deep_copy(ktensor, kt_host);


   //
   // Set up the array to get the information of individual rows
   // Use the largest mode size
   //
   // Get the largest mode size
   int maxModeSize =0 ;
   for (auto iDim : kruskalOutput.get_dim() )
   {
      maxModeSize = (maxModeSize > kruskalOutput.get_nPerMode(iDim))  ? maxModeSize : kruskalOutput.get_nPerMode(iDim);
   }

   Kokkos::View<DampedNewtonRowStats<element_index_t, kruskal_value_t> *> row_stats("Stats", maxModeSize);
   typename Kokkos::View<DampedNewtonRowStats<element_index_t, kruskal_value_t> *>::HostMirror row_stats_host = Kokkos::create_mirror_view(
           row_stats);

   std::ostringstream msg;
   msg << "  i";
   msg << "         kkt-violation";
   msg << "          -log-likelihood";
   msg << "      time (s)";
   if (log.get_verbosity() > 1)
   {
       msg << "  inner iters";
       msg << "   func evals";
   }
   log.print(msg.str(), Log::RELEASE);

   // Initialize the cumulative time for outer loop
   kruskal_value_t outerLoopTime = 0;
   while(outer_iter < _config.max_outer_iterations && not converged)
   {
      Kokkos::Timer outerLoopTimer;
      auto anyRowVarsModified = false;

      for(auto iDim : sparseInput.get_dim()) {
         obj = 0;

         //auto row_stats = Kokkos::subview(row_stats,std::make_pair(0,kruskalOutput.get_nPerMode(iDim));

         // Distribute weights
         {
         // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
         //auto _ = ScopedTimerContribution(timing_data.distributeWeights);
         // TODO make this work on the device

            kruskalOutput.distribute_weights_to_factor_matrices(iDim);
         }

         // Pi is no longer computed
         {
#ifndef KOKKOS_ENABLE_CUDA
            Kokkos::Timer PiTimer;
#endif
         // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
#ifndef KOKKOS_ENABLE_CUDA
            piTime += PiTimer.seconds();
#endif
         }

         // The task parallel part
         {
            ScopedTimerContribution _t(timing_data.taskParallel);
            kruskal_value_t maxKktViolationForMode = 0.0;

            auto scratch_per_team = RowSubProblemDampedNewton<team_policy_t, NumericTypes>::required_scratch_size(
                  _config, kruskal_info, kruskal_info.num_components);

            sparse_tensor_info.max_nonzeros_per_row = max_nonz_per_row[iDim];

            // Parallel Computation Starts here
#if 0
            // No custom reducer for the time being
            using reducer_type = CpAprCustomReducer<kruskal_value_t>;
            using reducer_value_t = typename reducer_type::value_type;
            auto modeValues = reducer_value_t{};
#endif

            // TODO this is a workaround that will be much cleaner when Kokkos has a combined reducers feature (e.g. tuple)
            // TODO see: Kokkos issue #2013
            // For now, we pull out what we want by passing vector, row_stats, with a structure of timers and counters for each row
            // Then the results are coalesced below
            auto kData = kruskalOutput.get_factor_matrix(iDim);
            Kokkos::Timer SparseTimer;

            if( _config.solver_option == 1 )
            {
               auto team_policy = sparten::get_team_policy<team_policy_t>(kruskalOutput.get_nPerMode(iDim));
               _damped_newton_driver_1<NumericTypes, team_policy_t>(sparse_tensor_info, _config, kruskal_info, iDim, numMode,
                                                                    pi,kruskalOutput.get_factor_matrix(iDim), nonzLocs[iDim],
                                                                    nonzLocsIdx[iDim], ktensor, sparseInput.get_data_view(),
                                                                    sparseInput.get_indices_view(),
                                                                    row_stats, team_policy, scratch_per_team);
               Kokkos::fence(); // to get reduction data
               sparseTime += SparseTimer.seconds();
               deep_copy(row_stats_host, row_stats);

            } else if ( _config.solver_option == 2 ) {
               auto nRows = kruskalOutput.get_nPerMode(iDim);
               auto team_policy = sparten::get_team_policy<team_policy_t>(nRows-nLargeRows[iDim]);
               // Run bunch of small team kernels
               _damped_newton_driver_2<NumericTypes, team_policy_t>(sparse_tensor_info, _config, kruskal_info,
                                                                 nLargeRows[iDim], nRows, scheduler[iDim], iDim, numMode,
                                                                 pi,kruskalOutput.get_factor_matrix(iDim), nonzLocs[iDim],
                                                                 nonzLocsIdx[iDim], ktensor, sparseInput.get_data_view(),
                                                                 sparseInput.get_indices_view(), row_stats, team_policy, scratch_per_team);
              // Kokkos::fence(); // to get reduction data

               sparseTime += SparseTimer.seconds();
               deep_copy(row_stats_host, row_stats);
               Kokkos::Timer DenseTimer;
               // Run big kernels
               for (int ii = 0 ; ii < nLargeRows[iDim]; ++ii )
               {
                  const int iRow = scheduler_offset[iDim][ii].second;
                  _damped_newton_driver_big<NumericTypes>(
                        sparse_tensor_info, _config, kruskal_info, iRow, locOffset[iDim][iRow], locOffset[iDim][iRow+1], iDim, numMode,
                        kruskalOutput.get_factor_matrix(iDim), nonzLocs[iDim], nonzLocsIdx[iDim], ktensor, sparseInput.get_data_view(),
                        sparseInput.get_indices_view(), bigTemp.FreeIndice, bigTemp.Var, bigTemp.NewVar, bigTemp.Phi,
                        bigTemp.Grad, bigTemp.Search, bigTemp.FreeHessianFull, bigTemp.FreeHessianActive, row_stats_host );

                  Kokkos::fence(); // to get reduction data
               }
               denseTime += DenseTimer.seconds();
            }


            // kktViolationPerMode[iDim] = modeValues.max_kkt_violation ;
            // anyRowVarsModified = anyRowVarsModified || modeValues.row_vars_modified;
            // TODO implement with parallel for
            // Aggregate stats from each row per outer iteration
            // Custom Reducer is not working

            for (sub_index_t j = 0; j < kruskalOutput.get_nPerMode(iDim); j++) {
               timing_data.row.time_elapsed += row_stats_host[j].t_elapsed/num_threads;
               timing_data.row.compute_phi += row_stats_host[j].t_compute_phi;
               timing_data.row.search_direction += row_stats_host[j].t_search_direction/num_threads;
               timing_data.row.line_search += row_stats_host[j].t_line_search;
               timing_data.row.failed_search += row_stats_host[j].t_failed_search;
               func_evals += row_stats_host[j].n_func_evals;
               inner_iters += row_stats_host[j].n_iters;
               obj += row_stats_host[j].obj;
               maxKktViolationForMode = std::max<kruskal_value_t>(maxKktViolationForMode, row_stats_host[j].KKT);
               anyRowVarsModified = anyRowVarsModified || row_stats_host[j].row_modified;
            }
            kktViolationPerMode[iDim] = maxKktViolationForMode;
         }

         {

            // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
            kruskalOutput.normalize(iDim);

         }
      }

#ifdef  _SPARTEN_DEBUG_1
      std::cout << kruskalOutput.show();
#endif
      errorNorm= * std::max_element( kktViolationPerMode.begin(), kktViolationPerMode.end() );

      if(not anyRowVarsModified) converged = true;

	   ++outer_iter;

      // TODO objective function value
      outerLoopTime += outerLoopTimer.seconds();
      if(outer_iter % _output_config.progress_print_interval == 0)
      {
	      _log_progress(outer_iter, inner_iters, func_evals, errorNorm, obj, outerLoopTime);
      }
   }

   {
      // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
      //auto _ = ScopedTimerContribution(timing_data.normalize);
      kruskalOutput.normalize();

   }

   {
      // TODO get this timing data some other way; it won't be accurate here without an unnecessary Kokkos::fence()
      //auto _ = ScopedTimerContribution(timing_data.permute);
      kruskalOutput.permute_factor_matrix_columns();
   }

   computeTime += cpAprTimer.seconds();

   // Final logging
   std::ostringstream msg_time;
   msg_time << "Sparse Time: " << sparseTime << " sec\n";
   msg_time << "Dense Time: " << denseTime << " sec";
   log.print(msg_time.str(),Log::DEBUG_1);
   _log_summary(outer_iter, inner_iters, func_evals, errorNorm, obj, timing_data.row.time_elapsed,
	             timing_data.row.compute_phi, timing_data.row.search_direction,
	             timing_data.row.line_search, piTime, computeTime,
	             max_nonz_per_row, min_nonz_per_row, mean_nonz_per_row, stdev_nonz_per_row);
   std::cout << "----------------------------------------------------------------------";
   if (log.get_verbosity() > 1)
       std::cout << "--------------------";
   std::cout << std::endl;
}

//==============================================================================

template <class NumericalTypes>
void DampedNewtonCpApr<NumericalTypes>::_log_progress(
	sub_index_t outer_iter,
	element_index_t inner_iter,
	element_index_t func_evals,
	kruskal_value_t error_norm,
	kruskal_value_t obj,
	kruskal_value_t loopTimer
) const
{
	Log &log = Log::new_log();

    std::ostringstream message_release;
    message_release << std::setfill(' ')
                    << std::right
                    << std::setw(5) << outer_iter
                    << "  " << std::scientific << std::setprecision(16) << error_norm
                    << "  " << std::scientific << std::setprecision(16) << obj
                    << "  " << std::scientific << std::setprecision(3) << loopTimer;
    if (log.get_verbosity() > 1)
    {
        message_release << " " << std::setfill(' ') << std::setw(12) << inner_iter;
        message_release << " " << std::setfill(' ') << std::setw(12) << func_evals;
    }
    log.print(message_release.str(), Log::RELEASE);
}

//==============================================================================

template <class NumericalTypes>
void DampedNewtonCpApr<NumericalTypes>::_log_history(
	sub_index_t outer_iter,
	element_index_t inner_iter,
	element_index_t func_evals,
	kruskal_value_t error_norm,
	kruskal_value_t obj,
	kruskal_value_t loopTimer
) const
{
	Log &log = Log::new_log();

	std::stringstream message_release;
	message_release << outer_iter;
	message_release << "," << inner_iter;
	message_release << "," << func_evals;
	message_release << "," << std::scientific << std::setprecision(16) << error_norm;
	message_release << "," << std::scientific << std::setprecision(16) << obj;
	message_release << "," << std::fixed << std::setprecision(3) << loopTimer;
	log.print(message_release.str(), Log::RELEASE);
}

//==============================================================================

template <class NumericalTypes>
void DampedNewtonCpApr<NumericalTypes>::_log_summary(

	sub_index_t outer_iter,
	element_index_t inner_iter,
	element_index_t func_evals,
	kruskal_value_t error_norm,
	kruskal_value_t obj,
	kruskal_value_t elapsed_time,
	kruskal_value_t compute_phi,
	kruskal_value_t search_direction,
	kruskal_value_t line_search,
	kruskal_value_t piTime,
	kruskal_value_t computeTime,
	std::vector<sub_index_t > max_nonz_per_row,
	std::vector<sub_index_t > min_nonz_per_row,
	std::vector<kruskal_value_t> mean_nonz_per_row,
	std::vector<kruskal_value_t> stdev_nonz_per_row
) const 
{
	Log &log = Log::new_log();

    std::ostringstream message_release;
    message_release << "\nAdvanced stats";
    message_release << "\n--------------";
    message_release << "\nFinal Log Likelihood: " << std::setw(16) << std::setprecision(9) << obj;
    message_release << "\nFinal KKT Violation: " << std::setw(16) << std::setprecision(9) << error_norm;
    message_release << "\nTotal Outer Iterations: " << std::setw(16) << outer_iter;
    message_release << "\nTotal Inner Iterations: " << std::setw(16) << inner_iter;
    message_release << "\nTotal Function Evaluations: " << std::setw(16) << func_evals;
    message_release << "\nTotal Time PDNR.compute() (s): " << std::setw(16) << std::setprecision(9) << computeTime;
    message_release << "\nTotal Time Pi (s): " << std::setw(16) << std::setprecision(9) << piTime;
    message_release << "\nTotal Time Phi (s): " << std::setw(16) << std::setprecision(9) << compute_phi;
    message_release << "\nTotal Time Perform Line Search (s): " << std::setw(16) << std::setprecision(9) << line_search;
#ifndef KOKKOS_ENABLE_CUDA
    message_release << "\nAverage Time per Thread Row Subproblem (s): " << std::setw(16) << std::setprecision(9) << elapsed_time;
    message_release << "\nAverage Time per Thread Compute Search Directions (s): " << std::setw(16) << std::setprecision(9) << search_direction;
#else
    message_release << "\nTotal Time Row Subproblem (s) " << std::setw(16) << std::setprecision(9) << elapsed_time;
	message_release << "\nTotal Time Compute Search Directions (s) " << std::setw(16) << std::setprecision(9) << search_direction;
#endif
    message_release << std::endl;
    message_release << "\nNonzeros Per Row";
    message_release << "\n----------------\n";
    message_release << std::left << std::setw(6) << "Mode";
    message_release << std::right << std::setw(8) << "Max";
    message_release << std::right << std::setw(8) << "Min";
    message_release << std::right << std::setw(25) << "Mean";
    message_release << std::right << std::setw(25) << "Stdev";
    for (int i = 0; i < max_nonz_per_row.size(); ++i)
    {
        message_release << std::fixed;
        message_release << std::left << "\n" << std::setw(6) << i;
        message_release << std::right << std::setfill(' ') << std::setw(8) << max_nonz_per_row[i];
        message_release << std::right << std::setfill(' ') << std::setw(8) << min_nonz_per_row[i];
        message_release << std::right << std::setfill(' ') << std::setw(25) << mean_nonz_per_row[i];
        message_release << std::right << std::setfill(' ') << std::setw(25) << stdev_nonz_per_row[i];
    }
    message_release << std::endl;

    log.print(message_release.str(), Log::DEBUG_1);
}
//==============================================================================

} // end namespace sparten
