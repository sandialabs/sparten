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

#include <sparten/PrecisionTraits.h>
#include <Kokkos_Core.hpp>
#include <sparten/spartenAlgorithms.h>
#include <sparten/SolverConfiguration.h>
#include <iomanip>
// TODO The cleanest way to get rid of these macros??
#ifndef KOKKOS_ENABLE_CUDA
#define reset_timer(timer) timer.reset()
#define stop_timer(timer) timer.seconds()
#else
#define reset_timer(timer)
#define stop_timer(timer) 0
#endif

namespace sparten {

template <class Derived>
class RowSubProblemCommon
{
private:
  // CRTP boilerplate
  KOKKOS_FORCEINLINE_FUNCTION
  Derived& self() noexcept { return static_cast<Derived&>(*this); }
  KOKKOS_FORCEINLINE_FUNCTION
  Derived const& self() const noexcept { return static_cast<Derived const&>(*this); }


protected:

  template <class KruskalValue>
  KOKKOS_INLINE_FUNCTION
  void _eliminate_small_gradient_values(KruskalValue &val) const
  {
    // If this shouldn't be checked, Derived will override this function to do
    // no checking on small values
    if (val < self()._config.eps_div_zero_grad)
    {
      val = self()._config.eps_div_zero_grad;
    }
  }


  KOKKOS_INLINE_FUNCTION
  void _compute_phi() const
  {
    using sub_index_t = typename Derived::sub_index_t;
    using kruskal_value_t = typename Derived::kruskal_value_t;

#ifndef KOKKOS_ENABLE_CUDA // TODO Get rid of these ugly macros
	  Kokkos::Timer timer;
#else
	  double timer = 0.0;
#endif
	  double elapsed_time = 0.0;


    // This is a matrix vector multiply: _daWork1 = _localPi x _daVars
    // with a filter.  Then _daWorkObj = _sparseData(...) / _daWork1 (value-wise)
    // THIS IS A REALLY IMPORTANT LOOP BECAUSE nNonz is likely large
    Kokkos::parallel_for( Kokkos::ThreadVectorRange(self()._team_member, self()._nNonz), [&] (const sub_index_t &iNonz) {
       // v = m_row * Pi

       auto mySum = sparten::dot_product(
          Kokkos::subview(self()._pi, self()._nonzLoc(iNonz) , Kokkos::ALL),
          self()._daVars);

       self()._daWork1(iNonz) = mySum;

        // w(inonz) = x_row(inonz) ./max(v(inonz),eps)
        // Eliminate too small values
        self()._eliminate_small_gradient_values(self()._daWork1(iNonz));
        self()._daWorkObj(iNonz) = self()._sparseData(self()._nonzLoc(iNonz)) / self()._daWork1(iNonz);
    });
    self()._team_member.team_barrier();

    // ComputePhi(iComp) = w(:,)*Pi
#if 0
     Kokkos::parallel_for( Kokkos::ThreadVectorRange(self()._team_member, self()._kruskal_info.num_components), [&] ( const sub_index_t &iComp) {
       kruskal_value_t mySum = 0;
       for (sub_index_t iNonz = 0; iNonz < self()._nNonz; ++iNonz )  {
          mySum += self()._pi(self()._nonzLoc(iNonz), iComp) * self()._daWorkObj(iNonz);
       }
       self()._daPhi(iComp) = mySum;
    });
#else
    for ( sub_index_t iComp = 0; iComp < self()._kruskal_info.num_components; ++iComp )
    {
       kruskal_value_t mySum = 0;
       Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(self()._team_member, self()._nNonz), [&] (const sub_index_t &iNonz, kruskal_value_t &lcMySum)
       {
          lcMySum += self()._pi(self()._nonzLoc(iNonz), iComp) * self()._daWorkObj(iNonz);
       }, mySum );
       Kokkos::single(Kokkos::PerTeam( self()._team_member), [&] ()
       {
          self()._daPhi(iComp) = mySum;
       });
    }
#endif
     elapsed_time = stop_timer(timer);

     Kokkos::single( Kokkos::PerTeam( self()._team_member), [&] () {
        const_cast<double&>(self().stats.t_compute_phi) = elapsed_time;
     });

  };


  template <class KruskalView1D>
  KOKKOS_INLINE_FUNCTION
  typename KruskalView1D::value_type _obj_likelihood( KruskalView1D  &daRowVars  )
  {
     using sub_index_t = typename Derived::sub_index_t;
     using kruskal_value_t = typename Derived::kruskal_value_t;
     using element_index_t = typename Derived::element_index_t;

     kruskal_value_t f  ;//= -sparten::sum_of_elements( self()._team_member, daRowVars);
     Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(self()._team_member, self()._kruskal_info.num_components), [&] ( const sub_index_t &iComp, kruskal_value_t& local_f )
     {
        local_f -= daRowVars(iComp);
     }, Kokkos::Sum<kruskal_value_t>(f) );
     self()._team_member.team_barrier();


     // Compute f = - (sum r=1:R m[r]) + (sum j=1:nnz X[j] * log(v' * Pi[j])).
     self()._team_member.team_barrier();

     kruskal_value_t f_part ;
     kruskal_value_t dSafeGuard =  self()._config.log_zero_safeguard;
     // element_index_t flag;  // Need when dealing with bad values
     // The value in _daWorkObj explodes in a random manner.
     Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(self()._team_member, self()._nNonz), [&] (const sub_index_t &iNonz, kruskal_value_t &f_inner)
     {
        auto mySum = sparten::dot_product( Kokkos::subview( self()._pi, self()._nonzLoc(iNonz), Kokkos::ALL), daRowVars );

        auto d = static_cast<kruskal_value_t>(mySum);
        if( d < dSafeGuard ) {
           d = dSafeGuard;
           // if( d < 0 ) lflag++;
        }
        f_inner += self()._sparseData(self()._nonzLoc(iNonz)) * static_cast<kruskal_value_t>(log(d));
     },Kokkos::Sum<kruskal_value_t >(f_part));

     f += f_part;

      // Increment function evaluations
     Kokkos::single( Kokkos::PerTeam( self()._team_member), [&] () {
        ++(self().stats.n_func_evals);
     });


    return f;
   }


   template <class KruskalView1D>
   KOKKOS_INLINE_FUNCTION
   typename KruskalView1D::value_type _obj_likelihood2( KruskalView1D  &daRowVars  )
   {
      using sub_index_t = typename Derived::sub_index_t;
      using kruskal_value_t = typename Derived::kruskal_value_t;
      using element_index_t = typename Derived::element_index_t;
      auto nComps = self()._kruskal_info.num_components;
      const  int32_t threadLocalSize=128;
      kruskal_value_t f  ;//= -sparten::sum_of_elements( self()._team_member, daRowVars);
      Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(self()._team_member, nComps), [&] ( const sub_index_t &iComp, kruskal_value_t& local_f )
      {
         local_f -= daRowVars(iComp);
      }, Kokkos::Sum<kruskal_value_t>(f) );
      self()._team_member.team_barrier();


      // Compute f = - (sum r=1:R m[r]) + (sum j=1:nnz X[j] * log(v' * Pi[j])).
      self()._team_member.team_barrier();

      kruskal_value_t f_part ;
      kruskal_value_t dSafeGuard =  self()._config.log_zero_safeguard;
      // element_index_t flag;  // Need when dealing with bad values
      // The value in _daWorkObj explodes in a random manner.
      Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(self()._team_member, self()._nNonz), [&] (const sub_index_t &iNonz, kruskal_value_t &f_inner)
      {
         // Need to compute UpsRow
         // i is cov = m_row * Pi
         kruskal_value_t UpsRow=0;
         kruskal_value_t mypi[threadLocalSize];

         // Initialize mypi to 1

          for( sub_index_t iComp = 0; iComp < nComps; ++iComp )
          {
             mypi[iComp] = 1;
          }


         // Compute Partial Derivative and Local Pi
         for( sub_index_t imode = 0 ; imode < self()._numModes; ++imode )
         {
            if( imode != self()._iModeSkip )
            {
               const auto myKdata = self()._kTensor(imode);
               const auto index = self()._sparseIndex(imode, self()._nonzLoc(iNonz));
               for( sub_index_t iComp = 0; iComp < nComps; ++iComp )
               {
                  mypi[iComp] *= myKdata(index,iComp);
                  //  UpsRow += _daVars(iComp)*myKdata(index,iComp);
               }
            }
         }

         for( sub_index_t iComp = 0; iComp < nComps; ++iComp )
         {

            UpsRow += self()._daVars(iComp)* mypi[iComp];
         }
         auto mySum = UpsRow;
       //  auto mySum = sparten::dot_product( Kokkos::subview( self()._pi, self()._nonzLoc(iNonz), Kokkos::ALL), daRowVars );

         auto d = static_cast<kruskal_value_t>(mySum);
         if( d < dSafeGuard ) {
            d = dSafeGuard;
            // if( d < 0 ) lflag++;
         }
         f_inner += self()._sparseData(self()._nonzLoc(iNonz)) * static_cast<kruskal_value_t>(log(d));
      },Kokkos::Sum<kruskal_value_t >(f_part));

      f += f_part;

      // Increment function evaluations
      Kokkos::single( Kokkos::PerTeam( self()._team_member), [&] () {
         ++(self().stats.n_func_evals);
      });


      return f;
   }

  // stats.nFnEvaluations
  template <class RowVarsView, class RowGradView, class NewRowVarsView, class KruskalValue>
  KOKKOS_INLINE_FUNCTION
  void _line_search(
    RowVarsView const& daRowVars,
    RowGradView const& daRowGrad,
    NewRowVarsView &daNewRowVars,
    KruskalValue& dObjNew,
    KruskalValue& dUnitStepAred,
    int32_t& nRetCode
  )
  {

    using sub_index_t = typename Derived::sub_index_t;
    using kruskal_value_t = typename Derived::kruskal_value_t;

    kruskal_value_t dStepLength = 1.0;

    sub_index_t nNumSteps = 0;
    kruskal_value_t dNewL1Norm;
    kruskal_value_t  dObj1;

#ifndef KOKKOS_ENABLE_CUDA // TODO Get rid of these ugly macros
	  Kokkos::Timer timer;
#else
	  double timer = 0.0;
#endif
	  double elapsed_time = 0.0;

    // Negative objective reduction means the unit step failed.
    dUnitStepAred = -1.0;

    // f_old = -1 * tt_loglikelihood_row(isSparse, x_row, m_old, Pi);
    // kruskal_value_t
    typename RowVarsView::value_type dObjOld = -this->_obj_likelihood(daRowVars);

    reset_timer(timer); // don't measure time in _obj_likelihood
    while (nNumSteps < self()._config.max_backtrack_steps)
    {
      // Compute a new projected iterate at alpha times the step length.
      dNewL1Norm = 0.0;

      Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(self()._team_member, self()._kruskal_info.num_components),
      [&] (sub_index_t iComp, kruskal_value_t& newL1Norm) {
        daNewRowVars(iComp) = daRowVars(iComp) + (dStepLength * self()._daSearchDirection(iComp));

        if (daNewRowVars(iComp) < 0.0)
        {
          daNewRowVars(iComp) = 0.0;
        }
        else
        {
          newL1Norm += daNewRowVars(iComp);
        }
      }, Kokkos::Sum<kruskal_value_t>(dNewL1Norm) );


      // TODO this could be part of the previous loop
      // Compute the gradient times the new step.
      kruskal_value_t dGradDotNew = 0.0;
      Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(self()._team_member, self()._kruskal_info.num_components),
        [&] (sub_index_t iComp, kruskal_value_t& gradDotNew) {
          gradDotNew += daRowGrad(iComp) * (daNewRowVars(iComp) - daRowVars(iComp));
      }, Kokkos::Sum<kruskal_value_t>(dGradDotNew) );

      // Check that the step is a descent direction.
      // Also reject the step if its length is nearly zero, because the
      // objective value is unreliable due to logarithm functions.
      if ((dGradDotNew > 0.0) || (dNewL1Norm < self()._config.min_variable_nonzero_tolerance))
      {
        // No descent.  Treat this as a failed step.
        dObjNew = std::numeric_limits<kruskal_value_t>::max();
        if (nNumSteps == 0) {
           dObj1 = dObjNew;
        }
      }
      else
      {
	      elapsed_time += stop_timer(timer);
	      // Compute the new value for the objective function
	      dObjNew = -this->_obj_likelihood(daNewRowVars);
	      reset_timer(timer);

        if (nNumSteps == 0)
        {
          // Only update the dUnitStepAred the first time???
          dObj1 = dObjNew;
        }

        // Break out if there is sufficient decrease.
        if (dObjNew <= dObjOld + dGradDotNew * self()._config.suff_decrease_tolerance)
        {
          break;
        }
      }

      // Backtrack the step length.
      dStepLength = dStepLength * self()._config.step_reduction_factor;

      // Increment the number of steps
      ++nNumSteps;
    }

    if ( dObj1 == std::numeric_limits<kruskal_value_t>::max() )
    {
        dObj1 = dObjOld;
    }
    if (nRetCode != -2)
    {
      nRetCode = nNumSteps;
    }

    // Test if the line search failed.
    if (((nNumSteps >= self()._config.max_backtrack_steps) && (dObjNew > dObjOld)) ||
    (dNewL1Norm < self()._config.min_variable_nonzero_tolerance))
    {
      // Fall back on a multiplicative update step (scaled steepest descent).
      // Experiments indicate it works better than a unit step in the direction
      // of steepest descent.
      // One way of computing the multiplicative update is to use
      // timeComputePhi = 1 - grad; however, it suffers from round-off error when timeComputePhi
      // is tiny.  The more accurate way computes directly from timeComputePhi.
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange( self()._team_member, self()._kruskal_info.num_components),
        [&] (sub_index_t iComp)
      {
          daNewRowVars(iComp) = daRowVars(iComp) * self()._daPhi(iComp);
        // std::cout << "DaNewRowVars2( " << iComp << " ) " << daNewRowVars(iComp) << std::endl;
      } );
      self()._team_member.team_barrier();
	    elapsed_time += stop_timer(timer);
	    dObjNew = -this->_obj_likelihood(daRowVars);
#if 0
      Kokkos::single( Kokkos::PerTeam( self()._team_member ), [&] () {
      });
#endif
    // Negative objective reduction means the unit step failed.
      dUnitStepAred = -1.0;
      if (nRetCode != -2)
      {
        nRetCode = -1;
      }
    } else {
      dUnitStepAred = dObjOld-dObj1;
	    elapsed_time += stop_timer(timer);
    }


    Kokkos::single( Kokkos::PerTeam( self()._team_member), [&] () {
       self().stats.t_line_search = elapsed_time;
    });

  }

   // stats.nFnEvaluations
   template <class RowVarsView, class RowGradView, class NewRowVarsView, class KruskalValue>
   KOKKOS_INLINE_FUNCTION
   void _line_search2(
           RowVarsView const& daRowVars,
           RowGradView const& daRowGrad,
           NewRowVarsView &daNewRowVars,
           KruskalValue& dObjNew,
           KruskalValue& dUnitStepAred,
           int32_t& nRetCode
   )
   {

      using sub_index_t = typename Derived::sub_index_t;
      using kruskal_value_t = typename Derived::kruskal_value_t;

      kruskal_value_t dStepLength = 1.0;

      sub_index_t nNumSteps = 0;
      kruskal_value_t dNewL1Norm;
      kruskal_value_t  dObj1;

#ifndef KOKKOS_ENABLE_CUDA // TODO Get rid of these ugly macros
      Kokkos::Timer timer;
#else
      double timer = 0.0;
#endif
      double elapsed_time = 0.0;

      // Negative objective reduction means the unit step failed.
      dUnitStepAred = -1.0;

      // f_old = -1 * tt_loglikelihood_row(isSparse, x_row, m_old, Pi);
      // kruskal_value_t
      typename RowVarsView::value_type dObjOld = -this->_obj_likelihood2(daRowVars);

      reset_timer(timer); // don't measure time in _obj_likelihood
      while (nNumSteps < self()._config.max_backtrack_steps)
      {
         // Compute a new projected iterate at alpha times the step length.
         dNewL1Norm = 0.0;

         Kokkos::parallel_reduce(
                 Kokkos::ThreadVectorRange(self()._team_member, self()._kruskal_info.num_components),
                 [&] (sub_index_t iComp, kruskal_value_t& newL1Norm) {
                    daNewRowVars(iComp) = daRowVars(iComp) + (dStepLength * self()._daSearchDirection(iComp));

                    if (daNewRowVars(iComp) < 0.0)
                    {
                       daNewRowVars(iComp) = 0.0;
                    }
                    else
                    {
                       newL1Norm += daNewRowVars(iComp);
                    }
         }, Kokkos::Sum<kruskal_value_t>(dNewL1Norm) );


         // TODO this could be part of the previous loop
         // Compute the gradient times the new step.
         kruskal_value_t dGradDotNew = 0.0;
         Kokkos::parallel_reduce(
                 Kokkos::ThreadVectorRange(self()._team_member, self()._kruskal_info.num_components),
                 [&] (sub_index_t iComp, kruskal_value_t& gradDotNew) {
                    gradDotNew += daRowGrad(iComp) * (daNewRowVars(iComp) - daRowVars(iComp));
                 }, Kokkos::Sum<kruskal_value_t>(dGradDotNew) );

         // Check that the step is a descent direction.
         // Also reject the step if its length is nearly zero, because the
         // objective value is unreliable due to logarithm functions.
         if ((dGradDotNew > 0.0) || (dNewL1Norm < self()._config.min_variable_nonzero_tolerance))
         {
            // No descent.  Treat this as a failed step.
            dObjNew = std::numeric_limits<kruskal_value_t>::max();
            if (nNumSteps == 0) {
               dObj1 = dObjNew;
            }
         }
         else
         {
            elapsed_time += stop_timer(timer);
            // Compute the new value for the objective function
            dObjNew = -this->_obj_likelihood2(daNewRowVars);
            reset_timer(timer);

            if (nNumSteps == 0)
            {
               // Only update the dUnitStepAred the first time???
               dObj1 = dObjNew;
            }

            // Break out if there is sufficient decrease.
            if (dObjNew <= dObjOld + dGradDotNew * self()._config.suff_decrease_tolerance)
            {
               break;
            }
         }

         // Backtrack the step length.
         dStepLength = dStepLength * self()._config.step_reduction_factor;

         // Increment the number of steps
         ++nNumSteps;
      }

      if ( dObj1 == std::numeric_limits<kruskal_value_t>::max() )
      {
         dObj1 = dObjOld;
      }
      if (nRetCode != -2)
      {
         nRetCode = nNumSteps;
      }

      // Test if the line search failed.
      if (((nNumSteps >= self()._config.max_backtrack_steps) && (dObjNew > dObjOld)) ||
          (dNewL1Norm < self()._config.min_variable_nonzero_tolerance))
      {
         // Fall back on a multiplicative update step (scaled steepest descent).
         // Experiments indicate it works better than a unit step in the direction
         // of steepest descent.
         // One way of computing the multiplicative update is to use
         // timeComputePhi = 1 - grad; however, it suffers from round-off error when timeComputePhi
         // is tiny.  The more accurate way computes directly from timeComputePhi.
         Kokkos::parallel_for(
                 Kokkos::ThreadVectorRange( self()._team_member, self()._kruskal_info.num_components),
                 [&] (sub_index_t iComp)
                 {
                    daNewRowVars(iComp) = daRowVars(iComp) * self()._daPhi(iComp);
                    // std::cout << "DaNewRowVars2( " << iComp << " ) " << daNewRowVars(iComp) << std::endl;
                 } );
         self()._team_member.team_barrier();
         elapsed_time += stop_timer(timer);
         dObjNew = -this->_obj_likelihood2(daRowVars);
#if 0
         Kokkos::single( Kokkos::PerTeam( self()._team_member ), [&] () {
      });
#endif
         // Negative objective reduction means the unit step failed.
         dUnitStepAred = -1.0;
         if (nRetCode != -2)
         {
            nRetCode = -1;
         }
      } else {
         dUnitStepAred = dObjOld-dObj1;
         elapsed_time += stop_timer(timer);
      }


      Kokkos::single( Kokkos::PerTeam( self()._team_member), [&] () {
         self().stats.t_line_search = elapsed_time;
      });

   }

  KOKKOS_INLINE_FUNCTION
  void _setup_local_pi()
  {
     using sub_index_t = typename Derived::sub_index_t;
     using element_index_t = typename Derived::element_index_t;
     // Extract Pi for iRow
     Kokkos::parallel_for( Kokkos::ThreadVectorRange(self()._team_member, self()._nNonz), [&] (const sub_index_t &iNonz) {
        for (int iComp = 0; iComp < self()._kruskal_info.num_components; ++iComp) {
           self()._localPi( iNonz, iComp ) = self()._pi(self()._nonzLoc(iNonz), iComp );
        }
     });
     self()._team_member.team_barrier();
  }

};

//------------------------------------------------------------------------------

} // end namespace sparten


