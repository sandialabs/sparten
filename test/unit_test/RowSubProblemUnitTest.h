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
 \brief Class declaration for Dimension testing
 */

#pragma once

#include <sparten/SolverConfiguration.h>
#include <sparten/RowSubProblemQuasiNewton.h>
#include <sparten/RowSubProblemDampedNewton.h>
#include <sparten/Log.h>
#include <sparten/spartenUtils.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "../sparten/SolverConfiguration.h"

namespace unitTest
{
using numeric_traits = sparten::PrecisionTraits<sparten::type::ElemIdx, double, sparten::type::ElemIdx,sparten::type::SubIdx>;


template <class TeamPolicy,class NumericalTypes>
class RowSubProblemQuasiNewtonDriver : public sparten::RowSubProblemQuasiNewton<TeamPolicy, NumericalTypes>
{
public:


  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sparse_value_t = typename NumericalTypes::sparse_value_t;
  using element_index_t = typename NumericalTypes::element_index_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;

  using team_policy_t = TeamPolicy;
  using execution_space = typename team_policy_t::execution_space;
  using team_member_t = typename team_policy_t::member_type;
  using scratch_memory_space = typename execution_space::scratch_memory_space;
  using device_memory_space = typename execution_space::memory_space;



    template <
    class PiView,
    class KDataView,
    class NonzLocView,
    class NonzLocIdxView,
    class SparseDataView,
    class SparseIndexView,
    class inputVarView,
    class RhoType,
    class KTensorType
  >

  KOKKOS_INLINE_FUNCTION
  RowSubProblemQuasiNewtonDriver(
    int32_t iRow,
    int32_t iMode,
    int32_t nMode,
    int32_t nNonz,
    typename TeamPolicy::member_type const& team_member,
    sparten::QuasiNewtonSolverConfiguration<NumericalTypes> config,
    sparten::SparseTensorInfo<NumericalTypes> const& input_info,
    sparten::KruskalTensorInfo<NumericalTypes> const& kruskal_info,
    PiView const& pi,
    KDataView const& kdata,
    NonzLocView const& nonzLoc,
    NonzLocIdxView const& nonzLocIdx,
    SparseDataView const& sparseData,
    inputVarView const &inRowVar,
    inputVarView const &inRowGrad,
    inputVarView const &inSearchDir,
    inputVarView const &inDeltaM,
    inputVarView const &inDeltaG,
    RhoType const inRho,
    SparseIndexView const &index,
    KTensorType const &ktensor

  )  : sparten::RowSubProblemQuasiNewton <TeamPolicy,NumericalTypes>
      (  iRow,
        team_member,
        config,
        input_info,
        kruskal_info,
        pi,
        kdata,
        nonzLoc,
        nonzLocIdx,
        sparseData )
  {


    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(team_member,this->_kruskal_info.num_components),
    [&] (const sub_index_t iFixed) {

      this->_daVars(iFixed) = inRowVar(iFixed);
      this->_daGrad(iFixed) = inRowGrad(iFixed);
      this->_daSearchDirection(iFixed) = inSearchDir(iFixed);
      this->_daDeltaM(0,iFixed) = inDeltaM(iFixed);
      this->_daDeltaG(0,iFixed) = inDeltaG(iFixed);

    });


    // Need to modify the following
    Kokkos::single(Kokkos::PerTeam( team_member ), [&] ()
    {
       this->_daRho(0) = inRho;
       for (int i = 1; i < 3; ++i) // Testing BGFS steps of 3,
       {
          this->_daRho(i) = 0.0;
       }
    });
    team_member.team_barrier();

    for (int i = 1; i < 3; ++i) {
       Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, this->_kruskal_info.num_components),[&](sub_index_t iFixed) {
          this->_daDeltaM(i, iFixed) = 0.0;
          this->_daDeltaG(i, iFixed) = 0.0;
       });
     }
  }


  template <class ViewLocalPi, class ElemIdx>
  KOKKOS_INLINE_FUNCTION
  void get_local_pi( ViewLocalPi &result, ElemIdx begin, ElemIdx end, int iRow )
  {

    // std::ofstream myfile("local_pi_"+ std::to_string(iRow) + ".txt");
     element_index_t nNonz= end-begin;
     //for( element_index_t iNonz = begin; iNonz < end; ++iNonz)
     Kokkos::parallel_for(Kokkos::ThreadVectorRange( this->_team_member, nNonz ), [&]( const element_index_t iNonz)
     {

       // Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->_team_member, this->_kruskal_info.num_components), [&]( const sub_index_t iComp)
        for ( int iComp = 0; iComp < this->_kruskal_info.num_components; ++iComp )
        {
           result(iNonz+begin,iComp) = this->_localPi(iNonz, iComp);
      //     myfile.precision(16);
        //   myfile <<  std::fixed << std::scientific << std::setprecision(16) <<  result(iNonz,iComp)  ;
          // if( iComp+1 < this->_kruskal_info.num_components )  myfile << " ";
        } //);
      // myfile << std::endl;
     });
     this->_team_member.team_barrier();
 //    myfile.close();
  }


  template <class KruskalView1D>
  KOKKOS_INLINE_FUNCTION
  kruskal_value_t test_obj_likelihood( KruskalView1D const& daRowVars )
  {
    return this->obj_likelihood(this->_daVars);
  }

  KOKKOS_INLINE_FUNCTION
  void test_compute_phi( )
  {
    this->_compute_phi();
  }

  template <class RowVarsView, class RowGradView>
  KOKKOS_INLINE_FUNCTION
  void test_line_search(
    RowVarsView const& daRowVars,
    RowGradView const& daRowGrad,
    kruskal_value_t& dObjNew,
    kruskal_value_t& dUnitStepAred,
    int & nRetCode
    )
  {
    this->line_search( daRowVars, daRowGrad, this->_daVars, dObjNew, dUnitStepAred, nRetCode  );
  }



  KOKKOS_INLINE_FUNCTION
  void test_compute_search_dir(int32_t nlbfgsPos, int32_t iters, kruskal_value_t dMuDumping, kruskal_value_t &dPredictedReduction )
  {
    this->_compute_search_dir(  nlbfgsPos,  iters);
  }


  template <class KDataView>
  KOKKOS_INLINE_FUNCTION
  void set_daVars( KDataView const &inputVar )
  {
    this->_set_daVars(inputVar);
  }

  template <class PhiDataView>
  KOKKOS_INLINE_FUNCTION
  void get_daPhi( PhiDataView &outPhi )
  {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components),
    [&] (sub_index_t iFixed) {

      outPhi(iFixed) = this->_daPhi(iFixed);
    });
  }

   template <class PhiDataView, class SubIdx>
   KOKKOS_INLINE_FUNCTION
   void get_daPhi( PhiDataView &outPhi, SubIdx iRow  )
   {
      Kokkos::parallel_for( Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components), [&] (const sub_index_t &iFixed) {
         outPhi(iRow, iFixed) = this->_daPhi(iFixed);
      });
   }

  template <class DataView,class SubIdx>
  KOKKOS_INLINE_FUNCTION
  void get_SearchDirection( DataView &outSearchDir, SubIdx iRow )
  {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components),
    [&] (const sub_index_t &iFixed) {

      outSearchDir(iRow,iFixed) = this->_daSearchDirection(iFixed);
    });
  }

  template <class DataView, class SubIdx>
  KOKKOS_INLINE_FUNCTION
  void get_daWorkObj( DataView &outWorkObj,  SubIdx iRow )
  {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(this->_team_member, this->_nNonz ),
    [&] (const sub_index_t &iFixed) {
      outWorkObj(iRow,iFixed) = this->_daWorkObj(iFixed);
    });
  }

   template <class DataView, class SubIdx>
   KOKKOS_INLINE_FUNCTION
   void get_daNewRowVars( DataView &newRowVars, SubIdx iRow )
   {
      Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(this->_team_member, this->_kruskal_info.num_components ),
            [&] (const sub_index_t &iFixed) {
               newRowVars(iRow,iFixed) = this->_daVars(iFixed);
            });
   }
};


template <class TeamPolicy,class NumericalTypes>
class RowSubProblemDampedNewtonDriver : public sparten::RowSubProblemDampedNewton<TeamPolicy, NumericalTypes>
{


public:


  using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
  using sparse_value_t = typename NumericalTypes::sparse_value_t;
  using element_index_t = typename NumericalTypes::element_index_t;
  using sub_index_t = typename NumericalTypes::sub_index_t;

  using team_policy_t = TeamPolicy;
  using execution_space = typename team_policy_t::execution_space;
  using team_member_t = typename team_policy_t::member_type;
  using scratch_memory_space = typename execution_space::scratch_memory_space;
  using device_memory_space = typename execution_space::memory_space;

  int32_t nNonzPDNR;

   template <
           class PiView,
           class KDataView,
           class NonzLocView,
           class NonzLocIdxView,
           class SparseDataView,
           class SparseIndexView,
           class inputVarView,
           class RhoType,
           class KTensorType
   >
   KOKKOS_INLINE_FUNCTION
   RowSubProblemDampedNewtonDriver(
           int32_t iRow,
           int32_t iMode,
           int32_t nMode,
           int32_t nNonz,
           typename TeamPolicy::member_type const& team_member,
           sparten::DampedNewtonSolverConfiguration<NumericalTypes> config,
           sparten::SparseTensorInfo<NumericalTypes> const& input_info,
           sparten::KruskalTensorInfo<NumericalTypes> const& kruskal_info,
           PiView const& pi,
           KDataView const& kdata,
           NonzLocView const& nonzLoc,
           NonzLocIdxView const& nonzLocIdx,
           SparseDataView const& sparseData,
           inputVarView const &inRowVar,
           inputVarView const &inRowGrad,
           inputVarView const &inSearchDir,
           inputVarView const &inDeltaM,
           inputVarView const &inDeltaG,
           RhoType const inRho,
           SparseIndexView const &index,
           KTensorType const &ktensor
   )  : sparten::RowSubProblemDampedNewton <TeamPolicy,NumericalTypes>
                (  iRow,
                   iMode,
                   nMode,
                   team_member,
                   config,
                   input_info,
                   kruskal_info,
                   pi,
                   kdata,
                   nonzLoc,
                   nonzLocIdx,
                   sparseData, index, ktensor )
   {


      Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member,this->_kruskal_info.num_components),
              [&] (const sub_index_t iFixed) {

                 this->_daVars(iFixed) = inRowVar(iFixed);
                 this->_daGrad(iFixed) = inRowGrad(iFixed);
                 this->_daSearchDirection(iFixed) = inSearchDir(iFixed);
              });
   }





  template <class RedudcitonValType>
  KOKKOS_INLINE_FUNCTION
  void test_compute_search_dir(  RedudcitonValType mu, RedudcitonValType &red )
  {
    this->_compute_search_dir( mu, red, this->_kruskal_info.num_components );
  }




   template <class DataView>
   KOKKOS_INLINE_FUNCTION
   void get_Hessian( DataView &outHessian )
   {
      Kokkos::parallel_for( Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components*this->_kruskal_info.num_components),
              [&] (sub_index_t iFixed) {
         outHessian(iFixed) = this->_daWork2(iFixed);
      });
   }

   template <class ViewLocalPi, class ElemIdx>
   KOKKOS_INLINE_FUNCTION
   void get_local_pi( ViewLocalPi &result, ElemIdx begin, ElemIdx end, int iRow )
   {
      // std::ofstream myfile("local_pi_"+ std::to_string(iRow) + ".txt");
      element_index_t nNonz= end-begin;
      //for( element_index_t iNonz = begin; iNonz < end; ++iNonz)
      Kokkos::parallel_for(Kokkos::ThreadVectorRange( this->_team_member, nNonz ), [&]( const element_index_t iNonz)
      {
        // Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->_team_member, this->_kruskal_info.num_components), [&](sub_index_t iComp)
         for ( int iComp = 0; iComp < this->_kruskal_info.num_components; ++iComp )
         {
            result(iNonz+begin, iComp) = this->_localPi(iNonz, iComp);
            //     myfile.precision(16);
            //   myfile <<  std::fixed << std::scientific << std::setprecision(16) <<  result(iNonz,iComp)  ;
            // if( iComp+1 < this->_kruskal_info.num_components )  myfile << " ";
         }  //);
         // myfile << std::endl;
      });
      this->_team_member.team_barrier();
      //    myfile.close();
   }

   template <class KruskalView1D>
   KOKKOS_INLINE_FUNCTION
   kruskal_value_t test_obj_likelihood( KruskalView1D const& daRowVars )
   {
      return this->obj_likelihood(daRowVars);
   }

   KOKKOS_INLINE_FUNCTION
   void test_compute_phi( )
   {
      this->compute_phi();
   }

   template <class RowVarsView, class RowGradView>
   KOKKOS_INLINE_FUNCTION
   void test_line_search(
           RowVarsView const& daRowVars,
           RowGradView const& daRowGrad,
           kruskal_value_t& dObjNew,
           kruskal_value_t& dUnitStepAred,
           int & nRetCode
   )
   {

      this->line_search( daRowVars, daRowGrad, this->_daVars, dObjNew, dUnitStepAred, nRetCode  );
   }


   KOKKOS_INLINE_FUNCTION
   void test_compute_search_dir(int32_t nlbfgsPos, int32_t iters, kruskal_value_t dMuDumping, kruskal_value_t &dPredictedReduction)
   {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->_team_member, this->_nNonz), [&] (const sub_index_t &iNonz) {
         auto sqVal = this->_daWork1(iNonz) * this->_daWork1(iNonz);
         this->_daWork1(iNonz) = this->_sparseData(this->_nonzLoc(iNonz)) / sparten::max(sqVal,this->_config.eps_div_zero_grad);
      });

      this->_team_member.team_barrier();

      this->_compute_search_dir( dMuDumping, dPredictedReduction, this->_kruskal_info.num_components );
   }


   template <class KDataView>
   KOKKOS_INLINE_FUNCTION
   void set_daVars( KDataView const &inputVar )
   {
      this->_set_daVars(inputVar);
   }

   template <class PhiDataView>
   KOKKOS_INLINE_FUNCTION
   void get_daPhi( PhiDataView &outPhi )
   {
      Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components),
              [&] (sub_index_t iFixed) {

                 outPhi(iFixed) = this->_daPhi(iFixed);
              });
   }

   template <class PhiDataView, class SubIdx>
   KOKKOS_INLINE_FUNCTION
   void get_daPhi( PhiDataView &outPhi, SubIdx iRow  )
   {
      Kokkos::parallel_for( Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components), [&] (const sub_index_t &iFixed) {
         outPhi(iRow, iFixed) = this->_daPhi(iFixed);
      });
   }

   template <class DataView, class SubIdx>
   KOKKOS_INLINE_FUNCTION
   void get_SearchDirection( DataView &outPhi, SubIdx iRow )
   {
      Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(this->_team_member,this->_kruskal_info.num_components),
              [&] (const sub_index_t &iFixed) {

                 outPhi(iRow,iFixed) = this->_daSearchDirection(iFixed);
              });
   }

   template <class DataView, class SubIdx>
   KOKKOS_INLINE_FUNCTION
   void get_daWorkObj( DataView &outWorkObj,  SubIdx iRow )
   {
      Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(this->_team_member, this->_nNonz ),
              [&] (const sub_index_t &iFixed) {
                 outWorkObj(iRow,iFixed) = this->_daWorkObj(iFixed);
              });
   }

   template <class DataView, class SubIdx>
   KOKKOS_INLINE_FUNCTION
   void get_daNewRowVars( DataView &newRowVars,  SubIdx iRow )
   {
      Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(this->_team_member, this->_kruskal_info.num_components ),
            [&] (const sub_index_t &iFixed) {
               newRowVars(iRow,iFixed) = this->_daVars(iFixed);
            });
   }
};



//! RowSubProblem Test Fixture
class RowSubProblemUnitTest : public ::testing::Test {
public:
    virtual void SetUp();         //!<
    virtual void TearDown();      //!<

protected:
   int _precision;
   sparten::CpAprConfiguration<numeric_traits> _config = sparten::CpAprConfiguration<numeric_traits>{10, 10, 0.01, 1.0e-4, 1e-16};
   sparten::QuasiNewtonSolverConfiguration<numeric_traits> _config_quasi = sparten::QuasiNewtonSolverConfiguration<numeric_traits>(_config);
   sparten::DampedNewtonSolverConfiguration<numeric_traits> _config_Damped = sparten::DampedNewtonSolverConfiguration<numeric_traits>(_config);
   sparten::SparseTensorInfo<numeric_traits> _mode_info;
   sparten::KruskalTensorInfo<numeric_traits> _kruskal_info;

};

} // unitTest
