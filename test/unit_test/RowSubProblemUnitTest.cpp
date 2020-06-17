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
// Copyright 201 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
// ************************************************************************
//@HEADER

/*!
 \file
 \brief
 */

#include <sparten/Log.h>
#include <sparten/KruskalTensor.h>
#include <sparten/KruskalTensorInitializer.h>
#include <sparten/spartenUtils.h>
#include <sparten/AsciiIO.h>
#include <sparten/SparseTensor.h>
#include <sparten/PrecisionTraits.h>
#include <sparten/SolverConfiguration.h>
#include <sparten/CpAprCommon.h>
#include <unit_test/RowSubProblemUnitTest.h>
#include <generator/FactTestSetGenerator.h>
#include <generator/RandGenMersenneTwister.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <limits>
#include <memory>

namespace unitTest {

typedef enum _test_type_t
{
   set_up_pi,
   compute_phi,
   obj_likelihood,
   lineSearch,
   searchDirPdnr,
   searchDirPqnr,

} test_type_t;

template<class ElemIdx, class SubIdx, class PiView, class factMT, class IndicesT>
void  _sparten_compute_pi_test_internal(
           ElemIdx sparse_nElement,
           SubIdx sparse_nDim,
           SubIdx iDimSkip,
           SubIdx kruskal_nComponent,
           PiView &pi,
           factMT &factorMatrices,
           IndicesT &indices)
{
#if defined(KOKKOS_ENABLE_CUDA)
   const SubIdx VectorSize = kruskal_nComponent == 1 ? 1 : std::min(16,2 << int(std::log2(kruskal_nComponent))-1);
   const SubIdx TeamSize  = 128/VectorSize;
   const SubIdx LeagueSize = ((SubIdx)sparse_nElement+(SubIdx)TeamSize-1)/(SubIdx)TeamSize;
#else
   const SubIdx VectorSize = 16;
   const SubIdx TeamSize = 1;
   const SubIdx LeagueSize = (sparse_nElement+(SubIdx)TeamSize-1)/(SubIdx)TeamSize;
#endif

 //  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> Policy;
   typedef Kokkos::TeamPolicy<> Policy;
   Policy policy(LeagueSize,TeamSize,VectorSize);
   for (SubIdx iDim = 0; iDim < sparse_nDim; ++iDim)
   {
      if (iDim != iDimSkip)
      {
         const auto kData = factorMatrices[iDim];
         Kokkos::parallel_for (policy, KOKKOS_LAMBDA (Policy::member_type team)
         {
            const auto team_index = team.team_rank();
            const auto iNonz = team.league_rank()*TeamSize+team_index;
            if( iNonz >= sparse_nElement ) return;
            const auto index = indices(iDim,iNonz);
            Kokkos::parallel_for (Kokkos::ThreadVectorRange(team, kruskal_nComponent), [&] (const SubIdx iComp)
            {
               pi(iNonz, iComp) *= kData(index, iComp);
            });
         });
         Kokkos::fence();
      }
   }
}

template<class inputVector>
void _getMatrixFromFile(inputVector &inVec, std::string filename )
{


   std::ifstream myfile(filename);
   std::string line;
   if (myfile.is_open()) {
      int i = 0;
      while (getline(myfile, line))
      {
          size_t pos = 0;
          std::string token;
          std::string delimiter =" ";
          std::string::size_type sz;
          while ((pos = line.find(delimiter)) != std::string::npos) {
              token = line.substr(0, pos);

              inVec[i] = std::stod(token, &sz);
              line.erase(0, pos + delimiter.length());
              i++;
          }
          inVec[i] = std::stod(line, &sz);
          i++;
      }
      myfile.close();

   } else {
      std::cout << "Error: File " << filename << " not found!" << std::endl;
   }
}

template<class inputVector, class SubIdx>
void _getMatrixFromFile(inputVector &inVec, std::string filename, SubIdx nRows )
{

   std::ifstream myfile(filename);
   std::string line;
   if (myfile.is_open()) {
      int i = 0;
      int j = 0;
      while (getline(myfile, line))
      {
         size_t pos = 0;
         std::string token;
         std::string delimiter =" ";
         std::string::size_type sz;
         while ((pos = line.find(delimiter)) != std::string::npos) {
               token = line.substr(0, pos);

               inVec(i,j) = std::stod(token, &sz);
               line.erase(0, pos + delimiter.length());
               j++;
         }
         inVec(i,j) = std::stod(line, &sz);
         i++;
         j = 0;
      }
      myfile.close();

   } else {
      std::cout << "Error: File " << filename << " not found!" << std::endl;
   }
}

template<class inputVector>
void _getVectorFromFile(inputVector &inVec, std::string filename) {
  //bool flag = true;

   std::ifstream myfile(filename);
   std::string line;

   if (myfile.is_open()) {
      int i = 0;
      while (getline(myfile, line)) {
         std::string::size_type sz;
         inVec[i] = std::stod(line, &sz);
         i++;
      }
      myfile.close();

   } else {
      std::cout << "Error: File " << filename << " not found!" << std::endl;
   }
}

template<class SparseIndicesHostView, class NonzeroLocsHostView, class NonzeroLocIdxsHostView>
void _reorder_nonzeros_host(
        sparten::type::SubIdx nRow,
        sparten::type::SubIdx nNonz,
        SparseIndicesHostView spidx,
        NonzeroLocsHostView nonzLocs,
        NonzeroLocIdxsHostView nonzLocIdxs) {

   auto mystack = sparten::make_unique_array<std::vector<sparten::type::ElemIdx >>(nRow);
   for (sparten::type::ElemIdx iElem = 0; iElem < nNonz; ++iElem) {
      auto iRow = spidx[iElem];
      mystack[iRow].push_back(iElem);
   }

   sparten::type::ElemIdx nz_count = 0;
   nonzLocIdxs(0) = 0;
   for (sparten::type::SubIdx iRow = 0; iRow < nRow; ++iRow) {
      nonzLocIdxs(iRow + 1) = nonzLocIdxs(iRow) + mystack[iRow].size();
      for (sparten::type::SubIdx iElem = 0; iElem < mystack[iRow].size(); ++iElem) {
         nonzLocs(nz_count) = mystack[iRow][iElem];
            ++nz_count;
      }
   }
}

template<class solverType, class NumericalTypes,class solver_conf_t, class kruskal_conf_t, class mode_conf_t>
void _sparten_rowsubproblem_test_driver(
           std::string problem,
           solver_conf_t _config,
           kruskal_conf_t _kruskal_info,
           mode_conf_t _mode_info,
           test_type_t _test_info )
{

   using kruskal_value_t = typename NumericalTypes::kruskal_value_t;
   using sparse_value_t = typename NumericalTypes::sparse_value_t;
   using element_index_t = typename NumericalTypes::element_index_t;
   using sub_index_t = typename NumericalTypes::sub_index_t;
   using host_memory_space = typename Kokkos::DefaultHostExecutionSpace::memory_space;
   using device_memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;

   using team_policy_t = Kokkos::TeamPolicy<>;
   sub_index_t CurrentMode = 0;
   auto nComponents = _kruskal_info.num_components;


   // Initialize the input data
   sub_index_t _precision = 16;

   // This part can be moved to Setup
   sparten::AsciiIO<element_index_t, kruskal_value_t, element_index_t , sub_index_t > _sptensorIO(
              "test/data/"+problem+"/tensor.txt", sparten::AsciiIO<element_index_t ,
              kruskal_value_t , element_index_t , sub_index_t >::READ,
              _precision);

   sparten::SparseTensor<sparten::type::ElemIdx, sparten::type::ElemIdx, sparten::type::SubIdx> _spTen = _sptensorIO.read_sptensor();

   // Get Factor Matrix
   auto nRows = _spTen.get_nRow(0);  // Getting the first mode
   auto nModes = _spTen.get_nDim();
   auto nElements = _spTen.get_nElement();

   sparten::KruskalTensor<kruskal_value_t, sub_index_t> kTensor =  sparten::KruskalTensor<kruskal_value_t , sub_index_t >(_spTen.get_nDim(), nComponents, _spTen.get_size_dim());
   sparten::KruskalTensorInitializer<kruskal_value_t, sparten::type::SubIdx> kruskalInit;
   kruskalInit.set_random_seed(131);

   kruskalInit.initialize(kTensor, sparten::KruskalTensorInitializer<kruskal_value_t, sub_index_t>::RANDOM);
   kTensor.distribute_weights_to_factor_matrices(0);
   auto factorMatrices = kTensor.get_factor_matrices();

   std::vector<Kokkos::View<sub_index_t *>> nonzLocsIdx(nModes);
   std::vector<Kokkos::View<element_index_t *>> nonzLocs(nModes);

   Kokkos::View<int *> numNonz("NonzInfo", nRows);
   typename Kokkos::View<int *>::HostMirror numNonzHost = Kokkos::create_mirror(numNonz);
   typename Kokkos::View<element_index_t *>::HostMirror nonzLocsHost("LocHost",nElements);

   // Set up data structure for row-wise access to the sparse tensor
   for (sub_index_t iDim = 0; iDim < nModes; ++iDim) {

      auto mynRows = _spTen.get_nRow(iDim);
      Kokkos::resize(nonzLocsIdx[iDim],mynRows+1);
      Kokkos::resize(nonzLocs[iDim],nElements);

      typename Kokkos::View<sub_index_t *>::HostMirror nonzLocsIdxHost=Kokkos::create_mirror_view(nonzLocsIdx[iDim]);


      _reorder_nonzeros_host(
                 mynRows, nElements,
                 Kokkos::subview(_spTen.get_indices_host_view(), iDim, Kokkos::ALL),
                 nonzLocsHost, nonzLocsIdxHost );


      if( iDim == CurrentMode ) {
         for (int iRow = 0; iRow < mynRows; ++iRow) {
            numNonzHost(iRow) = nonzLocsIdxHost(iRow+1)-nonzLocsIdxHost(iRow);
         }
      }
      Kokkos::deep_copy( nonzLocsIdx[iDim], nonzLocsIdxHost);
      Kokkos::deep_copy( nonzLocs[iDim], nonzLocsHost);
   }


   Kokkos::deep_copy( numNonz, numNonzHost);

   // Preparing the solver and data
   auto pi = Kokkos::View<kruskal_value_t **>( "pi matrix temporary", nElements, nComponents );
   Kokkos::deep_copy(pi,1.0);

   // auto factorMatrices = kTensor.get_factor_matrices();
   auto sp_indices = _spTen.get_indices_view();
   _sparten_compute_pi_test_internal(nElements, nModes, CurrentMode, nComponents, pi, factorMatrices, sp_indices );

   const auto kData = factorMatrices[CurrentMode];
   const auto myNonzLoc = nonzLocs[CurrentMode];
   const auto myNonzLocIdx = nonzLocsIdx[CurrentMode];
   const auto sparseData = _spTen.get_data_view();
   const auto sparseIndex =_spTen.get_indices_view();
   using subproblem_driver_t = solverType;
   const sub_index_t  max_nonz = _mode_info.max_nonzeros_per_row;
   const auto scratch_size = subproblem_driver_t::required_scratch_size( _config, _kruskal_info, max_nonz);

   auto team_policy = sparten::get_team_policy<team_policy_t>(nRows);


   Kokkos::View<kruskal_value_t **> _daRowVars("Vars",nRows,nComponents);
   Kokkos::View<kruskal_value_t **> _daNewRowVars("NewVars",nRows,nComponents);
   Kokkos::View<kruskal_value_t **> _daRowGrad("Grad",nRows,nComponents);
   Kokkos::View<kruskal_value_t **> _daSearchDir("SearchDir",nRows,nComponents);

   Kokkos::View<kruskal_value_t **> _daInitialDeltaM("DeltaM",nRows,nComponents);
   Kokkos::View<kruskal_value_t **> _daInitialDeltaG("DeltaG",nRows,nComponents);

   Kokkos::View<kruskal_value_t *> _daObjNew("Obj",nRows);
   Kokkos::View<kruskal_value_t *> _daUnitStepAred("Ared",nRows);
   Kokkos::View<kruskal_value_t *> _daInitialRho("Rho",nRows);

   typename Kokkos::View<kruskal_value_t **>::HostMirror _daRowGradHost = Kokkos::create_mirror_view(_daRowGrad) ;
   typename Kokkos::View<kruskal_value_t **>::HostMirror _daRowVarsHost =  Kokkos::create_mirror_view(_daRowVars);
   typename Kokkos::View<kruskal_value_t **>::HostMirror _daSearchDirHost =  Kokkos::create_mirror_view(_daSearchDir);

   typename Kokkos::View<kruskal_value_t **>::HostMirror _daInitialDeltaMHost =  Kokkos::create_mirror_view(_daInitialDeltaM);
   typename Kokkos::View<kruskal_value_t **>::HostMirror _daInitialDeltaGHost = Kokkos::create_mirror_view(_daInitialDeltaG);
   typename Kokkos::View <kruskal_value_t *>::HostMirror  _daObjNewHost=  Kokkos::create_mirror_view(_daObjNew);
   typename Kokkos::View <kruskal_value_t *>::HostMirror  _daUnitStepAredHost = Kokkos::create_mirror_view(_daUnitStepAred);
   typename Kokkos::View <kruskal_value_t *>::HostMirror  _daInitialRhoHost = Kokkos::create_mirror_view(_daInitialRho);

   _getMatrixFromFile( _daRowGradHost, "test/data/" + problem + "/daGrad.txt",nRows);
   Kokkos::deep_copy(_daRowGrad,_daRowGradHost);

   _getMatrixFromFile( _daRowVarsHost, "test/data/" + problem + "/daOldVar.txt",nRows);
   Kokkos::deep_copy(_daRowVars, _daRowVarsHost);

   _getMatrixFromFile( _daInitialDeltaGHost, "test/data/" + problem + "/daDeltaG.txt",nRows);
   Kokkos::deep_copy(_daInitialDeltaG, _daInitialDeltaGHost);

   _getMatrixFromFile( _daInitialDeltaMHost, "test/data/" + problem + "/daDeltaM.txt",nRows);
   Kokkos::deep_copy(_daInitialDeltaM, _daInitialDeltaMHost);

   _getVectorFromFile( _daInitialRhoHost, "test/data/" + problem + "/daRho.txt");
   Kokkos::deep_copy( _daInitialRho, _daInitialRhoHost);

   _getMatrixFromFile( _daSearchDirHost, "test/data/" + problem + "/daSearchDir.txt",nRows);
   Kokkos::deep_copy(_daSearchDir, _daSearchDirHost);

   if( _test_info == set_up_pi ) {
      auto numMode = kTensor.get_nDim();
      Kokkos::View<sparten::FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp",numMode);
      auto kt_host = Kokkos::create_mirror_view(ktensor);

      for( int i = 0; i < numMode; ++i )
      {
         kt_host(i)= kTensor.get_factor_matrix(i);
      }

      Kokkos::deep_copy( ktensor,kt_host );

      Kokkos::View<kruskal_value_t **> result_pi("pi_verify", nElements,nComponents);
      Kokkos::parallel_for(team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)),KOKKOS_LAMBDA(team_policy_t::member_type team) {
         const auto iRow = team.league_rank();
          // Initialize the test instances

         auto numMode = kTensor.get_nDim();


         auto test_instance = subproblem_driver_t
                 (iRow, 0, 3, numNonz(iRow), team, _config, _mode_info, _kruskal_info,
                  pi, kData, myNonzLoc,myNonzLocIdx,
                  sparseData,
                  Kokkos::subview(_daRowVars, iRow, Kokkos::ALL),
                  Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL),
                  Kokkos::subview(_daSearchDir, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaM, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaG, iRow, Kokkos::ALL),
                 _daInitialRho(iRow),sparseIndex,ktensor);

          test_instance.get_local_pi(result_pi, myNonzLocIdx(iRow),myNonzLocIdx(iRow+1),iRow);
      });

      Kokkos::fence();

      // Verify the solution
      std::vector<kruskal_value_t> answer_pi(nElements * nComponents);
      _getMatrixFromFile(answer_pi, "./test/data/"+ problem+"/local_pi.txt");

      typename Kokkos::View<kruskal_value_t **>::HostMirror result_pi_host = Kokkos::create_mirror_view (result_pi);
      Kokkos::deep_copy(result_pi_host, result_pi);
      for (int iNonz = 0; iNonz < nElements; ++iNonz)
      {
         for (int iComp = 0; iComp < nComponents; ++iComp)
         {
            if( std::abs(result_pi_host(iNonz, iComp)- answer_pi[iNonz * nComponents + iComp] ) > std::abs(1e-10 * answer_pi[iNonz * nComponents + iComp])) 
                std::cout << "At Row " << iComp << std::endl;
            EXPECT_NEAR(result_pi_host(iNonz, iComp), answer_pi[iNonz * nComponents + iComp],
                        std::abs(1e-10 * answer_pi[iNonz * nComponents + iComp]));
         }
      }

   } else if ( _test_info ==  compute_phi ) {  // Test Compute Phi
 
      Kokkos::View<kruskal_value_t **> _daPhi("Phi",nRows,nComponents);

      typename Kokkos::View<kruskal_value_t **>::HostMirror phiHost = Kokkos::create_mirror_view(_daPhi);

      Kokkos::deep_copy(_daRowVars,kData);

      auto numMode = kTensor.get_nDim();
      Kokkos::View<sparten::FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp",numMode);
      auto kt_host = Kokkos::create_mirror_view(ktensor);

      for( int i = 0; i < numMode; ++i )
      {
         kt_host(i)= kTensor.get_factor_matrix(i);
      }

      Kokkos::deep_copy( ktensor,kt_host );

      // Do some testing
      Kokkos::parallel_for ( team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)), KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team)
      {
         const auto iRow = team.league_rank();

         auto test_instance = subproblem_driver_t
                 (iRow, 0, 3, numNonz(iRow), team, _config, _mode_info, _kruskal_info,
                  pi, kData, myNonzLoc, myNonzLocIdx, sparseData,
                  Kokkos::subview(_daRowVars, iRow, Kokkos::ALL),
                  Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL),
                  Kokkos::subview(_daSearchDir, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaM, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaG, iRow, Kokkos::ALL),
                  _daInitialRho(iRow),sparseIndex,ktensor);

         team.team_barrier();
         test_instance.test_compute_phi();
         test_instance.get_daPhi( _daPhi, iRow );
         team.team_barrier();

      });

      Kokkos::fence();

      Kokkos::deep_copy( phiHost, _daPhi );   // Verify the answer./
      std::vector<kruskal_value_t> answer(nComponents*nRows);
      _getMatrixFromFile( answer, "test/data/" + problem + "/phi.txt");

      for( int i = 0; i < nRows; ++i ) {
         for (int j = 0; j < nComponents; ++j) {
            EXPECT_NEAR(phiHost(i, j), answer[i * nComponents + j], std::abs(1e-10 * answer[i* nComponents + j]) );
         }
      }
   } else if ( _test_info == obj_likelihood )       { // obj_likelihood

      Kokkos::deep_copy(_daRowVars,kData);
      Kokkos::View<kruskal_value_t *> _daValues("ObjValues",nRows);
      typename Kokkos::View<kruskal_value_t *>::HostMirror ValuesHost= Kokkos::create_mirror_view(_daValues);

      //auto &values = _daValues;

      auto numMode = kTensor.get_nDim();
      Kokkos::View<sparten::FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp",numMode);
      auto kt_host = Kokkos::create_mirror_view(ktensor);

      for( int i = 0; i < numMode; ++i )
      {
         kt_host(i)= kTensor.get_factor_matrix(i);
      }

      Kokkos::deep_copy( ktensor,kt_host );

      Kokkos::parallel_for ( team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)), KOKKOS_LAMBDA(team_policy_t::member_type team)
      {
         const auto iRow = team.league_rank(); 

         auto test_instance = subproblem_driver_t
                 (iRow,0, 3, numNonz(iRow), team, _config, _mode_info, _kruskal_info,
                  pi, kData, myNonzLoc, myNonzLocIdx, sparseData,
                  Kokkos::subview(kData, iRow, Kokkos::ALL),
                  Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL),
                  Kokkos::subview(_daSearchDir, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaM, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaG, iRow, Kokkos::ALL),
                 _daInitialRho(iRow),sparseIndex,ktensor);
         team.team_barrier();
         kruskal_value_t val;
         val = test_instance.test_obj_likelihood( Kokkos::subview(_daRowVars, iRow, Kokkos::ALL) ) ;
         _daValues(iRow) = val; // All threads accesse to the same cache line
      });

      Kokkos::fence();
      // Verify the answer
      Kokkos::deep_copy( ValuesHost, _daValues);

      std::vector<kruskal_value_t> answer(nRows);
      _getVectorFromFile( answer, "test/data/" + problem + "/likelihood.txt");
      Kokkos::fence();

      for ( int i = 0; i < nRows; ++i )
      {
         EXPECT_NEAR( ValuesHost[i], answer[i], std::abs(1e-10 * answer[i]) );
      }

   } else if ( _test_info == lineSearch ) { // Line Search
      auto numMode = kTensor.get_nDim();
      Kokkos::View<sparten::FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp",numMode);
      auto kt_host = Kokkos::create_mirror_view(ktensor);

      for( int i = 0; i < numMode; ++i )
      {
         kt_host(i)= kTensor.get_factor_matrix(i);
      }

      Kokkos::deep_copy( ktensor,kt_host );

      // Do some testing
      Kokkos::parallel_for ( team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)), KOKKOS_LAMBDA (team_policy_t::member_type team) {
         const auto iRow = team.league_rank();
         // Initialize test instance
         auto test_instance = subproblem_driver_t
                 (iRow, 0, 3, numNonz(iRow), team, _config, _mode_info, _kruskal_info,
                  pi, kData, myNonzLoc, myNonzLocIdx, sparseData,
                  Kokkos::subview(_daRowVars, iRow, Kokkos::ALL),
                  Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL),
                  Kokkos::subview(_daSearchDir, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaM, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaG, iRow, Kokkos::ALL),
                  _daInitialRho(iRow),sparseIndex,ktensor);

         // Declare some scalar values
         // int nRetCode;
         kruskal_value_t dObj, dAred;
         int nRetCode;
         // Do some testing

         test_instance.test_line_search(Kokkos::subview(_daRowVars, iRow, Kokkos::ALL),
                                        Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL), dObj, dAred, nRetCode);

         test_instance.get_daNewRowVars(_daNewRowVars, iRow);

         Kokkos::single(Kokkos::PerTeam(team), [&]() {
            _daObjNew(iRow) = dObj;
            _daUnitStepAred(iRow) = dAred;
         });

      });

      Kokkos::fence();

      _getMatrixFromFile(  _daRowGradHost, "test/data/" + problem + "/daNewVar.txt",nRows); // Reusing the space for RowGrad_Host


      Kokkos::deep_copy( _daRowVarsHost, _daNewRowVars );
      Kokkos::deep_copy( _daObjNewHost, _daObjNew );
      Kokkos::deep_copy( _daUnitStepAredHost, _daUnitStepAred);

      std::vector<kruskal_value_t> answerObjNew(nRows);
      std::vector<kruskal_value_t> answerAred(nRows);
      _getVectorFromFile( answerObjNew, "test/data/" + problem + "/daObj.txt" );
      _getVectorFromFile( answerAred, "test/data/" + problem + "/daAred.txt" );
      for ( int i = 0; i < nRows; ++i ) {

         EXPECT_NEAR( answerObjNew[i], _daObjNewHost[i], std::abs(1e-10 *  answerObjNew[i]) );
         EXPECT_NEAR( answerAred[i], _daUnitStepAredHost[i], std::abs(1e-10 *  answerAred[i]) );
         for ( int j = 0; j < nComponents; ++j ) {
            kruskal_value_t answer = _daRowGradHost(i,j);
            kruskal_value_t val = _daRowVarsHost(i,j);

            EXPECT_NEAR( answer, val, std::abs(1e-10 *  answer) );
         }
      }

   } else if ( _test_info == searchDirPqnr ) { // search direction for pqnr
      auto numMode = kTensor.get_nDim();
      Kokkos::View<sparten::FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp",numMode);
      auto kt_host = Kokkos::create_mirror_view(ktensor);

      for( int i = 0; i < numMode; ++i )
      {
         kt_host(i)= kTensor.get_factor_matrix(i);
      }

      Kokkos::deep_copy( ktensor,kt_host );

      Kokkos::parallel_for ( team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)), KOKKOS_LAMBDA (team_policy_t::member_type team) {
         const auto iRow = team.league_rank();
         // Initialize test instance

         // Initialize test instance
         auto test_instance =subproblem_driver_t
                 (iRow, 0, 3, numNonz(iRow), team, _config, _mode_info, _kruskal_info,
                  pi, kData, myNonzLoc, myNonzLocIdx, sparseData,
                  Kokkos::subview(_daRowVars, iRow, Kokkos::ALL),
                  Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL),
                  Kokkos::subview(_daSearchDir, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaM, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaG, iRow, Kokkos::ALL),
                  _daInitialRho(iRow),sparseIndex,ktensor);

         // Declare some scalar values
         int32_t iters = 0;
         int32_t nlbfgsPos = 0;
         kruskal_value_t dMu = 1.0;
         kruskal_value_t dPredRed = 0.0;

         // Do some testing
         test_instance.test_compute_search_dir(nlbfgsPos, iters, dMu, dPredRed );
         test_instance.get_SearchDirection(_daSearchDir, iRow);
      });
      Kokkos::fence();
      Kokkos::deep_copy( _daSearchDirHost, _daSearchDir );
      std::vector<kruskal_value_t> answer(nComponents*nRows);
      _getMatrixFromFile( answer, "test/data/" + problem + "/daSearchDir.txt");
      for ( int i = 0; i < nRows; ++i ) {
         for ( int j = 0; j < nComponents; ++j ) {
           EXPECT_NEAR( _daSearchDirHost(i,j), answer[i*nComponents+j],std::abs(1e-10 *  answer[i*nComponents+j]) );
         }
      }

   } else if ( _test_info == searchDirPdnr ) { // search direction for pdnr
      _getMatrixFromFile( _daRowGradHost, "test/data/" + problem + "/daGradPdnr.txt",nRows);
      Kokkos::deep_copy(_daRowGrad,_daRowGradHost);

      _getMatrixFromFile( _daRowVarsHost, "test/data/" + problem + "/daVarsPdnr.txt",nRows);
      Kokkos::deep_copy(_daRowVars, _daRowVarsHost);

      auto numMode = kTensor.get_nDim();
      Kokkos::View<sparten::FactorMatrix<kruskal_value_t> *> ktensor("ktensor_tmp",numMode);
      auto kt_host = Kokkos::create_mirror_view(ktensor);

      for( int i = 0; i < numMode; ++i )
      {
         kt_host(i)= kTensor.get_factor_matrix(i);
      }

      Kokkos::deep_copy( ktensor,kt_host );

      Kokkos::parallel_for ( team_policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)), KOKKOS_LAMBDA (team_policy_t::member_type team) {
         const auto iRow = team.league_rank();
         // Initialize test instance

         // Initialize test instance
         auto test_instance =subproblem_driver_t
                 (iRow, 0, 3, numNonz(iRow), team, _config, _mode_info, _kruskal_info,
                  pi, kData, myNonzLoc, myNonzLocIdx, sparseData,
                  Kokkos::subview(_daRowVars, iRow, Kokkos::ALL),
                  Kokkos::subview(_daRowGrad, iRow, Kokkos::ALL),
                  Kokkos::subview(_daSearchDir, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaM, iRow, Kokkos::ALL),
                  Kokkos::subview(_daInitialDeltaG, iRow, Kokkos::ALL),
                  _daInitialRho(iRow),sparseIndex,ktensor);

         // Declare some scalar values
         int32_t iters = 0;
         int32_t nlbfgsPos = 0;
         kruskal_value_t dMyReduction;

         // Do some testing
         test_instance.test_compute_phi();
         test_instance.test_compute_search_dir(nlbfgsPos, iters, 0.00001 , dMyReduction );
         test_instance.get_SearchDirection(_daSearchDir, iRow);
      });

      Kokkos::fence();
      Kokkos::deep_copy( _daSearchDirHost, _daSearchDir );

      std::vector<kruskal_value_t> answer(nComponents*nRows);
      _getMatrixFromFile( answer, "test/data/" + problem + "/daSearchDirPdnr.txt");

      for ( int i = 0; i < nRows; ++i ) {
         for ( int j = 0; j < nComponents; ++j ) {
            EXPECT_NEAR( _daSearchDirHost(i,j), answer[i*nComponents+j],std::abs(1e-10 *  answer[i*nComponents+j]) );
         }
      }

   }


}

void RowSubProblemUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  // Initialize the input data
   _precision=16;
   // Setup
   _config_quasi.eps_div_zero_grad = 1e-10;
   _config_quasi.log_zero_safeguard = 1e-16;
   _config_quasi.row_tolerance = 1e-3;
   _config_quasi.eps_active_set = 1e-3;
   _config_quasi.max_backtrack_steps = 10;
   _config_quasi.min_variable_nonzero_tolerance = 1e-7;
   _config_quasi.step_reduction_factor = 0.5;
   _config_quasi.suff_decrease_tolerance = 1e-4;
   _config_quasi.size_LBFGS = 3;


   _config_Damped.eps_div_zero_grad = 1e-10;
   _config_Damped.log_zero_safeguard = 1e-16;
   _config_Damped.row_tolerance = 1e-3;
   _config_Damped.eps_active_set = 1e-3;
   _config_Damped.max_backtrack_steps = 10;
   _config_Damped.min_variable_nonzero_tolerance = 1e-7;
   _config_Damped.step_reduction_factor = 0.5;
   _config_Damped.suff_decrease_tolerance = 1e-4;


}

void RowSubProblemUnitTest::TearDown()
{

}



#ifdef __NVCC__
    // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_setup_localpi_small)
#ifdef __NVCC__
    // CUDA lambda issue UGLY workaround
#undef private
#endif
{   _kruskal_info = {5};
   _mode_info={100};
   _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small",
           _config_quasi, _kruskal_info, _mode_info, set_up_pi);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, pqnr_compute_phi_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small",
              _config_quasi, _kruskal_info, _mode_info, compute_phi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, pqnr_obj_likelihood_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small",
            _config_quasi, _kruskal_info, _mode_info, obj_likelihood);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, pqnr_linesearch_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small",
            _config_quasi, _kruskal_info, _mode_info, lineSearch);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, pqnr_searchDir_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small",
       _config_quasi, _kruskal_info, _mode_info, searchDirPqnr);
   }



#ifdef __NVCC__
           // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_setup_localpi_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
   _kruskal_info = {10};
   _mode_info={870};
   _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("medium",
           _config_quasi, _kruskal_info, _mode_info, set_up_pi);
}



#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_compute_phi_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("medium",
              _config_quasi, _kruskal_info, _mode_info, compute_phi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_obj_likelihood_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("medium",
              _config_quasi, _kruskal_info, _mode_info, obj_likelihood);
   }


#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_linesearch_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("medium",
                                                                                                                               _config_quasi, _kruskal_info, _mode_info, lineSearch);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_searchDir_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("medium",
                                                                                                                               _config_quasi, _kruskal_info, _mode_info, searchDirPqnr);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_setup_localpi_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
   _kruskal_info = {40};
   _mode_info={2661};
   _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large", _config_quasi, _kruskal_info, _mode_info, set_up_pi);
}


#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_compute_phi_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
{
   _kruskal_info = {40};
   _mode_info={2661};
   _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large", _config_quasi, _kruskal_info, _mode_info, compute_phi);
}

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_obj_likelihood_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large", _config_quasi, _kruskal_info, _mode_info, obj_likelihood);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_linesearch_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large",
                   _config_quasi, _kruskal_info, _mode_info, lineSearch);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pqnr_searchDir_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemQuasiNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large",
                                                                                                                               _config_quasi, _kruskal_info, _mode_info, searchDirPqnr );
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_setup_localpi_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small", _config_Damped, _kruskal_info, _mode_info, set_up_pi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_compute_phi_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small", _config_Damped, _kruskal_info, _mode_info, compute_phi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_obj_likelihood_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small", _config_Damped, _kruskal_info, _mode_info, obj_likelihood);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_linesearch_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small", _config_Damped, _kruskal_info, _mode_info, lineSearch);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_searchDir_small)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {   _kruskal_info = {5};
      _mode_info={100};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("small", _config_Damped, _kruskal_info, _mode_info, searchDirPdnr);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_setup_localpi_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("medium", _config_Damped, _kruskal_info, _mode_info, set_up_pi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_compute_phi_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>, numeric_traits >("medium",_config_Damped, _kruskal_info, _mode_info, compute_phi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_obj_likelihood_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>, numeric_traits >("medium", _config_Damped, _kruskal_info, _mode_info, obj_likelihood);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_linesearch_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>, numeric_traits >("medium", _config_Damped, _kruskal_info, _mode_info, lineSearch);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_searchDir_medium)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {10};
      _mode_info={870};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>, numeric_traits >("medium", _config_Damped, _kruskal_info, _mode_info, searchDirPdnr);
   }


#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_setup_localpi_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large", _config_Damped, _kruskal_info, _mode_info, set_up_pi);
   }


#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_compute_phi_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large", _config_Damped, _kruskal_info, _mode_info, compute_phi);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_obj_likelihood_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large", _config_Damped, _kruskal_info, _mode_info, obj_likelihood);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_linesearch_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large",
                                                                                                               _config_Damped, _kruskal_info, _mode_info, lineSearch);
   }

#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#define private public
#endif
   TEST_F(RowSubProblemUnitTest, DISABLED_pdnr_searchDir_large)
#ifdef __NVCC__
   // CUDA lambda issue UGLY workaround
#undef private
#endif
   {
      _kruskal_info = {40};
      _mode_info={2661};
      _sparten_rowsubproblem_test_driver<RowSubProblemDampedNewtonDriver<Kokkos::TeamPolicy<>, numeric_traits>,numeric_traits >("large",
                                                                                                                                _config_Damped, _kruskal_info, _mode_info, searchDirPdnr);
   }

} // unitTest
