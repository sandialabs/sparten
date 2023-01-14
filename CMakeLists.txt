## ************************************************************************
##     SparTen: Software for Sparse Tensor Decompositions
##     by Sandia National Laboratories
##
## Sandia National Laboratories is a multimission laboratory managed
## and operated by National Technology and Engineering Solutions of Sandia,
## LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
## U.S. Department of Energy's National Nuclear Security Administration under
## contract DE-NA0003525.
##
## Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
## (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
## Government retains certain rights in this software.
## ************************************************************************

cmake_minimum_required(VERSION 3.16)

project(Sparten CXX C)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put all the executables in a /bin subdirectory.
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Sparten_BINARY_DIR}/bin)
# Put all the libraries in a /lib subdirectory.
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Sparten_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Sparten_BINARY_DIR}/lib)

#SET(Kokkos_ENABLE_OPENMP ON)
#add_definitions(-DSPARTEN_CHECK_NANS)
include(CTest)
SET(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Use Kokkos OpenMP backend")
IF(Kokkos_ENABLE_CUDA)
  SET(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA_LAMBDA")
ENDIF()


##############################################################################
# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)


##############################################################################
# Third Party Libraries (Kokkos)
add_subdirectory(${Sparten_SOURCE_DIR}/tpl)


##############################################################################
# SparTen version number
SET(SPARTEN_MAJOR_VERSION 1)
SET(SPARTEN_MINOR_VERSION 1)
SET(SPARTEN_BUILD_VERSION "0")
SET(SPARTEN_VERSION
  "${SPARTEN_MAJOR_VERSION}.${SPARTEN_MINOR_VERSION}.${SPARTEN_BUILD_VERSION}")

##############################################################################
# Make compiler info available through SparTen
SET(SPARTEN_COMPILER "${CMAKE_CXX_COMPILER}")
SET(SPARTEN_HOSTNAME "$ENV{HOSTNAME}")

##############################################################################
# SparTen configuration file creation
CONFIGURE_FILE (
  ${Sparten_SOURCE_DIR}/cmake/Sparten_Configure.h.in
  ${Sparten_BINARY_DIR}/include/Sparten_Configure.h
)
INCLUDE_DIRECTORIES (${Sparten_BINARY_DIR}/include)
INCLUDE_DIRECTORIES (${Sparten_SOURCE_DIR}/tpl/kokkos/core/src)
INCLUDE_DIRECTORIES (${Sparten_BINARY_DIR}/tpl/kokkos)

##############################################################################
# HWLOC
find_library(HWLOC_LIBRARIES NAMES libhwloc.so
    HINTS ${HWLOC_DIR} ${PROJECT_SOURCE_DIR}/tpl/hwloc)

if (HWLOC_LIBRARIES-NOTFOUND)
  message("HWLOC_LIBRARIES not found!")
else (HWLOC_LIBRARIES-NOTFOUND)
  message("HWLOC_LIBRARIES = ${HWLOC_LIBRARIES}")
  add_definitions(-DKokkos_HAVE_HWLOC)
  add_definitions("-DHWLOC_PATH=${HWLOC_LIBRARIES}")
endif (HWLOC_LIBRARIES-NOTFOUND)


##############################################################################
# Compiler checks
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.5)
    message(FATAL_ERROR "SparTen requires GCC version 4.8.5 or later.")
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    message(FATAL_ERROR "SparTen requires AppleClang version 9.0 or later.")
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5.2)
    message(FATAL_ERROR "SparTen requires Clang version 3.5.2 or later.")
  endif()
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
#  message(FATAL_ERROR "SparTen has not been tested the Intel compiler yet.")
endif()

set(Kokkos_DIR "${PROJECT_SOURCE_DIR}/tpl/kokkos")
set(Kokkos_BUILD_DIR "${PROJECT_BINARY_DIR}/tpl/kokkos/build")
set(Kokkos_INSTALL_DIR "${PROJECT_BINARY_DIR}/tpl/kokkos/install")
if(Kokkos_HWLOC_DIR)
  set(HWLOC_STRING "--with-hwloc=${Kokkos_HWLOC_DIR}")
endif(Kokkos_HWLOC_DIR)

# Build main library
file(GLOB SPARTEN_SOURCES sparten/*.cpp sparten/*.c sparten/eti/*.cpp)

# Remove Newton methods on GPU for now
# Comment the following lines of code when dev/testing CUDA + Newton
if(Kokkos_ENABLE_CUDA)
  LIST(REMOVE_ITEM SPARTEN_SOURCES
    #${PROJECT_SOURCE_DIR}/sparten/CpAprDampedNewton.cpp
    ${PROJECT_SOURCE_DIR}/sparten/CpAprQuasiNewton.cpp
    ${PROJECT_SOURCE_DIR}/sparten/CpRowSubProblem.cpp
  )
endif()

if(DEFINED ENV{CRAYPE_VERSION})
add_library(sparten ${SPARTEN_SOURCES})
target_link_libraries(
  sparten
  kokkos
  )
else( )
add_library(sparten ${SPARTEN_SOURCES})
target_link_libraries(
  sparten
  kokkos
)
endif( )

target_include_directories(
  sparten
  PUBLIC driver
  PUBLIC ${Sparten_SOURCE_DIR}
  )

# Build Sparten.
add_executable(
  Sparten_main
  driver/main.cpp
  )

if(DEFINED ENV{CRAYPE_VERSION})
target_link_libraries(
  Sparten_main
  sparten
  sparten_generator
  )
else( )
target_link_libraries(
  Sparten_main
  sparten
  sparten_generator
  )
endif( )

if (Kokkos_ENABLE_CUDA)
  target_link_libraries(
  Sparten_main
  )
endif (Kokkos_ENABLE_CUDA)

# Build generator library
file(GLOB GENERATOR_SOURCES generator/*.cpp)
add_library(sparten_generator ${GENERATOR_SOURCES})
target_link_libraries(sparten_generator kokkos)
if (Kokkos_ENABLE_CUDA)
  target_link_libraries(
  sparten_generator
  )
endif (Kokkos_ENABLE_CUDA)

target_include_directories(
  sparten_generator
  PUBLIC ${Sparten_SOURCE_DIR}
  )

# Build Sparten tensor generator.
add_executable(
  Sparten_tensor_gen
  driver/main_problemgen.cpp
  )

target_link_libraries(
  Sparten_tensor_gen
  sparten_generator
  sparten
  )

if (Kokkos_ENABLE_CUDA)
  target_link_libraries(
  Sparten_tensor_gen
  )
endif (Kokkos_ENABLE_CUDA)

INCLUDE_DIRECTORIES (${Sparten_SOURCE_DIR}/test)

# Build Sparten unit tests.
SET(SPARTEN_SOURCES_UNIT
  test/unit_test/AsciiIOUnitTest.cpp
  test/unit_test/blasWrapperUnitTest.cpp
  test/unit_test/FactTestSetGeneratorUnitTest.cpp
  test/unit_test/DimensionUnitTest.cpp
  test/unit_test/DiscreteCumulativeDistributionUnitTest.cpp
  test/unit_test/LogUnitTest.cpp
  test/unit_test/KruskalTensorUnitTest.cpp
  test/unit_test/RandGenMersenneTwisterUnitTest.cpp
  test/unit_test/RowSubProblemUnitTest.cpp
  test/unit_test/SparseTensorUnitTest.cpp
  test/unit_test/SpartenUtilsUnitTest.cpp
  test/unit_test/unitTestMain.cpp
  test/unit_test/CommandLineOptionsUnitTest.cpp
  )
#message(${SPARTEN_SOURCES_UNIT})
add_executable(
  Sparten_unit_test
  ${SPARTEN_SOURCES_UNIT}
  )

# Build Sparten regression tests.
SET(SPARTEN_SOURCES_REGRESSION
  test/regression_test/AsciiIORegressionTest.cpp
  test/regression_test/CpAprRegressionTest_multiplicativeUpdate.cpp
  test/regression_test/FactTestSetGeneratorRegressionTest.cpp
  test/regression_test/DimensionRegressionTest.cpp
  test/regression_test/DiscreteCumulativeDistributionRegressionTest.cpp
  test/regression_test/LogRegressionTest.cpp
  test/regression_test/KruskalTensorRegressionTest.cpp
  test/regression_test/RandGenMersenneTwisterRegressionTest.cpp
  test/regression_test/SparseTensorRegressionTest.cpp
  test/regression_test/SpartenUtilsRegressionTest.cpp
  test/regression_test/regressionTestMain.cpp
)
#message(${SPARTEN_SOURCES_REGRESSION})
add_executable(
  Sparten_regression_test
  ${SPARTEN_SOURCES_REGRESSION}
  )

if(DEFINED ENV{CRAYPE_VERSION})
target_link_libraries(
  Sparten_unit_test
  gtest
  sparten
  sparten_generator
  )
target_link_libraries(
  Sparten_regression_test
  gtest
  sparten
  sparten_generator
  )

else( )
target_link_libraries(
  Sparten_unit_test
  gtest
  sparten
  sparten_generator
  )
target_link_libraries(
  Sparten_regression_test
  gtest
  sparten
  sparten_generator
  )
endif(  )

if (Kokkos_ENABLE_CUDA)
  target_link_libraries(
  Sparten_unit_test
  )
  target_link_libraries(
  Sparten_regression_test
  )
endif (Kokkos_ENABLE_CUDA)

# Copy test data to binary directory
file(COPY ${Sparten_SOURCE_DIR}/test/data DESTINATION ${Sparten_BINARY_DIR}/test)
add_test(NAME GTESTS_Sparten_unit_test COMMAND $<TARGET_FILE:Sparten_unit_test> WORKING_DIRECTORY ${Sparten_BINARY_DIR})

# Add a target to generate API documentation with Doxygen
find_package(Doxygen)
  if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/index.doc ${CMAKE_CURRENT_BINARY_DIR}/doc/index.doc @ONLY)
    add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)

##############################################################################
# CODE COVERAGE
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  if(CMAKE_COMPILER_IS_GNUCXX)
    message ("Setting up coverage")
    set(CMAKE_C_FLAGS_COVERAGE "-g -O0 --coverage")
    set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage")
    link_libraries(gcov)
    target_link_libraries(Sparten_unit_test gcov)
    target_link_libraries(Sparten_regression_test gcov)    
  else( )
    message ("Code coverage support not available (must use GCC compiler)")
  endif(CMAKE_COMPILER_IS_GNUCXX)
endif(CMAKE_BUILD_TYPE STREQUAL "Coverage")


set(PRINT_VARS 1)
if(PRINT_VARS)
  message("CMAKE_VERSION = '${CMAKE_VERSION}'")
  message("CMAKE_CXX_COMPILER = '${CMAKE_CXX_COMPILER}'")
  message("CMAKE_CXX_COMPILER_ID = '${CMAKE_CXX_COMPILER_ID}'")
  message("CMAKE_CXX_COMPILER_VERSION = '${CMAKE_CXX_COMPILER_VERSION}'")
  message("CMAKE_CXX_STANDARD = '${CMAKE_CXX_STANDARD}'")
  message("CMAKE_CXX_EXTENSIONS = '${CMAKE_CXX_EXTENSIONS}'")
  message("CMAKE_CXX_FLAGS = '${CMAKE_CXX_FLAGS}'")
  message("CMAKE_CPP_FLAGS = '${CMAKE_CPP_FLAGS}'")
  message("CMAKE_C_COMPILER = '${CMAKE_C_COMPILER}'")
  message("CMAKE_C_COMPILER_ID = '${CMAKE_C_COMPILER_ID}'")
  message("CMAKE_C_COMPILER_VERSION = '${CMAKE_C_COMPILER_VERSION}'")
  message("CMAKE_C_FLAGS = '${CMAKE_C_FLAGS}'")
  message("CMAKE_EXE_LINKER_FLAGS = '${CMAKE_EXE_LINKER_FLAGS}'")
  message("PROJECT_SOURCE_DIR = '${PROJECT_SOURCE_DIR}'")
  message("PROJECT_BINARY_DIR = '${PROJECT_BINARY_DIR}'")
endif(PRINT_VARS)

SET(${PROJECT_NAME}_CXX_COMPILER_INFO "compiler='${CMAKE_CXX_COMPILER}';id='${CMAKE_CXX_COMPILER_ID}';version='${CMAKE_CXX_COMPILER_VERSION}';flags='${CMAKE_CXX_FLAGS}';standard='${CMAKE_CXX_STANDARD}';extensions='${CMAKE_CXX_EXTENSIONS}'" CACHE INTERNAL "CXX compiler info")
SET(${PROJECT_NAME}_C_COMPILER_INFO "compiler='${CMAKE_C_COMPILER}';id='${CMAKE_C_COMPILER_ID}';version='${CMAKE_C_COMPILER_VERSION}';flags='${CMAKE_C_FLAGS}'" CACHE INTERNAL "C complier info")
