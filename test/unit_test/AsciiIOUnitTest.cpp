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

#include <unit_test/AsciiIOUnitTest.h>
#include <sparten/Log.h>
#include <sparten/SparseTensor.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <Sparten_Configure.h>

namespace unitTest
{

void AsciiIOUnitTest::SetUp()
{
  sparten::Log &log = sparten::Log::new_log(sparten::Log::TO_SCREEN, "noFileNeeded", sparten::Log::QUIET);

  _sptensorFilename = SPARTEN_TEST_DIR "/test/data/regression/sptensor_one_based_index.tns";
  _nDim = 3;
  _nElement = 18;
  _nIndices.push_back(5);
  _nIndices.push_back(4);
  _nIndices.push_back(3);
  _maxIndex = 5;
  _precision = 16;
  _sptensorHeader = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(_sptensorFilename, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  _sptensorHeader->readHeaderData();
}

void AsciiIOUnitTest::TearDown()
{
  delete _sptensorHeader;
  std::remove("./AsciiIOUnitTest.cpp.ktensor_error.temp");
  std::remove("./AsciiIOUnitTest.cpp.ktensor_fixed.temp");
  std::remove("./AsciiIOUnitTest.cpp.ktensor.temp");
  std::remove("./AsciiIOUnitTest.cpp.one-based.temp");
  std::remove("./AsciiIOUnitTest.cpp.temp");
  std::remove("./AsciiIOUnitTest.cpp.zero-based.temp");
}

TEST_F(AsciiIOUnitTest, usingOneIndex)
{
  ASSERT_EQ(_sptensorHeader->usingOneIndex(), true);
}

TEST_F(AsciiIOUnitTest, fileType)
{
  ASSERT_EQ(_sptensorHeader->fileType(), "sptensor");
}

TEST_F(AsciiIOUnitTest, nDimension)
{
  ASSERT_EQ(_sptensorHeader->nDimension(), _nDim);
}

TEST_F(AsciiIOUnitTest, nIndices)
{
  const sparten::type::SubIdx *nIndices;
  nIndices = _sptensorHeader->nIndices();
  for (sparten::type::SubIdx iDim = 0; iDim < _nDim; iDim++)
  {
    ASSERT_EQ(nIndices[iDim], _nIndices[iDim]);
  }
}

TEST_F(AsciiIOUnitTest, maxIndex)
{
  ASSERT_EQ(_sptensorHeader->maxIndex(), _maxIndex);
}

TEST_F(AsciiIOUnitTest, nElement)
{
  ASSERT_EQ(_sptensorHeader->nElement(), _nElement);
}


TEST_F(AsciiIOUnitTest, constructor_desctructor)
{
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor;
  // parameters: filename, mode, precision
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ASSERT_TRUE(sptensor != NULL);
  delete sptensor;
  // parameters: filename, mode, format, precision
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, _precision);
  ASSERT_TRUE(sptensor != NULL);
  delete sptensor;
  // parameters: filename, mode, format, precision, index style
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, _precision, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::INDEX_0);
  ASSERT_TRUE(sptensor != NULL);
  // destructor
  delete sptensor;
}

TEST_F(AsciiIOUnitTest, open_file)
{
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor;
  // READ
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ASSERT_TRUE(sptensor != NULL);
  delete sptensor;

  // WRITE_APPEND
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::WRITE_APPEND, _precision);
  ASSERT_TRUE(sptensor != NULL);
  delete sptensor;

  // OVERWRITE
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, _precision);
  ASSERT_TRUE(sptensor != NULL);
  delete sptensor;

  // READ_WRITE
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ_WRITE, _precision);
  ASSERT_TRUE(sptensor != NULL);
  delete sptensor;

  // failure: incorrect file mode
  ASSERT_ANY_THROW((sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.temp", (sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::Access)(-1), _precision)));

  // failure: file cannot be opened
  ASSERT_ANY_THROW((sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("/etcshouldnotbealbetoopenfile.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision)));
}

TEST_F(AsciiIOUnitTest, close_file)
{
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor;
  // normal file open and close
  sptensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::WRITE_APPEND, _precision);
  sptensor->close_file();
  // failure: trying to close file that is not open
  ASSERT_ANY_THROW(sptensor->close_file());

  //delete sptensor;
  delete sptensor;
}

TEST_F(AsciiIOUnitTest, read)
{
  // zero-based indexing
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor0;
  sptensor0 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  sptensor0->read();
  ASSERT_EQ(sptensor0->usingOneIndex(), false);

  // one-based indexing
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor1;
  sptensor1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_one_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  // check that the tensor matches what is in the file
  // Below are the contents of the file: test/data/regression/sptensor_zero_based_index.tns
  /*
    sptensor
    3
    5 4 3
    18
    1 1 1 1.0000000000000000e+00
    1 3 3 2.0000000000000000e+00
    2 2 2 3.0000000000000000e+00
    2 3 1 4.0000000000000000e+00
    2 3 2 5.0000000000000000e+00
    2 3 3 6.0000000000000000e+00
    2 4 2 7.0000000000000000e+00
    3 1 1 8.0000000000000000e+00
    3 1 2 9.0000000000000000e+00
    3 3 1 1.0000000000000000e+01
    3 3 2 1.1000000000000000e+01
    3 4 1 1.2000000000000000e+01
    3 4 3 1.3000000000000000e+01
    4 1 1 1.4000000000000000e+01
    4 1 2 1.5000000000000000e+01
    4 3 1 1.6000000000000000e+01
    5 1 3 1.7000000000000000e+01
    5 4 3 1.8000000000000000e+01
  */
  // header info
  sparten::SparseTensor<sparten::type::ElemIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *sparseOutput = sptensor1->read();
  ASSERT_STREQ((sptensor1->fileType()).c_str(), "sptensor");
  ASSERT_EQ(sptensor1->usingOneIndex(), true);
  ASSERT_EQ(sptensor1->nDimension(), 3);
  const sparten::type::SubIdx *nIndices = sptensor1->nIndices();
  ASSERT_EQ(nIndices[0], 5);
  ASSERT_EQ(nIndices[1], 4);
  ASSERT_EQ(nIndices[2], 3);
  ASSERT_EQ(sptensor1->maxIndex(), 5);
  ASSERT_EQ(sptensor1->nElement(), 18);

  // tensor indices (stored as zero-based in SparseTensor data objects)
#if defined(KOKKOS_ENABLE_CUDA)
  typename sparten::SparseIndices<sparten::type::SubIdx>::HostMirror idsView = sparseOutput->get_indices_host_view();
#else
  sparten::SparseIndices<sparten::type::SubIdx> idsView = sparseOutput->get_indices_view();
#endif
  ASSERT_EQ(0, idsView(0,0));  ASSERT_EQ(0, idsView(1,0));  ASSERT_EQ(0, idsView(2,0));
  ASSERT_EQ(0, idsView(0,1));  ASSERT_EQ(2, idsView(1,1));  ASSERT_EQ(2, idsView(2,1));
  ASSERT_EQ(1, idsView(0,2));  ASSERT_EQ(1, idsView(1,2));  ASSERT_EQ(1, idsView(2,2));
  ASSERT_EQ(1, idsView(0,3));  ASSERT_EQ(2, idsView(1,3));  ASSERT_EQ(0, idsView(2,3));
  ASSERT_EQ(1, idsView(0,4));  ASSERT_EQ(2, idsView(1,4));  ASSERT_EQ(1, idsView(2,4));
  ASSERT_EQ(1, idsView(0,5));  ASSERT_EQ(2, idsView(1,5));  ASSERT_EQ(2, idsView(2,5));
  ASSERT_EQ(1, idsView(0,6));  ASSERT_EQ(3, idsView(1,6));  ASSERT_EQ(1, idsView(2,6));
  ASSERT_EQ(2, idsView(0,7));  ASSERT_EQ(0, idsView(1,7));  ASSERT_EQ(0, idsView(2,7));
  ASSERT_EQ(2, idsView(0,8));  ASSERT_EQ(0, idsView(1,8));  ASSERT_EQ(1, idsView(2,8));
  ASSERT_EQ(2, idsView(0,9));  ASSERT_EQ(2, idsView(1,9));  ASSERT_EQ(0, idsView(2,9));
  ASSERT_EQ(2, idsView(0,10)); ASSERT_EQ(2, idsView(1,10)); ASSERT_EQ(1, idsView(2,10));
  ASSERT_EQ(2, idsView(0,11)); ASSERT_EQ(3, idsView(1,11)); ASSERT_EQ(0, idsView(2,11));
  ASSERT_EQ(2, idsView(0,12)); ASSERT_EQ(3, idsView(1,12)); ASSERT_EQ(2, idsView(2,12));
  ASSERT_EQ(3, idsView(0,13)); ASSERT_EQ(0, idsView(1,13)); ASSERT_EQ(0, idsView(2,13));
  ASSERT_EQ(3, idsView(0,14)); ASSERT_EQ(0, idsView(1,14)); ASSERT_EQ(1, idsView(2,14));
  ASSERT_EQ(3, idsView(0,15)); ASSERT_EQ(2, idsView(1,15)); ASSERT_EQ(0, idsView(2,15));
  ASSERT_EQ(4, idsView(0,16)); ASSERT_EQ(0, idsView(1,16)); ASSERT_EQ(2, idsView(2,16));
  ASSERT_EQ(4, idsView(0,17)); ASSERT_EQ(3, idsView(1,17)); ASSERT_EQ(2, idsView(2,17));
  // tensor values
#if defined(KOKKOS_ENABLE_CUDA)
  sparten::SparseData<sparten::type::ElemIdx>::HostMirror data = sparseOutput->get_data_host_view();
#else
  sparten::SparseData<sparten::type::ElemIdx> data = sparseOutput->get_data_view();
#endif
  for (int i=0; i < sptensor1->nElement(); i++)
    ASSERT_EQ(data[i], i+1);

  // failure: incorrect tensor type
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensorF0;
  sptensorF0 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/tensor_incorrect_tensor_type.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ASSERT_ANY_THROW(sptensorF0->read());

  // failure: file not open
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensorF1;
  sptensorF1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_one_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  sptensorF1->close_file();
  ASSERT_ANY_THROW(sptensorF1->read());
  delete(sptensor0);
  delete(sptensor1);
  delete(sptensorF0);
  delete(sptensorF1);
}

TEST_F(AsciiIOUnitTest, read_header_data)
{
  // zero-based indexing
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *header0;
  header0 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  header0->readHeaderData();
  ASSERT_EQ(header0->usingOneIndex(), false);
  ASSERT_STREQ((header0->fileType()).c_str(), "sptensor");

  // one-based indexing
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *header1;
  header1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_one_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  header1->readHeaderData();
  ASSERT_EQ(header1->usingOneIndex(), true);
  ASSERT_STREQ((header1->fileType()).c_str(), "sptensor");

  // failure: incorrect tensor type
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *headerF0;
  headerF0 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/tensor_incorrect_tensor_type.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ASSERT_ANY_THROW(headerF0->readHeaderData());

  // failure: file not open
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *headerF1;
  headerF1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_one_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  headerF1->close_file();
  ASSERT_ANY_THROW(headerF1->readHeaderData());
  delete(header0);
  delete(header1);
  delete(headerF0);
  delete(headerF1);
}

TEST_F(AsciiIOUnitTest, write)
{
  // read sparse tensor file
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor0;
  sptensor0 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/sptensor_zero_based_index.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  sparten::SparseTensor<sparten::type::ElemIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *sparseInput = sptensor0->read();

  // write and read tensor: one-based
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor1;
  sptensor1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.one-based.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, _precision);
  sptensor1->write(*sparseInput);
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *sptensor2;
  sptensor2 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.one-based.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  sparten::SparseTensor<sparten::type::ElemIdx, sparten::type::ElemIdx, sparten::type::SubIdx> *sparseOutput = sptensor2->read();
  // compare header info
  ASSERT_STREQ((sptensor2->fileType()).c_str(), "sptensor");
  ASSERT_EQ(sptensor2->usingOneIndex(), true);
  ASSERT_EQ(sptensor2->nDimension(), 3);
  const sparten::type::SubIdx *nIndices = sptensor2->nIndices();
  ASSERT_EQ(nIndices[0], 5);
  ASSERT_EQ(nIndices[1], 4);
  ASSERT_EQ(nIndices[2], 3);
  ASSERT_EQ(sptensor2->maxIndex(), 5);
  ASSERT_EQ(sptensor2->nElement(), 18);

  // tensor indices (stored as zero-based in SparseTensor data objects)
#if defined(KOKKOS_ENABLE_CUDA)
 // typename sparten::SparseIndices<sparten::type::SubIdx>::HostMirror idsView = sparseOutput->get_indices_host_view();
  auto idsView = sparseOutput->get_indices_host_view();
#else
  sparten::SparseIndices<sparten::type::SubIdx> idsView = sparseOutput->get_indices_view();
#endif
  ASSERT_EQ(0, idsView(0,0));  ASSERT_EQ(0, idsView(1,0));  ASSERT_EQ(0, idsView(2,0));
  ASSERT_EQ(0, idsView(0,1));  ASSERT_EQ(2, idsView(1,1));  ASSERT_EQ(2, idsView(2,1));
  ASSERT_EQ(1, idsView(0,2));  ASSERT_EQ(1, idsView(1,2));  ASSERT_EQ(1, idsView(2,2));
  ASSERT_EQ(1, idsView(0,3));  ASSERT_EQ(2, idsView(1,3));  ASSERT_EQ(0, idsView(2,3));
  ASSERT_EQ(1, idsView(0,4));  ASSERT_EQ(2, idsView(1,4));  ASSERT_EQ(1, idsView(2,4));
  ASSERT_EQ(1, idsView(0,5));  ASSERT_EQ(2, idsView(1,5));  ASSERT_EQ(2, idsView(2,5));
  ASSERT_EQ(1, idsView(0,6));  ASSERT_EQ(3, idsView(1,6));  ASSERT_EQ(1, idsView(2,6));
  ASSERT_EQ(2, idsView(0,7));  ASSERT_EQ(0, idsView(1,7));  ASSERT_EQ(0, idsView(2,7));
  ASSERT_EQ(2, idsView(0,8));  ASSERT_EQ(0, idsView(1,8));  ASSERT_EQ(1, idsView(2,8));
  ASSERT_EQ(2, idsView(0,9));  ASSERT_EQ(2, idsView(1,9));  ASSERT_EQ(0, idsView(2,9));
  ASSERT_EQ(2, idsView(0,10)); ASSERT_EQ(2, idsView(1,10)); ASSERT_EQ(1, idsView(2,10));
  ASSERT_EQ(2, idsView(0,11)); ASSERT_EQ(3, idsView(1,11)); ASSERT_EQ(0, idsView(2,11));
  ASSERT_EQ(2, idsView(0,12)); ASSERT_EQ(3, idsView(1,12)); ASSERT_EQ(2, idsView(2,12));
  ASSERT_EQ(3, idsView(0,13)); ASSERT_EQ(0, idsView(1,13)); ASSERT_EQ(0, idsView(2,13));
  ASSERT_EQ(3, idsView(0,14)); ASSERT_EQ(0, idsView(1,14)); ASSERT_EQ(1, idsView(2,14));
  ASSERT_EQ(3, idsView(0,15)); ASSERT_EQ(2, idsView(1,15)); ASSERT_EQ(0, idsView(2,15));
  ASSERT_EQ(4, idsView(0,16)); ASSERT_EQ(0, idsView(1,16)); ASSERT_EQ(2, idsView(2,16));
  ASSERT_EQ(4, idsView(0,17)); ASSERT_EQ(3, idsView(1,17)); ASSERT_EQ(2, idsView(2,17));
  // tensor values
#if defined(KOKKOS_ENABLE_CUDA)
 // sparten::SparseData<sparten::type::ElemIdx>::HostMirror data = sparseOutput->get_data_host_view();
  auto data = sparseOutput->get_data_host_view();
#else
  sparten::SparseData<sparten::type::ElemIdx> data = sparseOutput->get_data_view();
#endif
  for (int i=0; i < sptensor2->nElement(); i++)
    ASSERT_EQ(data[i], i+1);

  delete(sptensor0);
  delete(sptensor1);
  delete(sptensor2);
  // write and read tensor: zero-based, FIXED_PRECISION
  sptensor1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.zero-based.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, _precision, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::INDEX_0);
  sptensor1->write(*sparseInput);
  sptensor2 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.zero-based.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, _precision, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::INDEX_0);
  sparseOutput = sptensor2->read();
  // compare header info
  ASSERT_STREQ((sptensor2->fileType()).c_str(), "sptensor");
  ASSERT_EQ(sptensor2->usingOneIndex(), false);
  ASSERT_EQ(sptensor2->nDimension(), 3);
  nIndices = sptensor2->nIndices();
  ASSERT_EQ(nIndices[0], 5);
  ASSERT_EQ(nIndices[1], 4);
  ASSERT_EQ(nIndices[2], 3);
  ASSERT_EQ(sptensor2->maxIndex(), 5);
  ASSERT_EQ(sptensor2->nElement(), 18);
  // tensor indices (stored as zero-based in SparseTensor data objects)
#if defined(KOKKOS_ENABLE_CUDA)
  idsView = sparseOutput->get_indices_host_view();
#else
  idsView = sparseOutput->get_indices_view();
#endif
  ASSERT_EQ(0, idsView(0,0));  ASSERT_EQ(0, idsView(1,0));  ASSERT_EQ(0, idsView(2,0));
  ASSERT_EQ(0, idsView(0,1));  ASSERT_EQ(2, idsView(1,1));  ASSERT_EQ(2, idsView(2,1));
  ASSERT_EQ(1, idsView(0,2));  ASSERT_EQ(1, idsView(1,2));  ASSERT_EQ(1, idsView(2,2));
  ASSERT_EQ(1, idsView(0,3));  ASSERT_EQ(2, idsView(1,3));  ASSERT_EQ(0, idsView(2,3));
  ASSERT_EQ(1, idsView(0,4));  ASSERT_EQ(2, idsView(1,4));  ASSERT_EQ(1, idsView(2,4));
  ASSERT_EQ(1, idsView(0,5));  ASSERT_EQ(2, idsView(1,5));  ASSERT_EQ(2, idsView(2,5));
  ASSERT_EQ(1, idsView(0,6));  ASSERT_EQ(3, idsView(1,6));  ASSERT_EQ(1, idsView(2,6));
  ASSERT_EQ(2, idsView(0,7));  ASSERT_EQ(0, idsView(1,7));  ASSERT_EQ(0, idsView(2,7));
  ASSERT_EQ(2, idsView(0,8));  ASSERT_EQ(0, idsView(1,8));  ASSERT_EQ(1, idsView(2,8));
  ASSERT_EQ(2, idsView(0,9));  ASSERT_EQ(2, idsView(1,9));  ASSERT_EQ(0, idsView(2,9));
  ASSERT_EQ(2, idsView(0,10)); ASSERT_EQ(2, idsView(1,10)); ASSERT_EQ(1, idsView(2,10));
  ASSERT_EQ(2, idsView(0,11)); ASSERT_EQ(3, idsView(1,11)); ASSERT_EQ(0, idsView(2,11));
  ASSERT_EQ(2, idsView(0,12)); ASSERT_EQ(3, idsView(1,12)); ASSERT_EQ(2, idsView(2,12));
  ASSERT_EQ(3, idsView(0,13)); ASSERT_EQ(0, idsView(1,13)); ASSERT_EQ(0, idsView(2,13));
  ASSERT_EQ(3, idsView(0,14)); ASSERT_EQ(0, idsView(1,14)); ASSERT_EQ(1, idsView(2,14));
  ASSERT_EQ(3, idsView(0,15)); ASSERT_EQ(2, idsView(1,15)); ASSERT_EQ(0, idsView(2,15));
  ASSERT_EQ(4, idsView(0,16)); ASSERT_EQ(0, idsView(1,16)); ASSERT_EQ(2, idsView(2,16));
  ASSERT_EQ(4, idsView(0,17)); ASSERT_EQ(3, idsView(1,17)); ASSERT_EQ(2, idsView(2,17));
  // tensor values
#if defined(KOKKOS_ENABLE_CUDA)
  data = sparseOutput->get_data_host_view();
#else
  data = sparseOutput->get_data_view();
#endif
  for (int i=0; i < sptensor2->nElement(); i++)
    ASSERT_EQ(data[i], i+1);

  delete(sptensor1);
  delete(sptensor2);
  // failure: write when file is closed
  sptensor1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.zero-based.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, _precision, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::INDEX_0);
  sptensor1->close_file();
  ASSERT_ANY_THROW((sptensor1->write(*sparseInput)));
  delete(sptensor1);

}

  //#if defined(KOKKOS_ENABLE_CUDA)
  //TEST_F(AsciiIOUnitTest, DISABLED_read_ktensor)
  //#else
TEST_F(AsciiIOUnitTest, read_ktensor)
//#endif
{
  // read_ktensor
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensor;
  ktensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/ktensor.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);

  // check that the tensor matches what is in the file
  // Below are the contents of the file: test/data/regression/ktensor.tns
  /*
     ktensor
     3
     5 4 3
     2
     1.2444021606445312e+02 4.6559776306152344e+01
     matrix
     2
     5 2
     2.3490464314818382e-02 0.0000000000000000e+00
     1.9575387239456177e-01 5.1848043180018232e-44
     4.2839869856834412e-01 1.9147317111492157e-01
     3.5235697031021118e-01 8.5291572459440748e-40
     6.1867687622257916e-13 8.0852687358856201e-01
     matrix
     2
     4 2
     3.7429088354110718e-01 3.7420859932899475e-01
     2.3891076445579529e-02 0.0000000000000000e+00
     4.3003934621810913e-01 2.8286926787001008e-30
     1.7177873849868774e-01 6.2579137086868286e-01
     matrix
     2
     3 2
     5.2233910560607910e-01 1.3735417780935677e-07
     4.0179938077926636e-01 2.4244684038166042e-10
     7.5861521065235138e-02 9.9999988079071045e-01
  */
  // header info
  sparten::KruskalTensor<double, sparten::type::SubIdx> *ktensorOutput = ktensor->read_ktensor();
  ASSERT_STREQ((ktensor->fileType()).c_str(), "ktensor");
  ASSERT_EQ(ktensorOutput->get_nDim(), 3);
  ASSERT_EQ(ktensorOutput->get_nComponent(), 2);
  ASSERT_EQ(ktensorOutput->get_factor_matrix_nRow(0), 5);
  ASSERT_EQ(ktensorOutput->get_factor_matrix_nRow(1), 4);
  ASSERT_EQ(ktensorOutput->get_factor_matrix_nRow(2), 3);
  ASSERT_EQ(ktensorOutput->get_factor_matrix_nColumn(0), 2);
  ASSERT_EQ(ktensorOutput->get_factor_matrix_nColumn(1), 2);
  ASSERT_EQ(ktensorOutput->get_factor_matrix_nColumn(2), 2);
  delete(ktensor);
  delete(ktensorOutput);

  // failure: incorrect factor matrix type
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensorF0;
  ktensorF0 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/ktensor_factor_matrix_error.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ASSERT_ANY_THROW(ktensorF0->read_ktensor());
  delete(ktensorF0);

  // failure: file not open
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensorF1;
  ktensorF1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/ktensor.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ktensorF1->close_file();
  ASSERT_ANY_THROW(ktensorF1->read_ktensor());
  delete(ktensorF1);

  // failure: incorrect tensor type
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensorF2;
  ktensorF2 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/ktensor_incorrect_tensor_type.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  ASSERT_ANY_THROW(ktensorF2->read_ktensor());
  delete(ktensorF2);

}

TEST_F(AsciiIOUnitTest, write_ktensor)
{
  // Currently disabled due to unimplemented KruskalTensor::compare
  // read_ktensor
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensor;
  ktensor = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>(SPARTEN_TEST_DIR "/test/data/regression/ktensor.tns", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  sparten::KruskalTensor<double, sparten::type::SubIdx> *ktensorInput = ktensor->read_ktensor();

  // check that the tensor matches what is in the file
  // Below are the contents of the file: test/data/regression/ktensor.tns
  /*
     ktensor
     3
     5 4 3
     2
     1.2444021606445312e+02 4.6559776306152344e+01
     matrix
     2
     5 2
     2.3490464314818382e-02 0.0000000000000000e+00
     1.9575387239456177e-01 5.1848043180018232e-44
     4.2839869856834412e-01 1.9147317111492157e-01
     3.5235697031021118e-01 8.5291572459440748e-40
     6.1867687622257916e-13 8.0852687358856201e-01
     matrix
     2
     4 2
     3.7429088354110718e-01 3.7420859932899475e-01
     2.3891076445579529e-02 0.0000000000000000e+00
     4.3003934621810913e-01 2.8286926787001008e-30
     1.7177873849868774e-01 6.2579137086868286e-01
     matrix
     2
     3 2
     5.2233910560607910e-01 1.3735417780935677e-07
     4.0179938077926636e-01 2.4244684038166042e-10
     7.5861521065235138e-02 9.9999988079071045e-01
  */
  // header info

  delete(ktensor);
  // write_ktensor
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensor1;
  ktensor1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.ktensor.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, _precision);
  ktensor1->write(*ktensorInput);
  delete(ktensor1);

  // read_ktensor
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensor2;
  ktensor2 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.ktensor.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, _precision);
  sparten::KruskalTensor<double, sparten::type::SubIdx> *ktensorOutput = ktensor2->read_ktensor();
  delete(ktensor2);

  // compare input and output
  ASSERT_TRUE(ktensorOutput->compare(*ktensorInput, 1e-5));
  delete(ktensorOutput);

  // write_ktensor, fixed precision
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensor3;
  ktensor3 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.ktensor_fixed.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, 5);
  ktensor3->write(*ktensorInput);
  delete(ktensor3);

  // read_ktensor, fixed precision
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensor4;
  ktensor4 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.ktensor_fixed.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::READ, sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::FIXED_PRECISION, 5);
  sparten::KruskalTensor<double, sparten::type::SubIdx> *ktensorOutput2 = ktensor4->read_ktensor();
  delete(ktensor4);
  // compare input and output
  ASSERT_TRUE(ktensorOutput2->compare(*ktensorInput, 1e-5));
  delete(ktensorInput);

  // failure: file not open
  sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx> *ktensorF1;
  ktensorF1 = new sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>("./AsciiIOUnitTest.cpp.ktensor_error.temp", sparten::AsciiIO<sparten::type::ElemIdx, double, sparten::type::ElemIdx, sparten::type::SubIdx>::OVERWRITE, _precision);
  ktensorF1->close_file();
  ASSERT_ANY_THROW(ktensorF1->write(*ktensorOutput2));
  delete(ktensorOutput2);
  delete(ktensorF1);
}

} // unitTest
