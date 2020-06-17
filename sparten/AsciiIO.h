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
 \brief Class declaration for ASCII file I/O
 */

#pragma once

#include <sparten/KruskalTensor.h>
#include <sparten/SparseTensor.h>
#include <sparten/spartenUtils.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sparten
{

//! File I/O for ASCII files
template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
class AsciiIO
{
public:
  //! File I/O mode
  enum Access
  {
    READ,           //!< Read-only
    WRITE_APPEND,   //!< Write, append to end of file
    OVERWRITE,      //!< Write, erase any previous data in file
    READ_WRITE      //!< Read and write access to same file
  };

  //! ASCII number formatting
  enum Format
  {
    FIXED_PRECISION,  //!< Prints numbers in ASCII file using fixed precision (user-specified precision)
    SCIENTIFIC        //!< Prints numbers in ASCII file using scientific notation
  };

  //! File index style (0-index or 1-index for MatLab compatability)
  enum IndexStyle
  {
    INDEX_0,  //!< Indices start at 0
    INDEX_1   //!< Indices start at 1 for Matlab compatability
  };

  AsciiIO(std::string const &filename, Access mode, SubIdx precision);                                    //!< Constructor
  AsciiIO(std::string const &filename, Access mode, Format number, SubIdx precision);                     //!< Constructor
  AsciiIO(std::string const &filename, Access mode, Format number, SubIdx precision, IndexStyle style);   //!< Constructor

  virtual ~AsciiIO();                                                                                     //!< Destructor

  virtual void open_file(Access mode);     //!< Opens the specified file
  virtual void close_file();               //!< Closes the specified file

  virtual SparseTensor<SparseValue, ElemIdx, SubIdx> *read();                           //!< Reads SparseTensor data from file
  SparseTensor<SparseValue, ElemIdx, SubIdx> read_sptensor();                           //!< Reads SparseTensor data from file
  virtual KruskalTensor<KruskalValue, SubIdx> *read_ktensor();                          //!< Reads KruskalTensor data from file
  virtual void readHeaderData();                                                        //!< Reads only the header data from file (fast)

  virtual void write(SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput);    //!< Writes SparseTensor data to file
  virtual void write(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput);          //!< Writes KruskalTensor data to file

  virtual bool usingOneIndex() const;                                                   //!< Returns true if using a 1-based index
  virtual std::string fileType() const;                                                 //!< Returns string containing file type
  virtual SubIdx nDimension() const;                                                    //!< Returns the number of dimensions
  virtual SubIdx const *nIndices() const;                                               //!< Returns the number of indices
  virtual SubIdx maxIndex() const;                                                      //!< Returns the maximum index
  virtual ElemIdx nElement() const;                                                     //!< Returns the number of elements

protected:
  bool _isFileOpen;       //!< File open status
  bool _usingOneIndex;    //!< Flag for using 1-index (default is 0-index)
  SubIdx _precision;      //!< Formatting precision
  SubIdx _nDim;           //!< Number of dimensions
  SubIdx *_nIndices;      //!< Array of sizes of each dimension
  SubIdx _maxIndex;       //!< Maximum dimension size
  ElemIdx _nElement;      //!< Number of non-zero elements
  Access _mode;           //!< File access mode
  Format _format;         //!< Number format
  std::string _filename;  //!< Filename
  std::fstream _file;     //!< File
  std::string _fileType;  //!< Format type ("sptensor", etc.)

  //void read_factored_matrix(FactorMatrix<KruskalValue> inputMatrix);                    //!< Reads a factor matrix
  void read_factored_matrix(KruskalTensor<KruskalValue, SubIdx> *kruskalOutput, SubIdx iFactor);                    //!< Reads a factor matrix
  void get_line_error_check(std::fstream const &input);                                 //!< Checks for errors in reading line from file
};

} // sparten
