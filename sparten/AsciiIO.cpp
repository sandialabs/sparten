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
 \brief Class definition for ASCII file I/O
 */

#include <sparten/AsciiIO.h>
#include <sparten/Log.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sparten
{

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::AsciiIO(std::string const &filename, Access mode, SubIdx precision) :
    _filename(filename),
    _mode(mode),
    _isFileOpen(false),
    _format(SCIENTIFIC),
    _precision(precision),
    _usingOneIndex(true),
    _fileType(""),
    _nDim(0),
    _nIndices(nullptr),
    _maxIndex(0),
    _nElement(0)
{
  open_file(mode);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::AsciiIO(std::string const &filename, Access mode, Format number, SubIdx precision) :
    _filename(filename),
    _mode(mode),
    _isFileOpen(false),
    _format(number),
    _precision(precision),
    _usingOneIndex(true),
    _fileType(""),
    _nDim(0),
    _nIndices(nullptr),
    _maxIndex(0),
    _nElement(0)
{
  open_file(mode);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::AsciiIO(std::string const &filename, Access mode, Format number, SubIdx precision, IndexStyle style) :
    _filename(filename),
    _mode(mode),
    _isFileOpen(false),
    _format(number),
    _precision(precision),
    _usingOneIndex(style),
    _fileType(""),
    _nDim(0),
    _nIndices(nullptr),
    _maxIndex(0),
    _nElement(0)
{
  open_file(mode);
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::~AsciiIO()
{
  if (_isFileOpen)
  {
    _file.close();
  }
  
  if (_nIndices != nullptr)
  {
    delete [] _nIndices;
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::open_file(Access mode)
{
  Log &log = Log::new_log();

  switch (mode)
  {
  case READ:
    _file.open(_filename.c_str(), std::fstream::in);
    _isFileOpen = true;
    break;
  case WRITE_APPEND:
    _file.open(_filename.c_str(), std::fstream::out | std::fstream::app);
    _isFileOpen = true;
    break;
  case OVERWRITE:
    _file.open(_filename.c_str(), std::fstream::out);
    _isFileOpen = true;
    break;
  case READ_WRITE:
    _file.open(_filename.c_str(), std::fstream::in | std::fstream::out);
    _isFileOpen = true;
    break;
  default:
    throw std::runtime_error("Invalid file I/O mode selected.");
  }

  if (!_file.is_open())
  {
    throw std::runtime_error("ASCII file failed to open. " + _filename);
  }
  else
  {
    log.print("\tOpening ASCII file: " + _filename, Log::DEBUG_1);
    _isFileOpen = true;
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::close_file()
{
  Log &log = Log::new_log();

  if (_isFileOpen)
  {
    _file.close();
    _isFileOpen = false;
    log.print("\tClosing ASCII file: " + _filename, Log::DEBUG_1);
  }
  else
  {
    throw std::runtime_error("File is already closed.");
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx> *AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::read()
{
  Log &log = Log::new_log();
  log.print("\tReading ASCII file: " + _filename, Log::DEBUG_1);

  SparseTensor<SparseValue, ElemIdx, SubIdx> *sparseOutput = nullptr;

  readHeaderData();
  
  std::string line;

  // Create sparse tensor
  std::vector<SubIdx> nIndicesVector(_nIndices, _nIndices + _nDim);
  sparseOutput = new SparseTensor<SparseValue, ElemIdx, SubIdx>(_nDim, _nElement, nIndicesVector);
  // OLD: sparseOutput = new SparseTensor<SparseValue, ElemIdx, SubIdx>(nDim, nElement, nIndicesVector);

  // Read sparse tensor contents
  ElemIdx elemId = 0;
  while (std::getline(_file, line))
  {
    get_line_error_check(_file);
    std::istringstream iss(line);

    // Read indices
    for (SubIdx iDim = 0; iDim < _nDim; ++iDim)
    {
      SubIdx tmpIndex;
      iss >> tmpIndex;

      sparseOutput->set_element_index(elemId, iDim, tmpIndex - (_usingOneIndex ? 1 : 0));
    }

    double tmpDValue;
    SparseValue tmpValue;

    iss >> tmpDValue;
    tmpValue =(SparseValue)tmpDValue;
    sparseOutput->set_element_data(elemId, tmpValue);

      ++elemId;
    }
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  // Move Data to the device memory
  sparseOutput->copy_to_device(); 
#endif
  log.print("\tDone reading ASCII file: " + _filename, Log::DEBUG_1);
  return sparseOutput;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SparseTensor<SparseValue, ElemIdx, SubIdx> AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::read_sptensor()
{
  Log &log = Log::new_log();
  log.print("\tReading ASCII file: " + _filename, Log::DEBUG_1);

  readHeaderData();

  std::string line;

  // Create sparse tensor
  std::vector<SubIdx> nIndicesVector(_nIndices, _nIndices + _nDim);
  SparseTensor<SparseValue, ElemIdx, SubIdx> sparseOutput(_nDim, _nElement, nIndicesVector);

  // Read sparse tensor contents
  ElemIdx elemId = 0;
  while (std::getline(_file, line))
  {
    get_line_error_check(_file);
    std::istringstream iss(line);

    // Read indices
    for (SubIdx iDim = 0; iDim < _nDim; ++iDim)
    {
      SubIdx tmpIndex;
      iss >> tmpIndex;

      sparseOutput.set_element_index(elemId, iDim, tmpIndex - (_usingOneIndex ? 1 : 0));
    }

    double tmpDValue;
    SparseValue tmpValue;

    iss >> tmpDValue;
    tmpValue =(SparseValue)tmpDValue;
    sparseOutput.set_element_data(elemId, tmpValue);

      ++elemId;
    }
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_USE_CUDA_UVM)
  // Move Data to the device memory
  sparseOutput.copy_to_device();
#endif
  log.print("\tDone reading ASCII file: " + _filename, Log::DEBUG_1);
  return sparseOutput;
}


template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::read_factored_matrix(KruskalTensor<KruskalValue, SubIdx> *kruskalOutput, SubIdx iFactor)
{
  if (_isFileOpen)
  {
    std::string line;
    std::getline(_file, line); /// \todo Add error check here
    if (line.compare("matrix") == 0)
    {
      std::getline(_file, line);  // Always "2"
      std::getline(_file, line);  // Get matrix size
      std::istringstream line_ss(line);
      SubIdx matSize[2];
      const SubIdx nDim = 2;
      for (SubIdx iDim = 0; iDim < nDim; ++iDim)
      {
        line_ss >> matSize[iDim];
      }
      for (SubIdx iRow = 0; iRow < matSize[0]; ++iRow)
      {
        std::getline(_file, line);
        std::istringstream line_elements(line);
	KruskalValue tmpValue;
        for (SubIdx iCol = 0; iCol < matSize[1]; ++iCol)
        {
	  line_elements >> tmpValue;
	  kruskalOutput->set_factor_matrix_element(tmpValue, iFactor, iRow, iCol);
        }
      }
    }
    else
    {
      throw std::runtime_error("Incorrect factor matrix type (should be 'matrix')  in ASCII file.");
    }
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
KruskalTensor<KruskalValue, SubIdx> * AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::read_ktensor()
{
  KruskalTensor<KruskalValue, SubIdx> *kruskalOutput = nullptr;

  if (_isFileOpen)
  {
    std::string line;
    // Check type
    std::getline(_file, line); /// \todo Add error check here
    if (line.compare("ktensor") == 0)
    {
        _usingOneIndex = false;
        _fileType = "ktensor";
    }
    else
    {
      throw std::runtime_error("Incorrect tensor type (should be 'ktensor') in ASCII file.");
    }
    // Read number of dimensions
    std::getline(_file, line); /// \todo Add error check here
    SubIdx nDim = std::stoll(line);

    std::vector<SubIdx> cDims(nDim);

    // Read size of each dimension
    std::getline(_file, line); /// \todo Add error check here
    std::istringstream line_ss(line);
    for (SubIdx iDim = 0; iDim < nDim; ++iDim)
    {
      line_ss >> cDims[iDim];
    }

    // Read the number of components
    std::getline(_file, line); /// \todo Add error check here
    SubIdx nComponents = std::stoll(line);;

    kruskalOutput = new KruskalTensor<KruskalValue, SubIdx>(cDims.size(), nComponents, cDims);
    std::vector<KruskalValue> lambda(nComponents);

    // Read Lambda (weight factor)
    std::getline(_file, line); /// \todo Add error check here

    std::istringstream line_ss2(line);

    for (SubIdx iComp = 0; iComp < nComponents; ++iComp)
    {
      line_ss2 >> lambda[iComp];
    }
    kruskalOutput->set_weight_vector(lambda);

    // Read Factored Matrix
    for (SubIdx iDim = 0; iDim < nDim; ++iDim)
    {
      read_factored_matrix(kruskalOutput, iDim);
    }
    kruskalOutput->move_to_device();
  }
  else
  {
    throw std::runtime_error("File needs to be open before reading file.");
  }

  return kruskalOutput;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::write(SparseTensor<SparseValue, ElemIdx, SubIdx> const &sparseInput)
{
  if (_isFileOpen)
  {
    _file << (_usingOneIndex ? "sptensor" : "sptensor indices-start-at-zero") << std::endl;
    _file << sparseInput.get_nDim() << std::endl;

    for (SubIdx i = 0; i < sparseInput.get_nDim(); ++i)
    {
      if (i > 0)
      {
        _file << " ";
      }
      _file << sparseInput.get_size_dim()[i];
    }

    _file << std::endl;
    _file << sparseInput.get_nElement() << std::endl;

    if (_format == SCIENTIFIC)
    {
      _file << std::setiosflags(std::ios::scientific);
    }
    else
    {
      _file << std::fixed;
    }

    _file << std::setprecision(_precision);

    // get views for indices and data before writing to file
#if defined(KOKKOS_ENABLE_CUDA)
    auto indices = sparseInput.get_indices_host_view();
    auto data = sparseInput.get_data_host_view();
#else
    auto indices = sparseInput.get_indices_view();
    auto data = sparseInput.get_data_view();
#endif
    for (ElemIdx elemId = 0; elemId < sparseInput.get_nElement(); ++elemId)
    {
      for (SubIdx index = 0; index < sparseInput.get_nDim(); ++index)
      {
        _file << indices(index, elemId) + static_cast<SubIdx>(_usingOneIndex) << " ";
      }

      _file << data(elemId) << std::endl;
    }

  }
  else
  {
    throw std::runtime_error("File needs to be open before writing file.");
  }

}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::write(KruskalTensor<KruskalValue, SubIdx> const &kruskalInput)
{
  if (_isFileOpen)
  {

#if defined(KOKKOS_ENABLE_CUDA)
    kruskalInput.move_to_host();
#endif
    _file << "ktensor" << std::endl;
    _file << kruskalInput.get_nDim() << std::endl;
    for (SubIdx iDim = 0; iDim < kruskalInput.get_nDim(); ++iDim)
    {
      if (iDim > 0)
      {
        _file << " ";
      }
#if defined(KOKKOS_ENABLE_CUDA)
      _file << kruskalInput.get_factor_matrix_nRow_host(iDim);
#else
      _file << kruskalInput.get_factor_matrix_nRow(iDim);
#endif
    }
    _file << std::endl;
 
    _file << kruskalInput.get_nComponent() << std::endl;

    if (_format == SCIENTIFIC)
    {
      _file << std::setiosflags(std::ios::scientific);
    }
    else
    {
      _file << std::fixed;
    }

    _file << std::setprecision(_precision);

    for (SubIdx iComp = 0; iComp < kruskalInput.get_nComponent(); ++iComp)
    {
      if (iComp > 0 )
      {
        _file << " ";
      }
#if defined(KOKKOS_ENABLE_CUDA)
      _file << kruskalInput.get_weight_element_host(iComp);
#else
      _file << kruskalInput.get_weight_element(iComp);
#endif
    }
    _file << std::endl;

    // Write the factored matrices
    for (SubIdx iDim = 0; iDim < kruskalInput.get_nDim(); ++iDim)
    {

      _file << "matrix" << std::endl;
      _file << "2" << std::endl;
      _file << kruskalInput.get_factor_matrix_nRow(iDim) << " "
            << kruskalInput.get_factor_matrix_nColumn(iDim) << std::endl;

      // Write elements for each row on one line.
      for (SubIdx iRow = 0; iRow < kruskalInput.get_factor_matrix_nRow(iDim) ; ++iRow)
      {
        for (SubIdx iCol = 0;  iCol < kruskalInput.get_factor_matrix_nColumn(iDim); ++iCol)
        {
          if (iCol > 0)
          {
            _file << " ";
          }
#if defined(KOKKOS_ENABLE_CUDA)
          _file << kruskalInput.get_factor_matrix_element_host(iDim, iRow, iCol);
#else
          _file << kruskalInput.get_factor_matrix_element(iDim, iRow, iCol);
#endif
        }
        _file << std::endl;
      }
    }
  }
  else
  {
    throw std::runtime_error("File needs to be open before writing file.");
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::get_line_error_check(std::fstream const &input)
{
  if ((input.fail()) != 0)
  {
    throw std::runtime_error("Logical I/O failure.");
  }

  if ((input.bad()) != 0)
  {
    throw std::runtime_error("I/O operation error.");
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
void AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::readHeaderData()
{
  if (_isFileOpen)
  {
    std::string line;

    // Check type
    std::getline(_file, line);
    get_line_error_check(_file);

    if (line.compare("sptensor") == 0)
    {
      _usingOneIndex = true;
      _fileType = "sptensor";
    } else if (line.compare("sptensor indices-start-at-zero") == 0)
    {
      _usingOneIndex = false;
      _fileType = "sptensor";
    } else
    {
      throw std::runtime_error("Incorrect tensor type (should be 'sptensor') in ASCII file.");
    }

    // Read number of dimensions
    std::getline(_file, line);
    get_line_error_check(_file);
    _nDim = std::stoll(line);

    if (_nIndices == nullptr)
    {
      _nIndices = new SubIdx[_nDim];
    }
    else
    {
      throw std::runtime_error("_nIndices already allocated!");
    }

    // Read size of each dimension
    std::getline(_file, line);
    get_line_error_check(_file);
    std::istringstream line_ss(line);
    for (SubIdx iDim = 0; iDim < _nDim; ++iDim)
    {
      line_ss >> _nIndices[iDim];
      if (_nIndices[iDim] > _maxIndex)
      {
        _maxIndex = _nIndices[iDim];
      }
    }

    // Read number of non-zero elements
    std::getline(_file, line);
    get_line_error_check(_file);
    _nElement = std::stoll(line);
  } else
  {
    throw std::runtime_error("File needs to be open before reading header information.");
  }
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
bool AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::usingOneIndex() const
{
  return _usingOneIndex;
} 

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
std::string AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::fileType() const
{
  return _fileType;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::nDimension() const
{
  return _nDim;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx const *AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::nIndices() const
{
  return _nIndices;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
SubIdx AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::maxIndex() const
{
  return _maxIndex;
}

template<class SparseValue, class KruskalValue, class ElemIdx, class SubIdx>
ElemIdx AsciiIO<SparseValue, KruskalValue, ElemIdx, SubIdx>::nElement() const
{
  return _nElement;
}

// Explicit instantiation
template class AsciiIO<type::SubIdx, float, type::SubIdx, type::SubIdx>;
template class AsciiIO<type::SubIdx, float, type::ElemIdx, type::SubIdx>;
template class AsciiIO<type::SubIdx, float, type::ElemIdx, type::ElemIdx>;

template class AsciiIO<type::ElemIdx, float, type::SubIdx, type::SubIdx>;
template class AsciiIO<type::ElemIdx, float, type::ElemIdx, type::SubIdx>;
template class AsciiIO<type::ElemIdx, float, type::ElemIdx, type::ElemIdx>;

template class AsciiIO<type::SubIdx, double, type::SubIdx, type::SubIdx>;
template class AsciiIO<type::SubIdx, double, type::ElemIdx, type::SubIdx>;
template class AsciiIO<type::SubIdx, double, type::ElemIdx, type::ElemIdx>;

template class AsciiIO<type::ElemIdx, double, type::SubIdx, type::SubIdx>;
template class AsciiIO<type::ElemIdx, double, type::ElemIdx, type::SubIdx>;
template class AsciiIO<type::ElemIdx, double, type::ElemIdx, type::ElemIdx>;

} // sparten
