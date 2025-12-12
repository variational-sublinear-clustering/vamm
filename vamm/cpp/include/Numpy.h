/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

// define the macro CLUSTERING_PRECISION to choose between double and float (default is double)
// use
//   -DCLUSTERING_PRECISION=double
// or
//   -DCLUSTERING_PRECISION=float
#ifndef CLUSTERING_PRECISION
#define CLUSTERING_PRECISION double
#endif

typedef CLUSTERING_PRECISION precision_t;

#include <Eigen/Dense>
#include <Eigen/Sparse>

// use RowMajor Matrix to have same storage order as in numpy

template <typename T = precision_t>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = precision_t>
using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = precision_t>
using ColMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T = precision_t>
using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename T = precision_t>
using SparseVector = Eigen::SparseVector<T, Eigen::RowMajor>;

template <typename T = precision_t>
using SparseMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;

// reference to a matrix or vector
template <typename T>
using Ref = Eigen::Ref<T>;

// constant reference to a matrix or vector
template <typename T>
using cRef = const Eigen::Ref<const T>;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/eigen.h>  // the magic translation between numpy and eigen happens here
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::
    literals;  // for "arg"_a, see:
               // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html?highlight=_a#keyword-arguments
#endif
//--------------------------------------------------------------------------------------------------------------------//