/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Numpy.h"
#include "checks.h"

// calculates variance over one axis in parallel without creating a copy of the data X
Vector<> var(cRef<Matrix<>> X, const size_t axis, const size_t ddof) {
    size_t N = 0;
    size_t D = 0;

    if (!(axis == 0 or axis == 1)) {
        std::stringstream msg;
        msg << "'axis' must be 0 or 1, but got " << axis;
        throw std::invalid_argument(msg.str());
    }
    if (axis == 0) {
        N = X.rows();
        D = X.cols();
    } else if (axis == 1) {
        // transpose the meaning of N and D
        N = X.cols();
        D = X.rows();
    }

    Vector<> mean(D);
    Vector<> var_(D);
    mean.fill(0);
    var_.fill(0);

    precision_t normalizer = 1. / (precision_t)N;
    precision_t normalizer_ddoff = 1. / (precision_t)(N - ddof);

#pragma omp parallel
    {
        Vector<> sum(D);
        Vector<> sum2(D);
        sum.fill(0.);
        sum2.fill(0.);

        if (axis == 0) {
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                sum += X.row(n);
                sum2 += X.row(n).cwiseProduct(X.row(n));
            }
        } else if (axis == 1) {
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                sum += X.col(n);
                sum2 += X.col(n).cwiseProduct(X.col(n));
            }
        }

#pragma omp critical
        {
            mean += sum;
            var_ += sum2;
        }
    }
    mean *= normalizer;
    var_ *= normalizer_ddoff;
    var_ -= mean.cwiseProduct(mean) * normalizer_ddoff / normalizer;

    return var_;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/numpy.h>
namespace stats {
void bind(py::module_& m) {
    py::module_ submodule_ = m.def_submodule("stats");
    submodule_.def("var", &var, "X"_a.noconvert(), "axis"_a = 0, "ddof"_a = 0);
}
}  // namespace stats
#endif
