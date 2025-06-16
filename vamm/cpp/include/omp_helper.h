/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

// helper functions for OpenMP
#pragma once

#include <random>

#include "Numpy.h"

#ifdef _OPENMP
#include <omp.h>

#include <vector>
#endif

// TODO: rename to get_num_threads()?
size_t get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

size_t get_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

#ifdef _OPENMP
void set_num_threads(const long num_threads) { omp_set_num_threads(num_threads); }
#else
void set_num_threads(const long) { return; }
#endif

template <class RandomNumberEngine>
class Random {
    size_t seed;
    int num_threads;
#ifdef _OPENMP
    std::vector<RandomNumberEngine> engine;
#else
    RandomNumberEngine engine;
#endif

   public:
    Random(const size_t seed_) : seed(seed_) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
        engine.resize(num_threads);
#pragma omp parallel
        {
            size_t thread_num = omp_get_thread_num();
            engine[thread_num] = RandomNumberEngine(seed + thread_num);
        }
#else
        engine = RandomNumberEngine(seed);
#endif
    };

    RandomNumberEngine& operator()() {
#ifdef _OPENMP
        return engine[omp_get_thread_num()];
#else
        return engine;
#endif
    };
};

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

namespace omp {
void bind(py::module_& m) {
    py::module_ submodule_ = m.def_submodule("omp");

    submodule_.def("get_max_threads", &get_max_threads);
    submodule_.def("set_num_threads", &set_num_threads, "num_threads"_a);
}
}  // namespace omp
#endif
