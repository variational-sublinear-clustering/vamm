/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once
#include <Eigen/Dense>
#include <random>

#include "Numpy.h"
#include "checks.h"
#include "omp_helper.h"

// squared Euclidean norm
struct squaredNorm {
    precision_t operator()(cRef<Vector<>> x, cRef<Vector<>> y) const { return (x - y).squaredNorm(); }
};

namespace afkmc2 {
template <class Distance>
size_t afkmc2(cRef<Matrix<>> X, size_t C, size_t seed, size_t chain, Ref<Vector<size_t>> indices,
              const Distance& distance) {
    checkSize(indices, C);

    size_t num_eval = 0;  // number of distance evaluations
    size_t N = X.rows();
    precision_t sum_inv = 0;
    precision_t N_inv = 1. / (precision_t)N;

    Random<std::mt19937_64> rng(seed);
    std::uniform_int_distribution<size_t> rand_int(0, N - 1);

    Vector<> q(N);
    Vector<> d(C);
    precision_t prob_x = 0.;
    precision_t prob_y = 0.;
    size_t idx_x = 0;
    size_t idx_y = 0;

    // draw first center
    indices[0] = rand_int(rng());

// calculate proposal distribution q
#pragma omp parallel for
    for (size_t n = 0; n < N; n++) {
        q[n] = distance(X.row(n), X.row(indices[0]));
    }
    q = q.cwiseMax(0);  // to prevent numerical errors, distance to itself may not be exactly zero
    num_eval += N;
    sum_inv = 1. / q.sum();

#pragma omp parallel for
    for (size_t n = 0; n < N; n++) {
        q[n] = 0.5 * (q[n] * sum_inv + N_inv);
    }

    std::discrete_distribution<size_t> draw_from_q(q.begin(), q.end());
    std::uniform_real_distribution<precision_t> uniform(0.0, 1.0);

    // draw other centers
    for (size_t c = 1; c < C; c++) {
        // Markov chain of length 'chain'
        for (size_t j = 0; j < chain; j++) {
            // draw new candidate
            idx_y = draw_from_q(rng());

// compute distance to centers
#pragma omp parallel for
            for (size_t i = 0; i < c; i++) {
                d[i] = distance(X.row(idx_y), X.row(indices[i]));
            }
            num_eval += c;

            // compute probability for new candidate
            // max(d,0) is used to prevent numerical errors, see above
            prob_y = std::max(d.head(c).minCoeff() / q[idx_y], (precision_t)0.);
            if ((j == 0) or (prob_x == 0) or ((prob_y / prob_x) > uniform(rng()))) {
                // accept new candidate, first candidate will always be excepted (j == 0)
                idx_x = idx_y;
                prob_x = prob_y;
            }
        }
        // select data point as center
        indices[c] = idx_x;
    }
    return num_eval;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/numpy.h>

void bind(py::module_& m) {
    m.def(
        "afkmc2",
        [](cRef<Matrix<>> X, size_t C, size_t seed, size_t chain, Ref<Vector<size_t>> indices) -> size_t {
            squaredNorm distance;  // use squared norm as default
            return afkmc2(X, C, seed, chain, indices, distance);
        },
        "X"_a.noconvert(), "C"_a, "seed"_a, "chain"_a, "indices"_a);
}
#endif

}  // namespace afkmc2