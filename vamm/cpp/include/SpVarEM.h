/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/functional.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <iostream>
#include <list>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Numpy.h"
#include "checks.h"
#include "omp_helper.h"

#ifdef _OPENMP
#include <omp.h>
#endif

template <class key_t = size_t, class val_t = precision_t>
using map_t = boost::unordered::unordered_flat_map<key_t, val_t>;

template <class key_t = size_t>
using set_t = boost::unordered::unordered_flat_set<key_t>;

// datatype to store truncated variational distribution
using q_t = std::vector<std::pair<size_t, precision_t>>;

template <typename SpVarEMAlgorithm>
class SpVarEM {
   public:
    static constexpr precision_t infty_ = std::numeric_limits<precision_t>::infinity();

    size_t N;
    size_t C;

    std::vector<q_t> qs;
    std::vector<std::vector<std::vector<std::pair<std::size_t, q_t::iterator>>>> partition_E;
    std::vector<std::vector<q_t>> partition_M;
    bool M_step_hard;

    template <class Model>
    size_t E_step_ljs(cRef<Matrix<>> X, const Model &model);

    precision_t E_step_normalize(const precision_t &);

   public:
    SpVarEM(size_t N_, size_t C_, bool hard_);
    // ---

    template <class Model>
    precision_t EM_step(cRef<Matrix<>> X, Model &model, const bool fit, const bool update_var_params,
                        const precision_t &beta);

    template <class Model>
    void M_step(cRef<Matrix<>> X, Model &model);

    // ---

    // protected:
    void get_partition_E(void);
    void get_partition_M(void);

    std::unordered_map<size_t, precision_t> q_map(size_t n) const;

    void q_in(size_t n, const std::unordered_map<size_t, precision_t> &map);

    auto approx_map(size_t n) const;

    auto q_to_sparse_matrix(void) const;
    void q_from_sparse_matrix(const SparseMatrix<> sp_mat);

    auto q_shrinked_to_sparse_matrix(cRef<Vector<bool>> Mask) const;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind_q_sparse_matrix_interface(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_);

    template <typename T, typename... Model>
    static void bind_EM_step(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_);

    template <typename T, typename... Model>
    static void bind_E_step(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_);

    template <typename T, typename... Model>
    static void bind_M_step(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_);

#endif
};

template <typename SpVarEMAlgorithm>
SpVarEM<SpVarEMAlgorithm>::SpVarEM(size_t N_, size_t C_, bool hard_) :
    N(N_), C(C_), qs(N_), partition_E(get_max_threads()), partition_M(get_max_threads()), M_step_hard(hard_) {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();
        partition_E[thread_num] = std::vector<std::vector<std::pair<std::size_t, q_t::iterator>>>(C);
        partition_M[thread_num] = std::vector<q_t>(C);
    }
}

// helper function to get soft partition via the qs
// for each c, the resulting list contains pairs {n, q_nc(c)} for all data points n with c in there qs

template <typename SpVarEMAlgorithm>
void SpVarEM<SpVarEMAlgorithm>::get_partition_E(void) {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();
        std::fill(partition_E[thread_num].begin(), partition_E[thread_num].end(),
                  std::vector<std::pair<size_t, q_t::iterator>>(0));
#pragma omp for
        for (size_t n = 0; n < N; n++) {
            for (auto q = qs[n].begin(); q != qs[n].end(); q++) {
                partition_E[thread_num][q->first].emplace_back(n, q);
            }
        }
        for (size_t c = 0; c < C; c++) {
            partition_E[thread_num][c].shrink_to_fit();
        }
    }
}

template <typename SpVarEMAlgorithm>
void SpVarEM<SpVarEMAlgorithm>::get_partition_M(void) {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();
        if (M_step_hard) {
            size_t c = 0;
            std::fill(partition_M[thread_num].begin(), partition_M[thread_num].end(),
                      std::vector<std::pair<std::size_t, precision_t>>(0));
#pragma omp for
            for (size_t n = 0; n < N; n++) {
                c = std::max_element(qs[n].begin(), qs[n].end(), [](auto &lhs, auto &rhs) -> bool {
                        return lhs.second < rhs.second;
                    })->first;
                partition_M[thread_num][c].emplace_back(n, 1.);
            }
        } else {
            std::fill(partition_M[thread_num].begin(), partition_M[thread_num].end(),
                      std::vector<std::pair<std::size_t, precision_t>>(0));
#pragma omp for
            for (size_t n = 0; n < qs.size(); n++) {
                for (const auto &[c, q_nc] : qs[n]) {
                    if (q_nc > 0.) {  // ignore if q rounds to zero
                        partition_M[thread_num][c].emplace_back(n, q_nc);
                    }
                }
            }
        }
        for (size_t c = 0; c < C; c++) {
            partition_M[thread_num][c].shrink_to_fit();
        }
    }
}

template <typename SpVarEMAlgorithm>
template <class Model>
size_t SpVarEM<SpVarEMAlgorithm>::E_step_ljs(cRef<Matrix<>> X, const Model &model) {
    if constexpr (Model::loop_order_n) {
#pragma omp parallel
        {
            model.E_step_allocate([&](auto &...e_step_args) -> auto {
#pragma omp for  // schedule(dynamic,1)
                for (size_t n = 0; n < N; n++) {
                    for (auto &[c, log_prob] : qs[n]) {
                        model.E_step_log_joint(X.row(n), c, log_prob, e_step_args...);
                    }
                }
            });
        }
    } else {
        get_partition_E();
#pragma omp parallel
        {
            model.E_step_allocate([&](auto &...e_step_args) -> auto {
#pragma omp for schedule(dynamic, 1)
                for (std::size_t c = 0; c < C; c++) {
                    for (const auto &partition_E_ : partition_E) {
                        if (!partition_E_[c].empty()) {
                            for (auto &IT : partition_E_[c]) {
                                model.E_step_log_joint(X.row(IT.first), c, IT.second->second, e_step_args...);
                            }
                        }
                    }
                }
            });
        }
    }

    size_t total_ljs = 0;
    for (size_t n = 0; n < N; n++) {
        total_ljs += qs[n].size();
    }

    return total_ljs;
}

template <typename SpVarEMAlgorithm>
precision_t SpVarEM<SpVarEMAlgorithm>::E_step_normalize(const precision_t &beta) {
    precision_t objective = 0;
#pragma omp parallel
    {
        precision_t lim = 0;
        precision_t sum = 0;
        precision_t objective_thread = 0;
#pragma omp for
        for (std::size_t n = 0; n < N; n++) {
            lim = std::max_element(qs[n].begin(), qs[n].end(), [](auto &lhs, auto &rhs) -> bool {
                      return lhs.second < rhs.second;
                  })->second;
            if (!std::isfinite(lim)) {
                throw std::runtime_error("NaN in logjoints!");
            }
            lim *= beta;
            sum = 0;
            for (auto &[c, q_nc] : qs[n]) {
                q_nc = std::exp(beta * q_nc - lim);
                sum += q_nc;
            }
            for (auto &[c, q_nc] : qs[n]) {
                q_nc /= sum;
            }
            if (M_step_hard) {
                objective_thread += lim;
            } else {
                objective_thread += std::log(sum) + lim;
            }
        }
#pragma omp atomic
        objective += objective_thread;
    }
    return objective;
}

template <typename SpVarEMAlgorithm>
template <class Model>
void SpVarEM<SpVarEMAlgorithm>::M_step(cRef<Matrix<>> X, Model &model) {
    get_partition_M();

    model.M_step_update(X, partition_M);
    model.M_step_finalize(N);
}

template <typename SpVarEMAlgorithm>
template <class Model>
precision_t SpVarEM<SpVarEMAlgorithm>::EM_step(cRef<Matrix<>> X, Model &model, const bool fit,
                                               const bool update_var_params, const precision_t &beta) {
    precision_t objective = static_cast<SpVarEMAlgorithm *>(this)->E_step(X, model, update_var_params, beta);
    if (fit) {
        static_cast<SpVarEMAlgorithm *>(this)->M_step(X, model);
    }
    return objective;
}

template <typename SpVarEMAlgorithm>
std::unordered_map<size_t, precision_t> SpVarEM<SpVarEMAlgorithm>::q_map(size_t n) const {
    checkIndex(n, qs.size());
    std::unordered_map<size_t, precision_t> map;
    for (const auto &el : qs[n]) {
        map.insert(el);
    }
    return map;
}

template <typename SpVarEMAlgorithm>
void SpVarEM<SpVarEMAlgorithm>::q_in(size_t n, const std::unordered_map<size_t, precision_t> &map) {
    checkIndex(n, N);
    if (map.size() != qs[n].size()) {
        throw std::invalid_argument("Invalid map.size()!\n");
    }
    for (const auto &it : map) {
        checkIndex(it.first, C);
    }
    qs[n].clear();
    qs[n].reserve(map.size());
    for (const auto &it : map) {
        qs[n].push_back(it);
    }
}

template <typename SpVarEMAlgorithm>
auto SpVarEM<SpVarEMAlgorithm>::approx_map(size_t n) const {
    auto it = std::max_element(qs[n].begin(), qs[n].end(),
                               [](auto &lhs, auto &rhs) -> bool { return lhs.second < rhs.second; });
    return std::make_pair(it->first, it->second);
}

template <typename SpVarEMAlgorithm>
auto SpVarEM<SpVarEMAlgorithm>::q_to_sparse_matrix(void) const {
    std::vector<Eigen::Triplet<precision_t>>
        coeff;  // each Triplet in this vector is a non-zero entry with (row index, column index, value)
    SparseMatrix<> sp_mat(N, C);
    coeff.reserve(qs[0].size() * qs.size());
    for (size_t n = 0; n < N; n++) {
        for (const auto &it : qs[n]) {
            coeff.emplace_back(n, it.first, it.second);
        }
    }
    sp_mat.setFromTriplets(coeff.begin(), coeff.end());
    return sp_mat;
}

template <typename SpVarEMAlgorithm>
auto SpVarEM<SpVarEMAlgorithm>::q_shrinked_to_sparse_matrix(cRef<Vector<bool>> Mask) const {
    checkSize(Mask, C);
    size_t C_active = Mask.count();
    int i = 0;
    Vector<int> idx(C);
    Vector<size_t> rev_idx(C_active);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            idx[c] = i;
            rev_idx[i] = c;
            i++;
        } else {
            idx[c] = -1;
        }
    }
    std::vector<Eigen::Triplet<precision_t>>
        coeff;  // each Triplet in this vector is a non-zero entry with (row index, column index, value)
    SparseMatrix<> sp_mat(N, C_active);
    coeff.reserve(qs[0].size() * qs.size());
    for (size_t n = 0; n < N; n++) {
        for (const auto &it : qs[n]) {
            coeff.emplace_back(n, idx[it.first], it.second);
        }
    }

    sp_mat.setFromTriplets(coeff.begin(), coeff.end());
    return std::make_tuple(sp_mat, rev_idx);
}

template <typename SpVarEMAlgorithm>
void SpVarEM<SpVarEMAlgorithm>::q_from_sparse_matrix(const SparseMatrix<> sp_mat) {
    checkSize(sp_mat, N, C);
    if ((size_t)sp_mat.nonZeros() != qs[0].size() * qs.size()) {
        std::stringstream msg;
        msg << "expected input having " << qs[0].size() * qs.size() << " non zero coefficients, but got "
            << sp_mat.nonZeros();
        throw std::invalid_argument(msg.str());
    }
    for (size_t n = 0; n < N; n++) {
        if ((size_t)sp_mat.row(n).nonZeros() != qs[n].size()) {
            std::stringstream msg;
            msg << "expected row " << n << " having " << qs[n].size() << " non zero coefficients, but got "
                << sp_mat.row(n).nonZeros();
            throw std::invalid_argument(msg.str());
        }
        qs[n].clear();
        for (SparseMatrix<>::InnerIterator it(sp_mat, n); it; ++it) {
            qs[n].emplace_back(it.col(), it.value());
        }
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

template <typename SpVarEMAlgorithm>
void SpVarEM<SpVarEMAlgorithm>::bind_q_sparse_matrix_interface(
    py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_) {
    SpVarEMAlgorithm_class_.def_property("q", &SpVarEMAlgorithm::q_to_sparse_matrix,
                                         &SpVarEMAlgorithm::q_from_sparse_matrix,
                                         "The variational distributions `q` as a scipy sparse csr matrix. "
                                         "This requires copies in both directions.");
    // gets converted to or from a scipy sparse csr matrix, creates a copy in both directions

    SpVarEMAlgorithm_class_.def("q_shrinked", &SpVarEMAlgorithm::q_shrinked_to_sparse_matrix,
                                "mask"_a.noconvert(),
                                R"(
                Get the variational distributions `q` as a scipy sparse csr matrix.
                The mask is used to disregard components.

                Parameters
                ----------
                mask : np.array[bool]
                    Masking of the components. 

                Returns
                -------
                q: scipy sparse csr matrix
                    The variational distributions.
                idx: np.array[unit64]
                    to map from column index in q to component index c.
                )");
}

template <typename SpVarEMAlgorithm>
template <typename T, typename... Model>
void SpVarEM<SpVarEMAlgorithm>::bind_E_step(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_) {
    (SpVarEMAlgorithm_class_.def(
         "E_step",
         [](SpVarEMAlgorithm &self, cRef<Matrix<>> X, Model &model, bool update_var_params,
            precision_t beta) -> auto { return self.E_step(X, model, update_var_params, beta); },
         "X"_a.noconvert(), "model"_a, "update_var_params"_a = true, "beta"_a = 1.0, R"(
                Perform one Expectation step of the variational EM algorithm.

                Parameters
                ----------
                X : npt.ndarray
                    Input data.
                model : Model
                    The mixture model to train.
                update_var_params : bool
                    Update variational parameters? Defaults to True.
                beta : np.float64
                    Inverse temperature. Defaults to 1.

                Returns
                -------
                float
                    The training objective.
                )"),
     ...);
}

template <typename SpVarEMAlgorithm>
template <typename T, typename... Model>
void SpVarEM<SpVarEMAlgorithm>::bind_M_step(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_) {
    (SpVarEMAlgorithm_class_.def(
         "M_step",
         [](SpVarEMAlgorithm &self, cRef<Matrix<>> X, Model &model) -> auto { return self.M_step(X, model); },
         "X"_a.noconvert(), "model"_a, R"(
                Perform one Maximization step of the variational EM algorithm.

                Parameters
                ----------
                X : npt.ndarray
                    Input data.
                model : Model
                    The mixture model to train.

                Returns
                -------
                None
                )"),
     ...);
}

template <typename SpVarEMAlgorithm>
template <typename T, typename... Model>
void SpVarEM<SpVarEMAlgorithm>::bind_EM_step(py::class_<SpVarEMAlgorithm> &SpVarEMAlgorithm_class_) {
    (SpVarEMAlgorithm_class_.def(
         "EM_step",
         [](SpVarEMAlgorithm &self, cRef<Matrix<>> X, Model &model, bool fit, bool update_var_params,
            precision_t beta) -> auto { return self.EM_step(X, model, fit, update_var_params, beta); },
         "X"_a.noconvert(), "model"_a, "fit"_a = true, "update_var_params"_a = true, "beta"_a = 1.0, R"(
                Perform one iteration of the variational EM algorithm.

                Parameters
                ----------
                X : npt.ndarray
                    Input data.
                model : Model
                    The mixture model to train.
                fit : bool
                    Whether to fit the model to the input data (M-step). Defaults to True.
                update_var_params : bool
                    Update variational parameters? Defaults to True.
                beta : np.float64
                    Inverse temperature. Defaults to 1.

                Returns
                -------
                float
                    The training objective.
                )"),
     ...);
}

#endif