/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include "Mixture.h"
#include "Numpy.h"
#include "omp_helper.h"

class Full : public Mixture<Full> {
   public:
    size_t C;
    size_t D;

    size_t first_active_c;

    const precision_t reg_covar;  // regularization added to the diagonal of the covariance
    const precision_t pi_factor;

    /* Model parameters */
    Matrix<> M;
    Matrix<> Covar;
    Vector<> Covar_shared;

    /* Utility */
    Matrix<> Eye;  // identity matrix

    std::vector<Matrix<>, Eigen::aligned_allocator<Matrix<>>> Prec_chol;

    Vector<> det_log;

    std::vector<Eigen::LLT<Matrix<>, Eigen::Upper>> Cholesky;  // to compute Cholesky decomposition
                                                               // Performance: use Upper for row-major matrices

    static constexpr bool loop_order_n = false;  // Shadows Mixture<Full>::loop_order_n
    bool shared;

    Full(size_t C_, size_t D_, bool flat_prior_, bool shared_, precision_t reg_covar_);

    size_t get_C(void) const;

    size_t get_D(void) const;

    Matrix<>& get_M();

    Matrix<>& get_Covar();

    void set_M(cRef<Matrix<>>);

    void set_Covar(cRef<Matrix<>>);

    void auxiliary();

    void auxiliary_(const size_t c);

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const;

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob, Vector<>& T0) const;

    void M_step_update(cRef<Matrix<>> X, const std::vector<std::vector<q_t>>& partition);

    void M_step_finalize(size_t N);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(py::module_& m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

Full::Full(size_t C_, size_t D_, bool flat_prior_ = false, bool shared_ = false,
           precision_t reg_covar_ = 1e-6) :
    Mixture(C_, D_, flat_prior_),
    C(C_),
    D(D_),
    first_active_c(0),
    reg_covar(reg_covar_),
    pi_factor(D_ * std::log(2.0 * M_PI)),
    M(Matrix<>::Zero(C_, D_)),
    Covar(Matrix<>::Ones(C_, D_ * D_)),
    Covar_shared(Vector<>::Zero(D_ * D_)),
    Eye(Matrix<>::Identity(D_, D_)),
    // Prec(C_, Matrix<>::Zero(D_, D_)),
    Prec_chol(C_, Matrix<>::Zero(D_, D_)),
    det_log(C_),
    Cholesky(get_max_threads()),
    shared(shared_) {
    P_adjust();
#pragma omp parallel
    { Cholesky[get_thread_num()] = Eigen::LLT<Matrix<>, Eigen::Upper>(D); }
}

size_t Full::get_C(void) const { return C; }

size_t Full::get_D(void) const { return D; }

Matrix<>& Full::get_M() { return M; }

Matrix<>& Full::get_Covar() { return Covar; }

void Full::set_Covar(cRef<Matrix<>> Covar_) {
    checkSize(Covar_, C, D * D);
    Covar = Covar_;
}

void Full::set_M(cRef<Matrix<>> M_) {
    checkSize(M_, C, D);
    M = M_;
}

void Full::auxiliary() {
    if (shared) {
        const size_t thread_num = get_thread_num();
        first_active_c = first_active();

        // Cholesky decomposition such that U^T U = Covar
        Cholesky[thread_num].compute(Covar.row(first_active_c).reshaped<Eigen::RowMajor>(D, D));
        if (Cholesky[thread_num].info() != 0) {
            throw std::runtime_error("Cholesky decomposition failed");
        }
        Prec_chol[first_active_c] = Cholesky[thread_num].matrixU().solve(Eye);
        det_log[first_active_c] = -2.0 * Prec_chol[first_active_c].diagonal().array().log().sum();
    }

    #pragma omp parallel for
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            auxiliary_(c);
        }
    }
}

void Full::auxiliary_(const size_t c) {
    if (shared) {
        Prec_chol[c] = Prec_chol[first_active_c];
        det_log[c] = det_log[first_active_c];
    }
    else {
        const size_t thread_num = get_thread_num();

        // Cholesky decomposition such that U^T U = Covar
        Cholesky[thread_num].compute(Covar.row(c).reshaped<Eigen::RowMajor>(D, D));
        if (Cholesky[thread_num].info() != 0) {
            discard(c, "Cholesky decomposition failed");
            return;
        }
        Prec_chol[c] = Cholesky[thread_num].matrixU().solve(Eye);
        det_log[c] = -2.0 * Prec_chol[c].diagonal().array().log().sum();  // log(det(Cov)) = -2 log(det(Prec_chol))
        // sum of logs is numerically much more stable than log of product.

        // Prec[c] = Cholesky[thread_num].solve(Eye);  // using Cholesky decomposition for inverse
        // Prec[c].noalias() = Prec_chol[c] * Prec_chol[c].transpose();
    }
}

template <class Lmbd>
void Full::E_step_allocate(const Lmbd& lmbd) const {
    Vector<> T0(D);

    lmbd(T0);
}

void Full::E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob, Vector<>& T0) const {
    T0.noalias() = (x - M.row(c)) * Prec_chol[c].triangularView<Eigen::Upper>();

    log_prob = T0.dot(T0);
    log_prob += det_log[c] + pi_factor;
    log_prob *= -0.5;
    log_prob += P_log[c];
}

void Full::M_step_update(cRef<Matrix<>> X, const std::vector<std::vector<q_t>>& partition) {
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                P[c] = 0.;
                M.row(c).fill(0);
                Covar.row(c).fill(0.);
                for (const auto& partition_ : partition) {
                    for (const auto& [n, q_nc] : partition_[c]) {
                        M.row(c) += q_nc * X.row(n);
                        Covar.row(c).reshaped<Eigen::RowMajor>(D, D).noalias() +=
                            q_nc * X.row(n).transpose() * X.row(n);
                        P[c] += q_nc;
                    }
                }
                if (P[c] <= 0) {
                    discard(c, "prior not positive");
                    continue;
                }
                M.row(c) /= P[c];

                if (shared) {
                    Covar.row(c).reshaped<Eigen::RowMajor>(D, D).noalias() -=
                            P[c] * M.row(c).transpose() * M.row(c);
                }
                else {
                    Covar.row(c) /= P[c];
                    Covar.row(c).reshaped<Eigen::RowMajor>(D, D).noalias() -= M.row(c).transpose() * M.row(c);
                    Covar.row(c).reshaped<Eigen::RowMajor>(D, D).diagonal().array() += reg_covar;
                }
            }
        }
    }
}

void Full::M_step_finalize(size_t N) {
    if (shared) {
        Covar_shared.fill(0.);
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                Covar_shared += Covar.row(c);
            }
        }
        Covar_shared /= N;
        Covar_shared.reshaped<Eigen::RowMajor>(D, D).diagonal().array() += reg_covar;
#pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                Covar.row(c) = Covar_shared;
            }
        }
    }
    P_adjust();
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

void Full::bind(py::module_& m) {
    py::class_<Full> Full_class_(m, "Full", py::module_local(), R"(
    Mixture of Factor Analyzer.

    Parameters
    ----------
    C : int
        Number of components.
    D : int
        Dimensionality of the data.
    flat_prior : bool, optional
        Whether to use a flat prior for the mixture components. Defaults to False.
    shared : bool, optional
        Whether the covariance matrices are shared among components. Defaults to False.
    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 1e-6.
    )");

    Full_class_.def(py::init<size_t, size_t, bool, bool, precision_t>(), "C"_a, "D"_a, "flat_prior"_a = false,
                    "shared"_a = false, "reg_covar"_a = 1e-6);

    Full_class_.def_property_readonly("C", &Full::get_C, "The number of components.");
    Full_class_.def_property_readonly("D", &Full::get_D, "The dimensionality of the data");

    Full_class_.def_property("means", &Full::get_M, &Full::set_M,
                             "The mean values of all mixture components.");
    Full_class_.def_property("variance", &Full::get_Covar, &Full::set_Covar,
                             "The covariance matrices of all mixture components.");

    Full_class_.def_readwrite("shared", &Full::shared,
                              "Whether the covariance matrices are shared among components.");

    bind_base<precision_t>(Full_class_);
}

#endif
