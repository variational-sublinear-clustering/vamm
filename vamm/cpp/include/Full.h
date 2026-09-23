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
    size_t first_active_c;

    const precision_t reg_covar;  // regularization added to the diagonal of the covariance
    const precision_t pi_factor;

    /* Model parameters */
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

    Matrix<>& get_Covar();

    void set_Covar(cRef<Matrix<>>);

    void auxiliary();

    void auxiliary_(const size_t c);

    void relocate_discarded_components(std::vector<size_t>& from, std::vector<size_t>& to,
                                       Random<std::mt19937_64>& rng);

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const;

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob, Vector<>& T0) const;

    template <class Lmbd>
    void M_step_allocate(const Lmbd& lmbd) const;
    void M_step_reset(size_t c);
    void M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c);
    void M_step_update(size_t c);

    void M_step_finalize(size_t N);

    Matrix<> generate_data(size_t, int, bool, int);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(py::module_& m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

Full::Full(size_t C_, size_t D_, bool flat_prior_ = false, bool shared_ = false,
           precision_t reg_covar_ = 1e-6) :
    Mixture(C_, D_, flat_prior_),
    first_active_c(0),
    reg_covar(reg_covar_),
    pi_factor(D_ * std::log(2.0 * M_PI)),
    Covar(C_, D_ * D_),  // will be filled below
    Covar_shared(Vector<>::Zero(D_ * D_)),
    Eye(Matrix<>::Identity(D_, D_)),
    // Prec(C_, Matrix<>::Zero(D_, D_)),
    Prec_chol(C_, Matrix<>::Zero(D_, D_)),
    det_log(C_),
    Cholesky(get_max_threads()),
    shared(shared_) {
#pragma omp parallel
    {
        Cholesky[get_thread_num()] = Eigen::LLT<Matrix<>, Eigen::Upper>(D);
#pragma omp for
        for (size_t c = 0; c < C; c++) {
            Covar.row(c).reshaped<Eigen::RowMajor>(D, D) = Eye;
        }
    }
}

Matrix<>& Full::get_Covar() { return Covar; }

void Full::set_Covar(cRef<Matrix<>> Covar_) {
    checkSize(Covar_, C, D * D);
    Covar = Covar_;
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
    } else {
        const size_t thread_num = get_thread_num();

        // Cholesky decomposition such that U^T U = Covar
        Cholesky[thread_num].compute(Covar.row(c).reshaped<Eigen::RowMajor>(D, D));
        if (Cholesky[thread_num].info() != 0) {
            discard(c, "Cholesky decomposition failed");
            return;
        }
        Prec_chol[c] = Cholesky[thread_num].matrixU().solve(Eye);
        det_log[c] =
            -2.0 * Prec_chol[c].diagonal().array().log().sum();  // log(det(Cov)) = -2 log(det(Prec_chol))
        // sum of logs is numerically much more stable than log of product.

        // Prec[c] = Cholesky[thread_num].solve(Eye);  // using Cholesky decomposition for inverse
        // Prec[c].noalias() = Prec_chol[c] * Prec_chol[c].transpose();
    }
}

void Full::relocate_discarded_components(std::vector<size_t>& from, std::vector<size_t>& to,
                                         Random<std::mt19937_64>& rng) {
#pragma omp parallel
    {
        // const size_t thread_num = get_thread_num();
        const size_t size = from.size();
        size_t c1;
        size_t c2;
        std::normal_distribution<precision_t> std_normal{0.0, 1.0};
        Vector<> noise(D);
#pragma omp for
        for (size_t i = 0; i < size; i++) {
            c1 = from[i];
            c2 = to[i];
            noise = noise.unaryExpr([&](precision_t) { return std_normal(rng()); });
            P[c1] = P[c2];

            // // Cholesky decomposition such that U^T U = Covar
            // Cholesky[thread_num].compute(Covar.row(c2).reshaped<Eigen::RowMajor>(D, D));
            // if (Cholesky[thread_num].info() != 0) {
            //     discard(c2, "Cholesky decomposition failed");
            //     continue;
            // }
            // M.row(c1).noalias() = 0.5 * noise * Cholesky[thread_num].matrixU();
            // M.row(c1) += M.row(c2);

            // using only the diagonal part of the covariance as noise
            M.row(c1) =
                M.row(c2) +
                0.1 * (Covar.row(c2).reshaped<Eigen::RowMajor>(D, D).diagonal().array().sqrt() * noise.array())
                          .matrix();

            Covar.row(c1) = Covar.row(c2);
            if (!checkFinite(M.row(c1))) {
                discard(c1, "relocated mean not finite");
            }
        }
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

template <class Lmbd>
void Full::M_step_allocate(const Lmbd& lmbd) const {
    lmbd();
}

void Full::M_step_reset(size_t c) {
    P[c] = 0.;
    M.row(c).fill(0);
    Covar.row(c).fill(0.);
}

void Full::M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c) {
    M.row(c) += q_nc * x;
    Covar.row(c).reshaped<Eigen::RowMajor>(D, D).noalias() += q_nc * x.transpose() * x;
    P[c] += q_nc;
}

void Full::M_step_update(size_t c) {
    if (P[c] <= 0) {
        discard(c, "prior not positive");
        return;
    }
    M.row(c) /= P[c];

    if (shared) {
        Covar.row(c).reshaped<Eigen::RowMajor>(D, D).noalias() -= P[c] * M.row(c).transpose() * M.row(c);
    } else {
        Covar.row(c) /= P[c];
        Covar.row(c).reshaped<Eigen::RowMajor>(D, D).noalias() -= M.row(c).transpose() * M.row(c);
        Covar.row(c).reshaped<Eigen::RowMajor>(D, D).diagonal().array() += reg_covar;
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

Matrix<> Full::generate_data(size_t N, int c = -1, bool add_noise = true, int seed = -1) {
    if (N <= 0) throw std::invalid_argument("N must be > 0");

    if (seed < 0) seed = std::random_device{}();
    Random<std::mt19937> rng(seed);

    Vector<size_t> hidden_states(N);

    if (c < 0) {
        std::discrete_distribution<size_t> categorical(P.begin(), P.end());
        hidden_states = hidden_states.unaryExpr([&](precision_t) { return categorical(rng()); });
    } else {
        hidden_states.setConstant(c);
    }

    Matrix<> X(N, D);

#pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        X.row(n) = M.row(hidden_states[n]);
    }

    if (!add_noise) return X;

    std::normal_distribution<precision_t> std_normal(0.0, 1.0);
    Matrix<> noise(Matrix<>::NullaryExpr(N, D, [&]() { return std_normal(rng()); }));

    if (shared) {
        first_active_c = first_active();
        Eigen::LLT<Matrix<>> llt(Covar.row(first_active_c).reshaped<Eigen::RowMajor>(D, D));

        if (llt.info() != Eigen::Success) {
            std::cerr << "WARNING: Cholesky decomposition of covariance failed! "
                         "Generate data without noise."
                      << std::endl;

            return X;
        }
        X.noalias() += noise * llt.matrixU();
    } else {
        std::vector<Matrix<>> U(C);

        for (size_t c = 0; c < C; ++c) {
            if (!Mask[c]) continue;

            Eigen::LLT<Matrix<>> llt(Covar.row(c).reshaped<Eigen::RowMajor>(D, D));
            if (llt.info() != Eigen::Success) {
                std::cerr << "WARNING: Cholesky decomposition of covariance failed! "
                             "Generate data without noise."
                          << std::endl;

                return X;
            }
            U[c] = llt.matrixU();
        }

#pragma omp parallel for
        for (size_t n = 0; n < N; ++n) {
            X.row(n).noalias() += noise.row(n) * U[hidden_states(n)].triangularView<Eigen::Upper>();
        }
    }
    return X;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

void Full::bind(py::module_& m) {
    py::class_<Full> Full_class_(m, "Full", py::module_local());

    Full_class_.def(py::init<size_t, size_t, bool, bool, precision_t>(), "C"_a, "D"_a, "flat_prior"_a = false,
                    "shared"_a = false, "reg_covar"_a = 1e-6);
    Full_class_.def_property("variance", &Full::get_Covar, &Full::set_Covar);
    Full_class_.def_readwrite("shared", &Full::shared);
    Full_class_.def("generate_data", &Full::generate_data, "N"_a, "c"_a = -1, "add_noise"_a = true,
                    "seed"_a = -1);

    bind_base<precision_t>(Full_class_);
}

#endif
