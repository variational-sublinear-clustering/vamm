/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/Dense>
#include <stdexcept>

#include "Mixture.h"

class Diagonal : public Mixture<Diagonal> {
   public:
    size_t C;
    size_t D;
    std::string var_type;

    bool iso;
    bool sv;

    Matrix<> M;
    Matrix<> S;
    Matrix<> S_inv;
    Vector<> S_shared;
    precision_t S_iso;
    precision_t reg_covar;
    Vector<> D_log;
    const precision_t pi_factor;

    void auxiliary_(const size_t c);

    Diagonal(size_t C_, size_t D_, bool flat_prior_, const std::string& var_type_, bool shared_, precision_t reg_covar_);

    size_t get_C(void) const;

    size_t get_D(void) const;

    Matrix<>& get_M(void);

    Matrix<>& get_S(void);

    void set_M(cRef<Matrix<>> M_);

    void set_S(cRef<Matrix<>> S_);

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const;

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob) const;

    void M_step_update(cRef<Matrix<>> X, const std::vector<std::vector<q_t>>& partition);

    void M_step_finalize(size_t N);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(pybind11::module_& m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

Diagonal::Diagonal(size_t C_, size_t D_, bool flat_prior_ = false, const std::string& var_type_ = "diagonal", bool shared_ = false,
                   precision_t reg_covar_ = 1e-3) :
    Mixture(C_, D_, flat_prior_),
    C(C_),
    D(D_),
    M(C_, D_),
    S(C_, D_),
    S_inv(C_, D_),
    S_shared(D),
    S_iso(0.),
    reg_covar(reg_covar_),
    D_log(C_),
    pi_factor(D_ * std::log(2.0 * M_PI)) {
    iso = (var_type_ == "isotropic");
    sv = shared_;
    if (!iso and var_type_ != "diagonal") {
        throw std::invalid_argument(
            "'covariance_type' should be one of 'isotropic' or 'diagonal', but got '" +
            var_type_ + "'");
    }

    M.fill(0);
    S.fill(1);
    P_adjust();
}

size_t Diagonal::get_C(void) const { return C; }

size_t Diagonal::get_D(void) const { return D; }

Matrix<>& Diagonal::get_M(void) { return M; }

Matrix<>& Diagonal::get_S(void) { return S; }

void Diagonal::set_M(cRef<Matrix<>> M_) {
    checkSize(M_, C, D);
    M = M_;  // TODO: avoid copy here?
}

void Diagonal::set_S(cRef<Matrix<>> S_) {
    checkSize(S_, C, D);
    checkLow(S_, 0., false);  // TODO: or S >= reg_covar?
    S = S_;                   // TODO: avoid copy here?
}

void Diagonal::auxiliary_(const size_t c) {
    S_inv.row(c) = S.row(c).cwiseInverse();
    if (!checkFinite(S_inv.row(c))) {
        discard(c, "zero variance");
        return;
    }
    D_log[c] = S.row(c).array().log().sum();
    if (!std::isfinite(D_log[c])) {
        discard(c, "log of determinant not finite");
    }
}

template <class Lmbd>
void Diagonal::E_step_allocate(const Lmbd& lmbd) const {
    lmbd();
}

void Diagonal::E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob) const {
    log_prob = S_inv.row(c).dot((x - M.row(c)).array().square().matrix());
    log_prob += D_log[c] + pi_factor;
    log_prob *= -0.5;
    log_prob += P_log[c];
}

void Diagonal::M_step_update(cRef<Matrix<>> X, const std::vector<std::vector<q_t>>& partition) {
/* */
#pragma omp parallel
    {
        Vector<> T_sq(D);
#pragma omp for schedule(dynamic, 1)
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                T_sq.fill(0);
                M.row(c).fill(0);
                P[c] = 0.;
                for (const auto& partition_ : partition) {
                    for (const auto& [n, q_nc] : partition_[c]) {
                        T_sq += q_nc * X.row(n).array().square().matrix();
                        M.row(c) += q_nc * X.row(n);
                        P[c] += q_nc;
                    }
                }
                if (P[c] <= 0) {
                    discard(c, "prior not positive");
                    continue;
                }
                M.row(c) /= P[c];
                if (!checkFinite(M.row(c))) {
                    discard(c, "mean not finite");
                    continue;
                }

                if (iso and sv) {
#pragma omp atomic
                    S_iso += T_sq.sum() - M.row(c).array().square().sum() * P[c];
                }
                if (iso and !sv) {
                    S(c,0) = T_sq.sum() / P[c] - M.row(c).array().square().sum();
                    S(c,0) /= D;
                    S(c,0) = std::max(S(c,0), reg_covar);
                    if (!std::isfinite(S(c,0))) {
                        discard(c, "variance not finite");
                        continue;
                    }
                    S.row(c).fill(S(c,0));
                }
                if (!iso and sv) {
                    S.row(c) = T_sq - M.row(c).array().square().matrix() * P[c];
                }
                if (!iso and !sv) {
                    S.row(c) = T_sq / P[c] - M.row(c).array().square().matrix();
                    S.row(c) = S.row(c).array().max(reg_covar);
                    if (!checkFinite(S.row(c))) {
                        discard(c, "variance not finite");
                        continue;
                    }
                }
            }
        }
    }
}

void Diagonal::M_step_finalize(size_t N) {
    /* */
    if (sv) {
        if (iso) {
            S_iso /= N * D;
            S_iso = std::max(S_iso, reg_covar);
            #pragma omp parallel for
            for (size_t c = 0; c < C; c++) {
                if (Mask[c]) {
                    S.row(c).fill(S_iso);
                }
            }
            S_iso = 0;
        }
        if (!iso) {
            S_shared.fill(0.);
            for (size_t c = 0; c < C; c++) {
                if (Mask[c]) {
                    S_shared += S.row(c);
                }
            }
            S_shared /= N;
            S_shared.array() = S_shared.array().max(reg_covar);
    #pragma omp parallel for
            for (size_t c = 0; c < C; c++) {
                if (Mask[c]) {
                    S.row(c) = S_shared;
                }
            }
        }
    }
    P_adjust();
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void Diagonal::bind(pybind11::module_& m) {
    pybind11::class_<Diagonal> Diagonal_class_(m, "Diagonal", pybind11::module_local(), R"(
    Gaussian mixture model with diagonal covariances.

    Parameters
    ----------
    C : int
        Number of components.
    D : int
        Dimensionality of the data.
    flat_prior : bool, optional
        Whether to use a flat prior for the mixture components. Defaults to False.
    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 1e-3.
    )");

    Diagonal_class_.def(pybind11::init<size_t, size_t, bool, const std::string, bool, precision_t>(), "C"_a, "D"_a,
                        "flat_prior"_a, "var_type"_a, "shared"_a, "reg_covar"_a = 1e-3);

    Diagonal_class_.def_property_readonly("C", &Diagonal::get_C, "The number of components.");
    Diagonal_class_.def_property_readonly("D", &Diagonal::get_D, "The dimensionality of the data");

    Diagonal_class_.def_property("means", &Diagonal::get_M, &Diagonal::set_M,
                                 "The mean values of all mixture components.");
    Diagonal_class_.def_property("variance", &Diagonal::get_S, &Diagonal::set_S,
                                 "The diagonal variances of all mixture components.");

    bind_base<precision_t>(Diagonal_class_);
}

#endif
