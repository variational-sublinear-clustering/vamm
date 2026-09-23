/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/Dense>
#include <stdexcept>

#include "Mixture.h"

class Diagonal : public Mixture<Diagonal> {
   public:
    std::string var_type;

    bool iso;
    bool sv;

    Matrix<> S;
    Matrix<> S_inv;
    Vector<> S_shared;
    precision_t S_iso;
    precision_t reg_covar;
    Vector<> D_log;
    const precision_t pi_factor;

    void auxiliary_(const size_t c);

    void relocate_discarded_components(std::vector<size_t>& from, std::vector<size_t>& to,
                                       Random<std::mt19937_64>& rng);

    Diagonal(size_t C_, size_t D_, bool flat_prior_, const std::string& var_type_, bool shared_,
             precision_t reg_covar_);

    Matrix<>& get_S(void);

    void set_S(cRef<Matrix<>> S_);

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const;

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob) const;

    template <class Lmbd>
    void M_step_allocate(const Lmbd& lmbd) const;
    void M_step_reset(size_t c, Vector<>& T_sq);
    void M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c, Vector<>& T_sq);
    void M_step_update(size_t c, Vector<>& T_sq);

    std::function<void(size_t, const Vector<>&)> update_variance;
    void update_isotropic_shared_variance(size_t c, const Vector<>& T_sq);
    void update_isotropic_variance(size_t c, const Vector<>& T_sq);
    void update_diagonaltied_variance(size_t c, const Vector<>& T_sq);
    void update_diagonal_variance(size_t c, const Vector<>& T_sq);

    void M_step_finalize(size_t N);

    std::function<void(size_t N)> finalize_variance;
    void finalize_isotropic_shared_variance(size_t N);
    void finalize_diagonaltied_variance(size_t N);

    Matrix<> generate_data(size_t, int, bool, int);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(pybind11::module_& m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

Diagonal::Diagonal(size_t C_, size_t D_, bool flat_prior_ = false, const std::string& var_type_ = "diagonal",
                   bool shared_ = false, precision_t reg_covar_ = 1e-3) :
    Mixture(C_, D_, flat_prior_),
    S(Matrix<>::Ones(C_, D_)),
    S_inv(C_, D_),
    S_shared(D_),
    S_iso(0.),
    reg_covar(reg_covar_),
    D_log(C_),
    pi_factor(D_ * std::log(2.0 * M_PI)) {
    finalize_variance = [](size_t) {
        // no-op placeholder
    };
    if ((var_type_ == "isotropic") && shared_) {
        update_variance = [&](size_t c, const Vector<>& T_sq) { update_isotropic_shared_variance(c, T_sq); };
        finalize_variance = [&](size_t N) { finalize_isotropic_shared_variance(N); };

    } else if ((var_type_ == "isotropic") && !shared_) {
        update_variance = [&](size_t c, const Vector<>& T_sq) { update_isotropic_variance(c, T_sq); };

    } else if ((var_type_ == "diagonal") && shared_) {
        update_variance = [&](size_t c, const Vector<>& T_sq) { update_diagonaltied_variance(c, T_sq); };
        finalize_variance = [&](size_t N) { finalize_diagonaltied_variance(N); };

    } else if ((var_type_ == "diagonal") && !shared_) {
        update_variance = [&](size_t c, const Vector<>& T_sq) { update_diagonal_variance(c, T_sq); };

    } else {
        throw std::invalid_argument("'covariance_type' should be one of 'isotropic' or 'diagonal', but got '" +
                                    var_type_ + "'");
    }
}

Matrix<>& Diagonal::get_S(void) { return S; }

void Diagonal::set_S(cRef<Matrix<>> S_) {
    checkSize(S_, C, D);
    checkLow(S_, 0., false);  // TODO: or S >= reg_covar?
    S = S_;
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

void Diagonal::relocate_discarded_components(std::vector<size_t>& from, std::vector<size_t>& to,
                                             Random<std::mt19937_64>& rng) {
#pragma omp parallel
    {
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
            M.row(c1) = M.row(c2) + (S.row(c2).array().sqrt() * noise.array()).matrix();
            S.row(c1) = S.row(c2);
            if (!checkFinite(M.row(c1))) {
                discard(c1, "relocated mean not finite");
            }
        }
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

template <class Lmbd>
void Diagonal::M_step_allocate(const Lmbd& lmbd) const {
    Vector<> T_sq(D);

    lmbd(T_sq);
}

void Diagonal::M_step_reset(size_t c, Vector<>& T_sq) {
    T_sq.fill(0);
    M.row(c).fill(0);
    P[c] = 0.0;
}

void Diagonal::M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c, Vector<>& T_sq) {
    T_sq += q_nc * x.array().square().matrix();
    M.row(c) += q_nc * x;
    P[c] += q_nc;
}

void Diagonal::M_step_update(size_t c, Vector<>& T_sq) {
    /* */
    if (P[c] <= 0) {
        discard(c, "prior not positive");
        return;
    }
    M.row(c) /= P[c];
    if (!checkFinite(M.row(c))) {
        discard(c, "mean not finite");
        return;
    }
    update_variance(c, T_sq);
}

void Diagonal::update_isotropic_shared_variance(size_t c, const Vector<>& T_sq) {
#pragma omp atomic
    S_iso += T_sq.sum() - M.row(c).array().square().sum() * P[c];
}

void Diagonal::update_isotropic_variance(size_t c, const Vector<>& T_sq) {
    S(c, 0) = T_sq.sum() / P[c] - M.row(c).array().square().sum();
    S(c, 0) /= D;
    S(c, 0) = std::max(S(c, 0), reg_covar);

    if (!std::isfinite(S(c, 0))) {
        discard(c, "variance not finite");
        return;
    }

    S.row(c).fill(S(c, 0));
}

void Diagonal::update_diagonaltied_variance(size_t c, const Vector<>& T_sq) {
    S.row(c) = T_sq - M.row(c).array().square().matrix() * P[c];
}

void Diagonal::update_diagonal_variance(size_t c, const Vector<>& T_sq) {
    S.row(c) = T_sq / P[c] - M.row(c).array().square().matrix();

    S.row(c) = S.row(c).array().max(reg_covar);

    if (!checkFinite(S.row(c))) {
        discard(c, "variance not finite");
        return;
    }
}

void Diagonal::M_step_finalize(size_t N) {
    /* */
    finalize_variance(N);
    P_adjust();
}

void Diagonal::finalize_isotropic_shared_variance(size_t N) {
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

void Diagonal::finalize_diagonaltied_variance(size_t N) {
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

Matrix<> Diagonal::generate_data(size_t N, int c = -1, bool add_noise = true, int seed = -1) {
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

    if (add_noise) {
#pragma omp parallel
        {
            Vector<> noise_vec(D);
            std::normal_distribution<precision_t> std_normal(0.0, 1.0);
#pragma omp for
            for (size_t n = 0; n < N; ++n) {
                noise_vec = noise_vec.unaryExpr([&](precision_t) { return std_normal(rng()); });
                X.row(n) += (S.row(hidden_states[n]).array().sqrt() * noise_vec.array()).matrix();
            }
        }
    }
    return X;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void Diagonal::bind(pybind11::module_& m) {
    pybind11::class_<Diagonal> Diagonal_class_(m, "Diagonal", pybind11::module_local());
    Diagonal_class_.def(pybind11::init<size_t, size_t, bool, const std::string, bool, precision_t>(), "C"_a,
                        "D"_a, "flat_prior"_a, "var_type"_a, "shared"_a, "reg_covar"_a = 1e-3);
    Diagonal_class_.def_property("variance", &Diagonal::get_S, &Diagonal::set_S);
    Diagonal_class_.def("generate_data", &Diagonal::generate_data, "N"_a, "c"_a = -1, "add_noise"_a = true,
                        "seed"_a = -1);

    bind_base<precision_t>(Diagonal_class_);
}

#endif
