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

class MFA : public Mixture<MFA> {
   public:
    size_t H;

    const precision_t reg_covar; /* regularization added to the diagonal variance */
    const precision_t pi_factor;

    /* Model parameters */
    Matrix<> A;
    Matrix<> S_diag; /* Maybe use sparse matrix here */

    /* Utility */
    std::vector<Matrix<>, Eigen::aligned_allocator<Matrix<>>> LM;
    std::vector<ColMatrix<>, Eigen::aligned_allocator<ColMatrix<>>> UV;

    Matrix<> S_diag_inv;
    Vector<> D_log;
    Vector<> T;

    static constexpr bool loop_order_n = false;  // Shadows Mixture<MFA>::loop_order_n
    bool shared;

    MFA(size_t C_, size_t D_, size_t H_, bool flat_prior_, bool shared_, precision_t reg_covar_);

    Matrix<>& get_A();

    Matrix<>& get_S();

    void set_A(cRef<Matrix<>>);

    void set_S(cRef<Matrix<>>);

    void auxiliary_(const size_t c);

    void relocate_discarded_components(std::vector<size_t>& from, std::vector<size_t>& to,
                                       Random<std::mt19937_64>& rng);

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const;

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob, Vector<>& T0,
                          Vector<>& T1) const;

    template <class Lmbd>
    void M_step_allocate(const Lmbd& lmbd) const;

    void M_step_reset(size_t c, ColMatrix<>& YE, ColMatrix<>& EE, Vector<>&, Vector<>&, ColMatrix<>&);

    void M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c, ColMatrix<>& YE, ColMatrix<>& EE,
                           Vector<>& T0, Vector<>& T1, ColMatrix<>& AM);

    void M_step_update(size_t c, ColMatrix<>& YE, ColMatrix<>& EE, Vector<>& T0, Vector<>& T1,
                       ColMatrix<>& AM);

    void M_step_finalize(size_t N);

    std::function<void(size_t N)> finalize_variance;
    void finalize_shared_variance(size_t N);
    void finalize_diagonal_variance(size_t N);

    Vector<> z_projection(cRef<Vector<>> x, size_t c) const;

    precision_t mahalanobis_distance(cRef<Vector<>> x, size_t c) const;

    Matrix<> generate_data(size_t, int, bool, int);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(py::module_& m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

MFA::MFA(size_t C_, size_t D_, size_t H_, bool flat_prior_ = false, bool shared_ = false,
         precision_t reg_covar_ = 1e-3) :
    Mixture(C_, D_, flat_prior_),
    H(H_),
    reg_covar(reg_covar_),
    pi_factor(D_ * std::log(2.0 * M_PI)),
    A(Matrix<>::Ones(C_, D_ * H_)),
    S_diag(Matrix<>::Ones(C_, D_)),
    LM(C_, Matrix<>::Zero(H_, H_)),
    UV(C_, Matrix<>::Zero(D_, 2 * H_)),
    S_diag_inv(Matrix<>::Ones(C_, D_)),
    D_log(C_),
    T(Vector<>::Zero(D)),
    shared(shared_) {
    if ((H == 0) || (H > D)) {
        throw std::invalid_argument("( ( H == 0 ) || ( H > D ) )");
    }
    if (shared_) {
        finalize_variance = [&](size_t N) { finalize_shared_variance(N); };
    } else {
        finalize_variance = [&](size_t N) { finalize_diagonal_variance(N); };
    }
}

Matrix<>& MFA::get_A() { return A; }

Matrix<>& MFA::get_S() { return S_diag; }

void MFA::set_A(cRef<Matrix<>> A_) {
    checkSize(A_, C, D * H);
    A = A_;
}

void MFA::set_S(cRef<Matrix<>> S_) {
    checkSize(S_, C, D);
    checkLow(S_, 0, false);
    S_diag = S_;
}

void MFA::auxiliary_(const size_t c) {
    Matrix<> LMat(H, H);
    S_diag_inv.row(c) = S_diag.row(c).cwiseInverse();
    if (!checkFinite(S_diag_inv.row(c))) {
        discard(c, "zero in diagonal variance");
        return;
    }

    UV[c].leftCols(H).noalias() = S_diag_inv.row(c).asDiagonal() * A.row(c).reshaped<Eigen::RowMajor>(D, H);

    LM[c] = Matrix<>::Identity(H, H);
    LM[c].noalias() += A.row(c).reshaped<Eigen::RowMajor>(D, H).transpose() * UV[c].leftCols(H);

    LMat = LM[c].llt().matrixU();  // Cholesky decomposition such that LMat LMat^T = LM[c]
    // inside llt() a temp object is created
    if (!checkFinite(LMat)) {
        discard(c, "Cholesky decomposition of LM failed");
        return;
    }
    D_log[c] = 2.0 * LMat.diagonal().array().log().sum() + S_diag.row(c).array().log().sum();

    LMat = LMat.inverse();  // using Cholesky decomposition for inverse
    LM[c].noalias() = LMat * LMat.transpose();
    UV[c].rightCols(H).noalias() = UV[c].leftCols(H) * LM[c];  // LM symmetric matrix .transpose()
}

void MFA::relocate_discarded_components(std::vector<size_t>& from, std::vector<size_t>& to,
                                        Random<std::mt19937_64>& rng) {
    const size_t size = from.size();
#pragma omp parallel
    {
        size_t c1;
        size_t c2;
        std::normal_distribution<precision_t> std_normal{0.0, 1.0};
        ColVector<> noise(H);
#pragma omp for
        for (size_t i = 0; i < size; i++) {
            c1 = from[i];
            c2 = to[i];
            noise = noise.unaryExpr([&](precision_t) { return std_normal(rng()); });
            P[c1] = P[c2];
            // we could maybe add diag noise, but the new components are discarded more frequently then
            // (take care with storage order: z noise is colMajor, diag noise must be rowMajor)
            M.row(c1) = M.row(c2);
            M.row(c1).noalias() += A.row(c2).reshaped<Eigen::RowMajor>(D, H) * noise;
            S_diag.row(c1) = S_diag.row(c2);
            A.row(c1) = A.row(c2);

            if (!checkFinite(M.row(c1))) {
                discard(c1, "relocated mean not finite");
            }
        }
    }
}

template <class Lmbd>
void MFA::E_step_allocate(const Lmbd& lmbd) const {
    Vector<> T0(D);
    Vector<> T1(2 * H);

    lmbd(T0, T1);
}

void MFA::E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob, Vector<>& T0,
                           Vector<>& T1) const {
    T0 = x - M.row(c);
    T1.noalias() = T0 * UV[c];

    log_prob = S_diag_inv.row(c).dot(T0.array().square().matrix());
    log_prob -= T1.head(H).dot(T1.tail(H));
    log_prob += D_log[c] + pi_factor;
    log_prob *= -0.5;
    log_prob += P_log[c];
}

template <class Lmbd>
void MFA::M_step_allocate(const Lmbd& lmbd) const {
    ColMatrix<> YE(D, H + 1);
    ColMatrix<> EE(H + 1, H + 1);
    Vector<> T0(D);
    Vector<> T1(H + 1);
    ColMatrix<> AM(D, H + 1);

    lmbd(YE, EE, T0, T1, AM);
}

void MFA::M_step_reset(size_t c, ColMatrix<>& YE, ColMatrix<>& EE, Vector<>&, Vector<>&, ColMatrix<>&) {
    YE.fill(0.);
    EE.fill(0.);
    P[c] = 0.;
    S_diag.row(c).fill(0.);
}

void MFA::M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c, ColMatrix<>& YE, ColMatrix<>& EE,
                            Vector<>& T0, Vector<>& T1, ColMatrix<>&) {
    T1[H] = 1.0;
    T0 = x - M.row(c);
    T1.head(H).noalias() = T0 * UV[c].rightCols(H);

    EE.noalias() += q_nc * T1.transpose() * T1;

    T1 *= q_nc;
    YE.noalias() += x.transpose() * T1;
    S_diag.row(c) += q_nc * x.array().square().matrix();
    P[c] += q_nc;
}

void MFA::M_step_update(size_t c, ColMatrix<>& YE, ColMatrix<>& EE, Vector<>&, Vector<>&, ColMatrix<>& AM) {
    if (P[c] <= 0) {
        discard(c, "prior not positive");
        return;
    }
    EE.block(0, 0, H, H) += P[c] * LM[c];

    EE = EE.inverse();
    // AM = EE.transpose().bdcSvd(Eigen::ComputeThinU |
    // Eigen::ComputeThinV).solve(YE.transpose()).transpose();
    if (!checkFinite(EE)) {
        discard(c, "EE matrix inversion failed");
        return;
    }
    AM.noalias() = YE * EE;

    S_diag.row(c) -= AM.cwiseProduct(YE).rowwise().sum();  // -= diag(YE @ EE^-1 @ YE^T)

    S_diag.row(c) = S_diag.row(c).array().max(0.);
    M.row(c) = AM.col(H);
    A.row(c) = AM.block(0, 0, D, H).reshaped<Eigen::RowMajor>();
}

void MFA::M_step_finalize(size_t N) {
    /* */
    finalize_variance(N);
    P_adjust();
}

void MFA::finalize_shared_variance(size_t N) {
    T.fill(0.);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            T += S_diag.row(c);
        }
    }
    T /= N;
    T.array() = T.array().max(reg_covar);
#pragma omp parallel for
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            S_diag.row(c) = T;
        }
    }
}

void MFA::finalize_diagonal_variance(size_t) {
#pragma omp parallel for
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            S_diag.row(c) /= P[c];
            S_diag.row(c) = S_diag.row(c).array().max(reg_covar);
        }
    }
}

Vector<> MFA::z_projection(cRef<Vector<>> x, size_t c) const {
    Vector<> z((x - M.row(c)) * UV[c].rightCols(H));
    return z;
}

precision_t MFA::mahalanobis_distance(cRef<Vector<>> x, size_t c) const {
    Vector<> T0(D);
    Vector<> T1(2 * H);
    precision_t val;

    T0 = x - M.row(c);
    T1.noalias() = T0 * UV[c];

    val = S_diag_inv.row(c).dot(T0.array().square().matrix());
    val -= T1.head(H).dot(T1.tail(H));

    return val;
}

Matrix<> MFA::generate_data(size_t N, int c = -1, bool add_noise = true, int seed = -1) {
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

    std::normal_distribution<precision_t> std_normal(0.0, 1.0);
    Matrix<> z(Matrix<>::NullaryExpr(N, H, [&]() { return std_normal(rng()); }));

    Matrix<> X(N, D);

#pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        X.row(n).noalias() =
            (A.row(hidden_states[n]).reshaped<Eigen::RowMajor>(D, H) * z.row(n).transpose()).transpose();
        X.row(n) += M.row(hidden_states[n]);
    }

    if (add_noise) {
#pragma omp parallel
        {
            Vector<> noise_vec(D);
            std::normal_distribution<precision_t> std_normal(0.0, 1.0);
#pragma omp for
            for (size_t n = 0; n < N; ++n) {
                noise_vec = noise_vec.unaryExpr([&](precision_t) { return std_normal(rng()); });
                X.row(n) += (S_diag.row(hidden_states[n]).array().sqrt() * noise_vec.array()).matrix();
            }
        }
    }
    return X;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

void MFA::bind(py::module_& m) {
    py::class_<MFA> MFA_class_(m, "MFA", py::module_local());

    MFA_class_.def(py::init<size_t, size_t, size_t, bool, bool, precision_t>(), "C"_a, "D"_a, "H"_a,
                   "flat_prior"_a = false, "shared"_a = false, "reg_covar"_a = 1e-3);

    MFA_class_.def_readonly("H", &MFA::H);

    MFA_class_.def_property("variance", &MFA::get_S, &MFA::set_S);
    MFA_class_.def_property("A", &MFA::get_A, &MFA::set_A);

    MFA_class_.def_readwrite("shared", &MFA::shared);

    MFA_class_.def("z_projection", &MFA::z_projection, "x"_a.noconvert(), "c"_a);
    MFA_class_.def("mahalanobis_distance", &MFA::mahalanobis_distance, "x"_a.noconvert(), "c"_a);
    MFA_class_.def("generate_data", &MFA::generate_data, "N"_a, "c"_a = -1, "add_noise"_a = true,
                   "seed"_a = -1);

    bind_base<precision_t>(MFA_class_);
}

#endif
