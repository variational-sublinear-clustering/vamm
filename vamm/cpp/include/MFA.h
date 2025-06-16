/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
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
    size_t C;
    size_t D;
    size_t H;

    const precision_t reg_covar; /* regularization added to the diagonal variance */
    const precision_t pi_factor;

    /* Model parameters */
    Matrix<> A;
    Matrix<> M;
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

    size_t get_C(void) const;

    size_t get_D(void) const;

    size_t get_H(void) const;

    Matrix<>& get_A();

    Matrix<>& get_M();

    Matrix<>& get_S();

    void set_A(cRef<Matrix<>>);

    void set_M(cRef<Matrix<>>);

    void set_S(cRef<Matrix<>>);

    void auxiliary_(const size_t c);

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const;

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob, Vector<>& T0,
                          Vector<>& T1) const;

    void M_step_update(cRef<Matrix<>> X, const std::vector<std::vector<q_t>>& partition);

    void M_step_finalize(size_t N);
    Vector<> z_projection(cRef<Vector<>> x, size_t c) const;

    precision_t mahalanobis_distance(cRef<Vector<>> x, size_t c) const;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(py::module_& m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

MFA::MFA(size_t C_, size_t D_, size_t H_, bool flat_prior_ = false, bool shared_ = false,
         precision_t reg_covar_ = 1e-3) :
    Mixture(C_, D_, flat_prior_),
    C(C_),
    D(D_),
    H(H_),
    reg_covar(reg_covar_),
    pi_factor(D_ * std::log(2.0 * M_PI)),
    A(Matrix<>::Zero(C_, D_ * H_)),
    M(Matrix<>::Zero(C_, D_)),
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
    P_adjust();
}

size_t MFA::get_C(void) const { return C; }

size_t MFA::get_D(void) const { return D; }

size_t MFA::get_H(void) const { return H; }

Matrix<>& MFA::get_A() { return A; }

Matrix<>& MFA::get_M() { return M; }

Matrix<>& MFA::get_S() { return S_diag; }

void MFA::set_A(cRef<Matrix<>> A_) {
    checkSize(A_, C, D * H);
    A = A_;
}

void MFA::set_M(cRef<Matrix<>> M_) {
    checkSize(M_, C, D);
    M = M_;
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

void MFA::M_step_update(cRef<Matrix<>> X, const std::vector<std::vector<q_t>>& partition) {
#pragma omp parallel
    {
        ColMatrix<> YE(D, H + 1);
        ColMatrix<> EE(H + 1, H + 1);
        Vector<> T0(D);
        Vector<> T1(H + 1);
        ColMatrix<> AM(D, H + 1);
        Vector<> T2(D);
        size_t data_per_component;

#pragma omp for schedule(dynamic, 1)
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                EE.fill(0.);
                YE.fill(0.);
                P[c] = 0.;
                S_diag.row(c).fill(0.);
                data_per_component = 0;
                for (const auto& partition_ : partition) {
                    for (const auto& [n, q_nc] : partition_[c]) {
                        T1[H] = 1.0;
                        T0 = X.row(n) - M.row(c);
                        T1.head(H).noalias() = T0 * UV[c].rightCols(H);

                        EE.noalias() += q_nc * T1.transpose() * T1;

                        T1 *= q_nc;
                        YE.noalias() += X.row(n).transpose() * T1;
                        S_diag.row(c) += q_nc * X.row(n).array().square().matrix();
                        P[c] += q_nc;
                        data_per_component++;
                    }
                }
                if (!shared and (reg_covar == 0.0) and (data_per_component < H + 2)) {
                    discard(c, "Contains less than H+2 data points!");
                    continue;
                }
                if (P[c] <= 0) {
                    discard(c, "prior not positive");
                    continue;
                }
                EE.block(0, 0, H, H) += P[c] * LM[c];

                EE = EE.inverse();
                // AM = EE.transpose().bdcSvd(Eigen::ComputeThinU |
                // Eigen::ComputeThinV).solve(YE.transpose()).transpose();
                if (!checkFinite(EE)) {
                    discard(c, "EE matrix inversion failed");
                    continue;
                }
                AM.noalias() = YE * EE;

                S_diag.row(c) -= AM.cwiseProduct(YE).rowwise().sum();  // -= diag(YE @ EE^-1 @ YE^T)

                S_diag.row(c) = S_diag.row(c).array().max(0.);
                M.row(c) = AM.col(H);
                A.row(c) = AM.block(0, 0, D, H).reshaped<Eigen::RowMajor>();
            }
        }
    }
}

void MFA::M_step_finalize(size_t N) {
    if (shared) {
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
    } else {
#pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            if (Mask[c]) {
                S_diag.row(c) /= P[c];
                S_diag.row(c) = S_diag.row(c).array().max(reg_covar);
            }
        }
    }
    P_adjust();
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

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

void MFA::bind(py::module_& m) {
    py::class_<MFA> MFA_class_(m, "MFA", py::module_local(), R"(
    Mixture of Factor Analyzer.

    Parameters
    ----------
    C : int
        Number of components.
    D : int
        Dimensionality of the data.
    H : int
        Dimensionality of the factors.
    flat_prior : bool, optional
        Whether to use a flat prior for the mixture components. Defaults to False.
    shared : bool, optional
        Whether the diagonal variances are shared among components. Defaults to False.
    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 1e-3.
    )");

    MFA_class_.def(py::init<size_t, size_t, size_t, bool, bool, precision_t>(), "C"_a, "D"_a, "H"_a,
                   "flat_prior"_a = false, "shared"_a = false, "reg_covar"_a = 1e-3);

    MFA_class_.def_property_readonly("C", &MFA::get_C, "The number of components.");
    MFA_class_.def_property_readonly("D", &MFA::get_D, "The dimensionality of the data");
    MFA_class_.def_property_readonly("H", &MFA::get_H, "The dimensionality of the factors.");

    MFA_class_.def_property("means", &MFA::get_M, &MFA::set_M, "The mean values of all mixture components.");
    MFA_class_.def_property("variance", &MFA::get_S, &MFA::set_S,
                            "The diagonal variances of all mixture components.");
    MFA_class_.def_property("A", &MFA::get_A, &MFA::set_A,
                            "The factor loading matrices of all mixture components.");

    MFA_class_.def_readwrite("shared", &MFA::shared,
                             "Whether the diagonal variances are shared among components.");

    MFA_class_.def("z_projection", &MFA::z_projection, "x"_a.noconvert(), "c"_a, R"(
    z_projection(x: numpy.ndarray[numpy.float64[1, n]], c: int)
    Projects the given data point into the latent space by calculating the most likely factor 'z' given component 'c'.

    Parameters
    ----------
    x : npt.ndarray
        Data point.
    c : int
        Component index.

    Returns
    -------
    z : npt.ndarray
        Factor represented as an H-dimensional vector.
    )");

    MFA_class_.def("mahalanobis_distance", &MFA::mahalanobis_distance, "x"_a.noconvert(), "c"_a, R"(
    mahalanobis distance(x: numpy.ndarray[numpy.float64[1, n]], c: int)

    Parameters
    ----------
    x : npt.ndarray
        Data point.
    c : int
        Component index.

    Returns
    -------
    val : float
        mahalanobis distance
    )");

    bind_base<precision_t>(MFA_class_);
}

#endif
