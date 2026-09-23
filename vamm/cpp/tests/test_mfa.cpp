/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#include <gtest/gtest.h>

#include "MFA.h"

class MFATest : public ::testing::Test {};

TEST_F(MFATest, ConstructorInitializesCorrectly) {
    MFA mfa(3, 5, 2, false, false, 1e-3);

    EXPECT_EQ(mfa.C, 3);
    EXPECT_EQ(mfa.D, 5);
    EXPECT_EQ(mfa.H, 2);

    EXPECT_NEAR(mfa.P.sum(), 1.0, 1e-12);
    EXPECT_NEAR(mfa.P[0], 1.0 / 3.0, 1e-12);

    EXPECT_TRUE(mfa.get_M().isZero());
    EXPECT_TRUE(mfa.get_A().isOnes());
    EXPECT_TRUE((mfa.get_S().array() == 1.0).all());
}

TEST_F(MFATest, ConstructorRejectsInvalidFactorDimension) {
    EXPECT_THROW(MFA(1, 3, 0), std::invalid_argument);
    EXPECT_THROW(MFA(1, 3, 4), std::invalid_argument);
}

TEST_F(MFATest, SetAndGetMeans) {
    MFA mfa(2, 3, 1);

    Matrix<> M(2, 3);
    M << 1, 2, 3, 4, 5, 6;

    mfa.set_M(M);

    EXPECT_TRUE(mfa.get_M().isApprox(M));
}

TEST_F(MFATest, SetMeanRejectsWrongDimension) {
    MFA mfa(2, 3, 1);
    Matrix<> M(2, 2);

    EXPECT_THROW(mfa.set_M(M), std::exception);
}

TEST_F(MFATest, SetAndGetVariance) {
    MFA mfa(2, 2, 1);

    Matrix<> S(2, 2);
    S << 1, 2, 3, 4;

    mfa.set_S(S);

    EXPECT_TRUE(mfa.get_S().isApprox(S));
}

TEST_F(MFATest, SetVarianceRejectsNegativeValues) {
    MFA mfa(1, 2, 1);

    Matrix<> S(1, 2);
    S << 1, -1;

    EXPECT_THROW(mfa.set_S(S), std::exception);
}

TEST_F(MFATest, SetAndGetLoadingMatrix) {
    MFA mfa(1, 3, 2);

    Matrix<> A(1, 6);
    A << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    mfa.set_A(A);

    EXPECT_TRUE(mfa.get_A().isApprox(A));
}

TEST_F(MFATest, AuxiliaryComputesCorrectLMAndDLog) {
    /*
    * For A = [2,3]:
    *
    * LM = I + A^T S^-1 A
    *
    *    = 1 + 2^2/4 + 3^2/9
    *    = 3
    * LM^-1 = 1/3

    * D_log = log(det(LM)) + log(4) + log(9)
    *       = log(3) + log(36)
    *       = log(108)
    */
    MFA mfa(1, 2, 1);

    Matrix<> A(1, 2);
    A << 2.0, 3.0;

    Matrix<> S(1, 2);
    S << 4.0, 9.0;

    mfa.set_A(A);
    mfa.set_S(S);

    mfa.auxiliary_(0);

    EXPECT_NEAR(mfa.S_diag_inv(0, 0), 0.25, 1e-12);
    EXPECT_NEAR(mfa.S_diag_inv(0, 1), 1.0 / 9.0, 1e-12);

    EXPECT_NEAR(mfa.LM[0](0, 0), 1.0 / 3.0, 1e-12);  // LM is here LM^(-1)
    EXPECT_NEAR(mfa.D_log[0], std::log(108.0), 1e-12);
}

TEST_F(MFATest, EStepLogJointMatchesAnalyticalValue) {
    /*
     * For this MFA:
     *
     * S^-1 = diag(1/4, 1/9)
     *
     * x^T S^-1 x = 2
     *
     * LM = 1 + A^T S^-1 A = 3
     *
     * z^2 = x^T Uc Vc x
     *     = x^T S^-1 A LM^-1 A^T S^-1 x
     *     = (A^T S^-1 x)^2 / LM
     *     = 4 / 3
     *
     * Mahalanobis distance:
     *
     * d² = 2 - 4/3 = 2/3
     *
     * log(det(S)) + log(det(LM))
     * = log(4) + log(9) + log(3)
     * = log(108)
     */
    MFA mfa(1, 2, 1);

    Matrix<> M(1, 2);
    M << 0.0, 0.0;

    Matrix<> S(1, 2);
    S << 4.0, 9.0;

    Matrix<> A(1, 2);
    A << 2.0, 3.0;

    mfa.set_M(M);
    mfa.set_S(S);
    mfa.set_A(A);

    mfa.auxiliary_(0);

    Vector<> x(2);
    x << 2.0, 3.0;

    Vector<> T0(2);
    Vector<> T1(2);

    precision_t log_prob;

    mfa.E_step_log_joint(x, 0, log_prob, T0, T1);

    const precision_t expected = -0.5 * (2.0 / 3.0 + std::log(108.0) + 2.0 * std::log(2.0 * M_PI));

    EXPECT_NEAR(log_prob, expected, 1e-12);
}

TEST_F(MFATest, MStepResetClearsValues) {
    MFA mfa(1, 2, 1);

    ColMatrix<> YE(2, 2);
    YE.setConstant(5.0);

    ColMatrix<> EE(2, 2);
    EE.setConstant(7.0);

    Vector<> T0(2);
    Vector<> T1(2);

    ColMatrix<> AM(2, 2);
    AM.setConstant(3.0);

    mfa.P[0] = 4.0;

    mfa.S_diag.row(0) << 10.0, 20.0;

    mfa.M_step_reset(0, YE, EE, T0, T1, AM);

    EXPECT_TRUE(YE.isZero());
    EXPECT_TRUE(EE.isZero());
    EXPECT_TRUE(mfa.get_P().isZero());
    EXPECT_TRUE(mfa.S_diag.isZero());
}

TEST_F(MFATest, MStepAccumulateUpdatesStatistics) {
    /*
     * A = [1,1], S = [1,1]
     *
     * UV = S^-1 A = [1,1]
     *
     * LM = I + A^T S^-1 A
     *    = 1 + 2
     *    = 3
     *
     * LM^-1 = 1/3
     *
     * z = x * UV * LM
     *   = [2,4] * [1,1] * 1/3
     *   = 2
     *
     * T1 = [z,1] = [2,1]
     *
     * EE = T1^T*T1
     *    = [[4,2],
     *       [2,1]]
     *
     * YE = x*T1
     *    = [[4,2],
     *       [8,4]]
     */
    MFA mfa(1, 2, 1);

    Matrix<> M(1, 2);
    M << 0.0, 0.0;
    mfa.set_M(M);

    Matrix<> A(1, 2);
    A << 1.0, 1.0;
    mfa.set_A(A);

    Matrix<> S(1, 2);
    S << 1.0, 1.0;
    mfa.set_S(S);

    mfa.auxiliary_(0);

    ColMatrix<> YE(2, 2);
    ColMatrix<> EE(2, 2);
    Vector<> T0(2);
    Vector<> T1(2);
    ColMatrix<> AM(2, 2);

    mfa.M_step_reset(0, YE, EE, T0, T1, AM);

    Vector<> x(2);
    x << 2.0, 4.0;

    mfa.M_step_accumulate(x, 1.0, 0, YE, EE, T0, T1, AM);

    EXPECT_NEAR(T1[0], 2.0, 1e-12);
    EXPECT_NEAR(T1[1], 1.0, 1e-12);

    EXPECT_NEAR(EE(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(EE(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(EE(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(EE(1, 1), 1.0, 1e-12);

    EXPECT_NEAR(YE(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(YE(0, 1), 2.0, 1e-12);

    EXPECT_NEAR(YE(1, 0), 8.0, 1e-12);
    EXPECT_NEAR(YE(1, 1), 4.0, 1e-12);

    EXPECT_NEAR(mfa.P[0], 1.0, 1e-12);
}

TEST_F(MFATest, MStepUpdateComputesMeanAndFactors) {
    /*
     * After adding P*LM:
     *
     * EE =
     * [7 2]
     * [2 1]
     *
     * AM = YE * inverse(EE)
     *
     *      = [0 2]
     *        [0 4]
     *
     * Therefore:
     *
     * M = AM.col(1)
     * A = AM.col(0)
     */
    MFA mfa(1, 2, 1);

    Matrix<> M(1, 2);
    M.setZero();
    mfa.set_M(M);

    Matrix<> A(1, 2);
    A << 1.0, 1.0;
    mfa.set_A(A);

    Matrix<> S(1, 2);
    S << 1.0, 1.0;
    mfa.set_S(S);

    mfa.auxiliary_(0);

    ColMatrix<> YE(2, 2);
    YE << 4.0, 2.0, 8.0, 4.0;

    ColMatrix<> EE(2, 2);
    EE << 4.0, 2.0, 2.0, 1.0;

    Vector<> T0(2);
    Vector<> T1(2);

    ColMatrix<> AM(2, 2);

    mfa.P[0] = 1.0;

    mfa.M_step_update(0, YE, EE, T0, T1, AM);

    EXPECT_NEAR(mfa.get_M()(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(mfa.get_M()(0, 1), 4.0, 1e-12);

    EXPECT_NEAR(mfa.get_A()(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(mfa.get_A()(0, 1), 0.0, 1e-12);
}

TEST_F(MFATest, DiagonalVarianceFinalizeProducesExpectedValues) {
    /*
     * Component 0:
     *
     * (4,8) / 2 = (2,4)
     *
     * Component 1:
     *
     * (6,0) / 2 = (3, 1e-3)
     */
    MFA mfa(2, 2, 1, false, false, 1e-3);

    mfa.P[0] = 2;
    mfa.P[1] = 2;

    mfa.S_diag.row(0) << 4, 8;
    mfa.S_diag.row(1) << 6, 0;

    mfa.M_step_finalize(4);

    EXPECT_NEAR(mfa.S_diag(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(mfa.S_diag(0, 1), 4.0, 1e-12);

    EXPECT_NEAR(mfa.S_diag(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(mfa.S_diag(1, 1), 1e-3, 1e-12);
}

TEST_F(MFATest, SharedVarianceFinalizeProducesExpectedValues) {
    /*
     * Shared variance:
     *
     * ((2,4) + (-6,8)) / 2
     *
     * = (1e-3,6)
     */
    MFA mfa(2, 2, 1, false, true, 1e-3);

    mfa.S_diag.row(0) << 2, 4;
    mfa.S_diag.row(1) << -6, 8;

    mfa.M_step_finalize(2);

    EXPECT_NEAR(mfa.S_diag(0, 0), 1e-3, 1e-12);
    EXPECT_NEAR(mfa.S_diag(0, 1), 6.0, 1e-12);

    EXPECT_NEAR(mfa.S_diag(1, 0), 1e-3, 1e-12);
    EXPECT_NEAR(mfa.S_diag(1, 1), 6.0, 1e-12);
}

TEST_F(MFATest, ZProjectionComputesExpectedValues) {
    /*
     * UV_right = S^-1 A * LM^-1
     *
     *            = [1/6, 1/9]^T
     *
     * z = x * UV_right
     *
     *   = 2/6 + 3/9
     *   = 2/3
     */
    MFA mfa(1, 2, 1);

    Matrix<> M(1, 2);
    M << 0.0, 0.0;
    mfa.set_M(M);

    Matrix<> S(1, 2);
    S << 4.0, 9.0;
    mfa.set_S(S);

    Matrix<> A(1, 2);
    A << 2.0, 3.0;
    mfa.set_A(A);

    mfa.auxiliary_(0);

    Vector<> x(2);
    x << 2.0, 3.0;

    Vector<> z = mfa.z_projection(x, 0);

    EXPECT_NEAR(z[0], 2.0 / 3.0, 1e-12);
}

TEST_F(MFATest, MahalanobisDistanceComputesExpectedValues) {
    /*
     * Mahalanobis distance:
     *
     * x^T S^-1 x = 2
     *
     * factor correction:
     *
     * (x^T S^-1 A)^2 / LM
     * = 4 / 3
     *
     * result:
     *
     * 2 - 4/3 = 2/3
     */
    MFA mfa(1, 2, 1);

    Matrix<> M(1, 2);
    M << 0.0, 0.0;
    mfa.set_M(M);

    Matrix<> S(1, 2);
    S << 4.0, 9.0;
    mfa.set_S(S);

    Matrix<> A(1, 2);
    A << 2.0, 3.0;
    mfa.set_A(A);

    mfa.auxiliary_(0);

    Vector<> x(2);
    x << 2.0, 3.0;

    precision_t distance = mfa.mahalanobis_distance(x, 0);

    EXPECT_NEAR(distance, 2.0 / 3.0, 1e-12);
}

TEST_F(MFATest, GenerateDataWithoutNoiseUsesCorrectMeans) {
    const size_t N = 5;
    const size_t C = 3;
    const size_t D = 2;
    const size_t H = 1;
    const int c = 1;

    MFA mfa(C, D, H);

    // Give each hidden state a distinct, easily identifiable mean.
    Matrix<> M(C, D);
    M << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    mfa.set_M(M);

    // A = 0 to just get the mean
    mfa.A.setZero();

    Matrix<> X = mfa.generate_data(N, c, false, 42);

    ASSERT_EQ(X.rows(), N);
    ASSERT_EQ(X.cols(), D);

    // With c=1 and no noise, every row must equal M.row(1).
    for (size_t n = 0; n < N; ++n) {
        EXPECT_TRUE(X.row(n).isApprox(mfa.get_M().row(c)));
    }
}

TEST_F(MFATest, GenerateDataIsDeterministicWithFixedSeed) {
    const size_t N = 100;
    const size_t C = 3;
    const size_t D = 2;
    const size_t H = 1;
    const int c = 1;
    const int seed = 42;

    MFA mfa(C, D, H);

    mfa.M << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    mfa.S_diag << 1.0, 4.0, 9.0, 16.0, 25.0, 36.0;
    mfa.A.setOnes();

    Matrix<> X1 = mfa.generate_data(N, c, true, seed);
    Matrix<> X2 = mfa.generate_data(N, c, true, seed);

    ASSERT_EQ(X1.rows(), N);
    ASSERT_EQ(X1.cols(), D);

    EXPECT_TRUE(X1.isApprox(X2));
}

TEST_F(MFATest, GenerateDataAddsNoiseWithExpectedMeanAndCovariance) {
    MFA mfa(1, 2, 1);

    mfa.M << 10.0, 20.0;
    mfa.S_diag << 4.0, 8.0;
    mfa.A << 3.0, 2.0;

    auto A = mfa.A.reshaped<Eigen::RowMajor>(2, 1);
    Matrix<> Covar(2, 2);
    Covar.noalias() = A * A.transpose();
    Covar.diagonal() += mfa.S_diag.row(0);

    const size_t N = 10000;

    Matrix<> X = mfa.generate_data(N, 0, true, 42);

    Vector<> mean = X.colwise().mean();
    Matrix<> empirical_covar = (X.rowwise() - mean);
    empirical_covar = (empirical_covar.transpose() * empirical_covar) / (N - 1);

    EXPECT_NEAR(mean(0), 10.0, 0.05);
    EXPECT_NEAR(mean(1), 20.0, 0.05);
    EXPECT_TRUE(Covar.isApprox(empirical_covar, 0.05));
}