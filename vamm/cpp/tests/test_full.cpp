/*
 * Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg
 * and Artificial Intelligence Lab of the University of Innsbruck.
 *
 * Tests for Full mixture model.
 */

#include <gtest/gtest.h>

#include "Full.h"

class FullTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

TEST_F(FullTest, ConstructorInitializesCorrectly) {
    Full f(3, 2);

    Matrix<> Eye(Matrix<>::Identity(2, 2));

    EXPECT_EQ(f.C, 3);
    EXPECT_EQ(f.D, 2);

    EXPECT_NEAR(f.P.sum(), 1.0, 1e-12);
    EXPECT_NEAR(f.P[0], 1.0 / 3.0, 1e-12);

    EXPECT_TRUE(f.get_M().isZero());
    for (size_t c = 0; c < f.C; c++) {
        EXPECT_TRUE(f.Covar.row(c).reshaped<Eigen::RowMajor>(f.D, f.D).isApprox(Eye));
    }
}

TEST_F(FullTest, SetAndGetMeans) {
    Full f(2, 3);

    Matrix<> M(2, 3);
    M << 1, 2, 3, 4, 5, 6;

    f.set_M(M);

    EXPECT_TRUE(f.get_M().isApprox(M));
}

TEST_F(FullTest, SetMeanRejectsWrongDimension) {
    Full f(2, 3);
    Matrix<> M(2, 2);

    EXPECT_THROW(f.set_M(M), std::exception);
}

TEST_F(FullTest, SetAndGetCovariance) {
    Full f(2, 2);

    Matrix<> Cov(2, 4);
    Cov << 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 6.0;

    f.set_Covar(Cov);

    EXPECT_TRUE(f.get_Covar().isApprox(Cov));
}

TEST_F(FullTest, EStepLogJointMatchesAnalyticalValue) {
    Full f(1, 2);

    Matrix<> M(1, 2);
    M.setZero();
    f.set_M(M);

    Matrix<> Cov(1, 4);
    Cov << 1.0, 0.0, 0.0, 1.0;

    f.set_Covar(Cov);

    f.auxiliary();

    Vector<> x(2);
    x << 1.0, 1.0;

    Vector<> T0(2);

    precision_t log_prob;

    f.E_step_log_joint(x, 0, log_prob, T0);

    /*
     * Mahalanobis distance:
     *
     * 1² + 1² = 2
     *
     * det(Cov)=1
     *
     * log p(x)
     *
     * = -0.5*(2 + 2log(2pi)) + log(1)
     */
    const precision_t expected = -0.5 * (2.0 + 2.0 * std::log(2.0 * M_PI)) + std::log(1.0);

    EXPECT_NEAR(log_prob, expected, 1e-12);
}

TEST_F(FullTest, MStepResetClearsValues) {
    Full f(1, 2);

    f.P[0] = 5.0;

    f.M.row(0) << 1.0, 2.0;

    f.Covar.row(0).setOnes();

    f.M_step_reset(0);

    EXPECT_NEAR(f.P[0], 0.0, 1e-12);
    EXPECT_NEAR(f.M.row(0).norm(), 0.0, 1e-12);
    EXPECT_NEAR(f.Covar.row(0).norm(), 0.0, 1e-12);
}

TEST_F(FullTest, MStepAccumulateUpdatesStatistics) {
    /*
     * Stored x*x^T:
     *
     * [4 8]
     * [8 16]
     */
    Full f(1, 2);

    f.M_step_reset(0);

    Vector<> x(2);
    x << 2.0, 4.0;

    f.M_step_accumulate(x, 1.0, 0);

    EXPECT_NEAR(f.P[0], 1.0, 1e-12);
    EXPECT_NEAR(f.M(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(f.M(0, 1), 4.0, 1e-12);
    EXPECT_NEAR(f.Covar(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(f.Covar(0, 3), 16.0, 1e-12);
}

TEST_F(FullTest, MStepUpdateComputesMeanAndCovariance) {
    Full full(1, 2, false, false, 1e-3);

    full.M_step_reset(0);

    Vector<> x(2);
    x << 2.0, 4.0;

    full.M_step_accumulate(x, 1.0, 0);

    full.M_step_update(0);

    EXPECT_NEAR(full.M(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(full.M(0, 1), 4.0, 1e-12);

    /*
     * One sample:
     *
     * covariance becomes zero
     * plus regularization.
     */
    EXPECT_NEAR(full.Covar(0, 0), 1e-3, 1e-12);
    EXPECT_NEAR(full.Covar(0, 3), 1e-3, 1e-12);
}

TEST_F(FullTest, SharedCovarianceIsAppliedToAllComponents) {
    Full full(2, 2, false, true);

    full.Covar.row(0) << 1.0, 0.0, 0.0, 1.0;

    full.Covar.row(1) << 3.0, 0.0, 0.0, 3.0;

    full.M_step_finalize(2);

    EXPECT_NEAR(full.Covar(0, 0), full.Covar(1, 0), 1e-12);
    EXPECT_NEAR(full.Covar(0, 3), full.Covar(1, 3), 1e-12);
}

TEST_F(FullTest, GenerateDataWithoutNoiseUsesCorrectMeans) {
    const size_t N = 5;
    const size_t C = 3;
    const size_t D = 2;
    const int c = 1;

    Full full(C, D);

    // Give each hidden state a distinct, easily identifiable mean.
    full.M << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    Matrix<> X = full.generate_data(N, c, false, 42);

    ASSERT_EQ(X.rows(), N);
    ASSERT_EQ(X.cols(), D);

    // With c=1 and no noise, every row must equal M.row(1).
    for (size_t n = 0; n < N; ++n) {
        EXPECT_TRUE(X.row(n).isApprox(full.get_M().row(c)));
    }
}

TEST_F(FullTest, GenerateDataIsDeterministicWithFixedSeed) {
    const size_t N = 100;
    const size_t C = 1;
    const size_t D = 2;
    const int c = 0;
    const int seed = 42;

    Full full(C, D);

    full.M << 1.0, 2.0;
    full.Covar << 2.0, 1.0, 1.0, 2.0;

    Matrix<> X1 = full.generate_data(N, c, true, seed);
    Matrix<> X2 = full.generate_data(N, c, true, seed);

    ASSERT_EQ(X1.rows(), N);
    ASSERT_EQ(X1.cols(), D);

    EXPECT_TRUE(X1.isApprox(X2));
}

TEST_F(FullTest, GenerateDataAddsNoiseWithExpectedMeanAndCovariance) {
    Full full(1, 2);

    Matrix<> Covar(2, 2);
    Covar << 2.0, 1.0, 1.0, 2.0;

    full.M << 10.0, 20.0;
    full.set_Covar(Covar.reshaped<Eigen::RowMajor>(1, 4));

    const size_t N = 10000;

    Matrix<> X = full.generate_data(N, 0, true, 42);

    Vector<> mean = X.colwise().mean();
    Matrix<> empirical_covar = (X.rowwise() - mean);
    empirical_covar = (empirical_covar.transpose() * empirical_covar) / (N - 1);

    EXPECT_NEAR(mean(0), 10.0, 0.05);
    EXPECT_NEAR(mean(1), 20.0, 0.05);
    EXPECT_TRUE(Covar.isApprox(empirical_covar, 0.05));
}