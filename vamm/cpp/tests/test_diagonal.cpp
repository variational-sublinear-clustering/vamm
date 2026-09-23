/*
 * Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg
 * and Artificial Intelligence Lab of the University of Innsbruck.
 *
 * Tests for Diagonal mixture model.
 */

#include <gtest/gtest.h>

#include "Diagonal.h"

class DiagonalTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

TEST_F(DiagonalTest, ConstructorInitializesCorrectly) {
    Diagonal d(3, 2);

    EXPECT_EQ(d.C, 3);
    EXPECT_EQ(d.D, 2);

    EXPECT_NEAR(d.P.sum(), 1.0, 1e-12);
    EXPECT_NEAR(d.P[0], 1.0 / 3.0, 1e-12);

    EXPECT_TRUE(d.get_M().isZero());
    EXPECT_TRUE((d.get_S().array() == 1.0).all());
}

TEST_F(DiagonalTest, ConstructorRejectsInvalidVarianceType) {
    EXPECT_THROW({ Diagonal d(2, 3, false, "invalid_type", false); }, std::invalid_argument);
}

TEST_F(DiagonalTest, SetAndGetMeans) {
    Diagonal d(2, 3);

    Matrix<> M(2, 3);
    M << 1, 2, 3, 4, 5, 6;

    d.set_M(M);

    EXPECT_TRUE(d.get_M().isApprox(M));
}

TEST_F(DiagonalTest, SetMeanRejectsWrongDimension) {
    Diagonal d(2, 3);
    Matrix<> M(2, 2);

    EXPECT_THROW(d.set_M(M), std::exception);
}

TEST_F(DiagonalTest, SetAndGetVariance) {
    Diagonal d(2, 2);

    Matrix<> S(2, 2);
    S << 1, 2, 3, 4;

    d.set_S(S);

    EXPECT_TRUE(d.get_S().isApprox(S));
}

TEST_F(DiagonalTest, SetVarianceRejectsNegativeValues) {
    Diagonal d(1, 2);

    Matrix<> S(1, 2);
    S << 1, -1;

    EXPECT_THROW(d.set_S(S), std::exception);
}

TEST_F(DiagonalTest, EStepLogJointMatchesAnalyticalValue) {
    /*
   Manual calculation:
   diff = [2,3]
   Mahalanobis: 2^2/4 + 3^2/9 = 1 + 1 = 2
   log(det): log(4)+log(9)
   D log(2*pi): 2 log(2*pi)
   result: -0.5*(2+log(4)+log(9)+2log(2pi))
    */

    Diagonal d(1, 2);

    Matrix<> M(1, 2);
    M << 1.0, -2.0;
    d.set_M(M);

    Matrix<> S(1, 2);
    S << 4.0, 9.0;
    d.set_S(S);

    // x = [3,1]
    Vector<> x(2);
    x << 3.0, 1.0;

    // Gaussian component prior should be 1 for this test
    d.P[0] = 1.0;
    d.P_log[0] = 0.0;
    precision_t log_prob = 0.0;

    d.auxiliary_(0);
    d.E_step_log_joint(x, 0, log_prob);

    precision_t expected = -0.5 * (2.0 + std::log(4.0) + std::log(9.0) + 2.0 * std::log(2.0 * M_PI));
    EXPECT_NEAR(log_prob, expected, 1e-12);
}

TEST_F(DiagonalTest, MStepResetClearsValues) {
    Diagonal d(1, 2);

    Vector<> T_sq(2);
    T_sq << 10, 20;

    Matrix<> M(1, 2);
    M << 1.0, -2.0;
    d.set_M(M);

    Vector<> P(1);
    P << 1.0;
    d.set_P(P);

    d.M_step_reset(0, T_sq);

    EXPECT_TRUE(T_sq.isZero());
    EXPECT_TRUE(d.get_M().row(0).isZero());
    EXPECT_TRUE(d.get_P().isZero());
}

TEST_F(DiagonalTest, MStepAccumulateUpdatesStatistics) {
    Diagonal d(1, 2);

    Vector<> T_sq(2);
    T_sq.setZero();

    Vector<> x(2);
    x << 2, 3;

    d.M_step_reset(0, T_sq);

    d.M_step_accumulate(x, 2.0, 0, T_sq);

    EXPECT_NEAR(T_sq(0), 8.0, 1e-12);
    EXPECT_NEAR(T_sq(1), 18.0, 1e-12);

    EXPECT_NEAR(d.get_M()(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(d.get_M()(0, 1), 6.0, 1e-12);

    EXPECT_NEAR(d.get_P()(0), 2.0, 1e-12);
}

TEST_F(DiagonalTest, MStepUpdateComputesMean) {
    Diagonal d(1, 2, false, "diagonal");

    Vector<> T_sq(2);
    T_sq.setZero();

    Vector<> x(2);
    x << 4, 6;

    d.M_step_reset(0, T_sq);
    d.M_step_accumulate(x, 1.0, 0, T_sq);
    d.M_step_update(0, T_sq);

    EXPECT_NEAR(d.get_M()(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(d.get_M()(0, 1), 6.0, 1e-12);
}

TEST_F(DiagonalTest, DiagonalVarianceUpdateComputesVariance) {
    /*
     * Datapoints:
     *   x₁ = (1, 2)
     *   x₂ = (3, 6)
     *
     * Responsibilities:
     *   q₁ = q₂ = 1
     *
     * Accumulated statistics:
     *   P    = 2
     *   M    = x₁ + x₂ = (4, 8)
     *   T_sq = x₁² + x₂²
     *        = (1² + 3², 2² + 6²)
     *        = (10, 40)
     *
     * Mean:
     *   μ = M / P
     *     = (4, 8) / 2
     *     = (2, 4)
     *
     * Variance:
     *   σ² = T_sq / P - μ²
     *      = (10, 40)/2 - (2², 4²)
     *      = (5, 20) - (4, 16)
     *      = (1, 4)
     */
    Diagonal d(1, 2, false, "diagonal");

    Vector<> T_sq(2);
    d.M_step_reset(0, T_sq);

    Vector<> x1(2);
    x1 << 1, 2;

    Vector<> x2(2);
    x2 << 3, 6;

    d.M_step_accumulate(x1, 1.0, 0, T_sq);
    d.M_step_accumulate(x2, 1.0, 0, T_sq);

    d.M_step_update(0, T_sq);

    EXPECT_NEAR(d.get_S()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(d.get_S()(0, 1), 4.0, 1e-12);
}

TEST_F(DiagonalTest, IsotropicVarianceUpdateComputesVariance) {
    /*
     * Using the same datapoints as above:
     *
     * Diagonal variance:
     *   (1, 4)
     *
     * Isotropic variance:
     *   σ² = (1 + 4) / 2
     *      = 2.5
     *
     * The implementation stores this value for every dimension:
     *   S = (2.5, 2.5)
     */
    Diagonal d(1, 2, false, "isotropic");

    Vector<> T_sq(2);
    d.M_step_reset(0, T_sq);

    Vector<> x1(2);
    x1 << 1, 2;

    Vector<> x2(2);
    x2 << 3, 6;

    d.M_step_accumulate(x1, 1.0, 0, T_sq);
    d.M_step_accumulate(x2, 1.0, 0, T_sq);

    d.M_step_update(0, T_sq);

    EXPECT_NEAR(d.get_S()(0, 0), 2.5, 1e-12);
    EXPECT_NEAR(d.get_S()(0, 1), 2.5, 1e-12);
}

TEST_F(DiagonalTest, SharedDiagonalVarianceComputesVariance) {
    /*
     * Datapoints:
     *   x1 = (1, 2)
     *   x2 = (3, 6)
     *   x3 = (2, 4)
     *   x4 = (4, 8)
     *
     * Responsibilities:
     *   q1 = q2 = q3 = q4 1
     *
     * Accumulated statistics:
     *   P1=P2  = 2
     *   M1     = x₁ + x₂ = (4, 8)
     *   M2     = x3 + x4 = (6, 12)
     *   T_sq1  = x₁² + x₂²
     *          = (1² + 3², 2² + 6²)
     *          = (10, 40)
     *   T_sq2  = x3² + x4²
     *          = (2² + 4², 4² + 8²)
     *          = (20, 80)
     *
     * Mean:
     *  μ1 = M1 / P
     *     = (4, 8) / 2
     *     = (2, 4)
     *  μ2 = M2 / P
     *     = (6, 12) / 2
     *     = (3, 6)
     *
     * Variance:
     *  σ1² = T_sq - μ² * P
     *      = (10, 40) - (2², 4²) * 2
     *      = (10, 40) - (8, 32)
     *      = (2, 8)
     *  σ2² = T_sq - μ² * P
     *      = (20, 80) - (3², 6²) * 2
     *      = (20, 80) - (18, 72)
     *      = (2, 8)
     */
    /*
     * Component 0:
     *   σ² = (2, 8)
     *
     * Component 1:
     *   σ² = (2, 8)
     *
     * Shared variance:
     *   S_shared = ((2,8) + (2,8)) / N
     *            = (4,16) / 4
     *            = (1, 4)
     *
     * Both components receive the same shared variance.
     */
    Diagonal d(2, 2, false, "diagonal", true);

    Vector<> T_sq0(2), T_sq1(2);
    d.M_step_reset(0, T_sq0);
    d.M_step_reset(1, T_sq1);

    d.M_step_accumulate((Vector<>(2) << 1, 2).finished(), 1.0, 0, T_sq0);
    d.M_step_accumulate((Vector<>(2) << 3, 6).finished(), 1.0, 0, T_sq0);

    d.M_step_accumulate((Vector<>(2) << 2, 4).finished(), 1.0, 1, T_sq1);
    d.M_step_accumulate((Vector<>(2) << 4, 8).finished(), 1.0, 1, T_sq1);

    d.M_step_update(0, T_sq0);
    d.M_step_update(1, T_sq1);

    d.M_step_finalize(4);

    EXPECT_NEAR(d.get_S()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(d.get_S()(0, 1), 4.0, 1e-12);

    EXPECT_NEAR(d.get_S()(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(d.get_S()(1, 1), 4.0, 1e-12);
}

TEST_F(DiagonalTest, SharedIsotropicVarianceComputesVariance) {
    /*
     * Datapoints:
     *   x1 = (1, 2)
     *   x2 = (3, 6)
     *   x3 = (2, 4)
     *   x4 = (4, 8)
     *
     * Responsibilities:
     *   q1 = q2 = q3 = q4 1
     *
     * Accumulated statistics:
     *   P1=P2  = 2
     *   M1     = x₁ + x₂ = (4, 8)
     *   M2     = x3 + x4 = (6, 12)
     *   T_sq1  = x₁² + x₂²
     *          = (1² + 3², 2² + 6²)
     *          = (10, 40)
     *   T_sq2  = x3² + x4²
     *          = (2² + 4², 4² + 8²)
     *          = (20, 80)
     *
     * Mean:
     *  μ1 = M1 / P
     *     = (4, 8) / 2
     *     = (2, 4)
     *  μ2 = M2 / P
     *     = (6, 12) / 2
     *     = (3, 6)
     *
     * Variance:
     *  σ1² = T_sq.sum() - μ².sum() * P
     *      = 10 + 40 - (2² + 4²)  * 2
     *      = 50 - 40
     *      = 10
     *  σ2² = T_sq.sum() - μ².sum() * P
     *      = 20 + 80 - (3² + 6²) * 2
     *      = 100 - 90
     *      = 10
     */
    /*
     * Component 0:
     *   σ² = 10
     *
     * Component 1:
     *   σ² = 10
     *
     * Shared variance:
     *   S_iso = (10 + 10)/ (N*D)
     *            = 20 / (4*2)
     *            = 2.5
     *
     * Both components receive the same shared variance.
     */
    Diagonal d(2, 2, false, "isotropic", true);

    Vector<> T_sq0(2), T_sq1(2);
    d.M_step_reset(0, T_sq0);
    d.M_step_reset(1, T_sq1);

    d.M_step_accumulate((Vector<>(2) << 1, 2).finished(), 1.0, 0, T_sq0);
    d.M_step_accumulate((Vector<>(2) << 3, 6).finished(), 1.0, 0, T_sq0);

    d.M_step_accumulate((Vector<>(2) << 2, 4).finished(), 1.0, 1, T_sq1);
    d.M_step_accumulate((Vector<>(2) << 4, 8).finished(), 1.0, 1, T_sq1);

    d.M_step_update(0, T_sq0);
    d.M_step_update(1, T_sq1);

    d.M_step_finalize(4);

    EXPECT_NEAR(d.get_S()(0, 0), 2.5, 1e-12);
    EXPECT_NEAR(d.get_S()(0, 1), 2.5, 1e-12);

    EXPECT_NEAR(d.get_S()(1, 0), 2.5, 1e-12);
    EXPECT_NEAR(d.get_S()(1, 1), 2.5, 1e-12);
}

TEST_F(DiagonalTest, GenerateDataWithoutNoiseUsesCorrectMeans) {
    const size_t N = 5;
    const size_t C = 3;
    const size_t D = 2;
    const int c = 1;

    Diagonal d(C, D);

    // Give each hidden state a distinct, easily identifiable mean.
    Matrix<> M(C, D);
    M << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    d.set_M(M);

    Matrix<> X = d.generate_data(N, c, false, 42);

    ASSERT_EQ(X.rows(), N);
    ASSERT_EQ(X.cols(), D);

    // With c=1 and no noise, every row must equal M.row(1).
    for (size_t n = 0; n < N; ++n) {
        EXPECT_TRUE(X.row(n).isApprox(d.get_M().row(c)));
    }
}

TEST_F(DiagonalTest, GenerateDataIsDeterministicWithFixedSeed) {
    const size_t N = 100;
    const size_t C = 3;
    const size_t D = 2;
    const int c = 1;
    const int seed = 42;

    Diagonal d(C, D);

    d.M << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    d.S << 1.0, 4.0, 9.0, 16.0, 25.0, 36.0;

    Matrix<> X1 = d.generate_data(N, c, true, seed);
    Matrix<> X2 = d.generate_data(N, c, true, seed);

    ASSERT_EQ(X1.rows(), N);
    ASSERT_EQ(X1.cols(), D);

    EXPECT_TRUE(X1.isApprox(X2));
}

TEST_F(DiagonalTest, GenerateDataAddsNoiseWithExpectedMeanAndVariance) {
    Diagonal d(1, 2);

    d.M << 10.0, 20.0;
    d.S << 4.0, 8.0;

    const size_t N = 100000;

    Matrix<> X = d.generate_data(N, 0, true, 42);

    Vector<> mean = X.colwise().mean();
    Vector<> variance = (X.rowwise() - mean).array().square().colwise().sum() / (N - 1);

    EXPECT_NEAR(mean(0), 10.0, 0.05);
    EXPECT_NEAR(mean(1), 20.0, 0.05);
    EXPECT_NEAR(variance(0), 4.0, 0.05);
    EXPECT_NEAR(variance(1), 8.0, 0.05);
}