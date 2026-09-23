#include <gtest/gtest.h>

#include "Mixture.h"

class DummyMixture : public Mixture<DummyMixture> {
   public:
    DummyMixture(size_t C, size_t D, bool flat_prior = false) : Mixture(C, D, flat_prior) {}

    void auxiliary_(const size_t) {}

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const {
        Vector<> dummy(D);
        lmbd(dummy);
    }

    void E_step_log_joint(cRef<Vector<>> x, size_t c, precision_t& log_prob, Vector<>&) const {
        /*
         * Simple deterministic likelihood:
         *
         * component with larger index
         * has larger likelihood.
         */
        log_prob = -x.squaredNorm() + static_cast<precision_t>(c);
    }
};

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

TEST(MixtureTest, ConstructorInitializesCorrectly) {
    DummyMixture mix(3, 2);

    EXPECT_EQ(mix.C, 3);
    EXPECT_EQ(mix.D, 2);

    EXPECT_EQ(mix.C_active, 3);

    EXPECT_TRUE(mix.Mask.all());

    EXPECT_NEAR(mix.P.sum(), 1.0, 1e-12);
    EXPECT_NEAR(mix.P[0], 1.0 / 3.0, 1e-12);

    EXPECT_TRUE(mix.get_M().isZero());
}

TEST(MixtureTest, RejectsInvalidDimensions) {
    EXPECT_THROW(DummyMixture(0, 2), std::invalid_argument);
    EXPECT_THROW(DummyMixture(2, 0), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Prior
// -----------------------------------------------------------------------------

TEST(MixtureTest, SetPriorNormalizesValues) {
    DummyMixture mix(3, 2);

    Vector<> P(3);
    P << 1.0, 2.0, 3.0;

    mix.set_P(P);

    EXPECT_NEAR(mix.P[0], 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(mix.P[1], 2.0 / 6.0, 1e-12);
    EXPECT_NEAR(mix.P[2], 3.0 / 6.0, 1e-12);
    EXPECT_NEAR(mix.P_log[0], std::log(1.0 / 6.0), 1e-12);
}

TEST(MixtureTest, FlatPriorCannotBeChanged) {
    DummyMixture mix(4, 2, true);

    Vector<> P(4);
    P << 1, 2, 3, 4;

    EXPECT_THROW(mix.set_P(P), std::invalid_argument);
    EXPECT_NEAR(mix.P[0], 0.25, 1e-12);
    EXPECT_NEAR(mix.P_log[0], std::log(0.25), 1e-12);
}

// -----------------------------------------------------------------------------
// Mask / discard
// -----------------------------------------------------------------------------

TEST(MixtureTest, SetMaskDisablesComponents) {
    DummyMixture mix(3, 2);

    Vector<bool> mask(3);
    mask << true, false, true;

    mix.set_Mask(mask);

    EXPECT_EQ(mix.C_active, 2);
    EXPECT_FALSE(mix.Mask[1]);
    EXPECT_NEAR(mix.P[1], 0.0, 1e-12);
    EXPECT_TRUE(std::isnan(mix.P_log[1]));
}

TEST(MixtureTest, DiscardRemovesComponent) {
    DummyMixture mix(3, 2);

    mix.discard(1);

    EXPECT_FALSE(mix.Mask[1]);
    EXPECT_EQ(mix.C_active, 2);
    EXPECT_NEAR(mix.P[1], 0.0, 1e-12);
}

TEST(MixtureTest, CannotDiscardLastComponent) {
    DummyMixture mix(1, 2);

    EXPECT_THROW(mix.discard(0), std::runtime_error);
}

// -----------------------------------------------------------------------------
// Component queries
// -----------------------------------------------------------------------------

TEST(MixtureTest, ValidChecksMaskAndIndex) {
    DummyMixture mix(3, 2);

    EXPECT_TRUE(mix.valid(0));

    mix.discard(1);

    EXPECT_FALSE(mix.valid(1));
    EXPECT_FALSE(mix.valid(10));
}

TEST(MixtureTest, FirstActiveReturnsFirstEnabledComponent) {
    DummyMixture mix(3, 2);

    mix.discard(0);

    EXPECT_EQ(mix.first_active(), 1);
}

// -----------------------------------------------------------------------------
// MAP
// -----------------------------------------------------------------------------

TEST(MixtureTest, MapReturnsMaximumLikelihoodComponent) {
    DummyMixture mix(3, 2);

    Vector<> x(2);
    x << 0.0, 0.0;

    size_t index = mix.map(x);

    EXPECT_EQ(index, 2);
}

TEST(MixtureTest, MapIgnoresDiscardedComponents) {
    DummyMixture mix(3, 2);

    mix.discard(2);

    Vector<> x(2);
    x << 0.0, 0.0;

    size_t index = mix.map(x);

    EXPECT_EQ(index, 1);
}

TEST(MixtureTest, MapKReturnsBestComponents) {
    DummyMixture mix(4, 2);

    Vector<> x(2);
    x << 0.0, 0.0;

    auto result = mix.map_k(x, 2);

    EXPECT_EQ(result.size(), 2);
    EXPECT_TRUE(result.count(3));
    EXPECT_TRUE(result.count(2));
}

// -----------------------------------------------------------------------------
// Likelihood
// -----------------------------------------------------------------------------

TEST(MixtureTest, NLLMatchesNegativeLogLikelihood) {
    DummyMixture mix(2, 2);

    Matrix<> X(2, 2);
    X << 0.0, 0.0, 1.0, 1.0;

    precision_t ll = mix.log_likelihood(X);
    precision_t nll = mix.nll(X);

    EXPECT_NEAR(nll, -ll / X.rows(), 1e-12);
}

TEST(MixtureTest, LogProbSingleIndexMatchesLogJoint) {
    DummyMixture mix(4, 2);

    // Two observations
    Matrix<> X(2, 2);
    X << 0.0, 0.0, 1.0, 2.0;

    // For each observation, provide a single component index to evaluate
    // Use different components to ensure index is used in log joint
    Matrix<size_t> indices(2, 1);
    indices << 0, 3;

    Vector<> lp = mix.log_prob(X, indices);

    // Expected log-probabilities: -||x||^2 + c (as implemented in DummyMixture)
    precision_t expected0 = -X.row(0).squaredNorm() + static_cast<precision_t>(0);
    precision_t expected1 = -X.row(1).squaredNorm() + static_cast<precision_t>(3);

    EXPECT_NEAR(lp(0), expected0, 1e-12);
    EXPECT_NEAR(lp(1), expected1, 1e-12);
}

TEST(MixtureTest, LogProbMultipleIndicesMatchesLogSumExp) {
    DummyMixture mix(5, 2);

    // Two observations
    Matrix<> X(2, 2);
    X << 0.0, 0.0, 1.0, 2.0;

    // Each observation considers three candidate components
    Matrix<size_t> indices(2, 3);
    indices << 0, 1, 2, 1, 2, 4;

    Vector<> lp = mix.log_prob(X, indices);

    // Helper: compute stable log-sum-exp using Eigen Vector<>
    auto log_sum_exp = [](const Vector<>& vals) -> precision_t {
        precision_t m = vals.maxCoeff();
        precision_t s = (vals.array() - m).exp().sum();
        return m + std::log(s);
    };

    // Compute expected values using DummyMixture's log-joint: -||x||^2 + c
    {
        Vector<> v0(indices.cols());
        for (int j = 0; j < static_cast<int>(indices.cols()); ++j) {
            size_t c = indices(0, j);
            v0(j) = -X.row(0).squaredNorm() + static_cast<precision_t>(c);
        }
        precision_t expected0 = log_sum_exp(v0);
        EXPECT_NEAR(lp(0), expected0, 1e-12);
    }

    {
        Vector<> v1(indices.cols());
        for (int j = 0; j < static_cast<int>(indices.cols()); ++j) {
            size_t c = indices(1, j);
            v1(j) = -X.row(1).squaredNorm() + static_cast<precision_t>(c);
        }
        precision_t expected1 = log_sum_exp(v1);
        EXPECT_NEAR(lp(1), expected1, 1e-12);
    }
}