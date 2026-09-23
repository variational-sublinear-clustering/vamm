/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "Variational.h"

TEST(VariationalPure, ComputeKLDivergence) {
    EXPECT_DOUBLE_EQ(compute_KL_divergence(2.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(compute_KL_divergence(-1.0, -3.0), 2.0);
}

TEST(VariationalPure, ComputeEuclidean) {
    EXPECT_DOUBLE_EQ(compute_euclidean(99.0, 2.5), -2.5);
}

TEST(VariationalPure, NormAveragesCandidates) {
    std::vector<Triplet> candidates = {
        {0, 4, 8.0},
        {1, 2, 6.0},
    };
    Vector<> P_log(2);
    P_log.fill(0.0);
    norm(candidates, P_log, 0);
    EXPECT_DOUBLE_EQ(candidates[0].sum_ljs, 2.0);
    EXPECT_DOUBLE_EQ(candidates[1].sum_ljs, 3.0);
}

TEST(VariationalPure, RemovePriorNormAdjustsByPrior) {
    std::vector<Triplet> candidates = {
        {0, 2, 4.0},
        {1, 2, 6.0},
    };
    Vector<> P_log(2);
    P_log << std::log(0.25), std::log(0.75);
    remove_prior_norm(candidates, P_log, 0);
    EXPECT_DOUBLE_EQ(candidates[0].sum_ljs, 2.0 + std::log(0.25) - std::log(0.25));
    EXPECT_DOUBLE_EQ(candidates[1].sum_ljs, 3.0 + std::log(0.75) - std::log(0.25));
}

TEST(VariationalPure, NormalizeProducesValidDistribution) {
    q_t q = {{0, 0.0}, {1, 1.0}, {2, 0.5}};
    precision_t lim = 0;
    precision_t sum = 0;
    normalize(q, 1.0, lim, sum);

    precision_t total = 0;
    for (const auto& [c, p] : q) {
        (void)c;
        total += p;
        EXPECT_GE(p, 0.0);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
    EXPECT_GT(sum, 0.0);
    EXPECT_TRUE(std::isfinite(lim));
}

TEST(VariationalPure, NormalizeThrowsOnNaN) {
    q_t q = {{0, 0.0}, {1, std::numeric_limits<precision_t>::quiet_NaN()}};
    precision_t lim = 0;
    precision_t sum = 0;
    EXPECT_THROW(normalize(q, 1.0, lim, sum), std::runtime_error);
}

TEST(VariationalPure, NormalizeRespectsBeta) {
    q_t q = {{0, 0.0}, {1, 2.0}};
    precision_t lim_a = 0;
    precision_t sum_a = 0;
    q_t q_copy = q;
    normalize(q, 2.0, lim_a, sum_a);

    precision_t lim_b = 0;
    precision_t sum_b = 0;
    normalize(q_copy, 0.5, lim_b, sum_b);

    EXPECT_NE(q[0].second, q_copy[0].second);
}
