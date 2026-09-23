/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#include <gtest/gtest.h>

#include <random>

#include "Diagonal.h"
#include "Variational.h"
#include "mock_model.hpp"
#include "omp_helper.h"

namespace {

void set_single_thread() {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

Matrix<> make_data(size_t N, size_t D, size_t seed) {
    Matrix<> X(N, D);
    std::mt19937_64 rng(seed);
    std::normal_distribution<precision_t> normal(0.0, 1.0);
    for (size_t n = 0; n < N; ++n) {
        for (size_t d = 0; d < D; ++d) {
            X(n, d) = normal(rng);
        }
    }
    return X;
}

}  // namespace

class VariationalEMTest : public ::testing::Test {
   protected:
    void SetUp() override { set_single_thread(); }
};

TEST_F(VariationalEMTest, EStepWithMockModelReturnsFiniteObjective) {
    const size_t N = 12;
    const size_t C = 4;
    const size_t D = 2;
    Variational var(N, C, 3, C, 1, 42, false);
    MockModel model(C, D);
    const Matrix<> X = make_data(N, D, 7);

    const precision_t obj = var.E_step(X, model, true, 1.0);
    EXPECT_TRUE(std::isfinite(obj));
    EXPECT_GT(var.number_ljs, 0u);

    for (size_t n = 0; n < N; ++n) {
        precision_t sum = 0;
        for (const auto& [c, p] : var.qs[n]) {
            (void)c;
            sum += p;
            EXPECT_GE(p, 0.0);
        }
        EXPECT_NEAR(sum, 1.0, 1e-10);
    }
}

TEST_F(VariationalEMTest, EMStepWithMockModelRunsEAndM) {
    const size_t N = 10;
    const size_t C = 3;
    const size_t D = 2;
    Variational var(N, C, 2, C, 1, 11, false);
    MockModel model(C, D);
    const Matrix<> X = make_data(N, D, 3);

    const precision_t obj = var.EM_step(X, model, true, true, 1.0);
    EXPECT_TRUE(std::isfinite(obj));
}

TEST_F(VariationalEMTest, EStepWithoutVarParamUpdate) {
    const size_t N = 8;
    const size_t C = 5;
    const size_t C_prime = 2;
    const size_t G = 3;
    Variational var(N, C, C_prime, G, 1, 99, false);
    MockModel model(C, 2);
    const Matrix<> X = make_data(N, 2, 1);

    const auto qs_before = var.qs;
    const auto gc_set_before = var.gc_set;
    const precision_t obj = var.E_step(X, model, false, 1.0);
    EXPECT_TRUE(std::isfinite(obj));

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C_prime; ++c) {
            EXPECT_EQ(var.qs[n][c].first, qs_before[n][c].first);
        }
    }
    for (size_t c = 0; c < C; ++c) {
        for (size_t g = 0; g < G; ++g) {
            EXPECT_EQ(var.gc_set[c][g], gc_set_before[c][g]);
        }
    }
}

TEST_F(VariationalEMTest, EMStepWithDiagonalModel) {
    const size_t N = 15;
    const size_t C = 4;
    const size_t D = 3;
    Variational var(N, C, 2, C, 1, 21, false);
    Diagonal model(C, D, false, "diagonal", false, 1e-3);
    const Matrix<> X = make_data(N, D, 5);

    const precision_t obj = var.EM_step(X, model, true, true, 1.0);
    EXPECT_TRUE(std::isfinite(obj));
    EXPECT_EQ(model.C, C);
}

TEST_F(VariationalEMTest, GetPartitionHardAssignsSingleComponent) {
    const size_t N = 5;
    const size_t C = 4;
    Variational var(N, C, 2, C, 1, 0, false);
    var.qs[0] = {{0, 0.1}, {2, 0.9}, {3, 0.2}};

    var.get_partition_hard();

    size_t assignments = 0;
    for (const auto& thread_part : var.partition) {
        for (size_t c = 0; c < C; ++c) {
            assignments += thread_part[c].size();
        }
    }
    EXPECT_GE(assignments, 1u);
}
