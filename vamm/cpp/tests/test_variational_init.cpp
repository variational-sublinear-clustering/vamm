/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#include <gtest/gtest.h>

#include <set>
#include <stdexcept>

#include "Variational.h"
#include "omp_helper.h"

namespace {

void set_single_thread() {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

}  // namespace

class VariationalInitTest : public ::testing::Test {
   protected:
    void SetUp() override { set_single_thread(); }
};

TEST_F(VariationalInitTest, ConstructorSetsDimensions) {
    Variational var(10, 5, 3, 4, 1, 42);
    EXPECT_EQ(var.N, 10u);
    EXPECT_EQ(var.C, 5u);
    EXPECT_EQ(var.C_prime, 3u);
    EXPECT_EQ(var.G, 4u);
    EXPECT_EQ(var.E, 1u);
    EXPECT_EQ(var.initial_seed, 42u);
}

TEST_F(VariationalInitTest, RejectsInvalidN) {
    EXPECT_THROW(Variational(0, 5, 3, 4, 1, 42), std::invalid_argument);
}

TEST_F(VariationalInitTest, RejectsInvalidC) {
    EXPECT_THROW(Variational(10, 0, 3, 4, 1, 42), std::invalid_argument);
}

TEST_F(VariationalInitTest, RejectsInvalidCPrime) {
    EXPECT_THROW(Variational(10, 5, 0, 4, 1, 42), std::invalid_argument);
    EXPECT_THROW(Variational(10, 5, 6, 4, 1, 42), std::invalid_argument);
}

TEST_F(VariationalInitTest, RejectsInvalidG) {
    EXPECT_THROW(Variational(10, 5, 3, 0, 1, 42), std::invalid_argument);
    EXPECT_THROW(Variational(10, 5, 3, 6, 1, 42), std::invalid_argument);
}

TEST_F(VariationalInitTest, RandomInitHasCPrimeComponentsPerPoint) {
    const size_t N = 20;
    const size_t C = 8;
    const size_t C_prime = 3;
    Variational var(N, C, C_prime, 5, 1, 123);

    for (size_t n = 0; n < N; ++n) {
        EXPECT_EQ(var.qs[n].size(), C_prime);
        std::set<size_t> components;
        for (const auto& [c, w] : var.qs[n]) {
            (void)w;
            EXPECT_LT(c, C);
            components.insert(c);
        }
        EXPECT_EQ(components.size(), C_prime);
    }
}

TEST_F(VariationalInitTest, FullCPrimeInitializesAllComponents) {
    const size_t N = 5;
    const size_t C = 4;
    Variational var(N, C, C, C, 1, 7);

    for (size_t n = 0; n < N; ++n) {
        EXPECT_EQ(var.qs[n].size(), C);
    }
}

TEST_F(VariationalInitTest, IndexInitSizeN) {
    const size_t N = 6;
    const size_t C = 5;
    Vector<size_t> indices(N);
    for (size_t n = 0; n < N; ++n) indices(n) = n % C;

    Variational var(N, C, 3, 4, 1, 99, indices);
    for (size_t n = 0; n < N; ++n) {
        EXPECT_EQ(var.qs[n].size(), 3u);
        bool has_seed = false;
        for (const auto& [c, w] : var.qs[n]) {
            (void)w;
            if (c == indices(n)) has_seed = true;
        }
        EXPECT_TRUE(has_seed);
    }
}

TEST_F(VariationalInitTest, IndexInitSizeC) {
    const size_t N = 6;
    const size_t C = 5;
    Vector<size_t> indices(C);
    for (size_t c = 0; c < C; ++c) indices(c) = c % N;

    Variational var(N, C, 3, 4, 1, 99, indices);
    for (size_t c = 0; c < C; ++c) {
        size_t n = indices(c);
        EXPECT_EQ(var.qs[n].size(), 3u);
        bool has_seed = false;
        for (const auto& [k, w] : var.qs[n]) {
            (void)w;
            if (c == k) has_seed = true;
        }
        EXPECT_TRUE(has_seed);
    }
}

TEST_F(VariationalInitTest, IndexInitRejectsWrongSize) {
    Vector<size_t> indices(7);
    EXPECT_THROW(Variational(6, 5, 3, 4, 1, 99, indices), std::invalid_argument);
}

TEST_F(VariationalInitTest, GcSetHasGNeighbors) {
    const size_t C = 6;
    const size_t G = 4;
    Variational var(10, C, 3, G, 1, 55);

    for (size_t c = 0; c < C; ++c) {
        EXPECT_EQ(var.gc_set[c].size(), G);
        bool contains_self = false;
        for (size_t g : var.gc_set[c]) {
            if (g == c) contains_self = true;
        }
        EXPECT_TRUE(contains_self);
    }
}

TEST_F(VariationalInitTest, EuclideanSimilarityMeasure) {
    Variational var(5, 4, 2, 3, 1, 1, true, false, "Euclidean");
    EXPECT_EQ(var.compute_relevance, compute_euclidean);
}

TEST_F(VariationalInitTest, KLSimilarityMeasureDefault) {
    Variational var(5, 4, 2, 3, 1, 1);
    EXPECT_EQ(var.compute_relevance, compute_KL_divergence);
}
