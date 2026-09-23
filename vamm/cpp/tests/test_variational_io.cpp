/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#include <gtest/gtest.h>

#include <stdexcept>
#include <unordered_map>

#include "Variational.h"
#include "omp_helper.h"

namespace {

void set_single_thread() {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

}  // namespace

class VariationalIOTest : public ::testing::Test {
   protected:
    void SetUp() override { set_single_thread(); }

    static Variational make_var() { return Variational(4, 5, 3, 4, 1, 42, true, false, "KL"); }
};

TEST_F(VariationalIOTest, QMapRoundTrip) {
    auto var = make_var();
    const auto original = var.q_map(0);

    var.q_in(0, original);
    const auto roundtrip = var.q_map(0);
    EXPECT_EQ(original.size(), roundtrip.size());
    for (const auto& [k, v] : original) {
        EXPECT_DOUBLE_EQ(roundtrip.at(k), v);
    }
}

TEST_F(VariationalIOTest, QInRejectsWrongSize) {
    auto var = make_var();
    std::unordered_map<size_t, precision_t> bad = {{0, 1.0}, {1, 1.0}};
    EXPECT_THROW(var.q_in(0, bad), std::invalid_argument);
}

TEST_F(VariationalIOTest, QInRejectsOutOfRangeComponent) {
    auto var = make_var();
    std::unordered_map<size_t, precision_t> bad = {{0, 1.0}, {1, 1.0}, {99, 1.0}};
    EXPECT_THROW(var.q_in(0, bad), std::out_of_range);
}

TEST_F(VariationalIOTest, QSparseMatrixRoundTrip) {
    auto var = make_var();
    const auto sp = var.q_to_sparse_matrix();
    EXPECT_EQ(sp.rows(), static_cast<int>(var.N));
    EXPECT_EQ(sp.cols(), static_cast<int>(var.C));

    Variational copy = var;
    copy.q_from_sparse_matrix(sp);
    for (size_t n = 0; n < var.N; ++n) {
        EXPECT_EQ(copy.q_map(n).size(), var.q_map(n).size());
    }
}

TEST_F(VariationalIOTest, GcSetSparseMatrixRoundTrip) {
    auto var = make_var();
    const auto sp = var.gc_set_to_sparse_matrix();
    EXPECT_EQ(sp.rows(), static_cast<int>(var.C));
    EXPECT_EQ(sp.cols(), static_cast<int>(var.C));

    Variational copy = var;
    copy.gc_set_from_sparse_matrix(sp);
    for (size_t c = 0; c < var.C; ++c) {
        EXPECT_EQ(copy.gc_set[c].size(), var.gc_set[c].size());
    }
}

TEST_F(VariationalIOTest, GcSetFromSparseRequiresSelfLoop) {
    auto var = make_var();
    auto sp = var.gc_set_to_sparse_matrix();

    // Remove self-entry from row 0
    std::vector<Eigen::Triplet<precision_t>> coeff;
    for (int c = 0; c < sp.rows(); ++c) {
        for (SparseMatrix<>::InnerIterator it(sp, c); it; ++it) {
            if (c == 0 && static_cast<size_t>(it.col()) == 0) continue;
            coeff.emplace_back(c, it.col(), it.value());
        }
    }
    SparseMatrix<> bad(sp.rows(), sp.cols());
    bad.setFromTriplets(coeff.begin(), coeff.end());

    EXPECT_THROW(var.gc_set_from_sparse_matrix(bad), std::invalid_argument);
}

TEST_F(VariationalIOTest, ApproxMapReturnsMaxEntry) {
    auto var = make_var();
    var.qs[0] = {{0, 0.1}, {2, 0.9}, {4, 0.2}};

    const auto [idx, val] = var.approx_map(0);
    EXPECT_EQ(idx, 2u);
    EXPECT_DOUBLE_EQ(val, 0.9);
}

TEST_F(VariationalIOTest, IndicesReturnsArgmaxPerPoint) {
    auto var = make_var();
    var.qs[0] = {{1, 0.2}, {3, 0.8}, {4, 0.1}};
    var.qs[1] = {{0, 0.5}, {2, 0.3}, {4, 0.7}};

    const Vector<size_t> idx = var.indices();
    EXPECT_EQ(idx(0), 3u);
    EXPECT_EQ(idx(1), 4u);
}

TEST_F(VariationalIOTest, EStepSelectTruncatesToCPrime) {
    auto var = make_var();
    var.qs[0] = {{0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 4.0}, {4, 5.0}};
    var.E_step_select();
    EXPECT_EQ(var.qs[0].size(), var.C_prime);
}

TEST_F(VariationalIOTest, QShrinkedRespectsMask) {
    auto var = make_var();
    Vector<bool> mask(5);
    mask << true, false, true, false, true;

    const auto [sp, rev_idx] = var.q_shrinked_to_sparse_matrix(mask);
    EXPECT_EQ(sp.cols(), 3);
    EXPECT_EQ(rev_idx.size(), 3u);
}
