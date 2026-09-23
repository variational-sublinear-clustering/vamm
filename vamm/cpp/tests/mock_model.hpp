/* Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg  */
/* and Artificial Intelligence Lab of the University of Innsbruck.         */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <cmath>
#include <vector>

#include "Variational.h"
#include "omp_helper.h"

/**
 * Minimal Model stub for Variational template methods in unit tests.
 *
 * Uses uniform log-joints derived from data point and component indices so
 * E-step / M-step behaviour is deterministic without a full mixture model.
 */
class MockModel {
   public:
    static constexpr bool loop_order_n = true;

    size_t C;
    size_t C_active;
    size_t D;
    precision_t objective_correction;
    Vector<bool> Mask;
    Vector<> P;
    Vector<> P_log;

    explicit MockModel(size_t C_, size_t D_ = 2)
        : C(C_),
          C_active(C_),
          D(D_),
          objective_correction(0.0),
          Mask(C_),
          P(C_),
          P_log(C_) {
        Mask.fill(true);
        P.fill(1.0 / static_cast<precision_t>(C_));
        P_log = P.array().log().matrix();
    }

    void auxiliary() {}

    template <class Lmbd>
    void E_step_allocate(const Lmbd& lmbd) const {
        lmbd();
    }

    void E_step_log_joint(cRef<Vector<>> x, const size_t c, precision_t& log_prob) const {
        // Deterministic pseudo log-joint: higher for lower-index components.
        log_prob = -static_cast<precision_t>(c) - x.squaredNorm();
        log_prob += P_log[c];
    }

    template <class Lmbd>
    void M_step_allocate(const Lmbd& lmbd) const {
        Vector<> acc(D);
        lmbd(acc);
    }

    void M_step_reset(size_t c, Vector<>& acc) {
        acc.fill(0);
        P[c] = 0;
    }

    void M_step_accumulate(cRef<Vector<>> x, precision_t q_nc, size_t c, Vector<>& acc) {
        acc += q_nc * x;
        P[c] += q_nc;
    }

    void M_step_update(size_t c, Vector<>& /*acc*/) {
        if (P[c] <= 0) {
            Mask[c] = false;
            --C_active;
        }
    }

    void M_step_finalize(size_t /*N*/) { P_adjust(); }

    void relocate_discarded_components(std::vector<size_t>& /*from*/, std::vector<size_t>& /*to*/,
                                       Random<std::mt19937_64>& /*rng*/) {}

    void P_adjust() {
        precision_t sum = 0;
        for (size_t c = 0; c < C; ++c) {
            if (Mask[c]) sum += P[c];
        }
        if (sum <= 0) return;
        for (size_t c = 0; c < C; ++c) {
            if (Mask[c]) {
                P[c] /= sum;
                P_log[c] = std::log(P[c]);
            }
        }
    }
};
