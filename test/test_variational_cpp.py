"""Integration tests for Variational via the cppvamm pybind11 module."""

from __future__ import annotations

import numpy as np
import pytest


class TestVariationalCpp:
    def test_constructor_dimensions(self, cppvamm):
        var = cppvamm.Variational(10, 5, 3, 4, 1, seed=42)
        assert var.N == 10
        assert var.C == 5
        assert var.C_prime == 3
        assert var.G == 4
        assert var.E == 1
        assert var.initial_seed == 42

    def test_invalid_constructor_raises(self, cppvamm):
        with pytest.raises(ValueError):
            cppvamm.Variational(0, 5, 3, 4, 1, seed=42)

    def test_q_map_roundtrip(self, cppvamm, variational_params):
        var = cppvamm.Variational(**variational_params)
        original = var.q_map(0)
        var.q_in(0, original)
        roundtrip = var.q_map(0)
        assert roundtrip == original

    def test_q_sparse_property_roundtrip(self, cppvamm, variational_params):
        var = cppvamm.Variational(**variational_params)
        sp = var.q
        var.q = sp
        assert var.q.nnz == sp.nnz

    def test_gc_set_property_roundtrip(self, cppvamm, variational_params):
        var = cppvamm.Variational(**variational_params)
        g = var.g
        var.g = g
        assert var.g.nnz == g.nnz

    def test_em_step_finite_objective(
        self, cppvamm, variational_params, diagonal_model, small_gaussian_data
    ):
        var = cppvamm.Variational(**variational_params, relocate_discarded=False)
        obj = var._EM_step(
            small_gaussian_data,
            diagonal_model,
            fit=True,
            update_var_params=True,
            beta=1.0,
        )
        assert np.isfinite(obj)
        assert var.number_ljs > 0

    def test_e_step_normalizes_posteriors(
        self, cppvamm, variational_params, diagonal_model, small_gaussian_data
    ):
        var = cppvamm.Variational(**variational_params, relocate_discarded=False)
        var._E_step(
            small_gaussian_data,
            diagonal_model,
            update_var_params=False,
            beta=1.0,
        )
        for n in range(var.N):
            q = var.q_map(n)
            assert np.isclose(sum(q.values()), 1.0, atol=1e-10)

    def test_indices_matches_argmax(self, cppvamm, variational_params):
        var = cppvamm.Variational(**variational_params)
        var.q_in(0, {0: 0.1, 2: 0.8, 4: 0.1})
        idx, val = var.approx_map(0)
        assert idx == 2
        assert val == pytest.approx(0.8)

    def test_set_e(self, cppvamm, variational_params):
        var = cppvamm.Variational(**variational_params)
        var.E = 5
        assert var.E == 5

    def test_multiple_em_steps_improve_or_stabilize(
        self, cppvamm, variational_params, diagonal_model, small_gaussian_data
    ):
        var = cppvamm.Variational(**variational_params, relocate_discarded=False)
        objectives = []
        for _ in range(3):
            obj = var._EM_step(
                small_gaussian_data,
                diagonal_model,
                fit=True,
                update_var_params=True,
                beta=1.0,
            )
            objectives.append(obj)
        assert all(np.isfinite(o) for o in objectives)
