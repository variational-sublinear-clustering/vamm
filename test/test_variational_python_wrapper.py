"""Integration tests for the Python Variational wrapper around cppvamm."""

from __future__ import annotations

import numpy as np
import pytest

objectives = {}


class TestVariationalPythonWrapper:
    def test_wrapper_exposes_cpp_attributes(self, vamm_variational_class):
        Variational = vamm_variational_class
        var = Variational(N=10, C=5, C_prime=3, G=4, E=1, rng=42)
        assert var.N == 10
        assert var.C == 5
        assert var.C_prime == 3
        assert var.G == 4
        assert var.E == 1

    def test_wrapper_fit_runs(
        self, vamm_variational_class, cppvamm, small_gaussian_data
    ):
        pytest.importorskip("vamm")
        from vamm import Gaussian

        gaussian_model = Gaussian(C=6, D=3, covariance_type="diagonal", shared=False)
        gaussian_model._initialize(X=small_gaussian_data)
        Variational = vamm_variational_class
        var = Variational(
            N=small_gaussian_data.shape[0], C=6, C_prime=3, G=4, E=1, rng=0
        )
        obj, log = var.fit(
            model=gaussian_model, X=small_gaussian_data, limit=2, verbose=False
        )
        # assert np.isclose(obj, 3.4515013857602153)
        assert np.isfinite(obj)
        assert len(log) >= 1

    def test_wrapper_delegates_q_map(self, vamm_variational_class):
        Variational = vamm_variational_class
        var = Variational(N=5, C=4, C_prime=2, G=3, E=1, rng=1)
        q = var.q_map(0)
        assert isinstance(q, dict)
        assert len(q) == 2

    def test_wrapper_and_cpp_same_dimensions(self, vamm_variational_class, cppvamm):
        Variational = vamm_variational_class
        py_var = Variational(N=8, C=4, C_prime=2, G=3, E=1, rng=99)
        cpp_var = cppvamm.Variational(8, 4, 2, 3, 1, seed=py_var.initial_seed)
        assert py_var.N == cpp_var.N
        assert py_var.C == cpp_var.C
        assert py_var.C_prime == cpp_var.C_prime
