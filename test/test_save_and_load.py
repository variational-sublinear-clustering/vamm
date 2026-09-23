"""Integration tests for the Python Variational wrapper around cppvamm."""

from __future__ import annotations

import numpy as np

from vamm.utils.save_and_load import save_params, load_params


class TestSaveAndLoad:
    def test_save_load_params(self, mfa_model, tmp_path, small_gaussian_data):

        mfa_model._initialize(X=small_gaussian_data)
        path = tmp_path / "model"

        save_params(mfa_model, path)

        assert (tmp_path / "model.h5").exists()

        loaded = load_params(path.with_suffix(".h5"))

        np.testing.assert_allclose(loaded.prior, mfa_model.prior)
        np.testing.assert_allclose(loaded.means, mfa_model.means)
        np.testing.assert_allclose(loaded.A, mfa_model.A)
        np.testing.assert_allclose(loaded.variance, mfa_model.variance)

        assert loaded.covariance_type == mfa_model.covariance_type
        assert loaded.shared == mfa_model.shared
