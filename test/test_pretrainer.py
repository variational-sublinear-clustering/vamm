import pytest
import importlib
import numpy as np
from unittest.mock import MagicMock, patch

from vamm import Gaussian, Variational

gaussian_module = importlib.import_module("vamm.models.Gaussian")


@pytest.fixture
def gaussian():
    """Create a Gaussian model without running any training."""
    return Gaussian(
        C=3,
        D=2,
        covariance_type="diagonal",
        flat_prior=False,
        init_prior=np.array([1 / 3, 1 / 3, 1 / 3]),
        init_means=np.zeros((3, 2)),
        init_variance=np.ones((3, 2)),
    )


def test_pretrainer_creates_isotropic_shared_gaussian(gaussian):
    """The pretrainer should use an isotropic, shared Gaussian model."""
    X = np.random.default_rng(42).normal(size=(20, 2))

    fake_pretrainer = MagicMock()
    fake_pretrainer.means = gaussian.means.copy()
    fake_pretrainer.mask = np.array([True, True, False])
    fake_pretrainer.prior = np.array([0.5, 0.5, 0.0])
    fake_pretrainer.variance = np.array([[2.0, 3.0]] * 3)

    fake_em = MagicMock()
    fake_em._log = [{"objective": 1.0}]
    fake_pretrainer.em = fake_em

    with patch.object(
        gaussian_module,
        "Gaussian",
        return_value=fake_pretrainer,
    ) as GaussianMock:
        gaussian._pretrainer(
            X=X,
            limit=[10, 20],
            rng=42,
            eps=[1e-3, 1e-4],
            C_prime=2,
            G=2,
            E=1,
            relocate_discarded=False,
            indices=np.array([0, 5, 10]),
            verbose=False,
        )

    GaussianMock.assert_called_once()

    kwargs = GaussianMock.call_args.kwargs

    assert kwargs["C"] == gaussian.C
    assert kwargs["D"] == gaussian.D
    assert kwargs["covariance_type"] == "isotropic"
    assert kwargs["shared"] is True
    assert kwargs["flat_prior"] is True

    np.testing.assert_array_equal(
        kwargs["init_means"],
        gaussian.means,
    )
    np.testing.assert_array_equal(
        kwargs["init_variance"],
        np.ones(1),
    )


def test_pretrainer_passes_arguments_to_fit(gaussian):
    """All relevant arguments should be forwarded to pretrainer.fit()."""
    X = np.random.default_rng(42).normal(size=(20, 2))
    indices = np.array([0, 5, 10])

    fake_pretrainer = MagicMock()
    fake_pretrainer.means = gaussian.means.copy()
    fake_pretrainer.mask = np.ones(3, dtype=bool)
    fake_pretrainer.prior = np.ones(3) / 3
    fake_pretrainer.variance = np.array([[2.0, 3.0]] * 3)

    fake_em = MagicMock()
    fake_em._log = [{"i": 0}]
    fake_pretrainer.em = fake_em

    with patch.object(
        gaussian_module,
        "Gaussian",
        return_value=fake_pretrainer,
    ):
        gaussian._pretrainer(
            X=X,
            em="existing-em",
            limit=[10, 20],
            rng=42,
            eps=[1e-3, 1e-4],
            C_prime=2,
            G=2,
            E=1,
            relocate_discarded=False,
            indices=indices,
            verbose=False,
        )

    fake_pretrainer.fit.assert_called_once_with(
        X=X,
        em="existing-em",
        limit=[10, 20],
        rng=42,
        eps=[1e-3, 1e-4],
        C_prime=2,
        G=2,
        E=1,
        relocate_discarded=False,
        indices=indices,
        verbose=False,
    )


def test_pretrainer_copies_training_state(gaussian):
    """The training state of the pretrainer should be copied to the model."""
    X = np.random.default_rng(42).normal(size=(20, 2))

    expected_mask = np.array([True, True, False])
    expected_prior = np.array([0.4, 0.6, 0.0])
    expected_means = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    expected_variance = np.array([[2.0, 3.0]] * 3)

    fake_em = MagicMock()
    fake_em._log = [{"i": 0}, {"i": 1}]

    fake_pretrainer = MagicMock()
    fake_pretrainer.em = fake_em
    fake_pretrainer.mask = expected_mask
    fake_pretrainer.prior = expected_prior
    fake_pretrainer.means = expected_means
    fake_pretrainer.variance = expected_variance

    with patch.object(
        gaussian_module,
        "Gaussian",
        return_value=fake_pretrainer,
    ):
        gaussian._pretrainer(X=X, C_prime=2, G=2, E=1, verbose=False)

    assert gaussian.em is fake_em
    assert gaussian.em._log == fake_em._log

    np.testing.assert_array_equal(gaussian.mask, expected_mask)
    np.testing.assert_array_equal(gaussian.prior, expected_prior)
    np.testing.assert_array_equal(gaussian.means, expected_means)

    # The original model has a variance with shape (C, D).
    # It should receive the variance of the first pretrainer component.
    np.testing.assert_array_equal(
        gaussian.variance,
        expected_variance[0, 0],
    )


def test_pretrainer_uses_first_pretrainer_variance(gaussian):
    """The variance assignment should use pretrainer.variance[0, 0]."""
    X = np.random.default_rng(42).normal(size=(20, 2))

    fake_pretrainer = MagicMock()
    fake_pretrainer.em = MagicMock()
    fake_pretrainer.em._log = []

    fake_pretrainer.mask = np.ones(3, dtype=bool)
    fake_pretrainer.prior = np.ones(3) / 3
    fake_pretrainer.means = gaussian.means.copy()

    # Deliberately use different values to verify exactly which
    # element is used by the implementation.
    fake_pretrainer.variance = np.array(
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )

    with patch.object(
        gaussian_module,
        "Gaussian",
        return_value=fake_pretrainer,
    ):
        gaussian._pretrainer(X=X, C_prime=2, G=2, E=1, verbose=False)

    # The variance of the first pretrainer component
    # should be copied to all components of the model.
    expected_variance = fake_pretrainer.variance[0, 0]

    np.testing.assert_array_equal(
        gaussian.variance,
        np.tile(
            expected_variance,
            (gaussian.C, gaussian.D),
        ),
    )


def test_pretrainer_continues_training_with_same_em():
    """Pretraining and final training use the same Variational EM instance."""

    rng = np.random.default_rng(42)

    X = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.3, size=(20, 2)),
            rng.normal(loc=[4.0, 4.0], scale=0.3, size=(20, 2)),
        ]
    )

    model = Gaussian(
        C=2,
        D=2,
        covariance_type="diagonal",
        flat_prior=False,
        init_prior="flat",
        init_means="random",
        init_variance="data_variance",
    )
    indices = model._initialize(X=X, rng=rng, verbose=False)

    # Create the EM object explicitly.
    em = Variational(
        N=X.shape[0],
        C=model.C,
        C_prime=2,
        G=2,
        E=1,
        relocate_discarded=True,
        indices=indices,
        rng=42,
    )

    objective, log = model.fit(
        X=X,
        em=em,
        limit=[2, 2],
        rng=42,
        eps=[1e-4, 1e-4],
        C_prime=2,
        G=2,
        E=1,
        relocate_discarded=True,
        indices=indices,
        use_pretrainer=True,
        verbose=False,
    )

    assert model.em is em
    assert len(em._log) > 0
    assert len(log) == len(em._log)
    assert log["i"].eq(0).sum() == 3
