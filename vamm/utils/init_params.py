# Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg 
# and Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vamm.utils import afkmc2 as cpp_afkmc2
from vamm.utils import stats


def flat(
    C: int,
    dtype: npt.DTypeLike,
    verbose: bool = False,
) -> npt.NDArray:
    """
    Initializes a flat prior distribution.

    This function creates a flat (uniform) prior distribution for `C` components, where each component has an equal probability.

    Parameters
    ----------
    C : int
        The number of components in the distribution.
    dtype : dtype
        The desired data type for the output array (e.g., `np.float32`, `np.float64`).
    verbose : bool, optional
        If `True`, prints a message indicating that the prior is being initialized. Default is `False`.

    Returns
    -------
    ndarray
        An array of shape `(C,)` containing the flat prior distribution, where each entry is equal to `1/C`.
    """
    if verbose:
        print("Initialize Prior with 1/C...", flush=True)
    return np.ones(C, dtype=dtype) / C


def afkmc2(
    X: npt.NDArray,
    C: int,
    chain: int = 10,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
):
    """
    Initializes component centers (means) using the AFK-MC² algorithm.

    This function selects `C` initial centers from the dataset `X` using the AFK-MC² algorithm, a variant of
    the k-means++ initialization, which is designed to provide a more efficient sampling process.

    Parameters
    ----------
    X : npt.NDArray
        The dataset from which the centers are to be initialized.
    C : int
        The number of centers (components) to initialize.
    chain : int, optional
        The length of the Markov chain. Default is 10.
    rng : np.random.Generator, int, or None, optional
        The random number generator or seed used for initialization. If `None`, the default random number generator is used.
    verbose : bool, optional
        If `True`, prints a message indicating that the means is being initialized. Default is `False`.

    Returns
    -------
    centers : npt.NDArray
        An array containing the initialized centers.
    indices : npt.NDArray
        An array containing the indices of the selected centers from the dataset `X`.

    References
    ----------
    Bachem, O., Lucic, M., Hassani, H. & Krause, A. (2016) Fast and Provably Good Seedings for k-Means
    Advances in Neural Information Processing Systems , Vol. 29
    """
    rng = np.random.default_rng(rng)
    seed = rng.integers(low=0, high=np.iinfo(np.uint32).max)
    indices = np.empty(shape=(C,), dtype=np.uint64)
    if verbose:
        print("Initialize Means with afkmc2...", flush=True)

    cpp_afkmc2(X, C, seed, chain, indices)
    return X[indices], indices


def random_data(
    X: npt.NDArray,
    C: int,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
):
    """
    Initializes component centers (means) by randomly selecting data points from the dataset.

    This function selects `C` random data points from the dataset `X` to be used as the initial centers for componenting.

    Parameters
    ----------
    X : npt.NDArray
        The dataset from which the centers are to be initialized.
    C : int
        The number of centers (components) to initialize.
    rng : np.random.Generator, int, or None, optional
        The random number generator or seed used for initialization. If `None`, the default random number generator is used.
    verbose : bool, optional
        If `True`, prints a message indicating that the means are being initialized. Default is `False`.

    Returns
    -------
    centers : npt.NDArray
        An array containing the initialized centers.
    indices : npt.NDArray
        An array containing the indices of the selected centers from the dataset `X`.
    """
    rng = np.random.default_rng(rng)
    if verbose:
        print("Initialize Means with random data points...", flush=True)
    indices = rng.choice(a=X.shape[0], size=C, replace=False)
    return X[indices], indices


def uniform(
    C: int,
    D: int,
    H: int,
    low: float = 0.0,
    high: float = 1.0,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
):
    """
    Initializes an 3D array with values drawn from a uniform distribution.

    This function generates a `(C, D, H)` shaped array `A` where each element is sampled uniformly between `low` and `high`.

    Parameters
    ----------
    C : int
        The size of the first dimension of the output array.
    D : int
        The size of the second dimension of the output array.
    H : int
        The size of the third dimension of the output array.
    low : float, optional
        The lower bound of the uniform distribution. Default is `0.0`.
    high : float, optional
        The upper bound of the uniform distribution. Default is `1.0`.
    rng : np.random.Generator, int, or None, optional
        The random number generator or seed used to generate the uniform distribution. If `None`, the default random number generator is used.
    verbose : bool, optional
        If `True`, prints a message indicating that `A` is being initialized. Default is `False`.

    Returns
    -------
    ndarray
        An array of shape `(C, D, H)` filled with uniformly distributed random values.
    """
    rng = np.random.default_rng(rng)
    if verbose:
        print(f"Initialize A uniformly between {low} and {high}...", flush=True)
    return rng.uniform(low=low, high=high, size=(C, D, H))


def data_variance(
    X: npt.NDArray,
    C: int,
    covariance_type: str,
    shared: bool,
    means: npt.NDArray | None = None,
    indices: npt.NDArray | None = None,
    verbose: bool = False,
):
    """
    Initializes variance parameters based on the data variance.

    This function computes the variance of the dataset `X` and uses it to initialize the variance parameters according to the specified `covariance_type`.
    The initialization can be done with or without reference to provided component means.
    When `means` and `indices` are provided, the variance is computed per component; otherwise, a global variance is computed and repeated across components.

    Parameters
    ----------
    X : npt.NDArray
        The dataset from which the variances are to be initialized.
    C : int
        The number of components.
    covariance_type : str
        The type of covariance structure. Must be one of `"isotropic"`, `"diagonaltied"`, `"mfatied"`, `"diagonal"`, or `"mfa"`.
    means : npt.NDArray or None, optional
        An array of shape `(C, D)` containing the means of the components. If `None`, the variance is computed without reference to component means. Default is `None`.
    indices : npt.NDArray or None, optional
        An array of shape `(N,)` containing the component assignments for each data point in `X`. This is required if `means` is provided. Default is `None`.
    verbose : bool, optional
        If `True`, prints a message indicating that the variances are being initialized. Default is `False`.

    Returns
    -------
    param : npt.NDArray
        The initialized variance parameters. The shape of the output array depends on the `covariance_type`:
        - `"isotropic"`: `(1,)`
        - `"diagonaltied"` `(D,)`
        - `"mfatied"`: `(D,)`
        - `"diagonal"`: `(C, D)`
        - `"mfa"`: `(C, D)`
    """
    N, D = X.shape

    assert covariance_type in ("isotropic", "diagonal", "mfa", "full")

    if verbose:
        print(f"Initialize variance with the data variance...", flush=True)

    if covariance_type in ("isotropic",):
        var = np.array([np.var(X)])
        return var if shared else np.tile(var, (C,))

    if covariance_type in ("diagonal", "mfa"):
        var = stats.var(X, axis=0)
        return var if shared else np.tile(var, (C, 1))

    if covariance_type in ("full",):
        var = np.cov(X.T)
        return var if shared else np.tile(var, (C, 1, 1))

    # if means is not None:
    #     assert len(shp) == 2
    #     assert len(indices) == N
    #     param = np.array([stats.var(X[indices == c], axis=0) for c in range(C)])


def random_variance(
    X: npt.NDArray,
    C: int,
    covariance_type: str,
    shared: bool,
    reg_covar: float = 1e-3,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
):
    """
    Initializes variance parameters randomly.

    Parameters
    ----------
    X : npt.NDArray
        The dataset from which the variances are to be initialized.
    C : int
        The number of components.
    covariance_type : str
        The type of covariance structure. Must be one of `"isotropic"`, `"diagonaltied"`, `"mfatied"`, `"diagonal"`, `"mfa"` or `"full"`.
    rng : np.random.Generator, int, or None, optional
        The random number generator or seed used to generate the uniform distribution. If `None`, the default random number generator is used.
    verbose : bool, optional
        If `True`, prints a message indicating that the variances are being initialized. Default is `False`.

    Returns
    -------
    param : npt.NDArray
        The initialized variance parameters. The shape of the output array depends on the `covariance_type`:
        - `"isotropic"`: `(1,)`
        - `"diagonaltied"` `(D,)`
        - `"mfatied"`: `(D,)`
        - `"diagonal"`: `(C, D)`
        - `"mfa"`: `(C, D)`
        - `"full"`: `(C, D, D)`
    """
    rng = np.random.default_rng(rng)
    N, D = X.shape
    variance_shape = {
        "isotropic": {"shared": (1,), "not_shared": (C,)},
        "diagonal": {"shared": (D,), "not_shared": (C, D)},
        "mfa": {"shared": (D,), "not_shared": (C, D)},
        "full": {"shared": (D, D), "not_shared": (C, D, D)},
    }
    assert covariance_type in variance_shape.keys()
    _shared = "shared" if shared else "not_shared"
    shp = variance_shape[covariance_type][_shared]

    if verbose:
        print(f"Initialize variance with random values...", flush=True)

    var = rng.normal(size=(shp))

    if covariance_type in ("full",):
        var = (
            var @ var.transpose()
            if shared
            else np.array([var[c] @ var[c].transpose() for c in range(C)])
        )  # TODO: improve this
    else:
        np.square(var, out=var)
        np.maximum(var, reg_covar, out=var)
    return var
