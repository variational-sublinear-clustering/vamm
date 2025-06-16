# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
from __future__ import annotations
import numpy as np
import numpy.typing as npt


def check_X(
    C: int,
    D: int,
    X: npt.NDArray,
    check_C: bool = False,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray:
    """
    Validates and adjusts the input array `X` to ensure it has the correct shape and memory layout for use in C++ code without requiring a copy.
    When `check_C` is `True`, this function enforces a minimum number of data points to match the number of components.

    Parameters
    ----------
    C : int
        The expected number of components. Used to validate that the number of data points is sufficient when `check_C` is `True`.
    D : int
        The expected number of features (dimensionality) for each data point.
    X : npt.NDArray
        The input data array of shape `(N, D)`
    check_C : bool, optional
        If `True`, checks that the number of samples `N` is at least `C`. Default is `False`.
    dtype: npt.DTypeLike or None, optional
        Enforce dtype of `X`. Defaults to None, which keeps the original dtype.

    Returns
    -------
    X : npt.NDArray
        The validated and possibly adjusted data array. The array is guaranteed to be contiguous in memory (row-major order).
    """
    # check ndim
    if X.ndim != 2:
        raise ValueError("Invalid X.ndim != 2")

    # check dimensionality
    if X.shape[1] != D:
        raise ValueError(f"Invalid X.shape[1] != {D}")

    # require at least one data point per component
    if check_C and (X.shape[0] < C):
        raise ValueError("Invalid X.shape[0]")

    dtype = dtype or X.dtype
    # usually row-major is the default, but ensure that it is row-major
    X = np.ascontiguousarray(X, dtype=dtype)
    return X
