# Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg
# and Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd

from vamm.utils.sanity import check_X
from vamm.utils.init_params import flat, afkmc2, random_data
from ..em.Variational import Variational


class Models:
    """
    Base wrapper class for mixture models.

    Parameters
    ----------
    cpp : Any
        C++ object of the underlying mixture model.
    """

    def __init__(
        self,
        cpp,
        init_prior: npt.NDArray | str = "flat",
        init_means: npt.NDArray | str = "afkmc2",
        flat_prior: bool = False,
    ) -> None:

        self.__dict__["_cpp"] = cpp
        self.__dict__["flat_prior"] = flat_prior

        self.em = None

        self._init = {"prior": None, "means": None}

        if type(init_prior) is np.ndarray and not self.flat_prior:
            self.prior = init_prior
        elif type(init_prior) is str and not self.flat_prior:
            self.prior[:] = np.nan
            self._init["prior"] = init_prior

        if type(init_means) is np.ndarray:
            self.means = init_means
        elif type(init_means) is str and not self.flat_prior:
            self.means[:] = np.nan
            self._init["means"] = init_means

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set attribute value.

        This method sets the value of an attribute. If the attribute exists in the C++ object and
        is not already defined in the Python object, it sets the value of the attribute in the C++ object.
        Otherwise, it sets the value in the Python object.

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : Any
            Value to be set for the attribute.

        Returns
        -------
        None
        """
        if hasattr(self._cpp, name) and name not in dir(self):
            setattr(self._cpp, name, value)
        else:
            super(Models, self).__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """
        The value of an attribute from the C++ object.

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        -------
        Any
            Value of the attribute.
        """
        return self._cpp.__getattribute__(name)

    def _initialize(
        self,
        X: npt.NDArray,
        indices: npt.NDArray | None = None,
        rng: np.random.Generator | int | None = None,
        verbose: bool = False,
    ):
        rng = np.random.default_rng(rng)
        if np.isnan(self.prior).all() and self._init["prior"] is not None:
            assert self._init["prior"] in (
                "flat",
            ), "Initialization method for prior unknown."
            if not self.flat_prior:
                self.prior = flat(self.C, dtype=self.prior.dtype, verbose=verbose)

        if np.isnan(self.means).all() and self._init["means"] is not None:
            assert self._init["means"] in (
                "afkmc2",
                "random",
            ), "Initialization method for means unknown."
            self.means, indices = (
                afkmc2(X, self.C, rng=rng, verbose=verbose)
                if self._init["means"] == "afkmc2"
                else random_data(X, self.C, rng=rng, verbose=verbose)
            )
        return indices

    def fit(
        self,
        X: npt.NDArray,
        em: Variational = None,
        limit: list[int] | int | None = 1000,
        rng: np.random.generator | int | None = None,
        eps: list[float] | float = 1.0e-4,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        relocate_discarded: bool = True,
        hard: bool = False,
        sim_measure: str = "KL",
        indices: npt.NDArray | None = None,
        use_pretrainer: bool = False,
        verbose: bool = False,
    ) -> tuple[float, pd.DataFrame]:
        """
        Fit the model to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        limit : int, list[int] or None, optional
            Convergence limit(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. None means no limit. Defaults to 1000.
        rng : np.random.Generator, int or None, optional
            Random number generator or seed. For None, a random seed is used.
        eps : float or list[float]
            Convergence threshold(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. Defaults to 1.0e-4.
        C_prime : int, optional
            Number of non-zero elements in truncated posterior. Defaults to 3.
        G : int, optional
            Component neighborhood size. Defaults to 15.
        E : int, optional
            Number of randomly added components. Defaults to 1.
        hard : bool, optional
            Whether to use hard assignment in the M-step. Defaults to False.
        sim_measure : {"KL","Euclidean"}, optional
            Whether to use 'KL' (Kullback-Leibler divergence) or 'Euclidean' distance as the similarity measure
            for updating the neighborhood set. Defaults to 'KL'.
        indices : np.ndarray or None, optional
            Indices of data points uses as initial component centers. Used for initializing the K-Sets and sets g_c. Defaults to None.
        use_pretrainer : bool, optional
            Whether to use pretraining. Defaults to False.
        verbose : bool, optional
            Whether to print progress messages. Defaults to False.

        Returns
        -------
        objective : float
            the final objective value.
        log : pd.DataFrame
            a DataFrame with training history.
        """
        self.em = em
        if use_pretrainer:
            self._pretrainer(
                X=X,
                em=em,
                limit=limit,
                rng=rng,
                eps=eps,
                C_prime=C_prime,
                G=G,
                E=E,
                relocate_discarded=relocate_discarded,
                indices=indices,
                verbose=verbose,
            )
            limit = limit if isinstance(limit, (list, tuple)) else [limit]
            limit = [0, limit[-1]]  # avoid warmup after pretrainer

        X = check_X(self.C, self.D, X, check_C=True, dtype=self.dtype)
        indices = self._initialize(X=X, indices=indices, rng=rng, verbose=verbose)
        if indices is not None:
            assert np.unique(indices).shape[0] == self.C, "indices are not unique!"

        if self.em is None:
            self.em = Variational(
                N=X.shape[0],
                C=self.C,
                C_prime=C_prime,
                G=G,
                E=E,
                rng=rng,
                indices=indices,
                relocate_discarded=relocate_discarded,
                hard=hard,
                sim_measure=sim_measure,
            )
        return self.em.fit(model=self, X=X, limit=limit, eps=eps, verbose=verbose)

    def _pretrainer(
        self,
        X: np.ndarray,
        em: Variational = None,
        limit: list[int] | int | None = 1000,
        rng: np.random.generator | int | None = None,
        eps: list[float] | float = 1.0e-4,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        relocate_discarded: bool = True,
        indices: npt.NDArray | None = None,
    ):
        """
        Pretrain the model.
        Only implemented for Gaussian so far.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        limit : int, list[int] or None, optional
            Convergence limit(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. None means no limit. Defaults to 1000.
        rng : np.random.Generator, int or None, optional
            Random number generator or seed. For None, a random seed is used.
        eps : float or list[float]
            Convergence threshold(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. Defaults to 1.0e-4.
        C_prime : int, optional
            Number of non-zero elements in truncated posterior. Defaults to 3.
        G : int, optional
            Component neighborhood size. Defaults to 15.
        E : int, optional
            Number of randomly added components. Defaults to 1.
        indices : np.ndarray, optional
            Indices of data points uses as seeds. Used for initializing the K-sets and sets g_c. Defaults to None.

        Returns
        -------
        eps: float
            Convergence threshold to use after pertaining
        limit: int
            Convergence limit to use after pertaining
        """
        pass

    def discard(self, c: int):
        """
        Discards the given component.

        The component at the given index will be ignored and the prior is adjusted accordingly.

        Parameters
        ----------
        c : int
            Component index.

        Returns
        -------
        None
        """
        self._cpp.discard(c)

    def log_joint(self, x: npt.NDArray, c: int):
        """

        Calculates log-joint value for the given data point and component.

        Parameters
        ----------
        x : npt.ndarray
            Data point.
        c : int
            Component index.

        Returns
        -------
        logjoint : float
            Value of the log-joint of x and c.
        """
        return self._cpp.log_joint(x, c)

    def map(self, x: npt.NDArray, c: int):
        """
        Finds the maximum a posteriori (MAP) component for the given data point.

        This method finds the index of the component with the maximum a posteriori (MAP)
        probability for the given data point using exhaustive search.

        Parameters
        ----------
        x : npt.ndarray
            Data point.

        Returns
        -------
        int
            The index of the MAP component.
        """
        return self._cpp.map(x, c)

    def map_k(self, x: npt.NDArray, k: int):
        """
        Finds the indices and log-joints of the k components with the larges log-joints for the given data point.

        This method finds the indices and of the k components with the larges log-joints probabilities for the given
        data point using exhaustive search. A dictionary is returned where keys are component indices and values are
        corresponding log-joints.

        Parameters
        ----------
        x : npt.ndarray
            Data point.
        k : int
            Number of components to consider.

        Returns
        -------
        Dict[int, float]
            Dictionary with indices and log-joints of the k components with the larges log-joints.
        """
        return self._cpp.map_k(x, k)

    def ll(self, X: npt.NDArray):
        """
        Calculate the log-likelihood of the model given the input data.

        Parameters
        ----------
        X : npt.ndarray
            Input data.

        Returns
        -------
        float
            The log-likelihood of the model.
        """
        X = check_X(self.C, self.D, X, check_C=True, dtype=self.dtype)
        return self._cpp.ll(X)

    def nll(self, X: npt.NDArray):
        """
        Calculate the negative log-likelihood per data point of the model given the input data.

        Parameters
        ----------
        X : npt.ndarray
            Input data.

        Returns
        -------
        float
            The negative log-likelihood of the model.
        """
        X = check_X(self.C, self.D, X, check_C=True, dtype=self.dtype)
        return self._cpp.nll(X)

    def log_prob(self, X: npt.NDArray, indices: npt.NDArray = None):
        """
        Calculate the log probabilities per data point of the model given the input data.

        Parameters
        ----------
        X : npt.ndarray
            Input data.

        X : npt.ndarray
            Components per data point to consider.

        Returns
        -------
        npt.ndarray
            The probabilities per data point.
        """
        X = check_X(self.C, self.D, X, check_C=True, dtype=self.dtype)
        if indices is None:
            assert self.em is not None, "If Indices is None, we need a em trainer."
            N = X.shape[0]
            C_prime = self.em.C_prime
            indices = self.em.q.indices.reshape(N, C_prime)

        return self._cpp.log_prob(X, indices.astype(np.uint64))

    def probability(self, X: npt.NDArray, indices: npt.NDArray = None):
        """
        Calculate the probabilities per data point of the model given the input data.

        Parameters
        ----------
        X : npt.ndarray
            Input data.

        X : npt.ndarray
            Components per data point to consider.

        Returns
        -------
        npt.ndarray
            The probabilities per data point.
        """
        return np.exp(self.log_prob(X, indices.astype(np.uint64)))
