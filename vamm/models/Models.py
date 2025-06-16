# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import time
from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd

from vamm.utils.sanity import check_X
from vamm.utils.commons import format_time
from vamm.utils.init_params import flat, afkmc2, random_data

from vamm.cpp import EM


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

        self.objective = None
        self._objective_last = None

        self._log = []

        self._init = {
            param: init
            for param, init in zip(
                ["prior", "means"],
                [init_prior, init_means],
            )
            if type(init) is str
        }

        if type(init_prior) is np.ndarray and not self.flat_prior:
            self.prior = init_prior

        if type(init_means) is np.ndarray:
            self.means = init_means

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

    def initialize(
        self,
        X: npt.NDArray,
        indices: npt.NDArray | None = None,
        rng: np.random.Generator | int | None = None,
        verbose: bool = False,
    ):
        rng = np.random.default_rng(rng)
        if "prior" in self._init.keys():
            assert self._init["prior"] in (
                "flat",
            ), "Initialization method for prior unknown."
            if not self.flat_prior:
                self.prior = flat(self.C, dtype=self.prior.dtype, verbose=verbose)
            self._init.pop("prior")

        if "means" in self._init.keys():
            assert self._init["means"] in (
                "afkmc2",
                "random",
            ), "Initialization method for means unknown."
            self.means, indices = (
                afkmc2(X, self.C, rng=rng, verbose=verbose)
                if self._init["means"] == "afkmc2"
                else random_data(X, self.C, rng=rng, verbose=verbose)
            )
            self._init.pop("means")

        return indices

    def fit(
        self,
        X: npt.NDArray,
        limit: list[int] | int | None = 1000,
        rng: np.random.generator | int | None = None,
        eps: list[float] | float = 1.0e-4,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        hard: bool = False,
        g_approx: str = "",
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
        eps, limit = self._check_conv_criteria(eps, limit)
        if use_pretrainer:
            eps, limit = self.pretrainer(
                X, limit, rng, eps, C_prime, G, E, indices, verbose
            )

        for _ in self._fit(
            X=X,
            limit=limit,
            rng=rng,
            eps=eps,
            C_prime=C_prime,
            G=G,
            E=E,
            hard=hard,
            g_approx=g_approx,
            indices=indices,
            verbose=verbose,
        ):
            pass

        return self.objective, self.log

    # TODO: make 'private'?
    def pretrainer(
        self,
        X: np.ndarray,
        limit: list[int] | int | None = 1000,
        rng: np.random.generator | int | None = None,
        eps: list[float] | float = 1.0e-4,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
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
        return eps, limit

    def _fit(
        self,
        X: npt.NDArray,
        limit: list | int | None = 1000,
        rng: np.random.Generator | int | None = None,
        eps: list | float = 1.0e-4,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        hard: bool = False,
        g_approx: str = "",
        indices: npt.NDArray = None,
        verbose: bool = False,
    ) -> GeneratorExit:
        """
        This method fits the model to the input data using the EM algorithm and
        yields the result for each iteration and is internally used by the '.fit()' method.
        It can also serve as an interface for accessing the fitting process between iterations.

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
        indices : np.ndarray, optional
            Indices of data points uses as seeds. Used for initializing the K-sets and sets g_c. Defaults to None.
        use_pretrainer : bool, optional
            Whether to use pretraining. Defaults to False.
        verbose : bool, optional
            Whether to print progress messages. Defaults to False.

        Yields
        -------
        self
            The object after each iteration.
        """

        eps, limit = self._check_conv_criteria(eps, limit)
        rng = np.random.default_rng(rng)

        X = check_X(self.C, self.D, X, check_C=True, dtype=self.dtype)
        N = X.shape[0]

        indices = self.initialize(X=X, indices=indices, rng=rng, verbose=verbose)
        self._precompute(X)

        if indices is not None:
            assert np.unique(indices).shape[0] == self.C, "indices are not unique!"
        # initialization
        if self.em is None:
            seed = rng.integers(low=0, high=np.iinfo(np.uint32).max)
            self.em = EM(
                N=X.shape[0],
                C=self.C,
                C_prime=C_prime,
                G=G,
                E=E,
                seed=seed,
                indices=indices,
                hard=hard,
                g_approx=g_approx,
            )
        # yield self
        for f, M_step in enumerate([False, True]):
            i = 0
            l = limit[f]
            e = eps[f]
            while l is None or i != l:
                active = self.active

                tic = time.monotonic()
                self.objective = self.em.E_step(X=X, model=self._cpp)
                if M_step:
                    self.em.M_step(X=X, model=self._cpp)
                dt = time.monotonic() - tic

                # show progress
                if verbose:
                    self._message(i, M_step, dt, active)

                self._log.append(
                    {
                        "i": i,
                        "active": self.active,
                        "M_step": M_step,
                        "objective": self.objective,
                        "eval": self.em.number_ljs,
                        "time": dt,
                    }
                )

                # check for increasing lower bound
                if i != 0:  # -initial
                    if (
                        self.objective > self._objective_last
                        and not np.isclose(self.objective, self._objective_last)
                        and verbose
                    ):
                        print(
                            f"Increasing objective (from {self._objective_last:<10.5f} to {self.objective:<10.5f})!",
                            flush=True,
                            end="\n\n",
                        )

                yield self
                # convergence criterion
                if i > 0:
                    if e is not None and self._stop(e):
                        break

                self._objective_last = self.objective
                i += 1

        if i == l and e > 0.0:
            print("Warning: Max. Iterations reached. Model did not converge.")
        # final objective after the last M-step
        self.objective = self.em.E_step(X=X, model=self._cpp)

    def _stop(self, e):
        """
        Checks if the algorithm should stop based on the relative change of the training objective.

        Parameters
        ----------
        e : float
            Convergence threshold.

        Returns
        -------
        bool
            True if the relative change in the objective is less than the threshold, False otherwise.
        """
        return abs(self.objective / self._objective_last - 1) < e

    @property
    def log(self):
        """
        The training history as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The training history.
        """
        return pd.DataFrame(self._log)

    def _check_conv_criteria(
        self, eps: list[float] | float, limit: list[int] | int
    ) -> tuple[list[float], list[int]]:
        """
        Checks variables for the convergence criteria and iteration limits and adjust them.

        Parameters
        ----------
        eps : float or list[float]
            Convergence threshold(s).
        limit : int or list[int]
            Convergence limit(s).

        Returns
        -------
        list[float]
            The convergence thresholds.
        list[int]
            The maximum number of EM iterations.
        """
        if not isinstance(eps, (list, tuple)):
            eps = [eps]
        if not isinstance(limit, (list, tuple)):
            limit = [limit]

        assert len(eps) == 1 or len(eps) == 2
        assert len(limit) == 1 or len(limit) == 2

        # if self.em is already initialized, skip warm-up steps by default
        # but do them if limit or eps have explicitly two values
        if self.em is not None and len(limit) == 1 and len(eps) == 1:
            limit = [0, limit[0]]

        eps = 2 * eps if len(eps) == 1 else eps
        limit = 2 * limit if len(limit) == 1 else limit

        return eps, limit

    def _message(self, i, M_step, dt, active):
        msg = f"Iteration {i+1} "
        msg += "(Warm-Up)\n\t" if not M_step else "\n\t"
        msg += f"Objective: {self.objective:<10.4f}\t"
        msg += f"Time: {format_time(dt)}\t"
        msg += f"Active Components: {self.active}/{self.C}\n"
        msg += (
            f"\tDiscarded {active - self.active} component(s)!\n"
            if active != self.active
            else ""
        )
        print(msg, flush=True)
