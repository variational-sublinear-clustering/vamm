# Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg
# and Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from vamm.utils.sanity import check_X
from vamm.utils.commons import format_time

from vamm.cpp import EM

if TYPE_CHECKING:
    from vamm.models.Models import Models


class Variational:
    """
    Variational EM algorithm based on truncated posteriors.

    Parameters
    ----------
    N : int
        Number of data points.

    C : int
        Number of mixture components.

    C_prime : int, optional
        Number of non-zero elements in truncated posterior. Defaults to 3.

    G : int, optional
        Component neighborhood size. Defaults to 15.

    E : int, optional
        Number of randomly added components. Defaults to 1.

    relocate_discarded : bool, optional
        Whether to relocate or discard components

    hard : bool, optional
        Whether to use hard assignment in the M-step. Defaults to False.

    sim_measure : {"KL","Euclidean"}, optional
        Whether to use 'KL' (Kullback-Leibler divergence) or 'Euclidean' distance as the similarity measure
        for updating the neighborhood set. Defaults to 'KL'.

    indices : npt.NDArray or None, optional
        Indices of data points uses as initial component centers. Used for initializing the K-Sets and sets g_c. Defaults to None.

    rng : np.random.Generator, int or None, optional
        Random number generator or seed. For None, a random seed is used.

    Attributes
    ----------
    N : int
        The number of data points (read-only).

    C : int
        The number of components (read-only).

    C_prime : int
        The number of non-zero elements in truncated posterior (read-only).

    G : int
        The size of the gc set (read-only).

    E : np.ndarray
        The number of randomly added components for each data point.
        The index of the array corresponds to the index of the data point.

    q : scipy sparse csr matrix
        The variational distributions ``q`` as a scipy sparse csr matrix.
        This requires copies in both directions.

    g : scipy sparse csr matrix
        The neighborhood gc_set to or from a scipy sparse csr matrix.
        This requires copies in both directions.

    number_ljs : int
        The number of log-joint evaluations in the last E-Step (read-only).

    initial_seed : int
        The initial seed for the random number generator (read-only).

    C_relocate : int
        Number of components relocated in last iteration (read-only).

    log : pd.DataFrame
        A DataFrame with training history (read-only).

    objective : float
        The training objective of the last iteration.
    """

    def __init__(
        self,
        N: int,
        C: int,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        relocate_discarded: bool = True,
        hard: bool = False,
        sim_measure: str = "KL",
        indices: npt.NDArray | None = None,
        rng: np.random.generator | int | None = None,
    ) -> None:
        assert sim_measure in (
            "Euclidean",
            "KL",
        ), f"Similarity measure must be either 'Euclidean' or 'KL', but got: '{sim_measure}'."

        if indices is None:
            print(
                "Warning: No initial indices provided. Random initialization may affect convergence and performance."
            )

        rng = np.random.default_rng(rng)
        seed = rng.integers(low=0, high=np.iinfo(np.uint32).max)
        self.__dict__["_cpp"] = EM(
            N=N,
            C=C,
            C_prime=C_prime,
            G=G,
            E=E,
            seed=seed,
            indices=indices,
            relocate_discarded=relocate_discarded,
            hard=hard,
            sim_measure=sim_measure,
        )
        self.objective = None
        self._objective_last = None
        self._log = []

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
            super(Variational, self).__setattr__(name, value)

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

    def fit(
        self,
        model: Models,
        X: npt.NDArray,
        limit: list[int] | int | None = 1000,
        eps: list[float] | float = 1.0e-4,
        verbose: bool = False,
    ) -> tuple[float, pd.DataFrame]:
        """
        Fit the model to the input data.

        Parameters
        ----------
        model : Models
            Model to train.
        X : np.ndarray
            Input data.
        limit : int, list[int] or None, optional
            Convergence limit(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. None means no limit. Defaults to 1000.
        eps : float or list[float]
            Convergence threshold(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. Defaults to 1.0e-4.
        verbose : bool, optional
            Whether to print progress messages. Defaults to False.

        Returns
        -------
        objective : float
            the final objective value.
        log : pd.DataFrame
            a DataFrame with training history.
        """
        for _ in self.fit_iter(model=model, X=X, limit=limit, eps=eps, verbose=verbose):
            pass
        return self.objective, self.log

    def fit_iter(
        self,
        model: Models,
        X: npt.NDArray,
        limit: list | int | None = 1000,
        eps: list | float = 1.0e-4,
        verbose: bool = False,
    ) -> GeneratorExit:
        """
        Fits the model to the input data and yields the result after each iteration.
        It serve as an interface for accessing the fitting process between iterations.

        Parameters
        ----------
        model : Models
            Model to train.
        X : np.ndarray
            Input data.
        limit : int, list[int] or None, optional
            Convergence limit(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. None means no limit. Defaults to 1000.
        eps : float or list[float]
            Convergence threshold(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. Defaults to 1.0e-4.
        verbose : bool, optional
            Whether to print progress messages. Defaults to False.

        Yields
        -------
        self
            The object after each iteration.

        Examples
        --------
        Print the objective and learned means after each iteration:

        .. code-block:: python

            for v in var.fit_iter(model, X):
                print(v.objective)
                print(model.means)

        """
        X = check_X(model.C, model.D, X, check_C=True, dtype=model.dtype)
        eps, limit = self._check_conv_criteria(eps, limit)
        model._precompute(X)
        model.em = self

        # yield self
        for f, M_step in enumerate([False, True]):
            i = 0
            l = limit[f]
            e = eps[f]
            while l is None or i != l:
                active = model.active

                tic = time.monotonic()
                self.objective = self._cpp._E_step(X=X, model=model._cpp)
                if M_step:
                    self._cpp._M_step(X=X, model=model._cpp)
                dt = time.monotonic() - tic

                # show progress
                if verbose:
                    self._message(model, i, M_step, dt, active)

                self._log.append(
                    {
                        "i": i,
                        "active": model.active,
                        "M_step": M_step,
                        "objective": self.objective,
                        "eval": self.number_ljs,
                        "time": dt,
                    }
                )

                # check for increasing lower bound
                if (
                    i != 0
                    and verbose
                    and self.objective > self._objective_last
                    and not np.isclose(self.objective, self._objective_last)
                ):
                    print(
                        f"Increasing objective (from {self._objective_last:<10.5f} to {self.objective:<10.5f})!",
                        flush=True,
                        end="\n\n",
                    )

                yield self
                # convergence criterion
                if i > 0:
                    if e is not None and self.stop(e):
                        break

                self._objective_last = self.objective
                i += 1

        if i == l and e > 0.0:
            print("Warning: Max. Iterations reached. Model did not converge.")
        # final objective after the last M-step
        self.objective = self._cpp._E_step(X=X, model=model._cpp)

    def stop(self, tol):
        """
        Checks if the algorithm should stop based on the relative change of the training objective.

        Used by the methods ``fit`` and ``fit_iter``.

        Parameters
        ----------
        tol : float
            Convergence tolerance.

        Returns
        -------
        bool
            True if the relative change in the objective is less than the tolerance, False otherwise.
        """
        return abs(self.objective / self._objective_last - 1) < tol

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
        self, eps: list[float] | float, limit: list[int] | int, warmup: bool = True
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

        eps = 2 * eps if len(eps) == 1 else eps
        limit = 2 * limit if len(limit) == 1 else limit

        return eps, limit

    def _message(self, model, i, M_step, dt, active):
        msg = f"Iteration {i+1} "
        msg += "(Warm-Up)\n\t" if not M_step else "\n\t"
        msg += f"Objective: {self.objective:<10.4f}\t"
        msg += f"Time: {format_time(dt)}\t"
        msg += f"Active Components: {model.active}/{model.C}\n"
        msg += (
            f"\tDiscarded {active - model.active} component(s)!\n"
            if active > model.active
            else ""
        )
        msg += (
            f"\tRelocated {self.C_relocate} component(s)!\n"
            if self.C_relocate > 0
            else ""
        )
        print(msg, flush=True)

    def fill_E(self, E: int):
        """
        Set the number of randomly added components for all data points.

        Parameters
        ----------
        E : int
            Number of randomly added components.

        Returns
        -------
        None
        """
        self._cpp.fill_E(E)

    def set_num_random(self, n: int, E: int):
        """
        Set the number of randomly added components for the data point with index ``n``.

        Parameters
        ----------
        n : int
            Index of the data point.
        E : int
            Number of randomly added components.

        Returns
        -------
        None
        """
        self._cpp.set_num_random(n, E)

    def q_map(self, n: int):
        """
        The variational distribution for the data point with index ``n``.

        This method returns a dictionary with the C_prime non-zero elements in the variational distribution
        of the data point with index ``n``. Keys are component indices and values are corresponding values
        of the variational distribution.

        Parameters
        ----------
        n : int
            Index of the data point.

        Returns
        -------
        Dict[int, float]
            Dictionary representing the variational distribution.
        """
        return self._cpp.q_map(n)

    def q_in(self, n: int, map: dict[int, float]):
        """
        Set the variational distribution for the data point with index ``n``.

        This method accepts a dictionary with C_prime elements that contain the variational distribution
        of the data point with index ``n``. Keys are component indices and values are corresponding values
        of the variational distribution.

        Parameters
        ----------
        n : int
            Index of the data point.
        map : Dict[int, float]
            Dictionary representing the variational distribution.

        Returns
        -------
        None
        """
        self._cpp.q_in(n, map)

    def gc_set(self, c: int):
        """
        Neighborhood of component ``c``.

        This method retrieves the approximate neighborhood of the component with index ``c``.
        The neighborhood always includes the index ``c`` itself and the G-1 indices of the
        nearest components found so far.

        Parameters
        ----------
        c : int
            Component index.

        Returns
        -------
        np.array[int]
            An array containing the indices of the component neighborhood.
        """
        return self._cpp.gc_set(c)

    def approx_map(self, n: int):
        """
        Finds the index of the component with the largest variational distribution for the data
        point with index ``n``.

        This method finds the index of the component with the larges value of the variational distribution
        for the given data point within the k non-zero elements.

        Parameters
        ----------
        n : int
            Index of the data point.

        Returns
        -------
        index: int
            The index of the component.
        value: float
            The value of the variational distribution
        """
        return self._cpp.approx_map(n)

    def indices(self):
        """
        Gets the indices of the components with the largest variational distribution for all data
        point.

        This method finds the indices of the components with the largest value of the variational distribution for all data points. The index of the resulting array corresponds to the
        index of the data point.

        Returns
        -------
        indices: npt.ndarray
            An array containing the indices of the components.
        """
        return self._cpp.indices

    def q_shrinked(self, mask):
        """
        Get the variational distributions ``q`` as a scipy sparse csr matrix.
        The mask is used to disregard components.

        Parameters
        ----------
        mask : np.array[bool]
            Masking of the components.

        Returns
        -------
        q: scipy sparse csr matrix
            The variational distributions.
        idx: np.array[int]
            To map from column index in q to component index c.
        """
        return self._cpp.q_shrinked(mask)
