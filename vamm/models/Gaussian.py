# Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg
# and Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vamm.models.Models import Models
from ..em.Variational import Variational
from vamm.cpp import MFA, Diagonal, Full
from vamm.utils.init_params import uniform, data_variance, random_variance


class Gaussian(Models):
    """
    Gaussian Mixture Model.

    Parameters
    ----------
    C : int
        Number of components.

    D : int
        Dimensionality of the data.

    covariance_type : {"isotropic", "diagonal", "mfa", "full"}
        Type of covariance matrix.

    shared : bool, optional
        Whether the covariance matrix is shared among the Gaussian components. Defaults to False.

    H : int
        Dimensionality of the factors. (Only used for "mfa".)

    flat_prior : bool, optional
        Whether to use a flat prior for the mixture components. Defaults to False.

    init_prior : np.ndarray or "flat", optional
        Initial values for the priors of the mixture components. Defaults to "flat", which initializes flat priors ``1/C``.

    init_means : np.ndarray or {"afkmc2", "random"}, optional
        Initial values for the means of the mixture components. "afkmc2": AFK-MC² initialization.
        "random": randomly selected data point. Defaults to "afkmc2".

    init_A : np.ndarray or "uniform", optional
        Initial values for the factor loading matrices. Defaults to "uniform", which fills the factor loadings with uniform random numbers in [0, 1].
        (Only used for "mfa".)

    init_variance : np.ndarray or "data_variance", optional
        Initial values for the diagonal variance. Defaults to "data_variance", which uses the variance of the data.

    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 1e-3.

    Attributes
    ----------
    C : int
        Number of mixture components (read-only).

    D : int
        Dimensionality of the input data (read-only).

    H : int
        Dimensionality of the latent factor space (read-only). (Only used for "mfa".)

    prior : ndarray of shape (C,)
        Prior probabilities of the mixture components. The values sum to one
        over all active components.

    log_prior : ndarray of shape (C,)
        The natural logarithm of the prior (read-only).

    means : ndarray of shape (C, D)
        Mean vectors of the mixture components.

    variance : ndarray of shape (C, D)
        Diagonal variance vectors of the mixture components.

    A : ndarray of shape (C, D, H)
        Factor loading matrices. Each row contains the loading
        matrix of one mixture component. (Only used for "mfa".)

    mask : ndarray of shape (C,), dtype=bool
        Boolean mask indicating which mixture components are active. Components
        marked as ``False`` are ignored during inference and training.

    active : int
        Number of active mixture components (read-only).

    flat_prior : bool
        Whether all active mixture components share the same prior probability.

    dtype : numpy.dtype
        Floating-point data type used to store the model parameters (read-only).

    verbose_discard : bool
        Whether to print a message when a component gets discarded. Defaults to False.

    reg_covar : float
        The regularization strength of the covariance matrix (read-only).

    em : Variational
        The last used EM trainer.
    """

    def __init__(
        self,
        C: int,
        D: int,
        covariance_type: str,
        shared: bool = False,
        H: int | None = None,
        flat_prior: bool = False,
        init_prior: npt.NDArray | str = "flat",
        init_means: npt.NDArray | str = "afkmc2",
        init_A: npt.NDArray | str = "uniform",
        init_variance: npt.NDArray | str = "data_variance",
        reg_covar: float = 1e-3,
    ) -> None:
        assert covariance_type in (
            "isotropic",
            "diagonal",
            "mfa",
            "full",
        ), f"'covariance_type' should be one of 'isotropic', 'diagonal', 'mfa' or 'full', but got '{covariance_type}'"

        self.__dict__["_covariance_type"] = covariance_type
        self.__dict__["_shared"] = shared
        self.__dict__["_reg_covar"] = reg_covar

        if covariance_type in ("mfa",):
            assert type(H) == int, f"Type of 'H' should be integer, but got '{type(H)}'"
            super().__init__(
                MFA(C, D, H, flat_prior, shared, reg_covar),
                init_prior,
                init_means,
                flat_prior,
            )
        elif covariance_type in ("full",):
            super().__init__(
                Full(C, D, flat_prior, shared, reg_covar),
                init_prior,
                init_means,
                flat_prior,
            )
        else:
            super().__init__(
                Diagonal(C, D, flat_prior, covariance_type, shared, reg_covar),
                init_prior,
                init_means,
                flat_prior,
            )

        self._init |= {"variance": None, "A": None}

        if type(init_A) is np.ndarray and covariance_type in ("mfa",):
            assert init_A.shape == (
                C,
                D,
                H,
            ), f"Shape of Factors should be ({C,D,H},), but got {init_A.shape}"
            self.A = init_A
        elif type(init_A) is str and covariance_type in ("mfa",):
            self._cpp.A[:] = np.nan
            self._init["A"] = init_A

        if type(init_variance) is np.ndarray:
            self._check_variance(
                init_variance, covariance_type, shared, C, D, reg_covar
            )
        elif type(init_variance) is str:
            self._cpp.variance[:] = np.nan
            self._init["variance"] = init_variance

    def _check_variance(self, init_variance, covariance_type, shared, C, D, reg_covar):
        if covariance_type == "isotropic" and shared:
            assert init_variance.shape == (
                1,
            ), f"Shape of isotropic variance should be (1,), but got {init_variance.shape}"
            init_variance = np.full((C, D), init_variance)

        if covariance_type == "isotropic" and not shared:
            assert init_variance.shape == (
                C,
            ), f"Shape of isotropic variance should be (C,), but got {init_variance.shape}"
            init_variance = np.repeat(init_variance, D).reshape(C, D)

        if covariance_type in ("diagonal", "mfa") and shared:
            assert init_variance.shape == (
                D,
            ), f"Shape of diagonal variance should be ({D},), but got {init_variance.shape}"
            init_variance = np.tile(init_variance, (C, 1))

        if covariance_type in ("diagonal", "mfa") and not shared:
            assert init_variance.shape == (
                C,
                D,
            ), f"Shape of diagonal variance should be ({C,D}), but got {init_variance.shape}"

        if covariance_type == "full" and shared:
            assert init_variance.shape == (
                D,
                D,
            ), f"Shape of covariance should be ({D,D}), but got {init_variance.shape}"
            if (init_variance.diagonal() < reg_covar).any():
                print(f"add {reg_covar} to diagonal of covariance")
                np.maximum(
                    init_variance.flat[:: D + 1],
                    reg_covar,
                    out=init_variance.flat[:: D + 1],
                )
                # init_variance.flat[:: D + 1] += reg_covar
            self._cpp.variance = np.tile(init_variance.reshape(-1), (C, 1))

        if covariance_type == "full" and not shared:
            assert init_variance.shape == (
                C,
                D,
                D,
            ), f"Shape of covariance should be ({C,D,D}), but got {init_variance.shape}"
            if (init_variance.diagonal(0, 1, 2) < reg_covar).any():
                print(f"add {reg_covar} to diagonal of covariance")
                for c in range(C):  # TODO: this without a loop?
                    init_variance[c].flat[:: D + 1] += reg_covar
            self._cpp.variance = init_variance.reshape(self.C, self.D * self.D)

        if covariance_type != "full":
            if (init_variance < reg_covar).any():
                print(f"minimum value of diagonal variance is increased to {reg_covar}")
                np.maximum(init_variance, reg_covar, out=init_variance)
            self.variance = init_variance

    def _initialize(
        self,
        X: npt.NDArray,
        indices: npt.NDArray | None = None,
        rng: np.random.Generator | int | None = None,
        verbose: bool = False,
    ):
        rng = np.random.default_rng(rng)
        indices = Models._initialize(
            self, X=X, indices=indices, rng=rng, verbose=verbose
        )

        if np.isnan(self.variance).all() and self._init["variance"] is not None:
            init_method_variance = self._init["variance"]
            assert init_method_variance in (
                "data_variance",
                "random",
            ), "Initialization method for variance unknown."
            if init_method_variance in ("data_variance",):
                init_variance = data_variance(
                    X, self.C, self.covariance_type, self.shared, verbose=verbose
                )
            elif init_method_variance in ("random",):
                init_variance = random_variance(
                    X,
                    self.C,
                    self.covariance_type,
                    self.shared,
                    self._reg_covar,
                    rng=rng,
                    verbose=verbose,
                )
            self._check_variance(
                init_variance,
                self.covariance_type,
                self.shared,
                self.C,
                self.D,
                self._reg_covar,
            )

        if (
            self.covariance_type == "mfa"
            and np.isnan(self.A).all()
            and self._init["A"] is not None
        ):
            assert self._init["A"] in (
                "uniform",
            ), "Initialization method for A unknown."
            self.A = uniform(self.C, self.D, self.H, rng=rng, verbose=verbose)

        return indices

    @property
    def covariance_type(self) -> str:
        """
        The used covariance type.

        This property retrieves the chosen covariance type, i.e.,
        'isotropic', 'diagonal', 'mfa' or 'full' (read-only).

        Returns
        -------
        str
            covariance type.
        """
        return self._covariance_type

    @property
    def shared(self) -> bool:
        """
        This property retrieves whether the covariance matrix is
        shared among the Gaussian components (read-only).

        Returns
        -------
        bool
            shared.
        """
        return self._shared

    @property
    def reg_covar(self) -> float:
        """
        The used value of 'reg_covar'.

        This property retrieves the regularization strength of the
        covariance matrix (read-only).

        Returns
        -------
        float
            reg_covar.
        """
        return self._reg_covar

    @property
    def A(self) -> np.ndarray:
        """
        The factor loading matrices.

        This property retrieves the factor loading matrices A from the MFA C++ object.

        Raises
        ------
        AttributeError
            If the underlying object is not an instance of MFA, i.e., if the covariance type is
            not "mfa".

        Returns
        -------
        np.ndarray
            factor loading matrices with shape (C, D, H).
        """
        if not isinstance(self._cpp, MFA):
            raise AttributeError(f"'{self._cpp}' has no attribute 'A'")
        return self._cpp.A.reshape(self.C, self.D, self.H)

    @A.setter
    def A(self, _A: np.ndarray) -> None:
        """
        Set the factor loading matrices.

        This setter property sets the factor loading matrices A in the MFA C++ object.

        Parameters
        ----------
        _A : np.ndarray
            The factor loading matrices to be set, expecting the shape (C, D, H).

        Raises
        ------
        AttributeError
            If the underlying object is not an instance of MFA, i.e., if the covariance type is
            not "mfa".

        Returns
        -------
        None
        """
        if not isinstance(self._cpp, MFA):
            raise AttributeError(f"'{self._cpp}' has no attribute 'A'")
        self._cpp.A = _A.reshape(self.C, self.D * self.H)

    def covariances(self) -> np.ndarray:
        """
        Compute the covariance matrices of the model.

        This method returns the covariance matrices based on the covariance type specified in the model.

        - For ``"full"`` the covariance matrices are directly returned.

        - For ``"mfa"``, the covariance matrix for each component is computed as:

            .. math::
                \\Sigma_c = A_c A_c^T + D_c

        - For other covariance types, the covariance matrices are diagonal matrices derived directly from the variance.

        Returns
        -------
        np.ndarray
            An array of shape ``(C, D, D)`` representing the covariance matrices of the model.

        Notes
        -----
        - This method is intended for convenience and should only be used when the number of components ``C`` and the data dimensionality ``D`` are small to medium-sized.
        - Computing and storing full covariance matrices can be computationally expensive for large ``C`` and ``D``. In this case it is recommended to directly work with ``variance`` and/or ``A``.
        """
        if self.covariance_type in ("full",):
            return self.variance
        covar = np.stack([np.diag(self.variance[c]) for c in range(self.C)], axis=0)
        if self.covariance_type in ("mfa",):
            covar += np.matmul(self.A, self.A.transpose(0, 2, 1))
        return covar

    @property
    def variance(self) -> np.ndarray:
        """
        The variances.

        This property gets the variances from the C++ object.

        Parameters
        ----------
        None

        Returns
        -------
        variance : np.ndarray
            The variance, shape depends on the covariance type.
        """
        if isinstance(self._cpp, Full):
            return self._cpp.variance.reshape(self.C, self.D, self.D)
        else:
            return self._cpp.variance

    @variance.setter
    def variance(self, _variance: np.ndarray) -> None:
        """
        Set the variances.

        This setter property sets the variances in the C++ object.

        Parameters
        ----------
        _variance : np.ndarray
            The variances matrices to be set, shape depends on the covariance type.

        Returns
        -------
        None
        """
        if isinstance(self._cpp, Full):
            self._cpp.variance = _variance.reshape(self.C, self.D * self.D)
        else:
            self._cpp.variance = _variance

    def z_projection(self, x: np.ndarray, c: int):
        """
        Projects the given data point into the latent space by calculating the most likely factor 'z' given component 'c'. (Only used for "mfa".)

        Parameters
        ----------
        x : npt.ndarray
            Data point.
        c : int
            Component index.

        Returns
        -------
        z : npt.ndarray
            Factor represented as an H-dimensional vector.
        """
        if not isinstance(self._cpp, MFA):
            raise AttributeError(f"'{self._cpp}' has no function 'z_projection'")
        return self._cpp.z_projection(x, c)

    def mahalanobis_distance(self, x: np.ndarray, c: int):
        """
        Computes the Mahalanobis distance for data point x and component c.
        (Only used for "mfa".)

        Parameters
        ----------
        x : npt.ndarray
            Data point.
        c : int
            Component index.

        Returns
        -------
        val : float
            mahalanobis distance
        """
        if not isinstance(self._cpp, MFA):
            raise AttributeError(
                f"'{self._cpp}' has no function 'mahalanobis_distance'"
            )
        return self._cpp.mahalanobis_distance(x, c)

    def generate_data(
        self, N: int, c: int = None, add_noise: bool = True, seed: int = None
    ):
        """
        Generate synthetic data based on the model parameters.

        Parameters
        ----------
        N : int
            Number of data points to generate.
        c : int, optional
            If given, only data from this component are generated
        add_noise : bool, optional
            Whether to add noise to the generated data. Defaults to True.
        seed : int, optional
            Seed for random number generation. If None, the global generator is used.

        Returns
        -------
        X : np.ndarray
            Synthetic data generated by the model.
        """
        c = -1 if c is None else c
        seed = -1 if seed is None else seed
        return self._cpp.generate_data(N, c, add_noise, seed)

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
        verbose: bool = True,
    ) -> tuple[float, int]:
        """
        Pretrain the model with an isotropic GMM trained with variational EM using the input data.

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
            Convergence threshold for the EM iterations (after pertaining)
        limit: int
            Convergence limit for the EM iterations (after pertaining)
        """
        if verbose:
            print("Pretrain means ... ", flush=True)

        pretrainer = Gaussian(
            C=self.C,
            D=self.D,
            covariance_type="isotropic",
            shared=True,
            flat_prior=True,
            init_means=self.means,
            init_variance=np.ones(1),
        )
        pretrainer.fit(
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
        self.em = pretrainer.em
        self.em._log = pretrainer.em._log

        self.mask = pretrainer.mask
        self.prior = pretrainer.prior
        self.means = pretrainer.means
        self.variance[:] = pretrainer.variance[0, 0]
