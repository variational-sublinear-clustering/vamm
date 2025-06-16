# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import numpy as np
import sklearn.datasets
from itertools import product

# TODO: find other solution for toy dataset and remove sklearn from requirements?

from vamm import Gaussian


def run(cov_type, shared, X, C, H, seed):
    N, D = X.shape
    rng = np.random.default_rng(seed)

    model = Gaussian(
        C=C,
        D=D,
        H=H,  # H is only needed for "mfa"
        covariance_type=cov_type,
        shared=shared,
    )
    _shared = "with shared covariance matrices" if shared else ""
    print(f"Train {cov_type} {_shared} ...")
    obj, logs = model.fit(
        X=X,
        rng=rng,
    )
    print(f"Objective = {obj:<10.5f}\n")


if __name__ == "__main__":
    cov_types = ["isotropic", "diagonal", "mfa", "full"]
    shared_list = [True, False]

    C = 100
    H = 2
    seed = 123

    rng = np.random.default_rng(seed)

    data = sklearn.datasets.load_digits().data.astype(np.float64)
    data += rng.standard_normal(size=data.shape)
    data = np.ascontiguousarray(data, dtype=np.float64)

    N, D = data.shape

    for cov_type, shared in product(cov_types, shared_list):
        obj = run(cov_type, shared, data, C, H, seed)
