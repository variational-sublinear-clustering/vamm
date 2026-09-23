"""Shared fixtures for vamm integration tests."""

from __future__ import annotations

import os
import numpy as np
import pytest
from pathlib import Path
import subprocess


def _set_single_thread() -> None:
    try:
        import cpputils

        cpputils.omp.set_num_threads(1)
    except (ImportError, AttributeError):
        pass


def build_cpp_tests():
    print("Building C++ tests")
    # #!/usr/bin/env bash
    # # Build and run C++ unit tests
    # set -euo pipefail

    # # ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    # BUILD_DIR="./build/cpp-tests"

    # if [[ ! -d "./vamm/extern/eigen/Eigen" ]]; then
    #     echo "Eigen submodule missing. Run from repo root:"
    #     echo "  git submodule update --init --recursive"
    #     exit 1
    # fi

    # cmake -S "vamm/cpp/tests" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Debug
    # cmake --build "${BUILD_DIR}" -j"$(nproc)"
    # ctest --test-dir "${BUILD_DIR}" --output-on-failure
    repo_root = Path(__file__).resolve().parents[1]

    eigen_path = repo_root / "vamm" / "extern" / "eigen" / "Eigen"
    assert eigen_path.exists(), (
        "Eigen submodule missing. " "Run: git submodule update --init --recursive"
    )

    build_dir = repo_root / "build" / "cpp-tests"

    subprocess.run(
        [
            "cmake",
            "-S",
            str(repo_root / "vamm" / "cpp" / "tests"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Debug",
        ],
        check=True,
    )

    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", str(os.cpu_count())],
        check=True,
    )


def pytest_sessionstart(session):
    build_cpp_tests()


@pytest.fixture(scope="session")
def cppvamm():
    """Import the compiled C++ extension; skip if not built."""
    mod = pytest.importorskip("cppvamm")
    required = ("Variational", "Diagonal")
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        pytest.skip(f"cppvamm missing: {', '.join(missing)} (run: pip install -e .)")
    _set_single_thread()
    return mod


@pytest.fixture
def small_gaussian_data():
    """Small reproducible dataset for integration tests."""
    rng = np.random.default_rng(42)
    N, D = 30, 3
    X = rng.standard_normal((N, D))
    return np.ascontiguousarray(X, dtype=np.float64)


@pytest.fixture
def variational_params():
    return dict(N=30, C=6, C_prime=3, G=4, E=1, seed=123)


@pytest.fixture
def diagonal_model(cppvamm, variational_params):
    C = variational_params["C"]
    D = 3
    return cppvamm.Diagonal(C, D, False, "diagonal", False, 1e-3)


@pytest.fixture(params=[True, False])
def mfa_model(request):
    from vamm import Gaussian

    return Gaussian(C=2, D=3, H=2, covariance_type="mfa", shared=request.param)


@pytest.fixture(scope="session")
def vamm_variational_class(cppvamm):
    """Import Python Variational wrapper without pulling the full vamm package."""
    from vamm.em.Variational import Variational

    return Variational
