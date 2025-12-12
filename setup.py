# Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg 
# and Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

# install: pip install .
# develop: pip install --editable . // pip install -e .

import sys
import toml
import sysconfig
from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile

pyproject_text = Path("pyproject.toml").read_text()
pyproject_data = toml.loads(pyproject_text)
build_type = pyproject_data["build-system"]["build-type"]

requirements = Path("requirements.txt").read_text().splitlines()

BUILD_TYPES = {
    "Release": ["-O3", "-DNDEBUG"],
    "Debug": ["-O0", "-g"],
    "RelWithDebInfo": ["-O2", "-g", "-DNDEBUG"],
    "MinSizeRel": ["-Os", "-DNDEBUG"],
}

include_dirs = [
    "vamm/extern/eigen",
    "vamm/cpp/include",
]

for lib in (
    "unordered",
    "assert",
    "container_hash",
    "config",
    "core",
    "predef",
    "throw_exception",
    "mp11",
    "describe",
    "static_assert",
):
    include_dirs += [f"vamm/extern/boost/{lib}/include"]


# check if submodules are fetched (Eigen and Boost)
for _path in include_dirs:
    path = Path(_path.replace("/include", ""))
    if not (path.exists() and any(path.iterdir())):
        raise ImportError(
            "Eigen or Boost seems to be missing! Please call 'git submodule init && git submodule update' to fetch these libraries."
        )

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += [
    "-Wall",
    "-Wextra",
    # "-Wshadow",
    "-pedantic",
    "-Wno-unknown-pragmas",
    "-march=native",
    "-DBOOST_ALLOW_DEPRECATED_HEADERS",
    "-fopenmp",
]
extra_compile_args += BUILD_TYPES.get(build_type, [])

define_macros = [("CLUSTERING_PRECISION", "double"), ("EIGEN_DONT_PARALLELIZE", None)]

ext_modules = [
    Pybind11Extension(
        "cppvamm",
        [
            "vamm/cpp/src/Bindings.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lgomp"],
        define_macros=define_macros,
        language="c++",
        cxx_std=17,
    ),
    Pybind11Extension(
        "cpputils",
        [
            "vamm/cpp/src/BindingsUtils.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lgomp"],
        define_macros=define_macros,
        language="c++",
        cxx_std=17,
    ),
]

with ParallelCompile(default=0):
    setup(
        name="vamm",
        version="0.1",
        packages=find_packages(),
        zip_safe=False,
        ext_modules=ext_modules,
        install_requires=requirements,
    )
