# VAMM

**Variational Accelerated Mixture Models (VAMM)** is a Python/C++ package for truncated variational optimization of mixture models, suitable for high-dimensional, large-scale datasets and large models.
Refer to the [related publications](#related-publications) for more details. To get started, check out the documentation (**TODO**) and explore the provided [example](#run-the-demo).

## Installation

### Requirements

Ensure the following requirements are met for installation:

- A C++ compiler that supports the C++17 Standard, such as the [GNU g++ Compiler](https://gcc.gnu.org/)
- [Python 3](https://www.python.org/) (version 3.9 or higher)
- [OpenMP](https://www.openmp.org/) (for parallel execution)
- [Git](https://git-scm.com/)

Please note that the code has only been tested on Linux distributions.

### Setup

1. **Clone the Repository**

    Clone this repository with the `--recursive` flag to include the required submodules:

    ```bash
    git clone --recursive git@gitlab.uni-oldenburg.de:ml-oldb/expectation-maximization-for-mixture-models/variational-em.git vamm
    cd vamm/
    ```

    If you have cloned the repository without the `--recursive` flag, run the following commands inside the repository to initialize and update the submodules:

    ```bash
    git submodule init
    git submodule update
    ```

    This will download the required C++ libraries, [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) and a subset of [Boost](https://www.boost.org/).

2. **Install Python Packages**

    We recommend using [Anaconda](https://www.anaconda.com/) to manage the installation and create a new environment for the project:

    ```bash
    conda create -n vamm python=3.9
    conda activate vamm
    ```

    Next, install the package with [pip](https://pypi.org/project/pip/):

    ```bash
    pip install .
    ```

    This command installs the required Python dependencies and builds the C++ libraries using [pybind11](https://github.com/pybind/pybind11).

### Different Builds and Versions (advanced)

By default, the C++ libraries are built with the `Release` configuration. You can change the build type (`Release`, `Debug`, `RelWithDebInfo`, or `MinSizeRel`) by modifying the `build-type` line in [`pyproject.toml`](./pyproject.toml).

[OpenMP](https://www.openmp.org/) for multiprocessing is linked by default. For a serial execution, multiprocessing can be disabled by removing `"-fopenmp"` from `extra_compile_args` and `"-lgomp"` from all `extra_link_args` in [`setup.py`](./setup.py).

Rebuild the package using `pip install .` for the changes to take effect.

## Run the Demo

After [installation](#installation), you will be able to run **VAMM**. Therefore, check out our [examples](./examples/README.md).
The demo fits various mixture models to a [dataset of digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) and reports the final training objectives.

## Related Publications

TODO
