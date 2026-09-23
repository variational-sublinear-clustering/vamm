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
    git clone --recursive git@github.com:variational-sublinear-clustering/vamm.git
    cd vamm/
    ```

    If you have cloned the repository without the `--recursive` flag, run the following commands inside the repository to initialize and update the submodules:

    ```bash
    git submodule update --init
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

## Run Tests

The integration and unit tests require additional dependencies, which can be installed with:

```bash
pip install .[test]
```

Once the dependencies are installed, run the test suite with:

```bash
pytest
```

## Types of Gaussian Mixtures

The `Gaussian` class implements Gaussian mixture models (GMMs) with various types of covariance structures. Currently, it supports isotropic, diagonal and full variances, as well as mixtures of factor analyzers (MFAs). The variances can optionally be shared among all components.
The type is specified via the `covariance_type` argument, while the `shared` boolean controls whether variances are shared. (For MFAs, only the diagonal variance is shared, but each component has its own factor loadings.)
The table below provides an overview of the supported types. The **sklearn** column shows the corresponding `covariance_type` used in scikit-learn's [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) class.

| `covariance_type` | `shared` | variance                                                             | description                                                                  | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)     |
|:-----------------:|:--------:|:--------------------------------------------------------------------:|------------------------------------------------------------------------------|:-----------:|
| `isotropic`       | :heavy_check_mark:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\sigma$" title="shared scalar variance" />                                                             | all components share the same scalar variance                                | -           |
| `isotropic`       | :x:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\sigma_c$" title="scalar variance" />                                                           | each component has its own scalar variance                                   | `spherical` |
| `diagonal`        | :heavy_check_mark:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\boldsymbol{D}$" title="shared diagonal variance" />                                                     | all components share the same diagonal covariance matrix                     | -           |
| `diagonal`        | :x:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\boldsymbol{D}_c$" title="diagonal variance" />                                                   | each component has its own diagonal covariance matrix                        | `diag`      |
| `mfa`             | :heavy_check_mark:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\boldsymbol{\Lambda}_c\boldsymbol{\Lambda}_c^T&plus;\boldsymbol{D}$" title="mixture of factor analyser (shared diagonal)" />   | each component has a low-rank + diagonal covariance matrix (shared diagonal) | -           |
| `mfa`             | :x:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\boldsymbol{\Lambda}_c\boldsymbol{\Lambda}_c^T&plus;\boldsymbol{D}_c$" title="mixture of factor analyser" /> | each component has its own low-rank + diagonal covariance matrix             | -           |
| `full`            | :heavy_check_mark:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\boldsymbol{\Sigma}$" title="shared full covariance" />                                                | all components share the same general covariance matrix                      | `tied`      |
| `full`            | :x:       | <img src="https://latex.codecogs.com/png.image?\dpi{110}\color{RoyalBlue}$\boldsymbol{\Sigma}_c$" title="full covariance" />                                              | each component has its own general covariance matrix                         | `full`      |

## Related Publications

If you use this work, please cite the following paper:

S. Salwig*, T. Kahlke*, F. Hirschberger, D. Forster, and J. Lücke.
"Sublinear Variational Optimization of Gaussian Mixture Models with Millions to Billions of Parameters".
*[Journal of Machine Learning Research, 27(167):1−70](https://jmlr.org/papers/v27/25-0639.html)* (2026).
*joint first authorship.

```bibtex
@article{SalwigKahlkeEtAl2026,
  author  = {Sebastian Salwig and Till Kahlke and Florian Hirschberger and Dennis Forster and J{{\"o}}rg L{{\"u}}cke},
  title   = {Sublinear Variational Optimization of Gaussian Mixture Models with Millions to Billions of Parameters},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
  volume  = {27},
  number  = {167},
  pages   = {1--70},
  url     = {http://jmlr.org/papers/v27/25-0639.html}
  note    = {Sebastian Salwig and Till Kahlke share first authorship on this work.}
}
```
