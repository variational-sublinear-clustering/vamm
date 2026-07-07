# Examples

We here provide an example of how to apply **VAMM** to train mixture models. This example include a small demo for quick testing.

## Requirements

Before running the example, ensure that you have successfully [installed](../README.md#installation) **VAMM**.
Next, navigate into the `examples/` directory.

## Quick Demo

For a quick introduction to **VAMM**, run the demo script.
The demo fits different types of Gaussian mixture models (including mixtures of factor analyzers) to a [dataset of hand-written digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Final training objectives for each optimization run are reported.

To run this demo:

```bash
python3 demo.py
```

This demo is designed to run quickly on most PCs and provides a starting point to understand how **VAMM** works.
