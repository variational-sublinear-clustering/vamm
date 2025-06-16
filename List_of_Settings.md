# List of Settings

Below is a list of behavior-changing settings via [Macros](#macros), [Environment Variables](#environment-variables) or [Flags](#flags).

## Macros

All macros are named using uppercase letters with snake-style.
These macros must be set at compile time.

Example on how to set a macros in the `setup.py`:

```python
define_macros = [("CLUSTERING_PRECISION", "double"), ...]
```

### CLUSTERING_PRECISION

- Possible values: `float`, `double`  
- Default value: `double`

Sets the datatype for floating-point numbers, i.e., the precision.
Using `float` leads to faster numerical operations compared to `double`, but can also lead to numerical instabilities. It is recommended to use `double`.

## Environment Variables

All environment variables are named using uppercase letters with snake-style.
An environment variables should be set **before** the python modules are loaded to make sure it takes effect.

Example on how to set a environment variable in bash:

```bash
export OMP_NUM_THREADS=4
```

Example on how to set a environment variable in python (using the module `os`):

```python
os.environ["OMP_NUM_THREADS"] = "4"
```

### OMP_NUM_THREADS

- Possible values: positive integers not larger than the number of threads
- Default value: None

Sets the maximum number of threads for shared memory parallelization

## Flags

Flags are variables that can be set or changed during the runtime of a model.

### model.verbose_discard

- Possible values: `True`, `False`
- Default value: `False`

Whether to print a verbose message whenever a component is discarded
