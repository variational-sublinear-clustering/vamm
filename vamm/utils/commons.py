# Copyright (C) 2025 Machine Learning Lab of the University of Oldenburg 
# and Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import contextlib
import time


@contextlib.contextmanager
def timer(msg: str = "Time taken:", decimal_digits=2):
    """
    Measure the runtime of functions or code snippets.

    This function can be used as either a decorator or a context manager to time the execution
    of code blocks, providing the elapsed time in seconds.

    Parameters
    ----------
    msg : str, optional
        Message to be printed together with the elapsed time, by default "Time taken:"

    Examples
    --------
    Using as a decorator or context manager:

    .. code-block:: python

        @timer("Function runtime:")
        def example_function():
            # function code
            pass

        with timer("Code block runtime:"):
            # code block
            pass
    """
    tic = time.monotonic()
    try:
        yield
    finally:
        t = round(time.monotonic() - tic, decimal_digits)
        print(f"{msg} {t} s")


def format_with_prefix(num: float, unit: str = "", base: float = 1000.0) -> str:
    """
    Converts a number into a more human-readable form using appropriate SI prefixes.

    Parameters
    ----------
    num : float
        The number to be formatted.
    unit : str, optional
        The unit to be appended to the formatted number. Default is an empty string.
    base : float, optional
        The base used to determine the prefix. For example, use `base=1024` for binary units like bytes.
        Default is 1000.0.

    Returns
    -------
    str
        The formatted number as a string with an appropriate SI prefix and unit.
    """
    if abs(num) < 1:
        for prefix in ["", "m", "µ", "n", "p", "f", "a", "z", "y", "r"]:
            if abs(num) >= 1:
                return f"{num:3.1f}{prefix}{unit}"
            num *= base
        return f"{num:f}q{unit}"
    else:
        for prefix in ["", "K", "M", "G", "T", "P", "E", "Z"]:
            if abs(num) <= base:
                return f"{num:3.1f}{prefix}{unit}"
            num /= base
    return f"{num:.1f}Y{unit}"


def format_time(seconds: float) -> str:
    """
    Converts a time span in seconds into a more human-readable format.

    Parameters
    ----------
    seconds : float
        The time duration in seconds.

    Returns
    -------
    str
        A string representing the time duration in a more readable form.

    Notes
    -----
    - The function formats time durations as days (d), hours (h), minutes (m), and seconds (s).
    - If the time is less than a minute, it is displayed in seconds with an appropriate SI prefix.
    """
    s = int(seconds)
    d = s // (3600 * 24)
    h = s // 3600 % 24
    m = s % 3600 // 60
    s = s % 3600 % 60
    if d > 0:
        return f"{d:2d}d {h:2d}h"
    elif h > 0:
        return f"{h:2d}h {m:2d}m"
    elif m > 0:
        return f"{m:2d}m {s:2d}s"
    return format_with_prefix(seconds, unit="s")
