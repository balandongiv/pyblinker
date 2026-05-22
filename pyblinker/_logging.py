"""Logging utilities."""

# Authors: The PyBlinker contributors.
# License: BSD-3-Clause
# Copyright: The PyBlinker contributors.
from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
from typing import Callable, Optional

_PACKAGE_NAME = __name__.split(".")[0]
_ROOT_LOGGER: logging.Logger | None = None


def _coerce_level(verbose: str | int | bool | None) -> int:
    """Coerce arbitrary verbosity representations to ``logging`` levels.

    Parameters
    ----------
    verbose : str | int | bool | None
        Verbosity specification.  ``True`` maps to ``INFO`` and ``False`` to
        ``WARNING``.  ``None`` falls back to the ``APP_LOGGING_LEVEL``
        environment variable, defaulting to ``INFO`` when unset.

    Returns
    -------
    int
        Logging level understood by :mod:`logging`.
    """
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    if isinstance(verbose, str):
        return getattr(logging, verbose.upper(), logging.INFO)
    if isinstance(verbose, int):
        return int(verbose)
    env = os.getenv("APP_LOGGING_LEVEL", "INFO")
    return getattr(logging, env.upper(), logging.INFO)


def _ensure_root() -> logging.Logger:
    """Configure and return the package root logger."""
    global _ROOT_LOGGER
    if _ROOT_LOGGER is None:
        logger = logging.getLogger(_PACKAGE_NAME)
        logger.propagate = False
        logger.setLevel(_coerce_level(None))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(name)s: %(message)s"))
        logger.addHandler(handler)
        _ROOT_LOGGER = logger
    return _ROOT_LOGGER


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger.

    Child loggers inherit the configuration of the package root logger.

    Parameters
    ----------
    name : str | None
        Name of the logger. ``None`` returns the package root logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    root = _ensure_root()
    if name in (None, root.name):
        return root
    return logging.getLogger(name)


def set_log_level(
    verbose: str | int | bool | None, *, return_old_level: bool = False
) -> Optional[int]:
    """Globally adjust the log level.

    Parameters
    ----------
    verbose : str | int | bool | None
        Desired verbosity.  See :func:`_coerce_level` for accepted values.
    return_old_level : bool, optional
        When ``True`` the previous level is returned for later restoration.

    Returns
    -------
    int | None
        Previous logging level if ``return_old_level`` is ``True``.
    """
    root = _ensure_root()
    old = root.getEffectiveLevel()
    new = _coerce_level(verbose)
    root.setLevel(new)
    return old if return_old_level else None


def set_log_file(
    fname: os.PathLike | str | None,
    *,
    output_format: str = "%(message)s",
    overwrite: bool | None = None,
) -> None:
    """Redirect logging output to a file or ``stdout``.

    Parameters
    ----------
    fname : path-like | str | None
        Destination file.  ``None`` restores ``stdout`` logging.
    output_format : str, optional
        Formatter pattern applied to log messages.  Defaults to ``%(message)s``
        for concise output similar to MNE.
    overwrite : bool | None, optional
        When ``True`` existing files are truncated.  ``False`` appends to the
        file and ``None`` mimics ``False``.
    """
    root = _ensure_root()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
        root.removeHandler(handler)
    if fname is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        mode = "w" if overwrite else "a"
        handler = logging.FileHandler(fname, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter(output_format))
    root.addHandler(handler)


@contextlib.contextmanager
def _use_log_level(level: str | int | bool | None):
    """Context manager to temporarily adjust the global log level."""

    old = set_log_level(level, return_old_level=True)
    try:
        yield
    finally:
        if old is not None:
            set_log_level(old)


def verbose(arg: Callable | str | int | bool | None | None = None):
    """Decorator and context manager for temporary verbosity changes.

    When used as ``@verbose`` it acts as a decorator that inspects the
    ``verbose`` keyword argument of the wrapped function. Supplying
    ``with verbose('DEBUG')`` returns a context manager that temporarily
    sets the global log level.
    """

    if callable(arg):  # used as a decorator without arguments
        func = arg

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            level = kwargs.pop("verbose", None)
            if level is None:
                return func(*args, **kwargs)
            with _use_log_level(level):
                return func(*args, **kwargs)

        return wrapper
    else:
        return _use_log_level(arg)
