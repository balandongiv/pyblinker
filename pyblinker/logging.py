"""Centralized logging utilities (MNE-style).

This module provides a single, centralized logging system for the package,
modeled closely after MNE-Python’s ``mne.utils._logging``. It offers:

- One **package logger** (``logger``) configured once, with ``propagate=False``.
- **Global verbosity control** via :func:`set_log_level`, accepting ``bool``,
  string level names, integers, or ``None`` (which reads an environment
  variable).
- **Temporary overrides** via :class:`use_log_level` (context manager) and the
  :func:`verbose` decorator/context manager (function-level override).
- **File routing** via :func:`set_log_file`, including doctest-/notebook-safe
  stdout handling and a user-facing warning when appending implicitly.
- **Frame breadcrumbs** (``add_frames``) that inject compact call-site stacks
  into log records, toggled at runtime, like MNE’s ``_FrameFilter`` does.
- **Testing/UX helpers**:
  - :class:`catch_logging` to capture logs in-memory.
  - :func:`wrapped_stdout` to funnel ``print()`` to the logger.
  - :func:`warn` to pin warnings to external call sites and avoid duplicate
    stdout prints unless writing to a file-like handler or frames are enabled.

**Environment variable**

- ``PYBLINKER_LOGGING_LEVEL`` controls the default log level when
  :func:`set_log_level` / :func:`_parse_verbose` receive ``None``. Accepts
  ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or ``CRITICAL``. Defaults to
  ``INFO`` if unset.

**Why this design?**

- MNE keeps console output minimal and consistent, while still allowing
  fine-grained control (global and per-call). Replicating that approach removes
  scattered logger configuration across modules and gives you predictable,
  testable behavior with clear entry points for users and developers.

"""

from __future__ import annotations

import contextlib
import functools
import inspect
import io
import logging
import os
import os.path as op
import re
import sys
import warnings
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Package/root logger
# --------------------------------------------------------------------------- #

_PACKAGE_NAME = __name__.split(".")[0]
#: Public package logger (like ``mne.utils.logger``). Modules should use either:
#: ``from <pkg>.logging import logger`` or ``get_logger(__name__)``.
logger = logging.getLogger(_PACKAGE_NAME)
logger.propagate = False  # do not bubble to root (avoids duplicate prints)


# --------------------------------------------------------------------------- #
# Frame breadcrumbs (add_frames)
# --------------------------------------------------------------------------- #

class _FrameFilter(logging.Filter):
    """Inject compact call-stack info into ``record.frame_info`` when enabled.

    The filter is attached once to the package logger. When ``add_frames`` is
    non-zero, the active formatter should include ``%(frame_info)s`` so that
    stack breadcrumbs are displayed.

    Notes
    -----
    The representation mirrors MNE’s format, using box-drawing characters to
    render a small tree of the call stack with the oldest frame at the top.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_frames: int = 0  # 0 disables frame info

    def filter(self, record: logging.LogRecord) -> bool:
        record.frame_info = "Unknown"
        if self.add_frames:
            # 5 frames to get out of this module and logging internals
            frames = _frame_info(5 + self.add_frames)[5:][::-1]
            if frames:
                frames[-1] = (frames[-1] + " :").ljust(30)
                if len(frames) > 1:
                    frames[0] = "┌" + frames[0]
                    frames[-1] = "└" + frames[-1]
                for i in range(1, len(frames) - 1):
                    frames[i] = "├" + frames[i]
                record.frame_info = "\n".join(frames)
        return True


_filter = _FrameFilter()
logger.addFilter(_filter)


def _ensure_stdout_handler() -> None:
    """Ensure a single stdout StreamHandler with a message-only formatter.

    Uses :class:`WrapStdOut` to cooperate with doctest / sphinx-gallery /
    notebook environments that monkey-patch ``sys.stdout``.
    """
    if not logger.handlers:
        h = logging.StreamHandler(WrapStdOut())
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)


_ensure_stdout_handler()


# --------------------------------------------------------------------------- #
# Verbosity parsing
# --------------------------------------------------------------------------- #

_LOGGING_TYPES = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


def _parse_verbose(verbose: bool | str | int | None) -> int:
    """Coerce verbosity into a logging level (MNE-style).

    Parameters
    ----------
    verbose : bool | str | int | None
        - ``None``: read ``PYBLINKER_LOGGING_LEVEL`` (default ``"INFO"``).
        - ``bool``: ``True`` → ``"INFO"``, ``False`` → ``"WARNING"``.
        - ``str``: one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
          ``CRITICAL`` (case-insensitive).
        - ``int``: a valid numeric logging level.

    Returns
    -------
    int
        The logging level to apply.

    Raises
    ------
    TypeError
        If the input type is not one of the supported types.
    ValueError
        If a string value is not a valid level name.

    Notes
    -----
    This is intentionally strict to catch typos early, matching MNE’s behavior.
    """
    if verbose is None:
        verbose = os.getenv("PYBLINKER_LOGGING_LEVEL", "INFO")
    if isinstance(verbose, bool):
        verbose = "INFO" if verbose else "WARNING"
    if isinstance(verbose, str):
        v = verbose.upper()
        if v not in _LOGGING_TYPES:
            raise ValueError(
                f"verbose must be one of {list(_LOGGING_TYPES)}, "
                f"an int, bool, or None; got {verbose!r}"
            )
        return _LOGGING_TYPES[v]
    if isinstance(verbose, int):
        return int(verbose)
    raise TypeError("verbose must be bool | str | int | None")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger (or a child logger).

    Parameters
    ----------
    name : str | None
        The logger name. ``None`` or the package name returns the package
        logger. Any other dotted name returns a namespaced child logger.

    Returns
    -------
    logging.Logger
        The logger instance. Child loggers inherit configuration from the
        package logger.
    """
    return logger if name in (None, logger.name) else logging.getLogger(name)


def set_log_level(
    verbose: bool | str | int | None,
    *,
    return_old_level: bool = False,
    add_frames: int | bool | None = None,
) -> Optional[int]:
    """Set the global log level (and optionally toggle frame breadcrumbs).

    Parameters
    ----------
    verbose : bool | str | int | None
        Coerced via :func:`_parse_verbose`.
    return_old_level : bool
        If ``True``, return the previous level (so callers can restore it).
    add_frames : int | bool | None
        When not ``None``, enable (``>0``/``True``) or disable (``0``/``False``)
        frame breadcrumbs and update all handler formatters to include or drop
        ``%(frame_info)s`` accordingly.

    Returns
    -------
    int | None
        Previous level when ``return_old_level`` is ``True``, else ``None``.
    """
    old_level = logger.level
    new_level = _parse_verbose(verbose)
    if new_level != old_level:
        logger.setLevel(new_level)

    if add_frames is not None:
        _filter.add_frames = int(add_frames)
        fmt = "%(message)s"
        if _filter.add_frames:
            fmt = "%(frame_info)s " + fmt
        for h in logger.handlers:
            h.setFormatter(logging.Formatter(fmt))
    return old_level if return_old_level else None


def _remove_close_handlers(lgr: logging.Logger) -> None:
    """Remove only our file/stream handlers and close file handlers we added.

    This intentionally avoids touching foreign handlers to play nicely with
    host applications or test runners that add their own handlers.
    """
    for h in list(lgr.handlers):
        if isinstance(h, (logging.FileHandler, logging.StreamHandler)):
            if isinstance(h, logging.FileHandler):
                try:
                    h.close()
                except Exception:
                    pass
            lgr.removeHandler(h)


def set_log_file(
    fname: os.PathLike | str | None,
    *,
    output_format: str = "%(message)s",
    overwrite: bool | None = None,
) -> None:
    """Route logs to a file (or back to stdout), mirroring MNE semantics.

    Parameters
    ----------
    fname : path-like | str | None
        Destination file path. ``None`` restores stdout logging. To suppress
        console logs, raise the level via :func:`set_log_level`.
    output_format : str
        Formatter string (e.g., ``"%(asctime)s - %(levelname)s - %(message)s"``).
        Defaults to ``"%(message)s"`` for minimal, MNE-like output.
    overwrite : bool | None
        - ``True``: truncate existing file.
        - ``False``: append to existing file.
        - ``None``: append **and** emit a warning once if the file exists,
          mirroring MNE’s UX.

    Notes
    -----
    When restoring stdout, a :class:`WrapStdOut` stream is used so doctest/
    sphinx-gallery/notebooks can capture output reliably.
    """
    _remove_close_handlers(logger)
    if fname is not None:
        if op.isfile(fname) and overwrite is None:
            warnings.warn(
                "Log entries will be appended to the file. Use overwrite=False "
                "to avoid this message in the future.",
                RuntimeWarning,
                stacklevel=2,
            )
            overwrite = False
        mode = "w" if overwrite else "a"
        h = logging.FileHandler(fname, mode=mode, encoding="utf-8")
    else:
        h = logging.StreamHandler(WrapStdOut())
    h.setFormatter(logging.Formatter(output_format))
    logger.addHandler(h)


class use_log_level:
    """Context manager to temporarily change log level (and frame breadcrumbs).

    Parameters
    ----------
    verbose : bool | str | int | None
        Temporary level to apply. ``None`` leaves the level unchanged.
    add_frames : int | bool | None
        Temporary breadcrumb setting. ``None`` keeps the current setting.

    Examples
    --------
    >>> from <yourpkg>.logging import use_log_level, logger
    >>> with use_log_level("DEBUG", add_frames=3):
    ...     logger.debug("debug with frames")

    See Also
    --------
    verbose : decorator/context manager version for per-call overrides
    set_log_level : set level globally
    """
    def __init__(self, verbose: bool | str | int | None = None, *, add_frames: int | bool | None = None):
        self._level = verbose
        self._add_frames = add_frames
        self._old_frames = _filter.add_frames

    def __enter__(self):
        self._old_level = set_log_level(self._level, return_old_level=True, add_frames=self._add_frames)

    def __exit__(self, *exc):
        add_frames = self._old_frames if self._add_frames is not None else None
        set_log_level(self._old_level, add_frames=add_frames)


def verbose(func_or_level: Callable | bool | str | int | None = None):
    """Decorator/context manager matching MNE’s ``verbose``.

    Usage
    -----
    - **Decorator**: ``@verbose`` on functions that accept ``verbose=None``.
      If the caller passes ``verbose="INFO"`` (or any accepted form),
      the level is applied for that call and restored afterward.
    - **Context manager**: ``with verbose("DEBUG"):``, which is equivalent to
      ``with use_log_level("DEBUG")``.

    Notes
    -----
    Like MNE, this enforces that decorated callables accept a ``verbose``
    keyword (typically defaulting to ``None``). If missing, a helpful runtime
    error is raised.
    """
    if not callable(func_or_level):  # context-manager usage
        return use_log_level(func_or_level)

    func = func_or_level
    sig = inspect.signature(func)
    if "verbose" not in sig.parameters:
        @functools.wraps(func)
        def _err_wrapper(*args, **kwargs):
            raise RuntimeError(f"Function/method {func.__qualname__} does not accept verbose parameter")
        return _err_wrapper

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        level = kwargs.pop("verbose", None)
        if level is None:
            return func(*args, **kwargs)
        with use_log_level(level):
            return func(*args, **kwargs)

    return _wrapper


# --------------------------------------------------------------------------- #
# Utilities mirrored from MNE (trimmed but faithful)
# --------------------------------------------------------------------------- #

class WrapStdOut:
    """Proxy that mirrors the current ``sys.stdout`` at attribute access time.

    Some tools (doctest, sphinx-gallery, notebook runners) monkey-patch
    ``sys.stdout`` dynamically. Using this wrapper ensures our StreamHandler
    always writes into the *current* ``stdout`` sink rather than a stale file
    descriptor captured at import time.
    """
    def __getattr__(self, name: str):
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        raise AttributeError(f"'file' object has no attribute '{name}'")


class ClosingStringIO(io.StringIO):
    """StringIO whose :meth:`getvalue` closes the buffer by default.

    This is handy in tests to release memory/file descriptors as soon as the
    captured value is read.
    """
    def getvalue(self, close: bool = True) -> str:  # type: ignore[override]
        out = super().getvalue()
        if close:
            self.close()
        return out


class catch_logging:
    """Capture package logger output to an in-memory buffer.

    This context manager removes existing package handlers, attaches an
    in-memory handler, and restores stdout logging on exit. It optionally
    changes the verbosity while active.

    Parameters
    ----------
    verbose : bool | str | int | None
        Temporary verbosity to apply inside the context.

    Returns
    -------
    io.StringIO
        A buffer whose content is the captured logger output.

    Notes
    -----
    The handler is marked as ``_pyblinker_file_like`` so :func:`warn`
    can decide whether to also mirror the message to the logger (to avoid
    duplicate console prints when not writing to files).
    """
    def __init__(self, verbose: bool | str | int | None = None):
        self.verbose = verbose

    def __enter__(self):
        self._ctx = use_log_level(self.verbose) if self.verbose is not None else contextlib.nullcontext()
        self._data = ClosingStringIO()
        self._lh = logging.StreamHandler(self._data)
        self._lh.setFormatter(logging.Formatter("%(message)s"))
        # mark for warn()
        self._lh._pyblinker_file_like = True  # type: ignore[attr-defined]
        _remove_close_handlers(logger)
        logger.addHandler(self._lh)
        self._ctx.__enter__()
        return self._data

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        logger.removeHandler(self._lh)
        set_log_file(None)


_verbose_dec_re = re.compile(r"^<decorator-gen-[0-9]+>$")


def warn(
    message: str,
    category: type[Warning] = RuntimeWarning,
    module: str = _PACKAGE_NAME,
    ignore_namespaces: tuple[str, ...] = (_PACKAGE_NAME,),
) -> None:
    """Emit a warning pinned to an external call site and (optionally) log it.

    This behaves like MNE’s ``warn``:
    1. Walk back the stack to find a frame **outside** our package namespace
       (or inside ``tests/``), so the warning points to user code.
    2. Emit via :func:`warnings.warn_explicit` at that location.
    3. Also call ``logger.warning`` **only if** we are writing to a file-like
       handler (including :class:`catch_logging`) or frame breadcrumbs are
       enabled. This avoids duplicate console output in typical setups.

    Parameters
    ----------
    message : str
        Warning message.
    category : type[Warning]
        Warning category (default ``RuntimeWarning``).
    module : str
        Module name for the warning metadata (default: package name).
    ignore_namespaces : tuple[str, ...]
        Package/module prefixes to treat as internal frames while searching for
        the external call site.
    """
    # Identify external call site
    root_dirs = []
    for ns in ignore_namespaces:
        try:
            root_dirs.append(op.dirname(__import__(ns).__file__))  # type: ignore[attr-defined]
        except Exception:
            pass

    frame = inspect.currentframe()
    fname, lineno = "<unknown>", 0
    if logger.level <= logging.WARNING:
        try:
            frame = frame.f_back if frame else None
            while frame:
                fn = frame.f_code.co_filename
                ln = frame.f_lineno
                if not _verbose_dec_re.search(fn):
                    if not any(fn.startswith(rd) for rd in root_dirs) or op.basename(op.dirname(fn)) == "tests":
                        fname, lineno = fn, ln
                        break
                frame = frame.f_back
        finally:
            del frame
        warnings.warn_explicit(
            message,
            category,
            fname,
            lineno,
            module,
            globals().get("__warningregistry__", {}),
        )
    # Mirror to logger only when appropriate (avoid duplicate console prints)
    if any(
        isinstance(h, logging.FileHandler) or getattr(h, "_pyblinker_file_like", False)
        for h in logger.handlers
    ) or _filter.add_frames:
        logger.warning(message)


@contextlib.contextmanager
def wrapped_stdout(indent: str = "", cull_newlines: bool = False):
    """Send ``print()`` output to ``logger.info`` with optional indentation.

    Parameters
    ----------
    indent : str
        Prefix added to each line of captured output.
    cull_newlines : bool
        If ``True``, collapse trailing blank lines.

    Notes
    -----
    This is useful when integrating third-party code that prints to stdout but
    you want consistent, centralized logging in your application.
    """
    orig = sys.stdout
    buf = ClosingStringIO()
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = orig
        pending = 0
        for line in buf.getvalue().split("\n"):
            if not line.strip() and cull_newlines:
                pending += 1
                continue
            for _ in range(pending):
                logger.info("\n")
            pending = 0
            logger.info(indent + line)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _frame_info(n: int) -> list[str]:
    """Collect up to ``n`` frames of ``module:lineno`` strings (inner→outer)."""
    frame = inspect.currentframe()
    infos: list[str] = []
    try:
        frame = frame.f_back if frame else None
        for _ in range(n):
            if not frame:
                break
            try:
                name = frame.f_globals["__name__"]
            except KeyError:
                pass
            else:
                infos.append(f"{name.lstrip(_PACKAGE_NAME + '.')}:{frame.f_lineno}")
            frame = frame.f_back
        return infos or ["unknown"]
    except Exception:
        return ["unknown"]
    finally:
        del frame
