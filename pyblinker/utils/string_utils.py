"""String manipulation helpers used across pyblinker utilities."""

from __future__ import annotations

import ast
from typing import Any

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def safe_literal_eval(value: str) -> Any:
    """Safely evaluate ``value`` using :func:`ast.literal_eval`.

    The function catches parsing errors and simply returns the original string
    when the content cannot be interpreted as a Python literal. Unlike
    :func:`ast.literal_eval`, the helper never raises ``SyntaxError`` or
    ``ValueError`` which makes it convenient when dealing with metadata that may
    contain user-provided strings.
    """

    logger.debug("Attempting literal eval for value: %s", value)
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        logger.debug("Falling back to original string for value: %s", value)
        return value


__all__ = ["safe_literal_eval"]
