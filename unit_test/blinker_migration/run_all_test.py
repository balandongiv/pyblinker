"""Execute all migration tests in this directory.

This convenience script discovers and runs every module in
``unit_test/blinker_migration`` matching the ``test_*.py`` pattern.
It is primarily intended for manual debugging during development.
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import pytest

# Ensure the repository root is on the Python path so imports like ``pyblinker``
# succeed when running this script directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Directory containing the migration tests.
TEST_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Discover and execute tests in :mod:`unit_test.blinker_migration`."""
    multiprocessing.set_start_method("spawn", force=True)
    pytest.main([str(TEST_DIR)])


if __name__ == "__main__":
    main()
