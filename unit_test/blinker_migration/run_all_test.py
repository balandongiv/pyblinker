"""Execute migration tests within the local ``pyblinker`` directory.

This convenience script discovers and runs every module in
``unit_test/blinker_migration/pyblinker`` matching the ``test_*.py`` pattern.
It is primarily intended for manual debugging during development.
"""

from __future__ import annotations

import multiprocessing
import sys
import unittest
from pathlib import Path

# Ensure the repository root is on the Python path so imports like ``pyblinker``
# succeed when running this script directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Directory containing the migration tests for ``pyblinker``.
TEST_DIR = Path(__file__).resolve().parent / "pyblinker"


def main() -> None:
    """Discover and execute tests in :mod:`unit_test.blinker_migration.pyblinker`."""
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(TEST_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
