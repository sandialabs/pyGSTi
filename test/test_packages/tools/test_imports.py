import subprocess
import sys


def test_import_pygsti_does_not_eagerly_import_cvxpy_when_available():
    code = """
import importlib.util
import sys

cvxpy_available = importlib.util.find_spec("cvxpy") is not None

import pygsti  # noqa: F401

if cvxpy_available and "cvxpy" in sys.modules:
    raise SystemExit("import pygsti eagerly imported optional dependency cvxpy")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
