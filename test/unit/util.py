"""Utilities shared by unit tests"""
from unittest import TestCase

import sys
import numpy as np
import numbers

# Test modules should import these generic names rather than importing the modules directly:

# `pathlib' is standard as of 3.4, but has been backported as pathlib2
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

# `mock' was refactored into unittest in 3.3
try:
    from unittest import mock
except ImportError:
    import mock

_TEST_ROOT_PATH = Path(__file__).parent.parent.absolute()
_TEST_DATA_PATH = _TEST_ROOT_PATH / "data"


def version_label():
    """Get the label used internally for this python version.

    This is mainly used to identify version-specific test fixtures
    """
    return "v{}".format(sys.version_info.major)


class BaseCase(TestCase):
    def assertArraysAlmostEqual(self, a, b, places=7, msg=None, delta=None):
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, places=places, msg=msg, delta=delta)

    def assertArraysEqual(self, a, b, msg=None):
        self.assertTrue(np.array_equal(a, b), msg=msg)

    def assertDictContainsSubset(self, subset, dictionary, places=7, msg=None):
        """Assert that `dictionary` contains each key-value pair in `subset`

        Array-type and numeric values are compared as "almost equal"
        """
        for k, v in subset.items():
            self.assertIn(k, dictionary, msg=msg)
            e = dictionary[k]
            if isinstance(v, np.ndarray) and isinstance(e, np.ndarray):
                self.assertArraysAlmostEqual(v, e, places=7, msg=msg)
            elif isinstance(v, numbers.Number) and isinstance(e, numbers.Number):
                self.assertAlmostEqual(v, e, places=7, msg=msg)
            else:
                self.assertEqual(v, e, msg=msg)

    def fixture_data(self, data_file_name):
        """Returns the absolute path to a test fixture data file"""
        # First try without a version or architecture
        noarch_file = _TEST_DATA_PATH / data_file_name
        if not noarch_file.exists():
            # If the no-arch data file doesn't exist, try looking in a python version-specific data path
            version_path = _TEST_DATA_PATH / version_label()
            if version_path.exists():
                version_file = version_path / data_file_name
                if version_file.exists():
                    return str(version_file)

        # As fallback, just return the no-arch filename and let the caller deal with it
        return str(noarch_file)

    def debug(self, debugger=None):
        """Helper factory for debugger breakpoints.

        This sets up certain useful debugging environment things, then returns a function to embed a debugger.

        To use this method, call the returned function, like this:

            self.debug()()

        This method is used in a weird way so that the debugger starts in the caller's stack frame, since you probably
        don't care about debugging this method itself.

        By default, if the `bpython` package is installed, this will use `bpdb`, the bpython debugger. bpython is an
        enhanced python interpreter that offers a number of advantages over IPython. If bpython is not installed, this
        will try the IPython embedded debugger, and if that's not installed either, we default to the built-in
        debugger. Alternatively, if the `debugger` argument is given, we'll use that as the debugger.

        Parameters
        ----------
        debugger : str, optional
            The debugger to use; one of ('bpdb', 'ipython', 'pdb'). By default, tries bpdb, falls back on ipython, then
            finally falls back on pdb if neither of the previous are available.

        Returns
        -------
        function
            Entry point to the debugger. In most cases you'll want to call this immediately, like this:

                self.debug()()
        """

        np.set_printoptions(precision=4,  # usually better for debugging
                            linewidth=120,  # this isn't the 40s, grandpa, we have 1080p now
                            suppress=True)  # fixed-point notation gets hard to read

        def debug_bpython():
            import bpdb
            return bpdb.set_trace

        def debug_ipython():
            import IPython
            return IPython.embed

        def debug_pdb():
            import pdb
            return pdb.set_trace

        if debugger is not None:
            return {
                'bpython': debug_bpython,
                'bpdb': debug_bpython,
                'ipython': debug_ipython,
                'pdb': debug_pdb,
                'default': debug_pdb
            }[debugger.lower()]()
        else:
            # Try bpython, fall back to ipython, then to pdb
            try:
                debug = debug_bpython()
            except ModuleNotFoundError:
                try:
                    debug = debug_ipython()
                except ModuleNotFoundError:
                    debug = debug_pdb()
            return debug
