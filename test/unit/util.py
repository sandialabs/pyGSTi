"""Utilities shared by unit tests"""
import sys
import numpy as np
import numbers
import tempfile
import functools
import types
from contextlib import contextmanager
from unittest import TestCase, skipUnless

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


def with_temp_path(filename=None):
    """Decorator version of ``BaseCase.temp_path``"""
    arg_fn = None
    if isinstance(filename, types.FunctionType):
        # Decorator was used without calling, so `filename' is actually the decorated function
        arg_fn = filename
        filename = None

    def decorator(fn):
        @functools.wraps(fn)
        def inner(self, *args, **kwargs):
            with self.temp_path(filename) as tmp_path:
                return fn(self, tmp_path, *args, **kwargs)
        return inner
    if arg_fn is not None:
        return decorator(arg_fn)
    else:
        return decorator


def with_temp_file(contents):
    """Helper decorator for file I/O testing.

    The decorated method will be called with the path of a temporary
    file containing the given contents.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def inner(self, tmp_path, *args, **kwargs):
            with open(tmp_path, 'w') as f:
                f.write(contents)
            return fn(self, tmp_path, *args, **kwargs)
        return with_temp_path(inner)
    return decorator


def _regenerate_fixtures(force=False):
    """Regenerate missing test fixture files.

    This call can be expensive, so use sparingly.
    """
    version_path = _TEST_DATA_PATH / version_label()
    version_path.mkdir(parents=True, exist_ok=True)
    from .fixture_gen import __main__ as gen
    gen._load_all_generators()
    gen.generate_all(force=force)


class BaseCase(TestCase):
    def assertArraysAlmostEqual(self, a, b, **kwargs):
        """Assert that two arrays are equal to within a certain precision.

        Internally, this just wraps a call to
        ``unittest.assertAlmostEqual`` with the operand difference
        norm and zero.

        Parameters
        ----------
        a, b: matrices or vectors
            The two operands to compare
        **kwargs:
            Additional arguments to pass to ``unittest.assertAlmostEqual``
        """
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, **kwargs)

    def assertArraysEqual(self, a, b, **kwargs):
        """Assert that two arrays are exactly equal.

        Internally, this just wraps a call to ``numpy.array_equal``
        in an assertion.

        Parameters
        ----------
        a, b: matrices or vectors
            The two operands to compare
        **kwargs:
            Additional arguments to pass to ``unittest.assertTrue``
        """
        self.assertTrue(np.array_equal(a, b), **kwargs)

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

    def fixture_path(self, filename, can_retry=True):
        """Returns the absolute path to a test fixture data file, if it exists.

        Storage of test fixtures in the filesystem is handled
        automatically. This is intentionally inconvenient; test
        fixtures written in the filesystem are inherently transient
        and are not git-tracked. Developers should not manually add
        test fixtures to the data directory; instead, they should
        write a module under ``test.unit.fixture_gen`` to generate
        their fixtures programatically, so their test results are
        reproducible.

        Internally, this method first looks for a
        non-python-version-specific, non-architecture-specific fixture
        with the given filename. Failing that, it will look for a
        version-specific fixture, and finally will either give up
        (failling the test where it was called) or optionally try
        again after regenerating missing test fixtures.

        You can manually generate or regenerate test fixtures by
        running ``python -m test.unit.fixture_gen`` or whatever
        fixture_gen module is appropriate. Use the ``--help`` flag to
        learn more. Alternatively, you can run any test under
        ``test.unit`` with the ``PYGSTI_REGEN_REF_FILES`` environment
        variable set.

        By default, if this method fails to locate a test fixture, it
        will try to regenerate missing fixtures. This can be an
        expensive operation. This behavior can be suppressed by
        calling with ``can_retry=False``, or by setting the
        ``NO_REGEN_TEST_DATA`` environment variable.

        Parameters
        ----------
        filename: str
            The filename of the fixture to load. This is the filename
            used to write it by the respective ``fixture_gen`` module.
        can_retry: bool
            If ``True`` (default), if the given fixture can't be
            found, try again after regenerating missing test fixtures.

        Returns
        -------
        ``pathlib.Path``
            The filesystem path of the fixture

        See Also
        --------
        ``test.unit.fixture_gen`` : test fixture generation
        """
        # First try without a version or architecture
        noarch_file = _TEST_DATA_PATH / filename
        if noarch_file.exists():
            return noarch_file
        else:
            # If the no-arch data file doesn't exist, try looking in a python version-specific data path
            version_path = _TEST_DATA_PATH / version_label()
            if version_path.exists():
                version_file = version_path / filename
                if version_file.exists():
                    return version_file

        # As fallback, regenerate fixtures and retry
        if can_retry:
            _regenerate_fixtures(force=False)
            return self.fixture_path(filename, can_retry=False)
        else:
            self.fail("Could not locate test fixture data {}".format(filename))

    @contextmanager
    def temp_path(self, filename=None):
        """Provides a context with the path of a temporary file.

        This is distinct from the contexts provided by tempfile in
        that this method yields the path of the temporary file, so the
        underlying file may be opened or closed inside the context as
        the caller pleases.

        Under the hood, this actually creates the file in a temporary
        directory. This directory will be cleaned up when the context
        closes, including the returned file and any other siblings.

        Parameters
        ----------
        filename: str, optional
            Optionally, the name of the file. By default, one will be
            randomly generated.

        Yields
        ------
        ``pathlib.Path``
            A Path object representing the path of the temporary file.

        See Also
        --------
        ``test.unit.util.with_temp_file`` : decorator version
        """

        filename = filename or "temp_file"  # yeah looks random to me
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            # Yield to context with temporary path
            yield tmp_path
            # TemporaryDirectory will be cleaned up on close

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
