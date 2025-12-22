"""Utilities shared by unit tests"""
import functools
import os
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def needs_cvxpy(fn):
    """Shortcut decorator for skipping tests that require CVXPY"""
    try:
        import cvxpy
        return unittest.skipIf('SKIP_CVXPY' in os.environ, "skipping cvxpy tests")(fn)
    except ImportError:
        return unittest.skip('cvxpy is not installed')


def needs_deap(fn):
    """Shortcut decorator for skipping tests that require deap"""
    try:
        import deap
        return unittest.skipIf('SKIP_DEAP' in os.environ, "skipping deap tests")(fn)
    except ImportError:
        return unittest.skip('deap is not installed')


def needs_matplotlib(fn):
    """Shortcut decorator for skipping tests that require matplotlib"""
    try:
        import matplotlib
        return unittest.skipIf('SKIP_MATPLOTLIB' in os.environ, "skipping matplotlib tests")(fn)
    except ImportError:
        return unittest.skip('matplotlib is not installed')


def with_temp_path(fn):
    """Decorator version of ``BaseCase.temp_path``"""
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        with self.temp_path() as tmp_path:
            return fn(self, tmp_path, *args, **kwargs)
    return inner


class BaseCase(unittest.TestCase):
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

    @contextmanager
    def assertNoWarns(self, category=Warning):
        """Asserts that nothing in the enclosed context generates a warning

        Parameters
        ----------
        category: ``Warning``, optional
            This assertion will fail only if a warning descended from
            this type is generated in the context. Since all warnings
            are derived from ``Warning``, by default this will fail on
            any warning.
        """

        with warnings.catch_warnings(record=True) as warns:
            yield  # yield to context

            for w in warns:
                if issubclass(w.category, category):
                    self.fail("{} was triggered".format(category.__name__))

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
        str
            The path of the temporary file.

        See Also
        --------
        ``test.unit.util.with_temp_file`` : decorator version
        """

        filename = filename or "temp_file"  # yeah looks random to me
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            # Yield to context with temporary path
            yield str(tmp_path)
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


class Namespace(object):
    """Namespace for shared test fixtures.

    This is included as an alternative to ``types.SimpleNamespace``,
    which may be absent from earlier python versions.

    This implementation is included for convenience and does not
    implicitly protect members from modification. When using a
    ``Namespace`` for module- or package-level fixtures, take care
    that any mutable members are used safely.

    Parameters
    ----------
    **kwargs
        Initial members of the namespace. Members may also be assigned
        after initialization, either directly or annotated via the
        ``Namespace.property`` and ``Namespace.memo`` decorators.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__ns_props__ = {}

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError as err:
            if name in self.__ns_props__:
                return self.__ns_props__[name](self)
            # else:
            #     raise err
            return None
            # ^ necessary to avoid cursed issues that can arise when a call to pyest.main(...)
            #   ends up triggering a test which needs the Namespace class.

    def property(self, fn):
        """Dynamic namespace property"""
        self.__ns_props__[fn.__name__] = fn

    def memo(self, fn):
        """Memoized namespace property

        Memoized properties may be used to efficiently compose
        namespace members from other memoized members, which could
        otherwise be prohibitively expensive to repeatedly generate.

        Memoization should only be used when you want to reuse
        previously computed values. Accordingly, it doesn't make sense
        to memoize functions with side-effects, or impure functions
        like time().
        """
        fn.__memo__ = None
        @functools.wraps(fn)
        def inner(self):
            if fn.__memo__ is None:
                fn.__memo__ = fn(self)
            return fn.__memo__
        self.property(inner)
