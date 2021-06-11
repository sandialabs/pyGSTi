"""Tests for pygsti file IO"""

import functools
import types
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from ..testutils.basecase import temp_files, BaseTestCase

TEMP_FILE_PATH = Path(__file__).parent.parent.absolute() / temp_files

from .references import generator


def with_temp_path(filename=None):
    """Decorator version of ``IOBase.temp_path``"""
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


class IOBase(BaseTestCase):
    # override BaseTestCase setup/teardown which would otherwise chdir
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def assertFilesEquivalent(self, path_a, path_b, mode='r'):
        """Helper method to assert that the contents of two files are equivalent."""
        def is_comment(line):
            return line.startswith('#') and not line.startswith('##')

        def next_semantic(f):
            while True:
                line = f.readline()
                if line == '':
                    return None
                if not is_comment(line):
                    return line.rstrip()

        with open(path_a, mode) as f_a:
            with open(path_b, mode) as f_b:
                while True:
                    line_a = next_semantic(f_a)
                    line_b = next_semantic(f_b)
                    if line_a is None or line_b is None:
                        if line_a is None and line_b is None:
                            break
                        else:
                            self.fail("Early end-of-file")
                    self.assertEqual(line_a, line_b)

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
        ``with_temp_file`` : decorator version
        """

        filename = filename or "temp_file"  # yeah looks random to me
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            # Yield to context with temporary path
            yield str(tmp_path)
            # TemporaryDirectory will be cleaned up on close

    def reference_path(self, filename):
        """Returns the absolute path to a test reference data file, if it exists.

        Parameters
        ----------
        filename: str
            The filename of the reference to load.

        Returns
        -------
        str
            The filesystem path of the reference file

        """
        file_path = TEMP_FILE_PATH / filename
        if not file_path.exists():
            generator.write(file_path)
        return str(file_path)
