"""Build or rebuild test fixtures on the disk"""

import functools
from warnings import warn
import types
import pkgutil
import importlib

from ..util import Path, version_label, _TEST_DATA_PATH


def _write(fn):
    """Helper wrapper for fixture generators.

    The underlying function must return two objects, a relative
    filename and a function that will write to a given path. This
    wrapper will check if the file exists and raise if it does, unless
    called with ``force=True``.
    """
    @functools.wraps(fn)
    def inner(*args, force=False, **kwargs):
        filename, write_fn = fn(*args, **kwargs)
        filepath = _TEST_DATA_PATH / filename

        if not force and filepath.exists():
            raise FileExistsError(str(filepath))
        else:
            write_fn(filepath)
    return inner


def _versioned(fn):
    """Indicates the generated data is python-version-specific"""
    fn.__versioned__ = True
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        filename, write_fn = fn(*args, **kwargs)
        return "{}/{}".format(version_label(), filename), write_fn
    return inner


def _memo(fn):
    fn.__memo__ = None
    @functools.wraps(fn)
    def inner(self):
        if fn.__memo__ is None:
            fn.__memo__ = fn(self)
        return fn.__memo__
    return property(inner)


def _generate(builders, *args, **kwargs):
    for fn in builders:
        try:
            fn(*args, **kwargs)
        except FileExistsError as e:
            warn("File already exists: {} (hint: use \u001b[31m--force\u001b[0m to overwrite)".format(e))


class _FixtureGenABC:
    """Base class for fixture data generators"""
    @property
    def __builders__(self):
        for name in dir(self):
            if name.startswith("build_"):
                yield getattr(self, name)

    def generate_all(self, *args, **kwargs):
        """Generate and write all test fixture data"""
        _generate(self.__builders__, *args, **kwargs)

    def generate_versioned(self, *args, **kwargs):
        """Generate and write all python-version-specific test fixture data"""
        _generate([b for b in self.__builders__ if hasattr(b, '__versioned__')], *args, **kwargs)

    def generate_nonversioned(self, *args, **kwargs):
        """Generate and write all non-python-version-specific test fixture data"""
        _generate([b for b in self.__builders__ if not hasattr(b, '__versioned__')], *args, **kwargs)


def _fixture_generators():
    for name, bound in globals().copy().items():
        if isinstance(bound, _FixtureGenABC):
            yield bound


def generate_all(force=False):
    """Generate and write all test fixture data from all fixture generators"""
    for gen in _fixture_generators():
        gen.generate_all(force=force)


def generate_versioned(force=False):
    """Generate and write all python-version-specific test fixture data from all fixture generators"""
    for gen in _fixture_generators():
        gen.generate_versioned(force=force)


def generate_nonversioned(force=False):
    """Generate and write all non-python-version-specific test fixture data from all fixture generators"""
    for gen in _fixture_generators():
        gen.generate_nonversioned(force=force)


# Dynamically import all submodules
__all__ = []
for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    # Import only non-private/magic modules
    if not name.startswith('_'):
        __all__.append(name)
        full_name = "{}.{}".format(__package__, name)
        importlib.import_module(full_name)
