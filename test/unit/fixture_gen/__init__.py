"""Build or rebuild test fixtures on the disk"""
from __future__ import absolute_import, unicode_literals
from future.utils import with_metaclass

import functools
from warnings import warn
import types
import pkgutil
import importlib
import sys

from ..util import Path, version_label, _TEST_DATA_PATH
from .. import _NO_REGEN_TEST_DATA

__fixture_generators__ = []


def _instantiate(name, cls):
    """Helper for fixture generator modules.

    Instantiates the given generator class and registers the instance
    as the module export.
    """
    instance = cls()
    if name == '__main__':
        # Run generation for module
        instance._run(_parse_args())
    else:
        # Load normally
        sys.modules[name] = instance
        __fixture_generators__.append(instance)


def _write(fn):
    """Helper wrapper for fixture generators.

    The underlying function must return two objects, a relative
    filename and a function that will write to a given path. This
    wrapper will check if the file exists and raise if it does, unless
    called with ``force=True``.
    """
    fn.__builder__ = True
    @functools.wraps(fn)
    def inner(self, force=False, *args, **kwargs):
        filename, write_fn = fn(self, *args, **kwargs)
        filepath = _TEST_DATA_PATH / filename

        if not force and filepath.exists():
            raise FixtureExistsError(str(filepath))
        else:
            write_fn(filepath)
            return filepath
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


class _FixtureGenMeta(type):
    def __init__(cls, name, bases, dict):
        super(_FixtureGenMeta, cls).__init__(name, bases, dict)
        if hasattr(cls, '__builders__'):
            cls.__builders__ = cls.__builders__.copy()
        else:
            cls.__builders__ = {}
        for name, bound in dict.items():
            if hasattr(bound, '__builder__'):
                cls.__builders__[name] = bound


class _FixtureGenABC(with_metaclass(_FixtureGenMeta, object)):
    """Base class for fixture data generators"""
    def __init__(self):
        super(_FixtureGenABC, self).__init__()
        self._module = sys.modules[self.__module__]
        self.__warningregistry__ = None

    def __getattr__(self, name):
        # Manually override with module lookup for python2 compat
        if name in dir(self):
            return object.__getattribute__(self, name)
        else:
            return self._module.__getattribute__(name)

    def _generate(self, builders, *args, **kwargs):
        if _NO_REGEN_TEST_DATA:
            warn("Skipping generation of {} (\u001b[31m`NO_REGEN_TEST_DATA'\u001b[0m set)".format(
                self.__class__.__name__
            ))
            return
        for fn in builders:
            try:
                filepath = fn(self, *args, **kwargs)
                warn("Wrote file \u001b[36m{}\u001b[0m".format(filepath))
            except FixtureExistsError as e:
                warn("File already exists: {} (hint: use \u001b[31m--force\u001b[0m to overwrite)".format(e))

    def generate_all(self, *args, **kwargs):
        """Generate and write all test fixture data"""
        self._generate(self.__builders__.values(), *args, **kwargs)

    def generate_versioned(self, *args, **kwargs):
        """Generate and write all python-version-specific test fixture data"""
        self._generate([b for b in self.__builders__.values() if hasattr(b, '__versioned__')], *args, **kwargs)

    def generate_nonversioned(self, *args, **kwargs):
        """Generate and write all non-python-version-specific test fixture data"""
        self._generate([b for b in self.__builders__.values() if not hasattr(b, '__versioned__')], *args, **kwargs)

    def _run(self, args):
        if args.only_versioned:
            gen = self.generate_versioned
        elif args.only_nonversioned:
            gen = self.generate_nonversioned
        else:
            gen = self.generate_all
        gen(force=args.force)


def generate_all(force=False):
    """Generate and write all test fixture data from all fixture generators"""
    for gen in __fixture_generators__:
        gen.generate_all(force=force)


def generate_versioned(force=False):
    """Generate and write all python-version-specific test fixture data from all fixture generators"""
    for gen in __fixture_generators__:
        gen.generate_versioned(force=force)


def generate_nonversioned(force=False):
    """Generate and write all non-python-version-specific test fixture data from all fixture generators"""
    for gen in __fixture_generators__:
        gen.generate_nonversioned(force=force)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--force', action='store_true', help="overwrite existing test fixtures")
    parser.add_argument('-p', '--only-versioned', action='store_true',
                        help="only build python-version-specific fixtures")
    parser.add_argument('-n', '--only-nonversioned', action='store_true',
                        help="only build non-python-version-specific fixtures")
    return parser.parse_args()


class FixtureExistsError(IOError):
    pass
