"""Build or rebuild test fixtures on the disk"""

import functools
from warnings import warn
import types
import pkgutil
import importlib
import sys

from ..util import Path, version_label, _TEST_DATA_PATH

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
    @functools.wraps(fn)
    def inner(*args, force=False, **kwargs):
        filename, write_fn = fn(*args, **kwargs)
        filepath = _TEST_DATA_PATH / filename

        if not force and filepath.exists():
            raise FileExistsError(str(filepath))
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


def _generate(builders, *args, **kwargs):
    for fn in builders:
        try:
            filepath = fn(*args, **kwargs)
            print("Wrote file \u001b[36m{}\u001b[0m".format(filepath))
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
    parser.add_argument('-p', '--only-versioned', action='store_true', help="only build python-version-specific fixtures")
    parser.add_argument('-n', '--only-nonversioned', action='store_true', help="only build non-python-version-specific fixtures")
    return parser.parse_args()
