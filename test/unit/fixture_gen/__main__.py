# Can be run as a script: `python -m test.unit.fixture_gen -h'
import argparse
import pkgutil
import importlib

from . import __path__ as modulepath, __fixture_generators__, \
    generate_all, generate_versioned, generate_nonversioned, _parse_args


def _load_all_generators():
    """Dynamically import all submodules"""
    for loader, name, is_pkg in pkgutil.walk_packages(modulepath):
        # Import only non-private/magic modules
        if not name.startswith('_'):
            full_name = "{}.{}".format(__package__, name)
            importlib.import_module(full_name)


if __name__ == '__main__':
    args = _parse_args()

    if args.only_versioned:
        gen = generate_versioned
    elif args.only_nonversioned:
        gen = generate_nonversioned
    else:
        gen = generate_all

    _load_all_generators()
    gen(force=args.force)
