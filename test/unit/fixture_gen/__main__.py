# Can be run as a script: `python -m test.unit.fixture_gen -h'

from . import generate_all, generate_versioned, generate_nonversioned, __doc__ as moduledoc

import argparse
parser = argparse.ArgumentParser(description=moduledoc)
parser.add_argument('-f', '--force', action='store_true', help="overwrite existing test fixtures")
parser.add_argument('-p', '--only-versioned', action='store_true', help="only build python-version-specific fixtures")
parser.add_argument('-n', '--only-nonversioned', action='store_true', help="only build non-python-version-specific fixtures")
args = parser.parse_args()

if args.only_versioned:
    gen = generate_versioned
elif args.only_nonversioned:
    gen = generate_nonversioned
else:
    gen = generate_all

gen(force=args.force)
