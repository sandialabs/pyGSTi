"""Unit test coverage for pyGSTi"""
import os

if 'FORCE_REGEN_TEST_DATA' in os.environ:
    from .util import _regenerate_fixtures
    _regenerate_fixtures(force=True)

# Suppress pygsti backwards-compatibility warning
os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
