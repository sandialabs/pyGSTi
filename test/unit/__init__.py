"""Unit test coverage for pyGSTi"""
import os

if 'PYGSTI_REGEN_REF_FILES' in os.environ:
    from .util import _regenerate_references
    _regenerate_references(force=True)

_NO_REGEN_TEST_DATA = 'NO_REGEN_TEST_DATA' in os.environ

# Suppress pygsti backwards-compatibility warning
os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
