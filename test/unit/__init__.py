"""Unit test coverage for pyGSTi"""
import os
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Suppress pygsti backwards-compatibility warning
os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
