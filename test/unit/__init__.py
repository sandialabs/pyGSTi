"""Unit test coverage for pyGSTi"""
import logging
import os

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Suppress pygsti backwards-compatibility warning
os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
