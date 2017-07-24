import warnings  as _warnings
import functools as _functools

from .parameterized import *

def warn_deprecated(name, replacement=None):
    message = 'The function {} is deprecated, and may not be present in future versions of pygsti.'.format(name)
    if replacement is not None:
        message += '\n    '
        message += 'Please use {} instead.'.format(replacement)
    _warnings.warn(message)

@parameterized
def deprecated_fn(fn, replacement=None):
    def inner(*args, **kwargs):
        warn_deprecated(fn.__name__, replacement)
        return fn(*args, **kwargs)
    return inner
