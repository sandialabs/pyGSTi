import hashlib as _hashlib
import functools as _functools
import sys as _sys
import numpy as _np
import functools as _functools
import inspect as _inspect

from .dataset import DataSet
from .datacomparator import DataComparator
from ..tools import compattools as _compat
from ..tools import timed_block as _timed_block
'''
elif isinstance(v,NotApplicable):
    md5.update("NOTAPPLICABLE".encode('utf-8'))
elif isinstance(v, SwitchValue):
    md5.update(v.base.tostring()) #don't recurse to parent switchboard
'''

class CustomDigestError(Exception):
    pass

def digest(obj, custom_digest=None):
    """Returns an MD5 digest of an arbitary Python object, `obj`."""
    if _sys.version_info > (3, 0): # Python3?
        longT = int      # define long and unicode
        unicodeT = str   #  types to mimic Python2
    else:
        longT = long
        unicodeT = unicode

    # a function to recursively serialize 'v' into an md5 object
    def add(md5, v):
        """Add `v` to the hash, recursively if needed."""
        md5.update(str(type(v)).encode('utf-8'))
        if isinstance(v, bytes):
            md5.update(v)  #can add bytes directly
        elif isinstance(v, float) or _compat.isstr(v) or _compat.isint(v):
            md5.update(str(v).encode('utf-8')) #need to encode strings
        elif isinstance(v, _np.ndarray):
            md5.update(v.tostring()) # numpy gives us bytes
        elif isinstance(v, (tuple, list)):
            for el in v:  add(md5,el)
        elif isinstance(v, dict):
            keys = list(v.keys())
            for k in sorted(keys):
                add(md5,k)
                add(md5,v[k])
        elif v is None:
            md5.update("NONE".encode('utf-8'))
        elif isinstance(v, (DataSet, DataComparator)):
            md5.update(v.timestamp.encode('utf-8'))
        else:
            try:
                if custom_digest is None:
                    raise CustomDigestError()
                custom_digest(md5, v)
            except CustomDigestError:
                attribs = list(sorted(dir(v)))
                for k in attribs:
                    if k.startswith('__'): continue
                    a = getattr(v, k)
                    if _inspect.isroutine(a): continue
                    add(md5,k)
                    add(md5,a)
        return

    M = _hashlib.md5()
    add(M, obj)
    return M.digest() #return the MD5 digest

def get_fn_name_key(fn):
    name = fn.__name__
    if hasattr(fn, '__self__'):
        name = fn.__self__.__class__.__name__ + '.' + name
    return name

def call_key(fn, args, custom_digest):
    """ 
    Returns a hashable key for caching the result of a function call.

    Parameters
    ----------
    fn : function
       The function itself

    args : list or tuple
       The function's arguments.

    Returns
    -------
    tuple
    """
    fnName = get_fn_name_key(fn)
    inner_digest = _functools.partial(digest, custom_digest=custom_digest)
    return (fnName,) + tuple(map(inner_digest,args))

class SmartCache:
    def __init__(self):
        self.cache       = dict()
        self.ineffective = set()

    def cached_compute(self, fn, argVals, custom_digest=None):
        name_key = get_fn_name_key(fn)
        if name_key in self.ineffective:
            key = 'NA'
            result = fn(*argVals)
        else:
            # argVals now contains all the arguments, so call the function if
            #  we need to and add result.
            times = dict()
            with _timed_block('hash', times):
                key = call_key(fn, argVals, custom_digest) # cache by call key
            if key not in self.cache:
                with _timed_block('call', times):
                    self.cache[key] = fn(*argVals)
            if 'call' in times:
                if times['hash'] > times['call']:
                    print('Added {} to hash-ineffective functions'.format(name_key))
                    self.ineffective.add(name_key)
            result = self.cache[key]
        return key, result

