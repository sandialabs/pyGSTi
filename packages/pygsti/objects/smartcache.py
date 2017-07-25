import hashlib   as _hashlib
import functools as _functools
import sys       as _sys
import numpy     as _np
import functools as _functools
import inspect   as _inspect

from collections import Counter, defaultdict
from pprint      import pprint

from .dataset          import DataSet
from .datacomparator   import DataComparator
from .verbosityprinter import VerbosityPrinter

from ..tools import compattools as _compat
from ..tools import timed_block as _timed_block

class SmartCache:
    def __init__(self):
        self.cache       = dict()
        self.ineffective = set()

        self.misses = Counter()
        self.hits   = Counter()

        self.requests            = Counter()
        self.ineffectiveRequests = Counter()

        self.effectiveTimes   = defaultdict(list)
        self.ineffectiveTimes = defaultdict(list)

        self.hashTimes = defaultdict(list)
        self.callTimes = defaultdict(list)

    def cached_compute(self, fn, argVals, custom_digest=None):
        name_key = get_fn_name_key(fn)
        self.requests[name_key] += 1 
        if name_key in self.ineffective:
            key = 'NA'
            result = fn(*argVals)
            self.ineffectiveRequests[name_key] += 1
            self.misses[key] += 1
        else:
            times = dict()
            with _timed_block('hash', times):
                key = call_key(fn, argVals, custom_digest) # cache by call key
            if key not in self.cache:
                with _timed_block('call', times):
                    self.cache[key] = fn(*argVals)
                self.misses[key] += 1
            else:
                self.hits[key] += 1
            if 'call' in times:
                hashtime = times['hash']
                calltime = times['call']
                if hashtime > calltime:
                    self.ineffective.add(name_key)
                    self.ineffectiveTimes[name_key].append(hashtime - calltime)
                else:
                    self.effectiveTimes[name_key].append(calltime - hashtime)
                self.hashTimes[name_key].append(hashtime)
                self.callTimes[name_key].append(calltime)
            result = self.cache[key]
        return key, result

    def status(self, printer=VerbosityPrinter(1)):
        size = lambda counter : len(list(counter.elements()))
        nRequests = size(self.requests)
        nHits     = size(self.hits)
        nMisses   = size(self.misses)
        printer.log('Status of smart cache:\n')
        printer.log('    {:<10} requests'.format(nRequests))
        printer.log('    {:<10} hits'.format(nHits))
        printer.log('    {:<10} misses'.format(nMisses))
        printer.log('    {}% effective\n'.format(int((nHits/nRequests) * 100)))

        printer.log('Most common requests:\n')
        for k, v in self.requests.most_common():
            printer.log('    {:<40} {}'.format(k, v))
        printer.log('')
        printer.log('Ineffective requests:\n')
        for k, v in self.ineffectiveRequests.most_common():
            printer.log('    {:<40} {}'.format(k, v))

        printer.log('')
        def saved_time(kv):
            k, v = kv
            nCalls = max(1, len(v))
            avg = sum(v) / nCalls
            return avg * nCalls

        printer.log('Effective total saved time, on average (potentially theoretical, if no cache hits):\n')
        saved = 0
        for k, v in sorted(self.effectiveTimes.items(), 
                           key=saved_time, reverse=True):
            v = saved_time((k, v))
            printer.log('    {:<45} {}'.format(k, v))
            saved += v
        printer.log('')

        printer.log('Ineffective differences:\n')
        overhead = 0
        for k, v in sorted(self.ineffectiveTimes.items(), 
                           key=saved_time):
            v = saved_time((k, v))
            printer.log('    {:<45} {}'.format(k, v))
            overhead += v
        printer.log('')

        printer.log('Hash v call differences:\n')
        for name_key in self.requests:
            keyHashTimes = self.hashTimes[name_key]
            hAvg = sum(keyHashTimes) / max(1, len(keyHashTimes))
            printer.log('    {:<45} hash avg: {}'.format(name_key, hAvg))
            keyCallTimes = self.callTimes[name_key]
            cAvg = sum(keyCallTimes) / max(1, len(keyCallTimes))
            printer.log('    {:<45} call avg: {}'.format(name_key, cAvg))
            printer.log('    {:<45} diff    : {}'.format(name_key, cAvg - hAvg))
            printer.log('')
        printer.log('')
        printer.log('overhead : {}'.format(overhead))
        printer.log('saved    : {}'.format(saved))
        printer.log('')
        printer.log('net benefit : {}'.format(saved - overhead))

def smart_cached(obj):
    cache = obj.cache = SmartCache()
    @_functools.wraps(obj)
    def cacher(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError('Cannot currently cache on kwargs')
        k, v = cache.cached_compute(obj, args)
        return v
    return cacher

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
