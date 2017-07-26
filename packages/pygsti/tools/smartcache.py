import hashlib   as _hashlib
import functools as _functools
import sys       as _sys
import numpy     as _np
import functools as _functools
import inspect   as _inspect

from collections import Counter, defaultdict
from pprint      import pprint

from . import compattools as _compat

from .opttools import timed_block as _timed_block

DIGEST_TIMES = defaultdict(list)

def average(l):
    nCalls = max(1, len(l))
    avg = sum(l) / nCalls
    return avg

class SmartCache(object):
    StaticCacheList = []

    def __init__(self, decorating=None):
        self.cache       = dict()
        self.ineffective = set()
        self.decorating    = decorating
        self.customDigests = []

        self.misses = Counter()
        self.hits   = Counter()
        self.fhits  = Counter()

        self.requests            = Counter()
        self.ineffectiveRequests = Counter()

        self.effectiveTimes   = defaultdict(list)
        self.ineffectiveTimes = defaultdict(list)

        self.hashTimes = defaultdict(list)
        self.callTimes = defaultdict(list)

        self.typesigs = dict()
        
        SmartCache.StaticCacheList.append(self)

    def add_digest(self, custom):
        self.customDigests.append(custom)

    def low_overhead_cached_compute(self, fn, argVals, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        name_key = get_fn_name_key(fn)
        if name_key in self.ineffective:
            key = 'NA'
            result = fn(*argVals, **kwargs)
        else:
            times = dict()
            with _timed_block('hash', times):
                key = call_key(fn, (argVals, kwargs), self.customDigests) # cache by call key
            if key not in self.cache:
                with _timed_block('call', times):
                    self.cache[key] = fn(*argVals, **kwargs)
                if times['hash'] > times['call']:
                    self.ineffective.add(name_key)
            result = self.cache[key]
        return key, result

    def cached_compute(self, fn, argVals, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        name_key = get_fn_name_key(fn)
        self.requests[name_key] += 1 
        if name_key in self.ineffective:
            key = 'NA'
            result = fn(*argVals, **kwargs)
            self.ineffectiveRequests[name_key] += 1
            self.misses[key] += 1
        else:
            times = dict()
            with _timed_block('hash', times):
                key = call_key(fn, (argVals, kwargs), self.customDigests) # cache by call key
            if key not in self.cache:
                typesig = str(tuple(str(type(arg)) for arg in argVals)) + \
                        str({k : str(type(v)) for k, v in kwargs.items()})
                self.typesigs[name_key] = typesig
                with _timed_block('call', times):
                    self.cache[key] = fn(*argVals, **kwargs)
                self.misses[key] += 1
                hashtime = times['hash']
                calltime = times['call']
                if hashtime > calltime:
                    self.ineffective.add(name_key)
                    self.ineffectiveTimes[name_key].append(hashtime - calltime)
                else:
                    self.effectiveTimes[name_key].append(calltime - hashtime)
                self.hashTimes[name_key].append(hashtime)
                self.callTimes[name_key].append(calltime)
            else:
                #print('The function {} experienced a cache hit'.format(name_key))
                self.hits[key] += 1
                self.fhits[name_key] += 1
            result = self.cache[key]
        return key, result

    @staticmethod
    def global_status(printer):
        for cache in SmartCache.StaticCacheList:
            cache.status(printer)
        printer.log('Average hash times by object:')
        for k, v in sorted(DIGEST_TIMES.items(), key=lambda t : sum(t[1])):
            total = sum(v)
            avg   = average(v)
            printer.log('    {:<65} avg | {}s'.format(k, avg))
            printer.log('    {:<65} tot | {}s'.format('', total))
            printer.log('-'*100)

    def status(self, printer):
        size = lambda counter : len(list(counter.elements()))
        nRequests = size(self.requests)
        nHits     = size(self.hits)
        nMisses   = size(self.misses)
        printer.log('Status of smart cache decorating {}:\n'.format(self.decorating))
        printer.log('    {:<10} requests'.format(nRequests))
        printer.log('    {:<10} hits'.format(nHits))
        printer.log('    {:<10} misses'.format(nMisses))
        printer.log('    {}% effective\n'.format(int((nHits/max(1, nRequests)) * 100)))

        printer.log('Most common requests:\n')
        for k, v in self.requests.most_common():
            printer.log('    {:<40} {}'.format(k, v))
        printer.log('')
        printer.log('Ineffective requests:\n')
        for k, v in self.ineffectiveRequests.most_common():
            printer.log('    {:<40} {}'.format(k, v))

        printer.log('')
        printer.log('Hits by name:\n')
        for k, v in self.fhits.most_common():
            printer.log('    {:<40} {}'.format(k, v))

        printer.log('')
        printer.log('Type signatures of functions and their hash times:\n')
        for k, v in self.typesigs.items():
            avg = average(self.hashTimes[k])
            printer.log('    {:<40} {}'.format(k, v))
            printer.log('    {:<40} {}'.format(k, avg))
            printer.log('')
        printer.log('')
        def saved_time(kv):
            k, v = kv
            if k not in self.fhits:
                return 0
            return average(v) * len(v)

        printer.log('Effective total saved time:\n')
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
            v = sum(v) / max(1, len(v))
            printer.log('    {:<45} {}'.format(k, v))
            overhead += v
        printer.log('')
        printer.log('overhead : {}'.format(overhead))
        printer.log('saved    : {}'.format(saved))
        printer.log('')
        printer.log('net benefit : {}'.format(saved - overhead))

def smart_cached(obj):
    cache = obj.cache = SmartCache(decorating=obj.__name__)
    @_functools.wraps(obj)
    def cacher(*args, **kwargs):
        k, v = cache.cached_compute(obj, args, kwargs)
        return v
    return cacher

class CustomDigestError(Exception):
    pass

def digest(obj, custom_digests=None):
    """Returns an MD5 digest of an arbitary Python object, `obj`."""
    if custom_digests is None:
        custom_digests = []
    if _sys.version_info > (3, 0): # Python3?
        longT = int      # define long and unicode
        unicodeT = str   #  types to mimic Python2
    else:
        longT = long
        unicodeT = unicode

    # a function to recursively serialize 'v' into an md5 object
    def add(md5, v):
        """Add `v` to the hash, recursively if needed."""
        with _timed_block(str(type(v)), DIGEST_TIMES):
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
            elif hasattr(v, 'timestamp'):
                md5.update(v.timestamp.encode('utf-8'))
            else:
                for custom_digest in custom_digests:
                    try:
                        custom_digest(md5, v)
                        break
                    except CustomDigestError:
                        pass
                else:
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

def call_key(fn, args, custom_digests):
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
    inner_digest = _functools.partial(digest, custom_digests=custom_digests)
    return (fnName,) + tuple(map(inner_digest,args))
