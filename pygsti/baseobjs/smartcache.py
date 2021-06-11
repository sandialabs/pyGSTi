"""
Defines SmartCache and supporting functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import hashlib as _hashlib
import inspect as _inspect
import pickle as _pickle
from collections import Counter, defaultdict

import numpy as _np

DIGEST_TIMES = defaultdict(list)


def _csize(counter):
    """
    Computes the size of (number of elements in) a given Counter.

    Parameters
    ----------
    counter : Counter
        The counter to get the size of.

    Returns
    -------
    int
    """
    return len(list(counter.elements()))


def _average(l):
    """
    Computes the _average of the items in a list

    Parameters
    ----------
    l : list
        the list

    Returns
    -------
    int
    """
    nCalls = max(1, len(l))
    return sum(l) / nCalls


def _show_cache_percents(hits, misses, printer):
    """
    Shows effectiveness of a cache

    Parameters
    ----------
    hits : Counter
        cache hits

    misses : Counter
        cache misses

    printer : pygsti.objects.VerbosityPrinter
        logging object

    Returns
    -------
    None
    """
    nHits = _csize(hits)
    nMisses = _csize(misses)
    nRequests = nHits + nMisses
    printer.log('    {:<10} requests'.format(nRequests))
    printer.log('    {:<10} hits'.format(nHits))
    printer.log('    {:<10} misses'.format(nMisses))
    printer.log('    {}% effective\n'.format(round((nHits / max(1, nRequests)) * 100, 2)))


def _show_kvs(title, kvs, printer):
    """
    Pretty-print key-value pairs w/ a title and printer object

    Parameters
    ----------
    title : str
        Title to print

    kvs : iterable
        An object that yields (key, value) pairs when you iterate it.

    printer : VerbosityPrinter
        The printer object.

    Returns
    -------
    None
    """
    printer.log(title)
    for k, v in kvs:
        printer.log('    {:<40} {}'.format(k, v))
    printer.log('')


class SmartCache(object):
    """
    Cache object that profiles itself

    Parameters
    ----------
    decorating : tuple
        module and function being decorated by the smart cache

    Attributes
    ----------
    StaticCacheList : list
        A list of all :class:`SmartCache` instances.
    """
    StaticCacheList = []

    def __init__(self, decorating=(None, None)):
        '''
        Construct a smart cache object

        Parameters
        ----------
        decorating : tuple
            module and function being decorated by the smart cache
        '''
        self.cache = dict()
        self.outargs = dict()
        self.ineffective = set()
        self.decoratingModule, self.decoratingFn = decorating
        self.customDigests = []

        self.misses = Counter()
        self.hits = Counter()
        self.fhits = Counter()

        self.requests = Counter()
        self.ineffectiveRequests = Counter()

        self.effectiveTimes = defaultdict(list)
        self.ineffectiveTimes = defaultdict(list)

        self.hashTimes = defaultdict(list)
        self.callTimes = defaultdict(list)

        self.typesigs = dict()
        self.saved = 0

        self.unpickleable = set()

        SmartCache.StaticCacheList.append(self)

    def __setstate__(self, d):
        return self.__dict__.update(d)

    def __getstate__(self):
        d = dict(self.__dict__)

        def get_pickleable_dict(cache_dict):
            pickleableCache = dict()
            for k, v in cache_dict.items():
                try:
                    _pickle.dumps(v)
                    pickleableCache[k] = v
                except TypeError as e:
                    if isinstance(v, dict):
                        self.unpickleable.add(str(k[0]) + str(type(v)) + str(e) + str(list(v.keys())))
                    else:
                        self.unpickleable.add(str(k[0]) + str(type(v)) + str(e))  # + str(list(v.__dict__.keys())))
                except _pickle.PicklingError as e:
                    self.unpickleable.add(str(k[0]) + str(type(v)) + str(e))
            return pickleableCache

        d['cache'] = get_pickleable_dict(self.cache)
        d['outargs'] = get_pickleable_dict(self.outargs)
        return d

    def __pygsti_getstate__(self):  # same but for json/msgpack
        d = dict(self.__dict__)
        from ..io.jsoncodec import encode_obj

        def get_jsonable_dict(cache_dict):

            jsonableCache = dict()
            for k, v in cache_dict.items():
                try:
                    encode_obj(v, False)
                    jsonableCache[k] = v
                except TypeError as e:
                    self.unpickleable.add(str(k[0]) + str(type(v)) + str(e))
            return jsonableCache

        d['cache'] = get_jsonable_dict(self.cache)
        d['outargs'] = get_jsonable_dict(self.outargs)
        return d

    def add_digest(self, custom):
        """
        Add a "custom" digest function, used for hashing otherwise un-hashable types.

        Parameters
        ----------
        custom : function
            A hashing function, which takes two arguments: `md5` (a running MD5
            hash) and `val` (the value to be hashed).  It should call
            `md5.update` to add to the running hash, and needn't return anything.

        Returns
        -------
        None
        """
        self.customDigests.append(custom)

    def low_overhead_cached_compute(self, fn, arg_vals, kwargs=None):
        """
        Cached compute with less profiling. See :method:`cached_compute` docstring.

        Parameters
        ----------
        fn : function
            Cached function

        arg_vals : tuple or list
            Arguments to cached function

        kwargs : dictionary
            Keyword arguments to cached function

        Returns
        -------
        key : the key used to hash the function call
        result : result of fn called with arg_vals and kwargs
        """
        if kwargs is None:
            kwargs = dict()
        name_key = _get_fn_name_key(fn)
        if name_key in self.ineffective:
            key = 'NA'
            result = fn(*arg_vals, **kwargs)
        else:
            from pygsti.tools.opttools import timed_block as _timed_block
            times = dict()
            with _timed_block('hash', times):
                key = _call_key(fn, tuple(arg_vals) + (kwargs,), self.customDigests)  # cache by call key
            if key not in self.cache:
                with _timed_block('call', times):
                    self.cache[key] = fn(*arg_vals, **kwargs)
                if times['hash'] > times['call']:
                    self.ineffective.add(name_key)
            result = self.cache[key]
        return key, result

    def cached_compute(self, fn, arg_vals, kwargs=None):
        """
        Shows effectiveness of a cache

        Parameters
        ----------
        fn : function
            Cached function

        arg_vals : tuple or list
            Arguments to cached function

        kwargs : dictionary
            Keyword arguments to cached function

        Returns
        -------
        key : the key used to hash the function call
        result : result of fn called with arg_vals and kwargs
        """
        special_kwargs = dict()
        if kwargs is None:
            kwargs = dict()
        else:
            for k, v in kwargs.items():
                if k.startswith('_'):
                    special_kwargs[k] = v
            for k in special_kwargs: del kwargs[k]

        name_key = _get_fn_name_key(fn)
        self.requests[name_key] += 1
        if name_key in self.ineffective:
            key = 'INEFFECTIVE'
            result = fn(*arg_vals, **kwargs)
            self.ineffectiveRequests[name_key] += 1
            self.misses[key] += 1
            #DB: print(fn.__name__, " --> Ineffective!") # DB
        else:
            from pygsti.tools.opttools import timed_block as _timed_block
            times = dict()
            with _timed_block('hash', times):
                key = _call_key(fn, tuple(arg_vals) + (kwargs,), self.customDigests)  # cache by call key
            if key not in self.cache:
                #DB: if "_compute_sub_mxs" in fn.__name__:
                #DB: print(fn.__name__, " --> computing... (not found in %d keys)" % len(list(self.cache.keys()))) # DB
                #DB: print("Key detail: ",key[0]) # DB
                #DB: for a,k in zip(tuple(arg_vals)+(kwargs,),key[1:]): print(type(a),": ",repr(k)) # DB
                typesig = str(tuple(str(type(arg)) for arg in arg_vals)) + \
                    str({k: str(type(v)) for k, v in kwargs.items()})
                self.typesigs[name_key] = typesig
                with _timed_block('call', times):
                    self.cache[key] = fn(*arg_vals, **kwargs)
                    if "_filledarrays" in special_kwargs:
                        self.outargs[key] = tuple((arg_vals[i] if isinstance(i, int) else kwargs[i]
                                                   for i in special_kwargs['_filledarrays']))  # copy?
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
                #DB: print('The function {} experienced a cache hit'.format(name_key)) # DB
                #DB: print(fn.__name__, " --> cache hit!") # DB
                self.hits[key] += 1
                self.fhits[name_key] += 1

                #Special kwarg processing: any keyword argument that starts with an
                # underscore is considered to be directed the SmartCache.
                if "_filledarrays" in special_kwargs:
                    for i, pos in enumerate(special_kwargs["_filledarrays"]):
                        if isinstance(pos, int):
                            arg_vals[pos][:] = self.outargs[key][i]
                        else:
                            kwargs[pos][:] = self.outargs[key][i]

            #Note - maybe we should .view or .copy arrays upon return
            # (now we just trust user not to alter mutable returned vals)
            result = self.cache[key]
        return key, result

    @staticmethod
    def global_status(printer):
        """
        Show the statuses of all Cache objects

        Parameters
        ----------
        printer : VerbosityPrinter
            The printer to use for output.

        Returns
        -------
        None
        """
        totalSaved = 0
        totalHits = Counter()
        totalMisses = Counter()

        suggestRemove = set()
        warnNoHits = set()
        notCalled = set()

        for cache in SmartCache.StaticCacheList:
            cache.status(printer)
            totalHits += cache.hits
            totalMisses += cache.misses
            totalSaved += cache.saved

            fullname = '{}.{}'.format(cache.decoratingModule, cache.decoratingFn)

            if cache.decoratingFn is not None and \
                    cache.decoratingFn in cache.ineffective:
                suggestRemove.add(fullname)

            if _csize(cache.hits) == 0:
                warnNoHits.add(fullname)

            if _csize(cache.requests) == 0:
                notCalled.add(fullname)

        with printer.verbosity_env(2):
            printer.log('Average hash times by object:')
            for k, v in sorted(DIGEST_TIMES.items(), key=lambda t: sum(t[1])):
                total = sum(v)
                avg = _average(v)
                printer.log('    {:<65} avg | {}s'.format(k, avg))
                printer.log('    {:<65} tot | {}s'.format('', total))
                printer.log('-' * 100)

        printer.log('\nBecause they take longer to hash than to calculate, \n'
                    'the following functions may be unmarked for caching:')
        for name in suggestRemove:
            printer.log('    {}'.format(name))

        with printer.verbosity_env(2):
            printer.log('\nThe following functions would provide a speedup, \n'
                        'but currently do not experience any cache hits:')
            for name in warnNoHits:
                printer.log('    {}'.format(name))

        with printer.verbosity_env(4):
            printer.log('\nThe following functions are not called:')
            for name in notCalled:
                printer.log('    {}'.format(name))

        printer.log('\nGlobal cache overview:')
        _show_cache_percents(totalHits, totalMisses, printer)
        printer.log('    {} seconds saved total'.format(totalSaved))

    def avg_timedict(self, d):
        """
        Given a dictionary of lists of times (`d`), returns a dict of the summed times.

        Parameters
        ----------
        d : dict
            A dictionary whose values are lists of times.

        Returns
        -------
        dict
        """
        ret = dict()
        for k, v in d.items():
            if k not in self.fhits:
                time = 0
            else:
                time = _average(v) * len(v)
            ret[k] = time
        return ret

    def status(self, printer):
        """
        Show the status of a cache object instance

        Parameters
        ----------
        printer : VerbosityPrinter
            The printer to use for output.

        Returns
        -------
        None
        """
        printer.log('Status of smart cache decorating {}.{}:\n'.format(
            self.decoratingModule, self.decoratingFn))
        _show_cache_percents(self.hits, self.misses, printer)

        with printer.verbosity_env(2):
            _show_kvs('Most common requests:\n', self.requests.most_common(), printer)
            _show_kvs('Ineffective requests:\n', self.ineffectiveRequests.most_common(), printer)
            _show_kvs('Hits:\n', self.fhits.most_common(), printer)

            printer.log('Type signatures of functions and their hash times:\n')
            for k, v in self.typesigs.items():
                avg = _average(self.hashTimes[k])
                printer.log('    {:<40} {}'.format(k, v))
                printer.log('    {:<40} {}'.format(k, avg))
                printer.log('')
            printer.log('')

            savedTimes = self.avg_timedict(self.effectiveTimes)
            saved = sum(savedTimes.values())
            _show_kvs('Effective total saved time:\n',
                      sorted(savedTimes.items(), key=lambda t: t[1], reverse=True),
                      printer)

            overTimes = self.avg_timedict(self.ineffectiveTimes)
            overhead = sum(overTimes.values())
            _show_kvs('Ineffective differences:\n',
                      sorted(overTimes.items(), key=lambda t: t[1]),
                      printer)

        printer.log('overhead    : {}'.format(overhead))
        printer.log('saved       : {}'.format(saved))
        printer.log('net benefit : {}'.format(saved - overhead))
        self.saved = saved - overhead
        printer.log(self.unpickleable)


def smart_cached(obj):
    """
    Decorator for applying a smart cache to a single function or method.

    Parameters
    ----------
    obj : function
        function to decorate.

    Returns
    -------
    function
    """
    cache = obj.cache = SmartCache(decorating=(obj.__module__, obj.__name__))

    @_functools.wraps(obj)
    def _cacher(*args, **kwargs):
        _, v = cache.cached_compute(obj, args, kwargs)
        return v
    return _cacher


class CustomDigestError(Exception):
    """
    Custom Digest Exception type
    """
    pass


def digest(obj, custom_digests=None):
    """
    Returns an MD5 digest of an arbitary Python object, `obj`.

    Parameters
    ----------
    obj : object
        Object to digest.

    custom_digests : list, optional
        A list of custom digest functions.  Each function should have the signature
        `digest(md5 : hashlib.md5, value)` and either digest `value` (calling `md5.update`
        or similar) or raise a :class:`CustomDigestError` to indicate it was unable to
        digest `value`.

    Returns
    -------
    MD5_digest
    """
    if custom_digests is None:
        custom_digests = []

    # a function to recursively serialize 'v' into an md5 object
    def add(md5, v):
        """Add `v` to the hash, recursively if needed."""
        from pygsti.tools.opttools import timed_block as _timed_block
        with _timed_block(str(type(v)), DIGEST_TIMES):
            md5.update(str(type(v)).encode('utf-8'))
            if isinstance(v, SmartCache): return  # don't hash SmartCache args
            if isinstance(v, bytes):
                md5.update(v)  # can add bytes directly
            elif v is None:
                md5.update(str(hash("(_NONE_)")).encode('utf-8'))  # make all None's hash the same
            else:
                try:
                    md5.update(str(hash(v)).encode('utf-8'))
                except TypeError:  # as hashException:
                    if isinstance(v, _np.ndarray):
                        md5.update(v.tostring() + str(v.shape).encode('utf-8'))  # numpy gives us bytes
                    elif isinstance(v, (tuple, list)):
                        for el in v: add(md5, el)
                    elif isinstance(v, dict):
                        keys = list(v.keys())
                        for k in sorted(keys):
                            add(md5, k)
                            add(md5, v[k])
                    elif type(v).__module__ == 'mpi4py.MPI':  # don't import mpi4py (not always available)
                        pass  # don't hash comm objects
                    else:
                        for custom_digest in custom_digests:
                            try:
                                custom_digest(md5, v)
                                break
                            except CustomDigestError:
                                pass
                        else:
                            attribs = sorted(v.__dict__.keys()) if hasattr(v, '__dict__') else list(sorted(dir(v)))
                            for k in attribs:
                                if k.startswith('__'):
                                    continue
                                a = getattr(v, k)
                                if _inspect.isroutine(a):
                                    continue
                                add(md5, k)
                                add(md5, a)
            return

    M = _hashlib.md5()
    add(M, obj)
    return M.digest()  # return native hash of the MD5 digest


def _get_fn_name_key(fn):
    """
    Get the name (str) used to hash the function `fn`

    Parameters
    ----------
    fn : function
        The function to get the hash key for.

    Returns
    -------
    str
    """
    name = fn.__name__
    if hasattr(fn, '__self__'):
        name = fn.__self__.__class__.__name__ + '.' + name
    return name


def _call_key(fn, args, custom_digests):
    """
    Returns a hashable key for caching the result of a function call.

    Parameters
    ----------
    fn : function
        The function itself

    args : list or tuple
        The function's arguments.

    custom_digests : list, optional
        A list of custom digest functions.  Each function should have the signature
        `digest(md5 : hashlib.md5, value)` and either digest `value` (calling `md5.update`
        or similar) or raise a :class:`CustomDigestError` to indicate it was unable to
        digest `value`.

    Returns
    -------
    tuple
    """
    fnName = _get_fn_name_key(fn)
    if fn.__name__ == "_create":
        pass  # special case: don't hash "self" in _create functions (b/c self doesn't matter - "self" is being created)
    elif hasattr(fn, '__self__'):  # add "self" to args when it's an instance's method call
        args = (fn.__self__,) + args
    inner_digest = _functools.partial(digest, custom_digests=custom_digests)
    return (fnName,) + tuple(map(inner_digest, args))
