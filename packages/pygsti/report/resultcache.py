from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
import collections as _collections
import re          as _re
import time        as _time
import itertools   as _itertools

from ..objects import VerbosityPrinter
#import warnings as _warnings

class ResultCache(object):
    def __init__(self, computeFns, parent=None, typenm="object"):
        self._computeFns = computeFns
        self._parent = parent # parent Results object
        self._typename = typenm # "type name" just for verbose output
        self._data = _collections.OrderedDict()
           #key = confidence level, value = ordered dict of actual data
           # (so dict of dicts where outer dict key is confidence level)

    def _setparent(self, computeFns, parent):
        """
        Set compute functions and parent.  These items
        are *not* pickled to avoid circular pickle references, so this
        method should be called by a parent object's _setstate_
        function.
        """
        self._computeFns = computeFns
        self._parent = parent

    def get(self, key, confidence_level="current",verbosity=0):

        """
        Retrieve the data associated with a given key, optionally
        specifying the confidence level and verbosity.

        Note that access via square-bracket indexing always uses
        the "current" confidence level, which is taken from the
        parent Results object, and verbosity == 0.

        Parameters
        ----------
        key : string
           The key to retrieve

        confidence_level : float or None or "current"
           The confidence level for which the value should be retrieved.  The
           special (and default) value "current" uses the value of the parent
           Results object.

        verbosity : int
           The level of detail to print to stdout.

        Returns
        -------
        stored_data_item
        """

        if self._parent:
            printer = VerbosityPrinter.build_printer(verbosity, comm=self._parent._comm)
        else:
            printer = VerbosityPrinter.build_printer(verbosity)

        if confidence_level=="current":
            level = self._parent.confidence_level if self._parent else None
        else: level = confidence_level

        if (level in self._data) == False:
            if self._get_compute_fn(key):
                self._data[level] = _collections.OrderedDict()
            else:
                raise KeyError("Invalid key: %s" % key)

        CIsuffix=" (w/%d%% CIs)" % round(level) if (level is not None) else ""

        if (key in self._data[level]) == False:
            computeFn = self._get_compute_fn(key)
            if computeFn:
                try:
                    tStart = _time.time()
                    printer.log("Generating %s: %s%s" % 
                                (self._typename, key, CIsuffix), end='')

                    self._data[level][key] = computeFn(key, level, printer)

                    printer.log("[%.1fs]" % (_time.time()-tStart))
                except ResultCache.NoCRDependenceError:
                    assert(level is not None)
                    self._data[level][key] = self.get(key, None, printer)
            else:
                raise KeyError("Invalid key: %s" % key)
            return self._data[level][key]

        else:
            printer.log("Retrieving cached %s: %s%s" %             
                        (self._typename, key, CIsuffix))
            return self._data[level][key]

    def _get_compute_fn(self, key):
        for expr,(computeFn, validateFn) in self._computeFns.items():
            if _re.match(expr, key) and len(validateFn(key)) > 0:
                return computeFn
        return None

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val):
        raise ValueError("ResultCache objects are read-only")

    def _compute_keys(self):
        #Passing the computeFn key "expr" to it's corresponding
        # validation function returns a list of all the computable
        # keys matching that expression, which by definition are all
        # the keys possibly computed by the given compute function.
        return list(_itertools.chain(
            *[ validateFn(expr) for expr,(computeFn, validateFn)
              in self._computeFns.items() ] ))

    def __iter__(self):
        #return self.dataset.slIndex.__iter__() #iterator over spam labels
        raise NotImplementedError("Would require mass evaluation of all keys.")

    def __contains__(self, key):
        return key in self._compute_keys()

    def keys(self):
        """ Returns a list of existing (or computable) keys."""
        return self._compute_keys()

    def has_key(self, key):
        """ Checks whether a given key exists (or is computable)."""
        return key in self._compute_keys()

    def iteritems(self):
        """ Iterates over (key, value) pairs. """
        #would use current confidence level
        raise NotImplementedError("Would require mass evaluation of all keys.")

    def values(self):
        """ Returns a list of values (for the current confidence level)."""
        #would use current confidence level
        raise NotImplementedError("Would require mass evaluation of all keys.")

    def copy(self):
        """ Creates a copy of this ResultCache.  Parent is *reset* """
        newCache = ResultCache(self._computeFns, parent=None,
                               typenm=self._typename)
        newCache._data = self._data.copy() #deepcopy()
        return newCache

    def __getstate__(self):
        #Return the state (for pickling) -- *don't* pickle parent or fns
        return  { '_data': self._data, '_typename': self._typename }

    def __setstate__(self, stateDict):
        self._computeFns = {}
        self._parent = None
        self._data = stateDict['_data']
        self._typename = stateDict['_typename']

    def clear_cached_data(self, except_these_keys):
        """
        Clears the cached data for all keys except those
        specified.

        Parameters
        ----------
        except_these_keys : list
            A list of keys to *keep* cached.

        Returns
        -------
        None
        """
        for level in self._data:
            for key in list(self._data[level].keys()):
                if key in except_these_keys: continue
                del self._data[level][key]


    class NoCRDependenceError(Exception):
        """
        Exception indicating that a function has no confidence region
        dependence and therefore doesn't need to be re-computed when the
        confidence level changes
        """
        pass
