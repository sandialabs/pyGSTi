#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
import collections as _collections
import numpy as _np
import re as _re
import sys as _sys
import itertools as _itertools
#import warnings as _warnings

class ResultCache(object):
    def __init__(self, computeFns, parent=None, typenm="object"):
        self._computeFns = computeFns
        self._parent = parent # parent Results object
        self._typename = typenm # "type name" just for verbose output
        self._data = _collections.OrderedDict()
           #key = confidence level, value = ordered dict of actual data
           # (so dict of dicts where outer dict key is confidence level)

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
        if confidence_level=="current":
            level = self._parent.confidence_level if self._parent else None
        else: level = confidence_level

        if self._data.has_key(level) == False:
            if self._get_compute_fn(key):
                self._data[level] = _collections.OrderedDict()
            else:
                raise KeyError("Invalid key: %s" % key)
        
        if self._data[level].has_key(key) == False:
            computeFn = self._get_compute_fn(key)
            if computeFn:
                try:
                    if verbosity > 0:
                        print "Generating %s: %s%s" % \
                            (self._typename,key," (w/%d%% CIs)" % round(level) 
                             if (level is not None) else "")
                        _sys.stdout.flush()

                    self._data[level][key] = computeFn(key, level, verbosity)
                except ResultCache.NoCRDependenceError:
                    assert(level is not None)
                    self._data[level][key] = self.get(key,None,verbosity)
            else:
                raise KeyError("Invalid key: %s" % key)
            return self._data[level][key]
    
        else:
            if verbosity > 0:
                print "Retrieving cached %s: %s%s" % \
                    (self._typename,key," (w/%d%% CIs)" % round(level) 
                     if (level is not None) else "")
                _sys.stdout.flush()
            return self._data[level][key]

    def _get_compute_fn(self, key):
        for expr,(computeFn, validateFn) in self._computeFns.iteritems():
            if _re.match(expr, key) and validateFn(key) is not None:
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
        return _itertools.chain( 
            [ validateFn(expr) for expr,(computeFn, validateFn) 
              in self._computeFns.iteritems() ] )

    def __iter__(self):
        #return self.dataset.slIndex.__iter__() #iterator over spam labels
        pass

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

    def __getstate__(self):
        #Return the state (for pickling) -- *don't* pickle parent or fns
        return  { '_data': self._data, '_typename': self._typename }

    def __setstate__(self, stateDict):
        self._computeFns = {}
        self._parent = None
        self._data = stateDict['_data']
        self._typename = stateDict['_typename']


    class NoCRDependenceError(Exception):
        """
        Exception indicating that a function has no confidence region
        dependence and therefore doesn't need to be re-computed when the
        confidence level changes
        """
        pass




class FigureCache(_collections.OrderedDict):
    pass


class SpecialsCache(_collections.OrderedDict):
    pass

class QtysCache(_collections.OrderedDict):
    pass




    def get_table(self, tableName, confidenceLevel=None, fmt="py", verbosity=0):
        """
        Get a report table in a specified format.  Tables are created on 
        the first request then cached for later requests for the same table.
        This method is typically used internally by other Results methods.

        Parameters
        ----------
        tableName : string
           The name of the table.

        confidenceLevel : float, optional
           If not None, then the confidence level (between 0 and 100) used to
           put error bars on the table's values (if possible). If None, no 
           confidence regions or intervals are included.

        fmt : { 'py', 'html', 'latex', 'ppt' }, optional
           The format of the table to be returned.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        string or object
           The requested table in the requested format.  'py' and 'ppt'
           tables are objects, 'html' and 'latex' tables are strings.
        """
        assert(self.bEssentialResultsSet)
        if self.tables.has_key(confidenceLevel) == False:
            self.tables[confidenceLevel] = {}
        if tableName not in self.tables[confidenceLevel]:
            self.tables[confidenceLevel][tableName] = self._generateTable(tableName, confidenceLevel, verbosity)
        return self.tables[confidenceLevel][tableName][fmt]


    def get_figure(self, figureName, verbosity=0):
        """
        Get a report figure.  Figures are created on the first
        request then cached for later requests for the same figure.
        This method is typically used internally by other Results methods.

        Parameters
        ----------
        figureName : string
           The name of the figure.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        ReportFigure
            The requested figure object.
        """
        assert(self.bEssentialResultsSet)
        if figureName not in self.figures:
            self.figures[figureName] = self._generateFigure(figureName, verbosity)
        return self.figures[figureName]



    def get_special(self, specialName, verbosity=0):
        """
        Get a "special item", which can be almost anything used in report
        or presentation construction.  This method is almost solely used 
        internally by other Results methods.

        Parameters
        ----------
        tableName : string
           The name of the special item.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        special item (type varies)
        """
        if specialName not in self.specials:
            self.specials[specialName] = self._generateSpecial(specialName, verbosity)
        return self.specials[specialName]
