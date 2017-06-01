from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the GatestringStructure class and supporting functionality."""

import collections as _collections
import itertools as _itertools
from ..tools import listtools as _lt

    
class GatestringPlaquette(object):
    """
    Encapsulates a single "plaquette" or "sub-matrix" within a
    gatestring-structure.  Typically this corresponds to a matrix
    whose rows and columns correspdond to measurement and preparation
    fiducial sequences.
    """
    
    def __init__(self, base, rows, cols, elements, aliases):
        """
        Create a new GatestringPlaquette.

        Parameters
        ----------
        base : GateString
            The "base" gate string of this plaquette.  Typically the sequence
            that is sandwiched between fiducial pairs.

        rows, cols : int
            The number of rows and columns of this plaquette.

        elements : list
            A list of `(i,j,s)` tuples where `i` and `j` are row and column
            indices and `s` is the corresponding `GateString`.

        aliases : dict
            A dictionary of gate label aliases that is carried along 
            for calls to :func:`expand_aliases`.
        """
        self.base = base
        self.rows = rows
        self.cols = cols
        self.elements = elements
        self.aliases = aliases

    def expand_aliases(self, dsFilter=None):
        """
        Returns a new GatestringPlaquette with any aliases
        expanded (within the gate strings).  Optionally keeps only
        those strings which, after alias expansion, are in `dsFilter`.

        Parameters
        ----------
        dsFilter : DataSet, optional
            If not None, keep only strings that are in this data set.

        Returns
        -------
        GatestringPlaquette
        """
        #find & replace aliased gate labels with their expanded form
        new_elements = []
        for i,j,s in self.elements:
            s2 = s if (self.aliases is None) else \
                 _lt.find_replace_tuple(s,self.aliases)
            
            if dsFilter is None or s2 in dsFilter:
                new_elements.append((i,j,s2))
            
        return GatestringPlaquette(self.base, self.rows, self.cols,
                                   new_elements, None)
        
    def __iter__(self):
        for i,j,s in self.elements:
            yield i,j,s
        #iterate over non-None entries (i,j,GateStr)

    def __len__(self):
        return len(self.elements)

    def copy(self):
        """
        Returns a copy of this `GatestringPlaquette`.
        """
        return GatestringPlaquette(self.base, self.rows, self.cols,
                                   self.elements, self.aliases)

    
class GatestringStructure(object):
    """
    Encapsulates a set of gate sequences, along with an associated structure.

    By "structure", we mean the ability to index the gate sequences by a 
    4-tuple (x, y, minor_x, minor_y) for displaying in nested color box plots,
    along with any aliases.
    """
    def __init__(self):
        pass

    def xvals(self):
        """ Returns a list of the x-values"""
        raise NotImplementedError("Derived class must implement this.")

    def yvals(self):
        """ Returns a list of the y-values"""
        raise NotImplementedError("Derived class must implement this.")

    def minor_xvals(self):
        """ Returns a list of the minor x-values"""
        raise NotImplementedError("Derived class must implement this.")

    def minor_yvals(self):
        """ Returns a list of the minor y-values"""
        raise NotImplementedError("Derived class must implement this.")
    
    def get_plaquette(self,x,y):
        """
        Returns a the plaquette at `(x,y)`.

        Parameters
        ----------
        x, y : values
            Coordinates which should be members of the lists returned by
            :method:`xvals` and :method:`yvals` respectively.

        Returns
        -------
        GatestringPlaquette
        """
        raise NotImplementedError("Derived class must implement this.")
    
    def create_plaquette(self, baseStr):
        """
        Creates a the plaquette for the given base string.

        Parameters
        ----------
        baseStr : GateString

        Returns
        -------
        GatestringPlaquette
        """
        raise NotImplementedError("Derived class must implement this.")

    def used_xvals(self):
        """Lists the x-values which have at least one non-empty plaquette"""
        return [ x for x in self.xvals() if any([ len(self.get_plaquette(x,y)) > 0
                                                  for y in self.yvals()]) ]
    
    def used_yvals(self):
        """Lists the y-values which have at least one non-empty plaquette"""
        return [ y for y in self.yvals() if any([ len(self.get_plaquette(x,y)) > 0
                                                  for x in self.xvals()]) ]
    
    def get_basestrings(self):
        """Lists the base strings (without duplicates) of all the plaquettes"""
        baseStrs = set()
        for x in self.xvals():
            for y in self.yvals():
                p = self.get_plaquette(x,y)
                if p is not None: baseStrs.add(p.base)
        return list(baseStrs)
        


    

class LsGermsStructure(GatestringStructure):
    """
    A type of gate string structure whereby sequences can be
    indexed by L, germ, preparation-fiducial, and measurement-fiducial.
    """
    def __init__(self, Ls, germs, prepStrs, effectStrs, aliases):
        """
        Create an empty gate string structure.

        Parameters
        ----------
        Ls : list of ints
            List of maximum lengths (x values)

        germs : list of GateStrings
            List of germ sequences (y values)

        prepStrs : list of GateStrings
            List of preparation fiducial sequences (minor x values)

        effecStrs : list of GateStrings
            List of measurement fiducial sequences (minor y values)

        aliases : dict
            Gate label aliases to be propagated to all plaquettes.
        """
        self.Ls = Ls[:]
        self.germs = germs[:]
        self.prepStrs = prepStrs[:]
        self.effectStrs = effectStrs[:]
        self.aliases = aliases.copy() if (aliases is not None) else None

        self.allstrs = []
        self._plaquettes = {}
        self._firsts = []
        self._baseStrToLGerm = {}

    #Base class access in terms of generic x,y coordinates
    def xvals(self):
        """ Returns a list of the x-values"""
        return self.Ls
    
    def yvals(self):
        """ Returns a list of the y-values"""
        return self.germs
    
    def minor_xvals(self):
        """ Returns a list of the minor x-values"""
        return self.prepStrs
    
    def minor_yvals(self):
        """ Returns a list of the minor y-values"""
        return self.effectStrs

    def add_plaquette(self, basestr, L, germ, fidpairs):
        """
        Adds a plaquette with the given fiducial pairs at the
        `(L,germ)` location.

        Parameters
        ----------
        basestr : GateString
            The base gate string of the new plaquette.

        L : int

        germ : GateString

        fidpairs : list
            A list if `(i,j)` tuples of integers, where `i` is a prepation
            fiducial index and `j` is a measurement fiducial index.

        Returns
        -------
        None
        """
        plaq = self.create_plaquette(basestr, fidpairs)

        for i,j,gatestr in plaq:
            if gatestr not in self.allstrs:
                self.allstrs.append(gatestr)
        self._plaquettes[(L,germ)] = plaq

        #keep track of which L,germ is the *first* one to "claim" a base string
        # (useful for *not* duplicating data in color box plots)
        if basestr not in self._baseStrToLGerm:
            self._firsts.append( (L,germ) )
            self._baseStrToLGerm[ basestr ] = (L,germ)
        
    def add_unindexed(self, gsList):
        """
        Adds unstructured gate strings (not in any plaquette).

        Parameters
        ----------
        gsList : list of GateStrings
            The gate strings to add.

        Returns
        -------
        None
        """
        for gatestr in gsList:
            if gatestr not in self.allstrs:
                self.allstrs.append(gatestr)

    def done_adding_strings(self):
        """
        Called to indicate the user is done adding plaquettes.
        """
        #placeholder in case there's some additional init we need to do.
        pass
                
    def get_plaquette(self, L, germ, onlyfirst=True):
        """
        Returns a the plaquette at `(L,germ)`.

        Parameters
        ----------
        L : int
            The maximum length.

        germ : Gatestring
            The germ.

        onlyfirst : bool, optional
            If True, then when multiple plaquettes have been added with the
            same base string, only the *first* added plaquette will be 
            returned normally.  Requests for the other plaquettes will be
            given an empty plaquette.  This behavior is useful for color 
            box plots where we wish to avoid duplicated data.

        Returns
        -------
        GatestringPlaquette
        """
        if not onlyfirst or (L,germ) in self._firsts:
            return self._plaquettes[(L,germ)]
        else:
            basestr = self._plaquettes[(L,germ)].base
            return self.create_plaquette(basestr,[]) # no elements

    def truncate(self, Ls=None, germs=None, prepStrs=None, effectStrs=None):
        """
        TODO: docstring
        """
        raise NotImplementedError("future capability")


    def create_plaquette(self, baseStr, fidpairs=None):
        """
        Creates a the plaquette for the given base string and pairs.

        Parameters
        ----------
        baseStr : GateString

        fidpairs : list
            A list if `(i,j)` tuples of integers, where `i` is a prepation
            fiducial index and `j` is a measurement fiducial index.  If
            None, then all pairs are included (a "full" plaquette is created).

        Returns
        -------
        GatestringPlaquette
        """
        if fidpairs is None:
            fidpairs = list(_itertools.product(range(len(self.prepStrs)),
                                               range(len(self.effectStrs))))

        elements = [ (j,i,self.prepStrs[i] + baseStr + self.effectStrs[j])
                     for i,j in fidpairs ] #note preps are *cols* not rows
        
        return GatestringPlaquette(baseStr, len(self.effectStrs),
                            len(self.prepStrs), elements, self.aliases)

    def copy(self):
        """
        Returns a copy of this `LsGermsStructure`.
        """
        cpy = LsGermsStructure(self.Ls, self.germs, self.prepStrs,
                               self.effectStrs, self.aliases)
        cpy.allstrs = self.allstrs[:]
        cpy._plaquettes = { k: v.copy() for k,v in self._plaquettes.items() }
        cpy._firsts = self._firsts[:]
        cpy._baseStrToGerm = self._baseStrToLGerm.copy()
        return cpy
