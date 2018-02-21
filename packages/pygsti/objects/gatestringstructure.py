""" Defines the GatestringStructure class and supporting functionality."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import copy as _copy
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
        self.elements = elements[:]
        self.aliases = aliases

        #After compiling:
        self._elementIndicesByStr = None
        self._outcomesByStr = None
        self.num_compiled_elements = None

    def expand_aliases(self, dsFilter=None, gatestring_compiler=None):
        """
        Returns a new GatestringPlaquette with any aliases
        expanded (within the gate strings).  Optionally keeps only
        those strings which, after alias expansion, are in `dsFilter`.

        Parameters
        ----------
        dsFilter : DataSet, optional
            If not None, keep only strings that are in this data set.

        gatestring_compiler : GateSet, optional
            Whether to call `compile_gatestrings(gatestring_compiler)`
            on the new GatestringPlaquette.

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
            
        ret = GatestringPlaquette(self.base, self.rows, self.cols,
                                   new_elements, None)
        if gatestring_compiler is not None:
            ret.compile_gatestrings(gatestring_compiler)
        return ret

    def get_all_strs(self):
        """Return a list of all the gate strings contained in this plaquette"""
        return [s for i,j,s in self.elements]

    def compile_gatestrings(self, gateset):
        """
        Compiles this plaquette so that the `num_compiled_elements` property and
        the `iter_compiled()` method may be used.

        Parameters
        ----------
        gateset : GateSet
            The gate set used to perform the compiling.
        """
        all_strs = self.get_all_strs()
        if len(all_strs) > 0:
            rawmap, self._elementIndicesByStr, self._outcomesByStr, nEls = \
              gateset.compile_gatestrings(all_strs)
        else:
            nEls = 0 #nothing to compile
        self.num_compiled_elements = nEls

    def iter_compiled(self):
        assert(self.num_compiled_elements is not None), \
            "Plaquette must be compiled first!"
        for k,(i,j,s) in enumerate(self.elements):
            yield i,j,s,self._elementIndicesByStr[k],self._outcomesByStr[k]
            
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
        aliases = _copy.deepcopy(self.aliases) if (self.aliases is not None) \
                  else None
        return GatestringPlaquette(self.base, self.rows, self.cols,
                                   self.elements[:], aliases)

    
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

    def plaquette_rows_cols(self):
        """
        Return the number of rows and columns contained in each plaquette of 
        this GateStringStructure.

        Returns
        -------
        rows, cols : int
        """
        return len(self.minor_yvals()), len(self.minor_xvals())

    
    def get_basestrings(self):
        """Lists the base strings (without duplicates) of all the plaquettes"""
        baseStrs = set()
        for x in self.xvals():
            for y in self.yvals():
                p = self.get_plaquette(x,y)
                if p is not None and p.base is not None:
                    baseStrs.add(p.base)
        return list(baseStrs)

    def compile_plaquettes(self, gateset):
        """
        Compiles all the plaquettes in this structure so that their
        `num_compiled_elements` property and the `iter_compiled()` methods
        may be used.

        Parameters
        ----------
        gateset : GateSet
            The gate set used to perform the compiling.
        """
        for x in self.xvals():
            for y in self.yvals():
                p = self.get_plaquette(x,y)
                if p is not None:
                    p.compile_gatestrings(gateset)
    

class LsGermsStructure(GatestringStructure):
    """
    A type of gate string structure whereby sequences can be
    indexed by L, germ, preparation-fiducial, and measurement-fiducial.
    """
    def __init__(self, Ls, germs, prepStrs, effectStrs, aliases=None,
                 sequenceRules=None):
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

        sequenceRules : list, optional
            A list of `(find,replace)` 2-tuples which specify string replacement
            rules.  Both `find` and `replace` are tuples of gate labels 
            (or `GateString` objects).
        """
        self.Ls = Ls[:]
        self.germs = germs[:]
        self.prepStrs = prepStrs[:]
        self.effectStrs = effectStrs[:]
        self.aliases = aliases.copy() if (aliases is not None) else None
        self.sequenceRules = sequenceRules[:] if (sequenceRules is not None) else None


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

    def add_plaquette(self, basestr, L, germ, fidpairs=None, dsfilter=None):
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
            fiducial index and `j` is a measurement fiducial index.  None
            can be used to mean all pairs.

        dsfilter : DataSet, optional
            If not None, check that this data set contains all of the 
            gate strings being added.  If dscheck does not contain a gate
            sequence, it is *not* added.
        
        Returns
        -------
        missing : list
            A list of `(prep_fiducial, germ, L, effect_fiducial, entire_string)`
            tuples indicating which sequences were not found in `dsfilter`.
        """

        missing_list = []
        from ..construction import gatestringconstruction as _gstrc #maybe move used routines to a gatestringtools.py?

        if fidpairs is None:
            fidpairs = list(_itertools.product(range(len(self.prepStrs)),
                                               range(len(self.effectStrs))))
        if dsfilter:
            inds_to_remove = []
            for k,(i,j) in enumerate(fidpairs):
                el = self.prepStrs[i] + basestr + self.effectStrs[j]
                trans_el = _gstrc.translate_gatestring(el, self.aliases)
                if trans_el not in dsfilter:
                    missing_list.append( (self.prepStrs[i],germ,L,self.effectStrs[j],el) )
                    inds_to_remove.append(k)

            if len(inds_to_remove) > 0:
                fidpairs = fidpairs[:] #copy
                for i in reversed(inds_to_remove):
                    del fidpairs[i]

        plaq = self.create_plaquette(basestr, fidpairs)
        self.allstrs.extend( [ _gstrc.manipulate_gatestring(gatestr,self.sequenceRules)
                               for i,j,gatestr in plaq ] )
        _lt.remove_duplicates_in_place(self.allstrs)

        self._plaquettes[(L,germ)] = plaq

        #keep track of which L,germ is the *first* one to "claim" a base string
        # (useful for *not* duplicating data in color box plots)
        if basestr not in self._baseStrToLGerm:
            self._firsts.append( (L,germ) )
            self._baseStrToLGerm[ basestr ] = (L,germ)

        return missing_list

        
    def add_unindexed(self, gsList, dsfilter=None):
        """
        Adds unstructured gate strings (not in any plaquette).

        Parameters
        ----------
        gsList : list of GateStrings
            The gate strings to add.

        dsfilter : DataSet, optional
            If not None, check that this data set contains all of the 
            gate strings being added.  If dscheck does not contain a gate
            sequence, it is *not* added.

        Returns
        -------
        missing : list
            A list of elements in `gsList` which were not found in `dsfilter`
            and therefore not added.
        """
        from ..construction import gatestringconstruction as _gstrc #maybe move used routines to a gatestringtools.py?
        
        missing_list = []
        for gatestr in gsList:
            if gatestr not in self.allstrs:
                if dsfilter:
                    trans_gatestr = _gstrc.translate_gatestring(gatestr, self.aliases)
                    if trans_gatestr not in dsfilter:
                        missing_list.append( gatestr )
                        continue
                self.allstrs.append(gatestr)
        return missing_list

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
        if (L,germ) not in self._plaquettes:
            p =  self.create_plaquette(None,[]) # no elements
            p.compile_gatestrings(None) # just marks as "compiled"
            return p
        
        if not onlyfirst or (L,germ) in self._firsts:
            return self._plaquettes[(L,germ)]
        else:
            basestr = self._plaquettes[(L,germ)].base
            p = self.create_plaquette(basestr,[]) # no elements
            p.compile_gatestrings(None) # just marks as "compiled"
            return p

    def truncate(self, Ls=None, germs=None, prepStrs=None, effectStrs=None):
        """
        A future capability: truncate this gate string structure to a
        subset of its current strings.
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

    def plaquette_rows_cols(self):
        """
        Return the number of rows and columns contained in each plaquette of 
        this LsGermsStructure.

        Returns
        -------
        rows, cols : int
        """
        return len(self.effectStrs), len(self.prepStrs)

    def copy(self):
        """
        Returns a copy of this `LsGermsStructure`.
        """
        cpy = LsGermsStructure(self.Ls, self.germs, self.prepStrs,
                               self.effectStrs, self.aliases, self.sequenceRules)
        cpy.allstrs = self.allstrs[:]
        cpy._plaquettes = { k: v.copy() for k,v in self._plaquettes.items() }
        cpy._firsts = self._firsts[:]
        cpy._baseStrToGerm = _copy.deepcopy(self._baseStrToLGerm.copy())
        return cpy
