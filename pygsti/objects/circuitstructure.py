""" Defines the CircuitStructure class and supporting functionality."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy
import uuid as _uuid
import itertools as _itertools
from ..tools import listtools as _lt
from .circuit import Circuit as _Circuit


class CircuitPlaquette(object):
    """
    Encapsulates a single "plaquette" or "sub-matrix" within a
    circuit-structure.  Typically this corresponds to a matrix
    whose rows and columns correspdond to measurement and preparation
    fiducial sequences.
    """

    def __init__(self, base, rows, cols, elements, aliases, fidpairs=None):
        """
        Create a new CircuitPlaquette.

        Parameters
        ----------
        base : Circuit
            The "base" operation sequence of this plaquette.  Typically the sequence
            that is sandwiched between fiducial pairs.

        rows, cols : int
            The number of rows and columns of this plaquette.

        elements : list
            A list of `(i,j,s)` tuples where `i` and `j` are row and column
            indices and `s` is the corresponding `Circuit`.

        aliases : dict
            A dictionary of operation label aliases that is carried along
            for calls to :func:`expand_aliases`.

        fidpairs : list, optional
            A list of `(prepStr, effectStr)` tuples specifying how
            `elements` is generated from `base`, i.e. by
            `prepStr + base + effectStr`.
        """
        self.base = base
        self.rows = rows
        self.cols = cols
        self.elements = elements[:]
        self.fidpairs = fidpairs[:] if (fidpairs is not None) else None
        self.aliases = aliases

        #After compiling:
        self._elementIndicesByStr = None
        self._outcomesByStr = None
        self.num_simplified_elements = None

    def expand_aliases(self, ds_filter=None, circuit_simplifier=None):
        """
        Returns a new CircuitPlaquette with any aliases
        expanded (within the operation sequences).  Optionally keeps only
        those strings which, after alias expansion, are in `ds_filter`.

        Parameters
        ----------
        ds_filter : DataSet, optional
            If not None, keep only strings that are in this data set.

        circuit_simplifier : Model, optional
            Whether to call `simplify_circuits(circuit_simplifier)`
            on the new CircuitPlaquette.

        Returns
        -------
        CircuitPlaquette
        """
        #find & replace aliased operation labels with their expanded form
        new_elements = []
        new_fidpairs = [] if (self.fidpairs is not None) else None
        for k, (i, j, s) in enumerate(self.elements):
            s2 = s if (self.aliases is None) else \
                s.replace_layers_with_aliases(self.aliases)

            if new_fidpairs:
                prep, effect = self.fidpairs[k]
                prep2 = prep if (self.aliases is None) else \
                    prep.replace_layers_with_aliases(self.aliases)
                effect2 = effect if (self.aliases is None) else \
                    effect.replace_layers_with_aliases(self.aliases)

            if ds_filter is None or s2 in ds_filter:
                new_elements.append((i, j, s2))
                if new_fidpairs: new_fidpairs.append((prep2, effect2))

        ret = CircuitPlaquette(self.base, self.rows, self.cols,
                               new_elements, None, new_fidpairs)
        if circuit_simplifier is not None:
            ret.simplify_circuits(circuit_simplifier, ds_filter)
        return ret

    def get_all_strs(self):
        """Return a list of all the operation sequences contained in this plaquette"""
        return [s for i, j, s in self.elements]

    def simplify_circuits(self, model, dataset=None):
        """
        Simplified this plaquette so that the `num_simplified_elements` property and
        the `iter_simplified()` and `elementvec_to_matrix` methods may be used.

        Parameters
        ----------
        model : Model
            The model used to perform the compiling.

        dataset : DataSet, optional
            If not None, restrict what is simplified to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.
        """
        all_strs = self.get_all_strs()
        if len(all_strs) > 0:
            rawmap, self._elementIndicesByStr, self._outcomesByStr, nEls = \
                model.simplify_circuits(all_strs, dataset)
        else:
            nEls = 0  # nothing to simplify
        self.num_simplified_elements = nEls

    def iter_simplified(self):
        assert(self.num_simplified_elements is not None), \
            "Plaquette must be simplified first!"
        for k, (i, j, s) in enumerate(self.elements):
            yield i, j, s, self._elementIndicesByStr[k], self._outcomesByStr[k]

    def __iter__(self):
        for i, j, s in self.elements:
            yield i, j, s
        #iterate over non-None entries (i,j,GateStr)

    def __len__(self):
        return len(self.elements)

    def elementvec_to_matrix(self, elementvec, mergeop="sum"):
        """
        Form a matrix of values for this plaquette from  a given vector,
        `elementvec` of individual-outcome elements (e.g. the bulk probabilities
        computed by a model).

        Parameters
        ----------
        elementvec : numpy array
            An array of length `self.num_simplified_elements` containting the
            values to use when constructing a matrix of values for this plaquette.

        mergeop : "sum" or format string, optional
            Dictates how to combine the `elementvec` components corresponding to a single
            plaquette entry (circuit).  If "sum", the returned array contains summed
            values.  If a format string, e.g. `"%.2f"`, then the so-formatted components
            are joined together with separating commas, and the resulting array contains
            string (object-type) entries.

        Returns
        -------
        numpy array
        """
        if mergeop == "sum":
            ret = _np.nan * _np.ones((self.rows, self.cols), 'd')
            for i, j, opstr, elIndices, _ in self.iter_simplified():
                ret[i, j] = sum(elementvec[elIndices])
        elif '%' in mergeop:
            fmt = mergeop
            ret = _np.nan * _np.ones((self.rows, self.cols), dtype=_np.object)
            for i, j, opstr, elIndices, _ in self.iter_simplified():
                ret[i, j] = ", ".join(["NaN" if _np.isnan(x) else
                                       (fmt % x) for x in elementvec[elIndices]])
        else:
            raise ValueError("Invalid `mergeop` arg: %s" % str(mergeop))
        return ret

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`
        and return a new `CircuitPlaquette` object.

        Parameters
        ----------
        processor_fn : function
            A function which takes a single Circuit argument and returns
            another (or the same) Circuit.

        updated_aliases : dict, optional
            Because the Label keys of an alias dictionary (maps
            Label -> Circuit) cannot be processed as a Circuit, one must
            supply a manualy processed alias dictionary.  If you don't use
            alias dictionaries just leave this set to None.

        Returns
        -------
        CircuitPlaquette
        """
        P = processor_fn
        updated_elements = [(i, j, P(s)) for i, j, s in self.elements]
        updated_fidpairs = [(P(prep), P(meas)) for prep, meas in self.fidpairs]
        return CircuitPlaquette(P(self.base), self.rows, self.cols,
                                updated_elements, updated_aliases, updated_fidpairs)

    def copy(self):
        """
        Returns a copy of this `CircuitPlaquette`.
        """
        aliases = _copy.deepcopy(self.aliases) if (self.aliases is not None) \
            else None
        return CircuitPlaquette(self.base, self.rows, self.cols,
                                self.elements[:], aliases, self.fidpairs)


class CircuitStructure(object):
    """
    Encapsulates a set of operation sequences, along with an associated structure.

    By "structure", we mean the ability to index the operation sequences by a
    4-tuple (x, y, minor_x, minor_y) for displaying in nested color box plots,
    along with any aliases.
    """

    def __init__(self):
        self.uuid = _uuid.uuid4()  # like a persistent id(),
        # useful for peristent (file) caches

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        if 'uuid' not in state_dict:
            self.uuid = _uuid.uuid4()  # create a new uuid

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

    def get_plaquette(self, x, y):
        """
        Returns a the plaquette at `(x,y)`.

        Parameters
        ----------
        x, y : values
            Coordinates which should be members of the lists returned by
            :method:`xvals` and :method:`yvals` respectively.

        Returns
        -------
        CircuitPlaquette
        """
        raise NotImplementedError("Derived class must implement this.")

    def create_plaquette(self, base_str):
        """
        Creates a the plaquette for the given base string.

        Parameters
        ----------
        base_str : Circuit

        Returns
        -------
        CircuitPlaquette
        """
        raise NotImplementedError("Derived class must implement this.")

    def used_xvals(self):
        """Lists the x-values which have at least one non-empty plaquette"""
        return [x for x in self.xvals() if any([len(self.get_plaquette(x, y)) > 0
                                                for y in self.yvals()])]

    def used_yvals(self):
        """Lists the y-values which have at least one non-empty plaquette"""
        return [y for y in self.yvals() if any([len(self.get_plaquette(x, y)) > 0
                                                for x in self.xvals()])]

    def plaquette_rows_cols(self):
        """
        Return the number of rows and columns contained in each plaquette of
        this CircuitStructure.

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
                p = self.get_plaquette(x, y)
                if p is not None and p.base is not None:
                    baseStrs.add(p.base)
        return list(baseStrs)

    def simplify_plaquettes(self, model, dataset=None):
        """
        Simplifies all the plaquettes in this structure so that their
        `num_simplified_elements` property and the `iter_simplified()` methods
        may be used.

        Parameters
        ----------
        model : Model
            The model used to perform the compiling.

        dataset : DataSet, optional
            If not None, restrict what is simplified to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.
        """
        for x in self.xvals():
            for y in self.yvals():
                p = self.get_plaquette(x, y)
                if p is not None:
                    p.simplify_circuits(model, dataset)


class LsGermsStructure(CircuitStructure):
    """
    A type of operation sequence structure whereby sequences can be
    indexed by L, germ, preparation-fiducial, and measurement-fiducial.
    """

    def __init__(self, max_lengths, germs, prep_fiducials, meas_fiducials, aliases=None,
                 sequence_rules=None):
        """
        Create an empty operation sequence structure.

        Parameters
        ----------
        max_lengths : list of ints
            List of maximum lengths (x values)

        germs : list of Circuits
            List of germ sequences (y values)

        prep_fiducials : list of Circuits
            List of preparation fiducial sequences (minor x values)

        effecStrs : list of Circuits
            List of measurement fiducial sequences (minor y values)

        aliases : dict
            Operation label aliases to be propagated to all plaquettes.

        sequence_rules : list, optional
            A list of `(find,replace)` 2-tuples which specify string replacement
            rules.  Both `find` and `replace` are tuples of operation labels
            (or `Circuit` objects).
        """
        self.Ls = max_lengths[:]
        self.germs = germs[:]
        self.prep_fiducials = prep_fiducials[:]
        self.meas_fiducials = meas_fiducials[:]
        self.aliases = aliases.copy() if (aliases is not None) else None
        self.sequenceRules = sequence_rules[:] if (sequence_rules is not None) else None

        self.allstrs = []
        self.allstrs_set = set()
        self.unindexed = []  # unindexed strings
        self._plaquettes = {}
        self._firsts = []
        self._baseStrToLGerm = {}
        super(LsGermsStructure, self).__init__()

    def __iter__(self):
        yield from self.allstrs

    def __len__(self):
        return len(self.allstrs)

    def __getitem__(self, key: int):
        return self.allstrs[key]

    #Base class access in terms of generic x,y coordinates
    def xvals(self):
        """ Returns a list of the x-values"""
        return self.Ls

    def yvals(self):
        """ Returns a list of the y-values"""
        return self.germs

    def minor_xvals(self):
        """ Returns a list of the minor x-values"""
        return self.prep_fiducials

    def minor_yvals(self):
        """ Returns a list of the minor y-values"""
        return self.meas_fiducials

    def add_plaquette(self, basestr, max_length, germ, fidpairs=None, dsfilter=None):
        """
        Adds a plaquette with the given fiducial pairs at the
        `(max_length,germ)` location.

        Parameters
        ----------
        basestr : Circuit
            The base operation sequence of the new plaquette.

        max_length : int

        germ : Circuit

        fidpairs : list
            A list if `(i,j)` tuples of integers, where `i` is a prepation
            fiducial index and `j` is a measurement fiducial index.  None
            can be used to mean all pairs.

        dsfilter : DataSet, optional
            If not None, check that this data set contains all of the
            operation sequences being added.  If dscheck does not contain a gate
            sequence, it is *not* added.

        Returns
        -------
        missing : list
            A list of `(prep_fiducial, germ, max_length, effect_fiducial, entire_string)`
            tuples indicating which sequences were not found in `dsfilter`.
        """

        missing_list = []
        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?

        if fidpairs is None:
            fidpairs = list(_itertools.product(range(len(self.prep_fiducials)),
                                               range(len(self.meas_fiducials))))
        if dsfilter:
            inds_to_remove = []
            for k, (i, j) in enumerate(fidpairs):
                el = self.prep_fiducials[i] + basestr + self.meas_fiducials[j]
                trans_el = _gstrc.translate_circuit(el, self.aliases)
                if trans_el not in dsfilter:
                    missing_list.append((self.prep_fiducials[i], germ, max_length, self.meas_fiducials[j], el))
                    inds_to_remove.append(k)

            if len(inds_to_remove) > 0:
                fidpairs = fidpairs[:]  # copy
                for i in reversed(inds_to_remove):
                    del fidpairs[i]

        plaq = self.create_plaquette(basestr, fidpairs)

        for x in (_gstrc.manipulate_circuit(opstr, self.sequenceRules) for i, j, opstr in plaq):
            if x not in self.allstrs_set:
                self.allstrs_set.add(x)
                self.allstrs.append(x)
        #_lt.remove_duplicates_in_place(self.allstrs) # above block does this more efficiently

        self._plaquettes[(max_length, germ)] = plaq

        #keep track of which max_length,germ is the *first* one to "claim" a base string
        # (useful for *not* duplicating data in color box plots)
        if basestr not in self._baseStrToLGerm:
            self._firsts.append((max_length, germ))
            self._baseStrToLGerm[basestr] = (max_length, germ)

        return missing_list

    def add_unindexed(self, gs_list, dsfilter=None):
        """
        Adds unstructured operation sequences (not in any plaquette).

        Parameters
        ----------
        gs_list : list of Circuits
            The operation sequences to add.

        dsfilter : DataSet, optional
            If not None, check that this data set contains all of the
            operation sequences being added.  If dscheck does not contain a gate
            sequence, it is *not* added.

        Returns
        -------
        missing : list
            A list of elements in `gs_list` which were not found in `dsfilter`
            and therefore not added.
        """
        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?
        #if dsfilter and len(dsfilter) > 8000: dsfilter = None # TEST DEBUG - remove dsfilter check

        missing_list = []
        for opstr in gs_list:
            if opstr not in self.allstrs_set:
                if dsfilter:
                    trans_opstr = _gstrc.translate_circuit(opstr, self.aliases)
                    if trans_opstr not in dsfilter:
                        missing_list.append(opstr)
                        continue
                self.allstrs_set.add(opstr)
                self.allstrs.append(opstr)
                self.unindexed.append(opstr)
        return missing_list

    def done_adding_strings(self):
        """
        Called to indicate the user is done adding plaquettes.
        """
        #placeholder in case there's some additional init we need to do.
        pass

    def get_plaquette(self, max_length, germ, onlyfirst=True):
        """
        Returns a the plaquette at `(max_length,germ)`.

        Parameters
        ----------
        max_length : int
            The maximum length.

        germ : Circuit
            The germ.

        onlyfirst : bool, optional
            If True, then when multiple plaquettes have been added with the
            same base string, only the *first* added plaquette will be
            returned normally.  Requests for the other plaquettes will be
            given an empty plaquette.  This behavior is useful for color
            box plots where we wish to avoid duplicated data.

        Returns
        -------
        CircuitPlaquette
        """
        if (max_length, germ) not in self._plaquettes:
            p = self.create_plaquette(None, [])  # no elements
            p.simplify_circuits(None)  # just marks as "simplified"
            return p

        if not onlyfirst or (max_length, germ) in self._firsts:
            return self._plaquettes[(max_length, germ)]
        else:
            basestr = self._plaquettes[(max_length, germ)].base
            p = self.create_plaquette(basestr, [])  # no elements
            p.simplify_circuits(None)  # just marks as "simplified"
            return p

    def truncate(self, max_lengths=None, germs=None, prep_fiducials=None, meas_fiducials=None, seqs=None):
        """
        Truncate this operation sequence structure to a subset of its current strings.

        Parameters
        ----------
        max_lengths : list, optional
            The integer L-values to keep.  If None, then all are kept.

        germs : list, optional
            The (Circuit) germs to keep.  If None, then all are kept.

        prep_fiducials, meas_fiducials : list, optional
            The (Circuit) preparation and measurement fiducial sequences to keep.
            If None, then all are kept.

        seqs : list
            Keep only sequences present in this list of Circuit objects.

        Returns
        -------
        LsGermsStructure
        """
        max_lengths = self.Ls if (max_lengths is None) else max_lengths
        germs = self.germs if (germs is None) else germs
        prep_fiducials = self.prep_fiducials if (prep_fiducials is None) else prep_fiducials
        meas_fiducials = self.meas_fiducials if (meas_fiducials is None) else meas_fiducials
        cpy = LsGermsStructure(max_lengths, germs, prep_fiducials,
                               meas_fiducials, self.aliases, self.sequenceRules)

        #OLD iPreps = [i for i, prepStr in enumerate(self.prep_fiducials) if prepStr in prep_fiducials]
        #OLD iEffects = [i for i, eStr in enumerate(self.meas_fiducials) if eStr in meas_fiducials]
        #OLD fidpairs = list(_itertools.product(iPreps, iEffects))
        all_fidpairs = list(_itertools.product(list(range(len(prep_fiducials))), list(range(len(meas_fiducials)))))

        for (L, germ), plaq in self._plaquettes.items():
            basestr = plaq.base
            if seqs is None:
                fidpairs = all_fidpairs
            else:
                fidpairs = []
                for i, j in all_fidpairs:
                    if prep_fiducials[i] + basestr + meas_fiducials[j] in seqs:
                        fidpairs.append((i, j))

            if (L in max_lengths) and (germ in germs):
                cpy.add_plaquette(basestr, L, germ, fidpairs)

        cpy.add_unindexed(self.unindexed)  # preserve unindexed strings
        return cpy

    def create_plaquette(self, base_str, fidpairs=None):
        """
        Creates a the plaquette for the given base string and pairs.

        Parameters
        ----------
        base_str : Circuit

        fidpairs : list
            A list if `(i,j)` tuples of integers, where `i` is a prepation
            fiducial index and `j` is a measurement fiducial index.  If
            None, then all pairs are included (a "full" plaquette is created).

        Returns
        -------
        CircuitPlaquette
        """
        if fidpairs is None:
            fidpairs = list(_itertools.product(range(len(self.prep_fiducials)),
                                               range(len(self.meas_fiducials))))

        elements = [(j, i, self.prep_fiducials[i] + base_str + self.meas_fiducials[j])
                    for i, j in fidpairs]  # note preps are *cols* not rows
        real_fidpairs = [(self.prep_fiducials[i], self.meas_fiducials[j]) for i, j in fidpairs]  # circuits, not indices

        return CircuitPlaquette(base_str, len(self.meas_fiducials),
                                len(self.prep_fiducials), elements,
                                self.aliases, real_fidpairs)

    def plaquette_rows_cols(self):
        """
        Return the number of rows and columns contained in each plaquette of
        this LsGermsStructure.

        Returns
        -------
        rows, cols : int
        """
        return len(self.meas_fiducials), len(self.prep_fiducials)

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`,
        returning a new circuit structure with processed circuits.

        Parameters
        ----------
        processor_fn : function
            A function which takes a single Circuit argument and returns
            another (or the same) Circuit.

        updated_aliases : dict, optional
            Because the Label keys of an alias dictionary (maps
            Label -> Circuit) cannot be processed as a Circuit, one must
            supply a manualy processed alias dictionary.  If you don't use
            alias dictionaries just leave this set to None.

        Returns
        -------
        LsGermsStructure
        """
        P = processor_fn  # shorhand
        cpy = LsGermsStructure(self.Ls, list(map(P, self.germs)),
                               list(map(P, self.prep_fiducials)), list(map(P, self.meas_fiducials)),
                               updated_aliases, self.sequenceRules)
        cpy.allstrs = list(map(P, self.allstrs))
        cpy.allstrs_set = set(cpy.allstrs)
        cpy.unindexed = list(map(P, self.unindexed))
        cpy._plaquettes = {k: v.process_circuits(P, updated_aliases) for k, v in self._plaquettes.items()}
        cpy._firsts = [(L, P(germ)) for (L, germ) in self._firsts]
        cpy._baseStrToLGerm = {P(base): (L, P(germ)) for base, (L, germ) in self._baseStrToLGerm.items()}
        return cpy

    def copy(self):
        """
        Returns a copy of this `LsGermsStructure`.
        """
        cpy = LsGermsStructure(self.Ls, self.germs, self.prep_fiducials,
                               self.meas_fiducials, self.aliases, self.sequenceRules)
        cpy.allstrs = self.allstrs[:]
        cpy.allstrs_set = self.allstrs_set.copy()
        cpy.unindexed = self.unindexed[:]
        cpy._plaquettes = {k: v.copy() for k, v in self._plaquettes.items()}
        cpy._firsts = self._firsts[:]
        cpy._baseStrToLGerm = _copy.deepcopy(self._baseStrToLGerm.copy())
        return cpy


class LsGermsSerialStructure(CircuitStructure):
    """
    A type of operation sequence structure whereby sequences can be
    indexed by L, germ, preparation-fiducial, and measurement-fiducial.
    """

    @classmethod
    def from_list(cls, circuit_list, dsfilter=None):
        """
        Creates a LsGermsSerialStructure out of a simple circuit list.

        This factory method is used when a default structure is required for a
        given simple list of circuits.
        """
        max_length = 0  # just a single "0" length
        empty_circuit = _Circuit((), line_labels=circuit_list[0].line_labels if len(circuit_list) > 0 else "auto")
        square_side = int(_np.ceil(_np.sqrt(len(circuit_list))))
        ret = cls([max_length], [empty_circuit], square_side, square_side)

        fidpairs = [(c, empty_circuit) for c in circuit_list]
        ret.add_plaquette(empty_circuit, max_length, empty_circuit, fidpairs, dsfilter)
        return ret

    def __init__(self, max_lengths, germs, n_minor_rows, n_minor_cols, aliases=None,
                 sequence_rules=None):
        """
        Create an empty LsGermSerialStructure.

        This type of operation sequence structure is useful for holding multi-qubit
        operation sequences which have a germ and max-length structure but which have
        widely varying fiducial sequences so that is it not useful to use the
        minor axes (rows/columns) to represent the *same* fiducials for all
        (L,germ) plaquettes.

        Parameters
        ----------
        max_lengths : list of ints
            List of maximum lengths (x values)

        germs : list of Circuits
            List of germ sequences (y values)

        n_minor_rows, n_minor_cols : int
            The number of minor rows and columns to allocate space for.
            These should be the maximum values required for any plaquette.

        aliases : dict
            Operation label aliases to be propagated to all plaquettes.

        sequence_rules : list, optional
            A list of `(find,replace)` 2-tuples which specify string replacement
            rules.  Both `find` and `replace` are tuples of operation labels
            (or `Circuit` objects).
        """
        self.Ls = max_lengths[:]
        self.germs = germs[:]
        self.nMinorRows = n_minor_rows
        self.nMinorCols = n_minor_cols
        self.aliases = aliases.copy() if (aliases is not None) else None
        self.sequenceRules = sequence_rules[:] if (sequence_rules is not None) else None

        self.allstrs = []
        self.allstrs_set = set()
        self.unindexed = []
        self._plaquettes = {}
        self._firsts = []
        self._baseStrToLGerm = {}
        super(LsGermsSerialStructure, self).__init__()

    #Base class access in terms of generic x,y coordinates
    def xvals(self):
        """ Returns a list of the x-values"""
        return self.Ls

    def yvals(self):
        """ Returns a list of the y-values"""
        return self.germs

    def minor_xvals(self):
        """ Returns a list of the minor x-values (0-based integers)"""
        return list(range(self.nMinorCols))

    def minor_yvals(self):
        """ Returns a list of the minor y-values (0-based integers)"""
        return list(range(self.nMinorRows))

    def add_plaquette(self, basestr, max_length, germ, fidpairs, dsfilter=None):
        """
        Adds a plaquette with the given fiducial pairs at the
        `(max_length,germ)` location.

        Parameters
        ----------
        basestr : Circuit
            The base operation sequence of the new plaquette, typically `germ^power`
            such that `len(germ^power) <= max_length`.

        max_length : int
            The maximum length value.

        germ : Circuit
            The germ string.

        fidpairs : list
            A list if `(prep,meas)` tuples of Circuit objects, specifying
            the fiducial pairs for this plaquette.  Note that this argument
            is different from the corresponding one in
            :method:`LsGermsStructure.add_plaquette` which takes pairs of
            *integer* indices and can be None.  In the present case, this
            argument is mandatory and contains tuples of operation sequences.

        dsfilter : DataSet, optional
            If not None, check that this data set contains all of the
            operation sequences being added.  If dscheck does not contain a gate
            sequence, it is *not* added.

        Returns
        -------
        missing : list
            A list of `(prep_fiducial, germ, max_length, effect_fiducial, entire_string)`
            tuples indicating which sequences were not found in `dsfilter`.
        """

        missing_list = []
        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?

        if dsfilter:  # and len(dsfilter) < 8000: # TEST DEBUG
            inds_to_remove = []
            for k, (prepStr, effectStr) in enumerate(fidpairs):
                el = prepStr + basestr + effectStr
                trans_el = _gstrc.translate_circuit(el, self.aliases)
                if trans_el not in dsfilter:
                    missing_list.append((prepStr, germ, max_length, effectStr, el))
                    inds_to_remove.append(k)

            if len(inds_to_remove) > 0:
                fidpairs = fidpairs[:]  # copy
                for i in reversed(inds_to_remove):
                    del fidpairs[i]

        plaq = self.create_plaquette(basestr, fidpairs)

        for x in (_gstrc.manipulate_circuit(opstr, self.sequenceRules) for i, j, opstr in plaq):
            if x not in self.allstrs_set:
                self.allstrs_set.add(x)
                self.allstrs.append(x)
        # _lt.remove_duplicates_in_place(self.allstrs) # above block does this more efficiently

        self._plaquettes[(max_length, germ)] = plaq

        #keep track of which max_length,germ is the *first* one to "claim" a base string
        # (useful for *not* duplicating data in color box plots)
        if basestr not in self._baseStrToLGerm:
            self._firsts.append((max_length, germ))
            self._baseStrToLGerm[basestr] = (max_length, germ)

        return missing_list

    def add_unindexed(self, gs_list, dsfilter=None):
        """
        Adds unstructured operation sequences (not in any plaquette).

        Parameters
        ----------
        gs_list : list of Circuits
            The operation sequences to add.

        dsfilter : DataSet, optional
            If not None, check that this data set contains all of the
            operation sequences being added.  If dscheck does not contain a gate
            sequence, it is *not* added.

        Returns
        -------
        missing : list
            A list of elements in `gs_list` which were not found in `dsfilter`
            and therefore not added.
        """
        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?

        missing_list = []
        for opstr in gs_list:
            if opstr not in self.allstrs_set:
                if dsfilter:
                    trans_opstr = _gstrc.translate_circuit(opstr, self.aliases)
                    if trans_opstr not in dsfilter:
                        missing_list.append(opstr)
                        continue
                self.allstrs_set.add(opstr)
                self.allstrs.append(opstr)
                self.unindexed.append(opstr)
        return missing_list

    def done_adding_strings(self):
        """
        Called to indicate the user is done adding plaquettes.
        """
        #placeholder in case there's some additional init we need to do.
        pass

    def get_plaquette(self, max_length, germ, onlyfirst=True):
        """
        Returns a the plaquette at `(max_length,germ)`.

        Parameters
        ----------
        max_length : int
            The maximum length.

        germ : Circuit
            The germ.

        onlyfirst : bool, optional
            If True, then when multiple plaquettes have been added with the
            same base string, only the *first* added plaquette will be
            returned normally.  Requests for the other plaquettes will be
            given an empty plaquette.  This behavior is useful for color
            box plots where we wish to avoid duplicated data.

        Returns
        -------
        CircuitPlaquette
        """
        if (max_length, germ) not in self._plaquettes:
            p = self.create_plaquette(None, [])  # no elements
            p.simplify_circuits(None)  # just marks as "simplified"
            return p

        if not onlyfirst or (max_length, germ) in self._firsts:
            return self._plaquettes[(max_length, germ)]
        else:
            basestr = self._plaquettes[(max_length, germ)].base
            p = self.create_plaquette(basestr, [])  # no elements
            p.simplify_circuits(None)  # just marks as "simplified"
            return p

    def truncate(self, max_lengths=None, germs=None, n_minor_rows=None, n_minor_cols=None):
        """
        Truncate this operation sequence structure to a subset of its current strings.

        Parameters
        ----------
        max_lengths : list, optional
            The integer L-values to keep.  If None, then all are kept.

        germs : list, optional
            The (Circuit) germs to keep.  If None, then all are kept.

        n_minor_rows, n_minor_cols : int or "auto", optional
            The number of minor rows and columns in the new structure.  If the
            special "auto" value is used, the number or rows/cols is chosen
            automatically (to be as small as possible). If None, then the values
            of the original (this) circuit structure are kept.

        Returns
        -------
        LsGermsSerialStructure
        """
        max_lengths = self.Ls if (max_lengths is None) else max_lengths
        germs = self.germs if (germs is None) else germs
        n_minor_cols = self.nMinorCols if (n_minor_cols is None) else n_minor_cols
        n_minor_rows = self.nMinorRows if (n_minor_rows is None) else n_minor_rows

        if n_minor_cols == "auto" or n_minor_rows == "auto":
            #Pre-compute fidpairs lists per plaquette to get #fidpairs for each
            maxEls = 0
            for (L, germ), plaq in self._plaquettes.items():
                if (L in max_lengths) and (germ in germs):
                    maxEls = max(maxEls, len(plaq.elements))

            if n_minor_cols == "auto" and n_minor_rows == "auto":
                #special behavior: make as square as possible
                n_minor_rows = n_minor_cols = int(_np.floor(_np.sqrt(maxEls)))
                if n_minor_rows * n_minor_cols < maxEls: n_minor_cols += 1
                if n_minor_rows * n_minor_cols < maxEls: n_minor_rows += 1
                assert(n_minor_rows * n_minor_cols >= maxEls), "Logic Error!"
            elif n_minor_cols == "auto":
                n_minor_cols = maxEls // n_minor_rows
                if n_minor_rows * n_minor_cols < maxEls: n_minor_cols += 1
            else:  # n_minor_rows == "auto"
                n_minor_rows = maxEls // n_minor_cols
                if n_minor_rows * n_minor_cols < maxEls: n_minor_rows += 1

        cpy = LsGermsSerialStructure(max_lengths, germs, n_minor_rows, n_minor_cols,
                                     self.aliases, self.sequenceRules)

        for (L, germ), plaq in self._plaquettes.items():
            basestr = plaq.base
            fidpairs = plaq.fidpairs
            if (L in max_lengths) and (germ in germs):
                cpy.add_plaquette(basestr, L, germ, fidpairs)

        cpy.add_unindexed(self.unindexed)  # preserve unindexed strings
        return cpy

    def create_plaquette(self, base_str, fidpairs):
        """
        Creates a the plaquette for the given base string and pairs.

        Parameters
        ----------
        base_str : Circuit

        fidpairs : list
            A list if `(prep,meas)` tuples of Circuit objects, specifying
            the fiducial pairs for this plaquette.  Note that this argument
            is mandatory and cannot be None as for :class:`LsGermsStructure`.

        Returns
        -------
        CircuitPlaquette
        """
        ji_list = list(_itertools.product(list(range(self.nMinorRows)),
                                          list(range(self.nMinorCols))))
        assert(len(ji_list) >= len(fidpairs)), "Number of minor rows/cols is too small!"

        elements = [(j, i, prepStr + base_str + effectStr)
                    for (j, i), (prepStr, effectStr) in
                    zip(ji_list[0:len(fidpairs)], fidpairs)]  # note preps are *cols* not rows

        return CircuitPlaquette(base_str, self.nMinorRows,
                                self.nMinorCols, elements,
                                self.aliases, fidpairs[:])

    def plaquette_rows_cols(self):
        """
        Return the number of rows and columns contained in each plaquette of
        this LsGermsStructure.

        Returns
        -------
        rows, cols : int
        """
        return self.nMinorRows, self.nMinorCols

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`,
        returning a new circuit structure with processed circuits.

        Parameters
        ----------
        processor_fn : function
            A function which takes a single Circuit argument and returns
            another (or the same) Circuit.

        updated_aliases : dict, optional
            Because the Label keys of an alias dictionary (maps
            Label -> Circuit) cannot be processed as a Circuit, one must
            supply a manualy processed alias dictionary.  If you don't use
            alias dictionaries just leave this set to None.

        Returns
        -------
        LsGermsSerialStructure
        """
        P = processor_fn  # shorthand
        cpy = LsGermsSerialStructure(self.Ls, list(map(P, self.germs)),
                                     self.nMinorRows, self.nMinorCols,
                                     updated_aliases, self.sequenceRules)
        cpy.allstrs = list(map(P, self.allstrs))
        cpy.allstrs_set = set(cpy.allstrs)
        cpy.unindexed = list(map(P, self.unindexed))
        cpy._plaquettes = {k: v.process_circuits(P, updated_aliases) for k, v in self._plaquettes.items()}
        cpy._firsts = [(L, P(germ)) for (L, germ) in self._firsts]
        cpy._baseStrToLGerm = {P(base): (L, P(germ)) for base, (L, germ) in self._baseStrToLGerm.items()}
        return cpy

    def copy(self):
        """
        Returns a copy of this `LsGermsSerialStructure`.
        """
        cpy = LsGermsSerialStructure(self.Ls, self.germs, self.nMinorRows,
                                     self.nMinorCols, self.aliases, self.sequenceRules)
        cpy.allstrs = self.allstrs[:]
        cpy.allstrs_set = self.allstrs_set.copy()
        cpy.unindexed = self.unindexed[:]
        cpy._plaquettes = {k: v.copy() for k, v in self._plaquettes.items()}
        cpy._firsts = self._firsts[:]
        cpy._baseStrToLGerm = _copy.deepcopy(self._baseStrToLGerm.copy())
        return cpy
