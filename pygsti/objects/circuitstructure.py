"""
Defines the CircuitStructure class and supporting functionality.
"""
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
import itertools as _itertools
from ..tools import listtools as _lt
from .circuit import Circuit as _Circuit
from .circuitlist import CircuitList as _CircuitList


class CircuitPlaquette(object):
    """
    Encapsulates a single "plaquette" or "sub-matrix" within a circuit plot.

    Parameters
    ----------
    elements : dict
        A dictionary with `(i,j)` keys, where `i` and `j` are row and column
v        indices and :class:`Circuit` values.

    num_rows : int, optional
        The number of rows in this plaquette.  If None, then this is set to one
        larger than the maximum row index in `elements`.

    num_cols : int, optional
        The number of columns in this plaquette. If None, then this is set to one
        larger than the maximum colum index in `elements`.

    op_label_aliases : dict, optional
        A dictionary of operation label aliases that is carried along
        for calls to :func:`expand_aliases`.
    """

    def __init__(self, elements, num_rows=None, num_cols=None, op_label_aliases=None):
        """
        Create a new CircuitPlaquette.
        """
        self.elements = elements.copy()
        self.op_label_aliases = op_label_aliases

        if num_rows is None:
            num_rows = max([i for i, _ in elements]) + 1 if len(elements) > 0 else 0
        if num_cols is None:
            num_cols = max([j for _, j in elements]) + 1 if len(elements) > 0 else 0
        self.num_rows = num_rows
        self.num_cols = num_cols

    @property
    def circuits(self):
        yield from self.elements.values()

    def __iter__(self):
        """
        Iterate over (row_index, col_index, circuit) tuples.
        """
        for (i, j), c in self.elements.items():
            yield i, j, c
        #iterate over non-None entries (i,j,GateStr)

    def __len__(self):
        return len(self.elements)

    def elementvec_to_matrix(self, elementvec, layout, mergeop="sum"):
        """
        Form a matrix of values corresponding to this plaquette from an element vector.

        An element vector holds individual-outcome elements (e.g. the bulk probabilities
        computed by a model).

        Parameters
        ----------
        elementvec : numpy array
            An array containting the values to use when constructing a
            matrix of values for this plaquette.  This array may contain more
            values than are needed by this plaquette.  Indices into this array
            are given by `elindices_lookup`.

        layout : CircuitOutcomeProbabilityArrayLayout
            The layout of `elementvec`, giving the mapping between its elements and
            circuit outcomes.

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
            ret = _np.nan * _np.ones((self.num_rows, self.num_cols), 'd')
            for (i, j), opstr in self.elements.items():
                ret[i, j] = sum(elementvec[layout.indices(opstr)])
        elif '%' in mergeop:
            fmt = mergeop
            ret = _np.nan * _np.ones((self.num_rows, self.num_cols), dtype=_np.object)
            for (i, j), opstr in self.elements.items():
                ret[i, j] = ", ".join(["NaN" if _np.isnan(x) else
                                       (fmt % x) for x in elementvec[layout.indices(opstr)]])
        else:
            raise ValueError("Invalid `mergeop` arg: %s" % str(mergeop))
        return ret

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`.

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
        updated_elements = {(i, j): P(c) for (i, j), c in self.elements.items()}
        return CircuitPlaquette(updated_elements, self.num_rows, self.num_cols, updated_aliases)

    def expand_aliases(self, ds_filter=None):
        """
        Returns a new CircuitPlaquette with any aliases expanded.

        Aliases are expanded (i.e. applied) within the circuits of this
        plaquette.  Optionally keeps only those strings which, after
        alias expansion, are in `ds_filter`.

        Parameters
        ----------
        ds_filter : DataSet, optional
            If not None, keep only strings that are in this data set.

        Returns
        -------
        CircuitPlaquette
        """
        #find & replace aliased operation labels with their expanded form
        new_elements = []
        for k, ((i, j), c) in enumerate(self.elements.items()):
            c2 = c.replace_layers_with_aliases(self.op_label_aliases)
            if ds_filter is None or c2 in ds_filter:
                new_elements.append((i, j, c2))

        return CircuitPlaquette(new_elements, self.num_rows, self.num_cols)

    def truncate(self, circuits_to_keep):
        """
        Remove any circuits from this plaquette that aren't in `circuits_to_keep`.

        Parameters
        ----------
        circuits_to_keep : list
            List of circuits to keep.  If None, then a copy of this object is returned.

        Returns
        -------
        CircuitPlaquette
        """
        if circuits_to_keep is None:
            return plaq.copy()

        elements = {(i, j): c for (i, j), c in self.elements.items() if c in circuits_to_keep}
        return CircuitPlaquette(elements, None, None, self.op_label_aliases)

    def copy(self):
        """
        Returns a copy of this `CircuitPlaquette`.

        Returns
        -------
        CircuitPlaquette
        """
        aliases = _copy.deepcopy(self.op_label_aliases) if (self.op_label_aliases is not None) else None
        return CircuitPlaquette(self.elements, self.num_rows, self.num_cols, aliases)

    def summary_label(self):
        return "%d circuits" % len(self)

    def element_label(self, irow, icol):
        c = self.elements.get((irow, icol), None)
        return f"{c.layerstr}" if (c is not None) else ""


class FiducialPairPlaquette(CircuitPlaquette):
    """
    A plaquette whose rows and columns correspond to measurement and preparation fiducial circuits.

    Theese fiducials sandwich a "base" circuit.

    Parameters
    ----------
    base : Circuit
        The "base" circuit of this plaquette.  Typically the sequence
        that is sandwiched between fiducial pairs.

    fidpairs : list or dict
        A list or dict of `(prepStr, effectStr)` tuples specifying how
        `elements` is generated from `base`, i.e. by `prepStr + base + effectStr`.
        If a dictionary, then `(i, j)` keys give the row and column indices of
        that fiducial pair (in the case of a list, items are placed sequentially by row.

    num_rows : int, optional
        The number of rows in this plaquette.  If None, then this is set to one
        larger than the maximum row index in `elements`.

    num_cols : int, optional
        The number of columns in this plaquette. If None, then this is set to one
        larger than the maximum colum index in `elements`.

    op_label_aliases : dict, optional
        A dictionary of operation label aliases that is carried along
        for calls to :func:`expand_aliases`.
    """

    def __init__(self, base, fidpairs, num_rows=None, num_cols=None, op_label_aliases=None):
        """
        Create a new FiducialPairPlaquette.
        """
        self.base = base
        if not isinstance(fidpairs, dict):
            cols = num_cols if (num_cols is not None) else int(_np.ceil(_np.sqrt(len(fidpairs))))
            fidpairs = {(k % cols, k // cols): fidpair for k, fidpair in enumerate(fidpairs)}
        self.fidpairs = fidpairs.copy()
        super().__init__({(i, j): prep + base + meas for (i, j), (prep, meas) in fidpairs.items()},
                         num_rows, num_cols, op_label_aliases)

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`.

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
        updated_fidpairs = {coords: (P(prep), P(meas)) for coords, (prep, meas) in self.fidpairs.items()}
        return FiducialPairPlaquette(P(self.base), updated_fidpairs, self.num_rows, self.num_cols, updated_aliases)

    def expand_aliases(self, ds_filter=None):
        """
        Returns a new CircuitPlaquette with any aliases expanded.

        Aliases are expanded (i.e. applied) within the circuits of this
        plaquette.  Optionally keeps only those strings which, after
        alias expansion, are in `ds_filter`.

        Parameters
        ----------
        ds_filter : DataSet, optional
            If not None, keep only strings that are in this data set.

        Returns
        -------
        CircuitPlaquette
        """
        #find & replace aliased operation labels with their expanded form
        new_base = self.base.replace_layers_with_aliases(self.op_label_aliases)
        new_fidpairs = {}
        for coords, (prep, meas) in self.fidpairs.items():
            prep2 = prep.replace_layers_with_aliases(self.op_label_aliases)
            meas2 = meas.replace_layers_with_aliases(self.op_label_aliases)
            if ds_filter is None or prep2 + new_base + meas2 in ds_filter:
                new_fidpairs[coords] = (prep2, meas2)
        return FiducialPairPlaquette(new_base, new_fidpairs, self.num_rows, self.num_cols, op_label_aliases=None)

    def truncate(self, circuits_to_keep):
        """
        Remove any circuits from this plaquette that aren't in `circuits_to_keep`.

        Parameters
        ----------
        circuits_to_keep : list
            List of circuits to keep.  If None, then a copy of this object is returned.

        Returns
        -------
        FiducialPairPlaquette
        """
        if circuits_to_keep is None:
            return plaq.copy()

        fidpairs = {}
        for (i, j), c in self.elements.items():
            if c in circuits_to_keep:
                fidpairs[(i, j)] = self.fidpairs[(i, j)]
        return FiducialPairPlaquette(self.base, fidpairs, None, None, self.op_label_aliases)

    def copy(self):
        """
        Returns a copy of this `CircuitPlaquette`.

        Returns
        -------
        FiducialPairPlaquette
        """
        aliases = _copy.deepcopy(self.aliases) if (self.aliases is not None) else None
        return FiducialPairPlaquette(self.base, self.fidpairs, self.num_rows, self.num_cols, aliases)

    def summary_label(self):
        return "{}" if len(self.base) == 0 else f"{self.base.layerstr}"

    def element_label(self, irow, icol):
        prep, meas = self.fidpairs.get((irow, icol), (None, None))
        if prep is None or meas is None:
            return ""
        else:
            return f"{prep.layerstr} + " + self.summary_label() + f" + {meas.layerstr}"


class GermFiducialPairPlaquette(FiducialPairPlaquette):
    """
    A plaquette whose rows and columns correspond to fiducial pairs and whose base is a germ-power.

    Parameters
    ----------
    germ : Circuit
        The "germ" circuit of this plaquette.

    power : int
        The number of times `germ` is repeated to get the base circuit (that
        is sandwiched between different fiducial pairs).

    fidpairs : list or dict
        A list or dict of `(prepStr, effectStr)` tuples specifying how
        `elements` is generated from `base`, i.e. by `prepStr + base + effectStr`.
        If a dictionary, then `(i, j)` keys give the row and column indices of
        that fiducial pair (in the case of a list, items are placed sequentially by row.

    num_rows : int, optional
        The number of rows in this plaquette.  If None, then this is set to one
        larger than the maximum row index in `elements`.

    num_cols : int, optional
        The number of columns in this plaquette. If None, then this is set to one
        larger than the maximum colum index in `elements`.

    op_label_aliases : dict, optional
        A dictionary of operation label aliases that is carried along
        for calls to :func:`expand_aliases`.
    """

    def __init__(self, germ, power, fidpairs, num_rows=None, num_cols=None, op_label_aliases=None):
        """
        Create a new GermFiducialPairPlaquette.
        """
        self.germ = germ
        self.power = power
        super().__init__(germ**power, fidpairs, num_rows, num_cols, op_label_aliases)

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`.

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
        updated_fidpairs = {coords: (P(prep), P(meas)) for coords, (prep, meas) in self.fidpairs.items()}
        return GermFiducialPairPlaquette(P(self.germ), self.power, updated_fidpairs,
                                         self.num_rows, self.num_cols, updated_aliases)

    def expand_aliases(self, ds_filter=None):
        """
        Returns a new CircuitPlaquette with any aliases expanded.

        Aliases are expanded (i.e. applied) within the circuits of this
        plaquette.  Optionally keeps only those strings which, after
        alias expansion, are in `ds_filter`.

        Parameters
        ----------
        ds_filter : DataSet, optional
            If not None, keep only strings that are in this data set.

        Returns
        -------
        CircuitPlaquette
        """
        #find & replace aliased operation labels with their expanded form
        new_germ = self.germ.replace_layers_with_aliases(self.op_label_aliases)
        new_base = new_germ ** self.power
        new_fidpairs = {}
        for coords, (prep, meas) in self.fidpairs.items():
            prep2 = prep.replace_layers_with_aliases(self.op_label_aliases)
            meas2 = meas.replace_layers_with_aliases(self.op_label_aliases)
            if ds_filter is None or prep2 + new_base + meas2 in ds_filter:
                new_fidpairs[coords] = (prep2, meas2)
        return GermFiducialPairPlaquette(new_germ, self.power, new_fidpairs, self.num_rows, self.num_cols,
                                         op_label_aliases=None)

    def truncate(self, circuits_to_keep):
        """
        Remove any circuits from this plaquette that aren't in `circuits_to_keep`.

        Parameters
        ----------
        circuits_to_keep : list
            List of circuits to keep.  If None, then a copy of this object is returned.

        Returns
        -------
        GermFiducialPairPlaquette
        """
        if circuits_to_keep is None:
            return self.copy()

        fidpairs = {}
        for (i, j), c in self.elements.items():
            if c in circuits_to_keep:
                fidpairs[(i, j)] = self.fidpairs[(i, j)]
        return GermFiducialPairPlaquette(self.germ, self.power, fidpairs, None, None, self.op_label_aliases)

    def copy(self):
        """
        Returns a copy of this `CircuitPlaquette`.

        Returns
        -------
        GermFiducialPairPlaquette
        """
        aliases = _copy.deepcopy(self.aliases) if (self.aliases is not None) else None
        return GermFiducialPairPlaquette(self.germ, self.power, self.fidpairs[:], self.num_rows, self.num_cols,
                                         aliases)

    def summary_label(self):
        if len(self.germ) == 0 or self.power == 0:
            return "{}"
        else:
            return f"({self.germ.layerstr})<sup>{self.power}</sup>"


class PlaquetteGridCircuitStructure(_CircuitList):
    """
    Encapsulates a set of circuits, along with an associated structure.

    By "structure", we mean the ability to index the circuits by a
    4-tuple (x, y, minor_x, minor_y) for displaying in nested color box plots,
    along with any aliases.
    """

    @classmethod
    def cast(cls, circuits_or_structure):
        """
        Convert (if needed) an object into a circuit structure.

        Parameters
        ----------
        circuits_or_structure : list or CircuitList
            The object to convert.  If a :class:`PlaquetteGridCircuitStructure`,
            then the object is simply returned.  Lists of circuits (including
            :class:`CircuitList`s are converted to structures having no
            plaquettes.

        Returns
        -------
        PlaquetteGridCircuitStructure
        """
        if isinstance(circuits_or_structure, PlaquetteGridCircuitStructure):
            return circuits_or_structure

        if isinstance(circuits_or_structure, _CircuitList):
            op_label_aliases = circuits_or_structure.op_label_aliases
            weights_dict = {c: wt for c, wt in zip(circuits_or_structure, circuits_or_structure.circuit_weights)}
            name = circuits_or_structure.name
        else:
            op_label_aliases = weights_dict = name = None

        return cls({}, [], [], circuits_or_structure,
                   op_label_aliases, weights_dict, name)

    def __init__(self, plaquettes, x_values, y_values, additional_circuits=None,
                 op_label_aliases=None, circuit_weights_dict=None, name=None):
        # plaquettes is a dict of plaquettes whose keys are tuples of length 2
        self._plaquettes = plaquettes
        self.xs = x_values
        self.ys = y_values
        self.xlabel = "L"    # TODO - allow customization?
        self.ylabel = "germ" # TODO - allow customization?

        circuits = set()
        for plaq in plaquettes.values():
            circuits.update(plaq.circuits)
        additional = set(additional_circuits) - circuits
        circuits = circuits.union(additional)
        self._additional_circuits = tuple(sorted(additional))

        circuits = sorted(circuits)
        circuit_weights = None if (circuit_weights_dict is None) else \
            _np.array([circuit_weights_dict.get(c, 0.0) for c in circuits], 'd')
        super().__init__(circuits, op_label_aliases, circuit_weights, name)

    @property
    def plaquettes(self):
        return self._plaquettes

    def iter_plaquettes(self):
        yield from self._plaquettes.items()

    def plaquette(self, x, y, empty_if_missing=False):
        """
        The plaquette at `(x,y)`.

        Parameters
        ----------
        x : various
            x-value (not index)

        y : various
            y-value (not index)

        empty_if_missing : bool, optional
            Whether an empty (0-element) plaquette
            should be returned when the requested `(x,y)` is
            missing.

        Returns
        -------
        CircuitPlaquette
        """
        if empty_if_missing and (x, y) not in self._plaquettes:
            return CircuitPlaquette([], 0, 0)  # an empty plaquette
        return self._plaquettes[(x, y)]

    @property
    def used_xs(self):
        """
        The x-values which have at least one non-empty plaquette

        Returns
        -------
        list
        """
        return [x for x in self.xs if any([len(self.plaquette(x, y, True)) > 0
                                           for y in self.ys])]

    @property
    def used_ys(self):
        """
        The y-values which have at least one non-empty plaquette

        Returns
        -------
        list
        """
        return [y for y in self.ys if any([len(self.plaquette(x, y, True)) > 0
                                           for x in self.xs])]

    def truncate(self, xs_to_keep=None, ys_to_keep=None, circuits_to_keep=None):
        """
        Truncate this circuit structure to a subset of its current strings.

        Parameters
        ----------
        xs_to_keep : list, optional
            The x-values to keep.  If None, then all are kept.

        ys_to_keep : list, optional
            The y-values to keep.  If None, then all are kept.

        circuits_to_keep : list
            Keep only the circuits present in this list (of Circuit objects).

        Returns
        -------
        PlaquetteGridCircuitStructure
        """
        xs = self.xs if (xs_to_keep is None) else xs_to_keep
        ys = self.ys if (ys_to_keep is None) else ys_to_keep

        plaquettes = {}
        for (x, y), plaq in self._plaquettes.items():
            if not ((x in xs) and (y in ys)): continue
            plaquettes[(x, y)] = plaq.truncate(circuits_to_keep)

        circuit_weights_dict = {c: weight for c, weight in zip(self, self.circuit_weights)} \
            if (self.circuit_weights is not None) else None
        additional = list(filter(lambda c: c in circuits_to_keep, self._additional_circuits)) \
            if (circuits_to_keep is not None) else self._additional_circuits
        return PlaquetteGridCircuitStructure(plaquettes, xs, ys, additional, self.op_label_aliases,
                                             circuit_weights_dict, self.name)

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Manipulate this object's circuits according to `processor_fn`.

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
        PlaquetteGridCircuitStructure
        """
        P = processor_fn  # shorhand

        plaquettes = {k: v.process_circuits(P, updated_aliases) for k, v in self._plaquettes.items()}
        if len(self.xs) > 0 and isinstance(self.xs[0], _Circuit):
            xs = list(map(P, self.xs))
            plaquettes = {(P(x), y): v for (x, y), v in plaquettes.items()}
        if len(self.ys) > 0 and isinstance(self.ys[0], _Circuit):
            ys = list(map(P, self.ys))
            plaquettes = {(x, P(y)): v for (x, y), v in plaquettes.items()}
        additional = list(map(P, self._additional_circuits))

        circuit_weights_dict = {P(c): weight for c, weight in zip(self, self.circuit_weights)} \
            if (self.circuit_weights is not None) else None
        return PlaquetteGridCircuitStructure(plaquettes, xs, ys, additional,
                                             updated_aliases, circuit_weights_dict, self.name)

    def copy(self):
        """
        Returns a copy of this circuit structure.

        Returns
        -------
        PlaquetteGridCircuitStructure
        """
        circuit_weights_dict = {c: weight for c, weight in zip(self, self.circuit_weights)} \
            if (self.circuit_weights is not None) else None
        return PlaquetteGridCircuitStructure(self._plaquettes.copy(), self.xs[:], self.ys[:],
                                             self._additional_circuits, self.op_label_aliases,
                                             circuit_weights_dict, self.name)

    #UNUSED?
    #@property
    #def basestrings(self):
    #    """
    #    Lists the base strings (without duplicates) of all the plaquettes
    #
    #    Returns
    #    -------
    #    list
    #    """
    #    baseStrs = set()
    #    for x in self.xs:
    #        for y in self.ys:
    #            p = self.plaquette(x, y)
    #            if p is not None and p.base is not None:
    #                baseStrs.add(p.base)
    #    return list(baseStrs)


#REMOVE
#class LsGermsStructure(CircuitStructure):
#    """
#    A circuit structure where circuits are indexed by L, germ, preparation-fiducial, and measurement-fiducial.
#
#    Parameters
#    ----------
#    max_lengths : list of ints
#        List of maximum lengths (x values)
#
#    germs : list of Circuits
#        List of germ sequences (y values)
#
#    prep_fiducials : list of Circuits
#        List of preparation fiducial sequences (minor x values)
#
#    effecStrs : list of Circuits
#        List of measurement fiducial sequences (minor y values)
#
#    aliases : dict
#        Operation label aliases to be propagated to all plaquettes.
#
#    sequence_rules : list, optional
#        A list of `(find,replace)` 2-tuples which specify string replacement
#        rules.  Both `find` and `replace` are tuples of operation labels
#        (or `Circuit` objects).
#    """
#
#    def __init__(self, max_lengths, germs, prep_fiducials, meas_fiducials, aliases=None,
#                 sequence_rules=None):
#        """
#        Create an empty circuit structure.
#
#        Parameters
#        ----------
#        max_lengths : list of ints
#            List of maximum lengths (x values)
#
#        germs : list of Circuits
#            List of germ sequences (y values)
#
#        prep_fiducials : list of Circuits
#            List of preparation fiducial sequences (minor x values)
#
#        effecStrs : list of Circuits
#            List of measurement fiducial sequences (minor y values)
#
#        aliases : dict
#            Operation label aliases to be propagated to all plaquettes.
#
#        sequence_rules : list, optional
#            A list of `(find,replace)` 2-tuples which specify string replacement
#            rules.  Both `find` and `replace` are tuples of operation labels
#            (or `Circuit` objects).
#        """
#        self._Ls = max_lengths[:]
#        self._germs = germs[:]
#        self._prep_fiducials = prep_fiducials[:]
#        self._meas_fiducials = meas_fiducials[:]
#        #self.aliases = aliases.copy() if (aliases is not None) else None
#        #self.sequenceRules = sequence_rules[:] if (sequence_rules is not None) else None
#
#        #self.allstrs = []
#        #self.allstrs_set = set()
#        self._unindexed = []  # unindexed strings
#        self._plaquettes = {} # by base string?
#        self._firsts = []
#        self._baseStrToLGerm = {}
#        super(LsGermsStructure, self).__init__()
#
#    @property
#    def xs(self):
#        return self._Ls
#
#    @property
#    def ys(self):
#        return self._germs
#
#    @property
#    def minor_xs(self):
#        return self._prep_fiducials
#
#    @property
#    def minor_ys(self):
#        return self._meas_fiducials
#
#    def add_plaquette(self, basestr, max_length, germ, fidpairs=None, dsfilter=None):
#        """
#        Adds a plaquette with the given fiducial pairs at the `(max_length,germ)` location.
#
#        Parameters
#        ----------
#        basestr : Circuit
#            The base circuit of the new plaquette.
#
#        max_length : int
#            The maximum length (x) coordinate of the new plaquette.
#
#        germ : Circuit
#            The germ (y) coordinate of the new plaquette.
#
#        fidpairs : list
#            A list if `(i,j)` tuples of integers, where `i` is a prepation
#            fiducial index and `j` is a measurement fiducial index.  None
#            can be used to mean all pairs.
#
#        dsfilter : DataSet, optional
#            If not None, check that this data set contains all of the
#            circuits being added.  If dscheck does not contain a gate
#            sequence, it is *not* added.
#
#        Returns
#        -------
#        missing : list
#            A list of `(prep_fiducial, germ, max_length, effect_fiducial, entire_string)`
#            tuples indicating which sequences were not found in `dsfilter`.
#        """
#        missing_list = []
#        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?
#
#        if fidpairs is None:
#            fidpairs = list(_itertools.product(range(len(self.prep_fiducials)),
#                                               range(len(self.meas_fiducials))))
#        if dsfilter:
#            inds_to_remove = []
#            for k, (i, j) in enumerate(fidpairs):
#                el = self.prep_fiducials[i] + basestr + self.meas_fiducials[j]
#                trans_el = _gstrc.translate_circuit(el, self.aliases)
#                if trans_el not in dsfilter:
#                    missing_list.append((self.prep_fiducials[i], germ, max_length, self.meas_fiducials[j], el))
#                    inds_to_remove.append(k)
#
#            if len(inds_to_remove) > 0:
#                fidpairs = fidpairs[:]  # copy
#                for i in reversed(inds_to_remove):
#                    del fidpairs[i]
#
#        plaq = self.create_plaquette(basestr, fidpairs)
#
#        for x in (_gstrc.manipulate_circuit(opstr, self.sequenceRules) for i, j, opstr in plaq):
#            if x not in self.allstrs_set:
#                self.allstrs_set.add(x)
#                self.allstrs.append(x)
#        #_lt.remove_duplicates_in_place(self.allstrs) # above block does this more efficiently
#
#        self._plaquettes[(max_length, germ)] = plaq
#
#        #keep track of which max_length,germ is the *first* one to "claim" a base string
#        # (useful for *not* duplicating data in color box plots)
#        if basestr not in self._baseStrToLGerm:
#            self._firsts.append((max_length, germ))
#            self._baseStrToLGerm[basestr] = (max_length, germ)
#
#        return missing_list
#
#    def add_unindexed(self, gs_list, dsfilter=None):
#        """
#        Adds unstructured circuits (not in any plaquette).
#
#        Parameters
#        ----------
#        gs_list : list of Circuits
#            The circuits to add.
#
#        dsfilter : DataSet, optional
#            If not None, check that this data set contains all of the
#            circuits being added.  If dscheck does not contain a gate
#            sequence, it is *not* added.
#
#        Returns
#        -------
#        missing : list
#            A list of elements in `gs_list` which were not found in `dsfilter`
#            and therefore not added.
#        """
#        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?
#        #if dsfilter and len(dsfilter) > 8000: dsfilter = None # TEST DEBUG - remove dsfilter check
#
#        missing_list = []
#        for opstr in gs_list:
#            if opstr not in self.allstrs_set:
#                if dsfilter:
#                    trans_opstr = _gstrc.translate_circuit(opstr, self.aliases)
#                    if trans_opstr not in dsfilter:
#                        missing_list.append(opstr)
#                        continue
#                self.allstrs_set.add(opstr)
#                self.allstrs.append(opstr)
#                self.unindexed.append(opstr)
#        return missing_list
#
#    def done_adding_strings(self):
#        """
#        Called to indicate the user is done adding plaquettes.
#
#        Returns
#        -------
#        None
#        """
#        #placeholder in case there's some additional init we need to do.
#        pass
#
#    def get_plaquette(self, max_length, germ, onlyfirst=True):
#        """
#        Returns a the plaquette at `(max_length,germ)`.
#
#        Parameters
#        ----------
#        max_length : int
#            The maximum length.
#
#        germ : Circuit
#            The germ.
#
#        onlyfirst : bool, optional
#            If True, then when multiple plaquettes have been added with the
#            same base string, only the *first* added plaquette will be
#            returned normally.  Requests for the other plaquettes will be
#            given an empty plaquette.  This behavior is useful for color
#            box plots where we wish to avoid duplicated data.
#
#        Returns
#        -------
#        CircuitPlaquette
#        """
#        if (max_length, germ) not in self._plaquettes:
#            return self.create_plaquette(None, [])  # no elements
#
#        if not onlyfirst or (max_length, germ) in self._firsts:
#            return self._plaquettes[(max_length, germ)]
#        else:
#            basestr = self._plaquettes[(max_length, germ)].base
#            return self.create_plaquette(basestr, [])  # no elements
#
#    def truncate(self, max_lengths=None, germs=None, prep_fiducials=None, meas_fiducials=None, seqs=None):
#        """
#        Truncate this circuit structure to a subset of its current strings.
#
#        Parameters
#        ----------
#        max_lengths : list, optional
#            The integer L-values to keep.  If None, then all are kept.
#
#        germs : list, optional
#            The (Circuit) germs to keep.  If None, then all are kept.
#
#        prep_fiducials : list, optional
#            The preparation fiducial circuits.
#
#        meas_fiducials : list, optional
#            The measurement fiducial circuits.
#
#        seqs : list
#            Keep only sequences present in this list of Circuit objects.
#
#        Returns
#        -------
#        LsGermsStructure
#        """
#        max_lengths = self.Ls if (max_lengths is None) else max_lengths
#        germs = self.germs if (germs is None) else germs
#        prep_fiducials = self.prep_fiducials if (prep_fiducials is None) else prep_fiducials
#        meas_fiducials = self.meas_fiducials if (meas_fiducials is None) else meas_fiducials
#        cpy = LsGermsStructure(max_lengths, germs, prep_fiducials,
#                               meas_fiducials, self.aliases, self.sequenceRules)
#
#        #OLD iPreps = [i for i, prepStr in enumerate(self.prep_fiducials) if prepStr in prep_fiducials]
#        #OLD iEffects = [i for i, eStr in enumerate(self.meas_fiducials) if eStr in meas_fiducials]
#        #OLD fidpairs = list(_itertools.product(iPreps, iEffects))
#        all_fidpairs = list(_itertools.product(list(range(len(prep_fiducials))), list(range(len(meas_fiducials)))))
#
#        for (L, germ), plaq in self._plaquettes.items():
#            basestr = plaq.base
#            if seqs is None:
#                fidpairs = all_fidpairs
#            else:
#                fidpairs = []
#                for i, j in all_fidpairs:
#                    if prep_fiducials[i] + basestr + meas_fiducials[j] in seqs:
#                        fidpairs.append((i, j))
#
#            if (L in max_lengths) and (germ in germs):
#                cpy.add_plaquette(basestr, L, germ, fidpairs)
#
#        cpy.add_unindexed(self.unindexed)  # preserve unindexed strings
#        return cpy
#
#    def _create_plaquette(self, base_circuit, fidpairs=None):
#        """
#        Creates a the plaquette for the given base string and pairs.
#
#        Parameters
#        ----------
#        base_circuit : Circuit
#            The base circuit to use.
#
#        fidpairs : list
#            A list if `(i,j)` tuples of integers, where `i` is a prepation
#            fiducial index and `j` is a measurement fiducial index.  If
#            None, then all pairs are included (a "full" plaquette is created).
#
#        Returns
#        -------
#        CircuitPlaquette
#        """
#        if fidpairs is None:
#            fidpairs = list(_itertools.product(range(len(self.prep_fiducials)),
#                                               range(len(self.meas_fiducials))))
#
#        elements = [(j, i, self.prep_fiducials[i] + base_circuit + self.meas_fiducials[j])
#                    for i, j in fidpairs]  # note preps are *cols* not rows
#        real_fidpairs = [(self.prep_fiducials[i], self.meas_fiducials[j]) for i, j in fidpairs] # circuits, not indices
#
#        return CircuitPlaquette(base_circuit, len(self.meas_fiducials),
#                                len(self.prep_fiducials), elements,
#                                self.aliases, real_fidpairs)
#
#    def num_plaquette_rows_cols(self):
#        """
#        Return the number of rows and columns contained in each plaquette of this circuit structure.
#
#        Returns
#        -------
#        rows, cols : int
#        """
#        return len(self.meas_fiducials), len(self.prep_fiducials)
#
#    def process_circuits(self, processor_fn, updated_aliases=None):
#        """
#        Manipulate this object's circuits according to `processor_fn`.
#
#        Parameters
#        ----------
#        processor_fn : function
#            A function which takes a single Circuit argument and returns
#            another (or the same) Circuit.
#
#        updated_aliases : dict, optional
#            Because the Label keys of an alias dictionary (maps
#            Label -> Circuit) cannot be processed as a Circuit, one must
#            supply a manualy processed alias dictionary.  If you don't use
#            alias dictionaries just leave this set to None.
#
#        Returns
#        -------
#        LsGermsStructure
#        """
#        P = processor_fn  # shorhand
#        cpy = LsGermsStructure(self.Ls, list(map(P, self.germs)),
#                               list(map(P, self.prep_fiducials)), list(map(P, self.meas_fiducials)),
#                               updated_aliases, self.sequenceRules)
#        cpy.allstrs = list(map(P, self.allstrs))
#        cpy.allstrs_set = set(cpy.allstrs)
#        cpy.unindexed = list(map(P, self.unindexed))
#        cpy._plaquettes = {k: v.process_circuits(P, updated_aliases) for k, v in self._plaquettes.items()}
#        cpy._firsts = [(L, P(germ)) for (L, germ) in self._firsts]
#        cpy._baseStrToLGerm = {P(base): (L, P(germ)) for base, (L, germ) in self._baseStrToLGerm.items()}
#        return cpy
#
#    def copy(self):
#        """
#        Returns a copy of this `LsGermsStructure`.
#
#        Returns
#        -------
#        LsGermsStructure
#        """
#        cpy = LsGermsStructure(self.Ls, self.germs, self.prep_fiducials,
#                               self.meas_fiducials, self.aliases, self.sequenceRules)
#        cpy.allstrs = self.allstrs[:]
#        cpy.allstrs_set = self.allstrs_set.copy()
#        cpy.unindexed = self.unindexed[:]
#        cpy._plaquettes = {k: v.copy() for k, v in self._plaquettes.items()}
#        cpy._firsts = self._firsts[:]
#        cpy._baseStrToLGerm = _copy.deepcopy(self._baseStrToLGerm.copy())
#        return cpy
#
#
#class LsGermsSerialStructure(CircuitStructure):
#    """
#    A circuit structure where circuits are indexed by L, germ, preparation-fiducial, and measurement-fiducial.
#
#    Parameters
#    ----------
#    max_lengths : list of ints
#        List of maximum lengths (x values)
#
#    germs : list of Circuits
#        List of germ sequences (y values)
#
#    n_minor_rows : int
#        The number of minor rows to allocate space for.
#        These should be the maximum values required for any plaquette.
#
#    n_minor_cols : int
#        The number of minor columns to allocate space for.
#        These should be the maximum values required for any plaquette.
#
#    aliases : dict
#        Operation label aliases to be propagated to all plaquettes.
#
#    sequence_rules : list, optional
#        A list of `(find,replace)` 2-tuples which specify string replacement
#        rules.  Both `find` and `replace` are tuples of operation labels
#        (or `Circuit` objects).
#    """
#
#    @classmethod
#    def from_list(cls, circuit_list, dsfilter=None):
#        """
#        Creates a LsGermsSerialStructure out of a simple circuit list.
#
#        This factory method is used when a default structure is required for a
#        given simple list of circuits.
#
#        Parameters
#        ----------
#        circuit_list : list
#            A list of :class:`Circuit` objects.
#
#        dsfilter : DataSet, optional
#            A data set which filters the elements of `circuit_list`, so that only
#            those circuits contained in `dsfilter` are included in the returned
#            circuit structure.
#
#        Returns
#        -------
#        LsGermsSerialStructure
#        """
#        max_length = 0  # just a single "0" length
#        empty_circuit = _Circuit((), line_labels=circuit_list[0].line_labels if len(circuit_list) > 0 else "auto")
#        square_side = int(_np.ceil(_np.sqrt(len(circuit_list))))
#        ret = cls([max_length], [empty_circuit], square_side, square_side)
#
#        fidpairs = [(c, empty_circuit) for c in circuit_list]
#        ret.add_plaquette(empty_circuit, max_length, empty_circuit, fidpairs, dsfilter)
#        return ret
#
#    def __init__(self, max_lengths, germs, n_minor_rows, n_minor_cols, aliases=None,
#                 sequence_rules=None):
#        """
#        Create an empty LsGermsSerialStructure.
#
#        This type of circuit structure is useful for holding multi-qubit
#        circuits which have a germ and max-length structure but which have
#        widely varying fiducial sequences so that is it not useful to use the
#        minor axes (rows/columns) to represent the *same* fiducials for all
#        (L,germ) plaquettes.
#
#        Parameters
#        ----------
#        max_lengths : list of ints
#            List of maximum lengths (x values)
#
#        germs : list of Circuits
#            List of germ sequences (y values)
#
#        n_minor_rows, n_minor_cols : int
#            The number of minor rows and columns to allocate space for.
#            These should be the maximum values required for any plaquette.
#
#        aliases : dict
#            Operation label aliases to be propagated to all plaquettes.
#
#        sequence_rules : list, optional
#            A list of `(find,replace)` 2-tuples which specify string replacement
#            rules.  Both `find` and `replace` are tuples of operation labels
#            (or `Circuit` objects).
#        """
#        self.Ls = max_lengths[:]
#        self.germs = germs[:]
#        self.nMinorRows = n_minor_rows
#        self.nMinorCols = n_minor_cols
#        self.aliases = aliases.copy() if (aliases is not None) else None
#        self.sequenceRules = sequence_rules[:] if (sequence_rules is not None) else None
#
#        self.allstrs = []
#        self.allstrs_set = set()
#        self.unindexed = []
#        self._plaquettes = {}
#        self._firsts = []
#        self._baseStrToLGerm = {}
#        super(LsGermsSerialStructure, self).__init__()
#
#    #Base class access in terms of generic x,y coordinates
#    def xvals(self):
#        """
#        Returns a list of the x-values
#
#        Returns
#        -------
#        list
#        """
#        return self.Ls
#
#    def yvals(self):
#        """
#        Returns a list of the y-values
#
#        Returns
#        -------
#        list
#        """
#        return self.germs
#
#    def minor_xvals(self):
#        """
#        Returns a list of the minor x-values (0-based integers)
#
#        Returns
#        -------
#        list
#        """
#        return list(range(self.nMinorCols))
#
#    def minor_yvals(self):
#        """
#        Returns a list of the minor y-values (0-based integers)
#
#        Returns
#        -------
#        list
#        """
#        return list(range(self.nMinorRows))
#
#    def add_plaquette(self, basestr, max_length, germ, fidpairs, dsfilter=None):
#        """
#        Adds a plaquette with the given fiducial pairs at the `(max_length,germ)` location.
#
#        Parameters
#        ----------
#        basestr : Circuit
#            The base circuit of the new plaquette, typically `germ^power`
#            such that `len(germ^power) <= max_length`.
#
#        max_length : int
#            The maximum length value.
#
#        germ : Circuit
#            The germ string.
#
#        fidpairs : list
#            A list if `(prep,meas)` tuples of Circuit objects, specifying
#            the fiducial pairs for this plaquette.  Note that this argument
#            is different from the corresponding one in
#            :method:`LsGermsStructure.add_plaquette` which takes pairs of
#            *integer* indices and can be None.  In the present case, this
#            argument is mandatory and contains tuples of circuits.
#
#        dsfilter : DataSet, optional
#            If not None, check that this data set contains all of the
#            circuits being added.  If dscheck does not contain a gate
#            sequence, it is *not* added.
#
#        Returns
#        -------
#        missing : list
#            A list of `(prep_fiducial, germ, max_length, effect_fiducial, entire_string)`
#            tuples indicating which sequences were not found in `dsfilter`.
#        """
#
#        missing_list = []
#        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?
#
#        if dsfilter:  # and len(dsfilter) < 8000: # TEST DEBUG
#            inds_to_remove = []
#            for k, (prepStr, effectStr) in enumerate(fidpairs):
#                el = prepStr + basestr + effectStr
#                trans_el = _gstrc.translate_circuit(el, self.aliases)
#                if trans_el not in dsfilter:
#                    missing_list.append((prepStr, germ, max_length, effectStr, el))
#                    inds_to_remove.append(k)
#
#            if len(inds_to_remove) > 0:
#                fidpairs = fidpairs[:]  # copy
#                for i in reversed(inds_to_remove):
#                    del fidpairs[i]
#
#        plaq = self.create_plaquette(basestr, fidpairs)
#
#        for x in (_gstrc.manipulate_circuit(opstr, self.sequenceRules) for i, j, opstr in plaq):
#            if x not in self.allstrs_set:
#                self.allstrs_set.add(x)
#                self.allstrs.append(x)
#        # _lt.remove_duplicates_in_place(self.allstrs) # above block does this more efficiently
#
#        self._plaquettes[(max_length, germ)] = plaq
#
#        #keep track of which max_length,germ is the *first* one to "claim" a base string
#        # (useful for *not* duplicating data in color box plots)
#        if basestr not in self._baseStrToLGerm:
#            self._firsts.append((max_length, germ))
#            self._baseStrToLGerm[basestr] = (max_length, germ)
#
#        return missing_list
#
#    def add_unindexed(self, gs_list, dsfilter=None):
#        """
#        Adds unstructured circuits (not in any plaquette).
#
#        Parameters
#        ----------
#        gs_list : list of Circuits
#            The circuits to add.
#
#        dsfilter : DataSet, optional
#            If not None, check that this data set contains all of the
#            circuits being added.  If dscheck does not contain a gate
#            sequence, it is *not* added.
#
#        Returns
#        -------
#        missing : list
#            A list of elements in `gs_list` which were not found in `dsfilter`
#            and therefore not added.
#        """
#        from ..construction import circuitconstruction as _gstrc  # maybe move used routines to a circuittools.py?
#
#        missing_list = []
#        for opstr in gs_list:
#            if opstr not in self.allstrs_set:
#                if dsfilter:
#                    trans_opstr = _gstrc.translate_circuit(opstr, self.aliases)
#                    if trans_opstr not in dsfilter:
#                        missing_list.append(opstr)
#                        continue
#                self.allstrs_set.add(opstr)
#                self.allstrs.append(opstr)
#                self.unindexed.append(opstr)
#        return missing_list
#
#    def done_adding_strings(self):
#        """
#        Called to indicate the user is done adding plaquettes.
#
#        Returns
#        -------
#        None
#        """
#        #placeholder in case there's some additional init we need to do.
#        pass
#
#    def get_plaquette(self, max_length, germ, onlyfirst=True):
#        """
#        Returns a the plaquette at `(max_length,germ)`.
#
#        Parameters
#        ----------
#        max_length : int
#            The maximum length.
#
#        germ : Circuit
#            The germ.
#
#        onlyfirst : bool, optional
#            If True, then when multiple plaquettes have been added with the
#            same base string, only the *first* added plaquette will be
#            returned normally.  Requests for the other plaquettes will be
#            given an empty plaquette.  This behavior is useful for color
#            box plots where we wish to avoid duplicated data.
#
#        Returns
#        -------
#        CircuitPlaquette
#        """
#        if (max_length, germ) not in self._plaquettes:
#            return self.create_plaquette(None, [])  # no elements
#
#        if not onlyfirst or (max_length, germ) in self._firsts:
#            return self._plaquettes[(max_length, germ)]
#        else:
#            basestr = self._plaquettes[(max_length, germ)].base
#            return self.create_plaquette(basestr, [])  # no elements
#
#    def truncate(self, max_lengths=None, germs=None, n_minor_rows=None, n_minor_cols=None):
#        """
#        Truncate this circuit structure to a subset of its current circuits.
#
#        Parameters
#        ----------
#        max_lengths : list, optional
#            The integer L-values to keep.  If None, then all are kept.
#
#        germs : list, optional
#            The (Circuit) germs to keep.  If None, then all are kept.
#
#        n_minor_rows : int or "auto", optional
#            The number of plaquette rows in the truncated circuit structure.
#            If "auto" then this is computed automatically.
#
#        n_minor_cols : int or "auto", optional
#            The number of plaquette columns in the truncated circuit structure.
#            If "auto" then this is computed automatically.
#
#        Returns
#        -------
#        LsGermsSerialStructure
#        """
#        max_lengths = self.Ls if (max_lengths is None) else max_lengths
#        germs = self.germs if (germs is None) else germs
#        n_minor_cols = self.nMinorCols if (n_minor_cols is None) else n_minor_cols
#        n_minor_rows = self.nMinorRows if (n_minor_rows is None) else n_minor_rows
#
#        if n_minor_cols == "auto" or n_minor_rows == "auto":
#            #Pre-compute fidpairs lists per plaquette to get #fidpairs for each
#            maxEls = 0
#            for (L, germ), plaq in self._plaquettes.items():
#                if (L in max_lengths) and (germ in germs):
#                    maxEls = max(maxEls, len(plaq.elements))
#
#            if n_minor_cols == "auto" and n_minor_rows == "auto":
#                #special behavior: make as square as possible
#                n_minor_rows = n_minor_cols = int(_np.floor(_np.sqrt(maxEls)))
#                if n_minor_rows * n_minor_cols < maxEls: n_minor_cols += 1
#                if n_minor_rows * n_minor_cols < maxEls: n_minor_rows += 1
#                assert(n_minor_rows * n_minor_cols >= maxEls), "Logic Error!"
#            elif n_minor_cols == "auto":
#                n_minor_cols = maxEls // n_minor_rows
#                if n_minor_rows * n_minor_cols < maxEls: n_minor_cols += 1
#            else:  # n_minor_rows == "auto"
#                n_minor_rows = maxEls // n_minor_cols
#                if n_minor_rows * n_minor_cols < maxEls: n_minor_rows += 1
#
#        cpy = LsGermsSerialStructure(max_lengths, germs, n_minor_rows, n_minor_cols,
#                                     self.aliases, self.sequenceRules)
#
#        for (L, germ), plaq in self._plaquettes.items():
#            basestr = plaq.base
#            fidpairs = plaq.fidpairs
#            if (L in max_lengths) and (germ in germs):
#                cpy.add_plaquette(basestr, L, germ, fidpairs)
#
#        cpy.add_unindexed(self.unindexed)  # preserve unindexed strings
#        return cpy
#
#    def _create_plaquette(self, base_circuit, fidpairs):
#        """
#        Creates a the plaquette for the given base string and pairs.
#
#        Parameters
#        ----------
#        base_circuit : Circuit
#            The base circuit to use.
#
#        fidpairs : list
#            A list if `(prep,meas)` tuples of Circuit objects, specifying
#            the fiducial pairs for this plaquette.  Note that this argument
#            is mandatory and cannot be None as for :class:`LsGermsStructure`.
#
#        Returns
#        -------
#        CircuitPlaquette
#        """
#        ji_list = list(_itertools.product(list(range(self.nMinorRows)),
#                                          list(range(self.nMinorCols))))
#        assert(len(ji_list) >= len(fidpairs)), "Number of minor rows/cols is too small!"
#
#        elements = [(j, i, prepStr + base_circuit + effectStr)
#                    for (j, i), (prepStr, effectStr) in
#                    zip(ji_list[0:len(fidpairs)], fidpairs)]  # note preps are *cols* not rows
#
#        return CircuitPlaquette(base_circuit, self.nMinorRows,
#                                self.nMinorCols, elements,
#                                self.aliases, fidpairs[:])
#
#    def num_plaquette_rows_cols(self):
#        """
#        Return the number of rows and columns contained in each plaquette of this LsGermsStructure.
#
#        Returns
#        -------
#        rows, cols : int
#        """
#        return self.nMinorRows, self.nMinorCols
#
#    def process_circuits(self, processor_fn, updated_aliases=None):
#        """
#        Manipulate this object's circuits according to `processor_fn`.
#
#        Parameters
#        ----------
#        processor_fn : function
#            A function which takes a single Circuit argument and returns
#            another (or the same) Circuit.
#
#        updated_aliases : dict, optional
#            Because the Label keys of an alias dictionary (maps
#            Label -> Circuit) cannot be processed as a Circuit, one must
#            supply a manualy processed alias dictionary.  If you don't use
#            alias dictionaries just leave this set to None.
#
#        Returns
#        -------
#        LsGermsSerialStructure
#        """
#        P = processor_fn  # shorthand
#        cpy = LsGermsSerialStructure(self.Ls, list(map(P, self.germs)),
#                                     self.nMinorRows, self.nMinorCols,
#                                     updated_aliases, self.sequenceRules)
#        cpy.allstrs = list(map(P, self.allstrs))
#        cpy.allstrs_set = set(cpy.allstrs)
#        cpy.unindexed = list(map(P, self.unindexed))
#        cpy._plaquettes = {k: v.process_circuits(P, updated_aliases) for k, v in self._plaquettes.items()}
#        cpy._firsts = [(L, P(germ)) for (L, germ) in self._firsts]
#        cpy._baseStrToLGerm = {P(base): (L, P(germ)) for base, (L, germ) in self._baseStrToLGerm.items()}
#        return cpy
#
#    def copy(self):
#        """
#        Returns a copy of this `LsGermsSerialStructure`.
#
#        Returns
#        -------
#        LsGermsSerialStructure
#        """
#        cpy = LsGermsSerialStructure(self.Ls, self.germs, self.nMinorRows,
#                                     self.nMinorCols, self.aliases, self.sequenceRules)
#        cpy.allstrs = self.allstrs[:]
#        cpy.allstrs_set = self.allstrs_set.copy()
#        cpy.unindexed = self.unindexed[:]
#        cpy._plaquettes = {k: v.copy() for k, v in self._plaquettes.items()}
#        cpy._firsts = self._firsts[:]
#        cpy._baseStrToLGerm = _copy.deepcopy(self._baseStrToLGerm.copy())
#        return cpy
