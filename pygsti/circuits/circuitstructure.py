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

import collections as _collections
import copy as _copy

import numpy as _np

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.circuits.circuitconstruction import manipulate_circuit as _manipulate_circuit


class CircuitPlaquette(_NicelySerializable):
    """
    Encapsulates a single "plaquette" or "sub-matrix" within a circuit plot.

    Parameters
    ----------
    elements : dict
        A dictionary with `(i,j)` keys, where `i` and `j` are row and column
        indices and :class:`Circuit` values.

    num_rows : int, optional
        The number of rows in this plaquette.  If None, then this is set to one
        larger than the maximum row index in `elements`.

    num_cols : int, optional
        The number of columns in this plaquette. If None, then this is set to one
        larger than the maximum colum index in `elements`.

    op_label_aliases : dict, optional
        A dictionary of operation label aliases that is carried along
        for calls to :func:`expand_aliases`.

    circuit_rules : list, optional
        A list of `(find,replace)` 2-tuples which specify string replacement
        rules.  Both `find` and `replace` are tuples of operation labels
        (or `Circuit` objects).
    """

    def __init__(self, elements, num_rows=None, num_cols=None, op_label_aliases=None, circuit_rules=None):
        """
        Create a new CircuitPlaquette.
        """
        super().__init__()
        self.elements = _collections.OrderedDict(elements)
        self.circuit_rules = circuit_rules
        self.op_label_aliases = op_label_aliases

        if self.circuit_rules is not None:
            self.elements = _collections.OrderedDict([(k, _manipulate_circuit(c, self.circuit_rules))
                                                      for k, c in self.elements.items()])

        if num_rows is None:
            num_rows = max([i for i, _ in elements]) + 1 if len(elements) > 0 else 0
        if num_cols is None:
            num_cols = max([j for _, j in elements]) + 1 if len(elements) > 0 else 0
        self.num_rows = num_rows
        self.num_cols = num_cols

    def _to_nice_serialization(self):  # memo holds already serialized objects
        state = super()._to_nice_serialization()
        state.update({'num_rows': self.num_rows,
                      'num_cols': self.num_cols,
                      # NOTE: op_label_aliases and circuit_rules are taken care of by parent structure
                      })
        if self.__class__ == CircuitPlaquette:  # class comparison (use 'is'?)
            # then this call is acting as the top-level serialization, not a base-class call, so serialize elements:
            state['elements'] = [(ij_key, c.str) for ij_key, c in self.elements.items()]
            # Note: tuples cannot be keys in JSON

        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        # Note: it's ok that this method assumes the presence of 'elements', since it should only
        # be called to de-serialize a serialized actual CircuitPlaquette object and not a serialized
        # derived class object.
        from pygsti.io import stdinput as _stdinput
        from pygsti.io.readers import convert_strings_to_circuits as _convert_strings_to_circuits
        std = _stdinput.StdInputParser()
        elements = {tuple(ij): std.parse_circuit(s, create_subcircuits=not _Circuit.default_expand_subcircuits)
                    for ij, s in state['elements']}

        return cls(elements, state['num_rows'], state['num_cols'], None, None)
        # Note: parent structure pipes in op_label_aliases & circuit_rules here, so we don't serialize it

    def _post_from_nice_serialization_init(self, op_label_aliases, circuit_rules):
        #Note: in case of a CircuitPlaquette, circuit_rules have already been applied to serialized elements,
        # so *don't* apply them to self.elements here
        self.circuit_rules = circuit_rules
        self.op_label_aliases = op_label_aliases

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
            ret = _np.nan * _np.ones((self.num_rows, self.num_cols), dtype=_np.object_)
            for (i, j), opstr in self.elements.items():
                ret[i, j] = ", ".join(["NaN" if _np.isnan(x) else
                                       (fmt % x) for x in elementvec[layout.indices(opstr)]])
        else:
            raise ValueError("Invalid `mergeop` arg: %s" % str(mergeop))
        return ret

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Create a new plaquette with circuits manipulated according to `processor_fn`.

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

    def truncate(self, circuits_to_keep, keep_rows_cols=False):
        """
        Remove any circuits from this plaquette that aren't in `circuits_to_keep`.

        Parameters
        ----------
        circuits_to_keep : list or set
            List of circuits to keep.  If None, then a copy of this object is returned.

        keep_rows_cols : bool
            Whether to retain the same number of rows as columns (even if entire rows
            and/or columns are empty).

        Returns
        -------
        CircuitPlaquette
        """
        if circuits_to_keep is None:
            return self.copy()

        if not isinstance(circuits_to_keep, set): circuits_to_keep = set(circuits_to_keep)
        num_rows, num_cols = (self.num_rows, self.num_cols) if keep_rows_cols else (None, None)
        elements = {(i, j): c for (i, j), c in self.elements.items() if c in circuits_to_keep}
        return CircuitPlaquette(elements, num_rows, num_cols, self.op_label_aliases)

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
        return ("%d circuits" % len(self)) if len(self) > 0 else ""

    def element_label(self, irow, icol):
        c = self.elements.get((irow, icol), None)
        return c.layerstr if (c is not None) else ""


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
        A list or dict of `(prepFiducial, effectFiducial)` tuples specifying how
        `elements` is generated from `base`, i.e. by `prepFiducial + base + effectFiducial`.
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

    circuit_rules : list, optional
        A list of `(find,replace)` 2-tuples which specify string replacement
        rules.  Both `find` and `replace` are tuples of operation labels
        (or `Circuit` objects).
    """

    def __init__(self, base, fidpairs, num_rows=None, num_cols=None, op_label_aliases=None, circuit_rules=None):
        """
        Create a new FiducialPairPlaquette.
        """
        self.base = base
        if not isinstance(fidpairs, dict):
            cols = num_cols if (num_cols is not None) else int(_np.ceil(_np.sqrt(len(fidpairs))))
            fidpairs = _collections.OrderedDict(
                [((k % cols, k // cols), fidpair) for k, fidpair in enumerate(fidpairs)])
        self.fidpairs = fidpairs.copy()
        super().__init__(_collections.OrderedDict([((i, j), prep + base + meas)
                                                   for (i, j), (prep, meas) in fidpairs.items()]),
                         num_rows, num_cols, op_label_aliases, circuit_rules)

    def _to_nice_serialization(self):  # memo holds already serialized objects
        state = super()._to_nice_serialization()
        state.update({'base_circuit': self.base.str,
                      'fiducial_pairs': [(ij_key, fidpair[0].str, fidpair[1].str)
                                         for ij_key, fidpair in self.fidpairs.items()]
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.io import stdinput as _stdinput
        std = _stdinput.StdInputParser()
        base = std.parse_circuit(state['base_circuit'], create_subcircuits=not _Circuit.default_expand_subcircuits)
        fidpairs = {tuple(ij): (std.parse_circuit(prepfid, create_subcircuits=not _Circuit.default_expand_subcircuits),
                                std.parse_circuit(measfid, create_subcircuits=not _Circuit.default_expand_subcircuits))
                    for ij, prepfid, measfid in state['fiducial_pairs']}
        return cls(base, fidpairs, state['num_rows'], state['num_cols'], None, None)
        # Note: parent calls _post_from_nice_serialization_init so we don't serialize op_label_aliases and circuit_rules

    def _post_from_nice_serialization_init(self, op_label_aliases, circuit_rules):
        super()._post_from_nice_serialization_init(op_label_aliases, circuit_rules)  # sets members
        if self.circuit_rules is not None:  # reset elements since these aren't serialzed
            self.elements = _collections.OrderedDict([(k, _manipulate_circuit(c, self.circuit_rules))
                                                      for k, c in self.elements.items()])

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Create a new plaquette with circuits manipulated according to `processor_fn`.

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
        updated_fidpairs = _collections.OrderedDict([(coords, (P(prep), P(meas)))
                                                     for coords, (prep, meas) in self.fidpairs.items()])
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
        new_fidpairs = _collections.OrderedDict()
        for coords, (prep, meas) in self.fidpairs.items():
            prep2 = prep.replace_layers_with_aliases(self.op_label_aliases)
            meas2 = meas.replace_layers_with_aliases(self.op_label_aliases)
            if ds_filter is None or prep2 + new_base + meas2 in ds_filter:
                new_fidpairs[coords] = (prep2, meas2)
        return FiducialPairPlaquette(new_base, new_fidpairs, self.num_rows, self.num_cols, op_label_aliases=None)

    def truncate(self, circuits_to_keep, keep_rows_cols=False):
        """
        Remove any circuits from this plaquette that aren't in `circuits_to_keep`.

        Parameters
        ----------
        circuits_to_keep : list or set
            List of circuits to keep.  If None, then a copy of this object is returned.

        keep_rows_cols : bool
            Whether to retain the same number of rows as columns (even if entire rows
            and/or columns are empty).

        Returns
        -------
        FiducialPairPlaquette
        """
        if circuits_to_keep is None:
            return self.copy()

        if not isinstance(circuits_to_keep, set): circuits_to_keep = set(circuits_to_keep)
        fidpairs = _collections.OrderedDict()
        for (i, j), c in self.elements.items():
            if c in circuits_to_keep:
                fidpairs[(i, j)] = self.fidpairs[(i, j)]
        num_rows, num_cols = (self.num_rows, self.num_cols) if keep_rows_cols else (None, None)
        return FiducialPairPlaquette(self.base, fidpairs, num_rows, num_cols, self.op_label_aliases)

    def copy(self):
        """
        Returns a copy of this `CircuitPlaquette`.

        Returns
        -------
        FiducialPairPlaquette
        """
        aliases = _copy.deepcopy(self.op_label_aliases) if (self.op_label_aliases is not None) else None
        return FiducialPairPlaquette(self.base, self.fidpairs, self.num_rows, self.num_cols, aliases)

    def summary_label(self):
        return "{}" if len(self.base) == 0 else self.base.layerstr

    def element_label(self, irow, icol):
        prep, meas = self.fidpairs.get((irow, icol), (None, None))
        if prep is None or meas is None:
            return ""
        else:
            #return f"{prep.layerstr} + " + self.summary_label() + f" + {meas.layerstr}"
            return " + ".join((prep.layerstr, self.summary_label(), meas.layerstr))


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

    circuit_rules : list, optional
        A list of `(find,replace)` 2-tuples which specify string replacement
        rules.  Both `find` and `replace` are tuples of operation labels
        (or `Circuit` objects).
    """

    def __init__(self, germ, power, fidpairs, num_rows=None, num_cols=None, op_label_aliases=None, circuit_rules=None):
        """
        Create a new GermFiducialPairPlaquette.
        """
        self.germ = germ
        self.power = power
        super().__init__(germ**power, fidpairs, num_rows, num_cols, op_label_aliases, circuit_rules)

    def _to_nice_serialization(self):  # memo holds already serialized objects
        state = super()._to_nice_serialization()
        state.update({'germ': self.germ.str,
                      'power': self.power,
                      'fiducial_pairs': [(ij_key, fidpair[0].str, fidpair[1].str)
                                         for ij_key, fidpair in self.fidpairs.items()]
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.io import stdinput as _stdinput
        std = _stdinput.StdInputParser()
        germ = std.parse_circuit(state['germ'], create_subcircuits=not _Circuit.default_expand_subcircuits)
        fidpairs = {tuple(ij): (std.parse_circuit(prepfid, create_subcircuits=not _Circuit.default_expand_subcircuits),
                                std.parse_circuit(measfid, create_subcircuits=not _Circuit.default_expand_subcircuits))
                    for ij, prepfid, measfid in state['fiducial_pairs']}
        return cls(germ, state['power'], fidpairs, state['num_rows'], state['num_cols'], None, None)
        # Note: parent calls _post_from_nice_serialization_init so we don't serialize op_label_aliases and circuit_rules

    def _post_from_nice_serialization_init(self, op_label_aliases, circuit_rules):
        super()._post_from_nice_serialization_init(op_label_aliases, circuit_rules)  # sets members
        if self.circuit_rules is not None:  # reset elements since these aren't serialzed
            self.elements = _collections.OrderedDict([(k, _manipulate_circuit(c, self.circuit_rules))
                                                      for k, c in self.elements.items()])

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Create a new plaquette with circuits manipulated according to `processor_fn`.

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
        updated_fidpairs = _collections.OrderedDict([(coords, (P(prep), P(meas)))
                                                     for coords, (prep, meas) in self.fidpairs.items()])
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
        new_fidpairs = _collections.OrderedDict()
        for coords, (prep, meas) in self.fidpairs.items():
            prep2 = prep.replace_layers_with_aliases(self.op_label_aliases)
            meas2 = meas.replace_layers_with_aliases(self.op_label_aliases)
            if ds_filter is None or prep2 + new_base + meas2 in ds_filter:
                new_fidpairs[coords] = (prep2, meas2)
        return GermFiducialPairPlaquette(new_germ, self.power, new_fidpairs, self.num_rows, self.num_cols,
                                         op_label_aliases=None)

    def truncate(self, circuits_to_keep, keep_rows_cols=False):
        """
        Remove any circuits from this plaquette that aren't in `circuits_to_keep`.

        Parameters
        ----------
        circuits_to_keep : list or set
            List of circuits to keep.  If None, then a copy of this object is returned.

        keep_rows_cols : bool
            Whether to retain the same number of rows as columns (even if entire rows
            and/or columns are empty).

        Returns
        -------
        GermFiducialPairPlaquette
        """
        if circuits_to_keep is None:
            return self.copy()

        if not isinstance(circuits_to_keep, set): circuits_to_keep = set(circuits_to_keep)
        fidpairs = _collections.OrderedDict()
        for (i, j), c in self.elements.items():
            if c in circuits_to_keep:
                fidpairs[(i, j)] = self.fidpairs[(i, j)]
        num_rows, num_cols = (self.num_rows, self.num_cols) if keep_rows_cols else (None, None)
        return GermFiducialPairPlaquette(self.germ, self.power, fidpairs, num_rows, num_cols, self.op_label_aliases)

    def copy(self):
        """
        Returns a copy of this `CircuitPlaquette`.

        Returns
        -------
        GermFiducialPairPlaquette
        """
        aliases = _copy.deepcopy(self.op_label_aliases) if (self.op_label_aliases is not None) else None
        return GermFiducialPairPlaquette(self.germ, self.power, self.fidpairs.copy(), self.num_rows, self.num_cols,
                                         aliases)

    def summary_label(self):
        if len(self.germ) == 0 or self.power == 0:
            return "{}"
        else:
            #return f"({self.germ.layerstr})<sup>{self.power}</sup>"
            return "(%s)<sup>%d</sup>" % (self.germ.layerstr, self.power)


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
            weights_dict = {c: wt for c, wt in zip(circuits_or_structure, circuits_or_structure.circuit_weights)} \
                if (circuits_or_structure.circuit_weights is not None) else None
            name = circuits_or_structure.name
        else:
            op_label_aliases = weights_dict = name = None

        return cls({}, [], [], '', '', circuits_or_structure,
                   op_label_aliases, weights_dict, name)

    def __init__(self, plaquettes, x_values, y_values, xlabel, ylabel, additional_circuits=None, op_label_aliases=None,
                 circuit_rules=None, circuit_weights_dict=None, additional_circuits_location='start', name=None):
        # plaquettes is a dict of plaquettes whose keys are tuples of length 2
        self._plaquettes = _collections.OrderedDict(plaquettes)
        self.xs = x_values
        self.ys = y_values
        self.xlabel = xlabel
        self.ylabel = ylabel

        assert(additional_circuits_location in ('start', 'end'))
        self._addl_location = additional_circuits_location

        circuits = _collections.OrderedDict()  # use as an ordered *set* (values all == None)
        if additional_circuits is None:
            additional_circuits = []

        if additional_circuits_location == 'start':
            circuits.update([(c, None) for c in additional_circuits])

        circuits_in_plaquettes = _collections.OrderedDict()
        for plaq in plaquettes.values():
            circuits_in_plaquettes.update([(c, None) for c in plaq.circuits])
        circuits.update(circuits_in_plaquettes)

        if additional_circuits_location == 'end':
            circuits.update([(c, None) for c in additional_circuits])

        # ordered-sets => tuples
        additional = _collections.OrderedDict([(c, None) for c in additional_circuits
                                               if (c not in circuits_in_plaquettes)])
        self._additional_circuits = tuple(additional.keys())  # only those circuits *not* in plaquettes
        circuits = tuple(circuits.keys())

        circuit_weights = None if (circuit_weights_dict is None) else \
            _np.array([circuit_weights_dict.get(c, 0.0) for c in circuits], 'd')
        super().__init__(circuits, op_label_aliases, circuit_rules, circuit_weights, name)

    def _to_nice_serialization(self):  # memo holds already serialized objects
        from pygsti.io.writers import convert_circuits_to_strings as _convert_circuits_to_strings

        def _gettype(vals):
            if all([isinstance(v, int) for v in vals]): return "int"
            if all([isinstance(v, str) for v in vals]): return "str"
            if all([isinstance(v, _Circuit) for v in vals]): return "circuit"
            raise ValueError("Unknown/mixed type(s) to serialize!")

        def _encodeval(typ, val):
            if typ in ("int", "str"): return val
            if typ == "circuit": return val.str
            assert(False), "Internal logic error!"

        xtype = _gettype(self.xs)
        ytype = _gettype(self.ys)

        state = _NicelySerializable._to_nice_serialization(self)  # don't call CircuitList version
        state.update({'name': self.name,
                      'xvalues': [_encodeval(xtype, x) for x in self.xs],
                      'yvalues': [_encodeval(ytype, y) for y in self.ys],
                      'xtype': xtype,
                      'ytype': ytype,
                      'xlabel': self.xlabel,  # str
                      'ylabel': self.ylabel,  # str
                      'op_label_aliases': _convert_circuits_to_strings(self.op_label_aliases),
                      'circuit_rules': _convert_circuits_to_strings(self.circuit_rules),
                      'additional_circuits_location': self._addl_location,
                      'plaquettes': [((_encodeval(xtype, x), _encodeval(ytype, y)), p.to_nice_serialization())
                                     for (x, y), p in self._plaquettes.items()],
                      'additional_circuits': [c.str for c in self._additional_circuits],
                      'circuit_weights': ({c.str: weight for c, weight in self.circuit_weights}
                                          if (self.circuit_weights is not None) else None)
                      })

        #state['plaquette_types'] = XXX
        #state['datacols'] = {
        #    'Circuit': [circuits]
        #    'PlaquetteX':
        #    'PlaquetteY':
        #    'X':
        #    'Y':
        #    'weight':
        #    # additional circuits have sentinels for Plaquette coords?

        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        from pygsti.io.readers import convert_strings_to_circuits as _convert_strings_to_circuits
        from pygsti.io import stdinput as _stdinput
        std = _stdinput.StdInputParser()

        def _decodeval(typ, val):
            if typ in ("int", "str"): return val
            if typ == "circuit": return std.parse_circuit(val,
                                                          create_subcircuits=not _Circuit.default_expand_subcircuits)
            raise ValueError("Unknown x/y type: %s" % str(typ))

        op_label_aliases = _convert_strings_to_circuits(state['op_label_aliases'])
        circuit_rules = _convert_strings_to_circuits(state['circuit_rules']) \
            if ('circuit_rules' in state) else None  # for backward compatibility (REMOVE in FUTURE?)

        plaquettes = {(_decodeval(state['xtype'], x), _decodeval(state['ytype'], y)):
                      CircuitPlaquette.from_nice_serialization(d) for (x, y), d in state['plaquettes']}
        for p in plaquettes.values():
            p._post_from_nice_serialization_init(op_label_aliases, circuit_rules)

        additional_circuits = [std.parse_circuit(s, create_subcircuits=not _Circuit.default_expand_subcircuits)
                               for s in state['additional_circuits']]
        circuit_weights = ({std.parse_circuit(s, create_subcircuits=not _Circuit.default_expand_subcircuits): weight
                           for s, weight in state['circuit_weights'].items()}
                           if (state['circuit_weights'] is not None) else None)
        xvalues = tuple([_decodeval(state['xtype'], x) for x in state['xvalues']])
        yvalues = tuple([_decodeval(state['ytype'], y) for y in state['yvalues']])

        return cls(plaquettes, xvalues, yvalues, state['xlabel'], state['ylabel'],
                   additional_circuits, op_label_aliases, circuit_rules, circuit_weights,
                   state['additional_circuits_location'], name=state['name'])

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
            #return CircuitPlaquette([], 0, 0)  # an empty plaquette
            nrows = ncols = 0
            if len(self._plaquettes) > 0:  # use number of rows & cols from first plaquette (if exists)
                first_plaq = next(iter(self._plaquettes.values()))
                nrows, ncols = first_plaq.num_rows, first_plaq.num_cols
            return CircuitPlaquette([], nrows, ncols)  # an empty plaquette
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

    def truncate(self, circuits_to_keep=None, xs_to_keep=None, ys_to_keep=None, keep_rows_cols=True):
        """
        Truncate this circuit structure to a subset of its current circuits.

        Parameters
        ----------
        circuits_to_keep : list
            Keep only the circuits present in this list (of Circuit objects).

        xs_to_keep : list, optional
            The x-values to keep.  If None, then all are kept.

        ys_to_keep : list, optional
            The y-values to keep.  If None, then all are kept.

        keep_rows_cols : bool
            Whether to retain the same number of rows as columns (even if entire rows
            and/or columns are empty).  By default, this is `True` because we usually
            want all the plaquettes of a :class:`PlaquetteGridCircuitStructure` to have
            the same number of rows and columns.

        Returns
        -------
        PlaquetteGridCircuitStructure
        """
        xs = self.xs if (xs_to_keep is None) else xs_to_keep
        ys = self.ys if (ys_to_keep is None) else ys_to_keep

        plaquettes = _collections.OrderedDict()
        for (x, y), plaq in self._plaquettes.items():
            if not ((x in xs) and (y in ys)): continue
            plaquettes[(x, y)] = plaq.truncate(circuits_to_keep, keep_rows_cols)

        circuit_weights_dict = {c: weight for c, weight in zip(self, self.circuit_weights)} \
            if (self.circuit_weights is not None) else None
        additional = list(filter(lambda c: c in circuits_to_keep, self._additional_circuits)) \
            if (circuits_to_keep is not None) else self._additional_circuits
        return PlaquetteGridCircuitStructure(plaquettes, xs, ys, self.xlabel, self.ylabel, additional,
                                             self.op_label_aliases, circuit_weights_dict=circuit_weights_dict,
                                             additional_circuits_location=self._addl_location, name=self.name)

    def nested_truncations(self, axis='x', keep_rows_cols=False):
        """
        Get the nested truncations of this circuit structure along an axis.

        When `axis == 'x'`, a list of truncations (of this structure)
        that keep an incrementally larger set of all the x-values.  E.g.,
        if the x-values are `[1,2,4]`, truncations to `[1]`, `[1,2]`,
        and `[1,2,4]` (no truncation) would be returned.

        Setting `axis =='y'` gives the same behavior except using
        the y-values.

        Parameters
        ----------
        axis : {'x', 'y'}
            Which axis to truncate along (see above).

        keep_rows_cols : bool
            Whether to retain the same number of rows as columns (even if entire rows
            and/or columns are empty).

        Returns
        -------
        list
            A list of :class:`PlaquetteGridCircuitStructure` objects
            (truncations of this object).
        """
        if axis == 'x':
            return [self.truncate(xs_to_keep=self.xs[0:i + 1], keep_rows_cols=keep_rows_cols)
                    for i in range(len(self.xs))]
        elif axis == 'y':
            return [self.truncate(ys_to_keep=self.ys[0:i + 1], keep_rows_cols=keep_rows_cols)
                    for i in range(len(self.ys))]
        else:
            #raise ValueError(f"Invalid `axis` argument: {axis} - must be 'x' or 'y'!")
            raise ValueError("Invalid `axis` argument: %s - must be 'x' or 'y'!" % str(axis))

    def process_circuits(self, processor_fn, updated_aliases=None):
        """
        Create a new plaquette with circuits manipulated according to `processor_fn`.

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

        plaquettes = _collections.OrderedDict([(k, v.process_circuits(P, updated_aliases))
                                               for k, v in self._plaquettes.items()])
        if len(self.xs) > 0 and isinstance(self.xs[0], _Circuit):
            xs = list(map(P, self.xs))
            plaquettes = _collections.OrderedDict([((P(x), y), v) for (x, y), v in plaquettes.items()])
        else:
            xs = self.xs

        if len(self.ys) > 0 and isinstance(self.ys[0], _Circuit):
            ys = list(map(P, self.ys))
            plaquettes = _collections.OrderedDict([((x, P(y)), v) for (x, y), v in plaquettes.items()])
        else:
            ys = self.ys

        additional = list(map(P, self._additional_circuits))

        circuit_weights_dict = {P(c): weight for c, weight in zip(self, self.circuit_weights)} \
            if (self.circuit_weights is not None) else None
        return PlaquetteGridCircuitStructure(plaquettes, xs, ys, self.xlabel, self.ylabel, additional, updated_aliases,
                                             circuit_weights_dict=circuit_weights_dict,
                                             additional_circuits_location=self._addl_location, name=self.name)

    def copy(self):
        """
        Returns a copy of this circuit structure.

        Returns
        -------
        PlaquetteGridCircuitStructure
        """
        circuit_weights_dict = {c: weight for c, weight in zip(self, self.circuit_weights)} \
            if (self.circuit_weights is not None) else None
        return PlaquetteGridCircuitStructure(self._plaquettes.copy(), self.xs[:], self.ys[:], self.xlabel, self.ylabel,
                                             self._additional_circuits, self.op_label_aliases,
                                             circuit_weights_dict=circuit_weights_dict,
                                             additional_circuits_location=self._addl_location, name=self.name)
