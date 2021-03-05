"""
Defines the Circuit class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numbers as _numbers
import numpy as _np
import copy as _copy
import itertools as _itertools
import warnings as _warnings
import collections as _collections

from . import labeldicts as _ld
from .label import Label as _Label, CircuitLabel as _CircuitLabel
from ..io import CircuitParser as _CircuitParser
from ..tools import internalgates as _itgs
from ..tools import compattools as _compat
from ..tools import slicetools as _slct


#Internally:
# when static: a tuple of Label objects labelling each top-level circuit layer
# when editable: a list of lists, one per top-level layer, holding just
# the non-LabelTupTup (non-compound) labels.

#Externally, we'd like to do thinks like:
# c = Circuit( LabelList )
# c.append_line("Q0")
# c.append_layer(layer_label)
# c[2]['Q0'] = 'Gx'  # puts Gx:Q0 into circuit (at 3rd layer)
# c[2,'Q0'] = 'Gx'
# c[2,('Q0','Q1')] = Label('Gcnot') # puts Gcnot:Q0:Q1 into circuit
# c[2,('Q1','Q0')] = 'Gcnot'        # puts Gcnot:Q1:Q0 into circuit
# c[2] = (Label('Gx','Q0'), Label('Gy','Q1')) # assigns a circuit layer
# c[2,:] = (Label('Gx','Q0'), Label('Gy','Q1')) # assigns a circuit layer
# del c[2]
# c.insert(2, (Label('Gx','Q0'), Label('Gy','Q1')) ) # inserts a layer
# c[:,'Q0'] = ('Gx','Gy','','Gx') # assigns the Q0 line
# c[1:3,'Q0'] = ('Gx','Gy') # assigns to a part of the Q0 line


def _np_to_quil_def_str(name, input_array):
    """
    Write a DEFGATE block for RQC quil for an arbitrary one- or two-qubit unitary gate.
    (quil/pyquil currently does not offer support for arbitrary n-qubit gates for
    n>2.)

    Parameters
    ----------
    name : str
        The name of the gate (e.g., 'Gc0' for the 0th Clifford gate)

    input_array : array_like
        The representation of the gate as a unitary map.
        E.g., for name = 'Gc0',input_array = np.array([[1,0],[0,1]])

    Returns
    -------
    output : str
        A block of quil code (as a string) that should be included before circuit
        declaration in any quil circuit that uses the specified gate.
    """
    output = 'DEFGATE {}:\n'.format(name)
    for line in input_array:
        output += '    '
        output += ', '.join(map(_num_to_rqc_str, line))
        output += '\n'
    return output


def _num_to_rqc_str(num):
    """Convert float to string to be included in RQC quil DEFGATE block
    (as written by _np_to_quil_def_str)."""
    num = _np.complex(_np.real_if_close(num))
    if _np.imag(num) == 0:
        output = str(_np.real(num))
        return output
    else:
        real_part = _np.real(num)
        imag_part = _np.imag(num)
        if imag_part < 0:
            sgn = '-'
            imag_part = imag_part * -1
        elif imag_part > 0:
            sgn = '+'
        else:
            assert False
        return '{}{}{}i'.format(real_part, sgn, imag_part)


def _label_to_nested_lists_of_simple_labels(lbl, default_sslbls=None, always_return_list=True):
    """ Convert lbl into nested lists of *simple* labels """
    if not isinstance(lbl, _Label):  # if not a Label, make into a label,
        lbl = _Label(lbl)  # e.g. a string or list/tuple of labels, etc.
    if lbl.is_simple():  # a *simple* label - the elements of our lists
        if lbl.sslbls is None and default_sslbls is not None:
            lbl = _Label(lbl.name, default_sslbls)
        return [lbl] if always_return_list else lbl
    return [_label_to_nested_lists_of_simple_labels(l, default_sslbls, False)
            for l in lbl.components]  # a *list*


def _sslbls_of_nested_lists_of_simple_labels(obj, labels_to_ignore=None):
    """ Get state space labels from a nested lists of simple (not compound) Labels. """
    if isinstance(obj, _Label):
        if labels_to_ignore and (obj in labels_to_ignore):
            return ()
        return obj.sslbls
    else:
        sub_sslbls = [_sslbls_of_nested_lists_of_simple_labels(sub, labels_to_ignore) for sub in obj]
        return None if (None in sub_sslbls) else set(_itertools.chain(*sub_sslbls))


def _accumulate_explicit_sslbls(obj):
    """
    Get all the explicitly given state-space labels within `obj`,
    which can be a Label or a list/tuple of labels.  Returns a *set*.
    """
    ret = set()
    if isinstance(obj, _Label):
        if not obj.is_simple():
            for lbl in obj.components:
                ret.update(_accumulate_explicit_sslbls(lbl))
        else:  # a simple label
            if obj.sslbls is not None:  # don't know how to interpet None sslbls
                return set(obj.sslbls)
    else:  # things that aren't labels we assume are iterable
        for lbl in obj:
            ret.update(_accumulate_explicit_sslbls(lbl))
    return ret


def _op_seq_str_suffix(line_labels, occurrence_id):
    """ The suffix (after the first "@") of a circuit's string rep"""
    if occurrence_id is None:
        return "" if line_labels is None or line_labels == ('*',) else \
            "@(" + ','.join(map(str, line_labels)) + ")"
    else:
        return "@()@" + str(occurrence_id) if (line_labels is None or line_labels == ('*',)) \
            else "@(" + ','.join(map(str, line_labels)) + ")@" + str(occurrence_id)


def _op_seq_to_str(seq, line_labels, occurrence_id):
    """ Used for creating default string representations. """
    if len(seq) == 0:  # special case of empty operation sequence (for speed)
        return "{}" + _op_seq_str_suffix(line_labels, occurrence_id)

    def process_lists(el): return el if not isinstance(el, list) else \
        ('[%s]' % ''.join(map(str, el)) if (len(el) != 1) else str(el[0]))

    return ''.join(map(str, map(process_lists, seq))) + _op_seq_str_suffix(line_labels, occurrence_id)


def to_label(x):
    """
    Helper function for converting `x` to a single Label object

    Parameters
    ----------
    x : various
        An object to (attempt to) convert into a :class:`Label`.

    Returns
    -------
    Label
    """
    if isinstance(x, _Label): return x
    # # do this manually when desired, as it "boxes" a circuit being inserted
    #elif isinstance(x,Circuit): return x.to_circuit_label()
    else: return _Label(x)


class Circuit(object):
    """
    A quantum circuit.

    A Circuit represents a quantum circuit, consisting of state preparation,
    gates, and measurement operations.  It is composed of some number of "lines",
    typically one per qubit, and stores the operations on these lines as a
    sequence of :class:`Label` objects, one per circuit layer, whose `.sslbls`
    members indicate which line(s) the label belongs on.  When a circuit is
    created with 'editable=True', a rich set of operations may be used to
    construct the circuit in place, after which `done_editing()` should be
    called so that the Circuit can be properly hashed as needed.

    Parameters
    ----------
    layer_labels : iterable of Labels or str
        This argument provides a list of the layer labels specifying the
        state preparations, gates, and measurements for the circuit.  This
        argument can also be a :class:`Circuit` or a string, in which case
        it is parsed as a text-formatted circuit.  Internally this will
        eventually be converted to a list of `Label` objects, one per layer,
        but it may be specified using anything that can be readily converted
        to a Label objects.  For example, any of the following are allowed:

        - `['Gx','Gx']` : X gate on each of 2 layers
        - `[Label('Gx'),Label('Gx')] : same as above
        - `[('Gx',0),('Gy',0)]` : X then Y on qubit 0 (2 layers)
        - `[[('Gx',0),('Gx',1)],[('Gy',0),('Gy',1)]]` : parallel X then Y on qubits 0 & 1

    line_labels : iterable, optional
        The (string valued) label for each circuit line.  If `'auto'`, then
        `line_labels` is taken to be the list of all state-space labels
        present within `layer_labels`.  If there are no such labels (e.g.
        if `layer_labels` contains just gate names like `('Gx','Gy')`), then
        the special value `'*'` is used as a single line label.

    num_lines : int, optional
        Specify this instead of `line_labels` to set the latter to the
        integers between 0 and `num_lines-1`.

    editable : bool, optional
        Whether the created `Circuit` is created in able to be modified.  If
        `True`, then you should call `done_editing()` once the circuit is
        completely assembled, as this makes the circuit read-only and
        allows it to be hashed.

    stringrep : string, optional
        A string representation for the circuit.  If `None` (the default),
        then this will be generated automatically when needed.  One
        reason you'd want to specify this is if you know of a nice compact
        string representation that you'd rather use, e.g. `"Gx^4"` instead
        of the automatically generated `"GxGxGxGx"`.  If you want to
        initialize a `Circuit` entirely from a string representation you
        can either specify the string in as `layer_labels` or set
        `layer_labels` to `None` and `stringrep` to any valid (one-line)
        circuit string.

    name : str, optional
        A name for this circuit (useful if/when used as a block within
        larger circuits).

    check : bool, optional
        Whether `stringrep` should be checked against `layer_labels` to
        ensure they are consistent, and whether the labels in `layer_labels`
        are a subset of `line_labels`.  The only reason you'd want to set
        this to `False` is if you're absolutely sure `stringrep` and
        `line_labels` are consistent and want to save computation time.

    expand_subcircuits : bool or "default"
        If `"default"`, then the value of `Circuit.default_expand_subcircuits`
        is used.  If True, then any sub-circuits (e.g. anything exponentiated
        like "(GxGy)^4") will be expanded when it is stored within the created
        Circuit.  If False, then such sub-circuits will be left as-is.  It's
        typically more robust to expand sub-circuits as this facilitates
        comparison (e.g. so "GxGx" == "Gx^2"), but in cases when you have
        massive exponents (e.g. "Gx^8192") it may improve performance to
        set `expand_subcircuits=False`.

    occurrence : hashable, optional
        A value to set as the "occurrence id" for this circuit.  This
        value doesn't affect the circuit an any way except by affecting
        it's hashing and equivalence testing.  Circuits with different
        occurrence ids are *not* equivalent.  Occurrence values effectively
        allow multiple copies of the same ciruit to be stored in a
        dictionary or :class:`DataSet`.

    Attributes
    ----------
    default_expand_subcircuits : bool
        By default, expand sub-circuit labels.

    line_labels : tuple
        The line labels (often qubit labels) of this circuit.

    layertup : tuple
        This Circuit's layers as a standard Python tuple of layer Labels.

    tup : tuple
        This Circuit as a standard Python tuple of layer Labels and line labels.

    str : str
        The Python string representation of this Circuit.
    """
    default_expand_subcircuits = True

    @classmethod
    def cast(cls, obj):
        """
        Convert `obj` into a :class:`Circuit`.

        Parameters
        ----------
        obj : object
            Object to convert

        Returns
        -------
        Circuit
        """
        if isinstance(obj, cls): return obj
        if isinstance(obj, (tuple, list)): return cls.from_tuple(obj)
        raise ValueError("Cannot create an %s object from '%s'" % (cls.__name__, str(type(obj))))

    @classmethod
    def from_tuple(cls, tup):
        """
        Creates a :class:`Circuit` from a tuple

        Parameters
        ----------
        tup : tuple
            The tuple to convert.

        Returns
        -------
        Circuit
        """
        if '@' in tup:
            k = tup.index('@')
            return cls(tup[0:k], tup[k + 1:])
        else:
            return cls(tup)

    def __init__(self, layer_labels=(), line_labels='auto', num_lines=None, editable=False,
                 stringrep=None, name='', check=True, expand_subcircuits="default",
                 occurrence=None):
        """
        Creates a new Circuit object, encapsulating a quantum circuit.

        You only need to supply the first `layer_labels` argument, though
        usually (except for just 1 or 2 qubits) you'll want to also supply
        `line_labels` or `num_lines`.  If you'll be adding to or altering
        the circuit before using it, you should set `editable=True`.

        Parameters
        ----------
        layer_labels : iterable of Labels or str
            This argument provides a list of the layer labels specifying the
            state preparations, gates, and measurements for the circuit.  This
            argument can also be a :class:`Circuit` or a string, in which case
            it is parsed as a text-formatted circuit.  Internally this will
            eventually be converted to a list of `Label` objects, one per layer,
            but it may be specified using anything that can be readily converted
            to a Label objects.  For example, any of the following are allowed:

            - `['Gx','Gx']` : X gate on each of 2 layers
            - `[Label('Gx'),Label('Gx')] : same as above
            - `[('Gx',0),('Gy',0)]` : X then Y on qubit 0 (2 layers)
            - `[[('Gx',0),('Gx',1)],[('Gy',0),('Gy',1)]]` : parallel X then Y on qubits 0 & 1

        line_labels : iterable, optional
            The (string valued) label for each circuit line.  If `'auto'`, then
            `line_labels` is taken to be the list of all state-space labels
            present within `layer_labels`.  If there are no such labels (e.g.
            if `layer_labels` contains just gate names like `('Gx','Gy')`), then
            the special value `'*'` is used as a single line label.

        num_lines : int, optional
            Specify this instead of `line_labels` to set the latter to the
            integers between 0 and `num_lines-1`.

        editable : bool, optional
            Whether the created `Circuit` is created in able to be modified.  If
            `True`, then you should call `done_editing()` once the circuit is
            completely assembled, as this makes the circuit read-only and
            allows it to be hashed.

        stringrep : string, optional
            A string representation for the circuit.  If `None` (the default),
            then this will be generated automatically when needed.  One
            reason you'd want to specify this is if you know of a nice compact
            string representation that you'd rather use, e.g. `"Gx^4"` instead
            of the automatically generated `"GxGxGxGx"`.  If you want to
            initialize a `Circuit` entirely from a string representation you
            can either specify the string in as `layer_labels` or set
            `layer_labels` to `None` and `stringrep` to any valid (one-line)
            circuit string.

        name : str, optional
            A name for this circuit (useful if/when used as a block within
            larger circuits).

        check : bool, optional
            Whether `stringrep` should be checked against `layer_labels` to
            ensure they are consistent, and whether the labels in `layer_labels`
            are a subset of `line_labels`.  The only reason you'd want to set
            this to `False` is if you're absolutely sure `stringrep` and
            `line_labels` are consistent and want to save computation time.

        expand_subcircuits : bool or "default"
            If `"default"`, then the value of `Circuit.default_expand_subcircuits`
            is used.  If True, then any sub-circuits (e.g. anything exponentiated
            like "(GxGy)^4") will be expanded when it is stored within the created
            Circuit.  If False, then such sub-circuits will be left as-is.  It's
            typically more robust to expand sub-circuits as this facilitates
            comparison (e.g. so "GxGx" == "Gx^2"), but in cases when you have
            massive exponents (e.g. "Gx^8192") it may improve performance to
            set `expand_subcircuits=False`.

        occurrence : hashable, optional
            A value to set as the "occurrence id" for this circuit.  This
            value doesn't affect the circuit an any way except by affecting
            it's hashing and equivalence testing.  Circuits with different
            occurrence ids are *not* equivalent.  Occurrence values effectively
            allow multiple copies of the same ciruit to be stored in a
            dictionary or :class:`DataSet`.
        """
        layer_labels_objs = None  # layer_labels elements as Label objects (only if needed)
        if isinstance(layer_labels, str):
            cparser = _CircuitParser(); cparser.lookup = None
            layer_labels, chk_labels, chk_occurrence = cparser.parse(layer_labels)
            if chk_labels is not None:
                if line_labels == 'auto':
                    line_labels = chk_labels
                elif tuple(line_labels) != chk_labels:
                    raise ValueError(("Error intializing Circuit: "
                                      " `line_labels` and line labels in `layer_labels` do not match: %s != %s")
                                     % (line_labels, chk_labels))
            if chk_occurrence is not None:
                if occurrence is None:  # Also acts as "auto"
                    occurrence = chk_occurrence
                elif occurrence != chk_occurrence:
                    raise ValueError(("Error intializing Circuit: "
                                      " `occurrence` and occurrence ID in `layer_labels` do not match: %s != %s")
                                     % (occurrence, chk_occurrence))

        if expand_subcircuits == "default":
            expand_subcircuits = Circuit.default_expand_subcircuits
        if expand_subcircuits and layer_labels is not None:
            layer_labels_objs = tuple(_itertools.chain(*[x.expand_subcircuits() for x in map(to_label, layer_labels)]))
            #print("DB: Layer labels = ",layer_labels_objs)

        #Parse stringrep if needed
        if stringrep is not None and (layer_labels is None or check):
            cparser = _CircuitParser()
            cparser.lookup = None  # lookup - functionality removed as it wasn't used
            chk, chk_labels, chk_occurrence = cparser.parse(stringrep)  # tuple of Labels
            if expand_subcircuits and chk is not None:
                chk = tuple(_itertools.chain(*[x.expand_subcircuits() for x in map(to_label, chk)]))
                #print("DB: Check Layer labels = ",chk)

            if layer_labels is None:
                layer_labels = chk
            else:  # check == True
                if layer_labels_objs is None:
                    layer_labels_objs = tuple(map(to_label, layer_labels))
                if layer_labels_objs != tuple(chk):
                    #print("DB: ",layer_labels_objs,"VS",tuple(chk))
                    raise ValueError(("Error intializing Circuit: "
                                      " `layer_labels` and `stringrep` do not match: %s != %s\n"
                                      "(set `layer_labels` to None to infer it from `stringrep`)")
                                     % (layer_labels, stringrep))
            if chk_labels is not None:
                if line_labels == 'auto':
                    line_labels = chk_labels
                elif tuple(line_labels) != chk_labels:
                    raise ValueError(("Error intializing Circuit: "
                                      " `line_labels` and `stringrep` do not match: %s != %s (from %s)\n"
                                      "(set `line_labels` to None to infer it from `stringrep`)")
                                     % (line_labels, chk_labels, stringrep))

            if chk_occurrence is not None:
                if occurrence is None:  # Also acts as "auto"
                    occurrence = chk_occurrence
                elif occurrence != chk_occurrence:
                    raise ValueError(("Error intializing Circuit: "
                                      " `occurrence` and occurrence ID in `layer_labels` do not match: %s != %s")
                                     % (occurrence, chk_occurrence))

        if layer_labels is None:
            raise ValueError("Must specify `stringrep` when `layer_labels` is None")

        # Set self._line_labels
        if line_labels == 'auto':
            if layer_labels_objs is None:
                layer_labels_objs = tuple(map(to_label, layer_labels))
            explicit_lbls = _accumulate_explicit_sslbls(layer_labels_objs)
            if len(explicit_lbls) == 0:
                if num_lines is not None:
                    assert(num_lines >= 0), "`num_lines` must be >= 0!"
                    if len(layer_labels) > 0:
                        assert(num_lines > 0), "`num_lines` must be > 0!"
                    my_line_labels = tuple(range(num_lines))
                elif len(layer_labels) > 0 or not editable:
                    my_line_labels = ('*',)  # special single line-label when no line labels are given
                else:
                    my_line_labels = ()  # empty *editable* circuits begin with zero line labels (this is ok)
            else:
                my_line_labels = tuple(sorted(explicit_lbls))
        else:
            explicit_lbls = None
            my_line_labels = tuple(line_labels)

        if (num_lines is not None) and (num_lines != len(my_line_labels)):
            if num_lines > len(my_line_labels) and \
               set(my_line_labels).issubset(set(range(num_lines))):
                # special case where we just add missing integer-labeled line(s)
                my_line_labels = tuple(range(num_lines))
            else:
                raise ValueError("`num_lines` was expected to be %d but equals %d!" %
                                 (len(my_line_labels), num_lines))

        if check:
            if explicit_lbls is None:
                if layer_labels_objs is None:
                    layer_labels_objs = tuple(map(to_label, layer_labels))
                explicit_lbls = _accumulate_explicit_sslbls(layer_labels_objs)
            if not set(explicit_lbls).issubset(my_line_labels):
                raise ValueError("line labels must contain at least %s" % str(explicit_lbls))

        #Set compute self._labels, which is either a nested list of simple labels (non-static case)
        #  or a tuple of Label objects (static case)
        if not editable:
            if layer_labels_objs is None:
                layer_labels_objs = tuple(map(to_label, layer_labels))
            labels = layer_labels_objs
        else:
            labels = [_label_to_nested_lists_of_simple_labels(layer_lbl)
                      for layer_lbl in layer_labels]

        #Set *all* class attributes (separated so can call bare_init separately for fast internal creation)
        self._bare_init(labels, my_line_labels, editable, name, stringrep, occurrence)

        # # Special case: layer_labels can be a single CircuitLabel or Circuit
        # # (Note: a Circuit would work just fine, as a list of layers, but this performs some extra checks)
        # isCircuit = isinstance(layer_labels, _Circuit)
        # isCircuitLabel = isinstance(layer_labels, _CircuitLabel)
        # if isCircuitLabel:
        #    assert(line_labels is None or line_labels == "auto" or line_labels == expected_line_labels), \
        #        "Given `line_labels` (%s) are inconsistent with CircuitLabel's sslbls (%s)" \
        #        % (str(line_labels),str(layer_labels.sslbls))
        #    assert(num_lines is None or layer_labels.sslbls == tuple(range(num_lines))), \
        #        "Given `num_lines` (%d) is inconsistend with CircuitLabel's sslbls (%s)" \
        #        % (num_lines,str(layer_labels.sslbls))
        #    if name is None: name = layer_labels.name # Note: `name` can be used to rename a CircuitLabel

        #    self._line_labels = layer_labels.sslbls
        #    self._reps = layer_labels.reps
        #    self._name = name
        #    self._static = not editable

    @classmethod
    def _fastinit(cls, labels, line_labels, editable, name='', stringrep=None, occurrence=None):
        ret = cls.__new__(cls)
        ret._bare_init(labels, line_labels, editable, name, stringrep, occurrence)
        return ret

    def _bare_init(self, labels, line_labels, editable, name='', stringrep=None, occurrence=None):
        self._labels = labels
        self._line_labels = line_labels
        self._occurrence_id = occurrence
        self._static = not editable
        #self._reps = reps # repetitions: default=1, which remains unless we initialize from a CircuitLabel...
        self._name = name  # can be None
        self._str = stringrep if self._static else None  # can be None (lazy generation)
        self._times = None  # for FUTURE expansion
        self.auxinfo = {}  # for FUTURE expansion / user metadata
        self._alignmarks = ()  # layer indices *before* which there is an alignment mark

    def to_label(self, nreps=1):
        """
        Construct and return this entire circuit as a :class:`CircuitLabel`.

        Note: occurrence-id information is not stored in a circuit label, so
        circuits that differ only in their `occurence_id` will return circuit
        labels that are equal.

        Parameters
        ----------
        nreps : int, optional
            The number of times this circuit will be repeated (`CircuitLabels`
            support exponentiation and you can specify this here).

        Returns
        -------
        CircuitLabel
        """
        #Note: self._occurrence_id is NOT a held by a CircuitLabel (this seems reasonable)
        eff_line_labels = None if self._line_labels == ('*',) else self._line_labels  # special case
        return _CircuitLabel(self._name, self._labels, eff_line_labels, nreps)

    @property
    def line_labels(self):
        """
        The line labels (often qubit labels) of this circuit.
        """
        return self._line_labels

    @line_labels.setter
    def line_labels(self, value):
        """
        The line labels (often qubit labels) of this circuit.
        """
        if value == self._line_labels: return
        #added_line_labels = set(value) - set(self._line_labels) # it's always OK to add lines
        removed_line_labels = set(self._line_labels) - set(value)
        if removed_line_labels:
            idling_line_labels = set(self.idling_lines())
            removed_not_idling = removed_line_labels - idling_line_labels
            if removed_not_idling and self._static:
                raise ValueError("Cannot remove non-idling lines %s from a read-only circuit!" %
                                 str(removed_not_idling))
            else:
                self.delete_lines(tuple(removed_not_idling))
        self._line_labels = tuple(value)
        self._str = None  # regenerate string rep (it may have updated)

    @property
    def name(self):
        """
        The name of this circuit.

        Note: the name is *not* a part of the hashed value.
        The name is used to name the :class:`CircuitLabel` returned from :method:`to_label`.
        """
        return self._name

    @property
    def occurrence(self):
        """
        The occurrence id of this circuit.
        """
        return self._occurrence_id

    @occurrence.setter
    def occurrence(self, value):
        """
        The occurrence id of this circuit.
        """
        self._occurrence_id = value
        self._str = None  # regenerate string rep (it may have updated)

    @property
    def layertup(self):
        """
        This Circuit's layers as a standard Python tuple of layer Labels.

        Returns
        -------
        tuple
        """
        if self._static:
            return self._labels
        else:
            return tuple([to_label(layer_lbl) for layer_lbl in self._labels])

    @property
    def tup(self):
        """
        This Circuit as a standard Python tuple of layer Labels and line labels.

        Returns
        -------
        tuple
        """
        if self._occurrence_id is None:
            if self._line_labels in (('*',), ()):  # No line labels
                return self.layertup
            else:
                return self.layertup + ('@',) + self._line_labels
        else:
            linelbl_tup = () if self._line_labels in (('*',), ()) else self._line_labels
            return self.layertup + ('@',) + linelbl_tup + ('@', self._occurrence_id)
            # Note: we *always* need line labels (even if they're empty) when using occurrence id

    @property
    def str(self):
        """
        The Python string representation of this Circuit.

        Returns
        -------
        str
        """
        if self._str is None:
            generated_str = _op_seq_to_str(self._labels, self.line_labels, self._occurrence_id)  # lazy generation
            if self._static:  # if we're read-only then cache the string one and for all,
                self._str = generated_str  # otherwise keep generating it as needed (unless it's set by the user?)
            return generated_str
        else:
            return self._str

    @property
    def layerstr(self):
        """ Just the string representation of the circuit layers (no '@<line_labels>' suffix) """
        return self._labels_lines_str()[0]

    @property
    def linesstr(self):
        """ Just the string representation of the circuit's line labels (the '@<line_labels>' suffix) """
        return self._labels_lines_str()[1]

    def _labels_lines_str(self):
        """ Split the string representation up into layer-labels & line-labels parts """
        if '@' in self.str:
            return self.str.split('@')
        else:
            return self.str, None

    @str.setter
    def str(self, value):
        """
        The Python string representation of this Circuit.

        Returns
        -------
        str
        """
        assert(not self._static), \
            ("Cannot edit a read-only circuit!  "
             "Set editable=True when calling pygsti.obj.Circuit to create editable circuit.")
        cparser = _CircuitParser()
        chk, chk_labels, chk_occurrence = cparser.parse(value)

        if not all([my_layer in (chk_lbl, [chk_lbl]) for chk_lbl, my_layer in zip(chk, self._labels)]):
            raise ValueError(("Cannot set .str to %s because it doesn't"
                              " evaluate to %s (this circuit)") %
                             (value, self.str))
        if chk_labels is not None:
            if tuple(self.line_labels) != chk_labels:
                raise ValueError(("Cannot set .str to %s because line labels evaluate to"
                                  " %s which is != this circuit's line labels (%s).") %
                                 (value, chk_labels, str(self.line_labels)))
        if chk_occurrence is not None:
            if self._occurrence_id != chk_occurrence:
                raise ValueError(("Cannot set .str to %s because occurrence evaluates to"
                                  " %s which is != this circuit's occurrence (%s).") %
                                 (value, str(chk_occurrence), str(self._occurrence_id)))

        self._str = value

    def __hash__(self):
        if not self._static:
            _warnings.warn(("Editable circuit is being converted to read-only"
                            " mode in order to hash it.  You should call"
                            " circuit.done_editing() beforehand."))
            self.done_editing()
        return hash(self.tup)
        #if self._line_labels in (('*',),()): #No line labels
        #    return hash(self._labels)
        #else:
        #    return hash(self._labels + ('@',) + self._line_labels)

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return self._labels.__iter__()

    def __contains__(self, x):
        """Note: this is not covered by __iter__ for case of contained CircuitLabels """
        return any([(x == layer or x in layer) for layer in self._labels])

    def __radd__(self, x):
        if not isinstance(x, Circuit):
            assert(all([isinstance(l, _Label) for l in x])), "Only Circuits and Label-tuples can be added to Circuits!"
            return Circuit._fastinit(x + self.layertup, self.line_labels, editable=False)
        return x.__add__(self)

    def __add__(self, x):
        if not isinstance(x, Circuit):
            assert(all([isinstance(l, _Label) for l in x])), "Only Circuits and Label-tuples can be added to Circuits!"
            return Circuit._fastinit(self.layertup + x, self.line_labels, editable=False)

        if self._str is None or x._str is None:
            s = None
        else:
            mystr, _ = self._labels_lines_str()
            xstr, _ = x._labels_lines_str()

            if mystr != "{}":
                s = (mystr + xstr) if xstr != "{}" else mystr
            else: s = xstr

        added_labels = tuple([l for l in x.line_labels if l not in self.line_labels])
        new_line_labels = self.line_labels + added_labels
        if s is not None:
            s += _op_seq_str_suffix(new_line_labels, occurrence_id=None)  # don't maintain occurrence_id

        return Circuit._fastinit(self.layertup + x.layertup, new_line_labels, editable=False, name='',
                                 stringrep=s, occurrence=None)

    def repeat(self, ntimes, expand="default"):
        """
        Repeat this circuit `ntimes` times.

        Parameters
        ----------
        ntimes : int
            Number of repetitions.

        expand : bool or "default", optional
            When `False`, the returned circuit will contain a :class:`CircuitLabel` encapsulating
            the repetitions without explicitly storing them.  When `True`, the returned circuit will
            be expanded into the `ntimes` repetitions.  `"default"` means to use the value in the
            class variable `Circuit.default_expand_subcircuits`.
        """
        if expand == "default": expand = Circuit.default_expand_subcircuits
        assert((_compat.isint(ntimes) or _np.issubdtype(ntimes, int)) and ntimes >= 0)
        mystr, mylines = self._labels_lines_str()
        if ntimes > 1: s = "(%s)^%d" % (mystr, ntimes)
        elif ntimes == 1: s = "(%s)" % mystr
        else: s = "{}"
        if mylines is not None:
            s += "@" + mylines  # add line labels
        if ntimes >= 1 and expand is False:
            reppedCircuitLbl = self.to_label(nreps=ntimes)
            return Circuit((reppedCircuitLbl,), self.line_labels, None, not self._static, s, check=False)
        else:
            # just adds parens to string rep & copies
            return Circuit(self.layertup * ntimes, self.line_labels, None, not self._static, s, check=False)

    def __mul__(self, x):
        return self.repeat(x)

    def __pow__(self, x):  # same as __mul__()
        return self.__mul__(x)

    def __eq__(self, x):
        if x is None: return False
        if isinstance(x, Circuit):
            return self.tup.__eq__(x.tup)
        else:
            return self.layertup == tuple(x)  # equality with non-circuits is just based on *labels*

    def __lt__(self, x):
        if isinstance(x, Circuit):
            return self.tup.__lt__(x.tup)
        else:
            return self.layertup < tuple(x)  # comparison with non-circuits is just based on *labels*

    def __gt__(self, x):
        if isinstance(x, Circuit):
            return self.tup.__gt__(x.tup)
        else:
            return self.layertup > tuple(x)  # comparison with non-circuits is just based on *labels*

    @property
    def num_lines(self):
        """
        The number of lines in this circuit.

        Returns
        -------
        int
        """
        return len(self.line_labels)

    def copy(self, editable="auto"):
        """
        Returns a copy of the circuit.

        Parameters
        ----------
        editable : {True,False,"auto"}
            Whether returned copy is editable.  If `"auto"` is given,
            then the copy is editable if and only if this Circuit is.

        Returns
        -------
        Circuit
        """
        if editable == "auto": editable = not self._static
        return Circuit(self.layertup, self.line_labels, None, editable, self._str, check=False,
                       occurrence=self.occurrence)

    def clear(self):
        """
        Removes all the gates in a circuit (preserving the number of lines).

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        self._labels = []

    def _proc_layers_arg(self, layers):
        """ Pre-process the layers argument used by many methods """
        if layers is None:
            layers = list(range(len(self._labels)))
        elif isinstance(layers, slice):
            if layers.start is None and layers.stop is None:
                layers = list(range(len(self._labels)))  # e.g. circuit[:]
            else:
                layers = _slct.indices(layers, len(self._labels))
        elif not isinstance(layers, (list, tuple)):
            layers = (layers,)
        return layers

    def _proc_lines_arg(self, lines):
        """ Pre-process the lines argument used by many methods """
        if lines is None:
            lines = self.line_labels
        elif isinstance(lines, slice):
            if lines.start is None and lines.stop is None:
                lines = self.line_labels
            else:
                lines = _slct.indices(lines)
        elif not isinstance(lines, (list, tuple)):
            lines = (lines,)
        return lines

    def _proc_key_arg(self, key):
        """ Pre-process the key argument used by many methods """
        if isinstance(key, tuple):
            if len(key) != 2: return IndexError("Index must be of the form <layerIndex>,<lineIndex>")
            layers = key[0]
            lines = key[1]
        else:
            layers = key
            lines = None
        return layers, lines

    def _layer_components(self, ilayer):
        """ Get the components of the `ilayer`-th layer as a list/tuple. """
        #(works for static and non-static Circuits)
        if self._static:
            if self._labels[ilayer].is_simple(): return [self._labels[ilayer]]
            else: return self._labels[ilayer].components
        else:
            return self._labels[ilayer] if isinstance(self._labels[ilayer], list) \
                else [self._labels[ilayer]]

    def _remove_layer_component(self, ilayer, indx):
        """ Removes the `indx`-th component from the `ilayer`-th layer """
        #(works for special case when layer is just a *single* component)
        assert(not self._static), "Cannot edit a read-only circuit!"
        if isinstance(self._labels[ilayer], list):
            del self._labels[ilayer][indx]
        else:
            assert(indx == 0), "Only index 0 exists for a single-simple-Label level"
            # don't remove *layer* - when final component is removed we're left with an empty layer
            self._labels[ilayer] = []

    def _append_layer_component(self, ilayer, val):
        """ Add `val` to the `ilayer`-th layer """
        #(works for special case when layer is just a *single* component)
        assert(not self._static), "Cannot edit a read-only circuit!"
        if isinstance(self._labels[ilayer], list):
            self._labels[ilayer].append(val)
        else:  # currently ilayer-th layer is a single component!
            self._labels[ilayer] = [self._labels[ilayer], val]

    def _replace_layer_component(self, ilayer, indx, val):
        assert(not self._static), "Cannot edit a read-only circuit!"
        """ Replace `indx`-th component of `ilayer`-th layer with `val` """
        #(works for special case when layer is just a *single* component)
        if isinstance(self._labels[ilayer], list):
            self._labels[ilayer][indx] = val
        else:
            assert(indx == 0), "Only index 0 exists for a single-simple-Label level"
            self._labels[ilayer] = val

    def extract_labels(self, layers=None, lines=None, strict=True):
        """
        Get a subregion - a "rectangle" - of this Circuit.

        This can be used to select multiple layers and/or lines of this Circuit.
        The `strict` argument controls whether gates need to be entirely within
        the given rectangle or can be intersecting it.  If `layers` is a single
        integer then a :class:`Label` is returned (representing a layer or a
        part of a layer), otherwise a :class:`Circuit` is returned.

        Parameters
        ----------
        layers : int, slice, or list/tuple of ints
            Which layers to select (the horizontal dimension of the selection
            rectangle).  Layers are always selected by index, and this
            argument can be a single (integer) index - in which case a `Label`
            is returned - or multiple indices as given by a slice or list -
            in which case a `Circuit` is returned.  Note that, even though
            we speak of a "rectangle", layer indices do not need to be
            contiguous.  The special value `None` selects all layers.

        lines : str/int, slice, or list/tuple of strs/ints
            Which lines to select (the vertical dimension of the selection
            rectangle).  Lines are selected by their line-labels (elements
            of the circuit's `.line_labels` property), which can be strings
            and/or integers.  A single or multiple line-labels can be
            specified.  If the line labels are integers a slice can be used,
            otherwise a list or tuple of labels is the only way to select
            multiple of them.  Note that line-labels do not need to be
            contiguous. The special value `None` selects all lines.

        strict : bool, optional
            When `True`, only gates lying completely within the selected
            region are included in the return value.  If a gate straddles
            the region boundary (e.g. if we select just line `1` and the
            circuit contains `"Gcnot:1:2"`) then it is *silently* not-included
            in the returned label or circuit.  If `False`, then gates which
            straddle the region boundary *are* included.  Note that this may
            result in a `Label` or `Circuit` containing more line labels than
            where requested in the call to `extract_labels(...)`..

        Returns
        -------
        Label or Circuit
            The requested portion of this circuit, given as a `Label` if
            `layers` is a single integer and as a `Circuit` otherwise.
            Note: if you want a `Circuit` when only selecting one layer,
            set `layers` to a slice or tuple containing just a single index.
        """
        nonint_layers = not isinstance(layers, int)

        #Shortcut for common case when lines == None and when we're only taking a layer slice/index
        if lines is None:
            assert(layers is not None)
            if nonint_layers is False: return self.layertup[layers]
            if isinstance(layers, slice) and strict is True:  # if strict=False, then need to recompute line labels
                return Circuit._fastinit(self._labels[layers], self.line_labels, not self._static)

        layers = self._proc_layers_arg(layers)
        lines = self._proc_lines_arg(lines)
        if len(layers) == 0 or len(lines) == 0:
            return Circuit._fastinit(() if self._static else [],
                                     lines, not self._static) if nonint_layers else None  # zero-area region

        ret = []
        if self._static:
            def get_sslbls(lbl): return lbl.sslbls
        else:
            get_sslbls = _sslbls_of_nested_lists_of_simple_labels

        for i in layers:
            ret_layer = []
            for l in self._layer_components(i):  # loop over labels in this layer
                sslbls = get_sslbls(l)
                if sslbls is None:
                    ## add in special case of identity layer
                    #if (isinstance(l,_Label) and l.name == self.identity): # ~ is_identity_layer(l)
                    #    ret_layer.append(l); continue
                    sslbls = set(self.line_labels)  # otherwise, treat None sslbs as *all* labels
                else:
                    sslbls = set(sslbls)
                if (strict and sslbls.issubset(lines)) or \
                   (not strict and len(sslbls.intersection(lines)) >= 0):
                    ret_layer.append(l)
            ret.append(_Label(ret_layer) if len(ret_layer) != 1 else ret_layer[0])  # Labels b/c we use _fastinit

        if nonint_layers:
            if not strict: lines = "auto"  # since we may have included lbls on other lines
            # don't worry about string rep for now...
            return Circuit._fastinit(tuple(ret) if self._static else ret, lines, not self._static)
        else:
            return _Label(ret[0])

    def set_labels(self, lbls, layers=None, lines=None):
        """
        Write `lbls` to the block defined by the `layers` and `lines` arguments.

        Note that `lbls` can be anything interpretable as a :class:`Label`
        or list of labels.

        Parameters
        ----------
        lbls : Label, list/tuple of Labels, or Circuit
            When `layers` is a single integer, `lbls` should be a single
            "layer label" of type `Label`.  Otherwise, `lbls` should be
            a list or tuple of `Label` objects with length equal to the
            number of layers being set.  A `Circuit` may also be used in this
            case.

        layers : int, slice, or list/tuple of ints
            Which layers to set (the horizontal dimension of the destination
            rectangle).  Layers are always selected by index, and this
            argument can be a single (integer) index  or multiple indices as
            given by a slice or list.  Note that these indices do not need to be
            contiguous.  The special value `None` stands for all layers.

        lines : str/int, slice, or list/tuple of strs/ints
            Which lines to set (the vertical dimension of the destination
            rectangle).  Lines are selected by their line-labels, which can be
            strings and/or integers.  A single or multiple line-labels can be
            specified.  If the line labels are integers a slice can be used,
            otherwise a list or tuple of labels is the only way to specify
            multiple of them.  The line-labels do not need to be contiguous.
            The special value `None` stands for all lines, and in this case
            new lines will be created if there are new state-space labels
            in `lbls` (when `lines` is not `None` an error is raised instead).

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        #Note: this means self._labels contains nested lists of simple labels

        #Convert layers to a list/tuple of layer indices
        all_layers = bool(layers is None)  # whether we're assigning to *all* layers
        int_layers = isinstance(layers, int)
        layers = self._proc_layers_arg(layers)

        #Convert lines to a list/tuple of line (state space) labels
        all_lines = bool(lines is None)  # whether we're assigning to *all* lines
        lines = self._proc_lines_arg(lines)

        #make lbls into either:
        # 1) a single Label (possibly compound) if layers is an int
        # 2) a tuple of Labels (possibly compound) otherwise
        if int_layers:
            if isinstance(lbls, Circuit):  # special case: "box" a circuit assigned to a single layer
                lbls = lbls.to_label()     # converts Circuit => CircuitLabel
            lbls = to_label(lbls)
            lbls_sslbls = None if (lbls.sslbls is None) else set(lbls.sslbls)
        else:
            if isinstance(lbls, Circuit):
                assert(set(lbls.line_labels).issubset(self.line_labels)), \
                    "Assigned circuit has lines (%s) not contained in this circuit (%s)!" \
                    % (str(lbls.line_labels), str(self.line_labels))
                lbls = lbls.layertup  # circuit layer labels as a tuple
            assert(isinstance(lbls, (tuple, list))), \
                ("When assigning to a layer range (even w/len=1) `lbls` "
                 "must be  a *list or tuple* of label-like items")
            lbls = tuple(map(to_label, lbls))
            lbls_sslbls = None if any([l.sslbls is None for l in lbls]) \
                else set(_itertools.chain(*[l.sslbls for l in lbls]))

        if len(layers) == 0 or len(lines) == 0: return  # zero-area block

        #If we're assigning to multiple layers, then divide up lbls into pieces to place in each layer
        if all_layers:  # then we'll add new layers as needed
            while len(lbls) > len(self._labels):
                self._labels.append([])
        elif len(layers) > 1:
            assert(len(layers) == len(lbls)), \
                "Block width mismatch: assigning %d layers to %d layers" % (len(lbls), len(layers))

        # when processing `lbls`: if a label has sslbls == None, then applies to all
        # the lines being assigned.  If sslbl != None, then the labels must be
        # contained within the line labels being assigned (unless we're allowed to expand)
        if lbls_sslbls is not None:
            new_line_labels = set(lbls_sslbls) - set(self.line_labels)
            if all_lines:  # then allow new lines to be added
                if len(new_line_labels) > 0:
                    self._line_labels = self.line_labels + tuple(sorted(new_line_labels))  # sort?
            else:
                assert(len(new_line_labels) == 0), "Cannot add new lines %s" % str(new_line_labels)
                assert(set(lbls_sslbls).issubset(lines)), \
                    "Unallowed state space labels: %s" % str(set(lbls_sslbls) - set(lines))

        assert(set(lines).issubset(self.line_labels)), \
            ("Specified lines (%s) must be a subset of this circuit's lines"
             " (%s).") % (str(lines), str(self.line_labels))

        #remove all labels in block to be assigned
        self._clear_labels(layers, lines)

        def_sslbls = None if all_lines else lines
        if not int_layers:
            for i, lbls_comp in zip(layers, lbls):
                self._labels[i].extend(_label_to_nested_lists_of_simple_labels(lbls_comp, def_sslbls))
        else:  # single layer using integer layer index (so lbls is a single Label)
            self._labels[layers[0]].extend(_label_to_nested_lists_of_simple_labels(lbls, def_sslbls))

    def insert_idling_layers(self, insert_before, num_to_insert, lines=None):
        """
        Inserts into this circuit one or more idling (blank) layers.

        By default, complete layer(s) are inserted.  The `lines` argument
        allows you to insert partial layers (on only a subset of the lines).

        Parameters
        ----------
        insert_before : int
            The layer index to insert the new layers before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The special value `None` inserts at the end.

        num_to_insert : int
            The number of new layers to insert.

        lines : str/int, slice, or list/tuple of strs/ints, optional
            Which lines should have new layers (blank circuit space)
            inserted into them.  A single or multiple line-labels can be
            specified, similarly as in :method:`extract_labels`.  The default
            value `None` stands for *all* lines.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.insert_idling_layers_inplace(insert_before, num_to_insert, lines)
        cpy.done_editing()
        return cpy

    def insert_idling_layers_inplace(self, insert_before, num_to_insert, lines=None):
        """
        Inserts into this circuit one or more idling (blank) layers.

        By default, complete layer(s) are inserted.  The `lines` argument
        allows you to insert partial layers (on only a subset of the lines).

        Parameters
        ----------
        insert_before : int
            The layer index to insert the new layers before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The special value `None` inserts at the end.

        num_to_insert : int
            The number of new layers to insert.

        lines : str/int, slice, or list/tuple of strs/ints, optional
            Which lines should have new layers (blank circuit space)
            inserted into them.  A single or multiple line-labels can be
            specified, similarly as in :method:`extract_labels`.  The default
            value `None` stands for *all* lines.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        if insert_before is None: insert_before = len(self._labels)
        elif insert_before < 0: insert_before = len(self._labels) + insert_before

        if lines is None:  # insert complete layers
            for i in range(num_to_insert):
                self._labels.insert(insert_before, [])
        else:  # insert layers only on given lines - shift existing labels to right
            for i in range(num_to_insert):
                self._labels.append([])  # add blank layers at end
            for i in range(insert_before, insert_before + num_to_insert):
                # move labels on `lines` to layer i+num_to_insert
                inds_to_delete = []
                for k, lbl in enumerate(self._labels[i]):
                    sslbls = _sslbls_of_nested_lists_of_simple_labels(lbl)
                    if len(sslbls.intersection(lines)) > 0:  # then we need to move this label
                        if not sslbls.issubset(lines):
                            raise ValueError("Cannot shift a block that is straddled by %s!" % _Label(lbl))
                            #FUTURE: recover from this error gracefully so we don't leave the circuit in an intermediate
                            #state?
                        inds_to_delete.append(k)  # remove it from current layer
                        self._labels[i + num_to_insert].append(lbl)  # and put it in the destination layer
                for k in reversed(inds_to_delete):
                    del self._labels[i][k]

    def _append_idling_layers_inplace(self, num_to_insert, lines=None):
        """
        Adds one or more idling (blank) layers to the end of this circuit.

        By default, complete layer(s) are appended.  The `lines` argument
        allows you to add partial layers (on only a subset of the lines).

        Parameters
        ----------
        num_to_insert : int
            The number of new layers to append.

        lines : str/int, slice, or list/tuple of strs/ints, optional
            Which lines should have new layers (blank circuit space)
            inserted into them.  A single or multiple line-labels can be
            specified, similarly as in :method:`extract_labels`.  The default
            value `None` stands for *all* lines.

        Returns
        -------
        None
        """
        self.insert_idling_layers_inplace(None, num_to_insert, lines)

    def insert_labels_into_layers(self, lbls, layer_to_insert_before, lines=None):
        """
        Inserts into this circuit the contents of `lbls` into new full or partial layers.

        By default, complete layer(s) are inserted.  The `lines` argument
        allows you to insert partial layers (on only a subset of the lines).

        Parameters
        ----------
        lbls : list/tuple of Labels, or Circuit
            The full or partial layer labels to insert.  The length of this
            list, tuple, or circuit determines the number of layers which are
            inserted.

        layer_to_insert_before : int
            The layer index to insert `lbls` before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The special value `None` inserts at the end.

        lines : str/int, slice, or list/tuple of strs/ints, optional
            Which lines should have `lbls` inserted into them.  Currently
            this can only be a larger set than the set of line labels present
            in `lbls` (in future versions this may allow filtering of `lbls`).
            value `None` stands for *all* lines.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.insert_labels_into_layers_inplace(lbls, layer_to_insert_before, lines)
        cpy.done_editing()
        return cpy

    def insert_labels_into_layers_inplace(self, lbls, layer_to_insert_before, lines=None):
        """
        Inserts into this circuit the contents of `lbls` into new full or partial layers.

        By default, complete layer(s) are inserted.  The `lines` argument
        allows you to insert partial layers (on only a subset of the lines).

        Parameters
        ----------
        lbls : list/tuple of Labels, or Circuit
            The full or partial layer labels to insert.  The length of this
            list, tuple, or circuit determines the number of layers which are
            inserted.

        layer_to_insert_before : int
            The layer index to insert `lbls` before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The special value `None` inserts at the end.

        lines : str/int, slice, or list/tuple of strs/ints, optional
            Which lines should have `lbls` inserted into them.  Currently
            this can only be a larger set than the set of line labels present
            in `lbls` (in future versions this may allow filtering of `lbls`).
            value `None` stands for *all* lines.

        Returns
        -------
        None
        """
        if isinstance(lbls, Circuit): lbls = tuple(lbls)
        # lbls is expected to be a list/tuple of Label-like items, one per inserted layer
        lbls = tuple(map(to_label, lbls))
        numLayersToInsert = len(lbls)
        self.insert_idling_layers_inplace(layer_to_insert_before, numLayersToInsert, lines)  # make space
        self.set_labels(lbls, slice(layer_to_insert_before, layer_to_insert_before + numLayersToInsert), lines)
        #Note: set_labels expects lbls to be a list/tuple of Label-like items b/c it's given a layer *slice*

    def insert_idling_lines(self, insert_before, line_labels):
        """
        Insert one or more idling (blank) lines into this circuit.

        Parameters
        ----------
        insert_before : str or int
            The line label to insert new lines before.  The special value `None`
            inserts lines at the bottom of this circuit.

        line_labels : list or tuple
            A list or tuple of the new line labels to insert (can be integers
            and/or strings).

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.insert_idling_lines_inplace(insert_before, line_labels)
        cpy.done_editing()
        return cpy

    def insert_idling_lines_inplace(self, insert_before, line_labels):
        """
        Insert one or more idling (blank) lines into this circuit.

        Parameters
        ----------
        insert_before : str or int
            The line label to insert new lines before.  The special value `None`
            inserts lines at the bottom of this circuit.

        line_labels : list or tuple
            A list or tuple of the new line labels to insert (can be integers
            and/or strings).

        Returns
        -------
        None
        """
        #assert(not self._static),"Cannot edit a read-only circuit!"
        # Actually, this is OK even for static circuits because it won't affect the hashed value (labels only)
        if insert_before is None:
            i = len(self.line_labels)
        else:
            i = self.line_labels.index(insert_before)
        self._line_labels = self.line_labels[0:i] + tuple(line_labels) + self.line_labels[i:]

    def _append_idling_lines(self, line_labels):
        """
        Add one or more idling (blank) lines onto the bottom of this circuit.

        Parameters
        ----------
        line_labels : list or tuple
            A list or tuple of the new line labels to insert (can be integers
            and/or strings).

        Returns
        -------
        None
        """
        self.insert_idling_lines_inplace(None, line_labels)

    def insert_labels_as_lines_inplace(self, lbls, layer_to_insert_before=None, line_to_insert_before=None,
                                       line_labels="auto"):
        """
        Inserts into this circuit the contents of `lbls` into new lines.

        By default, `lbls` is inserted at the beginning of the new lines(s). The
        `layer_to_insert_before` argument allows you to insert `lbls` beginning at
        a layer of your choice.

        Parameters
        ----------
        lbls : list/tuple of Labels, or Circuit
            A list of layer labels to insert as new lines.  The state-space
            (line) labels within `lbls` must not overlap with that of this
            circuit or an error is raised.  If `lbls` contains more layers
            than this circuit currently has, new layers are added automatically.

        layer_to_insert_before : int
            The layer index to insert `lbls` before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The default value of `None` inserts at the beginning.

        line_to_insert_before : str or int
            The line label to insert the new lines before.  The default value
            of `None` inserts lines at the bottom of the circuit.

        line_labels : list, tuple, or "auto"
            The labels of the new lines being inserted.  If `"auto"`, then
            these are inferred from `lbls`.

        Returns
        -------
        None
        """
        if layer_to_insert_before is None: layer_to_insert_before = 0
        elif layer_to_insert_before < 0: layer_to_insert_before = len(self._labels) + layer_to_insert_before

        if isinstance(lbls, Circuit):
            if line_labels == "auto": line_labels = lbls.line_labels
            lbls = lbls.tup
        elif line_labels == "auto":
            line_labels = tuple(sorted(_accumulate_explicit_sslbls(lbls)))

        existing_labels = set(line_labels).intersection(self.line_labels)
        if len(existing_labels) > 0:
            raise ValueError("Cannot insert line(s) labeled %s - they already exist!" % str(existing_labels))

        self.insert_idling_lines_inplace(line_to_insert_before, line_labels)

        #add additional layers to end of circuit if new lines are longer than current circuit depth
        numLayersToInsert = len(lbls)
        if layer_to_insert_before + numLayersToInsert > len(self._labels):
            self._append_idling_layers_inplace(layer_to_insert_before + numLayersToInsert - len(self._labels))

        #Note: set_labels expects lbls to be a list/tuple of Label-like items b/c it's given a layer *slice*
        self.set_labels(lbls, slice(layer_to_insert_before, layer_to_insert_before + numLayersToInsert), line_labels)

    def insert_labels_as_lines(self, lbls, layer_to_insert_before=None, line_to_insert_before=None, line_labels="auto"):
        """
        Inserts into this circuit the contents of `lbls` into new lines.

        By default, `lbls` is inserted at the beginning of the new lines(s). The
        `layer_to_insert_before` argument allows you to insert `lbls` beginning at
        a layer of your choice.

        Parameters
        ----------
        lbls : list/tuple of Labels, or Circuit
            A list of layer labels to insert as new lines.  The state-space
            (line) labels within `lbls` must not overlap with that of this
            circuit or an error is raised.  If `lbls` contains more layers
            than this circuit currently has, new layers are added automatically.

        layer_to_insert_before : int
            The layer index to insert `lbls` before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The default value of `None` inserts at the beginning.

        line_to_insert_before : str or int
            The line label to insert the new lines before.  The default value
            of `None` inserts lines at the bottom of the circuit.

        line_labels : list, tuple, or "auto"
            The labels of the new lines being inserted.  If `"auto"`, then
            these are inferred from `lbls`.

        Returns
        -------
        None
        """
        cpy = self.copy(editable=True)
        cpy.insert_labels_as_lines_inplace(lbls, layer_to_insert_before, line_to_insert_before, line_labels)
        cpy.done_editing()
        return cpy

    def _append_labels_as_lines(self, lbls, layer_to_insert_before=None, line_labels="auto"):
        """
        Adds the contents of `lbls` as new lines at the bottom of this circuit.

        By default, `lbls` is inserted at the beginning of the new lines(s). The
        `layer_to_insert_before` argument allows you to insert `lbls` beginning at
        a layer of your choice.

        Parameters
        ----------
        lbls : list/tuple of Labels, or Circuit
            A list of layer labels to append as new lines.  The state-space
            (line) labels within `lbls` must not overlap with that of this
            circuit or an error is raised.  If `lbls` contains more layers
            than this circuit currently has, new layers are added automatically.

        layer_to_insert_before : int
            The layer index to insert `lbls` before.  Can be from 0
            (insert at the beginning) to `len(self)-1` (insert at end), and
            negative indexing can be used to insert relative to the last layer.
            The default value of `None` inserts at the beginning.

        line_labels : list, tuple, or "auto"
            The labels of the new lines being added.  If `"auto"`, then
            these are inferred from `lbls`.

        Returns
        -------
        None
        """
        return self.insert_labels_as_lines(lbls, layer_to_insert_before, None, line_labels)

    def _clear_labels(self, layers, lines, clear_straddlers=False):
        """ remove all labels in a block given by layers and lines
            Note: layers & lines must be lists/tuples of values; they can't be slices or single vals
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        for i in layers:
            new_layer = []
            for l in self._layer_components(i):  # loop over labels in this layer
                sslbls = _sslbls_of_nested_lists_of_simple_labels(l)
                sslbls = set(self.line_labels) if (sslbls is None) else set(sslbls)
                if len(sslbls.intersection(lines)) == 0:
                    new_layer.append(l)
                elif not clear_straddlers and not sslbls.issubset(lines):
                    raise ValueError("Cannot operate on a block that is straddled by %s!" % str(_Label(l)))
            self._labels[i] = new_layer

    def clear_labels(self, layers=None, lines=None, clear_straddlers=False):
        """
        Removes all the gates within the given circuit region.  Does not reduce the number of layers or lines.

        Parameters
        ----------
        layers : int, slice, or list/tuple of ints
            Defines the horizontal dimension of the region to clear.  See
            :method:`extract_labels` for details.

        lines : str/int, slice, or list/tuple of strs/ints
            Defines the vertical dimension of the region to clear.  See
            :method:`extract_labels` for details.

        clear_straddlers : bool, optional
            Whether or not gates which straddle cleared and non-cleared lines
            should be cleared.  If `False` and straddling gates exist, an error
            will be raised.

        Returns
        -------
        None
        """
        layers = self._proc_layers_arg(layers)
        lines = self._proc_lines_arg(lines)
        self._clear_labels(layers, lines, clear_straddlers)

    def delete_layers(self, layers=None):
        """
        Deletes one or more layers from the circuit.

        Parameters
        ----------
        layers : int, slice, or list/tuple of ints
            The layer index or indices to delete.  See :method:`extract_labels`
            for details.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        layers = self._proc_layers_arg(layers)
        for i in reversed(sorted(layers)):
            del self._labels[i]

    def delete_lines(self, lines, delete_straddlers=False):
        """
        Deletes one or more lines from the circuit.

        Parameters
        ----------
        lines : str/int, slice, or list/tuple of strs/ints
            The line label(s) to delete.  See :method:`extract_labels` for details.

        delete_straddlers : bool, optional
            Whether or not gates which straddle deleted and non-deleted lines
            should be removed.  If `False` and straddling gates exist, an error
            will be raised.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        lines = self._proc_lines_arg(lines)
        for i in range(len(self._labels)):
            new_layer = []
            for l in self._layer_components(i):  # loop over labels in this layer
                sslbls = _sslbls_of_nested_lists_of_simple_labels(l)
                if sslbls is None or len(set(sslbls).intersection(lines)) == 0:
                    new_layer.append(l)
                elif not delete_straddlers and not set(sslbls).issubset(lines):
                    raise ValueError(("Cannot remove a block that is straddled by "
                                      "%s when `delete_straddlers` == False!") % _Label(l))
            self._labels[i] = new_layer
        self._line_labels = tuple([x for x in self.line_labels if x not in lines])

    def __getitem__(self, key):
        layers, lines = self._proc_key_arg(key)
        return self.extract_labels(layers, lines, strict=True)

    def __setitem__(self, key, val):
        layers, lines = self._proc_key_arg(key)
        return self.set_labels(val, layers, lines)

    def __delitem__(self, key):
        layers, lines = self._proc_key_arg(key)
        if layers is None:
            self.delete_lines(lines, delete_straddlers=True)
        elif lines is None:
            self.delete_layers(layers)
        else:
            raise IndexError("Can only delete entire layers or enire lines.")

    def to_pythonstr(self, op_labels):
        """
        Convert this circuit to an "encoded" python string.

        In the returned string each operation label is represented as a
        **single** character, starting with 'A' and continuing down the alphabet.
        This can be useful for processing operation sequences using python's
        string tools (regex in particular).

        Parameters
        ----------
        op_labels : tuple
            An iterable containing at least all the layer-Labels that appear
            in this Circuit, and which will be mapped to alphabet
            characters, beginning with 'A'.

        Returns
        -------
        string
            The converted operation sequence.

        Examples
        --------
        ('Gx','Gx','Gy','Gx') => "AABA"
        """
        assert(len(op_labels) < 26)  # Complain if we go beyond 'Z'
        translateDict = {}; c = 'A'
        for opLabel in op_labels:
            translateDict[opLabel] = c
            c = chr(ord(c) + 1)
        return "".join([translateDict[opLabel] for opLabel in self.layertup])

    @classmethod
    def from_pythonstr(cls, python_string, op_labels):
        """
        Decode an "encoded string" into a :class:`Circuit`.

        Create a Circuit from a python string where each operation label is
        represented as a **single** character, starting with 'A' and continuing
        down the alphabet.  This performs the inverse of :method:`to_pythonstr`.

        Parameters
        ----------
        python_string : string
            string whose individual characters correspond to the operation labels of a
            operation sequence.

        op_labels : tuple
            tuple containing all the operation labels that will be mapped from alphabet
            characters, beginning with 'A'.

        Returns
        -------
        Circuit

        Examples
        --------
        "AABA" => ('Gx','Gx','Gy','Gx')
        """
        assert(len(op_labels) < 26)  # Complain if we go beyond 'Z'
        translateDict = {}; c = 'A'
        for opLabel in op_labels:
            translateDict[c] = opLabel
            c = chr(ord(c) + 1)
        return cls(tuple([translateDict[cc] for cc in python_string]))

    def serialize(self):
        """
        Serialize the parallel gate operations of this Circuit.

        Construct a new Circuit whereby all layers containing multiple gates are
        converted to separate single-gate layers, effectively putting each
        elementary gate operation into its own layer.  Ordering is dictated by
        the ordering of the compound layer labels.

        Returns
        -------
        Circuit
        """
        serial_lbls = []
        for lbl in self.layertup:
            if len(lbl.components) == 0:  # special case of an empty-layer label,
                serial_lbls.append(lbl)  # which we serialize as an atomic object
            for c in lbl.components:
                serial_lbls.append(c)
        return Circuit._fastinit(tuple(serial_lbls), self.line_labels, editable=False, occurrence=self.occurrence)

    def parallelize(self, can_break_labels=True, adjacent_only=False):
        """
        Compress a circuit's gates by performing them in parallel.

        Construct a circuit with the same underlying labels as this one,
        but with as many gates performed in parallel as possible (with
        some restrictions - see the Parameters section below).  Generally,
        gates are moved as far left (toward the start) of the circuit as
        possible.

        Parameters
        ----------
        can_break_labels : bool, optional
            Whether compound (parallel-gate) labels in this Circuit can be
            separated during the parallelization process.  For example, if
            `can_break_labels=True` then `"Gx:0[Gy:0Gy:1]" => "[Gx:0Gy:1]Gy:0"`
            whereas if `can_break_labels=False` the result would remain
            `"Gx:0[Gy:0Gy:1]"` because `[Gy:0Gy:1]` cannot be separated.

        adjacent_only : bool, optional
            It `True`, then operation labels are only allowed to move into an
            adjacent label, that is, they cannot move "through" other
            operation labels.  For example, if `adjacent_only=True` then
            `"Gx:0Gy:0Gy:1" => "Gx:0[Gy:0Gy:1]"` whereas if
            `adjacent_only=False` the result would be `"[Gx:0Gy:1]Gy:0`.
            Setting this to `True` is sometimes useful if you want to
            parallelize a serial string in such a way that subsequently
            calling `.serialize()` will give you back the original string.

        Returns
        -------
        Circuit
        """
        parallel_lbls = []
        first_free = {'*': 0}
        for lbl in self.layertup:
            if can_break_labels:  # then process label components individually
                for c in lbl.components:
                    if c.sslbls is None:  # ~= acts on *all* sslbls
                        pos = max(list(first_free.values()))
                        #first position where all sslbls are free
                    else:
                        inds = [v for k, v in first_free.items() if k in c.sslbls]
                        pos = max(inds) if len(inds) > 0 else first_free['*']
                        #first position where all c.sslbls are free (uses special
                        # '*' "base" key if we haven't seen any of the sslbls yet)

                    if len(parallel_lbls) < pos + 1: parallel_lbls.append([])
                    assert(pos < len(parallel_lbls))
                    parallel_lbls[pos].append(c)  # add component in proper place

                    #update first_free
                    if adjacent_only:  # all labels/components following this one must at least be at 'pos'
                        for k in first_free: first_free[k] = pos
                    if c.sslbls is None:
                        for k in first_free: first_free[k] = pos + 1  # includes '*'
                    else:
                        for k in c.sslbls: first_free[k] = pos + 1

            else:  # can't break labels - treat as a whole
                if lbl.sslbls is None:  # ~= acts on *all* sslbls
                    pos = max(list(first_free.values()))
                    #first position where all sslbls are free
                else:
                    inds = [v for k, v in first_free.items() if k in lbl.sslbls]
                    pos = max(inds) if len(inds) > 0 else first_free['*']
                    #first position where all c.sslbls are free (uses special
                    # '*' "base" key if we haven't seen any of the sslbls yet)

                if len(parallel_lbls) < pos + 1: parallel_lbls.append([])
                assert(pos < len(parallel_lbls))
                for c in lbl.components:  # add *all* components of lbl in proper place
                    parallel_lbls[pos].append(c)

                #update first_free
                if adjacent_only:  # all labels/components following this one must at least be at 'pos'
                    for k in first_free: first_free[k] = pos
                if lbl.sslbls is None:
                    for k in first_free: first_free[k] = pos + 1  # includes '*'
                else:
                    for k in lbl.sslbls: first_free[k] = pos + 1

        # Convert elements of `parallel_lbls` into Labels (needed b/c we use _fastinit below)
        parallel_lbls = [_Label(lbl_list) if len(lbl_list) != 1 else lbl_list[0] for lbl_list in parallel_lbls]
        return Circuit._fastinit(tuple(parallel_lbls), self.line_labels, editable=False, occurrence=self.occurrence)

    def expand_subcircuits_inplace(self):
        """
        Expands all :class:`CircuitLabel` labels within this circuit.

        This operation is done in place and so can only be performed
        on an editable :class:`Circuit`.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        #Iterate in reverse so we don't have to deal with
        # added layers.
        for i in reversed(range(len(self._labels))):
            circuits_to_expand = []
            layers_to_add = 0

            for l in self._layer_components(i):  # loop over labels in this layer
                if isinstance(l, _CircuitLabel):
                    circuits_to_expand.append(l)
                    layers_to_add = max(layers_to_add, l.depth - 1)

            if layers_to_add > 0:
                self.insert_idling_layers_inplace(i + 1, layers_to_add)
            for subc in circuits_to_expand:
                self.clear_labels(slice(i, i + subc.depth), subc.sslbls)  # remove the CircuitLabel
                self.set_labels(subc.components * subc.reps, slice(i, i + subc.depth),
                                subc.sslbls)  # dump in the contents

    def expand_subcircuits(self):
        """
        Returns a new circuit with :class:`CircuitLabel` labels expanded.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.expand_subcircuits_inplace()
        cpy.done_editing()
        return cpy

    def factorize_repetitions_inplace(self):
        """
        Attempt to replace repeated sub-circuits with :class:`CircuitLabel` objects.

        More or less the reverse of :method:`expand_subcircuits`, this method
        attempts to collapse repetitions of the same labels into single
        :class:`CircuitLabel` labels within this circuit.

        This operation is done in place and so can only be performed
        on an editable :class:`Circuit`.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        nLayers = self.num_layers
        iLayersToRemove = []
        iStart = 0
        while iStart < nLayers - 1:
            iEnd = iStart + 1
            while iEnd < nLayers and self._labels[iStart] == self._labels[iEnd]:
                iEnd += 1
            nreps = iEnd - iStart
            #print("Start,End = ",iStart,iEnd)
            if nreps <= 1:  # just move to next layer
                iStart += 1; continue  # nothing to do

            #Construct a sub-circuit label that repeats layer[iStart] nreps times
            # and stick it at layer iStart
            #print("Constructing %d reps at %d" % (nreps, iStart))
            repCircuit = _CircuitLabel('', self._labels[iStart], None, nreps)
            self.clear_labels(iStart, None)  # remove existing labels (unnecessary?)
            self.set_labels(repCircuit, iStart, None)
            iLayersToRemove.extend(list(range(iStart + 1, iEnd)))
            iStart += nreps  # advance iStart to next unprocessed layer inde

        if len(iLayersToRemove) > 0:
            #print("Removing layers: ",iLayersToRemove)
            self.delete_layers(iLayersToRemove)

    def insert_layer(self, circuit_layer, j):
        """
        Inserts a single layer into a circuit.

        The input layer does not need to contain a gate that acts on
        every qubit, but it should not contain more than one gate on
        a qubit.

        Parameters
        ----------
        circuit_layer : Label
            The layer to insert.  A (possibly compound) Label object or
            something that can be converted into one, e.g.
            `(('Gx',0),('Gcnot',1,2))` or just `'Gx'`.

        j : int
            The layer index (depth) at which to insert the `circuit_layer`.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.insert_layer_inplace(circuit_layer, j)
        cpy.done_editing()
        return cpy

    def insert_layer_inplace(self, circuit_layer, j):
        """
        Inserts a single layer into a circuit.

        The input layer does not need to contain a gate that acts on
        every qubit, but it should not contain more than one gate on
        a qubit.

        Parameters
        ----------
        circuit_layer : Label
            The layer to insert.  A (possibly compound) Label object or
            something that can be converted into one, e.g.
            `(('Gx',0),('Gcnot',1,2))` or just `'Gx'`.

        j : int
            The layer index (depth) at which to insert the `circuit_layer`.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        if self.line_labels is None or self.line_labels == ():
            #Allow insertion of a layer into an empty circuit to update the circuit's line_labels
            layer_lbl = to_label(circuit_layer)
            self.line_labels = layer_lbl.sslbls if (layer_lbl.sslbls is not None) else ('*',)

        self.insert_labels_into_layers_inplace([circuit_layer], j)

    def insert_circuit(self, circuit, j):
        """
        Inserts a circuit into this circuit.

        The circuit to insert can be over more qubits than this circuit, as long
        as all qubits that are not part of this circuit are idling. In this
        case, the idling qubits are all discarded. The circuit to insert can
        also be on less qubits than this circuit: all other qubits are set to
        idling. So, the labels of the circuit to insert for all non-idling
        qubits must be a subset of the labels of this circuit.

        Parameters
        ----------
        circuit : Circuit
            The circuit to be inserted.

        j : int
            The layer index (depth) at which to insert the circuit.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.insert_circuit_inplace(circuit, j)
        cpy.done_editing()
        return cpy

    def insert_circuit_inplace(self, circuit, j):
        """
        Inserts a circuit into this circuit.

        The circuit to insert can be over more qubits than this circuit, as long
        as all qubits that are not part of this circuit are idling. In this
        case, the idling qubits are all discarded. The circuit to insert can
        also be on less qubits than this circuit: all other qubits are set to
        idling. So, the labels of the circuit to insert for all non-idling
        qubits must be a subset of the labels of this circuit.

        Parameters
        ----------
        circuit : Circuit
            The circuit to be inserted.

        j : int
            The layer index (depth) at which to insert the circuit.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        lines_to_insert = []
        for line_lbl in circuit.line_labels:
            if line_lbl in self.line_labels:
                lines_to_insert.append(line_lbl)
            else:
                assert(circuit._is_line_idling(line_lbl)), \
                    "There are non-idling lines in the circuit to insert that are *not* lines in this circuit!"

        labels_to_insert = circuit.extract_labels(layers=None, lines=lines_to_insert)
        self.insert_labels_into_layers_inplace(labels_to_insert, j)

    def append_circuit(self, circuit):
        """
        Append a circuit to the end of this circuit.

        This circuit must satisfy the requirements of
        :method:`insert_circuit()`. See that method for more details.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be appended.

        Returns
        -------
        Circuit
        """
        self.insert_circuit(circuit, self.num_layers)

    def append_circuit_inplace(self, circuit):
        """
        Append a circuit to the end of this circuit.

        This circuit must satisfy the requirements of
        :method:`insert_circuit()`. See that method for more details.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be appended.

        Returns
        -------
        None
        """
        self.insert_circuit_inplace(circuit, self.num_layers)

    def prefix_circuit(self, circuit):
        """
        Prefix a circuit to the beginning of this circuit.

        This circuit must satisfy the requirements of the
        :method:`insert_circuit()`. See that method for more details.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.

        Returns
        -------
        Circuit
        """
        self.insert_circuit(circuit, 0)

    def prefix_circuit_inplace(self, circuit):
        """
        Prefix a circuit to the beginning of this circuit.

        This circuit must satisfy the requirements of the
        :method:`insert_circuit()`. See that method for more details.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.

        Returns
        -------
        None
        """
        self.insert_circuit_inplace(circuit, 0)

    def tensor_circuit_inplace(self, circuit, line_order=None):
        """
        The tensor product of this circuit and `circuit`.

        That is, it adds `circuit` to this circuit as new lines.  The line
        labels of `circuit` must be disjoint from the line labels of this
        circuit, as otherwise applying the circuits in parallel does not make
        sense.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be tensored.

        line_order : List, optional
            A list of all the line labels specifying the order of the circuit in the updated
            circuit. If None, the lines of `circuit` are added below the lines of this circuit.
            Note that, for many purposes, the ordering of lines of the circuit is irrelevant.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        #assert(self.identity == circuit.identity), "The identity labels must be the same!"

        #Construct new line labels (of final circuit)
        overlap = set(self.line_labels).intersection(circuit.line_labels)
        if len(overlap) > 0:
            raise ValueError(
                "The line labels of `circuit` and this Circuit must be distinct, but overlap = %s!" % str(overlap))

        all_line_labels = set(self.line_labels + circuit.line_labels)
        if line_order is not None:
            line_order_set = set(line_order)
            if len(line_order_set) != len(line_order):
                raise ValueError("`line_order` == %s cannot contain duplicates!" % str(line_order))

            missing = all_line_labels - line_order_set
            if len(missing) > 0:
                raise ValueError("`line_order` is missing %s." % str(missing))

            extra = set(line_order) - all_line_labels
            if len(extra) > 0:
                raise ValueError("`line_order` had nonpresent line labels %s." % str(extra))

            new_line_labels = line_order
        else:
            new_line_labels = self.line_labels + circuit.line_labels

        #Add circuit's labels into this circuit
        self.insert_labels_as_lines_inplace(circuit._labels, line_labels=circuit.line_labels)
        self._line_labels = new_line_labels  # essentially just reorders labels if needed

    def tensor_circuit(self, circuit, line_order=None):
        """
        The tensor product of this circuit and `circuit`.

        That is, it adds `circuit` to this circuit as new lines.  The line
        labels of `circuit` must be disjoint from the line labels of this
        circuit, as otherwise applying the circuits in parallel does not make
        sense.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be tensored.

        line_order : List, optional
            A list of all the line labels specifying the order of the circuit in the updated
            circuit. If None, the lines of `circuit` are added below the lines of this circuit.
            Note that, for many purposes, the ordering of lines of the circuit is irrelevant.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.tensor_circuit_inplace(circuit, line_order)
        cpy.done_editing()
        return cpy

    def replace_layer_with_circuit_inplace(self, circuit, j):
        """
        Replaces the `j`-th layer of this circuit with `circuit`.

        Parameters
        ----------
        circuit : Circuit
            The circuit to insert

        j : int
            The layer index to replace.

        Returns
        -------
        None
        """
        del self[j]
        self.insert_labels_into_layers_inplace(circuit, j)

    def replace_layer_with_circuit(self, circuit, j):
        """
        Replaces the `j`-th layer of this circuit with `circuit`.

        Parameters
        ----------
        circuit : Circuit
            The circuit to insert

        j : int
            The layer index to replace.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.replace_layer_with_circuit_inplace(circuit, j)
        cpy.done_editing()
        return cpy

    def replace_gatename_inplace(self, old_gatename, new_gatename):
        """
        Changes the *name* of a gate throughout this Circuit.

        Note that the name is only a part of the label identifying each
        gate, and doesn't include the lines (qubits) a gate acts upon.  For
        example, the "Gx:0" and "Gx:1" labels both have the same name but
        act on different qubits.

        Parameters
        ----------
        old_gatename : str
            The gate name to replace.

        new_gatename : str
            The name to replace `old_gatename` with.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        def replace(obj):  # obj is either a simple label or a list
            if isinstance(obj, _Label):
                if obj.name == old_gatename:
                    newobj = _Label(new_gatename, obj.sslbls)
                else: newobj = obj
            else:
                newobj = [replace(sub) for sub in obj]
            return newobj

        self._labels = replace(self._labels)

    def replace_gatename(self, old_gatename, new_gatename):
        """
        Returns a copy of this Circuit except that `old_gatename` is changed to `new_gatename`.

        Note that the "name" is only a part of the "label" identifying each
        gate, and doesn't include the lines (qubits) a gate acts upon.  For
        example, the "Gx:0" and "Gx:1" labels both have the same name but
        act on different qubits.

        Parameters
        ----------
        old_gatename : str
            The gate name to replace.

        new_gatename : str
            The name to replace `old_gatename` with.

        Returns
        -------
        Circuit
        """
        if not self._static:
            #Could to this in both cases, but is slow for large static circuits
            cpy = self.copy(editable=True)
            cpy.replace_gatename_inplace(old_gatename, new_gatename)
            cpy.done_editing()
            return cpy
        else:  # static case: so self._labels is a tuple of Labels
            return Circuit([lbl.replace_name(old_gatename, new_gatename)
                            for lbl in self._labels], self.line_labels, occurrence=self.occurrence)

    def replace_gatename_with_idle_inplace(self, gatename):
        """
        Treats a given gatename as an idle gate throughout this Circuit.

        This effectively removes this gate name from the circuit, and replaces
        a layer containing only this gate name with an idle layer.

        Parameters
        ----------
        gatename : str
            The gate name to replace.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        def replace(obj):  # obj is either a simple label or a list
            ret = []
            for sub in obj:
                if isinstance(sub, _Label):
                    if sub.name == gatename: continue
                    ret.append(sub)
                else:
                    ret.append(replace(sub))
            return ret

        self._labels = replace(self._labels)

    def replace_gatename_with_idle(self, gatename):
        """
        Returns a copy of this Circuit with a given gatename treated as an idle gate.

        This effectively removes this gate name from the circuit, and replaces
        a layer containing only this gate name with an idle layer.

        Parameters
        ----------
        gatename : str
            The gate name to replace.

        Returns
        -------
        Circuit
        """
        # Slow for large static circuits - maybe make a faster case?
        cpy = self.copy(editable=True)
        cpy.replace_gatename_with_idle_inplace(gatename)
        cpy.done_editing()
        return cpy

    def replace_layer(self, old_layer, new_layer):
        """
        Returns a copy of this Circuit except that `old_layer` is changed to `new_layer`.

        Parameters
        ----------
        old_layer : str or Label
            The layer to find.

        new_layer : str or Label
            The layer to replace found layers with.

        Returns
        -------
        Circuit
        """
        old_layer = to_label(old_layer)
        new_layer = to_label(new_layer)
        if not self._static:
            #Could to this in both cases, but is slow for large static circuits
            cpy = self.copy(editable=False)  # convert our layers to Labels
            return Circuit._fastinit(tuple([new_layer if lbl == old_layer else lbl
                                            for lbl in cpy._labels]), self.line_labels, editable=False,
                                     occurrence=self.occurrence)
        else:  # static case: so self._labels is a tuple of Labels
            return Circuit(tuple([new_layer if lbl == old_layer else lbl
                                  for lbl in self._labels]), self.line_labels, editable=False,
                           occurrence=self.occurrence)

    def replace_layers_with_aliases(self, alias_dict):
        """
        Performs a find and replace using layer aliases.

        Returns a copy of this Circuit except that it's layers that match
        keys of `alias_dict` are replaced with the corresponding values.

        Parameters
        ----------
        alias_dict : dict
            A dictionary whose keys are layer Labels (or equivalent tuples or
            strings), and whose values are Circuits.

        Returns
        -------
        Circuit
        """
        static_self = self if self._static else self.copy(editable=False)  # convert our layers to Labels
        if not alias_dict: return static_self
        assert(all([c._static for c in alias_dict.values()])), "Alias dict values must be *static* circuits!"
        layers = static_self._labels  # a *tuple*
        for label, c in alias_dict.items():
            while label in layers:
                i = layers.index(label)
                layers = layers[:i] + c._labels + layers[i + 1:]
        return Circuit._fastinit(layers, self.line_labels, editable=False, occurrence=self.occurrence)

    #def replace_identity(self, identity, convert_identity_gates = True): # THIS module only
    #    """
    #    Changes the *name* of the idle/identity gate in the circuit. This replaces
    #    the name of the identity element in the circuit by setting self.identity = identity.
    #    If `convert_identity_gates` is True, this also changes the names of all the gates that
    #    had the old self.identity name.
    #
    #    Parameters
    #    ----------
    #    identity : string
    #        The new name for the identity gate.
    #
    #    convert_identity_gates : bool, optional
    #        If True, all gates that had the old identity name are converted to the new identity
    #        name. Otherwise, they keep the old name, and the circuit nolonger considers them to
    #        be identity gates.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    if convert_identity_gates:
    #        self.replace_gatename(self.identity, identity)
    #
    #    self._tup_dirty = self._str_dirty = True
    #    self.identity = identity

    def change_gate_library(self, compilation, allowed_filter=None, allow_unchanged_gates=False, depth_compression=True,
                            one_q_gate_relations=None):
        """
        Re-express a circuit over a different model.

        Parameters
        ----------
        compilation : dict or CompilationLibrary.
            If a dictionary, the keys are some or all of the gates that appear in the circuit, and the values are
            replacement circuits that are normally compilations for each of these gates (if they are not, the action
            of the circuit will be changed). The circuits need not be on all of the qubits, and need only satisfy
            the requirements of the `insert_circuit` method. There must be a key for every gate except the self.identity
            gate, unless `allow_unchanged_gates` is False. In that case, gate that aren't a key in this dictionary are
            left unchanged.

            If a CompilationLibrary, this will be queried via the get_compilation_of() method to find compilations
            for all of the gates in the circuit. So this CompilationLibrary must contain or be able to auto-generate
            compilations for the requested gates, except when `allow_unchanged_gates` is True. In that case, gates
            that a compilation is not returned for are left unchanged.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used when generating compilations from `compilation`. Can only be
            not None if `compilation` is a CompilationLibrary. If a `dict`, keys must be gate names (like `"Gcnot"`) and
            values :class:`QubitGraph` objects indicating where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in the current library that is confined within
            that set is allowed. If None, then all gates within the library are allowed.

        allow_unchanged_gates : bool, optional
            Whether to allow some gates to remain unchanged, and therefore to be absent from `compilation`.  When
            `True` such gates are left alone; when `False` an error is raised if any such gates are encountered.

        depth_compression : bool, optional
            Whether to perform depth compression after changing the gate library. If one_q_gate_relations is None this
            will only remove idle layers and compress the circuit by moving everything as far forward as is possible
            without knowledge of the action of any gates other than self.identity. See the `depth_compression` method
            for more details. Under most circumstances this should be true; if it is False changing gate library will
            often result in a massive increase in circuit depth.

        one_q_gate_relations : dict, optional
            Gate relations for the one-qubit gates in the new gate library, that are used in the depth compression, to
            cancel / combine gates. E.g., one key-value pair might be ('Gh','Gh') : 'I', to signify that two Hadamards c
            ompose to the idle gate 'Gi'. See the depth_compression() method for more details.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        # If it's a CompilationLibrary, it has this attribute. When it's a CompilationLibrary we use the
        # .get_compilation_of method, which will look to see if a compilation for a gate is already available (with
        # `allowed_filter` taken account of) and if not it will attempt to construct it.
        if hasattr(compilation, 'templates'):
            # The function we query to find compilations
            def get_compilation(gate):
                # Use try, because it will fail if it cannot construct a compilation, and this is fine under some
                # circumstances
                try:
                    circuit = compilation.get_compilation_of(gate, allowed_filter=allowed_filter, verbosity=0)
                    return circuit
                except:
                    return None
        # Otherwise, we assume it's a dict.
        else:
            assert(allowed_filter is None), \
                "`allowed_filter` can only been not None if the compilation is a CompilationLibrary!"
            # The function we query to find compilations

            def get_compilation(gate):
                return compilation.get(gate, None)

        for ilayer in range(self.num_layers - 1, -1, -1):
            icomps_to_remove = []
            for icomp, l in enumerate(self._layer_components(ilayer)):  # loop over labels in this layer
                replacement_circuit = get_compilation(l)
                if replacement_circuit is not None:
                    # Replace the gate with a circuit: remove the gate and add insert
                    # the replacement circuit as the following layers.
                    icomps_to_remove.append(icomp)
                    self.insert_labels_into_layers_inplace(replacement_circuit, ilayer + 1)
                else:
                    # We never consider not having a compilation for the identity to be a failure.
                    if not allow_unchanged_gates:
                        raise ValueError(
                            "`compilation` does not contain, or cannot generate a compilation for {}!".format(l))

            for icomp in reversed(icomps_to_remove):
                self._remove_layer_component(ilayer, icomp)

        # If specified, perform the depth compression.
        # It is better to do this *after* the identity name has been changed.
        if depth_compression:
            self.compress_depth_inplace(one_q_gate_relations=one_q_gate_relations, verbosity=0)

    def map_names_inplace(self, mapper):
        """
        The names of all of the simple labels are updated in-place according to the mapping function `mapper`.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing gate name values
            and whose values are the new names (strings) or a function
            which takes a single (existing name) argument and returns a new name.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        # If the mapper is a dict, turn it into a function
        def mapper_func(gatename): return mapper.get(gatename, None) \
            if isinstance(mapper, dict) else mapper

        def map_names(obj):  # obj is either a simple label or a list
            if isinstance(obj, _Label):
                if obj.is_simple():  # *simple* label
                    new_name = mapper_func(obj.name)
                    newobj = _Label(new_name, obj.sslbls) \
                        if (new_name is not None) else obj
                else:  # compound label
                    newobj = _Label([map_names(comp) for comp in obj.components])
            else:
                newobj = [map_names(sub) for sub in obj]
            return newobj
        self._labels = map_names(self._labels)

    def map_state_space_labels_inplace(self, mapper):
        """
        The labels of all of the lines (wires/qubits) are updated according to the mapping function `mapper`.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.line_labels values
            and whose value are the new labels, or a function which takes a
            single (existing line-label) argument and returns a new line-label.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        # If the mapper is a dict, turn it into a function
        def mapper_func(line_label): return mapper[line_label] \
            if isinstance(mapper, dict) else mapper

        self._line_labels = tuple((mapper_func(l) for l in self.line_labels))

        def map_sslbls(obj):  # obj is either a simple label or a list
            if isinstance(obj, _Label):
                new_sslbls = [mapper_func(l) for l in obj.sslbls] \
                    if (obj.sslbls is not None) else None
                newobj = _Label(obj.name, new_sslbls)
            else:
                newobj = [map_sslbls(sub) for sub in obj]
            return newobj
        self._labels = map_sslbls(self._labels)

    def map_state_space_labels(self, mapper):
        """
        Creates a new Circuit whose line labels are updated according to the mapping function `mapper`.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.line_labels values
            and whose value are the new labels, or a function which takes a
            single (existing line-label) argument and returns a new line-label.

        Returns
        -------
        Circuit
        """
        def mapper_func(line_label): return mapper[line_label] \
            if isinstance(mapper, dict) else mapper(line_label)
        mapped_line_labels = tuple(map(mapper_func, self.line_labels))
        return Circuit([l.map_state_space_labels(mapper_func) for l in self.layertup],
                       mapped_line_labels, None, not self._static, occurrence=self.occurrence)

    def reorder_lines_inplace(self, order):
        """
        Reorders the lines (wires/qubits) of the circuit.

        Note that the ordering of the lines is unimportant for most purposes.

        Parameters
        ----------
        order : list
            A list containing all of the circuit line labels (self.line_labels) in the
            order that the should be converted to.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        assert(set(order) == set(self.line_labels)), "The line labels must be the same!"
        self._line_labels = tuple(order)

    def reorder_lines(self, order):
        """
        Reorders the lines (wires/qubits) of the circuit.

        Note that the ordering of the lines is unimportant for most purposes.

        Parameters
        ----------
        order : list
            A list containing all of the circuit line labels (self.line_labels) in the
            order that the should be converted to.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.reorder_lines_inplace(order)
        cpy.done_editing()
        return cpy

    def _is_line_idling(self, line_label, idle_layer_labels=None):
        """
        Whether the line in question is idling in *every* circuit layer.

        Parameters
        ----------
        line_label : str or int
            The label of the line (i.e., "wire" or qubit).

        idle_layer_labels : iterable, optional
            A list or tuple of layer-labels that should be treated
            as idle operations, so their presence will not disqualify
            a line from being "idle".  E.g. `["Gi"]` will cause `"Gi"`
            layers to be considered idle layers.

        Returns
        -------
        bool
            True if the line is idling. False otherwise.
        """
        if self._static:
            layers = list(filter(lambda x: x not in idle_layer_labels, self._labels)) \
                if idle_layer_labels else self._labels
            all_sslbls = None if any([layer.sslbls is None for layer in layers]) \
                else set(_itertools.chain(*[layer.sslbls for layer in layers]))
        else:
            all_sslbls = _sslbls_of_nested_lists_of_simple_labels(self._labels, idle_layer_labels)  # None or a set

        if all_sslbls is None:
            return False  # no lines are idling
        return bool(line_label not in all_sslbls)

    def idling_lines(self, idle_layer_labels=None):
        """
        Returns the line labels corresponding to idling lines.

        Parameters
        ----------
        idle_layer_labels : iterable, optional
            A list or tuple of layer-labels that should be treated
            as idle operations, so their presence will not disqualify
            a line from being "idle".  E.g. `["Gi"]` will cause `"Gi"`
            layers to be considered idle layers.

        Returns
        -------
        tuple
        """
        if self._static:
            layers = list(filter(lambda x: x not in idle_layer_labels, self._labels)) \
                if idle_layer_labels else self._labels
            all_sslbls = None if any([layer.sslbls is None for layer in layers]) \
                else set(_itertools.chain(*[layer.sslbls for layer in layers]))
        else:
            all_sslbls = _sslbls_of_nested_lists_of_simple_labels(self._labels, idle_layer_labels)  # None or a set

        if all_sslbls is None:
            return ()
        else:
            return tuple([x for x in self.line_labels
                          if x not in all_sslbls])  # preserve order

    def delete_idling_lines_inplace(self, idle_layer_labels=None):
        """
        Removes from this circuit all lines that are idling at every layer.

        Parameters
        ----------
        idle_layer_labels : iterable, optional
            A list or tuple of layer-labels that should be treated
            as idle operations, so their presence will not disqualify
            a line from being "idle".  E.g. `["Gi"]` will cause `"Gi"`
            layers to be considered idle layers.

        Returns
        -------
        None
        """
        #assert(not self._static),"Cannot edit a read-only circuit!"
        # Actually, this is OK even for static circuits because it won't affect the hashed value (labels only)

        if idle_layer_labels:
            assert(all([to_label(x).sslbls is None for x in idle_layer_labels])), "Idle layer labels must be *global*"

        if self._static:
            layers = list(filter(lambda x: x not in idle_layer_labels, self._labels)) \
                if idle_layer_labels else self._labels
            all_sslbls = None if any([layer.sslbls is None for layer in layers]) \
                else set(_itertools.chain(*[layer.sslbls for layer in layers]))
        else:
            all_sslbls = _sslbls_of_nested_lists_of_simple_labels(self._labels, idle_layer_labels)  # None or a set

        if all_sslbls is None:
            return  # no lines are idling

        #All we need to do is update line_labels since there aren't any labels
        # to remove in self._labels (as all the lines are idling)
        self._line_labels = tuple([x for x in self.line_labels
                                   if x in all_sslbls])  # preserve order

    def delete_idling_lines(self, idle_layer_labels=None):
        """
        Removes from this circuit all lines that are idling at every layer.

        Parameters
        ----------
        idle_layer_labels : iterable, optional
            A list or tuple of layer-labels that should be treated
            as idle operations, so their presence will not disqualify
            a line from being "idle".  E.g. `["Gi"]` will cause `"Gi"`
            layers to be considered idle layers.

        Returns
        -------
        Circuit
        """
        cpy = self.copy(editable=True)
        cpy.delete_idling_lines_inplace(idle_layer_labels)
        cpy.done_editing()
        return cpy

    def replace_with_idling_line_inplace(self, line_label, clear_straddlers=True):
        """
        Converts the specified line to an idling line, by removing all its gates.

        If there are any multi-qubit gates acting on this line,
        this function will raise an error when `clear_straddlers=False`.

        Parameters
        ----------
        line_label : str or int
            The label of the line to convert to an idling line.

        clear_straddlers : bool, optional
            Whether or not gates which straddle the `line_label` should also
            be cleared.  If `False` and straddling gates exist, an error
            will be raised.

        Returns
        -------
        None
        """
        self.clear_labels(lines=line_label, clear_straddlers=clear_straddlers)

    def reverse_inplace(self):
        """
        Reverses the order of the circuit.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        self._labels = list(reversed(self._labels))  # reverses the layer order
        #FUTURE: would need to reverse_inplace each layer too, if layer can have *sublayers*

    def _combine_one_q_gates_inplace(self, one_q_gate_relations):
        """
        Compresses sequences of 1-qubit gates in the circuit, using the provided gate relations.

        One of the steps of the depth_compression() method, and in most cases that method will
        be more useful.

        Parameters
        ----------
        one_q_gate_relations : dict
            Keys that are pairs of strings, corresponding to 1-qubit gate names, with values that are
            a single string, also corresponding to a 1-qubit gate name. Whenever a 1-qubit gate with
            name `name1` is followed in the circuit by a 1-qubit gate with `name2` then, if
            one_q_gate_relations[name1,name2] = name3, name1 -> name3 and name2 -> self.identity, the
            identity name in the circuit. Moreover, this is still implemented when there are self.identity
            gates between these 1-qubit gates, and it is implemented iteratively in the sense that if there
            is a sequence of 1-qubit gates with names name1, name2, name3, ... and there are relations
            for all of (name1,name2) -> name12, (name12,name3) -> name123 etc then the entire sequence of
            1-qubit gates will be compressed into a single possibly non-idle 1-qubit gate followed by
            idle gates in place of the previous 1-qubit gates.  Note that `None` can be used as `name3`
            to signify that the result is the identity (no gate labels).

            If a ProcessorSpec object has been created for the gates/device in question, the
            ProcessorSpec.oneQgate_relations is the appropriate (and auto-generated) `one_q_gate_relations`.

            Note that this function will not compress sequences of 1-qubit gates that cannot be compressed by
            independently inspecting sequential non-idle pairs (as would be the case with, for example,
            Gxpi Gzpi Gxpi Gzpi, if the relation did not know that (Gxpi,Gzpi) -> Gypi, even though the sequence
            is the identity).

        Returns
        -------
        bool
            False if the circuit is unchanged, and True otherwise.
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        # A flag that is turned to True if any non-trivial re-arranging is implemented by this method.
        compression_implemented = False

        # A flag telling us when to stop iterating
        productive = True

        while productive:  # keep iterating
            #print("BEGIN ITER")
            productive = False
            # Loop through all the qubits, to try and compress squences of 1-qubit gates on the qubit in question.
            for ilayer in range(0, len(self._labels) - 1):
                layerA_comps = self._layer_components(ilayer)
                layerB_comps = self._layer_components(ilayer + 1)
                applies = []
                for a, lblA in enumerate(layerA_comps):
                    if not isinstance(lblA, _Label) or (lblA.sslbls is None) \
                       or (len(lblA.sslbls) != 1): continue  # only care about 1-qubit simple labels within a layer
                    #FUTURE: could relax the != 1 condition?

                    for b, lblB in enumerate(layerB_comps):
                        if isinstance(lblB, _Label) and lblB.sslbls == lblA.sslbls:
                            #queue an apply rule if one exists
                            #print("CHECK for: ", (lblA.name,lblB.name))
                            if (lblA.name, lblB.name) in one_q_gate_relations:
                                new_Aname = one_q_gate_relations[lblA.name, lblB.name]
                                applies.append((a, b, new_Aname, lblA.sslbls))
                                break

                layerA_sslbls = _sslbls_of_nested_lists_of_simple_labels(self._labels[ilayer])
                for b, lblB in enumerate(layerB_comps):
                    if isinstance(lblB, _Label):
                        #see if layerA happens to *not* have anything on lblB.sslbls:
                        if layerA_sslbls is None or \
                           (lblB.sslbls is not None and len(set(lblB.sslbls).intersection(layerA_sslbls)) == 0):
                            applies.append((-1, b, lblB.name, lblB.sslbls))  # shift label over
                            break

                if len(applies) > 0:
                    # Record that a compression has been implemented : the circuit has been changed.
                    compression_implemented = productive = True

                #execute queued applies (outside of above loops)
                sorted_applies = sorted(applies, key=lambda x: -x[1])  # sort in order of descending 'b' for removes
                ilayer_inds_to_remove = []
                for a, b, new_Aname, sslbls in sorted_applies:
                    if a == -1:  # Note: new_Aname cannot be None here
                        self._append_layer_component(ilayer, _Label(new_Aname, sslbls))
                    elif new_Aname is None:
                        ilayer_inds_to_remove.append(a)  # remove layer component - but wait to do so in order
                    else:
                        self._replace_layer_component(ilayer, a, _Label(new_Aname, sslbls))
                    self._remove_layer_component(ilayer + 1, b)

                for a in sorted(ilayer_inds_to_remove, reverse=True):
                    self._remove_layer_component(ilayer, a)

        # returns the flag that tells us whether the algorithm achieved anything.
        return compression_implemented

    def _shift_gates_forward_inplace(self):
        """
        Shift all gates forward (left) as far as is possible.

        This operation is performed without any knowledge of what any of the
        gates are.  One of the steps of :method:`depth_compression()`.

        Returns
        -------
        bool
            False if the circuit is unchanged, and True otherwise.
        """
        assert(not self._static), "Cannot edit a read-only circuit!"
        # Keeps track of whether any changes have been made to the circuit.
        compression_implemented = False

        #print("BEGIN")
        used_lines = {}
        for icurlayer in range(len(self._labels)):
            #print("LAYER ",icurlayer)
            #Slide labels in current layer to left ("forward")
            icomps_to_remove = []; used_lines[icurlayer] = set()
            for icomp, lbl in enumerate(self._layer_components(icurlayer)):
                #see if we can move this label forward
                #print("COMP%d: %s" % (icomp,str(lbl)))
                sslbls = _sslbls_of_nested_lists_of_simple_labels(lbl)
                if sslbls is None: sslbls = self.line_labels

                dest_layer = icurlayer
                while dest_layer > 0 and len(used_lines[dest_layer - 1].intersection(sslbls)) == 0:
                    dest_layer -= 1
                if dest_layer < icurlayer:
                    icomps_to_remove.append(icomp)  # remove this label from current layer
                    self._append_layer_component(dest_layer, lbl)  # add it to the destination layer
                    used_lines[dest_layer].update(sslbls)  # update used_lines at dest layer
                    #print(" <-- layer %d (used=%s)" % (dest_layer,str(used_lines[dest_layer])))
                else:
                    #can't move this label forward - update used_lines of current layer
                    used_lines[icurlayer].update(sslbls)  # update used_lines at dest layer
                    #print(" can't move: (cur layer used=%s)" % (str(used_lines[icurlayer])))

            #Remove components in current layer which were pushed forward
            #print("Removing ",icomps_to_remove," from layer ",icurlayer)
            for icomp in reversed(icomps_to_remove):
                self._remove_layer_component(icurlayer, icomp)

            if len(icomps_to_remove) > 0:  # keep track of whether we did anything
                compression_implemented = True

        # Only return the bool if requested
        return compression_implemented

    def delete_idle_layers_inplace(self):
        """
        Deletes all layers in this circuit that contain no gate operations.

        One of the steps of the `depth_compression()` method.

        Returns
        -------
        bool
            False if the circuit is unchanged, and True otherwise.
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        inds_to_remove = []
        for ilayer, layer_labels in enumerate(self._labels):
            if layer_labels == []:
                inds_to_remove.append(ilayer)
        for ilayer in reversed(inds_to_remove):
            del self._labels[ilayer]

        return bool(len(inds_to_remove) > 0)  # whether compression was implemented

    def compress_depth_inplace(self, one_q_gate_relations=None, verbosity=0):
        """
        Compresses the depth of this circuit using very simple re-write rules.

        1. If `one_q_gate_relations` is provided, all sequences of 1-qubit gates
           in the  circuit are compressed as far as is possible using only the
           pair-wise combination rules provided by this dict (see below).
        2. All gates are shifted forwarded as far as is possible without any
           knowledge of what any of the gates are.
        3. All idle-only layers are deleted.

        Parameters
        ----------
        one_q_gate_relations : dict
            Keys that are pairs of strings, corresponding to 1-qubit gate names, with values that are
            a single string, also corresponding to a 1-qubit gate name. Whenever a 1-qubit gate with
            name `name1` is followed in the circuit by a 1-qubit gate with `name2` then, if
            one_q_gate_relations[name1,name2] = name3, name1 -> name3 and name2 -> self.identity, the
            identity name in the circuit. Moreover, this is still implemented when there are self.identity
            gates between these 1-qubit gates, and it is implemented iteratively in the sense that if there
            is a sequence of 1-qubit gates with names name1, name2, name3, ... and there are relations
            for all of (name1,name2) -> name12, (name12,name3) -> name123 etc then the entire sequence of
            1-qubit gates will be compressed into a single possibly non-idle 1-qubit gate followed by
            idle gates in place of the previous 1-qubit gates.

            If a ProcessorSpec object has been created for the gates/device in question, the
            ProcessorSpec.oneQgate_relations is the appropriate (and auto-generated) `one_q_gate_relations`.

            Note that this function will not compress sequences of 1-qubit gates that cannot be compressed by
            independently inspecting sequential non-idle pairs (as would be the case with, for example,
            Gxpi Gzpi Gxpi Gzpi, if the relation did not know that (Gxpi,Gzpi) -> Gypi, even though the sequence
            is the identity).

        verbosity : int, optional
            If > 0, information about the depth compression is printed to screen.

        Returns
        -------
        None
        """
        assert(not self._static), "Cannot edit a read-only circuit!"

        if verbosity > 0:
            print("- Implementing circuit depth compression")
            print("  - Circuit depth before compression is {}".format(self.num_layers))

        flag1 = False
        if one_q_gate_relations is not None:
            flag1 = self._combine_one_q_gates_inplace(one_q_gate_relations)
        flag2 = self._shift_gates_forward_inplace()
        flag3 = self.delete_idle_layers_inplace()

        if verbosity > 0:
            if not (flag1 or flag2 or flag3):
                print("  - Circuit unchanged by depth compression algorithm")
            print("  - Circuit depth after compression is {}".format(self.num_layers))

    def layer(self, j):
        """
        Returns a tuple of the *components*, i.e. the (non-identity) gates, in the layer at depth `j`.

        These are the `.components` of the :class:`Label` returned by indexing
        this Circuit (using square brackets) with `j`, i.e. this returns
        `this_circuit[j].components`.

        Parameters
        ----------
        j : int
            The index (depth) of the layer to be returned

        Returns
        -------
        tuple
        """
        return tuple(self.layer_label(j).components)

    def layer_label(self, j):
        """
        Returns the layer, as a :class:`Label`, at depth j.

        This label contains as components all the (non-identity) gates in the layer..

        Parameters
        ----------
        j : int
            The index (depth) of the layer to be returned

        Returns
        -------
        Label
        """
        assert(j >= 0 and j < self.num_layers
               ), "Circuit layer label invalid! Circuit is only of depth {}".format(self.num_layers)
        return self[j]

    def layer_with_idles(self, j, idle_gate_name='I'):
        """
        Returns a tuple of the components of the layer at depth `j`, with `idle_gate_name` at empty circuit locations.

        This effectively places an explicit `idle_gate_name` gates wherever there is an implied
        identity operation in the circuit.

        Parameters
        ----------
        j : int
            The index (depth) of the layer to be returned

        idle_gate_name : str, optional
            The idle gate name to use.  Note that state space (qubit) labels
            will be added to this name to form a :class:`Label`.

        Returns
        -------
        tuple
        """
        return tuple(self.layer_label_with_idles(j, idle_gate_name).components)

    def layer_label_with_idles(self, j, idle_gate_name='I'):
        """
        Returns the layer, as a :class:`Label`, at depth j, with `idle_gate_name` at empty circuit locations.

        This effectively places an explicit `idle_gate_name` gates wherever there is an implied
        identity operation in the circuit.

        Parameters
        ----------
        j : int
            The index (depth) of the layer to be returned

        idle_gate_name : str, optional
            The idle gate name to use.  Note that state space (qubit) labels
            will be added to this name to form a :class:`Label`.

        Returns
        -------
        Label
        """
        layer_lbl = self.layer_label(j)  # (a Label)
        if layer_lbl.sslbls is None:
            if layer_lbl == ():  # special case - the completely empty layer: sslbls=None but needs padding
                return _Label([_Label(idle_gate_name, line_lbl) for line_lbl in self.line_labels])
            return layer_lbl  # all qubits used - no idles to pad

        components = list(layer_lbl.components)
        for line_lbl in self.line_labels:
            if line_lbl not in layer_lbl.sslbls:
                components.append(_Label(idle_gate_name, line_lbl))
        return _Label(components)

    @property
    def num_layers(self):
        """
        The number of circuit layers.

        In simple circuits, this is the same as the depth (given by :method:`depth`).
        For circuits containing sub-circuit blocks, this gives the number of
        top-level layers in this circuit.

        Returns
        -------
        int
        """
        return len(self._labels)

    @property
    def depth(self):
        """
        The circuit depth.

        This is the number of layers in simple circuits. For circuits containing
        sub-circuit blocks, this includes the full depth of these blocks.  If you
        just want the number of top-level layers, use :method:`num_layers`.

        Returns
        -------
        int
        """
        if self._static:
            return sum([lbl.depth for lbl in self._labels])
        else:
            return sum([_Label(layer_lbl).depth for layer_lbl in self._labels])

    @property
    def width(self):
        """
        The circuit width.

        This is the number of qubits on which the circuit acts. This includes
        qubits that only idle, but are included as part of the circuit according
        to self.line_labels.

        Returns
        -------
        int
        """
        return len(self.line_labels)

    @property
    def size(self):
        """
        Returns the circuit size.

        This is the sum of the sizes of all the gates in the circuit. A gate
        that acts on n-qubits has a size of n, with the exception of the idle
        which has a size of 0. Hence, the circuit is given by: `size = depth *
        num_lines - num_1Q_idles`.

        Returns
        -------
        int
        """
        #TODO HERE -update from here down b/c of sub-circuit blocks
        if self._static:
            def size(lbl):  # obj a Label, perhaps compound
                if lbl.is_simple():  # a simple label
                    return len(lbl.sslbls) if (lbl.sslbls is not None) else len(self.line_labels)
                else:
                    return sum([size(sublbl) for sublbl in lbl.components])
        else:
            def size(obj):  # obj is either a simple label or a list
                if isinstance(obj, _Label):  # all Labels are simple labels
                    return len(obj.sslbls) if (obj.sslbls is not None) else len(self.line_labels)
                else:
                    return sum([size(sub) for sub in obj])

        return sum([size(layer_lbl) for layer_lbl in self._labels])

    @property
    def duration(self):
        # similar to depth()
        if self._static:
            return sum([lbl.time for lbl in self._labels])
        else:
            return sum([_Label(layer_lbl).time for layer_lbl in self._labels])

    def two_q_gate_count(self):
        """
        The number of two-qubit gates in the circuit.

        (Note that this cannot distinguish between "true" 2-qubit gates and gate
        that have been defined to act on two qubits but that represent some
        tensor-product gate.)

        Returns
        -------
        int
        """
        return self.num_nq_gates(2)

    def num_nq_gates(self, nq):
        """
        The number of `nq`-qubit gates in the circuit.

        (Note that this cannot distinguish between "true" `nq`-qubit gates and
        gate that have been defined to act on `nq` qubits but that represent
        some tensor-product gate.)

        Parameters
        ----------
        nq : int
            The qubit-count of the gates to count.  For example, if `nq == 3`,
            this function returns the number of 3-qubit gates.

        Returns
        -------
        int
        """
        if self._static:
            def cnt(lbl):  # obj a Label, perhaps compound
                if lbl.is_simple():  # a simple label
                    return 1 if (lbl.sslbls is not None) and (len(lbl.sslbls) == nq) else 0
                else:
                    return sum([cnt(sublbl) for sublbl in lbl.components])
        else:
            def cnt(obj):  # obj is either a simple label or a list
                if isinstance(obj, _Label):  # all Labels are simple labels
                    return 1 if (obj.sslbls is not None) and (len(obj.sslbls) == nq) else 0
                else:
                    return sum([cnt(sub) for sub in obj])

        return sum([cnt(layer_lbl) for layer_lbl in self._labels])

    @property
    def num_multiq_gates(self):
        """
        The number of multi-qubit (2+ qubits) gates in the circuit.

        (Note that this cannot distinguish between "true" multi-qubit gates and
        gate that have been defined to act on more than one qubit but that
        represent some tensor-product gate.)

        Returns
        -------
        int
        """
        if self._static:
            def cnt(lbl):  # obj a Label, perhaps compound
                if lbl.is_simple():  # a simple label
                    return 1 if (lbl.sslbls is not None) and (len(lbl.sslbls) >= 2) else 0
                else:
                    return sum([cnt(sublbl) for sublbl in lbl.components])
        else:
            def cnt(obj):  # obj is either a simple label or a list
                if isinstance(obj, _Label):  # all Labels are simple labels
                    return 1 if (obj.sslbls is not None) and (len(obj.sslbls) >= 2) else 0
                else:
                    return sum([cnt(sub) for sub in obj])

        return sum([cnt(layer_lbl) for layer_lbl in self._labels])

    # UNUSED
    #def predicted_error_probability(self, gate_error_probabilities):
    #    """
    #    Predicts the probability that one or more errors occur in the circuit
    #    if the gates have the error probabilities specified by in the input
    #    dictionary. Given correct error rates for the gates and stochastic errors,
    #    this is predictive of the probability of an error in the circuit. But note
    #    that that is generally *not* the same as the probability that the circuit
    #    implemented is incorrect (e.g., stochastic errors can cancel).
    #
    #    Parameters
    #    ----------
    #    gate_error_probabilities : dict
    #        A dictionary where the keys are the labels that appear in the circuit, and
    #        the value is the error probability for that gate.
    #
    #    Returns
    #    -------
    #    float
    #        The probability that there is one or more errors in the circuit.
    #    """
    #    f = 1.
    #    depth = self.num_layers
    #    for i in range(0,self.num_lines):
    #        for j in range(0,depth):
    #            gatelbl = self.line_items[i][j]
    #
    #            # So that we don't include multi-qubit gates more than once.
    #            if gatelbl.qubits is None:
    #                if i == 0:
    #                    f = f*(1-gate_error_probabilities[gatelbl])
    #            elif gatelbl.qubits[0] == self.line_labels[i]:
    #                f = f*(1-gate_error_probabilities[gatelbl])
    #    return 1 - f

    def _togrid(self, identity_name):
        """ return a list-of-lists rep? """
        d = self.num_layers
        line_items = [[_Label(identity_name, ll)] * d for ll in self.line_labels]

        for ilayer in range(len(self._labels)):
            for layercomp in self._layer_components(ilayer):
                if isinstance(layercomp, _Label):
                    comp_label = layercomp
                    if layercomp.is_simple():
                        comp_sslbls = layercomp.sslbls
                    else:
                        #We can't intelligently flatten compound labels that occur within a layer-label yet...
                        comp_sslbls = layercomp.sslbls
                else:  # layercomp must be a list (and _static == False)
                    comp_label = _Label(layercomp)
                    comp_sslbls = _sslbls_of_nested_lists_of_simple_labels(layercomp)
                if comp_sslbls is None: comp_sslbls = self.line_labels
                for sslbl in comp_sslbls:
                    lineIndx = self.line_labels.index(sslbl)  # replace w/dict for speed...
                    line_items[lineIndx][ilayer] = comp_label
        return line_items

    def __str__(self):
        """
        A text rendering of the circuit.
        """

        # If it's a circuit over no lines, return an empty string
        if self.num_lines == 0: return ''

        s = ''
        Ctxt = 'C'
        Ttxt = 'T'
        identityName = 'I'  # can be anything that isn't used in circuit

        def abbrev(lbl, k):  # assumes a simple label w/ name & qubits
            """ Returns what to print on line 'k' for label 'lbl' """
            lbl_qubits = lbl.qubits if (lbl.qubits is not None) else self.line_labels
            nqubits = len(lbl_qubits)
            if nqubits == 1 and lbl.name is not None:
                if isinstance(lbl, _CircuitLabel):  # HACK
                    return "|" + str(lbl) + "|"
                elif lbl.args:
                    return lbl.name + "(" + ",".join(map(str, lbl.args)) + ")"
                else:
                    return lbl.name
            elif lbl.name in ('CNOT', 'Gcnot') and nqubits == 2:  # qubit indices = (control,target)
                if k == self.line_labels.index(lbl_qubits[0]):
                    return Ctxt + str(lbl_qubits[1])
                else:
                    return Ttxt + str(lbl_qubits[0])
            elif lbl.name in ('CPHASE', 'Gcphase') and nqubits == 2:
                if k == self.line_labels.index(lbl_qubits[0]):
                    otherqubit = lbl_qubits[1]
                else:
                    otherqubit = lbl_qubits[0]
                return Ctxt + str(otherqubit)
            elif isinstance(lbl, _CircuitLabel):
                return "|" + str(lbl) + "|"
            else:
                return str(lbl)

        line_items = self._togrid(identityName)
        max_labellen = [max([len(abbrev(line_items[i][j], i))
                             for i in range(0, self.num_lines)])
                        for j in range(0, self.num_layers)]

        max_linelabellen = max([len(str(llabel)) for llabel in self.line_labels])

        for i in range(self.num_lines):
            s += 'Qubit {} '.format(self.line_labels[i]) + ' ' * \
                (max_linelabellen - len(str(self.line_labels[i]))) + '---'
            for j, maxlbllen in enumerate(max_labellen):
                if line_items[i][j].name == identityName:
                    # Replace with special idle print at some point
                    #s += '-'*(maxlbllen+3) # 1 for each pipe, 1 for joining dash
                    s += '|' + ' ' * (maxlbllen) + '|-'
                else:
                    lbl = abbrev(line_items[i][j], i)
                    pad = maxlbllen - len(lbl)
                    s += '|' + ' ' * int(_np.floor(pad / 2)) + lbl + ' ' * int(_np.ceil(pad / 2)) + '|-'  # + '-'*pad
            s += '--\n'

        return s

    def __repr__(self):
        return "Circuit(%s)" % self.str

    def format_display_str(self, width=80):
        """
        Formats a string for displaying this circuit suject to a maximum `width`.

        Parameters
        ----------
        width : int, optional
            The maximum width in characters.  If the circuit is longer than this
            width it is wrapped using multiple lines (like a musical score).

        Returns
        -------
        str
        """
        ret = ""
        circuit_string = str(self).strip()  # get rid of trailing newline
        line_strings = circuit_string.split('\n')
        nLines = len(line_strings)  # e.g., number of qubits
        lineLen = len(line_strings[0])
        assert(nLines == self.num_lines)  # this is assumed...
        assert(all([len(linestr) == lineLen for linestr in line_strings]))  # assume all lines have same length

        iSegment = iStart = iEnd = 0
        while(iEnd < lineLen):
            iStart = iEnd  # start from our last ending point
            prefix = "" if iSegment == 0 else " >>> "
            usable_width = width - len(prefix)
            if iStart + usable_width > lineLen:
                iEnd = lineLen
            elif '-' not in line_strings[0][iStart:iStart + usable_width]:
                iEnd = iStart + usable_width
            else:
                iEnd = iStart + line_strings[0][iStart:iStart + usable_width].rfind('-')

            for iLine in range(nLines):
                ret += prefix + line_strings[iLine][iStart:iEnd] + "\n"
            ret += "\n"
            iSegment += 1

        return ret

    def _print_labelinfo(self):
        """A useful debug routine for printing the internal label structure of a circuit"""
        def plbl(x, lit):
            iscircuit = isinstance(x, _CircuitLabel)
            extra = "reps=%d" % x.reps if iscircuit else ""
            print(lit, ": str=", x, " type=", type(x), " ncomps=", len(x.components), extra)
            if len(x.components) > 1 or iscircuit:
                for i, cmp in enumerate(x.components):
                    plbl(cmp, "  %s[%d]" % (lit, i))

        print("--- LABEL INFO for %s (%d layers) ---" % (self.str, self.num_layers))
        for j in range(0, self.num_layers):
            plbl(self[j], "self[%d]" % j)

    def _write_q_circuit_tex(self, filename):  # TODO
        """
        Writes a LaTeX file for rendering this circuit nicely.

        Creates a file containing LaTex that will display this circuit using the
        Qcircuit.tex LaTex import (compiling the LaTex requires that you have the
        Qcircuit.tex file).

        Parameters
        ----------
        filename : str
            The file to write the LaTex into. Should end with '.tex'

        Returns
        -------
        None
        """
        raise NotImplementedError("TODO: need to upgrade this method")
        n = self.num_lines
        d = self.num_layers

        f = open(filename, 'w')
        f.write("\documentclass{article}\n")
        f.write("\\usepackage{mathtools}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage[paperwidth=" + str(5. + d * .3)
                + "in, paperheight=" + str(2 + n * 0.2) + "in,margin=0.5in]{geometry}")
        f.write("\input{Qcircuit}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{equation*}\n")
        f.write("\Qcircuit @C=1.0em @R=0.5em {\n")

        for q in range(0, n):
            qstring = '&'
            # The quantum wire for qubit q
            circuit_for_q = self.line_items[q]
            for gate in circuit_for_q:
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                nqubits = len(gate_qubits)
                if gate.name == self.identity:
                    qstring += ' \qw &'
                elif gate.name in ('CNOT', 'Gcnot') and nqubits == 2:
                    if gate_qubits[0] == q:
                        qstring += ' \ctrl{' + str(gate_qubits[1] - q) + '} &'
                    else:
                        qstring += ' \\targ &'
                elif gate.name in ('CPHASE', 'Gcphase') and nqubits == 2:
                    if gate_qubits[0] == q:
                        qstring += ' \ctrl{' + str(gate_qubits[1] - q) + '} &'
                    else:
                        qstring += ' \control \qw &'

                else:
                    qstring += ' \gate{' + str(gate.name) + '} &'

            qstring += ' \qw & \\' + '\\ \n'
            f.write(qstring)

        f.write("}\end{equation*}\n")
        f.write("\end{document}")
        f.close()

    def convert_to_cirq(self,
                        qubit_conversion,
                        wait_duration=None,
                        gatename_conversion=None,
                        idle_gate_name='Gi'):
        """
        Converts this circuit to a Cirq circuit.

        Parameters
        ----------
        qubit_conversion : dict
            Mapping from qubit labels (e.g. integers) to Cirq qubit objects.

        wait_duration : cirq.Duration, optional
            If no gatename_conversion dict is given, the idle operation is not
            converted to a gate. If wait_diration is specified and gatename_conversion
            is not specified, then the idle operation will be converted to a
            `cirq.WaitGate` with the specified duration.

        gatename_conversion : dict, optional
            If not None, a dictionary that converts the gatenames in the circuit to the
            Cirq gates that will appear in the Cirq circuit. If only standard pyGSTi names
            are used (e.g., 'Gh', 'Gp', 'Gcnot', 'Gcphase', etc) this dictionary need not
            be specified, and an automatic conversion to the standard Cirq names will be
            implemented.
        idle_gate_name : str, optional
            Name to use for idle gates. Defaults to 'Gi'

        Returns
        -------
        A Cirq Circuit object.
        """

        try:
            import cirq
        except ImportError:
            raise ImportError("Cirq is required for this operation, and it does not appear to be installed.")

        if gatename_conversion is None:
            gatename_conversion = _itgs.standard_gatenames_cirq_conversions()
            if wait_duration is not None:
                gatename_conversion[idle_gate_name] = cirq.WaitGate(wait_duration)

        moments = []
        for i in range(self.num_layers):
            layer = self.layer_with_idles(i, idle_gate_name)
            operations = []
            for gate in layer:
                operation = gatename_conversion[gate.name]
                if operation is None:
                    # This happens if no idle gate it specified because
                    # standard_gatenames_cirq_conversions maps 'Gi' to `None`
                    continue
                qubits = map(qubit_conversion.get, gate.qubits)
                operations.append(operation.on(*qubits))
            moments.append(cirq.Moment(operations))

        return cirq.Circuit(moments)

    def convert_to_quil(self,
                        num_qubits=None,
                        gatename_conversion=None,
                        qubit_conversion=None,
                        readout_conversion=None,
                        block_between_layers=True,
                        block_idles=True,
                        gate_declarations=None):  # TODO
        """
        Converts this circuit to a quil string.

        Parameters
        ----------
        num_qubits : int, optional
            The number of qubits for the quil file.  If None, then this is assumed
            to equal the number of line labels in this circuit.

        gatename_conversion : dict, optional
            A dictionary mapping gate names contained in this circuit to the corresponding
            gate names used in the rendered quil.  If None, a standard set of conversions
            is used (see :function:`standard_gatenames_quil_conversions`).

        qubit_conversion : dict, optional
            If not None, a dictionary converting the qubit labels in the circuit to the
            desired qubit labels in the quil output. Can be left as None if the qubit
            labels are either (1) integers, or (2) of the form 'Qi' for integer i. In
            this case they are converted to integers (i.e., for (1) the mapping is trivial,
            for (2) the mapping strips the 'Q').

        readout_conversion : dict, optional
            If not None, a dictionary converting the qubit labels mapped through qubit_conversion
            to the bit labels for readot.  E.g. Suppose only qubit 2 (on Rigetti hardware)
            is in use.  Then the pyGSTi string will have only one qubit (labeled 0); it
            will get remapped to 2 via qubit_conversion={0:2}.  At the end of the quil
            circuit, readout should go recorded in bit 0, so readout_conversion = {0:0}.
            (That is, qubit with pyGSTi label 0 gets read to Rigetti bit 0, even though
            that qubit has Rigetti label 2.)

        block_between_layers : bool, optional
            When `True`, add in a barrier after every circuit layer.  Including such "pragma" blocks
            can be important for QCVV testing, as this can help reduce the "behind-the-scenes"
            compilation (beyond necessary conversion to native instructions) experience by the circuit.

        block_idles : bool, optional
            In the special case of global idle gates, pragma-block barriers are inserted *even*
            when `block_between_layers=False`.  Set `block_idles=False` to disable this behavior,
            whcih typically results in global idle gates being removed by the compiler.

        gate_declarations : dict, optional
            If not None, a dictionary that provides unitary maps for particular gates that
            are not already in the quil syntax.

        Returns
        -------
        str
            A quil string.
        """

        # create standard conversations.
        if gatename_conversion is None:
            gatename_conversion = _itgs.standard_gatenames_quil_conversions()
        if qubit_conversion is None:
            # To tell us whether we have found a standard qubit labelling type.
            standardtype = False
            # Must first check they are strings, because cannot query q[0] for int q.
            if all([isinstance(q, str) for q in self.line_labels]):
                if all([q[0] == 'Q' for q in self.line_labels]):
                    standardtype = True
                    qubit_conversion = {llabel: int(llabel[1:]) for llabel in self.line_labels}
            if all([isinstance(q, int) for q in self.line_labels]):
                qubit_conversion = {q: q for q in self.line_labels}
                standardtype = True
            if not standardtype:
                raise ValueError(
                    "No standard qubit labelling conversion is available! Please provide `qubit_conversion`.")

        if num_qubits is None:
            num_qubits = len(self.line_labels)

        # Init the quil string.
        quil = ''

        if gate_declarations is not None:
            for gate_lbl in gate_declarations.keys():
                quil += _np_to_quil_def_str(gate_lbl, gate_declarations[gate_lbl])

        depth = self.num_layers

#        quil += 'DECLARE ro BIT[{0}]\n'.format(str(self.num_lines))
        quil += 'DECLARE ro BIT[{0}]\n'.format(str(num_qubits))

        quil += 'RESET\n'

        quil += 'PRAGMA INITIAL_REWIRING "NAIVE"\n'

        # Go through the layers, and add the quil for each layer in turn.
        for l in range(depth):

            # Get the layer, without identity gates and containing each gate only once.
            layer = self.layer_label(l)
            # For keeping track of which qubits have a gate on them in the layer.
            qubits_used = []

            # Go through the (non-self.identity) gates in the layer and convert them to quil
            for gate in layer.components:
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                assert(len(gate_qubits) <= 2 or gate.qubits is None), \
                    'Gate on more than 2 qubits given; this is currently not supported!'

                # Find the quil for the gate.
                quil_for_gate = gatename_conversion[gate.name]

                #If gate.qubits is None, gate is assumed to be single-qubit gate
                #acting in parallel on all qubits.  If the gate is a global idle, then
                #Pragma blocks are inserted (for tests like idle tomography) even
                #if block_between_layers==False.  Set block_idles=False to disable this as well.
                if gate.qubits is None:
                    if quil_for_gate == 'I':
                        if block_idles:
                            quil += 'PRAGMA PRESERVE_BLOCK\n'
                        for q in gate_qubits:
                            quil += quil_for_gate + ' ' + str(qubit_conversion[q]) + '\n'
                        if block_idles:
                            quil += 'PRAGMA END_PRESERVE_BLOCK\n'
                    else:
                        for q in gate_qubits:
                            quil += quil_for_gate + ' ' + str(qubit_conversion[q]) + '\n'

                #If gate.qubits is not None, then apply the one- or multi-qubit gate to
                #the explicitly specified qubits.
                else:
                    for q in gate_qubits: quil_for_gate += ' ' + str(qubit_conversion[q])
                    quil_for_gate += '\n'
                    # Add the quil for the gate to the quil string.
                    quil += quil_for_gate

                # Keeps track of the qubits that have been accounted for, and checks that hadn't been used
                # although that should already be checked in the .layer_label(), which checks for its a valid
                # circuit layer.
                assert(not set(gate_qubits).issubset(set(qubits_used)))
                qubits_used.extend(gate_qubits)

            # All gates that don't have a non-idle gate acting on them get an idle in the layer.
            for q in self.line_labels:
                if q not in qubits_used:
                    quil += 'I' + ' ' + str(qubit_conversion[q]) + '\n'

            # Add in a barrier after every circuit layer if block_between_layers==True.
            # Including pragma blocks are critical for QCVV testing, as circuits should usually
            # experience minimal "behind-the-scenes" compilation (beyond necessary
            # conversion to native instructions)
            # To do: Add "barrier" as native pygsti circuit instruction, and use for indicating
            # where pragma blocks should be.
            if block_between_layers:
                quil += 'PRAGMA PRESERVE_BLOCK\nPRAGMA END_PRESERVE_BLOCK\n'

        # Add in a measurement at the end.
        if readout_conversion is None:
            for q in self.line_labels:
                #            quil += "MEASURE {0} [{1}]\n".format(str(qubit_conversion[q]),str(qubit_conversion[q]))
                quil += "MEASURE {0} ro[{1}]\n".format(str(qubit_conversion[q]), str(qubit_conversion[q]))
        else:
            for q in self.line_labels:
                quil += "MEASURE {0} ro[{1}]\n".format(str(qubit_conversion[q]), str(readout_conversion[q]))

        return quil

    def convert_to_openqasm(self, num_qubits=None,
                            standard_gates_version='u3',
                            gatename_conversion=None, qubit_conversion=None,
                            block_between_layers=True,
                            block_between_gates=False,
                            gateargs_map=None):  # TODO
        """
        Converts this circuit to an openqasm string.

        Parameters
        ----------
        num_qubits : in, optional
            The number of qubits for the openqasm file.  If None, then this is assumed
            to equal the number of line labels in this circuit.

        gatename_conversion : dict, optional
            If not None, a dictionary that converts the gatenames in the circuit to the
            gatenames that will appear in the openqasm output. If only standard pyGSTi names
            are used (e.g., 'Gh', 'Gp', 'Gcnot', 'Gcphase', etc) this dictionary need not
            be specified, and an automatic conversion to the standard openqasm names will be
            implemented.

        qubit_conversion : dict, optional
            If not None, a dictionary converting the qubit labels in the circuit to the
            desired qubit labels in the openqasm output. Can be left as None if the qubit
            labels are either (1) integers, or (2) of the form 'Qi' for integer i. In
            this case they are converted to integers (i.e., for (1) the mapping is trivial,
            for (2) the mapping strips the 'Q').

        block_between_layers : bool, optional
            When `True`, add in a barrier after every circuit layer.  Including such barriers
            can be important for QCVV testing, as this can help reduce the "behind-the-scenes"
            compilation (beyond necessary conversion to native instructions) experience by the circuit.

        Returns
        -------
        str
            An openqasm string.
        """

        # create standard conversations.
        if gatename_conversion is None:
            gatename_conversion, gateargs_map = _itgs.standard_gatenames_openqasm_conversions(standard_gates_version)
        if qubit_conversion is None:
            # To tell us whether we have found a standard qubit labelling type.
            standardtype = False
            # Must first check they are strings, because cannot query q[0] for int q.
            if all([isinstance(q, str) for q in self.line_labels]):
                if all([q[0] == 'Q' for q in self.line_labels]):
                    standardtype = True
                    qubit_conversion = {llabel: int(llabel[1:]) for llabel in self.line_labels}
            if all([isinstance(q, int) for q in self.line_labels]):
                qubit_conversion = {q: q for q in self.line_labels}
                standardtype = True
            if not standardtype:
                raise ValueError(
                    "No standard qubit labelling conversion is available! Please provide `qubit_conversion`.")

        if num_qubits is None:
            num_qubits = len(self.line_labels)

        # if gateargs_map is None:
        #     gateargs_map = {}

        #Currently only using 'Iz' as valid intermediate measurement ('IM') label.
        #Todo:  Expand to all intermediate measurements.
        if 'Iz' in self.str:
            # using_IMs = True
            num_IMs = self.str.count('Iz')
        else:
            # using_IMs = False
            num_IMs = 0
        num_IMs_used = 0

        # Init the openqasm string.
        openqasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\n'

        openqasm += 'qreg q[{0}];\n'.format(str(num_qubits))
        # openqasm += 'creg cr[{0}];\n'.format(str(num_qubits))
        openqasm += 'creg cr[{0}];\n'.format(str(num_qubits + num_IMs))
        openqasm += '\n'

        depth = self.num_layers

        def trivial_arg_map(gatearg):
            return ''

        # Go through the layers, and add the openqasm for each layer in turn.
        for l in range(depth):

            # Get the layer, without identity gates and containing each gate only once.
            layer = self.layer_label(l)
            # For keeping track of which qubits have a gate on them in the layer.
            qubits_used = []

            # Go through the (non-self.identity) gates in the layer and convert them to openqasm
            for gate in layer.components:
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                assert(len(gate_qubits) <= 2), 'Gates on more than 2 qubits given; this is currently not supported!'

                # Find the openqasm for the gate.
                if gate.name.__str__() != 'Iz':
                    openqasmlist_for_gate = gatename_conversion[gate.name]  # + gatearg_str

                    if isinstance(openqasmlist_for_gate, str):
                        gatearg_str = gateargs_map.get(gate.name, trivial_arg_map)(gate.args)
                        openqasmlist_for_gate = [openqasmlist_for_gate + gatearg_str, ]
                    else:
                        assert(gate.name not in gateargs_map.keys())

                    openqasm_for_gate = ''
                    for subopenqasm_for_gate in openqasmlist_for_gate:

                        #If gate.qubits is None, gate is assumed to be single-qubit gate
                        #acting in parallel on all qubits.
                        if gate.qubits is None:
                            for q in gate_qubits:
                                openqasm_for_gate += subopenqasm_for_gate + ' q[' + str(qubit_conversion[q]) + '];\n'
                            if block_between_gates:
                                openqasm_for_gate += 'barrier '
                                for q in self.line_labels[:-1]:
                                    openqasm_for_gate += 'q[{0}], '.format(str(qubit_conversion[q]))
                                openqasm_for_gate += 'q[{0}];\n'.format(str(qubit_conversion[self.line_labels[-1]]))

                        else:
                            openqasm_for_gate += subopenqasm_for_gate
                            for q in gate_qubits:
                                openqasm_for_gate += ' q[' + str(qubit_conversion[q]) + ']'
                                if q != gate_qubits[-1]:
                                    openqasm_for_gate += ', '
                            openqasm_for_gate += ';\n'
                            if block_between_gates:
                                openqasm_for_gate += 'barrier '
                                for q in self.line_labels[:-1]:
                                    openqasm_for_gate += 'q[{0}], '.format(str(qubit_conversion[q]))
                                openqasm_for_gate += 'q[{0}];\n'.format(str(qubit_conversion[self.line_labels[-1]]))

                else:
                    assert len(gate.qubits) == 1
                    q = gate.qubits[0]
                    # classical_bit = num_IMs_used
                    openqasm_for_gate = "measure q[{0}] -> cr[{1}];\n".format(str(qubit_conversion[q]), num_IMs_used)
                    num_IMs_used += 1

                # Add the openqasm for the gate to the openqasm string.
                openqasm += openqasm_for_gate

                # Keeps track of the qubits that have been accounted for, and checks that hadn't been used
                # although that should already be checked in the .layer_label(), which checks for its a valid
                # circuit layer.
                assert(not set(gate_qubits).issubset(set(qubits_used)))
                qubits_used.extend(gate_qubits)

            # All gates that don't have a non-idle gate acting on them get an idle in the layer.
            if not block_between_gates:
                for q in self.line_labels:
                    if q not in qubits_used:
                        openqasm += 'id' + ' q[' + str(qubit_conversion[q]) + '];\n'

            # Add in a barrier after every circuit layer if block_between_layers==True.
            # Including barriers is critical for QCVV testing, circuits should usually
            # experience minimal "behind-the-scenes" compilation (beyond necessary
            # conversion to native instructions).
            # To do: Add "barrier" as native pygsti circuit instruction, and use for indicating
            # where pragma blocks should be.
            if block_between_layers:
                openqasm += 'barrier '
                for q in self.line_labels[:-1]:
                    openqasm += 'q[{0}], '.format(str(qubit_conversion[q]))
                openqasm += 'q[{0}];\n'.format(str(qubit_conversion[self.line_labels[-1]]))
                # openqasm += ';'

        # Add in a measurement at the end.
        for q in self.line_labels:
            # openqasm += "measure q[{0}] -> cr[{1}];\n".format(str(qubit_conversion[q]), str(qubit_conversion[q]))
            openqasm += "measure q[{0}] -> cr[{1}];\n".format(str(qubit_conversion[q]),
                                                              str(num_IMs_used + qubit_conversion[q]))

        return openqasm

    def simulate(self, model, return_all_outcomes=False):
        """
        Compute the outcome probabilities of this Circuit using `model` as a model for the gates.

        The order of the outcome strings (e.g., '0100') is w.r.t. to the
        ordering of the qubits in the circuit. That is, the ith element of the
        outcome string corresponds to the qubit with label
        `self.qubit_labels[i]`.

        Parameters
        ----------
        model : Model
            A description of the gate and SPAM operations corresponding to the
            labels stored in this Circuit. If this model is over more qubits
            than the circuit, the output will be the probabilities for the qubits
            in the circuit marginalized over the other qubits. But, the simulation
            is over the full set of qubits in the model, and so the time taken for
            the simulation scales with the number of qubits in the model. For
            models whereby "spectator" qubits do not affect the qubits in this
            circuit (such as with perfect gates), more efficient simulations will
            be obtained by first creating a model only over the qubits in this
            circuit.

        return_all_outcomes : bool, optional
            Whether to include outcomes in the returned dictionary that have zero
            probability. When False, the threshold for discarding an outcome as z
            ero probability is 10^-12.

        Returns
        -------
        probs : dictionary
            A dictionary with keys equal to all (`return_all_outcomes` is True) or
            possibly only some (`return_all_outcomes` is False) of the possible
            outcomes, and values that are float probabilities.
        """
        # These results is a dict with strings of outcomes (normally bits) ordered according to the
        # state space ordering in the model.
        results = model.probabilities(self)

        # Mapping from the state-space labels of the model to their indices.
        # (e.g. if model.state_space_labels is [('Qa','Qb')] then sslInds['Qb'] = 1
        # (and 'Qb' may be a circuit line label)
        sslInds = {sslbl: i for i, sslbl in enumerate(model.state_space_labels.labels[0])}
        # Note: we ignore all but the first tensor product block of the state space.

        def process_outcome(outcome):
            """Relabels an outcome tuple and drops state space labels not in the circuit."""
            processed_outcome = []
            for lbl in outcome:  # lbl is a string - an instrument element or POVM effect label, e.g. '010'
                relbl = ''.join([lbl[sslInds[ll]] for ll in self.line_labels])
                processed_outcome.append(relbl)
                #Note: above code *assumes* that each state-space label (and so circuit line label)
                # corresponds to a *single* letter of the instrument/POVM label `lbl`.  This is almost
                # always the case, as state space labels are usually qubits and so effect labels are
                # composed of '0's and '1's.
            return tuple(processed_outcome)

        processed_results = _ld.OutcomeLabelDict()
        for outcome, pr in results.items():
            if return_all_outcomes or pr > 1e-12:  # then process & accumulate pr
                p_outcome = process_outcome(outcome)  # rearranges and drops parts of `outcome`
                if p_outcome in processed_results:  # (may occur b/c processing can map many-to-one)
                    processed_results[p_outcome] += pr  # adding marginalizes the results.
                else:
                    processed_results[p_outcome] = pr

        return processed_results

    def done_editing(self):
        """
        Make this circuit read-only, so that it can be hashed (e.g. used as a dictionary key).

        This is done automatically when attempting to hash a :class:`Circuit`
        for the first time, so there's calling this function can usually be
        skipped (but it's good for code clarity).

        Returns
        -------
        None
        """
        if not self._static:
            self._static = True
            self._labels = tuple([_Label(layer_lbl) for layer_lbl in self._labels])

    def expand_instruments_and_separate_povm(self, model, observed_outcomes=None):
        """
        Creates a dictionary of :class:`SeparatePOVMCircuit` objects from expanding the instruments of this circuit.

        Each key of the returned dictionary replaces the instruments in this circuit with a selection
        of their members.  (The size of the resulting dictionary is the product of the sizes of
        each instrument appearing in this circuit when `observed_outcomes is None`).  Keys are stored
        as :class:`SeparatePOVMCircuit` objects so it's easy to keep track of which POVM outcomes (effects)
        correspond to observed data.  This function is, for the most part, used internally to process
        a circuit before computing its outcome probabilities.

        Parameters
        ----------
        model : TODO docstring

        Returns
        -------
        OrderedDict
            A dict whose keys are :class:`SeparatePOVMCircuit` objects and whose
            values are tuples of the outcome labels corresponding to this circuit,
            one per POVM effect held in the key.
        """
        complete_circuit = model.complete_circuit(self)
        expanded_circuit_outcomes = _collections.OrderedDict()
        povm_lbl = complete_circuit[-1]  # "complete" circuits always end with a POVM label
        circuit_without_povm = complete_circuit[0:len(complete_circuit) - 1]

        def create_tree(lst):
            subs = _collections.OrderedDict()
            for el in lst:
                if len(el) > 0:
                    if el[0] not in subs: subs[el[0]] = []
                    subs[el[0]].append(el[1:])
            return _collections.OrderedDict([(k, create_tree(sub_lst)) for k, sub_lst in subs.items()])

        def add_expanded_circuit_outcomes(circuit, running_outcomes, ootree, start):
            """
            """
            cir = circuit if start == 0 else circuit[start:]  # for performance, avoid uneeded slicing
            for k, layer_label in enumerate(cir, start=start):
                components = layer_label.components
                #instrument_inds = _np.nonzero([model._is_primitive_instrument_layer_lbl(component)
                #                               for component in components])[0]  # SLOWER than statement below
                instrument_inds = _np.array([i for i, component in enumerate(components)
                                             if model._is_primitive_instrument_layer_lbl(component)])
                if instrument_inds.size > 0:
                    # This layer contains at least one instrument => recurse with instrument(s) replaced with
                    #  all combinations of their members.
                    component_lookup = {i: comp for i, comp in enumerate(components)}
                    instrument_members = [model._member_labels_for_instrument(components[i])
                                          for i in instrument_inds]  # also components of outcome labels
                    for selected_instrmt_members in _itertools.product(*instrument_members):
                        expanded_layer_lbl = component_lookup.copy()
                        expanded_layer_lbl.update({i: components[i] + "_" + sel
                                                   for i, sel in zip(instrument_inds, selected_instrmt_members)})
                        expanded_layer_lbl = _Label([expanded_layer_lbl[i] for i in range(len(components))])

                        if ootree is not None:
                            new_ootree = ootree
                            for sel in selected_instrmt_members:
                                new_ootree = new_ootree.get(sel, {})
                            if len(new_ootree) == 0: continue  # no observed outcomes along this outcome-tree path
                        else:
                            new_ootree = None

                        add_expanded_circuit_outcomes(circuit[0:k] + Circuit((expanded_layer_lbl,)) + circuit[k + 1:],
                                                      running_outcomes + selected_instrmt_members, new_ootree, k + 1)
                    break

            else:  # no more instruments to process: `cir` contains no instruments => add an expanded circuit
                assert(circuit not in expanded_circuit_outcomes)  # shouldn't be possible to generate duplicates...
                elabels = model._effect_labels_for_povm(povm_lbl) if (observed_outcomes is None) \
                    else tuple(ootree.keys())
                outcomes = tuple((running_outcomes + (elabel,) for elabel in elabels))
                expanded_circuit_outcomes[SeparatePOVMCircuit(circuit, povm_lbl, elabels)] = outcomes

        ootree = create_tree(observed_outcomes) if observed_outcomes is not None else None  # tree of observed outcomes
        # e.g. [('0','00'), ('0','01'), ('1','10')] ==> {'0': {'00': {}, '01': {}}, '1': {'10': {}}}

        if model._has_instruments():
            add_expanded_circuit_outcomes(circuit_without_povm, (), ootree, start=0)
        else:
            # It may be helpful to cache the set of elabels for a POVM (maybe within the model?) because
            # currently the call to _effect_labels_for_povm may be a bottleneck.  It's needed, even when we have
            # observed outcomes, because there may be some observed outcomes that aren't modeled (e.g. leakage states)
            if observed_outcomes is None:
                elabels = model._effect_labels_for_povm(povm_lbl)
            else:
                possible_lbls = set(model._effect_labels_for_povm(povm_lbl))
                elabels = tuple([oo for oo in ootree.keys() if oo in possible_lbls])
            outcomes = tuple(((elabel,) for elabel in elabels))
            expanded_circuit_outcomes[SeparatePOVMCircuit(circuit_without_povm, povm_lbl, elabels)] = outcomes

        return expanded_circuit_outcomes


class CompressedCircuit(object):
    """
    A "compressed" Circuit that requires less disk space.

    The tuple of circuit layers is compressed using a custom algorithm which looks for
    repeated portions of the circuit.

    One place where CompressedCircuit objects can be useful is when saving
    large lists of long operation sequences in some non-human-readable format (e.g.
    pickle).  CompressedCircuit objects *cannot* be used in place of
    Circuit objects within pyGSTi, and so are *not* useful when manipulating
    and running algorithms which use operation sequences.

    Parameters
    ----------
    circuit : Circuit
        The operation sequence object which is compressed to create
        a new CompressedCircuit object.

    min_len_to_compress : int, optional
        The minimum length string to compress.  If len(circuit)
        is less than this amount its tuple is returned.

    max_period_to_look_for : int, optional
        The maximum period length to use when searching for periodic
        structure within circuit.  Larger values mean the method
        takes more time but could result in better compressing.
    """

    def __init__(self, circuit, min_len_to_compress=20, max_period_to_look_for=20):
        """
        Create a new CompressedCircuit object

        Parameters
        ----------
        circuit : Circuit
            The operation sequence object which is compressed to create
            a new CompressedCircuit object.

        min_len_to_compress : int, optional
            The minimum length string to compress.  If len(circuit)
            is less than this amount its tuple is returned.

        max_period_to_look_for : int, optional
            The maximum period length to use when searching for periodic
            structure within circuit.  Larger values mean the method
            takes more time but could result in better compressing.
        """
        if not isinstance(circuit, Circuit):
            raise ValueError("CompressedCircuits can only be created from existing Circuit objects")
        self._tup = CompressedCircuit.compress_op_label_tuple(
            circuit.layertup, min_len_to_compress, max_period_to_look_for)
        self._str = circuit.str
        self._line_labels = circuit.line_labels
        self._occurrence_id = circuit.occurrence

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state_dict):
        for k, v in state_dict.items():
            if k == 'tup':
                self._tup = state_dict['tup']  # backwards compatibility
            elif k == 'str':
                self._str = state_dict['str']  # backwards compatibility
            else:
                if k == "line_labels": k = "_line_labels"  # add underscore
                self.__dict__[k] = v
        if '_line_labels' not in state_dict and "line_labels" not in state_dict:
            self._line_labels = ('*',)

    def expand(self):
        """
        Expands this compressed operation sequence into a Circuit object.

        Returns
        -------
        Circuit
        """
        tup = CompressedCircuit.expand_op_label_tuple(self._tup)
        occurrence = self._occurrence_id if hasattr(self, '_occurrence_id') else None  # backward compatibility
        return Circuit(tup, self._line_labels, editable=False, stringrep=self._str,
                       check=False, occurrence=occurrence)

    @staticmethod
    def _get_num_periods(circuit, period_len):
        n = 0
        if len(circuit) < period_len: return 0
        while circuit[0:period_len] == circuit[n * period_len:(n + 1) * period_len]:
            n += 1
        return n

    @staticmethod
    def compress_op_label_tuple(circuit, min_len_to_compress=20, max_period_to_look_for=20):
        """
        Compress a operation sequence.

        The result is tuple with a special compressed- gate-string form form
        that is not useable by other GST methods but is typically shorter
        (especially for long operation sequences with a repetative structure)
        than the original operation sequence tuple.

        Parameters
        ----------
        circuit : tuple of operation labels or Circuit
            The operation sequence to compress.

        min_len_to_compress : int, optional
            The minimum length string to compress.  If len(circuit)
            is less than this amount its tuple is returned.

        max_period_to_look_for : int, optional
            The maximum period length to use when searching for periodic
            structure within circuit.  Larger values mean the method
            takes more time but could result in better compressing.

        Returns
        -------
        tuple
            The compressed (or raw) operation sequence tuple.
        """
        circuit = tuple(circuit)  # converts from Circuit or list to tuple if needed
        L = len(circuit)
        if L < min_len_to_compress: return tuple(circuit)
        compressed = ["CCC"]  # list for appending, then make into tuple at the end
        start = 0
        while start < L:
            #print "Start = ",start
            score = _np.zeros(max_period_to_look_for + 1, 'd')
            numperiods = _np.zeros(max_period_to_look_for + 1, _np.int64)
            for periodLen in range(1, max_period_to_look_for + 1):
                n = CompressedCircuit._get_num_periods(circuit[start:], periodLen)
                if n == 0: score[periodLen] = 0
                elif n == 1: score[periodLen] = 4.1 / periodLen
                else: score[periodLen] = _np.sqrt(periodLen) * n
                numperiods[periodLen] = n
            bestPeriodLen = _np.argmax(score)
            n = numperiods[bestPeriodLen]
            bestPeriod = circuit[start:start + bestPeriodLen]
            #print "Scores = ",score
            #print "num per = ",numperiods
            #print "best = %s ^ %d" % (str(bestPeriod), n)
            assert(n > 0 and bestPeriodLen > 0)
            if start > 0 and n == 1 and compressed[-1][1] == 1:
                compressed[-1] = (compressed[-1][0] + bestPeriod, 1)
            else:
                compressed.append((bestPeriod, n))
            start = start + bestPeriodLen * n

        return tuple(compressed)

    @staticmethod
    def expand_op_label_tuple(compressed_op_labels):
        """
        Expand a compressed tuple (created with :method:`compress_op_label_tuple`) into a tuple of operation labels.

        Parameters
        ----------
        compressed_op_labels : tuple
            a tuple in the compressed form created by
            compress_op_label_tuple(...).

        Returns
        -------
        tuple
            A tuple of operation labels specifying the uncompressed operation sequence.
        """
        if len(compressed_op_labels) == 0: return ()
        if compressed_op_labels[0] != "CCC": return compressed_op_labels
        expandedString = []
        for (period, n) in compressed_op_labels[1:]:
            expandedString += period * n
        return tuple(expandedString)


class SeparatePOVMCircuit(object):
    """
    Separately holds a POVM-less :class:`Circuit` object, a POVM label, and a list of effect labels.

    This is often used to hold "expanded" circuits whose instrument labels have been replaced with
    specific instrument members and whose POVMs have simillarly been "expanded" except that we keep
    the entire expanded POVM together in this one data structure.  (There's no especially good reason
    for this other than practicality - that since almost *all* circuits end with a POVM, holding each
    POVM outcome (effect) separately would be very wasteful.
    """
    def __init__(self, circuit_without_povm, povm_label, effect_labels):
        self.circuit_without_povm = circuit_without_povm
        self.povm_label = povm_label
        self.effect_labels = effect_labels

    @property
    def full_effect_labels(self):
        return [(self.povm_label + "_" + el) for el in self.effect_labels]

    def __len__(self):
        return len(self.circuit_without_povm)  # don't count POVM in length, so slicing works as expected

    def __getitem__(self, index):
        return SeparatePOVMCircuit(self.circuit_without_povm[index], self.povm_label, self.effect_labels)

    def __lt__(self, other):  # so we can sort a list of SeparatePOVMCircuits
        return self.circuit_without_povm < other.circuit_without_povm

    def __str__(self):
        return "SeparatePOVM(" + self.circuit_without_povm.str + "," \
            + str(self.povm_label) + "," + str(self.effect_labels) + ")"

    #LATER: add a method for getting the "POVM_effect" labels?
