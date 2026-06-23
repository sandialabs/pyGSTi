"""
Defines the Label class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

from typing import Union, Optional, Literal, Any, Sequence, Callable
from warnings import warn
import itertools as _itertools
import numbers as _numbers
import sys as _sys
import copy as _copy
import numpy as _np


# Unclear why we use _numbers.Integral below. We had it in isinstance checks
# before adding type annotations.
StateSpaceLabels     = tuple[Union[str, _numbers.Integral, int], ...]
StateSpaceLabelsCastable =   Union[str, _numbers.Integral, int, StateSpaceLabels]

# TODO: define a ConcreteLabel protocol! (See type alias at EOF.)


def _integerize_sslbls(state_space_labels):
    """Convert string-integers to ints in state space labels."""
    if state_space_labels is None:
        return None

    if not isinstance(state_space_labels, (tuple, list)):
        state_space_labels = (state_space_labels,)

    integerized_sslbls = []
    for ssl in state_space_labels:
        try:
            integerized_sslbls.append(int(ssl))
        except (ValueError, TypeError):
            # Keep as string if not an integer
            integerized_sslbls.append(_sys.intern(ssl))
    return tuple(integerized_sslbls)


def _check_concate_compatibility(label1, label2):
    """
    Checks for qubit overlap and time conflicts between two labels, and
    "unwraps" a depth-1 CircuitLabel.
    Returns the (possibly unwrapped) second label and the new time for the
    concatenated label.
    """
    # 1. Qubit overlap check
    if label1.sslbls and label2.sslbls and set(label1.sslbls) & set(label2.sslbls):
        raise ValueError("Cannot concatenate labels with overlapping qubits")

    # 2. Time compatibility check
    time1 = getattr(label1, 'time', 0.0)
    time2 = getattr(label2, 'time', 0.0)
    if time1 != 0.0 and time2 != 0.0 and time1 != time2:
        warn(f"Trying to concate two Labels with distinct time values {time1}, and {time2}", RuntimeWarning)
    new_time = max(time1, time2)

    # 3. CircuitLabel unwrapping
    if isinstance(label2, CircuitLabel):
        if label2.depth == 1:
            label2 = Label(label2.components, label2.sslbls, label2.time, label2.args)
        else:
            raise ValueError(f"Trying to concate a CircuitLabel with depth: {label2.depth} to a lbl with depth {label1.depth}")

    return label2, new_time


def _create_concatenated_label(components, time, args):
    """
    Creates a LabelTupTup, LabelTupTupWithTime, or LabelTupTupWithArgs based on
    the presence of time and args.
    """
    if args:
        return LabelTupTupWithArgs.init(components, time=time, args=args)
    elif time != 0.0:
        return LabelTupTupWithTime.init(components, time=time)
    else:
        return LabelTupTup.init(components)


class Label(object):
    """
    A label used to identify a gate, circuit layer, or (sub-)circuit.

    A label consisting of a string along with a tuple of
    integers or sector-names specifying which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by an object so-labeled.
    """

    # this is just an abstract base class for isinstance checking.
    # actual labels will either be LabelTup or LabelStr instances,
        # depending on whether the tuple of sector names exists or not.
        # (the reason for separate classes is for hashing speed)

    def __new__(cls, name: Any, state_space_labels: Optional[StateSpaceLabels]=None, time=None, args=None) -> ConcreteLabel:
        """
        Creates a new Model-item label, which is divided into a simple string
        label and a tuple specifying the part of the Hilbert space upon which the
        item acts (often just qubit indices).

        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.

        state_space_labels : list or tuple, optional
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.  If something other than
            a list or tuple is passed, a single-element tuple is created
            containing the passed object.

        time : float
            The time at which this label occurs (can be relative or absolute)

        args : iterable of hashable types, optional
            A list of "arguments" for this label.  Having arguments makes the
            Label even more resemble a function call, and supplies parameters
            for the object (often a gate or layer operation) being labeled that
            are fixed at circuit-creation time (i.e. are not optimized over).
            For example, the angle of a continuously-variable X-rotation gate
            could be an argument of a gate label, and one might create a label:
            `Label('Gx', (0,), args=(pi/3,))`
        """
        
        time_is_present = False      
        if isinstance(name, Label) and state_space_labels is None:
            return name  # type: ignore
            # ^ Note: Labels are immutable, so no need to copy

        if isinstance(name, str) and state_space_labels is None and time is None and args is None:
            if ':' in name or '!' in name or ';' in name:
                try:
                    from pygsti.circuits.circuitparser import parse_label
                    return parse_label(name)
                except Exception:
                    pass  # if parsing fails, just treat as a normal (un-parsable) string.

        if time is None and args is None:
            match name:
                # We need the | list() because we lose the tuple nature when written to a json file.
                case [float() | None as ctime, tuple() | list() as cargs, tuple() | list() as tup]:
                    return LabelTupTupWithArgs.init((Label(tmp) for tmp in tup), time=ctime, args=cargs)
                case [float() as ctime, None as cargs, tuple() | list() as tup]:
                    return LabelTupTupWithTime.init((Label(tmp) for tmp in tup), time=ctime)
                case [None as ctime, None as cargs, tuple() | list() as tup]: # Just a single tuple.
                    return LabelTupTup.init((Label(tmp) for tmp in tup))

        if isinstance(name, (tuple, list)) and state_space_labels is None:
            # We're being asked to initialize from a non-string with no
            # state_space_labels, explicitly given.  `name` could either be:
            # 0) an empty tuple: () -> LabelTupTup with *no* subLabels.
            # 1) a (name, ssl0, ssl1, ...) tuple -> LabelTup
            # 2) a (subLabel1_tup, subLabel2_tup, ...) tuple -> LabelTupTup if
            #     length > 1 otherwise just initialize from subLabel1_tup.
            # 3) a (str, sslbls, reps, subLabel1_tup, subLabel2_tup, ...) tuple -> CircuitLabel.
            #      even if there is only one subLabel we want to make it a CircuitLabel.
            # Note: subLabelX_tup could also be identified as a Label object
            #       (even a LabelStr)

            if len(name) == 0:
                if args: return LabelTupTupWithArgs.init((), time, args)
                elif time is None or time == 0:
                    return LabelTupTup.init(())
                else:
                    return LabelTupTupWithTime.init((), time)
            elif isinstance(name[0], (tuple, list, Label)):
                if len(name) > 1:
                    if args: return LabelTupTupWithArgs.init(name, time=time, args=args)
                    elif time is not None and time != 0.0: return LabelTupTupWithTime.init(name, time=time)
                    else: return LabelTupTup.init(name)
                else:
                    return Label(name[0], time=time, args=args)
            elif len(name) >= 3 and isinstance(name[0], str) and \
                (isinstance(name[1], tuple) or name[1] is None) and isinstance(name[2], int):
                # We are building a CircuitLabel.
                loc_name = name[0]
                sslbls = name[1]
                reps = name[2]
                tup = ()
                for x in name[3:]:
                    tup = (*tup, Label(x, time=time))
                return CircuitLabel(loc_name, tup, sslbls, reps, time)
            else:
                #Case when state_space_labels, etc, are given after name in a single tuple
                tup = name
                name = tup[0]
                tup_args = []; state_space_labels = []
                next_is_arg = False
                next_is_time = False
                for x in tup[1:]:
                    if next_is_arg:
                        next_is_arg = False
                        tup_args.append(x); continue
                    if next_is_time:
                        next_is_time = False
                        time = x; time_is_present = True; continue

                    if isinstance(x, str):
                        if x.startswith(';'):
                            assert(args is None), "Cannot supply args in tuple when `args` is given!"
                            if x == ';':
                                next_is_arg = True
                            else:
                                tup_args.append(x[1:])
                            continue
                        if x.startswith('!'):
                            assert(time is None), "Cannot supply time in tuple when `time` is given!"
                            time_is_present = True
                            if x == '!':
                                next_is_time = True
                            else:
                                time = float(x[1:])
                            continue
                    state_space_labels.append(x)

                # Take the args that were passed to the function.
                #Note that we will error out if tup args were to be set earlier and args was set.
                args = tup_args if len(tup_args) > 0 else args
                state_space_labels = tuple(state_space_labels)  # needed for () and (None,) comparison below

        if time is None:
            time = 0.0  # for non-TupTup labels not setting a time is equivalent to setting it to 0.0

        #print(" -> preproc with name=", name, "sslbls=", state_space_labels, "t=", time, "args=", args)
        # If numpy object, we have to check size=0 for empty; otherwise, check for empty tuple
        if state_space_labels is None \
            or (isinstance(state_space_labels, (_np.ndarray, _np.generic)) and state_space_labels.size == 0) \
            or (not isinstance(state_space_labels, (_np.ndarray, _np.generic)) and state_space_labels in ((), (None,))):
            if args is not None:
                return LabelTupWithArgs.init(name, (), time, args)  # just use empty sslbls
            else:
                return LabelStr.init(name, time)

        else:
            if args is not None:
                return LabelTupWithArgs.init(name, state_space_labels, time, args)
            else:
                if time == 0.0 and not time_is_present:
                    return LabelTup.init(name, state_space_labels)
                else:
                    return LabelTupWithTime.init(name, state_space_labels, time if time is not None else 0.0)

    @property
    def args(self) -> tuple:
        raise NotImplementedError()
    
    @property
    def components(self) -> tuple:
        raise NotImplementedError()

    @property
    def depth(self) -> int:
        """
        The depth of this label, viewed as a sub-circuit.
        """
        return 1  # most labels are depth=1

    @property
    def is_sorted(self) -> bool:
        """
        Whether the internal labels are sorted in increasing qubit order.
        """
        return self._is_sorted
    
    @is_sorted.setter
    def is_sorted(self, val: bool):
        self._is_sorted = val

    @property
    def qubits(self) -> Optional[StateSpaceLabels]:
        """
        An alias for sslbls, since commonly these are just qubit indices. (a tuple)
        """
        return self.sslbls

    @property
    def num_qubits(self) -> Optional[int]:
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all") -> bool:
        """
        Whether this label has the given `prefix`.

        Usually used to test whether the label names a given type.

        Parameters
        ----------
        prefix : str
            The prefix to check for.

        typ : {"any","all"}
            Whether, when there are multiple parts to the label, the prefix
            must occur in any or all of the parts.

        Returns
        -------
        bool
        """
        return self.name.startswith(prefix)

    @property
    def reps(self) -> int:
        """
        Number of repetitions (of this label's components) that this label represents.
        """
        return 1  # most labels have only reps==1

    @property
    def has_nontrivial_components(self) -> bool:
        return len(self.components) > 0 and self.components != (self,)

    def collect_args(self) -> tuple:  # or is it Optional[tuple] ??
        if not self.has_nontrivial_components:
            return self.args
        else:
            ret = list(self.args)
            for c in self.components:
                ret.extend(c.collect_args())
            return tuple(ret)

    def strip_args(self):
        # default, appropriate for a label without args or components
        return self

    def expand_subcircuits(self):
        """
        Expand any sub-circuits within this label.

        Returns a list of component labels which doesn't include any
        :class:`CircuitLabel` labels.  This effectively expands any "boxes" or
        "exponentiation" within this label.

        Returns
        -------
        tuple
            A tuple of component Labels (none of which should be
            :class:`CircuitLabel` objects).
        """
        return (self,)  # most labels just expand to themselves
    
    @property
    def is_simple(self) -> bool:
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.    
        """
        return self.IS_SIMPLE  # type: ignore

    def with_sorted_inner_labels(self) -> ConcreteLabel:
        """
        Returns `self` if either (1) we aren't a LabelTupTup or (2) we are a LabelTupTup,
        but one or more inner Label objects have sslbls == None.

        In all other situations, we return a LabelTupTup whose inner labels are sorted
        according to their sslbls. This may raise an error if any of the sslbls are not
        comparable to one another.
        
        Note that we are sorting in increasing order of the first qubit that the label
        acts on and respect the order of the sslbls for each label.
        """

        if not isinstance(self, LabelTupTup):
            return self  # type: ignore
        
        assert isinstance(self, tuple)
        # ^ make static analysis tools happy
        assert hasattr(self, 'components')
        
        if len(self) <= 1:
            # The inner labels are trivially sorted.
            return self  # type: ignore
        
        if self.is_sorted:
            return self

        if isinstance(self, CircuitLabel):
            # Each of the components are assumed to define a distinct layer in the circuit.
            # This is the case even if the circuit could be parallelized to have smaller depth.
            # As such we do not want to sort at this level but recurse until we reach the level of
            # multiple labels for the same layer. Then, we can sort those based upon the statespace
            # labels they are acting on.

            ret = CircuitLabel(self.name, [layer.with_sorted_inner_labels() for layer in self.components],
                               self.sslbls, self.reps, self.time)
            ret._is_sorted = True
            return ret

        tmp1 = dict()
        for inner in self.components:
            sslbls = inner.sslbls
            if sslbls is None:
                return self  # type: ignore
            tmp1[sslbls] = inner.with_sorted_inner_labels() # Recurse so that all parallel sections are in sorted order locally.
        tmp2 = tuple( (tmp1[k] for k in sorted(tmp1.keys())) )

        if len(tmp2) != len(self.components):
            msg = f"duplicate sslbls among inner labels of = {self}"
            raise ValueError(msg)
        
        args = None if getattr(self, 'args', ()) == () else self.args  # type: ignore
        # ^ We override an empty tuple with None to make sure we hit the right
        #   codepath in Label.__new__.
        out = Label(tmp2,
            time=getattr(self, 'time', None),
            args=args
        )
        out.is_sorted = True
        # ^ We don't pass state_space_labels=self.sslbls in order to make sure
        #   we hit the right codepath in Label.__new__; all codepaths that
        #   lead to LabelTupTup-like objects require state_space_labels=None.
        return out

    def copy(self):
        return _copy.deepcopy(self)

    def replace_name(self, oldname, newname):
        raise NotImplementedError()
    
    def concate(self, other: Label):
        """
        Combine two labels together so that they are one label which could be a single layer.
        """
        raise NotImplementedError("Derived Classes must implement this function.")


class LabelTup(Label, tuple):
    """
    A label consisting of a string along with a tuple of integers or state-space-names.

    These state-space sector names specify which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by the object this label refers to.
    """

    IS_SIMPLE = True  # access with self.is_simple property

    @classmethod
    def init(cls, name: str, state_space_labels: StateSpaceLabelsCastable):
        """
        Creates a new Model-item label.

        The created label is comprised of a simple string label and a tuple
        specifying the part of the Hilbert space upon which the item acts
        (often just qubit indices).

        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.

        state_space_labels : list or tuple
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.  If something other than
            a list or tuple is passed, a single-element tuple is created
            containing the passed object.

        Returns
        -------
        LabelTup
        """

        #Type checking
        assert(isinstance(name, str)), "`name` must be a string, but it's '%s'" % str(name)
        assert(state_space_labels is not None), "LabelTup must be initialized with non-None state-space labels"
        if not isinstance(state_space_labels, (tuple, list)):
            state_space_labels = (state_space_labels,)
        for ssl in state_space_labels:
            assert(isinstance(ssl, str) or isinstance(ssl, _numbers.Integral)), \
                "State space label '%s' must be a string or integer!" % str(ssl)

        #Try to convert integer-strings to ints (for parsing from files...)
        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = _integerize_sslbls(state_space_labels)
        tup = (_sys.intern(name),) + sslbls
        obj = tuple.__new__(cls, tup)
        obj._is_sorted = True
        return obj

    __new__ = tuple.__new__

    @property
    def is_sorted(self) -> Literal[True]:
        """
        There is only ever one gate recognized by this label.
        """
        return True

    @property
    def name(self) -> str:
        """
        This label's name (a string).
        """
        return self[0]

    @property
    def sslbls(self) -> Optional[StateSpaceLabels]:
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) > 1:
            return self[1:]
        else: return None

    @property
    def args(self) -> tuple:  # empty tuple
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self) -> tuple[LabelTup]:  # length-1
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return (self,)  # just a single "sub-label" component

    def map_state_space_labels(self, mapper) -> LabelTup:
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        Label
        """
        sslbls = self.sslbls
        if not isinstance(sslbls, tuple):
            raise ValueError()
        if isinstance(mapper, dict):
            mapped_sslbls = [mapper[sslbl] for sslbl in sslbls]
        else:  # assume mapper is callable
            mapped_sslbls = [mapper(sslbl) for sslbl in sslbls]
        return LabelTup.init(self.name, mapped_sslbls)

    def __str__(self) -> str:
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        s = str(self.name)
        if self.sslbls:  # test for None and len == 0
            s += ":" + ":".join(map(str, self.sslbls))
        return s

    def __repr__(self) -> str:
        return "Label(" + repr(self[:]) + ")"

    def __add__(self, s: str) -> LabelTup:
        if isinstance(s, str):
            return LabelTup.init(self.name + s, self.sslbls)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other) -> bool:
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        return tuple.__eq__(self, other)

    def __lt__(self, x) -> bool:
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x) -> bool:
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self) -> tuple[type, tuple, None]:
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTup, (self[:],), None)

    def to_native(self) -> tuple:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return tuple(self)

    def replace_name(self, oldname: str, newname: str) -> LabelTup:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelTup
        """
        return LabelTup.init(newname, self.sslbls) if (self.name == oldname) else self

    def concate(self, other: Label):
        other, new_time = _check_concate_compatibility(self, other)

        if isinstance(other, LabelTupTup):
            components = (self,) + tuple(other)
        elif isinstance(other, LabelTup):
            components = (self, other)
        elif isinstance(other, LabelStr):
            raise ValueError("Cannot concate `LabelStr` as they do not have an associated qubit set unless it is explicitly in the string.")
        else:
            return super().concate(other)

        all_args = self.collect_args() + other.collect_args()
        return _create_concatenated_label(components, new_time, all_args)

    __hash__ = tuple.__hash__
    # ^ this is why we derive from tuple - using the
    #   native tuple.__hash__ directly == speed boost


class LabelTupWithTime(LabelTup, tuple):
    """
    A label consisting of a string along with a tuple of integers or state-space-names.

    These state-space sector names specify which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by the object this label refers to.
    """

    @classmethod
    def init(cls, name: str, state_space_labels: StateSpaceLabelsCastable, time=0.0) -> LabelTupWithTime:
        """
        Creates a new Model-item label.

        The created label is comprised of a simple string label and a tuple
        specifying the part of the Hilbert space upon which the item acts
        (often just qubit indices).

        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.

        state_space_labels : list or tuple
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.  If something other than
            a list or tuple is passed, a single-element tuple is created
            containing the passed object.

        time : float
            The time at which this label occurs (can be relative or absolute)

        Returns
        -------
        LabelTupWithTime
        """

        #Type checking
        assert(isinstance(name, str)), "`name` must be a string, but it's '%s'" % str(name)
        assert(state_space_labels is not None), "LabelTupWithTime must be initialized with non-None state-space labels"
        assert(isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        if not isinstance(state_space_labels, (tuple, list)):
            state_space_labels = (state_space_labels,)
        for ssl in state_space_labels:
            assert(isinstance(ssl, str) or isinstance(ssl, _numbers.Integral)), \
                "State space label '%s' must be a string or integer!" % str(ssl)

        #Try to convert integer-strings to ints (for parsing from files...)
        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = _integerize_sslbls(state_space_labels)
        tup = (_sys.intern(name),) + sslbls
        return cls.__new__(cls, tup, time)

    def __new__(cls, prepended_tup : tuple[Any, ...], time=0.0) -> LabelTupWithTime:
        # require len(prepended_tup) >= 1
        # require prepended_tup[0] is a str
        ret = tuple.__new__(cls, prepended_tup)  # creates a LabelTupWithTime object using tuple's __new__
        ret.time : float = time  # type: ignore
        return ret
    
    @property
    def time(self) -> float:
        """
        The time value associated with this label.
        """
        return self._time
    
    @time.setter
    def time(self, val: float):
        self._time = val

    def map_state_space_labels(self, mapper) -> LabelTupWithTime:
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        Label
        """
        if isinstance(mapper, dict):
            mapped_sslbls = [mapper[sslbl] for sslbl in self.sslbls]
        else:  # assume mapper is callable
            mapped_sslbls = [mapper(sslbl) for sslbl in self.sslbls]
        mapped_prepended_tup = (self.name,) + tuple(mapped_sslbls)
        time :  float = self.time  # type: ignore
        return LabelTupWithTime.__new__(LabelTupWithTime, mapped_prepended_tup, time=time)

    def __str__(self) -> str:
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        #caller = inspect.getframeinfo(inspect.currentframe().f_back)
        #ky = "%s:%s:%d" % (caller[2],os.path.basename(caller[0]),caller[1])
        #_debug_record[ky] = _debug_record.get(ky, 0) + 1
        s = str(self.name)
        if self.sslbls:  # test for None and len == 0
            s += ":" + ":".join(map(str, self.sslbls))
        if self.time != 0.0:
            s += ("!%f" % self.time).rstrip('0').rstrip('.')
        return s

    def __repr__(self) -> str:
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __add__(self, s: str) -> LabelTupWithTime:
        if isinstance(s, str):
            return LabelTupWithTime.init(self.name + s, self.sslbls, self.time)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __reduce__(self) -> tuple[type, tuple[tuple, float], None]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupWithTime, (self[:], self.time), None)

    def to_native(self) -> tuple:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        if self.time != 0.0:
            return tuple(self) + ('!' + str(self.time),)
        return tuple(self)

    def replace_name(self, oldname: str, newname: str) -> LabelTupWithTime:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelTupWithTime
        """
        if self.name != oldname:
            return self
        sslbls = self.sslbls
        new_tup = (newname,) if sslbls is None else ((newname,) + sslbls)
        time : float = self.time  # type: ignore
        return LabelTupWithTime.__new__(LabelTupWithTime, new_tup, time)

class LabelStr(Label, str):
    """
    A string-valued label.

    A Label for the special case when only a name is present (no
    state-space-labels).  We create this as a separate class
    so that we can use the string hash function in a
    "hardcoded" way - if we put switching logic in __hash__
    the hashing gets *much* slower.
    """

    IS_SIMPLE = True  # access with self.is_simple property

    @classmethod
    def init(cls, name: str, time: float=0.0) -> LabelStr:
        """
        Creates a new Model-item label, which is just a simple string label.

        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.

        time : float
            The time at which this label occurs (can be relative or absolute)

        Returns
        -------
        LabelStr
        """
        #Type checking
        assert(isinstance(name, str)), "`name` must be a string, but it's '%s'" % str(name)
        assert(isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        return cls.__new__(cls, name, time)

    def __new__(cls, name, time=0.0) -> LabelStr:
        ret = str.__new__(cls, name)
        ret.time = time
        ret._is_sorted = True # there is only one value so it must be sorted.
        return ret

    @property
    def is_sorted(self) -> Literal[True]:
        """
        There is only ever one gate recognized by this label.
        """
        return True

    @property
    def name(self) -> str:
        """
        This label's name (a string).
        """
        return str(self[:])

    @property
    def sslbls(self) -> None:
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        return None

    @property
    def args(self) -> tuple:  # empty tuple
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self) -> tuple[LabelStr]:  # length-1
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return (self,)  # just a single "sub-label" component

    def __str__(self) -> str:
        s = self[:]  # converts to a normal str
        if self.time != 0.0:
            s += ("!%f" % self.time).rstrip('0').rstrip('.')
        return s

    def __repr__(self) -> str:
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __add__(self, s: str) -> LabelStr:
        if isinstance(s, str):
            return LabelStr(self.name + str(s), time=self.time)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other: str) -> bool:
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        # NOTE: does not depend on self.time!
        return str.__eq__(self, other)

    def __lt__(self, x) -> bool:
        # NOTE: does not depend on self.time!
        return str.__lt__(self, str(x))

    def __gt__(self, x) -> bool:
        # NOTE: does not depend on self.time!
        return str.__gt__(self, str(x))

    def __pygsti_reduce__(self) -> tuple[type, tuple[str, float], None]:
        return self.__reduce__()

    def __reduce__(self) -> tuple[type, tuple[str, float], None]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelStr, (str(self), self.time), None)
    
    def __contains__(self, x) -> bool:
        #need to get a string rep of the tested label.
        return str(x) in str(self)

    def to_native(self) -> str:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        str
        """
        if self.time != 0.0:
            return (str(self), '!' + str(self.time))
        return str(self)

    def replace_name(self, oldname: str, newname: str) -> LabelStr:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelStr
        """
        return LabelStr(newname) if (self.name == oldname) else self

    def concate(self, other):
        raise ValueError("Cannot concate a `LabelStr` to another `Label` as `LabelStr` " \
                        "do not have a qubits associated without parsing.")

    __hash__ = str.__hash__
    # ^ this is why we derive from tuple - using the
    #   native tuple.__hash__ directly == speed boost
    #
    #   NOTE: does not depend on self.time!
    #


def _tuptup_sslbls(obj: Union[LabelTupTup, LabelTupTupWithArgs, LabelTupTupWithTime], offset: int) -> Optional[StateSpaceLabels]:
    """
    A helper function used to retrieve obj.sslbls; the return value is cached 
    in obj._sslbls if it's not None.
    """
    if len(obj) == 0:
        return None
    if getattr(obj, '_sslbls', None) is not None:
        # The getattr(...) guard is for backward compatibility in case
        # someone picked a Label. In principle, all types of `obj`
        # allowed in this function have a None-initialized _sslbls 
        # attribute set in their __new__ methods.
        return obj._sslbls  # type: ignore
    s = set()
    for lbl in obj[offset:]:
        if lbl.sslbls is None:
            return None
        s.update(lbl.sslbls)
    obj._sslbls = tuple(sorted(list(s))) # type: ignore
    return obj._sslbls # type: ignore


class LabelTupTup(Label, tuple):
    """
    A label consisting of a *tuple* of ConcreteLabel objects.

    This typically labels a layer of a circuit (a parallel level of gates).
    """

    IS_SIMPLE = False  # access with self.is_simple property

    @classmethod
    def init(cls, tup_of_tups: Sequence[Any]) -> LabelTupTup:
        """
        Creates a new Model-item tuple-of-tuples label.

        This is a tuple of tuples of simple string labels and
        tuples specifying the part of the Hilbert space upon
        which that item acts (often just qubit indices).

        Parameters
        ----------
        tup_of_tups : tuple
            The item data - a tuple of (string, state-space-labels) tuples
            which labels a parallel layer/level of a circuit.

        Returns
        -------
        LabelTupTup
        """
        if isinstance(tup_of_tups, Label):
            tup_of_tups_to_iterate = tup_of_tups.components
        else:
            tup_of_tups_to_iterate = tup_of_tups
        tupOfLabels : tuple[ConcreteLabel, ...] = tuple([tup if isinstance(tup, Label) else Label(tup) for tup in tup_of_tups_to_iterate]) 
        # ^ Note: constituent `tup`s in the list comprehension can also be a Label obj
        if __debug__ and tupOfLabels: # Debug flag is so that the check gets optimized out later in production runs.
            atleast_one_missing_time = False
            somewith_nonemptyargs = False
            ctimes = set()
            for lbl in tupOfLabels:
                if hasattr(lbl, "time"):
                    ctimes.add(lbl.time)
                else:
                    atleast_one_missing_time = True
                if hasattr(lbl, "args"):
                    if len(lbl.args) > 0:
                        somewith_nonemptyargs = True
            if not atleast_one_missing_time and ctimes == set([0.0]) and somewith_nonemptyargs:
                # This is likely a LabelTupTup containing LabelsWithArgs. We do not know if they just got defaulted to time = 0.
                warn("The interior labels have args which sets the default time to 0.0." \
                "If you want to also label this collection LabelTupTup with time = 0.0 you must include the time in the constructor.", RuntimeWarning) 
            elif not atleast_one_missing_time and len(ctimes) == 1 and 0.0 not in ctimes:
                warn("You may want to consider a LabelTupTupWithTime as all of the entries have the same time" \
                    " and it had to be user specified.", RuntimeWarning)
            elif not atleast_one_missing_time and len(ctimes) == 1:
                warn("You may want to consider a LabelTupTupWithTime as all of the entries have the same time " \
                    "which was previously the default time.", RuntimeWarning)

        ret = cls.__new__(cls, tupOfLabels)
        return ret
    
    def __new__(cls, tup_of_labels: tuple[ConcreteLabel, ...]) -> LabelTupTup:
        ret = tuple.__new__(cls, tup_of_labels)
        ret._sslbls : Optional[tuple] = None  # type: ignore
        ret._is_sorted = False
        return ret

    @property
    def name(self) -> Literal['COMPOUND']:
        """
        This label's name (a string).
        """
        # TODO - something intelligent here?
        # no real "name" for a compound label... but want it to be a string so
        # users can use .startswith, etc.
        return "COMPOUND"

    @property
    def sslbls(self) -> Optional[StateSpaceLabels]:
        """
        This label's state-space labels, often qubit labels (a tuple).

        If any component has sslbls == None, which signifies operating on
        *all* qubits, then this label is on *all* qubits and sslbls is None.
        """
        return _tuptup_sslbls(self, 0)

    @property
    def args(self) -> tuple:  # empty tuple
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self) -> LabelTupTup:  # always `self`
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self  # self is a tuple of "sub-label" components

    def has_prefix(self, prefix: str, typ: Literal['any', 'all']='all') -> bool:
        """
        Whether this label has the given `prefix`.

        Usually used to test whether the label names a given type.

        Parameters
        ----------
        prefix : str
            The prefix to check for.

        typ : {"any","all"}
            Whether, when there are multiple parts to the label, the prefix
            must occur in any or all of the parts.

        Returns
        -------
        bool
        """
        if typ == "all":
            return all([lbl.has_prefix(prefix) for lbl in self])
        elif typ == "any":
            return any([lbl.has_prefix(prefix) for lbl in self])
        else:
            raise ValueError("Invalid `typ` arg: %s" % str(typ))

    def map_state_space_labels(self, mapper) -> LabelTupTup:
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        Label
        """
        return LabelTupTup.__new__(LabelTupTup, tuple((lbl.map_state_space_labels(mapper) for lbl in self)))

    def strip_args(self) -> LabelTupTup:
        """ Return version of self with all arguments removed """
        # default, appropriate for a label without args or components
        return LabelTupTup.__new__(LabelTupTup, tuple(comp.strip_args() for comp in self))

    def __str__(self) -> str:
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        return "[" + "".join([str(lbl) for lbl in self]) + "]"

    def __repr__(self) -> str:
        return "Label(" + repr(self[:]) + ")"

    def __add__(self, s: None) -> None:
        raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other) -> bool:
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        return tuple.__eq__(self, other)

    def __lt__(self, x) -> bool:
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x) -> bool:
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self) -> tuple[type, tuple[tuple], None]:
        return self.__reduce__()

    def __reduce__(self) -> tuple[type, tuple[tuple[ConcreteLabel, ...]], None]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupTup, (self[:],), None)

    def __contains__(self, x) -> bool:
        # "recursive" contains checks component containers
        return any([(x == layer or x in layer) for layer in self.components])

    def to_native(self) -> tuple:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return (None, None, tuple(x.to_native() for x in self),)

    def replace_name(self, oldname: str, newname: str) -> LabelTupTup:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelTupTup
        """
        return LabelTupTup(tuple((x.replace_name(oldname, newname) for x in self)))

    @property
    def depth(self) -> int:
        """
        The depth of this label, viewed as a sub-circuit.
        """
        if len(self.components) == 0: return 1  # still depth 1 even if empty
        return max([x.depth for x in self.components])

    def expand_subcircuits(self) -> tuple[LabelTupTup, ...]:
        """
        Expand any sub-circuits within this label.

        Returns a list of component labels which doesn't include any
        :class:`CircuitLabel` labels.  This effectively expands any "boxes" or
        "exponentiation" within this label.

        Returns
        -------
        tuple
            A tuple of component Labels (none of which should be
            :class:`CircuitLabel` objects).
        """
        ret = []
        expanded_comps = [x.expand_subcircuits() for x in self.components]

        for i in range(self.depth):  # depth == # of layers when expanded
            ec = []
            for expanded_comp in expanded_comps:
                if i < len(expanded_comp):
                    ec.extend(expanded_comp[i].components)  # .components = vertical expansion
            #assert(len(ec) > 0), "Logic error!" #this is ok (e.g. an idle subcircuit)
            ret.append(LabelTupTup.init(ec))
        return tuple(ret)

    def concate(self, other: Label):
        other, new_time = _check_concate_compatibility(self, other)

        if isinstance(other, LabelTupTup):
            components = (*self, *other)
        elif isinstance(other, LabelTup):
            components = (*self, other)
        elif isinstance(other, LabelStr):
            raise ValueError("Cannot concate `LabelStr` as they do not have an associated qubit set unless it is explicitly in the string.")
        else:
            return super().concate(other)

        all_args = self.collect_args() + other.collect_args()
        return _create_concatenated_label(components, new_time, all_args)

    __hash__ = tuple.__hash__
    # ^ this is why we derive from tuple - using the
    #   native tuple.__hash__ directly == speed boost


class LabelTupTupWithTime(LabelTupTup, tuple):
    """
    A label consisting of a *tuple* of ConcreteLabel objects.

    This typically labels a layer of a circuit (a parallel level of gates).
    """

    @classmethod
    def init(cls, tup_of_tups: Sequence[Any], time: Optional[float]=None) -> LabelTupTupWithTime:
        """
        Creates a new Model-item tuple-of-tuples label.

        This is a tuple of tuples of simple string labels and
        tuples specifying the part of the Hilbert space upon
        which that item acts (often just qubit indices).

        Parameters
        ----------
        tup_of_tups : tuple
            The item data - a tuple of (string, state-space-labels) tuples
            which labels a parallel layer/level of a circuit.

        time : float, optional
            A time value associated with this label.  Often this is the
            duration of the object or operation labeled.

        Returns
        -------
        LabelTupTupWithTime
        """
        if time is None and hasattr(tup_of_tups, 'time'):
            time = tup_of_tups.time

        assert(time is None or isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        if isinstance(tup_of_tups, Label):
            tup_of_tups_to_iterate = tup_of_tups.components
        else:
            tup_of_tups_to_iterate = tup_of_tups
        tupOfLabels = tuple(tup if isinstance(tup, Label) else Label(tup) for tup in tup_of_tups_to_iterate)  # Note: tup can also be a Label obj
        if time is None:
            time = 0.0 if len(tupOfLabels) == 0 else \
                max([lbl.time for lbl in tupOfLabels if hasattr(lbl, "time")] + [0.0])
        return cls.__new__(cls, tupOfLabels, time)

    def __new__(cls, tup_of_labels: tuple[ConcreteLabel, ...], time: float=0.0):
        ret = super(LabelTupTupWithTime, cls).__new__(cls, tup_of_labels)
        ret.time : float = time  # type: ignore
        return ret

    @property
    def time(self) -> float:
        """
        The time value associated with this label.
        """
        return self._time
    
    @time.setter
    def time(self, val: float):
        self._time = val

    def map_state_space_labels(self, mapper) -> LabelTupTupWithTime:
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        Label
        """
        mapped = tuple((lbl.map_state_space_labels(mapper) for lbl in self))
        time : float = self.time  # type: ignore
        return LabelTupTupWithTime.__new__(LabelTupTupWithTime, mapped, time)

    def strip_args(self) -> LabelTupTupWithTime:
        """ Return version of self with all arguments removed """
        # default, appropriate for a label without args or components
        stripped = tuple((comp.strip_args() for comp in self))
        time : float = self.time  # type: ignore
        return LabelTupTupWithTime.__new__(LabelTupTupWithTime, stripped, time)

    def __repr__(self) -> str:
        time : float = self.time  # type: ignore
        timearg = ",time=" + repr(time) if (time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __reduce__(self) -> tuple[type, tuple[tuple, float], None]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        time : float = self.time  # type: ignore
        return (LabelTupTupWithTime, (self[:], time), None)

    def to_native(self) -> tuple:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return (self.time, None, tuple(c.to_native() for c in self),)

    def replace_name(self, oldname: str, newname: str) -> LabelTupTupWithTime:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelTupTupWithTime
        """
        renamed = tuple((x.replace_name(oldname, newname) for x in self))
        time : float = self.time # type: ignore
        return LabelTupTupWithTime.__new__(LabelTupTupWithTime, renamed, time)

    def expand_subcircuits(self) -> tuple[LabelTupTupWithTime, ...]:
        """
        Expand any sub-circuits within this label.

        Returns a list of component labels which doesn't include any
        :class:`CircuitLabel` labels.  This effectively expands any "boxes" or
        "exponentiation" within this label.

        Returns
        -------
        tuple
            A tuple of component Labels (none of which should be
            :class:`CircuitLabel` objects).
        """
        ret = []
        expanded_comps = [x.expand_subcircuits() for x in self.components]

        for i in range(self.depth):  # depth == # of layers when expanded
            ec = []
            for expanded_comp in expanded_comps:
                if i < len(expanded_comp):
                    ec.extend(expanded_comp[i].components)  # .components = vertical expansion
            #assert(len(ec) > 0), "Logic error!" #this is ok (e.g. an idle subcircuit)
            ret.append(LabelTupTupWithTime.init(ec))
        return tuple(ret)

class CircuitLabel(LabelTupTupWithTime, tuple):
    """
    A (sub-)circuit label.

    This class encapsulates a complete circuit as a single layer.  It
    lacks some of the methods and metadata of a true :class:`Circuit`
    object, but contains the essentials: the tuple of layer labels
    (held as the label's components) and line labels (held as the label's
    state-space labels)
    """

    IS_SIMPLE = True  # access with self.is_simple property

    # NOTE: This class doesn't follow the pattern with an ".init(...)"
    # method that calls a ".__new__(...)" method after mild input parsing.

    def __new__(cls, name: str, tup_of_layers: Sequence[Any],
                state_space_labels: Optional[StateSpaceLabels],
                reps: int=1, time: Optional[float]=None
        ) -> CircuitLabel:
        """
        Creates a new Model-item label, which defines a set of other labels
        as a sub-circuit and allows that sub-circuit to be repeated some integer
        number of times.  A `CircuitLabel` can be visualized as placing a
        (named) box around some set of labels and optionally exponentiating
        that box.

        Internally, a circuit labels look very similar to `LabelTupTup` objects,
        holding a tuple of tuples defining the component labels (circuit layers).

        Parameters
        ----------
        name : str
            The name of the sub-circuit (box).  Cannot be `None`, but can be
            empty.

        tup_of_layers : tuple
            The item data - a tuple of tuples which label the components
            (layers) within this label.

        state_space_labels : list or tuple
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.

        reps : int, optional
            The "exponent" - the number of times the `tup_of_layers` labels are
            repeated.

        time : float
            The time at which this label occurs (can be relative or absolute)
        """
        assert(isinstance(reps, _numbers.Integral) and isinstance(name, str)
               ), "Invalid name or reps: %s %s" % (str(name), str(reps))
        tupOfLabels : tuple[ConcreteLabel, ...] = tuple((Label(tup) for tup in tup_of_layers))

        if state_space_labels is not None:
            # The PR that added type annotations to this class also added type this checking.
            # The type checking is necessary to comply with the Label API.
            assert isinstance(state_space_labels, tuple)
            for sslbl in state_space_labels:
                assert isinstance(sslbl, (int, str, _numbers.Integral)) 
        
        ret = tuple.__new__(cls, (name, state_space_labels, reps) + tupOfLabels)
        if time is None:
            ret.time = 0.0 if len(tupOfLabels) == 0 else \
                sum([lbl.time for lbl in tupOfLabels if hasattr(lbl, "time")] + [0.0])  # sum b/c components are *layers* of sub-circuit
        else:
            ret.time = time
        ret._is_sorted = False
        return ret

    @property
    def name(self) -> str:
        """
        This label's name (a string).
        """
        return self[0]

    @property
    def sslbls(self) -> Optional[StateSpaceLabels]:
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        return self[1]

    @property
    def reps(self) -> int:
        """
        Number of repetitions (of this label's components) that this label represents.
        """
        return self[2]

    def has_prefix(self, prefix, typ="all") -> bool:
        """
        Whether this label has the given `prefix`.

        Usually used to test whether the label names a given type.

        Parameters
        ----------
        prefix : str
            The prefix to check for.

        typ : {"any","all"}
            Whether, when there are multiple parts to the label, the prefix
            must occur in any or all of the parts.

        Returns
        -------
        bool
        """
        return self.name.startswith(prefix) # Needed since LabelTupTup overrides.

    @property
    def components(self) -> tuple[ConcreteLabel, ...]:
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self[3:]

    def map_state_space_labels(self, mapper) -> CircuitLabel:
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        CircuitLabel
        """
        if isinstance(mapper, dict):
            mapped_sslbls = tuple((mapper[sslbl] for sslbl in self.sslbls))
        else:  # assume mapper is callable
            mapped_sslbls = tuple((mapper(sslbl) for sslbl in self.sslbls))
        mapped_layers = [lbl.map_state_space_labels(mapper) for lbl in self.components]
        time : float = self.time  # type: ignore
        return CircuitLabel.__new__(CircuitLabel, self.name, mapped_layers, mapped_sslbls, self.reps, time)

    def strip_args(self) -> None:
        raise NotImplementedError("TODO!")

    def __str__(self) -> str:
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        if len(self.name) > 0:
            s = self.name
            if self.time != 0.0:
                s += ("!%f" % self.time).rstrip('0').rstrip('.')
        else:
            s = "".join([str(lbl) for lbl in self.components])
            if self.time != 0.0:
                s += ("!%f" % self.time).rstrip('0').rstrip('.')
            if len(self.components) > 0:
                s = "(" + s + ")"  # add parenthesis
        if self[2] != 1: s += "^%d" % self[2]
        return s

    def __repr__(self) -> str:
        return "CircuitLabel(" + repr(self.name) + "," + repr(self[3:]) + "," \
            + repr(self[1]) + "," + repr(self[2]) + "," + repr(self.time) + ")"

    def __reduce__(self) -> tuple[
            type, 
            tuple[str, tuple[ConcreteLabel, ...], Optional[StateSpaceLabels], int, float],
            None
        ]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (CircuitLabel, (self[0], self[3:], self[1], self[2], self.time), None)

    def to_native(self) -> tuple:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        tup = self[0:3]
        if self.time is not None and self.time != 0.0:
            tup += ('!' + str(self.time),)
        tup += tuple((x.to_native() for x in self.components))
        return tup

    def replace_name(self, oldname, newname):
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        CircuitLabel
        """
        mapped_layers = tuple((x.replace_name(oldname, newname) for x in self.components))
        return CircuitLabel.__new__(CircuitLabel, self.name, mapped_layers, self.sslbls, self[2], self.time)

    @property
    def depth(self) -> int:
        """
        The depth of this label, viewed as a sub-circuit.
        """
        return sum([x.depth for x in self.components]) * self.reps

    def expand_subcircuits(self) -> tuple:
        """
        Expand any sub-circuits within this label.

        Returns a list of component labels which doesn't include any
        :class:`CircuitLabel` labels.  This effectively expands any "boxes" or
        "exponentiation" within this label.

        Returns
        -------
        tuple
            A tuple of component Labels (none of which should be
            :class:`CircuitLabel` objects).
        """
        return tuple(_itertools.chain(*[x.expand_subcircuits() for x in self.components])) * self.reps

    def concate(self, other: Label):

        if isinstance(other, CircuitLabel):
            if other.depth == self.depth:
                # We can concate these.
                return Label((self, other)) # make a _LabelTupTup
            raise ValueError(f"Cannot concate two CircuitLabels of different depth: {self.depth} vs {other.depth}")

        tmp_lbl: Label = None
        if self.depth == 1:
            # Convert into not a CircuitLabel.
            tmp_lbl = Label(self.components, self.sslbls, self.time, self.args)
        else:
            raise ValueError(f"One cannot concate a `CircuitLabel` of depth > 1 (depth = {self.depth}) except to another `CircuitLabel` of the same depth.")
        return tmp_lbl.concate(other)


class LabelTupWithArgs(LabelTupWithTime, tuple):
    """
    A label consisting of a string along with a tuple of integers or state-space-names.

    These state-space sector names specify which qubits, or more generally,
    parts of the Hilbert space that is acted upon by the object this label
    refers to.  This label type also supports having arguments and a time value.
    """

    @classmethod
    def init(cls, name: str, state_space_labels: StateSpaceLabelsCastable, time: float = 0.0, args: Sequence[Any] = ()) -> LabelTupWithArgs:
        """
        Creates a new Model-item label.

        The created is divided into a simple string label, a tuple specifying
        the part of the Hilbert space upon which the item acts (often just qubit
        indices), a time, and arguments.

        Parameters
        ----------
        name : str
            The item name. E.g., 'CNOT' or 'H'.

        state_space_labels : list or tuple
            A list or tuple that identifies which sectors/parts of the Hilbert
            space is acted upon.  In many cases, this is a list of integers
            specifying the qubits on which a gate acts, when the ordering in the
            list defines the 'direction' of the gate.  If something other than
            a list or tuple is passed, a single-element tuple is created
            containing the passed object.

        time : float
            The time at which this label occurs (can be relative or absolute)

        args : iterable of hashable types
            A list of "arguments" for this label.

        Returns
        -------
        LabelTupWithArgs
        """
        #Type checking
        assert(isinstance(name, str)), "`name` must be a string, but it's '%s'" % str(name)
        assert(state_space_labels is not None), "LabelTup must be initialized with non-None state-space labels"
        if not isinstance(state_space_labels, (tuple, list)):
            state_space_labels = (state_space_labels,)
        for ssl in state_space_labels:
            assert(isinstance(ssl, str) or isinstance(ssl, _numbers.Integral)), \
                "State space label '%s' must be a string or integer!" % str(ssl)
        assert(isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        assert(len(args) > 0), "`args` must be a nonempty list/tuple of hashable arguments"
        #TODO: check that all args are hashable?

        #Try to convert integer-strings to ints (for parsing from files...)
        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = _integerize_sslbls(state_space_labels)
        args = tuple(args)
        tup = (_sys.intern(name), 2 + len(args)) + args + sslbls  # stores: (name, K, args, sslbls)
        # where K is the index of the start of the sslbls (or 1 more than the last arg index)

        return cls.__new__(cls, tup, time)

    @property
    def sslbls(self) -> Optional[StateSpaceLabels]:
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) > self[1]:
            return self[self[1]:]
        else:
            return None

    @property
    def args(self) -> tuple[Any, ...]:
        """
        This label's arguments.
        """
        return self[2:self[1]]

    def map_state_space_labels(self, mapper) -> LabelTupWithArgs:
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        Label
        """
        if isinstance(mapper, dict):
            mapped_sslbls = tuple((mapper[sslbl] for sslbl in self.sslbls))
        else:  # assume mapper is callable
            mapped_sslbls = tuple((mapper(sslbl) for sslbl in self.sslbls))
        time : float = self.time  # type: ignore
        return LabelTupWithArgs.init(self.name, mapped_sslbls, time, self.args)

    def strip_args(self) -> Union[LabelTup, LabelStr]:
        if self.sslbls is not None:
            stripped_tup = self[self[1]:]
            return LabelTup.init(self.name, stripped_tup)
        else:
            # special case of sslbls == None, which is just a string label without its args
            return LabelStr.__new__(LabelStr, self[0])

    def __str__(self) -> str:
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        s = str(self.name)
        if self.args:  # test for None and len == 0
            s += ";" + ";".join(map(str, self.args))
        if self.sslbls:  # test for None and len == 0
            s += ":" + ":".join(map(str, self.sslbls))
        if self.time != 0.0:
            s += ("!%f" % self.time).rstrip('0').rstrip('.')
        return s

    def __repr__(self) -> str:
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self.name) + "," + repr(self.sslbls) + ",args=" + repr(self.args) + timearg + ")"

    def __add__(self, s: str) -> LabelTupWithArgs:
        if isinstance(s, str):
            sslbls = tuple() if self.sslbls is None else self.sslbls
            return LabelTupWithArgs.init(self.name + s, sslbls, self.time, self.args)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __lt__(self, x) -> bool:
        # NOTE: does not depend on self.time!
        try:
            return tuple.__lt__(self, tuple(x))
        except:
            return tuple.__lt__(tuple(map(str, self)), tuple(map(str, x)))

    def __gt__(self, x) -> bool:
        # NOTE: does not depend on self.time!
        try:
            return tuple.__gt__(self, tuple(x))
        except:
            return tuple.__gt__(tuple(map(str, self)), tuple(map(str, x)))

    def __reduce__(self) -> tuple[type, tuple[tuple, float], None]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupWithArgs, (self[:], self.time), None)

    def to_native(self) -> tuple:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        ret = (self.name,)
        if self.sslbls is not None:
            ret += self.sslbls

        if self.args:
            ret += (';',) + self.args

        if self.time != 0.0:
            ret += ('!' + str(self.time),)
        return ret

    def replace_name(self, oldname: str, newname: str) -> LabelTupWithArgs:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelTupWithArgs
        """
        if self.name != oldname:
            return self
        sslbls = tuple() if self.sslbls is None else self.sslbls
        return LabelTupWithArgs.init(newname, sslbls, self.time, self.args)


class LabelTupTupWithArgs(LabelTupTupWithTime, tuple):
    """
    A label consisting of a *tuple* of (string, state-space-labels) tuples.

    This typically labels a layer of a circuit (a parallel level of gates).
    This label type also supports having arguments and a time value.

    Contents, as a tuple, are equal to `(K,) + tuple(args) + tup_of_lbls`,
    where K is the index of the start of tup_of_lbls.
    """

    @classmethod
    def init(cls, tup_of_tups: Sequence[Any], time: Optional[float]=None, args: Sequence[Any]=()):
        """
        Creates a new Model-item label.

        The created label is a tuple of tuples of simple string labels and
        tuples specifying the part of the Hilbert space upon which that item
        acts (often just qubit indices).

        Parameters
        ----------
        tup_of_tups : tuple
            The item data - a tuple of (string, state-space-labels) tuples
            which labels a parallel layer/level of a circuit.

        time : float
            The time at which this label occurs (can be relative or absolute)

        args : iterable of hashable types
            A list of "arguments" for this label.

        Returns
        -------
        LabelTupTupWithArgs
        """
        if time is None and hasattr(tup_of_tups, 'time'):
            time = tup_of_tups.time
        if not args and hasattr(tup_of_tups, 'args'):
            args = tup_of_tups.args

        assert(time is None or isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        if isinstance(tup_of_tups, Label):
            tup_of_tups_to_iterate = tup_of_tups.components
        else:
            tup_of_tups_to_iterate = tup_of_tups
        tup_of_lbls = tuple(tup if isinstance(tup, Label) else Label(tup) for tup in tup_of_tups_to_iterate)
        tup_rep = (1 + len(args),) + tuple(args) + tup_of_lbls
        if time is None:
            time = 0.0 if len(tup_of_lbls) == 0 else \
                max([lbl.time for lbl in tup_of_lbls if hasattr(lbl, "time")] + [0.0])
        return cls.__new__(cls, tup_rep, time)



    @property
    def sslbls(self) -> Optional[StateSpaceLabels]:
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        return _tuptup_sslbls(self, self[0])

    @property
    def args(self) -> tuple[Any, ...]:
        """
        This label's arguments.
        """
        return self[1:self[0]]

    @property
    def components(self) -> tuple[ConcreteLabel, ...]:
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self[self[0]:]  # a tuple of "sub-label" components


    def map_state_space_labels(self, mapper):
        """
        Apply a mapping to this Label's state-space (qubit) labels.

        Return a copy of this Label with all of the state-space labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the Label "Gcnot:0:1" would return "Gcnot:1:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing state-space-label) argument and returns a new state-space-label.

        Returns
        -------
        Label
        """
        mapped_inner_tups = tuple((lbl.map_state_space_labels(mapper) for lbl in self.components))
        return LabelTupTupWithArgs.init(mapped_inner_tups, self.time, self.args)

    def strip_args(self) -> LabelTupTupWithTime:
        """ Return version of self with all arguments removed """
        # default, appropriate for a label without args or components
        stripped_inner_tups = tuple((comp.strip_args() for comp in self))
        return LabelTupTupWithTime.__new__(LabelTupTupWithTime, stripped_inner_tups, self.time)

    def __str__(self) -> str:
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        if self.args:  # test for None and len == 0
            argstr = ";" + ";".join(map(str, self.args))
        else:
            argstr = ""

        if self.time != 0.0:  # if we're supposed to be holding a time
            timestr = ("!%f" % self.time).rstrip('0').rstrip('.')
        else:
            timestr = ""

        return "[" + "".join([str(lbl) for lbl in self]) + argstr + timestr + "]"

    def __repr__(self) -> str:
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __reduce__(self) -> tuple[type, tuple[tuple[Any, ...], float], None]:
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupTupWithArgs, (self[:], self.time), None)

    def to_native(self) -> tuple[Any, ...]:
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return (self.time, self.args, tuple(c.to_native() for c in self.components),)

    def replace_name(self, oldname: str, newname: str) -> LabelTupTupWithArgs:
        """
        Returns a label with `oldname` replaced by `newname`.

        Parameters
        ----------
        oldname : str
            Name to find.

        newname : str
            Name to replace found name with.

        Returns
        -------
        LabelTupTupWithArgs
        """
        replaced_components = tuple((x.replace_name(oldname, newname) for x in self.components))
        return LabelTupTupWithArgs.init(replaced_components, self.time, self.args)

    def expand_subcircuits(self) -> tuple['LabelTupTupWithArgs', ...]:
        """
        Expand any sub-circuits within this label.

        Returns a list of component labels which doesn't include any
        :class:`CircuitLabel` labels.  This effectively expands any "boxes" or
        "exponentiation" within this label.

        Returns
        -------
        tuple
            A tuple of component Labels (none of which should be
            :class:`CircuitLabel` objects).
        """
        ret = []
        expanded_comps = [x.expand_subcircuits() for x in self.components]

        for i in range(self.depth):  # depth == # of layers when expanded
            ec = []
            for expanded_comp in expanded_comps:
                if i < len(expanded_comp):
                    ec.extend(expanded_comp[i].components)
            ret.append(LabelTupTupWithArgs.init(ec, args=self.args))
        return tuple(ret)



ConcreteLabel = Union[
    LabelTup,    LabelTupWithArgs,    LabelTupWithTime,
    LabelTupTup, LabelTupTupWithArgs, LabelTupTupWithTime,
    LabelStr, CircuitLabel
]
