## user-exposure: low (EGN - using these objects directly is uncommon, though their ability to act like and be created from strings & tuples is utilized by even novice users perhaps unknowingly)
"""
Defines the Label class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import numbers as _numbers
import sys as _sys

_debug_record = {}


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

    def __new__(cls, name, state_space_labels=None, time=None, args=None):
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
        #print("Label.__new__ with name=", name, "sslbls=", state_space_labels, "t=", time, "args=", args)
        if isinstance(name, Label) and state_space_labels is None:
            return name  # Note: Labels are immutable, so no need to copy

        if not isinstance(name, str) and state_space_labels is None \
           and isinstance(name, (tuple, list)):

            #We're being asked to initialize from a non-string with no
            # state_space_labels, explicitly given.  `name` could either be:
            # 0) an empty tuple: () -> LabelTupTup with *no* subLabels.
            # 1) a (name, ssl0, ssl1, ...) tuple -> LabelTup
            # 2) a (subLabel1_tup, subLabel2_tup, ...) tuple -> LabelTupTup if
            #     length > 1 otherwise just initialize from subLabel1_tup.
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
                    if args: return LabelTupTupWithArgs.init(name, time, args)
                    elif time is None or time == 0: return LabelTupTup.init(name)
                    else: return LabelTupTupWithTime.init(name, time)
                else:
                    return Label(name[0], time=time, args=args)
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
                        time = x; continue

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
                            if x == '!':
                                next_is_time = True
                            else:
                                time = float(x[1:])
                            continue
                    state_space_labels.append(x)
                args = tup_args if len(tup_args) > 0 else None
                state_space_labels = tuple(state_space_labels)  # needed for () and (None,) comparison below

        if time is None:
            time = 0.0  # for non-TupTup labels not setting a time is equivalent to setting it to 0.0

        #print(" -> preproc with name=", name, "sslbls=", state_space_labels, "t=", time, "args=", args)
        if state_space_labels is None or state_space_labels in ((), (None,)):
            if args:
                return LabelTupWithArgs.init(name, (), time, args)  # just use empty sslbls
            else:
                return LabelStr.init(name, time)

        else:
            if args: return LabelTupWithArgs.init(name, state_space_labels, time, args)
            else:
                if time == 0.0:
                    return LabelTup.init(name, state_space_labels)
                else:
                    return LabelTupWithTime.init(name, state_space_labels, time)

    @property
    def depth(self):
        """
        The depth of this label, viewed as a sub-circuit.
        """
        return 1  # most labels are depth=1

    @property
    def reps(self):
        """
        Number of repetitions (of this label's components) that this label represents.
        """
        return 1  # most labels have only reps==1

    @property
    def has_nontrivial_components(self):
        return len(self.components) > 0 and self.components != (self,)

    def collect_args(self):
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


class LabelTup(Label, tuple):
    """
    A label consisting of a string along with a tuple of integers or state-space-names.

    These state-space sector names specify which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by the object this label refers to.
    """

    @classmethod
    def init(cls, name, state_space_labels):
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
        integerized_sslbls = []
        for ssl in state_space_labels:
            try: integerized_sslbls.append(int(ssl))
            except: integerized_sslbls.append(_sys.intern(ssl))

        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = tuple(integerized_sslbls)
        tup = (_sys.intern(name),) + sslbls
        return tuple.__new__(cls, tup)

    __new__ = tuple.__new__
    #def __new__(cls, tup, time=0.0):
    #    ret = tuple.__new__(cls, tup)  # creates a LabelTup object using tuple's __new__
    #    ret.time = time
    #    return ret

    @property
    def time(self):
        """
        This label's name time (always 0)
        """
        return 0

    @property
    def name(self):
        """
        This label's name (a string).
        """
        return self[0]

    @property
    def sslbls(self):
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) > 1:
            return self[1:]
        else: return None

    @property
    def args(self):
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return (self,)  # just a single "sub-label" component

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices. (a tuple)
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
        if isinstance(mapper, dict):
            mapped_sslbls = [mapper[sslbl] for sslbl in self.sslbls]
        else:  # assume mapper is callable
            mapped_sslbls = [mapper(sslbl) for sslbl in self.sslbls]
        return Label(self.name, mapped_sslbls)

    #OLD
    #def __iter__(self):
    #    return self.tup.__iter__()

    #OLD
    #def __iter__(self):
    #    """ Iterate over the name + state space labels """
    #    # Note: tuple(.) uses __iter__ to construct tuple rep.
    #    yield self.name
    #    if self.sslbls is not None:
    #        for ssl in self.sslbls:
    #            yield ssl

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        #caller = inspect.getframeinfo(inspect.currentframe().f_back)
        #ky = "%s:%s:%d" % (caller[2],os.path.basename(caller[0]),caller[1])
        #_debug_record[ky] = _debug_record.get(ky, 0) + 1
        s = str(self.name)
        if self.sslbls:  # test for None and len == 0
            s += ":" + ":".join(map(str, self.sslbls))
        return s

    def __repr__(self):
        return "Label(" + repr(self[:]) + ")"

    def __add__(self, s):
        if isinstance(s, str):
            return LabelTup.init(self.name + s, self.sslbls)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x):
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTup, (self[:],), None)

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return tuple(self)

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
        LabelTup
        """
        return LabelTup(newname, self.sslbls) if (self.name == oldname) else self

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return True

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


class LabelTupWithTime(Label, tuple):
    """
    A label consisting of a string along with a tuple of integers or state-space-names.

    These state-space sector names specify which qubits, or
    more generally, parts of the Hilbert space that is
    acted upon by the object this label refers to.
    """

    @classmethod
    def init(cls, name, state_space_labels, time=0.0):
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
        integerized_sslbls = []
        for ssl in state_space_labels:
            try: integerized_sslbls.append(int(ssl))
            except: integerized_sslbls.append(_sys.intern(ssl))

        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = tuple(integerized_sslbls)
        tup = (_sys.intern(name),) + sslbls
        return cls.__new__(cls, tup, time)

    def __new__(cls, tup, time=0.0):
        ret = tuple.__new__(cls, tup)  # creates a LabelTupWithTime object using tuple's __new__
        ret.time = time
        return ret

    @property
    def name(self):
        """
        This label's name (a string).
        """
        return self[0]

    @property
    def sslbls(self):
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) > 1:
            return self[1:]
        else: return None

    @property
    def args(self):
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return (self,)  # just a single "sub-label" component

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices. (a tuple)
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
        if isinstance(mapper, dict):
            mapped_sslbls = [mapper[sslbl] for sslbl in self.sslbls]
        else:  # assume mapper is callable
            mapped_sslbls = [mapper(sslbl) for sslbl in self.sslbls]
        return Label(self.name, mapped_sslbls)

    #OLD
    #def __iter__(self):
    #    return self.tup.__iter__()

    #OLD
    #def __iter__(self):
    #    """ Iterate over the name + state space labels """
    #    # Note: tuple(.) uses __iter__ to construct tuple rep.
    #    yield self.name
    #    if self.sslbls is not None:
    #        for ssl in self.sslbls:
    #            yield ssl

    def __str__(self):
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

    def __repr__(self):
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __add__(self, s):
        if isinstance(s, str):
            return LabelTupWithTime.init(self.name + s, self.sslbls)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x):
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupWithTime, (self[:], self.time), None)

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return tuple(self)

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
        LabelTupWithTime
        """
        return LabelTupWithTime(newname, self.sslbls) if (self.name == oldname) else self

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return True

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


class LabelStr(Label, str):
    """
    A string-valued label.

    A Label for the special case when only a name is present (no
    state-space-labels).  We create this as a separate class
    so that we can use the string hash function in a
    "hardcoded" way - if we put switching logic in __hash__
    the hashing gets *much* slower.
    """

    @classmethod
    def init(cls, name, time=0.0):
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

    def __new__(cls, name, time=0.0):
        ret = str.__new__(cls, name)
        ret.time = time
        return ret

    @property
    def name(self):
        """
        This label's name (a string).
        """
        return str(self[:])

    @property
    def sslbls(self):
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        return None

    @property
    def args(self):
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return (self,)  # just a single "sub-label" component

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices.
        """
        return None

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return None

    def has_prefix(self, prefix, typ="all"):
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
        return self.startswith(prefix)

    def __str__(self):
        s = self[:]  # converts to a normal str
        if self.time != 0.0:
            s += ("!%f" % self.time).rstrip('0').rstrip('.')
        return s

    def __repr__(self):
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __add__(self, s):
        if isinstance(s, str):
            return LabelStr(self.name + str(s))
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        return str.__eq__(self, other)

    def __lt__(self, x):
        return str.__lt__(self, str(x))

    def __gt__(self, x):
        return str.__gt__(self, str(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelStr, (str(self), self.time), None)

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        str
        """
        return str(self)

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
        LabelStr
        """
        return LabelStr(newname) if (self.name == oldname) else self

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return True

    __hash__ = str.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


class LabelTupTup(Label, tuple):
    """
    A label consisting of a *tuple* of (string, state-space-labels) tuples.

    This typically labels a layer of a circuit (a parallel level of gates).
    """

    @classmethod
    def init(cls, tup_of_tups):
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
        tupOfLabels = tuple((Label(tup) for tup in tup_of_tups))  # Note: tup can also be a Label obj
        if len(tupOfLabels) > 0:
            assert(max([lbl.time for lbl in tupOfLabels]) == 0.0), \
                "Cannot create a LabelTupTup containing labels with time != 0"
        return cls.__new__(cls, tupOfLabels)

    __new__ = tuple.__new__

    @property
    def time(self):
        """
        This label's name time (always 0)
        """
        return 0.0

    @property
    def name(self):
        """
        This label's name (a string).
        """
        # TODO - something intelligent here?
        # no real "name" for a compound label... but want it to be a string so
        # users can use .startswith, etc.
        return "COMPOUND"

    @property
    def sslbls(self):
        # Note: if any component has sslbls == None, which signifies operating
        # on *all* qubits, then this label is on *all* qubites
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) == 0: return None  # "idle" label containing no gates - *all* qubits idle
        s = set()
        for lbl in self:
            if lbl.sslbls is None: return None
            s.update(lbl.sslbls)
        return tuple(sorted(list(s)))

    @property
    def args(self):
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self  # self is a tuple of "sub-label" components

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices.
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
        else: raise ValueError("Invalid `typ` arg: %s" % str(typ))

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
        return LabelTupTup(tuple((lbl.map_state_space_labels(mapper) for lbl in self)))

    def strip_args(self):
        """ Return version of self with all arguments removed """
        # default, appropriate for a label without args or components
        return LabelTupTup.__new__(LabelTupTup, (comp.strip_args() for comp in self))

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        return "[" + "".join([str(lbl) for lbl in self]) + "]"

    def __repr__(self):
        return "Label(" + repr(self[:]) + ")"

    def __add__(self, s):
        raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x):
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupTup, (self[:],), None)

    def __contains__(self, x):
        # "recursive" contains checks component containers
        return any([(x == layer or x in layer) for layer in self.components])

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return tuple((x.to_native() for x in self))

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
        LabelTupTup
        """
        return LabelTupTup(tuple((x.replace_name(oldname, newname) for x in self)))

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return False

    @property
    def depth(self):
        """
        The depth of this label, viewed as a sub-circuit.
        """
        if len(self.components) == 0: return 1  # still depth 1 even if empty
        return max([x.depth for x in self.components])

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

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


class LabelTupTupWithTime(Label, tuple):
    """
    A label consisting of a *tuple* of (string, state-space-labels) tuples.

    This typically labels a layer of a circuit (a parallel level of gates).
    """

    @classmethod
    def init(cls, tup_of_tups, time=None):
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
        assert(time is None or isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        tupOfLabels = tuple((Label(tup) for tup in tup_of_tups))  # Note: tup can also be a Label obj
        if time is None:
            time = 0.0 if len(tupOfLabels) == 0 else \
                max([lbl.time for lbl in tupOfLabels])
        return cls.__new__(cls, tupOfLabels, time)

    def __new__(cls, tup_of_labels, time=0.0):
        ret = tuple.__new__(cls, tup_of_labels)  # creates a LabelTupTupWithTime object using tuple's __new__
        ret.time = time
        return ret

    @property
    def name(self):
        """
        This label's name (a string).
        """
        # TODO - something intelligent here?
        # no real "name" for a compound label... but want it to be a string so
        # users can use .startswith, etc.
        return "COMPOUND"

    @property
    def sslbls(self):
        # Note: if any component has sslbls == None, which signifies operating
        # on *all* qubits, then this label is on *all* qubites
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) == 0: return None  # "idle" label containing no gates - *all* qubits idle
        s = set()
        for lbl in self:
            if lbl.sslbls is None: return None
            s.update(lbl.sslbls)
        return tuple(sorted(list(s)))

    @property
    def args(self):
        """
        This label's arguments.
        """
        return ()

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self  # self is a tuple of "sub-label" components

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices.
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
        else: raise ValueError("Invalid `typ` arg: %s" % str(typ))

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
        return LabelTupTupWithTime(tuple((lbl.map_state_space_labels(mapper) for lbl in self)))

    def strip_args(self):
        """ Return version of self with all arguments removed """
        # default, appropriate for a label without args or components
        return LabelTupTupWithTime.__new__(LabelTupTupWithTime, (comp.strip_args() for comp in self), self.time)

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        return "[" + "".join([str(lbl) for lbl in self]) + "]"

    def __repr__(self):
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __add__(self, s):
        raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x):
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupTupWithTime, (self[:], self.time), None)

    def __contains__(self, x):
        # "recursive" contains checks component containers
        return any([(x == layer or x in layer) for layer in self.components])

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return tuple((x.to_native() for x in self))

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
        LabelTupTupWithTime
        """
        return LabelTupTupWithTime(tuple((x.replace_name(oldname, newname) for x in self)))

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return False

    @property
    def depth(self):
        """
        The depth of this label, viewed as a sub-circuit.
        """
        if len(self.components) == 0: return 1  # still depth 1 even if empty
        return max([x.depth for x in self.components])

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

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


class CircuitLabel(Label, tuple):
    """
    A (sub-)circuit label.

    This class encapsulates a complete circuit as a single layer.  It
    lacks some of the methods and metadata of a true :class:`Circuit`
    object, but contains the essentials: the tuple of layer labels
    (held as the label's components) and line labels (held as the label's
    state-space labels)
    """
    def __new__(cls, name, tup_of_layers, state_space_labels, reps=1, time=None):
        # Note: may need default args for all but 1st for pickling!
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
        tupOfLabels = tuple((Label(tup) for tup in tup_of_layers))  # Note: tup can also be a Label obj
        # creates a CircuitLabel object using tuple's __new__
        ret = tuple.__new__(cls, (name, state_space_labels, reps) + tupOfLabels)
        if time is None:
            ret.time = 0.0 if len(tupOfLabels) == 0 else \
                sum([lbl.time for lbl in tupOfLabels])  # sum b/c components are *layers* of sub-circuit
        else:
            ret.time = time
        return ret

    @property
    def name(self):
        """
        This label's name (a string).
        """
        return self[0]

    @property
    def sslbls(self):
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        return self[1]

    @property
    def reps(self):
        """
        Number of repetitions (of this label's components) that this label represents.
        """
        return self[2]

    @property
    def args(self):
        """
        This label's arguments.
        """
        raise NotImplementedError("TODO!")

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self[3:]

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices. (a tuple)
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
        CircuitLabel
        """
        if isinstance(mapper, dict):
            mapped_sslbls = [mapper[sslbl] for sslbl in self.sslbls]
        else:  # assume mapper is callable
            mapped_sslbls = [mapper(sslbl) for sslbl in self.sslbls]
        return CircuitLabel(self.name,
                            tuple((lbl.map_state_space_labels(mapper) for lbl in self.components)),
                            mapped_sslbls,
                            self[2])

    def strip_args(self):
        raise NotImplementedError("TODO!")

    def __str__(self):
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

    def __repr__(self):
        return "CircuitLabel(" + repr(self.name) + "," + repr(self[3:]) + "," \
            + repr(self[1]) + "," + repr(self[2]) + "," + repr(self.time) + ")"

    def __add__(self, s):
        raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x):
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (CircuitLabel, (self[0], self[3:], self[1], self[2], self.time), None)

    def __contains__(self, x):
        # "recursive" contains checks component containers
        return any([(x == layer or x in layer) for layer in self.components])

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return self[0:3] + tuple((x.to_native() for x in self.components))

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
        return CircuitLabel(self.name,
                            tuple((x.replace_name(oldname, newname) for x in self.components)),
                            self.sslbls,
                            self[2])

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return True  # still true - even though can have components!

    @property
    def depth(self):
        """
        The depth of this label, viewed as a sub-circuit.
        """
        return sum([x.depth for x in self.components]) * self.reps

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
        return tuple(_itertools.chain(*[x.expand_subcircuits() for x in self.components])) * self.reps

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


#class NamedLabelTupTup(Label,tuple):
#    def __new__(cls,name,tup_of_tups):
#        pass


class LabelTupWithArgs(Label, tuple):
    """
    A label consisting of a string along with a tuple of integers or state-space-names.

    These state-space sector names specify which qubits, or more generally,
    parts of the Hilbert space that is acted upon by the object this label
    refers to.  This label type also supports having arguments and a time value.
    """

    @classmethod
    def init(cls, name, state_space_labels, time=0.0, args=()):
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
        integerized_sslbls = []
        for ssl in state_space_labels:
            try: integerized_sslbls.append(int(ssl))
            except: integerized_sslbls.append(_sys.intern(ssl))

        # Regardless of whether the input is a list, tuple, or int, the state space labels
        # (qubits) that the item/gate acts on are stored as a tuple (because tuples are immutable).
        sslbls = tuple(integerized_sslbls)
        args = tuple(args)
        tup = (_sys.intern(name), 2 + len(args)) + args + sslbls  # stores: (name, K, args, sslbls)
        # where K is the index of the start of the sslbls (or 1 more than the last arg index)

        return cls.__new__(cls, tup, time)

    def __new__(cls, tup, time=0.0):
        ret = tuple.__new__(cls, tup)  # creates a LabelTup object using tuple's __new__
        ret.time = time
        return ret

    @property
    def name(self):
        """
        This label's name (a string).
        """
        return self[0]

    @property
    def sslbls(self):
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        if len(self) > self[1]:
            return self[self[1]:]
        else: return None

    @property
    def args(self):
        """
        This label's arguments.
        """
        return self[2:self[1]]

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return (self,)  # just a single "sub-label" component

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices. (a tuple)
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
        if isinstance(mapper, dict):
            mapped_sslbls = [mapper[sslbl] for sslbl in self.sslbls]
        else:  # assume mapper is callable
            mapped_sslbls = [mapper(sslbl) for sslbl in self.sslbls]
        return Label(self.name, mapped_sslbls, self.time, self.args)
        # FUTURE: use LabelTupWithArgs here instead of Label?

    def strip_args(self):
        if self.sslbls is not None:
            return LabelTup.__new__(LabelTup, (self[0],) + self[self[1]:])  # make a new LabelTup (no args)
        else:  # special case of sslbls == None, which is just a string label without its args
            return LabelStr.__new__(LabelStr, self[0])

    def __str__(self):
        """
        Defines how a Label is printed out, e.g. Gx:0 or Gcnot:1:2
        """
        #caller = inspect.getframeinfo(inspect.currentframe().f_back)
        #ky = "%s:%s:%d" % (caller[2],os.path.basename(caller[0]),caller[1])
        #_debug_record[ky] = _debug_record.get(ky, 0) + 1
        s = str(self.name)
        if self.args:  # test for None and len == 0
            s += ";" + ";".join(map(str, self.args))
        if self.sslbls:  # test for None and len == 0
            s += ":" + ":".join(map(str, self.sslbls))
        if self.time != 0.0:
            s += ("!%f" % self.time).rstrip('0').rstrip('.')
        return s

    def __repr__(self):
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self.name) + "," + repr(self.sslbls) + ",args=" + repr(self.args) + timearg + ")"

        #Alternate way of giving rep (this pattern could be repeated for other label classes too):
        #singletup = (self.name,) + self.sslbls
        #for arg in self.args: singletup += (';', arg)
        #if self.time != 0.0: singletup += ("!", self.time)
        #return "Label(" + repr(singletup) + ")"

    def __add__(self, s):
        if isinstance(s, str):
            return LabelTupWithArgs.init(self.name + s, self.sslbls, self.time, self.args)
        else:
            raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        try:
            return tuple.__lt__(self, tuple(x))
        except:
            tuple.__lt__(tuple(map(str, self)), tuple(map(str, x)))

    def __gt__(self, x):
        try:
            return tuple.__gt__(self, tuple(x))
        except:
            tuple.__gt__(tuple(map(str, self)), tuple(map(str, x)))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupWithArgs, (self[:], self.time), None)

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return tuple(self)

    def replacename(self, oldname, newname):
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
        return LabelTupWithArgs(newname, self.sslbls, self.time, self.args) if (self.name == oldname) else self

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return True

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost


class LabelTupTupWithArgs(Label, tuple):
    """
    A label consisting of a *tuple* of (string, state-space-labels) tuples.

    This typically labels a layer of a circuit (a parallel level of gates).
    This label type also supports having arguments and a time value.
    """

    @classmethod
    def init(cls, tup_of_tups, time=None, args=()):
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
        assert(time is None or isinstance(time, float)), "`time` must be a floating point value, received: " + str(time)
        assert(len(args) > 0), "`args` must be a nonempty list/tuple of hashable arguments"
        tupOfLabels = (1 + len(args),) + args + tuple((Label(tup) for tup in tup_of_tups))  # Note tup can be a Label
        # stores: (K, args, subLabels) where K is the index of the start of subLabels

        #if time is not None:
        #    assert(all([(time == l.time or l.time is None) for l in tupOfLabels[1 + len(args):]])), \
        #        "Component times do not match compound label time!"
        if time is None:
            time = 0.0 if len(tupOfLabels) == 0 else \
                max([lbl.time for lbl in tupOfLabels])
        return cls.__new__(cls, tupOfLabels, time)

    def __new__(cls, tup_of_labels, time=0.0):
        ret = tuple.__new__(cls, tup_of_labels)  # creates a LabelTupTup object using tuple's __new__
        ret.time = time
        return ret

    @property
    def name(self):
        # TODO - something intelligent here?
        # no real "name" for a compound label... but want it to be a string so
        # users can use .startswith, etc.
        """
        This label's name (a string).
        """
        return "COMPOUND"

    @property
    def sslbls(self):
        # Note: if any component has sslbls == None, which signifies operating
        # on *all* qubits, then this label is on *all* qubits
        """
        This label's state-space labels, often qubit labels (a tuple).
        """
        s = set()
        for lbl in self[self[0]:]:
            if lbl.sslbls is None: return None
            s.update(lbl.sslbls)
        return tuple(sorted(list(s)))

    @property
    def args(self):
        """
        This label's arguments.
        """
        return self[1:self[0]]

    @property
    def components(self):
        """
        The sub-label components of this label, or just `(self,)` if no sub-labels exist.
        """
        return self[self[0]:]  # a tuple of "sub-label" components

    @property
    def qubits(self):  # Used in Circuit
        """
        An alias for sslbls, since commonly these are just qubit indices. (a tuple)
        """
        return self.sslbls

    @property
    def num_qubits(self):  # Used in Circuit
        """
        The number of qubits this label "acts" on (an integer). `None` if `self.ssbls is None`.
        """
        return len(self.sslbls) if (self.sslbls is not None) else None

    def has_prefix(self, prefix, typ="all"):
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
            return all([lbl.has_prefix(prefix) for lbl in self.components])
        elif typ == "any":
            return any([lbl.has_prefix(prefix) for lbl in self.components])
        else: raise ValueError("Invalid `typ` arg: %s" % str(typ))

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
        return LabelTupTupWithArgs(tuple((lbl.map_state_space_labels(mapper)
                                          for lbl in self.components)), self.time, self.args)

    def strip_args(self):
        """ Return version of self with all arguments removed """
        # default, appropriate for a label without args or components
        return LabelTupTupWithTime.__new__(LabelTupTupWithTime, (comp.strip_args() for comp in self), self.time)

    def __str__(self):
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

    def __repr__(self):
        timearg = ",time=" + repr(self.time) if (self.time != 0.0) else ""
        return "Label(" + repr(self[:]) + timearg + ")"

    def __add__(self, s):
        raise NotImplementedError("Cannot add %s to a Label" % str(type(s)))

    def __eq__(self, other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        #Unnecessary now that we have a separate LabelStr
        #if isinstance(other, str):
        #    if self.sslbls: return False # tests for None and len > 0
        #    return self.name == other

        return tuple.__eq__(self, other)
        #OLD return self.name == other.name and self.sslbls == other.sslbls # ok to compare None

    def __lt__(self, x):
        return tuple.__lt__(self, tuple(x))

    def __gt__(self, x):
        return tuple.__gt__(self, tuple(x))

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __reduce__(self):
        # Need to tell serialization logic how to create a new Label since it's derived
        # from the immutable tuple type (so cannot have its state set after creation)
        return (LabelTupTupWithArgs, (self[:], self.time), None)

    def __contains__(self, x):
        # "recursive" contains checks component containers
        return any([(x == layer or x in layer) for layer in self.components])

    def to_native(self):
        """
        Returns this label as native python types.

        Useful for faster serialization.

        Returns
        -------
        tuple
        """
        return self[0:self[0]] + tuple((x.to_native() for x in self[self[0]:]))

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
        LabelTupTupWithArgs
        """
        return LabelTupTupWithArgs(tuple((x.replace_name(oldname, newname) for x in self.components)),
                                   self.time, self.args)

    def is_simple(self):
        """
        Whether this is a "simple" (opaque w/a true name, from a circuit perspective) label or not.

        Returns
        -------
        bool
        """
        return False

    @property
    def depth(self):
        """
        The depth of this label, viewed as a sub-circuit.
        """
        if len(self.components) == 0: return 1  # still depth 1 even if empty
        return max([x.depth for x in self.components])

    __hash__ = tuple.__hash__  # this is why we derive from tuple - using the
    # native tuple.__hash__ directly == speed boost
