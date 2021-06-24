"""
Defines the ProcessorSpec class and supporting functionality.
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

from pygsti.tools import internalgates as _itgs


class ProcessorSpec(object):
    """
    The device specification for a one or more qubit quantum computer.

    This is objected is geared towards multi-qubit devices; many of the contained
    structures are superfluous in the case of a single qubit.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the device.

    gate_names : list of strings
        The names of gates in the device.  This may include standard gate
        names known by pyGSTi (see below) or names which appear in the
        `nonstd_gate_unitaries` argument. The set of standard gate names
        includes, but is not limited to:

        - 'Gi' : the 1Q idle operation
        - 'Gx','Gy','Gz' : 1-qubit pi/2 rotations
        - 'Gxpi','Gypi','Gzpi' : 1-qubit pi rotations
        - 'Gh' : Hadamard
        - 'Gp' : phase or S-gate (i.e., ((1,0),(0,i)))
        - 'Gcphase','Gcnot','Gswap' : standard 2-qubit gates

        Alternative names can be used for all or any of these gates, but
        then they must be explicitly defined in the `nonstd_gate_unitaries`
        dictionary.  Including any standard names in `nonstd_gate_unitaries`
        overrides the default (builtin) unitary with the one supplied.

    nonstd_gate_unitaries: dictionary of numpy arrays
        A dictionary with keys that are gate names (strings) and values that are numpy arrays specifying
        quantum gates in terms of unitary matrices. This is an additional "lookup" database of unitaries -
        to add a gate to this `ProcessorSpec` its names still needs to appear in the `gate_names` list.
        This dictionary's values specify additional (target) native gates that can be implemented in the device
        as unitaries acting on ordinary pure-state-vectors, in the standard computationl basis. These unitaries
        need not, and often should not, be unitaries acting on all of the qubits. E.g., a CNOT gate is specified
        by a key that is the desired name for CNOT, and a value that is the standard 4 x 4 complex matrix for CNOT.
        All gate names must start with 'G'.  As an advanced behavior, a unitary-matrix-returning function which
        takes a single argument - a tuple of label arguments - may be given instead of a single matrix to create
        an operation *factory* which allows continuously-parameterized gates.  This function must also return
        an empty/dummy unitary when `None` is given as it's argument.

    availability : dict, optional
        A dictionary whose keys are some subset of the keys (which are gate names) `nonstd_gate_unitaries` and the
        strings (which are gate names) in `gate_names` and whose values are lists of qubit-label-tuples.  Each
        qubit-label-tuple must have length equal to the number of qubits the corresponding gate acts upon, and
        causes that gate to be available to act on the specified qubits. Instead of a list of tuples, values of
        `availability` may take the special values `"all-permutations"` and `"all-combinations"`, which as their
        names imply, equate to all possible permutations and combinations of the appropriate number of qubit labels
        (deterined by the gate's dimension). If a gate name is not present in `availability`, the default is
        `"all-permutations"`.  So, the availability of a gate only needs to be specified when it cannot act in every
        valid way on the qubits (e.g., the device does not have all-to-all connectivity).

    qubit_labels : list or tuple, optional
        The labels (integers or strings) of the qubits.  If `None`, then the integers starting with zero are used.
    """

    def __init__(self, num_qubits, gate_names, nonstd_gate_unitaries=None, availability=None, qubit_labels=None):
        assert(type(num_qubits) is int), "The number of qubits, n, should be an integer!"

        #Store inputs for adding models later
        self.gate_names = gate_names[:]  # copy this list
        self.nonstd_gate_unitaries = nonstd_gate_unitaries.copy() if (nonstd_gate_unitaries is not None) else {}
        #self.gate_names += list(self.nonstd_gate_unitaries.keys())  # must specify all names in `gate_names`
        self.availability = availability.copy() if (availability is not None) else {}

        # Stores the basic unitary matrices defining the gates, as it is convenient to have these easily accessable.
        self.gate_unitaries = _collections.OrderedDict()
        std_gate_unitaries = _itgs.standard_gatename_unitaries()
        for gname in gate_names:
            if gname in nonstd_gate_unitaries:
                self.gate_unitaries[gname] = nonstd_gate_unitaries[gname]
            elif gname in std_gate_unitaries:
                self.gate_unitaries[gname] = std_gate_unitaries[gname]
            else:
                raise ValueError(
                    str(gname) + " is not a valid 'standard' gate name, it must be given in `nonstd_gate_unitaries`")

        # If no qubit labels are provided it defaults to integers from 0 to num_qubits-1.
        if qubit_labels is None:
            self.qubit_labels = tuple(range(num_qubits))
        else:
            assert(len(qubit_labels) == num_qubits)
            self.qubit_labels = tuple(qubit_labels)

    @property
    def number_of_qubits(self):
        """ The number of qubits. """
        return len(self.qubit_labels)
