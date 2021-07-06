"""
Defines the QubitProcessorSpec class and supporting functionality.
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
import itertools as _itertools
import collections as _collections

from pygsti.tools import internalgates as _itgs
from pygsti.tools import symplectic as _symplectic
from pygsti.tools import optools as _ot
from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.baseobjs.label import Label as _Lbl

class ProcessorSpec(object):
    pass  # base class for potentially other types of processors (not composed of just qubits)


class QubitProcessorSpec(ProcessorSpec):
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
        to add a gate to this `QubitProcessorSpec` its names still needs to appear in the `gate_names` list.
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

    geometry : {"line","ring","grid","torus"} or QubitGraph, optional
        The type of connectivity among the qubits, specifying a graph used to
        define neighbor relationships.  Alternatively, a :class:`QubitGraph`
        object with `qubit_labels` as the node labels may be passed directly.
        This argument is only used as a convenient way of specifying gate
        availability (edge connections are used for gates whose availability
        is unspecified by `availability` or whose value there is `"all-edges"`).

    qubit_labels : list or tuple, optional
        The labels (integers or strings) of the qubits.  If `None`, then the integers starting with zero are used.

    aux_info : dict, optional
        Any additional information that should be attached to this processor spec.
    """

    def __init__(self, num_qubits, gate_names, nonstd_gate_unitaries=None, availability=None,
                 geometry=None, qubit_labels=None, aux_info=None):
        assert(type(num_qubits) is int), "The number of qubits, n, should be an integer!"
        if nonstd_gate_unitaries is None: nonstd_gate_unitaries = {}

        #Store inputs for adding models later
        self.gate_names = tuple(gate_names[:])  # copy & cast to tuple
        self.nonstd_gate_unitaries = nonstd_gate_unitaries.copy() if (nonstd_gate_unitaries is not None) else {}
        #self.gate_names += list(self.nonstd_gate_unitaries.keys())  # must specify all names in `gate_names`

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

       # Set self.qubit_graph (can be None)
        if geometry is None:
            if qubit_labels is None:
                qubit_labels = tuple(range(num_qubits))
            self.qubit_graph = _qgraph.QubitGraph(qubit_labels)  # creates a graph with no edges
        elif isinstance(geometry, _qgraph.QubitGraph):
            self.qubit_graph = geometry
            if qubit_labels is None:
                qubit_labels = self.qubit_graph.node_names
        else:  # assume geometry is a string
            if qubit_labels is None:
                qubit_labels = tuple(range(num_qubits))
            self.qubit_graph = _qgraph.QubitGraph.common_graph(num_qubits, geometry, directed=False,
                                                               qubit_labels=qubit_labels)

        # If no qubit labels are provided it defaults to integers from 0 to num_qubits-1.
        if qubit_labels is None:
            self.qubit_labels = tuple(range(num_qubits))
        else:
            assert(len(qubit_labels) == num_qubits)
            self.qubit_labels = tuple(qubit_labels)

        # Set availability
        if availability is None: availability = {}
        self.availability = _collections.OrderedDict([(gatenm, availability.get(gatenm, 'all-edges'))
                                                      for gatenm in self.gate_names])  #if _Lbl(gatenm).sslbls is not None NEEDED?

        self.compiled_from = None  # could hold (QubitProcessorSpec, compilations) tuple if not None
        self.aux_info = aux_info  # can hold anything additional (e.g. gate inverse relationships)
        self._symplectic_reps = {}  # lazily-evaluated symplectic representations for Clifford gates
        super(QubitProcessorSpec, self).__init__()

    @property
    def num_qubits(self):
        """ The number of qubits. """
        return len(self.qubit_labels)

    def gate_number_of_qubits(self, gate_name):
        unitary = self.gate_unitaries[gate_name]
        return int(round(_np.log2(unitary.udim if callable(unitary) else unitary.shape[0])))

    def resolved_availability(self, gate_name, tuple_or_function="auto"):
        """ TODO: docstring -- returns the availability resolved as either a tuple of sslbl-tuples or a fn"""
        assert(tuple_or_function in ('tuple', 'function', 'auto'))
        avail_entry = self.availability.get(gate_name, 'all-edges')
        gate_nqubits = self.gate_number_of_qubits(gate_name)
        return self._resolve_availability(avail_entry, gate_nqubits, tuple_or_function)

    def _resolve_availability(self, avail_entry, gate_nqubits, tuple_or_function="auto"):

        if callable(avail_entry):  # a boolean function(sslbls)
            if tuple_or_function == "tuple":
                return tuple([sslbls for sslbls in _itertools.permutations(self.qubit_labels, gate_nqubits)
                              if avail_entry(sslbls)])
            return avail_entry  # "auto" also comes here

        elif avail_entry == 'all-combinations':
            if tuple_or_function == "function":
                def _f(sslbls):
                    return set(sslbls).issubset(self.qubit_labels) and tuple(sslbls) == tuple(sorted(sslbls))
                return _f
            return tuple(_itertools.combinations(self.qubit_labels, gate_nqubits))  # "auto" also comes here

        elif avail_entry == 'all-permutations':
            if tuple_or_function == "function":
                def _f(sslbls):
                    return set(sslbls).issubset(self.qubit_labels)
                return _f
            return tuple(_itertools.permutations(self.qubit_labels, gate_nqubits))  # "auto" also comes here

        elif avail_entry == 'all-edges':
            assert(gate_nqubits in (1,2)), \
                "I don't know how to place a %d-qubit gate on graph edges yet" % gate_nqubits
            if tuple_or_function == "function":
                def _f(sslbls):
                    if len(sslbls) == 1: return True
                    elif len(sslbls) == 2: return self.qubit_graph.is_directly_connected(sslbls[0], sslbls[1])
                    else: raise NotImplementedError()
                return _f

            # "auto" also comes here:
            if gate_nqubits == 1: return tuple([(i,) for i in self.qubit_labels])
            elif gate_nqubits == 2: return tuple(self.qubit_graph.edges(double_for_undirected=True))
            else: raise NotImplementedError()

        elif avail_entry in ('arbitrary', '*'):  # indicates user supplied factory determines allowed sslbls
            return '*'  # special signal value for this case

        else:
            if not isinstance(avail_entry, (list, tuple)):
                raise ValueError("Unrecognized availability entry: " + str(avail_entry))
            if tuple_or_function == "function":
                def _f(sslbls):
                    return sslbls in avail_entry
                return _f
            return avail_entry  # "auto" also comes here

    def is_available(self, gate_label):
        """ TODO: docstring """
        if not isinstance(gate_label, _Lbl):
            gate_label = _Lbl(gate_label)
        test_fn = self.resolved_availability(gate_label.name, "function")
        if test_fn == '*':
            return True  # really should check gate factory function somehow? TODO
        else:
            return test_fn(gate_label.sslbls)

    def available_gatenames(self, sslbls):
        """ TODO: docstring - return all the gate names that are available for at least a subset of `sslbls`"""
        ret = []
        for gn in self.gate_names:
            gn_nqubits = self.gate_number_of_qubits(gn)
            avail_fn = self.resolved_availability(gn, tuple_or_function="function")
            if gn_nqubits > len(sslbls): continue  # gate has too many qubits to fit in sslbls
            if any((avail_fn(sslbls_subset) for sslbls_subset in _itertools.permutations(sslbls, gn_nqubits))):
                ret.append(gn)
        return tuple(ret)

    def available_gatelabels(self, gate_name, sslbls):
        """ TODO: docstring - return all the gate labels that are available for `gatename` on
            at least a subset of `sslbls`"""
        ret = []
        gate_nqubits = self.gate_number_of_qubits(gate_name)
        avail_fn = self.resolved_availability(gate_name, tuple_or_function="function")
        if gate_nqubits > len(sslbls): return ()  # gate has too many qubits to fit in sslbls
        return tuple((_Lbl(gate_name, sslbls_subset) for sslbls_subset in _itertools.permutations(sslbls, gate_nqubits)
                     if avail_fn(sslbls_subset)))

    def compute_clifford_symplectic_reps(self, gatename_filter=None):
        """
        Constructs a dictionary of the symplectic representations for all the Clifford gates in this processor spec.

        Parameters
        ----------
        gatename_filter : iterable, optional
            A list, tuple, or set of gate names whose symplectic
            representations should be returned (if they exist).

        Returns
        -------
        dict
            keys are gate names, values are
            `(symplectic_matrix, phase_vector)` tuples.
        """
        ret = {}
        for gn, unitary in self.gate_unitaries.items():
            if gatename_filter is not None and gn not in gatename_filter: continue
            if gn not in self._symplectic_reps:
                try:
                    self._symplectic_reps[gn] = _symplectic.unitary_to_symplectic(unitary)
                except ValueError:
                    self._symplectic_reps[gn] = None  # `gn` is not a Clifford
            if self._symplectic_reps[gn] is not None:
                ret[gn] = self._symplectic_reps[gn]
        return ret

    def compute_one_qubit_gate_relations(self):
        """
        Computes the basic pair-wise relationships relationships between the gates.

        1. It multiplies all possible combinations of two 1-qubit gates together, from
        the full model available to in this device. If the two gates multiple to
        another 1-qubit gate from this set of gates this is recorded in the dictionary
        self.oneQgate_relations. If the 1-qubit gate with name `name1` followed by the
        1-qubit gate with name `name2` multiple (up to phase) to the gate with `name3`,
        then self.oneQgate_relations[`name1`,`name2`] = `name3`.

        2. If the inverse of any 1-qubit gate is contained in the model, this is
        recorded in the dictionary self.gate_inverse.

        Returns
        -------
        gate_relations : dict
            Keys are `(gatename1, gatename2)` and values are either the gate name
            of the product of the two gates or `None`, signifying the identity.
        gate_inverses : dict
            Keys and values are gate names, mapping a gate name to its inverse
            gate (if one exists).
        """
        Id = _np.identity(4, float)
        nontrivial_gname_pauligate_pairs = []
        oneQgate_relations = {}
        gate_inverse = {}

        for gname in self.gate_names:
            if callable(self.gate_unitaries[gname]): continue  # can't pre-process factories

            # We convert to process matrices, to avoid global phase problems.
            u = _ot.unitary_to_pauligate(self.gate_unitaries[gname])
            if u.shape == (4, 4):
                #assert(not _np.allclose(u,Id)), "Identity should *not* be included in root gate names!"
                #if _np.allclose(u, Id):
                #    _warnings.warn("The identity should often *not* be included "
                #                   "in the root gate names of a QubitProcessorSpec.")
                nontrivial_gname_pauligate_pairs.append((gname, u))

        for gname1, u1 in nontrivial_gname_pauligate_pairs:
            for gname2, u2 in nontrivial_gname_pauligate_pairs:
                ucombined = _np.dot(u2, u1)
                for gname3, u3 in nontrivial_gname_pauligate_pairs:
                    if _np.allclose(u3, ucombined):
                        # If ucombined is u3, add the gate composition relation.
                        oneQgate_relations[gname1, gname2] = gname3  # != Id (asserted above)
                    if _np.allclose(ucombined, Id):
                        # If ucombined is the identity, add the inversion relation.
                        gate_inverse[gname1] = gname2
                        gate_inverse[gname2] = gname1
                        oneQgate_relations[gname1, gname2] = None
                        # special 1Q gate relation where result is the identity (~no gates)
        return oneQgate_relations, gate_inverse

    def compute_multiqubit_inversion_relations(self):
        """
        Computes the inverses of multi-qubit (>1 qubit) gates.

        Finds whether any of the multi-qubit gates in this device also have their
        inverse in the model. That is, if the unitaries for the  multi-qubit gate with
        name `name1` followed by the multi-qubit gate (of the same dimension) with
        name `name2` multiple (up to phase) to the identity, then
        gate_inverse[`name1`] = `name2` and gate_inverse[`name2`] = `name1`

        1-qubit gates are not computed by this method, as they are be computed by the method
        :method:`compute_one_qubit_gate_relations`.

        Returns
        -------
        gate_inverse : dict
            Keys and values are gate names, mapping a gate name to its inverse
            gate (if one exists).
        """
        gate_inverse = {}
        for gname1 in self.gate_names:
            if callable(self.gate_unitaries[gname1]): continue  # can't pre-process factories

            # We convert to process matrices, to avoid global phase problems.
            u1 = _ot.unitary_to_pauligate(self.gate_unitaries[gname1])
            if _np.shape(u1) != (4, 4):
                for gname2 in self.gate_names:
                    if callable(self.gate_unitaries[gname2]): continue  # can't pre-process factories
                    u2 = _ot.unitary_to_pauligate(self.gate_unitaries[gname2])
                    if _np.shape(u2) == _np.shape(u1):
                        ucombined = _np.dot(u2, u1)
                        if _np.allclose(ucombined, _np.identity(_np.shape(u2)[0], float)):
                            gate_inverse[gname1] = gname2
                            gate_inverse[gname2] = gname1
        return gate_inverse

    def compile(self, compilation_rules):
        """
        TODO: docstring

        Parameters
        ----------
        compilation_rules : CompilationRules

        Returns
        -------
        QubitProcessorSpec
        """
        from pygsti.processors.compilationrules import CompilationRules as _CompilationRules
        from pygsti.processors.compilationrules import CompilationError as _CompilationError
        compilation_rules = _CompilationRules.cast(compilation_rules)
        gate_names = tuple(compilation_rules.gate_unitaries.keys())
        gate_unitaries = compilation_rules.gate_unitaries.copy()  # can contain `None` entries we deal with below

        availability = {}
        for gn in gate_names:
            if gn in compilation_rules.local_templates:
                # merge availabilities from gates in local template
                compilation_circuit = compilation_rules.local_templates[gn]
                all_sslbls = compilation_circuit.line_labels
                gn_nqubits = len(all_sslbls)
                assert(all_sslbls == tuple(range(0,len(gn_nqubits)))), \
                    "Template circuits *must* have line labels == 0...(gate's #qubits-1), not %s!" % (str(all_sslbls))

                # To construct the availability for a circuit, we take the intersection
                # of the availability for each of the layers.  Each layer's availability is
                # the cartesian-like product of the availabilities for each of the components
                circuit_availability = None
                for layer in compilation_circuit[:]:
                    layer_availability_factors = []
                    layer_availability_sslbls = []
                    for gate in layer.components:
                        gate_availability = self.availability[gate.name]
                        if gate_availability in ('all-edges', 'all-combinations', 'all-permutations'):
                            raise NotImplementedError("Cannot merge special availabilities yet")
                        layer_availability_factors.append(gate_availability)

                        gate_sslbls = gate.sslbls
                        if gate_sslbls is None: gate_sslbls = all_sslbls
                        assert(len(set(layer_availability_sslbls).intersection(gate_sslbls)) == 0), \
                            "Duplicate state space labels in layer: %s" % str(layer)
                        layer_availability_sslbls.extend(gate_sslbls)  # integers
                    layer_availability = _itertools.product(*layer_availability_factors)
                    if tuple(layer_availability_sslbls) != all_sslbls:  # then need to permute availability elements
                        p = {to: frm for frm, to in enumerate(layer_availability_sslbls)}  # use sslbls as *indices*
                        new_order = [p[i] for i in range(gn_nqubits)]
                        layer_availability = tuple(map(lambda el: tuple([el[i] for i in new_order]),
                                                       layer_availability))
                    circuit_availability = layer_availability if (circuit_availability is None) else \
                        circuit_availability.intersection(layer_availability)
                assert(circuit_availability is not None), "Local template circuit cannot be empty!"
                availability[gn] = tuple(sorted(circuit_availability))

                if gate_unitaries[gn] is None:
                    #TODO: compute unitary via product of embedded unitaries of circuit layers, something like:
                    # gate_unitaries[gn] = product(
                    #      [kronproduct(
                    #           [embed(self.gate_unitaries[gate.name], gate.sslbls, range(gn_nqubits))
                    #            for gate in layer.components])
                    #       for layer in compilation_circuit)])
                    raise NotImplementedError("Still need to implement product of unitaries logic!")

            elif gn in compilation_rules.function_templates:
                # create boolean oracle function for availability
                def _fn(sslbls):
                    try:
                        compilation_rules.function_templates[gn](sslbls)  # (returns a circuit)
                        return True
                    except _CompilationError:
                        return False
                availability[gn] = _fn  # boolean function indicating availability
            else:
                availability[gn] = ()   # empty tuple for absent gates - OK b/c may have specific compilations

            if gate_unitaries[gn] is None:
                raise ValueError("Must specify unitary for gate name '%s'" % str(gn))

        # specific compilations add specific availability for their gate names:
        for gate_lbl in compilation_rules.specific_compilations.keys():
            assert(gate_lbl.name in gate_names), \
                "gate name '%s' missing from CompilationRules gate unitaries!" % gate_lbl.name
            assert(isinstance(availability[gate_lbl.name], tuple)), \
                "Cannot add specific values to non-explicit availabilities (e.g. given by functions)"
            availability[gate_lbl.name] += (gate_lbl.sslbls,)

        ret = QubitProcessorSpec(self.num_qubits, gate_names, gate_unitaries, availability,
                                 self.qubit_graph, self.qubit_labels)
        ret.compiled_from = (self, compilation_rules)
        return ret

    def subset(self, gate_names_to_include):
        gate_names = [gn for gn in gate_names_to_include if gn in self.gate_names]
        gate_unitaries = {gn: self.gate_unitaries[gn] for gn in gate_names}
        availability = {gn: self.availability[gn] for gn in gate_names}
        return QubitProcessorSpec(self.num_qubits, gate_names, gate_unitaries, availability,
                                  self.qubit_graph, self.qubit_labels)