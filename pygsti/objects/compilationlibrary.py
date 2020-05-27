"""
Defines CompilationLibrary class and supporting functions
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
import collections as _collections
import itertools as _itertools
from scipy.sparse.csgraph import floyd_warshall as _fw

from ..tools import symplectic as _symp
from ..tools import listtools as _lt
from .label import Label as _Label
from .circuit import Circuit as _Circuit
from .qubitgraph import QubitGraph as _QubitGraph

IDENT = 'I'  # internal 1Q-identity-gate name used for compilation
# MUST be the same as in processorspec.py


class CompilationError(Exception):
    """
    A compilation error, raised by :class:`CompilationLibrary`
    """
    pass


class CompilationLibrary(_collections.OrderedDict):
    """
    An collection of compilations for gates.

    Essentially an ordered dictionary whose keys are operation labels
    (:class:`Label` objects) and whose values are circuits
    (:class:`Circuit` objects).  A `CompilationLibrary` holds a :class:`Model`
    which specifies the "native" gates that all compilations are made up of.
    Currently, this model should only contain Clifford gates, so that its
    `compute_clifford_symplectic_reps` method gives representations for all of its
    gates.

    Compilations can be either "local" or "non-local". A local compilation
    ony uses gates that act on its target qubits.  All 1-qubit gates can be
    local.  A non-local compilation uses qubits outside the set of target
    qubits (e.g. a CNOT between two qubits between which there is no native
    CNOT).  Currently, non-local compilations can only be constructed for
    the CNOT gate.

    To speed up the creation of local compilations, a `CompilationLibrary`
    stores "template" compilations, which specify how to construct a
    compilation for some k-qubit gate on qubits labeled 0 to k-1.  When creating
    a compilation for a gate, a template is used if a suitable one can be found;
    otherwise a new template is created and then used.

    Compilation libraries are most often used within a :class:`ProcessorSpec`
    object.

    Parameters
    ----------
    clifford_model : Model
        The model of "native" Clifford gates which all compilations in
        this library are composed from.

    ctyp : {"absolute","paulieq"}
        The "compilation type" for this library.  If `"absolute"`, then
        compilations must match the gate operation being compiled exactly.
        If `"paulieq"`, then compilations only need to match the desired
        gate operation up to a Paui operation (which is useful for compiling
        multi-qubit Clifford gates / stabilizer states without unneeded 1-qubit
        gate over-heads).

    items : list, optional
        initial items (key,value pairs) to place in library.
    """

    def __init__(self, clifford_model, ctyp="absolute", items=[]):
        """
        Create a new CompilationLibrary.

        Parameters
        ----------
        clifford_model : Model
            The model of "native" Clifford gates which all compilations in
            this library are composed from.

        ctyp : {"absolute","paulieq"}
            The "compilation type" for this library.  If `"absolute"`, then
            compilations must match the gate operation being compiled exactly.
            If `"paulieq"`, then compilations only need to match the desired
            gate operation up to a Paui operation (which is useful for compiling
            multi-qubit Clifford gates / stabilizer states without unneeded 1-qubit
            gate over-heads).
        """
        self.model = clifford_model  # model of (all Clifford) gates to compile requested gates into
        self.ctype = ctyp  # "absolute" or "paulieq"
        self.templates = _collections.defaultdict(list)  # keys=gate names (strs); vals=tuples of Labels
        self.connectivity = {}  # QubitGraphs for gates currently compiled in library (key=gate_name)
        super(CompilationLibrary, self).__init__(items)
        #** Note: if change __init__ signature, update __reduce__ below

    def __reduce__(self):
        return (CompilationLibrary,
                (self.model, self.ctype, list(self.items())), None)

    def _create_local_compilation_of(self, oplabel, unitary=None, srep=None, max_iterations=10, verbosity=1):
        """
        Constructs a local compilation of `oplabel`.

        An existing template is used if one is available, otherwise a new
        template is created using an iterative procedure. Raises
        :class:`CompilationError` when no compilation can be found.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.  If `oplabel.name` is a
            recognized standard Clifford name (e.g. 'H', 'P', 'X', 'CNOT')
            then no further information is needed.  Otherwise, you must specify
            either (or both) of `unitary` or `srep` *unless* the compilation
            for this oplabel has already been previously constructed and force
            is `False`. In that case, the previously constructed compilation will
            be returned in all cases, and so this method does not need to know
            what the gate actually is.

        unitary : numpy.ndarray, optional
            The unitary action of the gate being compiled.  If, as is typical,
            you're compiling using Clifford gates, then this unitary should
            correspond to a Clifford operation.  If you specify `unitary`,
            you don't need to specify `srep` - it is computed automatically.

        srep : tuple, optional
            The `(smatrix, svector)` tuple giving the symplectic representation
            of the gate being compiled.

        max_iterations : int, optional
            The maximum number of iterations for the iterative compilation
            algorithm.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        Returns
        -------
        Circuit
        """
        # Template compilations always use integer qubit labels: 0 to N
        #  where N is the number of qubits in the template's overall label
        #  (i.e. its key in self.templates)
        def to_real_label(template_label):
            """ Convert a "template" operation label (which uses integer qubit labels
                0 to N) to a "real" label for a potential gate in self.model. """
            qlabels = [oplabel.qubits[i] for i in template_label.qubits]
            return _Label(template_label.name, qlabels)

        def to_template_label(real_label):
            """ The reverse (qubits in template == oplabel.qubits) """
            qlabels = [oplabel.qubits.index(lbl) for lbl in real_label.qubits]
            return _Label(real_label.name, qlabels)

        def is_local_compilation_feasible(template_labels):
            """ Whether template_labels can possibly be enough
                gates to compile a template for op_label with """
            if oplabel.number_of_qubits <= 1:
                return len(template_labels) > 0  # 1Q gates, anything is ok
            elif oplabel.number_of_qubits == 2:
                # 2Q gates need a compilation gate that is also 2Q (can't do with just 1Q gates!)
                return max([lbl.number_of_qubits for lbl in template_labels]) == 2
            else:
                # >2Q gates need to make sure there's some connected path
                return True  # future: update using graphs stuff?

        template_to_use = None

        for template_compilation in self.templates.get(oplabel.name, []):
            #Check availability of gates in self.model to determine
            # whether template_compilation can be applied.
            model_primitive_ops = self.model.primitive_op_labels()
            if all([(gl in model_primitive_ops) for gl in map(to_real_label,
                                                              template_compilation)]):
                template_to_use = template_compilation
                if verbosity > 0: print("Existing template found!")
                break  # compilation found!

        else:  # no existing templates can be applied, so make a new one

            #construct a list of the available gates on the qubits of
            # `oplabel` (or a subset of them)
            available_glabels = list(filter(lambda gl: set(gl.qubits).issubset(oplabel.qubits),
                                            self.model.primitive_op_labels()))
            available_glabels.extend([_Label(IDENT, k) for k in oplabel.qubits])
            available_template_labels = set(map(to_template_label, available_glabels))
            available_srep_dict = self.model.compute_clifford_symplectic_reps(available_glabels)
            available_srep_dict[IDENT] = _symp.unitary_to_symplectic(_np.identity(2, 'd'))
            #Manually add 1Q idle gate on each of the qubits, as this typically isn't stored in model.

            if is_local_compilation_feasible(available_template_labels):
                template_to_use = self.add_clifford_compilation_template(
                    oplabel.name, oplabel.number_of_qubits, unitary, srep,
                    available_template_labels, available_srep_dict,
                    verbosity=verbosity, max_iterations=max_iterations)

        #If a template has been found, use it.
        if template_to_use is not None:
            opstr = list(map(to_real_label, template_to_use))
            #REMOVE 'I's
            return _Circuit(layer_labels=opstr,
                            line_labels=self.model.state_space_labels.labels[0])
        else:
            raise CompilationError("Cannot locally compile %s" % str(oplabel))

    def get_local_compilation_of(self, oplabel, unitary=None, srep=None, max_iterations=10, force=False, verbosity=1):
        """
        Gets a new local compilation of `oplabel`.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.  If `oplabel.name` is a
            recognized standard Clifford name (e.g. 'H', 'P', 'X', 'CNOT')
            then no further information is needed.  Otherwise, you must specify
            either (or both) of `unitary` or `srep`.

        unitary : numpy.ndarray, optional
            The unitary action of the gate being compiled.  If, as is typical,
            you're compiling using Clifford gates, then this unitary should
            correspond to a Clifford operation.  If you specify `unitary`,
            you don't need to specify `srep` - it is computed automatically.

        srep : tuple, optional
            The `(smatrix, svector)` tuple giving the symplectic representation
            of the gate being compiled.

        max_iterations : int, optional
            The maximum number of iterations for the iterative compilation
            algorithm.

        force : bool, optional
            If True, then a compilation is recomputed even if `oplabel`
            already exists in this `CompilationLibrary`.  Otherwise
            compilations are only computed when they are *not* present.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        Returns
        -------
        None
        """
        if not force and oplabel in self:
            return self[oplabel]  # don't re-compute unless we're told to

        circuit = self._create_local_compilation_of(oplabel,
                                                      unitary=unitary,
                                                      srep=srep,
                                                      max_iterations=max_iterations,
                                                      verbosity=verbosity)
        return circuit

    def add_local_compilation_of(self, oplabel, unitary=None, srep=None, max_iterations=10, force=False, verbosity=1):
        """
        Adds a new local compilation of `oplabel`.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.  If `oplabel.name` is a
            recognized standard Clifford name (e.g. 'H', 'P', 'X', 'CNOT')
            then no further information is needed.  Otherwise, you must specify
            either (or both) of `unitary` or `srep`.

        unitary : numpy.ndarray, optional
            The unitary action of the gate being compiled.  If, as is typical,
            you're compiling using Clifford gates, then this unitary should
            correspond to a Clifford operation.  If you specify `unitary`,
            you don't need to specify `srep` - it is computed automatically.

        srep : tuple, optional
            The `(smatrix, svector)` tuple giving the symplectic representation
            of the gate being compiled.

        max_iterations : int, optional
            The maximum number of iterations for the iterative compilation
            algorithm.

        force : bool, optional
            If True, then a compilation is recomputed even if `oplabel`
            already exists in this `CompilationLibrary`.  Otherwise
            compilations are only computed when they are *not* present.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        Returns
        -------
        None
        """
        self[oplabel] = self.get_local_compilation_of(oplabel, unitary, srep,
                                                      max_iterations, force,
                                                      verbosity)

    def add_clifford_compilation_template(self, gate_name, nqubits, unitary, srep,
                                          available_glabels, available_sreps,
                                          verbosity=1, max_iterations=10):
        """
        Adds a new compilation template for `gate_name`.

        Parameters
        ----------
        gate_name : str
            The gate name to create a compilation for.  If it is
            recognized standard Clifford name (e.g. 'H', 'P', 'X', 'CNOT')
            then `unitary` and `srep` can be None. Otherwise, you must specify
            either (or both) of `unitary` or `srep`.

        nqubits : int
            The number of qubits this gate acts upon.

        unitary : numpy.ndarray
            The unitary action of the gate being templated.  If, as is typical,
            you're compiling using Clifford gates, then this unitary should
            correspond to a Clifford operation.  If you specify `unitary`,
            you don't need to specify `srep` - it is computed automatically.

        srep : tuple, optional
            The `(smatrix, svector)` tuple giving the symplectic representation
            of the gate being templated.

        available_glabels : list
            A list of the gate labels (:class:`Label` objects) that are available for
            use in compilations.

        available_sreps : dict
            A dictionary of available symplectic representations.  Keys are gate
            labels and values are numpy arrays.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        max_iterations : int, optional
            The maximum number of iterations for the iterative
            template compilation-finding algorithm.

        Returns
        -------
        tuple
            A tuple of the operation labels (essentially a circuit) specifying
            the template compilation that was generated.
        """
        # The unitary is specifed, this takes priority and we use it to construct the
        # symplectic rep of the gate.
        if unitary is not None:
            srep = _symp.unitary_to_symplectic(unitary, flagnonclifford=True)

        # If the unitary has not been provided and smatrix and svector are both None, then
        # we find them from the dictionary of standard gates.

        if srep is None:
            template_lbl = _Label(gate_name, tuple(range(nqubits)))  # integer ascending qubit labels
            smatrix, svector = _symp.symplectic_rep_of_clifford_layer(template_lbl, nqubits)
        else:
            smatrix, svector = srep

        assert(_symp.check_valid_clifford(smatrix, svector)), "The gate is not a valid Clifford!"
        assert(_np.shape(smatrix)[0] // 2 == nqubits), \
            "The gate acts on a different number of qubits to stated by `nqubits`"

        if verbosity > 0:
            if self.ctype == 'absolute':
                print("- Generating a template for a compilation of {}...".format(gate_name), end='\n')
            elif self.ctype == 'paulieq':
                print("- Generating a template for a pauli-equivalent compilation of {}...".format(gate_name), end='\n')

        obtained_sreps = {}

        #Separate the available operation labels by their target qubits
        available_glabels_by_qubit = _collections.defaultdict(list)
        for gl in available_glabels:
            available_glabels_by_qubit[tuple(sorted(gl.qubits))].append(gl)
            #sort qubit labels b/c order doesn't matter and can't hash sets

        # Construst all possible circuit layers acting on the qubits.
        all_layers = []

        #Loop over all partitions of the nqubits
        for p in _lt.partitions(nqubits):
            pi = _np.concatenate(([0], _np.cumsum(p)))
            to_iter_over = [available_glabels_by_qubit[tuple(range(pi[i], pi[i + 1]))] for i in range(len(p))]
            for gls_in_layer in _itertools.product(*to_iter_over):
                all_layers.append(gls_in_layer)

        # Find the symplectic action of all possible circuits of length 1 on the qubits
        for layer in all_layers:
            obtained_sreps[layer] = _symp.symplectic_rep_of_clifford_layer(layer, nqubits, srep_dict=available_sreps)

        # Main loop. We go through the loop at most max_iterations times
        found = False
        for counter in range(0, max_iterations):

            if verbosity > 0:
                print("  - Checking all length {} {}-qubit circuits... ({})".format(counter + 1,
                                                                                    nqubits,
                                                                                    len(obtained_sreps)))

            candidates = []  # all valid compilations, if any, of this length.

            # Look to see if we have found a compilation
            for seq, (s, p) in obtained_sreps.items():
                if _np.array_equal(smatrix, s):
                    if self.ctype == 'paulieq' or \
                       (self.ctype == 'absolute' and _np.array_equal(svector, p)):
                        candidates.append(seq)
                        found = True

            # If there is more than one way to compile gate at this circuit length, pick the
            # one containing the most idle gates.
            if len(candidates) > 1:

                number_of_idles = 0
                max_number_of_idles = 0

                # Look at each sequence, and see if it has more than or equal to max_number_of_idles.
                # If so, set it to the current chosen sequence.
                for seq in candidates:
                    number_of_idles = len([x for x in seq if x.name == IDENT])

                    if number_of_idles >= max_number_of_idles:
                        max_number_of_idles = number_of_idles
                        compilation = seq
            elif len(candidates) == 1:
                compilation = candidates[0]

            # If we have found a compilation, leave the loop
            if found:
                if verbosity > 0: print("Compilation template created!")
                break

            # If we have reached the maximum number of iterations, quit the loop
            # before we construct the symplectic rep for all sequences of a longer length.
            if (counter == max_iterations - 1):
                print("  - Maximum iterations reached without finding a compilation !")
                return None

            # Construct the gates obtained from the next length sequences.
            new_obtained_sreps = {}

            for seq, (s, p) in obtained_sreps.items():
                # Add all possible tensor products of single-qubit gates to the end of the sequence
                for layer in all_layers:

                    # Calculate the symp rep of this parallel gate
                    sadd, padd = _symp.symplectic_rep_of_clifford_layer(layer, nqubits, srep_dict=available_sreps)
                    key = seq + layer  # tuple/Circuit concatenation

                    # Calculate and record the symplectic rep of this gate sequence.
                    new_obtained_sreps[key] = _symp.compose_cliffords(s, p, sadd, padd)

            # Update list of potential compilations
            obtained_sreps = new_obtained_sreps

        #Compilation done: remove IDENT labels, as these are just used to
        # explicitly keep track of the number of identity gates in a circuit (really needed?)
        compilation = list(filter(lambda gl: gl.name != IDENT, compilation))

        #Store & return template that was found
        self.templates[gate_name].append(compilation)

        return compilation

    #PRIVATE
    def _compute_connectivity_of(self, gate_name):
        """
        Compute the connectivity for `gate_name` using the (compiled) gates available this library.

        Connectivity is defined in terms of nearest-neighbor links, and the
        resulting :class:`QubitGraph`, is stored in `self.connectivity[gate_name]`.

        Parameters
        ----------
        gate_name : str
            gate name to compute connectivity for.

        Returns
        -------
        None
        """
        nQ = int(round(_np.log2(self.model.dim)))  # assumes *unitary* mode (OK?)
        qubit_labels = self.model.state_space_labels.labels[0]
        d = {qlbl: i for i, qlbl in enumerate(qubit_labels)}
        assert(len(qubit_labels) == nQ), "Number of qubit labels is inconsistent with Model dimension!"

        connectivity = _np.zeros((nQ, nQ), dtype=bool)
        for compiled_gatelabel in self.keys():
            if compiled_gatelabel.name == gate_name:
                for p in _itertools.permutations(compiled_gatelabel.qubits, 2):
                    connectivity[d[p[0]], d[p[1]]] = True
                    # Note: d converts from qubit labels to integer indices

        self.connectivity[gate_name] = _QubitGraph(qubit_labels, connectivity)

    def filter_connectivity(self, gate_name, allowed_filter):
        """
        Compute the QubitGraph giving the available `gate_name` gates subject to `allowed_filter`.

        The filter adds constraints to by specifying the availability of `gate_name`.

        Parameters
        ----------
        gate_name : str
            The gate name.

        allowed_filter : dict or set
            See :method:`get_nonlocal_compilation_of`.

        Returns
        -------
        QubitGraph
        """
        if gate_name not in self.connectivity:  # need to recompute
            self._compute_connectivity_of(gate_name)

        init_qgraph = self.connectivity[gate_name]  # unconstrained

        if isinstance(allowed_filter, dict):
            graph_constraint = allowed_filter.get(gate_name, None)
            if graph_constraint is not None:
                directed = graph_constraint.directed or init_qgraph.directed
                init_nodes = set(init_qgraph.node_names())
                qlabels = [lbl for lbl in graph_constraint.node_names()
                           if lbl in init_nodes]  # labels common to both graphs
                qlset = set(qlabels)  # for faster lookups
                final_edges = []
                for edge in graph_constraint.edges(True):
                    if edge[0] in qlset and edge[1] in qlset and \
                       init_qgraph.has_edge(edge):
                        final_edges.append(edge)  # edge common to both
                return _QubitGraph(qlabels, initial_edges=final_edges, directed=directed)
            else:
                return init_qgraph

        else:
            if allowed_filter is None:
                return init_qgraph
            else:
                # assume allowed_filter is iterable and contains qubit labels
                return init_qgraph.subgraph(list(allowed_filter))

    def _create_nonlocal_compilation_of(self, oplabel, allowed_filter=None, verbosity=1, check=True):
        """
        Constructs a potentially non-local compilation of `oplabel`.

        This method currently only generates a compilation for a non-local CNOT,
        up to arbitrary Pauli gates, between a pair of unconnected qubits. It
        converts this CNOT into a circuit of CNOT gates between connected qubits,
        using a fixed circuit form. This compilation is not optimal in at least
        some circumstances.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.  Currently, `oplabel.name` must
            equal `"CNOT"`.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used in this non-local
            compilation.  If a `dict`, keys must be gate names (like
            `"CNOT"`) and values :class:`QubitGraph` objects indicating
            where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in
            the current library that is confined within that set is allowed.
            If None, then all gates within the library are allowed.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        check : bool, optional
            Whether to perform internal consistency checks.

        Returns
        -------
        Circuit
        """
        assert(oplabel.number_of_qubits > 1), "1-qubit gates can't be non-local!"
        assert(oplabel.name == "CNOT" and oplabel.number_of_qubits == 2), \
            "Only non-local CNOT compilation is currently supported."

        #Get connectivity of this gate (CNOT)
        #if allowed_filter is not None:
        qgraph = self.filter_connectivity(oplabel.name, allowed_filter)
        #else:
        #    qgraph = self.connectivity[oplabel.name]

        #CNOT specific
        q1 = oplabel.qubits[0]
        q2 = oplabel.qubits[1]
        dist = qgraph.shortest_path_distance(q1, q2)

        if verbosity > 0:
            print("")
            print("Attempting to generate a compilation for CNOT, up to Paulis,")
            print("with control qubit = {} and target qubit = {}".format(q1, q2))
            print("")
            print("Distance between qubits is = {}".format(dist))

        assert(qgraph.is_connected(q1, q2) >= 0), "There is no path between the qubits!"

        # If the qubits are directly connected, this algorithm may not behave well.
        assert(not qgraph.is_directly_connected(q1, q2)), "Qubits are connected! Algorithm is not needed or valid."

        # Find the shortest path between q1 and q2
        shortestpath = qgraph.shortest_path(q1, q2)

        # Part 1 of the circuit is CNOTs along the shortest path from q1 to q2.
        # To do: describe the circuit.
        part_1 = []
        for i in range(0, len(shortestpath) - 1):
            part_1.append(_Label('CNOT', [shortestpath[i], shortestpath[i + 1]]))

        # Part 2 is...
        # To do: describe the circuit.
        part_2 = _copy.deepcopy(part_1)
        part_2.reverse()
        del part_2[0]

        # To do: describe the circuit.
        part_3 = _copy.deepcopy(part_1)
        del part_3[0]

        # To do: describe the circuit.
        part_4 = _copy.deepcopy(part_3)
        del part_4[len(part_3) - 1]
        part_4.reverse()

        # Add the lists of gates together, in order
        cnot_circuit = part_1 + part_2 + part_3 + part_4

        # Convert the operationlist to a circuit.
        circuit = _Circuit(layer_labels=cnot_circuit,
                           line_labels=self.model.state_space_labels.labels[0],
                           editable=True)

        ## Change into the native gates, using the compilation for CNOTs between
        ## connected qubits.
        circuit.change_gate_library(self)
        circuit.done_editing()

        if check:
            # Calculate the symplectic matrix implemented by this circuit, to check the compilation
            # is ok, below.
            sreps = self.model.compute_clifford_symplectic_reps()
            s, p = _symp.symplectic_rep_of_clifford_circuit(circuit, sreps)

            # Construct the symplectic rep of CNOT between this pair of qubits, to compare to s.
            nQ = int(round(_np.log2(self.model.dim)))  # assumes *unitary* mode (OK?)
            iq1 = self.model.state_space_labels.labels[0].index(q1)  # assumes single tensor-prod term
            iq2 = self.model.state_space_labels.labels[0].index(q2)  # assumes single tensor-prod term
            s_cnot, p_cnot = _symp.symplectic_rep_of_clifford_layer(_Label('CNOT', (iq1, iq2)), nQ)

            assert(_np.array_equal(s, s_cnot)), "Compilation has failed!"
            if self.ctype == "absolute":
                assert(_np.array_equal(p, p_cnot)), "Compilation has failed!"

        return circuit

    def get_nonlocal_compilation_of(self, oplabel, force=False,
                                    allowed_filter=None, verbosity=1, check=True):
        """
        Get a potentially non-local compilation of `oplabel`.

        This function does *not* add this compilation to the library, it merely
        returns it. To add it, use :method:`add_nonlocal_compilation_of`.

        This method currently only generates a compilation for a non-local CNOT,
        up to arbitrary Pauli gates, between a pair of unconnected qubits. It
        converts this CNOT into a circuit of CNOT gates between connected qubits,
        using a fixed circuit form. This compilation is not optimal in at least
        some circumstances.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.  Currently, `oplabel.name` must
            equal `"CNOT"`.

        force : bool, optional
            If True, then a compilation is recomputed even if `oplabel`
            already exists in this `CompilationLibrary`.  Otherwise
            compilations are only computed when they are *not* present.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used in this non-local
            compilation.  If a `dict`, keys must be gate names (like
            `"CNOT"`) and values :class:`QubitGraph` objects indicating
            where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in
            the current library that is confined within that set is allowed.
            If None, then all gates within the library are allowed.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        check : bool, optional
            Whether to perform internal consistency checks.

        Returns
        -------
        Circuit
        """
        context_key = None
        if isinstance(allowed_filter, dict):
            context_key = frozenset(allowed_filter.items())
        elif isinstance(allowed_filter, set):
            context_key = frozenset(allowed_filter)

        if context_key is not None:
            key = (oplabel, context_key)
        else:
            key = oplabel

        if not force and key in self:
            return self[oplabel]  # don't re-compute unless we're told to

        circuit = self._create_nonlocal_compilation_of(
            oplabel, allowed_filter=allowed_filter, verbosity=verbosity, check=check)

        return circuit

    def add_nonlocal_compilation_of(self, oplabel, force=False,
                                    allowed_filter=None, verbosity=1, check=True):
        """
        Add a potentially non-local compilation of `oplabel` to this library.

        This method currently only generates a compilation for a non-local CNOT,
        up to arbitrary Pauli gates, between a pair of unconnected qubits. It
        converts this CNOT into a circuit of CNOT gates between connected qubits,
        using a fixed circuit form. This compilation is not optimal in at least
        some circumstances.

        If `allowed_filter` is None then the compilation is recorded under the key `oplabel`.
        Otherwise, the compilation is recorded under the key (`oplabel`,`context_key`) where
        `context_key` is frozenset(`allowed_filter`) when `allowed_filter` is a set, and
        `context_key` is frozenset(`allowed_filter`.items()) when `allowed_filter` is a dict.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.  Currently, `oplabel.name` must
            equal `"CNOT"`.

        force : bool, optional
            If True, then a compilation is recomputed even if `oplabel`
            already exists in this `CompilationLibrary`.  Otherwise
            compilations are only computed when they are *not* present.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used in this non-local
            compilation.  If a `dict`, keys must be gate names (like
            `"CNOT"`) and values :class:`QubitGraph` objects indicating
            where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in
            the current library that is confined within that set is allowed.
            If None, then all gates within the library are allowed.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        check : bool, optional
            Whether to perform internal consistency checks.

        Returns
        -------
        None
        """
        context_key = None
        if isinstance(allowed_filter, dict):
            context_key = frozenset(allowed_filter.items())
        elif isinstance(allowed_filter, set):
            context_key = frozenset(allowed_filter)

        if context_key is not None:
            key = (oplabel, context_key)
        else:
            key = oplabel

        if not force and key in self:
            return
        else:
            circuit = self.get_nonlocal_compilation_of(oplabel, force, allowed_filter,
                                                       verbosity, check)

            self[key] = circuit

    def get_compilation_of(self, oplabel, force=False, allowed_filter=None, verbosity=1, check=True):
        """
        Get a compilation of `oplabel` in the context of `allowed_filter`, if any.

        This is often more convenient than querying the CompilationLibrary directly as a dictionary,
        because:

        1. If allowed_filter is not None, this handles the correct querying of the dictionary
           to find out if there is a previously saved compilation with this `allowed_filter` context.
        2. If a compilation is not present, this method will try to compute one.

        This method does *not* store the compilation. To store the compilation first call the
        method `add_compilation_of()`.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.

        force : bool, optional
            If True, then an attempt is made to recompute a compilation
            even if `oplabel` already exists in this `CompilationLibrary`.
            Otherwise compilations are only computed when they are *not* present.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used in this non-local
            compilation.  If a `dict`, keys must be gate names (like
            `"CNOT"`) and values :class:`QubitGraph` objects indicating
            where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in
            the current library that is confined within that set is allowed.
            If None, then all gates within the library are allowed.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        check : bool, optional
            Whether to perform internal consistency checks.

        Returns
        -------
        Circuit
        """
        # first try and compile the gate locally. Future: this will not work properly if the allowed_filter removes
        # gates that the get_local_compilation_of uses, because it knows nothing of the filter. This inconsistence
        # should be removed somehow.
        try:
            # We don't have to account for `force` manually here, because it is dealt with inside this function
            circuit = self.get_local_compilation_of(
                oplabel, unitary=None, srep=None, max_iterations=10, force=force, verbosity=verbosity)
            # Check for the case where this function won't currently behave as expected.
            if isinstance(allowed_filter, dict):
                raise ValueError("This function may behave incorrectly when the allowed_filer is a dict "
                                 "*and* the gate can be compiled locally!")

        # If local compilation isn't possible, we move on and try non-local compilation
        except:
            circuit = self.get_nonlocal_compilation_of(
                oplabel, force=force, allowed_filter=allowed_filter, verbosity=verbosity, check=check)

        return circuit

    def add_compilation_of(self, oplabel, force=False, allowed_filter=None, verbosity=1, check=True):
        """
        Adds a compilation of `oplabel` in the context of `allowed_filter`, if any.

        If `allowed_filter` is None then the compilation is recorded under the key `oplabel`.
        Otherwise, the compilation is recorded under the key (`oplabel`,`context_key`) where
        `context_key` is frozenset(`allowed_filter`) when `allowed_filter` is a set, and
        `context_key` is frozenset(`allowed_filter`.items()) when `allowed_filter` is a dict.

        Parameters
        ----------
        oplabel : Label
            The label of the gate to compile.

        force : bool, optional
            If True, then an attempt is made to recompute a compilation
            even if `oplabel` already exists in this `CompilationLibrary`.
            Otherwise compilations are only computed when they are *not* present.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used in this non-local
            compilation.  If a `dict`, keys must be gate names (like
            `"CNOT"`) and values :class:`QubitGraph` objects indicating
            where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in
            the current library that is confined within that set is allowed.
            If None, then all gates within the library are allowed.

        verbosity : int, optional
            An integer >= 0 specifying how much detail to send to stdout.

        check : bool, optional
            Whether to perform internal consistency checks.

        Returns
        -------
        None
        """
        # first try and compile the gate locally. Future: this will not work properly if the allowed_filter removes
        # gates that the get_local_compilation_of uses, because it knows nothing of the filter. This inconsistence
        # should be removed somehow.
        try:
            # We don't have to account for `force` manually here, because it is dealt with inside this function
            self.add_local_compilation_of(oplabel, unitary=None, srep=None,
                                          max_iterations=10, force=force, verbosity=verbosity)
            # Check for the case where this function won't currently behave as expected.
            if isinstance(allowed_filter, dict):
                raise ValueError("This function may behave incorrectly when the allowed_filer is a dict "
                                 "*and* the gate can be compiled locally!")

        # If local compilation isn't possible, we move on and try non-local compilation
        except:
            pass

        self.add_nonlocal_compilation_of(
            oplabel, force=force, allowed_filter=allowed_filter, verbosity=verbosity, check=check)

        return
