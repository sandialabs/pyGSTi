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

import collections as _collections
import copy as _copy
import itertools as _itertools

import numpy as _np
from pygsti.baseobjs.label import Label as _Label

from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.tools import listtools as _lt
from pygsti.tools import symplectic as _symp
from pygsti.tools import internalgates as _itgs

IDENT = 'I'  # internal 1Q-identity-gate name used for compilation
# MUST be the same as in processorpack.py


class CompilationError(Exception):
    """
    A compilation error, raised by :class:`CompilationLibrary`
    """
    pass


class CompilationRules(object):

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, CompilationRules): return obj
        return cls(obj)

    def __init__(self, compilation_rules_dict=None):
        self.gate_unitaries = _collections.OrderedDict()  # gate_name => unitary mx, fn, or None
        self.local_templates = _collections.OrderedDict()  # gate_name => Circuit on gate's #qubits
        self.function_templates = _collections.OrderedDict()  # gate_name => fn(sslbls)
        self.specific_compilations = _collections.OrderedDict()  # gate_label => Circuit on absolute qubits
        if compilation_rules_dict is not None:
            raise NotImplementedError("Need to convert compilation_rules_dict into info")

    def add_compilation_rule(self, gate_name, template_circuit_or_fn, unitary=None):
        std_gate_unitaries = _itgs.standard_gatename_unitaries()
        std_gate_unitaries.update(_itgs.internal_gate_unitaries())  # internal gates ok too?
        if unitary is None:
            if gate_name in std_gate_unitaries: unitary = std_gate_unitaries[gate_name]
            else: raise ValueError("Must supply `unitary` for non-standard gate name '%s'" % gate_name)
        self.gate_unitaries[gate_name] = unitary

        if callable(template_circuit_or_fn):
            self.function_templates[gate_name] = template_circuit_or_fn
        else:
            self.local_templates[gate_name] = template_circuit_or_fn

    def add_specific_compilation_rule(self, gate_label, circuit, unitary):
        std_gate_unitaries = _itgs.standard_gatename_unitaries()
        std_gate_unitaries.update(_itgs.internal_gate_unitaries())  # internal gates ok too?
        if gate_label.name not in self.gate_unitaries:
            if unitary is None:
                if gate_label.name in std_gate_unitaries: unitary = std_gate_unitaries[gate_label.name]
                else: raise ValueError("Must supply `unitary` for non-standard gate name '%s'" % gate_label.name)
            self.gate_unitaries[gate_label.name] = unitary
        self.specific_compilations[gate_label] = circuit

    def create_aux_info(self):
        """ TODO: docstring -- compute any aux_info to be added to processorspec when
         compiling with these CompilationRules"""
        return {}


class CliffordCompilationRules(CompilationRules):
    """
    An collection of compilations for clifford gates.

    Holds mapping between operation labels (:class:`Label` objects) and circuits
    (:class:`Circuit` objects).

    TODO: update this docstring part:
    A `CliffordCompilationRules` holds a :class:`Model`
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

    Compilation libraries are most often used within a :class:`QubitProcessorSpec`
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

    @classmethod
    def create_standard(cls, base_processor_spec, compile_type="absolute", what_to_compile=("1Qcliffords",), verbosity=1):
        """ subctype : {"1Qcliffords", "localcnots", "allcnots", "paulis"} -- but depends on `ctype` """

        # A list of the 1-qubit gates to compile, in the std names understood inside the compilation code.
        one_q_gates = []
        # A list of the 2-qubit gates to compile, in the std names understood inside the compilation code.
        two_q_gates = []

        add_nonlocal_two_q_gates = False  # Defaults to not adding non-local compilations of 2-qubit gates.
        number_of_qubits = base_processor_spec.num_qubits
        qubit_labels = base_processor_spec.qubit_labels

        # We construct the requested Pauli-equivalent compilations.
        if compile_type == 'paulieq':
            for subctype in what_to_compile:
                if subctype == '1Qcliffords':
                    one_q_gates += ['H', 'P', 'PH', 'HP', 'HPH']
                elif subctype == 'localcnots':
                    # So that the default still makes sense with 1 qubit, we ignore the request to compile CNOTs
                    # in that case
                    if number_of_qubits > 1:
                        two_q_gates += ['CNOT', ]
                elif subctype == 'allcnots':
                    # So that the default still makes sense with 1 qubit, we ignore the request to compile CNOTs
                    # in that case
                    if number_of_qubits > 1:
                        two_q_gates += ['CNOT', ]
                        add_nonlocal_two_q_gates = True
                else:
                    raise ValueError("{} is invalid for the `{}` compile type!".format(subctype, compile_type))

        # We construct the requested `absolute` (i.e., not only up to Paulis) compilations.
        elif compile_type == 'absolute':
            for subctype in what_to_compile:
                if subctype == 'paulis':
                    one_q_gates += ['I', 'X', 'Y', 'Z']
                elif subctype == '1Qcliffords':
                    one_q_gates += ['C' + str(q) for q in range(24)]
                else:
                    raise ValueError("{} is invalid for the `{}` compile type!".format(subctype, compile_type))
        else:
            raise ValueError("Invalid `compile_type` argument: %s" % str(compile_type))

        descs = {'paulieq': 'up to paulis', 'absolute': ''}

        # Lists that are all the hard-coded 1-qubit and 2-qubit gates.
        # future: should probably import these from _itgss somehow.
        hardcoded_oneQgates = ['I', 'X', 'Y', 'Z', 'H', 'P', 'HP', 'PH', 'HPH'] + ['C' + str(i) for i in range(24)]

        # Currently we can only compile CNOT gates, although that should be fixed.
        for gate in two_q_gates:
            assert (gate == 'CNOT'), ("The only 2-qubit gate auto-generated compilations currently possible "
                                      "are for the CNOT gate (Gcnot)!")

        # Creates an empty library to fill
        compilation_rules = cls(base_processor_spec, compile_type)

        # 1-qubit gate compilations. These must be complied "locally" - i.e., out of native gates which act only
        # on the target qubit of the gate being compiled, and they are stored in the compilation rules.
        for q in qubit_labels:
            for gname in one_q_gates:
                # Check that this is a gate that is defined in the code, so that we can try and compile it.
                assert (gname in hardcoded_oneQgates), "{} is not an allowed hard-coded 1-qubit gate".format(gname)
                if verbosity > 0:
                    print(
                        "- Creating a circuit to implement {} {} on qubit {}...".format(gname, descs[compile_type],
                                                                                        q))
                # This does a brute-force search to compile the gate, by creating `templates` when necessary, and using
                # a template if one has already been constructed.
                compilation_rules.add_local_compilation_of(_Label(gname, q), verbosity=verbosity)
            if verbosity > 0: print("Complete.")

        # Manually add in the "obvious" compilations for CNOT gates as templates, so that we use the normal conversions
        # based on the Hadamard gate -- if this is possible. If we don't do this, we resort to random compilations,
        # which might not give the "expected" compilations (even if the alternatives might be just as good).
        if 'CNOT' in two_q_gates:
            # Look to see if we have a CNOT gate in the model (with any name).
            cnot_name = cls._find_std_gate(base_processor_spec, 'CNOT')
            H_name = cls._find_std_gate(base_processor_spec, 'H')
            I_name = cls._find_std_gate(base_processor_spec, 'I')

            # If we've failed to find a Hadamard gate but we only need paulieq compilation, we try
            # to find a gate that is Pauli-equivalent to Hadamard.
            if H_name is None and compile_type == 'paulieq':
                for gn, gunitary in base_processor_spec.gate_unitaries.items():
                    if callable(gunitary): continue  # can't pre-process factories
                    if _symp.unitary_is_clifford(gunitary):
                        if _itgs.is_gate_pauli_equivalent_to_this_standard_unitary(gunitary, 'H'):
                            H_name = gn; break

            # If CNOT is available, add it as a template for 'CNOT'.
            if cnot_name is not None:
                compilation_rules._clifford_templates['CNOT'] = [(_Label(cnot_name, (0, 1)),)]
                # If Hadamard is also available, add the standard conjugation as template for reversed CNOT.
                if H_name is not None:
                    compilation_rules._clifford_templates['CNOT'].append((_Label(H_name, 0), _Label(H_name, 1), _Label(
                        cnot_name, (1, 0)), _Label(H_name, 0), _Label(H_name, 1)))

            # If CNOT isn't available, look to see if we have CPHASE gate in the model (with any name). If we do *and*
            # we have Hadamards, we add the obvious construction of CNOT from CPHASE and Hadamards as a template
            else:
                cphase_name = cls._find_std_gate(base_processor_spec, 'CPHASE')

                # If we find CPHASE, and we have a Hadamard-like gate, we add used them to add a CNOT compilation
                # template.
                if H_name is not None:
                    if cphase_name is not None:
                        if I_name is not None:
                            # we explicitly put identity gates into template (so any noise on them is simluated correctly)

                            # Add it with CPHASE in both directions, in case the CPHASES have been specified as being
                            # available in only one direction
                            compilation_rules._clifford_templates['CNOT'] = [
                                (_Label(I_name, 0), _Label(H_name, 1), _Label(cphase_name, (0, 1)), _Label(I_name, 0),
                                 _Label(H_name, 1))]
                            compilation_rules._clifford_templates['CNOT'].append(
                                (_Label(I_name, 0), _Label(H_name, 1), _Label(cphase_name, (1, 0)), _Label(I_name, 0),
                                 _Label(H_name, 1)))
                        else:  # similar, but without explicit identity gates
                            compilation_rules._clifford_templates['CNOT'] = [
                                (_Label(H_name, 1), _Label(cphase_name, (0, 1)), _Label(H_name, 1))]
                            compilation_rules._clifford_templates['CNOT'].append(
                                (_Label(H_name, 1), _Label(cphase_name, (1, 0)), _Label(H_name, 1)))
                            

        # After adding default templates, we know generate compilations for CNOTs between all connected pairs. If the
        # default templates were not relevant or aren't relevant for some qubits, this will generate new templates by
        # brute force.
        for gate in two_q_gates:
            not_locally_compilable = []
            for q1 in base_processor_spec.qubit_labels:
                for q2 in base_processor_spec.qubit_labels:
                    if q1 == q2: continue  # 2Q gates must be on different qubits!
                    for gname in two_q_gates:
                        if verbosity > 0:
                            print("Creating a circuit to implement {} {} on qubits {}...".format(
                                gname, descs[compile_type], (q1, q2)))
                        try:
                            compilation_rules.add_local_compilation_of(
                                _Label(gname, (q1, q2)), verbosity=verbosity)
                        except CompilationError:
                            not_locally_compilable.append((gname, q1, q2))

            # If requested, try to compile remaining 2Q gates that are `non-local` (not between neighbouring qubits)
            # using specific algorithms.
            if add_nonlocal_two_q_gates:
                for gname, q1, q2 in not_locally_compilable:
                    compilation_rules.add_nonlocal_compilation_of(_Label(gname, (q1, q2)),
                                                        verbosity=verbosity)

        return compilation_rules

    @classmethod
    def _find_std_gate(cls, base_processor_spec, std_gate_name):
        for gn in base_processor_spec.gate_names:
            if callable(base_processor_spec.gate_unitaries[gn]): continue  # can't pre-process factories
            if _itgs.is_gate_this_standard_unitary(base_processor_spec.gate_unitaries[gn], std_gate_name):
                return gn
        return None

    def __init__(self, native_gates_processorspec, compile_type="absolute"):
        """
        TODO: update docstring
        Create a new CompilationLibrary.

        Parameters
        ----------
        clifford_model : Model
            The model of "native" Clifford gates which all compilations in
            this library are composed from.

        compile_type : {"absolute","paulieq"}
            The "compilation type" for this library.  If `"absolute"`, then
            compilations must match the gate operation being compiled exactly.
            If `"paulieq"`, then compilations only need to match the desired
            gate operation up to a Paui operation (which is useful for compiling
            multi-qubit Clifford gates / stabilizer states without unneeded 1-qubit
            gate over-heads).
        """

        # processor_spec: holds all native Clifford gates (requested gates compile into circuits of these)
        self.processor_spec = native_gates_processorspec

        self.compile_type = compile_type  # "absolute" or "paulieq"
        self._clifford_templates = _collections.defaultdict(list)  # keys=gate names (strs); vals=tuples of Labels
        self.connectivity = {}  # QubitGraphs for gates currently compiled in library (key=gate_name)

        super(CliffordCompilationRules, self).__init__()

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
        #  (i.e. its key in self._clifford_templates)
        def to_real_label(template_label):
            """ Convert a "template" operation label (which uses integer qubit labels
                0 to N) to a "real" label for a potential gate in self.processor_spec. """
            qlabels = [oplabel.sslbls[i] for i in template_label.sslbls]
            return _Label(template_label.name, qlabels)

        def to_template_label(real_label):
            """ The reverse (qubits in template == oplabel.qubits) """
            qlabels = [oplabel.sslbls.index(lbl) for lbl in real_label.sslbls]
            return _Label(real_label.name, qlabels)

        def is_local_compilation_feasible(allowed_gatenames):
            """ Whether template_labels can possibly be enough
                gates to compile a template for op_label with """
            if oplabel.num_qubits <= 1:
                return len(allowed_gatenames) > 0  # 1Q gates, anything is ok
            elif oplabel.num_qubits == 2:
                # 2Q gates need a compilation gate that is also 2Q (can't do with just 1Q gates!)
                return max([self.processor_spec.gate_num_qubits(gn) for gn in allowed_gatenames]) == 2
            else:
                # >2Q gates need to make sure there's some connected path
                return True  # future: update using graphs stuff?

        template_to_use = None

        for template_compilation in self._clifford_templates.get(oplabel.name, []):
            #Check availability of gates in self.model to determine
            # whether template_compilation can be applied.
            if all([self.processor_spec.is_available(gl) for gl in map(to_real_label, template_compilation)]):
                template_to_use = template_compilation
                if verbosity > 0: print("Existing template found!")
                break  # compilation found!

        else:  # no existing templates can be applied, so make a new one

            #construct a list of the available gates on the qubits of `oplabel` (or a subset of them)
            available_gatenames = self.processor_spec.available_gatenames(oplabel.sslbls)
            available_srep_dict = self.processor_spec.compute_clifford_symplectic_reps(available_gatenames)
            #available_srep_dict[IDENT] = _symp.unitary_to_symplectic(_np.identity(2, 'd'))  # REMOVE
            #Manually add 1Q idle gate, as this typically isn't stored in the processor spec.

            if is_local_compilation_feasible(available_gatenames):
                available_gatelabels = [to_template_label(gl) for gn in available_gatenames
                                        for gl in self.processor_spec.available_gatelabels(gn, oplabel.sslbls)]
                template_to_use = self.add_clifford_compilation_template(
                    oplabel.name, oplabel.num_qubits, unitary, srep,
                    available_gatelabels, available_srep_dict,
                    verbosity=verbosity, max_iterations=max_iterations)

        #If a template has been found, use it.
        if template_to_use is not None:
            opstr = list(map(to_real_label, template_to_use))
            #REMOVE 'I's ?
            return _Circuit(layer_labels=opstr, line_labels=self.processor_spec.qubit_labels)
        else:
            raise CompilationError("Cannot locally compile %s" % str(oplabel))

    def _get_local_compilation_of(self, oplabel, unitary=None, srep=None, max_iterations=10, force=False, verbosity=1):
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
        if not force and oplabel in self.specific_compilations:
            return self.specific_compilations[oplabel]  # don't re-compute unless we're told to

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
        circuit = self._get_local_compilation_of(oplabel, unitary, srep, max_iterations, force, verbosity)
        self.add_specific_compilation_rule(oplabel, circuit, unitary)

    def add_clifford_compilation_template(self, gate_name, nqubits, unitary, srep,
                                          available_gatelabels, available_sreps,
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
            if self.compile_type == 'absolute':
                print("- Generating a template for a compilation of {}...".format(gate_name), end='\n')
            elif self.compile_type == 'paulieq':
                print("- Generating a template for a pauli-equivalent compilation of {}...".format(gate_name), end='\n')

        obtained_sreps = {}

        #Separate the available operation labels by their target qubits
        available_glabels_by_qubit = _collections.defaultdict(list)
        for gl in available_gatelabels:
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

        # find the 1Q identity gate name 
        I_name = self._find_std_gate(self.processor_spec, 'I')

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
                    if self.compile_type == 'paulieq' or \
                       (self.compile_type == 'absolute' and _np.array_equal(svector, p)):
                        candidates.append(seq)
                        found = True

            # If there is more than one way to compile gate at this circuit length, pick the
            # one containing the most idle gates.
            if len(candidates) > 1:

                # Look at each sequence, and see if it has more than or equal to max_number_of_idles.
                # If so, set it to the current chosen sequence.
                if I_name is not None:
                    number_of_idles = 0
                    max_number_of_idles = 0

                    for seq in candidates:
                        number_of_idles = len([x for x in seq if x.name == I_name])

                        if number_of_idles >= max_number_of_idles:
                            max_number_of_idles = number_of_idles
                            compilation = seq
                else:
                    # idles are absent from circuits - just take one with smallest depth
                    min_depth = 1e100
                    for seq in candidates:
                        depth = len(seq)
                        if depth < min_depth:
                            min_depth = depth
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

        #Compilation done: remove identity labels, as these are just used to
        # explicitly keep track of the number of identity gates in a circuit (really needed?)
        compilation = list(filter(lambda gl: gl.name != I_name, compilation))

        #Store & return template that was found
        self._clifford_templates[gate_name].append(compilation)

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
        nQ = self.processor_spec.num_qubits
        qubit_labels = self.processor_spec.qubit_labels
        d = {qlbl: i for i, qlbl in enumerate(qubit_labels)}
        assert(len(qubit_labels) == nQ), "Number of qubit labels is inconsistent with Model dimension!"

        connectivity = _np.zeros((nQ, nQ), dtype=bool)
        for compiled_gatelabel in self.specific_compilations.keys():
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

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be to construct this
            connectivity.  If a `dict`, keys must be gate names (like
            `"CNOT"`) and values :class:`QubitGraph` objects indicating
            where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in
            the current library that is confined within that set is allowed.
            If None, then all gates within the library are allowed.

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
                init_nodes = set(init_qgraph.node_names)
                qlabels = [lbl for lbl in graph_constraint.node_names
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
        assert(oplabel.num_qubits > 1), "1-qubit gates can't be non-local!"
        assert(oplabel.name == "CNOT" and oplabel.num_qubits == 2), \
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
        line_labels = self.processor_spec.qubit_labels
        circuit = _Circuit(layer_labels=cnot_circuit,
                           line_labels=line_labels,
                           editable=True)

        ## Change into the native gates, using the compilation for CNOTs between
        ## connected qubits.
        circuit.change_gate_library(self)
        circuit.done_editing()

        if check:
            # Calculate the symplectic matrix implemented by this circuit, to check the compilation
            # is ok, below.
            sreps = self.processor_spec.compute_clifford_symplectic_reps()
            s, p = _symp.symplectic_rep_of_clifford_circuit(circuit, sreps)

            # Construct the symplectic rep of CNOT between this pair of qubits, to compare to s.
            nQ = self.processor_spec.num_qubits
            iq1 = line_labels.index(q1)  # assumes single tensor-prod term
            iq2 = line_labels.index(q2)  # assumes single tensor-prod term
            s_cnot, p_cnot = _symp.symplectic_rep_of_clifford_layer(_Label('CNOT', (iq1, iq2)), nQ)

            assert(_np.array_equal(s, s_cnot)), "Compilation has failed!"
            if self.compile_type == "absolute":
                assert(_np.array_equal(p, p_cnot)), "Compilation has failed!"

        return circuit

    def _get_nonlocal_compilation_of(self, oplabel, force=False,
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

        if not force and key in self.specific_compilations:
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

        if not force and key in self.specific_compilations:
            return
        else:
            circuit = self._get_nonlocal_compilation_of(oplabel, force, allowed_filter,
                                                        verbosity, check)

            self.add_specific_compilation_rule(key, circuit, unitary=None)  # Need to take unitary as arg?

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
            circuit = self._get_local_compilation_of(
                oplabel, unitary=None, srep=None, max_iterations=10, force=force, verbosity=verbosity)
            # Check for the case where this function won't currently behave as expected.
            if isinstance(allowed_filter, dict):
                raise ValueError("This function may behave incorrectly when the allowed_filer is a dict "
                                 "*and* the gate can be compiled locally!")

        # If local compilation isn't possible, we move on and try non-local compilation
        except:
            circuit = self._get_nonlocal_compilation_of(
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
