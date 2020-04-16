""" Defines the ProcessorSpec class and supporting functionality."""
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
import warnings as _warnings
import itertools as _iter

from .localnoisemodel import LocalNoiseModel as _LocalNoiseModel
from .compilationlibrary import CompilationLibrary as _CompilationLibrary
from .compilationlibrary import CompilationError as _CompilationError
from .qubitgraph import QubitGraph as _QubitGraph
from .label import Label as _Label
from ..tools import optools as _gt
from ..tools import internalgates as _itgs
from ..tools import symplectic as _symp

IDENT = 'I'  # internal 1Q-identity-gate name used for compilation
# MUST be the same as in compilationlibrary.py


class ProcessorSpec(object):
    """
    An object that can be used to encapsulate the device specification for a one or more qubit
    quantum computer. This is objected is geared towards multi-qubit devices; many of the contained
    structures are superfluous in the case of a single qubit.

    """

    def __init__(self, n_qubits, gate_names, nonstd_gate_unitaries={}, availability={},
                 qubit_labels=None, construct_models=('clifford', 'target'),
                 construct_clifford_compilations={'paulieq': ('1Qcliffords', 'allcnots'),
                                                  'absolute': ('paulis', '1Qcliffords')}, verbosity=0):
        """
        Initializes a ProcessorSpec object.

        The most basic information required for a ProcessorSpec object is the number of qubits in the
        device, and the library of "native" target gates, which are specified as unitary matrices
        (using `nonstd_gate_unitaries`) or using default gate-names known to pyGSTi, such as 'Gcnot'
        for the CNOT gate (using `gate_names`). The other core information is the availability of the
        gates (specified via `availability`).

        Parameters
        ----------
        n_qubits : int
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

        construct_models : tuple, optional
            Standard model for the gates to add.
                - If 'target' is in the tuple, "target" process matrices corresponding to ideal gates are added.
                - If 'clifford' is in the tuple, the Clifford gates in the model are represented in their efficient-in-n
                symplectic form (these are reps. of perfect gates).

        construct_clifford_compilations : dict, optional
            The compilations for "standard" Clifford gates that are constructed. These are mostly only of importance for
            compiling many-qubit Clifford gates, and similar tasks related to creating randomized benchmarking circuits.
            The standard option is exhaustive (i.e., there are no further allowed options), and leaving this as-is
            should be fine for most purposes.

        verbosity : int, optional
            If > 0, information about the generation of the ProcessorSpec is printed to the screen.

        Returns
        -------
        ProcessorSpec
        """
        assert(type(n_qubits) is int), "The number of qubits, n, should be an integer!"

        #Store inputs for adding models later
        self.number_of_qubits = n_qubits
        self.root_gate_names = gate_names[:]  # copy this list
        self.nonstd_gate_unitaries = nonstd_gate_unitaries.copy()
        #self.root_gate_names += list(self.nonstd_gate_unitaries.keys())  # must specify all names in `gate_names`
        self.availability = availability.copy()

        # Stores the basic unitary matrices defining the gates, as it is convenient to have these easily accessable.
        self.root_gate_unitaries = _collections.OrderedDict()
        std_gate_unitaries = _itgs.get_standard_gatename_unitaries()
        for gname in gate_names:
            if gname in nonstd_gate_unitaries:
                self.root_gate_unitaries[gname] = nonstd_gate_unitaries[gname]
            elif gname in std_gate_unitaries:
                self.root_gate_unitaries[gname] = std_gate_unitaries[gname]
            else:
                raise ValueError(
                    str(gname) + " is not a valid 'standard' gate name, it must be given in `nonstd_gate_unitaries`")

        # If no qubit labels are provided it defaults to integers from 0 to n_qubits-1.
        if qubit_labels is None:
            self.qubit_labels = list(range(n_qubits))
        else:
            assert(len(qubit_labels) == n_qubits)
            self.qubit_labels = list(qubit_labels)

        # A dictionary of models for the device (e.g., imperfect unitaries, process matrices etc).
        self.models = _collections.OrderedDict()

        # Compilations are stored here.
        self.compilations = _collections.OrderedDict()

        # The connectivity graph of the device, for the "clifford" model (future: perhaps this should not only be for
        # Clifford gates, or there should be a qubitgraph for each model.)
        self.qubitgraph = None

        # Holds a dictionary with keys that are 1Q gatename pairs (gn1, gn2), with a value gn3 that is the 1Q that
        # these gates combine to when gn1 is applied first and then gn2. There is no key for a pair if they don't
        # combine to a 1Q gate in the model.
        self.oneQgate_relations = {}
        # A dict from a gatename to the gatename of the inverse gate, if it is in the model.
        self.gate_inverse = {}

        # Add initial models
        for model_name in construct_models:
            self.add_std_model(model_name)

        # If we construct a Clifford model, we do add various things that are Clifford-specific.
        if 'clifford' in construct_models:
            # Add compilations for the "basic" Clifford gates specified, which are used for, e.g., Clifford compiler
            # algorithms.  We only do this if there is a 'clifford' model, as these are Clifford compilations.
            for ctype in list(construct_clifford_compilations.keys()):

                # We construct the requested Pauli-equivalent compilations.
                if ctype == 'paulieq':
                    # A list of the 1-qubit gates to compile, in the std names understood inside the compilers.
                    oneQgates = []
                    # A list of the 2-qubit gates to compile, in the std names understood inside the compilers.
                    twoQgates = []
                    add_nonlocal_twoQgates = False  # Defaults to not adding non-local compilations of 2-qubit gates.
                    for subctype in construct_clifford_compilations[ctype]:
                        if subctype == '1Qcliffords':
                            oneQgates += ['H', 'P', 'PH', 'HP', 'HPH']
                        elif subctype == 'localcnots':
                            # So that the default still makes sense with 1 qubit, we ignore the request to compile CNOTs
                            # in that case
                            if self.number_of_qubits > 1:
                                twoQgates += ['CNOT', ]
                        elif subctype == 'allcnots':
                            # So that the default still makes sense with 1 qubit, we ignore the request to compile CNOTs
                            # in that case
                            if self.number_of_qubits > 1:
                                twoQgates += ['CNOT', ]
                                add_nonlocal_twoQgates = True
                        else:
                            raise ValueError("One of the values for the key `{}` to "
                                             "`construct_clifford_compilations` is not a valid option!".format(ctype))
                    self.add_std_compilations(ctype, oneQgates, twoQgates, add_nonlocal_twoQgates, verbosity)

                # We construct the requested `absolute` (i.e., not only up to Paulis) compilations.
                elif ctype == 'absolute':
                    # A list of the 1-qubit gates to compile, in the std names understood inside the compilers.
                    oneQgates = []
                    # A list of the 2-qubit gates to compile, in the std names understood inside the compilers.
                    twoQgates = []
                    for subctype in construct_clifford_compilations[ctype]:
                        if subctype == 'paulis':
                            oneQgates += ['I', 'X', 'Y', 'Z']
                        elif subctype == '1Qcliffords':
                            oneQgates += ['C' + str(q) for q in range(24)]
                        else:
                            raise ValueError("One of the values for the key `{}` to "
                                             "`construct_clifford_compilations` is not a valid option!".format(ctype))
                    self.add_std_compilations(ctype, oneQgates, twoQgates, verbosity)

                else:
                    raise ValueError(
                        "The keys to `construct_clifford_compilations` can only be `paulieq` and `absolute!")

            # Generates the QubitGraph for the multi-qubit Clifford gates. If there are multi-qubit gates which are not
            # Clifford gates then these are not counted as "connections".
            connectivity = _np.zeros((self.number_of_qubits, self.number_of_qubits), dtype=bool)
            for oplabel in self.models['clifford'].get_primitive_op_labels():
                # This treats non-entangling 2-qubit gates as making qubits connected. Stopping that is
                # something we may need to do at some point.
                if oplabel.number_of_qubits is None: continue  # skip "global" gates in connectivity consideration?
                if oplabel.number_of_qubits > 1:
                    for p in _itertools.permutations(oplabel.qubits, 2):
                        connectivity[self.qubit_labels.index(p[0]), self.qubit_labels.index(p[1])] = True

            self.qubitgraph = _QubitGraph(self.qubit_labels, connectivity)

        #future : store this in a less clumsy way.
        if 'clifford' in self.models:
            # Compute the operation labels that act on an entire set of qubits
            self.clifford_ops_on_qubits = _collections.defaultdict(list)
            for gl in self.models['clifford'].get_primitive_op_labels():
                if gl.qubits is None: continue  # skip "global" gates (?)
                for p in _itertools.permutations(gl.qubits):
                    self.clifford_ops_on_qubits[p].append(gl)
        else:
            self.clifford_ops_on_qubits = None

        # Adds in the 1-qubit gate algebra
        self.add_one_q_gate_relations()
        # Records the inverses of gates that have an inverse in the model (only compares to gates of the same dimension)
        self.add_multiqubit_inversion_relations()

        return  # done with __init__(...)

    def get_edges(self, qubits):
        # todo
        edgelist = []

        for oplabel in self.models['clifford'].get_primitive_op_labels():
            # This treats non-entangling 2-qubit gates as making qubits connected. Stopping that is
            # something we may need to do at some point.
            if oplabel.number_of_qubits == 2:
                if oplabel.qubits[0] in qubits and oplabel.qubits[1] in qubits:
                    edgelist.append(oplabel.qubits)

        return edgelist

    def get_std_model(self, model_name, parameterization='auto', sim_type='auto'):
        # Erik future : improve docstring.
        """
        Builds a standard model for the gates. For example, "target" process matrices are added
        if model_name = 'target_name';  Target Clifford gates, represented in their efficient-in-n
        symplectic form, are added if model_name = 'clifford'.

        Returns
        -------
        Model
        """
        if model_name == 'clifford':
            assert(parameterization in ('auto', 'clifford')), "Clifford model must use 'clifford' parameterizations"
            assert(sim_type in ('auto', 'map')), "Clifford model must use 'map' simulation type"
            model = _LocalNoiseModel.build_from_parameterization(
                self.number_of_qubits,
                self.root_gate_names,
                self.nonstd_gate_unitaries, None,
                self.availability,
                self.qubit_labels,
                parameterization='clifford',
                sim_type=sim_type,
                on_construction_error='warn',  # *drop* gates that aren't cliffords
                independent_gates=False,
                ensure_composed_gates=False)  # change these? add `geometry`?

        elif model_name in ('target', 'Target', 'static', 'TP', 'full'):
            param = model_name if (parameterization == 'auto') \
                else parameterization
            if param in ('target', 'Target'): param = 'static'  # special case for 'target' model

            model = _LocalNoiseModel.build_from_parameterization(
                self.number_of_qubits, self.root_gate_names,
                self.nonstd_gate_unitaries, None, self.availability,
                self.qubit_labels, parameterization=param, sim_type=sim_type,
                independent_gates=False, ensure_composed_gates=False)  # change these? add `geometry`?

        else:  # unknown model name, so require parameterization
            if parameterization == 'auto':
                raise ValueError(
                    "Non-std model name '%s' means you must specify `parameterization` argument!" % model_name)
            model = _LocalNoiseModel.build_from_parameterization(
                self.number_of_qubits, self.root_gate_names,
                self.nonstd_gate_unitaries, None, self.availability,
                self.qubit_labels, parameterization=parameterization, sim_type=sim_type,
                independent_gates=False, ensure_composed_gates=False)  # change these? add `geometry`?

        return model

    def add_std_model(self, model_name, parameterization='auto', sim_type='auto'):
        # Erik future : improve docstring.
        """
        Adds a standard model for the gates. For example, "target" process matrices are added
        if model_name = 'target_name';  Target Clifford gates, represented in their efficient-in-n
        symplectic form, are added if model_name = 'clifford'.
        """
        self.models[model_name] = self.get_std_model(model_name, parameterization, sim_type)

    def add_std_compilations(self, compile_type, one_q_gates, two_q_gates, add_nonlocal_two_q_gates=False, verbosity=0):
        """
        Constructions compilations for the requested standard gates, and stores them in
        a CompilationLibrary object in the dictionary self.compilations. The key for
        is given by `compile_type`.

        Parameters
        ----------
        compile_type : str
            Either 'absolute' or 'paulieq'. If 'absolute' then circuits that exactly generate the
            requested gates are constructed. If 'paulieq' then the circuits are only guaranteed to
            generate the requested gates up to multiplication by Paulis. This latter option is useful
            for creating the building-blocks for efficient multi-qubit Clifford gate generation.

        one_q_gates : list
            The "standard" 1-qubit gates to generate compilations for, as strings that correspond to
            built-in fixed (perfect) gates. The allowed strings in this list are:

            - 'I', 'X', 'Y', 'Z' : the Pauli operators
            - 'H', 'P', 'HP', 'PH', 'HPH' : gates constructed for the Hadmard and phase gate.
            - 'C0', 'C1', 'C2', ...., 'C23' : the 1-qubit Clifford gates

             Note that there is some overlap between these gates: there are multiple internal labels
             for some of the gates, used in different contexts.

        two_q_gates : list
            The "standard" 2-qubit gates to generate compilations for, as strings that correspond to
            built-in fixed (perfect) gates. The allowed strings in this list are:

            - 'CNOT' : the CNOT gate.

        add_nonlocal_two_q_gates : bool, optional
            Whether to add compilations for CNOT gates between non-neighbouring qubits.

        verbosity : int, optional
            If > 0, information about the compilation generation is printed to screen.

        Returns
        -------
        None
        """
        # For printing to screen what the compiler is doing.
        descs = {'paulieq': 'up to paulis', 'absolute': ''}
        # Lists that are all the hard-coded 1-qubit and 2-qubit gates.
        # future: should probably import these from _itgss somehow.
        hardcoded_oneQgates = ['I', 'X', 'Y', 'Z', 'H', 'P', 'HP', 'PH', 'HPH'] + ['C' + str(i) for i in range(24)]

        # Currently we can only compile CNOT gates, although that should be fixed.
        for gate in two_q_gates:
            assert(gate == 'CNOT'), ("The only 2-qubit gate auto-generated compilations currently possible "
                                     "are for the CNOT gate (Gcnot)!")

        if 'clifford' not in self.models:
            raise ValueError("Cannot create standard compilations without a 'clifford' model!")
        # Creates an empty library to fill
        library = _CompilationLibrary(self.models['clifford'], compile_type)

        # 1-qubit gate compilations. These must be complied "locally" - i.e., out of native gates which act only
        # on the target qubit of the gate being compiled, and they are stored in the compilation library.
        for q in self.qubit_labels:
            for gname in one_q_gates:
                # Check that this is a gate that is defined in the code, so that we can try and compile it.
                assert(gname in hardcoded_oneQgates), "{} is not an allowed hard-coded 1-qubit gate".format(gname)
                if verbosity > 0:
                    print("- Creating a circuit to implement {} {} on qubit {}...".format(gname,
                                                                                          descs[compile_type], q))
                # This does a brute-force search to compile the gate, by creating `templates` when necessary, and using
                # a template if one has already been constructed.
                library.add_local_compilation_of(_Label(gname, q), verbosity=verbosity)
            if verbosity > 0: print("Complete.")

        # Manually add in the "obvious" compilations for CNOT gates as templates, so that we use the normal conversions
        # based on the Hadamard gate -- if this is possible. If we don't do this, we resort to random compilations,
        # which might not give the "expected" compilations (even if the alternatives might be just as good).
        if 'CNOT' in two_q_gates:
            # Look to see if we have a CNOT gate in the model (with any name).
            cnot_name = None
            for gn in self.root_gate_names:
                if callable(self.root_gate_unitaries[gn]): continue  # can't pre-process factories
                if _itgs.is_gate_this_standard_unitary(self.root_gate_unitaries[gn], 'CNOT'):
                    cnot_name = gn
                    break

            H_name = None
            for gn in self.root_gate_names:
                if callable(self.root_gate_unitaries[gn]): continue  # can't pre-process factories
                if _itgs.is_gate_this_standard_unitary(self.root_gate_unitaries[gn], 'H'):
                    H_name = gn
                    break

            # If we've failed to find a Hadamard gate, we try but we only need paulieq compilation, we try
            # to find a gate that is Pauli-equivalent to Hadamard.
            if H_name is None and compile_type == 'paulieq':
                for gn in self.root_gate_names:
                    if callable(self.root_gate_unitaries[gn]): continue  # can't pre-process factories
                    if _symp.unitary_is_a_clifford(self.root_gate_unitaries[gn]):
                        if _itgs.is_gate_pauli_equivalent_to_this_standard_unitary(self.root_gate_unitaries[gn], 'H'):
                            H_name = gn

            # If CNOT is available, add it as a template for 'CNOT'.
            if cnot_name is not None:
                library.templates['CNOT'] = [(_Label(cnot_name, (0, 1)),)]
                # If Hadamard is also available, add the standard conjugation as template for reversed CNOT.
                if H_name is not None:
                    library.templates['CNOT'].append((_Label(H_name, 0), _Label(H_name, 1), _Label(
                        cnot_name, (1, 0)), _Label(H_name, 0), _Label(H_name, 1)))

            # If CNOT isn't available, look to see if we have CPHASE gate in the model (with any name). If we do *and*
            # we have Hadamards, we add the obvious construction of CNOT from CPHASE and Hadamards as a template
            else:
                cphase_name = None
                for gn in self.root_gate_names:
                    if callable(self.root_gate_unitaries[gn]): continue  # can't pre-process factories
                    if _itgs.is_gate_this_standard_unitary(self.root_gate_unitaries[gn], 'CPHASE'):
                        cphase_name = gn
                        break

                # If we find CPHASE, and we have a Hadamard-like gate, we add used them to add a CNOT compilation
                # template.
                if H_name is not None:
                    if cphase_name is not None:
                        # Note: we need the identity gate for these templates, which we put explicitly
                        # in template layers and which all models should be able to deal with (as the
                        # lack of labels in a layer).

                        # Add it with CPHASE in both directions, in case the CPHASES have been specified as being
                        # available in only one direction
                        library.templates['CNOT'] = [(_Label(IDENT, 0), _Label(H_name, 1), _Label(
                            cphase_name, (0, 1)), _Label(IDENT, 0), _Label(H_name, 1))]
                        library.templates['CNOT'].append((_Label(IDENT, 0), _Label(H_name, 1), _Label(
                            cphase_name, (1, 0)), _Label(IDENT, 0), _Label(H_name, 1)))

        # After adding default templates, we know generate compilations for CNOTs between all connected pairs. If the
        # default templates were not relevant or aren't relevant for some qubits, this will generate new templates by
        # brute force.
        for gate in two_q_gates:
            not_locally_compilable = []
            for q1 in self.qubit_labels:
                for q2 in self.qubit_labels:
                    if q1 == q2: continue  # 2Q gates must be on different qubits!
                    for gname in two_q_gates:
                        if verbosity > 0:
                            print("Creating a circuit to implement {} {} on qubits {}...".format(
                                gname, descs[compile_type], (q1, q2)))
                        try:
                            library.add_local_compilation_of(
                                _Label(gname, (q1, q2)), verbosity=verbosity)
                        except _CompilationError:
                            not_locally_compilable.append((gname, q1, q2))

            # If requested, try to compile remaining 2Q gates that are `non-local` (not between neighbouring qubits)
            # using specific algorithms.
            if add_nonlocal_two_q_gates:
                for gname, q1, q2 in not_locally_compilable:
                    library.add_nonlocal_compilation_of(_Label(gname, (q1, q2)),
                                                        verbosity=verbosity)

        self.compilations[compile_type] = library

    def add_one_q_gate_relations(self):
        """
        Records the basic pair-wise relationships relationships between the gates.

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
        None
        """
        Id = _np.identity(4, float)
        nontrivial_gname_pauligate_pairs = []
        for gname in self.root_gate_names:
            if callable(self.root_gate_unitaries[gname]): continue  # can't pre-process factories

            # We convert to process matrices, to avoid global phase problems.
            u = _gt.unitary_to_pauligate(self.root_gate_unitaries[gname])
            if u.shape == (4, 4):
                #assert(not _np.allclose(u,Id)), "Identity should *not* be included in root gate names!"
                #if _np.allclose(u, Id):
                #    _warnings.warn("The identity should often *not* be included "
                #                   "in the root gate names of a ProcessorSpec.")
                nontrivial_gname_pauligate_pairs.append((gname, u))

        for gname1, u1 in nontrivial_gname_pauligate_pairs:
            for gname2, u2 in nontrivial_gname_pauligate_pairs:
                ucombined = _np.dot(u2, u1)
                for gname3, u3 in nontrivial_gname_pauligate_pairs:
                    if _np.allclose(u3, ucombined):
                        # If ucombined is u3, add the gate composition relation.
                        self.oneQgate_relations[gname1, gname2] = gname3  # != Id (asserted above)
                    if _np.allclose(ucombined, Id):
                        # If ucombined is the identity, add the inversion relation.
                        self.gate_inverse[gname1] = gname2
                        self.gate_inverse[gname2] = gname1
                        self.oneQgate_relations[gname1, gname2] = None
                        # special 1Q gate relation where result is the identity (~no gates)

    def add_multiqubit_inversion_relations(self):
        """
        Finds whether any of the multi-qubit gates in this device also have their
        inverse in the model, and if so stores this is in the dictionary
        self.gate_inverse. That is, if the unitaries for the  multi-qubit gate with
        name `name1` followed by the multi-qubit gate (of the same dimension) with
        name `name2` multiple (up to phase) to the identity, then
        self.gate_inverse[`name1`] = `name2` and self.gate_inverse[`name2`] = `name1`

        1-qubit gates are not added by this method, as they can be added by the method
        add_one_q_gate_relations().

        Returns
        -------
        None
        """
        for gname1 in self.root_gate_names:
            if callable(self.root_gate_unitaries[gname1]): continue  # can't pre-process factories

            # We convert to process matrices, to avoid global phase problems.
            u1 = _gt.unitary_to_pauligate(self.root_gate_unitaries[gname1])
            if _np.shape(u1) != (4, 4):
                for gname2 in self.root_gate_names:
                    if callable(self.root_gate_unitaries[gname2]): continue  # can't pre-process factories
                    u2 = _gt.unitary_to_pauligate(self.root_gate_unitaries[gname2])
                    if _np.shape(u2) == _np.shape(u1):
                        ucombined = _np.dot(u2, u1)
                        if _np.allclose(ucombined, _np.identity(_np.shape(u2)[0], float)):
                            self.gate_inverse[gname1] = gname2
                            self.gate_inverse[gname2] = gname1

    def get_all_connected_sets(self, n):
        """
        Returns all connected sets of `n` qubits. Note that for a large device with
        this will be often be an unreasonably large number of sets of qubits, and so
        the run-time of this method will be unreasonable.

        Parameters
        ----------
        n: int
            The number of qubits within each set.

        Returns
        -------
        list
            All sets of `n` connected qubits.

        """
        connectedqubits = []
        for combo in _iter.combinations(self.qubit_labels, n):
            if self.qubitgraph.subgraph(list(combo)).are_glob_connected(combo):
                connectedqubits.append(combo)

        return connectedqubits

    #Note:  Below method gets all subgraphs up to full graph size.
    def get_all_connected_sets_new(self):
        """
        todo
        """
        def fn(neighbor_dict, k_max, x, y, output_set):
            vertices = neighbor_dict.keys()
            if len(x) == k_max:
                return output_set
            if x:
                T = set(a for x in x for a in neighbor_dict[x] if a not in y and a not in x)
            else:
                T = vertices
            Y1 = set(y)
            for v in T:
                x.add(v)
#                print (X)
                output_set.add(frozenset(x))
                fn(neighbor_dict, k_max, x, Y1, output_set)
                x.remove(v)
                Y1.add(v)

        def addedge(a, b, neighbor_dict):
            neighbor_dict[a].append(b)
            neighbor_dict[b].append(a)

        def group_subgraphs(subgraph_list):
            processed_subgraph_dict = _collections.defaultdict(list)
            for subgraph in subgraph_list:
                k = len(subgraph)
                subgraph_as_list = list(subgraph)
                subgraph_as_list.sort()
                subgraph_as_tuple = tuple(subgraph_as_list)
                processed_subgraph_dict[k].append(subgraph_as_tuple)
            return processed_subgraph_dict

        neighbor_dict = _collections.defaultdict(list)
        directed_edge_list = self.qubitgraph.edges()
        undirected_edge_list = list(set([frozenset(edge) for edge in directed_edge_list]))
        undirected_edge_list = [list(edge) for edge in undirected_edge_list]

        for edge in undirected_edge_list:
            addedge(edge[0], edge[1], neighbor_dict)
        k_max = self.number_of_qubits
        output_set = set()
        fn(neighbor_dict, k_max, set(), set(), output_set)
        grouped_subgraphs = group_subgraphs(output_set)
        return grouped_subgraphs

    # Future : replace this with a way to specify how "costly" using different qubits/gates is estimated to be, so that
    # Clifford compilers etc can take this into account by auto-generating a costfunction from this information.
    # def construct_compiler_costs(self, custom_connectivity=None):
    #     """

    #     """
    #     self.qubitcosts = {}
    #     distances = self.qubitgraph.shortest_path_distance_matrix()
    #     for i in range(0,self.number_of_qubits):
    #         self.qubitcosts[i] = _np.sum(distances[i,:])

    #     temp_distances = list(_np.sum(distances,0))
    #     temp_qubits = list(_np.arange(0,self.number_of_qubits))
    #     self.costorderedqubits = []

    #     while len(temp_distances) > 0:

    #         longest_remaining_distance = max(temp_distances)
    #         qubits_at_this_distance = []

    #         while longest_remaining_distance == max(temp_distances):

    #             index = _np.argmax(temp_distances)
    #             qubits_at_this_distance.append(temp_qubits[index])
    #             del temp_distances[index]
    #             del temp_qubits[index]

    #             if len(temp_distances) == 0:
    #                 break

    #         self.costorderedqubits.append(qubits_at_this_distance)
