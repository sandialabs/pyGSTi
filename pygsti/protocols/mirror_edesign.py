from __future__ import annotations
from typing import Callable, Union, Literal, Optional, Tuple

import numpy as _np
import tqdm as _tqdm
import warnings as _warnings

from collections import defaultdict

from pygsti.circuits.circuit import Circuit as _Circuit

from pygsti.protocols.protocol import FreeformDesign as _FreeformDesign
from pygsti.protocols.protocol import CombinedExperimentDesign as _CombinedExperimentDesign

from pygsti.processors import random_compilation as _rc

from pygsti.circuits import subcircuit_selection as _subcircsel

import time

#TODO: OOP-ify this code?

def get_active_qubits(circ, ignore=None):
    from qiskit.converters.circuit_to_dag import circuit_to_dag

    dag = circuit_to_dag(circ)
    active_qubits = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires(ignore=ignore)]
    
    return(active_qubits)

def get_active_qubits_no_dag(circ, ignore=set()):
    active_qubits = set()
    for instruction, qargs, _ in circ.data:
        if instruction.name in ignore:
            continue
        for qubit in qargs:
            active_qubits.add(qubit)

    return active_qubits


def qiskit_circuits_to_fullstack_mirror_edesign(qk_circs, #not transpiled
                                                    qk_backend=None,
                                                    coupling_map=None,
                                                    basis_gates=None,
                                                    transpiler_kwargs_dict={},
                                                    mirroring_kwargs_dict={},
                                                    num_transpilation_attempts=100,
                                                    return_qiskit_time=False
                                                    ):

    """
    Create a full-stack benchmark from high-level Qiskit circuits.

    Parameters
    ----------
    qk_circs : List[qiskit.QuantumCircuit] | Dict[qiskit.QuantumCircuit]
        Qiskit QuantumCircuits from which a full-stack benchmark is to be created.
        If a dictionary is provided, those keys are used.
        If a list is provided, default integer keys are used.

    qk_backend : qiskit-ibm-runtime.IBMBackend, optional
        IBM backend whose native gate set, connectivity, and error rates
        are to be targeted when doing a full-stack transpilation. Fake backends
        are also acceptable. If not provided, both `coupling_map` and `basis_gates`
        must be provided, and certain transpiler optimizations that depend on
        backend error rates may not be accessible.

    coupling_map : qiskit.transpiler.CouplingMap, optional
        couplinp map for a target backend that must be provided if `qk_backend` is None.
        This argument is ignored if `qk_backend` is not None.

    basis_gates : list[str], optional
        list of native gates for a target backend that must be provided
        if `qk_backend` is None.
        This argument is ignored if `qk_backend` is not None.

    transpiler_kwargs_dict : dict, optional
        keyword arguments that are passed to the Qiskit transpiler for full-stack
        transpilation. Please see the Qiskit transpiler documentation
        for a comprehensive list of options. If any or all of the following keys
        are not provided, the listed defaults are used:

        'optimization_level': default 3
        'approximation_degree': default 1.0
        'seed_transpiler': default None

    mirroring_kwargs_dict : dict, optional
        dictionary of keyword arguments to be used in circuit mirroring. If an arg
        is not provided, a default value is used.
        The args are:
            'mirror_circuits_per_circ': default 10. The number of mirror circuits of the
            test-exact and exact-exact varieties to be used for the process fidelity estimation
            of each provided Qiskit circuit.

            'num_ref_per_qubit_subset': default 10. The number of SPAM reference circuits to use
            for each qubit subset that is represented among the provided Qiskit circuits.

            'rand_state': default None. np.random.RandomState to be used for circuit mirroring.

    num_transpilation_attempts : int, optional
        number of times to attempt full-stack circuit transpilation. Circuit mirroring requires
        that the transpilation not use ancilla qubits, which is difficult to enforce with the
        other transpiler options. Instead, we adopt a try-until-success strategy, which may fail
        if an insufficient number of attempts are allowed. Increase this number if you are
        having issues with the transpilation. If not provided, the default is 100 attempts.

    return_qiskit_time : bool, optional
        Debug flag that sets whether or not to report the time spent in the Qiskit transpiler.
        Qiskit transpilation is often the most costly part of benchmark creation and it can be
        helpful to know how much time it is consuming.

    Returns
    ---------
        Tuple
            pygsti.protocols.FreeformDesign
                Experiment design containing the pyGSTi conversion of all Qiskit circuits that
                were passed in. Does not need executed, but is needed for fidelity calculations.

            pygsti.protocols.CombinedExperimentDesign
                Experiment design containing all mirror circuits that must be executed
                in order to perform mirror circuit fidelity estimation.

            float, optional
                amount of time spent in Qiskit transpiler.
                
    """

    try:
        import qiskit
        if qiskit.__version__ != '1.1.1':
            print("warning: 'qiskit_circuits_to_fullstack_mirror_edesign' is designed for qiskit 1.1.1. Your version is " + qiskit.__version__)

        from qiskit import transpile
    except:
        raise RuntimeError('qiskit is required for this operation, and does not appear to be installed.')

    if qk_backend is None:
        assert coupling_map is not None and basis_gates is not None, "'coupling_map' and 'basis_gates' must be provided if 'qk_backend is not provided."

    test_circ_dict = defaultdict(list)
    ref_circ_dict = defaultdict(list)

    ref_circ_id_lookup_dict = {}

    if isinstance(qk_circs, list):
        qk_circs = {i: qk_circ for i, qk_circ in enumerate(qk_circs)}

    # set default transpilation kwargs
    transpiler_kwargs_dict['optimization_level'] = transpiler_kwargs_dict.get('optimization_level', 3)
    transpiler_kwargs_dict['approximation_degree'] = transpiler_kwargs_dict.get('approximation_degree', 1.0)
    transpiler_kwargs_dict['seed_transpiler'] = transpiler_kwargs_dict.get('seed_transpiler', None)

    if return_qiskit_time:
        qiskit_time = 0.0

    for k, qk_circ in qk_circs.items():
        num_virtual_qubits = qk_circ.num_qubits
        for _ in range(num_transpilation_attempts):
            
            if return_qiskit_time: start = time.time()

            if qk_backend is not None:
                if coupling_map is not None:
                    _warnings.warn("'coupling_map' is ignored when 'qk_backend' is provided.")
                if basis_gates is not None:
                    _warnings.warn("'basis_gates' is ignored when 'qk_backend' is provided.")

                qk_test_circ = transpile(qk_circ,
                                         backend=qk_backend,
                                         coupling_map=coupling_map,
                                         **transpiler_kwargs_dict)
            else:
                qk_test_circ = transpile(qk_circ,
                                         coupling_map=coupling_map,
                                         basis_gates=basis_gates,
                                         **transpiler_kwargs_dict
                                         )

            if return_qiskit_time: qiskit_time += time.time() - start

            # num_active_qubits = len(active_qubits(qk_test_circ))
            active_qubits = get_active_qubits_no_dag(qk_test_circ)

            uses_ancilla = False

            for qubit in active_qubits:
                idx = qubit._index
                init_register = qk_test_circ.layout.initial_layout[idx]._register
                if init_register.name == 'ancilla':
                    uses_ancilla = True
                    print(qubit)
                    break

            # assert num_active_qubits == num_active_qubits_2

            if not uses_ancilla:
                break
        else:
            raise RuntimeError('Could not generate transpilation that does not use ancilla qubits. Maybe try increasing `num_transpilation_attempts?`')

        
        # import ipdb

        # ipdb.set_trace()


        qk_init_opt_layout = qk_test_circ.layout.initial_index_layout(filter_ancillas=False)[:num_virtual_qubits]
        qk_final_opt_layout = qk_test_circ.layout.final_index_layout(filter_ancillas=False)[:num_virtual_qubits]

        assert set(qk_init_opt_layout) == set(qk_final_opt_layout)

        ps_test_circ, qubit_mapping = _Circuit.from_qiskit(qk_test_circ, allow_different_gates_in_same_layer=False)
        ps_test_circ = ps_test_circ.delete_idling_lines()

        assert len(ps_test_circ.line_labels) <= num_virtual_qubits
        # ps_test_circ = ps_test_circ.reorder_lines(sorted(ps_test_circ.line_labels))

        ps_init_opt_layout = [qubit_mapping[i] for i in qk_init_opt_layout]
        ps_final_opt_layout = [qubit_mapping[i] for i in qk_final_opt_layout]
        ps_routing_perm = {ps_init_opt_layout[i]: ps_final_opt_layout[i] for i in range(num_virtual_qubits)}

        assert set(ps_init_opt_layout) == set(ps_final_opt_layout)

        test_metadata = {
            'id': k,
            'width': ps_test_circ.width,
            'depth': ps_test_circ.depth,
            'initial_layout': ps_init_opt_layout,
            'final_layout': ps_final_opt_layout,
            'routing_permutation': ps_routing_perm,
        }

        # print(ps_test_circ.depth)
        
        test_circ_dict[ps_test_circ] += [test_metadata]

        if qk_backend is not None:
            reduced_coupling_map = qk_backend.coupling_map.reduce(qk_final_opt_layout) # Thanks Jordan
        else:
            reduced_coupling_map = coupling_map.reduce(qk_final_opt_layout)


        # generate exact reference circuit, ensuring that the line labels will align with the test circuit when inverted.
        ref_seed = 0
        for _ in range(num_transpilation_attempts):
            

            # if return_qiskit_time: start = time.time()
            # ipdb.set_trace()
            qk_ref_inv_circ = transpile(qk_circ.inverse(),
                                        coupling_map=reduced_coupling_map,
                                        basis_gates=['u3', 'cz'],
                                        initial_layout=list(range(num_virtual_qubits)),
                                        optimization_level=1,
                                        approximation_degree=1.0,
                                        seed_transpiler=ref_seed
                                        )
            
            # if return_qiskit_time: qiskit_time += time.time() - start

            # num_active_qubits = len(active_qubits(qk_ref_inv_circ))
            active_qubits = get_active_qubits_no_dag(qk_ref_inv_circ)

            uses_ancilla = False

            for qubit in active_qubits:
                idx = qubit._index
                init_register = qk_ref_inv_circ.layout.initial_layout[idx]._register
                if init_register.name == 'ancilla':
                    uses_ancilla = True
                    print(qubit)
                    break

            if not uses_ancilla:
                break

            ref_seed += 1

        else:
            raise RuntimeError('Could not generate transpilation that does not use ancilla qubits.')

        qk_init_ref_inv_reduced_layout = qk_ref_inv_circ.layout.initial_index_layout(filter_ancillas=False)[:num_virtual_qubits]

        qk_final_ref_inv_reduced_layout = qk_ref_inv_circ.layout.final_index_layout(filter_ancillas=False)[:num_virtual_qubits]

        qk_ref_circ = qk_ref_inv_circ.inverse()

        qubit_map = {qk_ref_inv_circ.qbit_argument_conversion(i)[0]: f'Q{qk_final_opt_layout[i]}'
                         for i in qk_init_ref_inv_reduced_layout}

        ps_ref_circ, qubit_idx_map = _Circuit.from_qiskit(qk_ref_circ, qubit_conversion=qubit_map, allow_different_gates_in_same_layer=False)
        ps_ref_circ = ps_ref_circ.delete_idling_lines()

        assert len(ps_ref_circ.line_labels) <= num_virtual_qubits
        # ps_ref_circ = ps_ref_circ.reorder_lines(sorted(ps_ref_circ.line_labels))

        ps_init_ref_inv_layout = [qubit_idx_map[i] for i in qk_init_ref_inv_reduced_layout]
        ps_final_ref_inv_layout = [qubit_idx_map[i] for i in qk_final_ref_inv_reduced_layout]
        routing_perm_ref_inv_ps = {ps_init_ref_inv_layout[i]: ps_final_ref_inv_layout[i] for i in range(num_virtual_qubits)}

        ref_metadata = {
            'id': k,
            'width': ps_ref_circ.width,
            'depth': ps_ref_circ.depth,
            'initial_layout_inv': ps_init_ref_inv_layout,
            'final_layout_inv': ps_final_ref_inv_layout,
            'routing_permutation_inv': routing_perm_ref_inv_ps
        }

        # print(ps_ref_circ.depth)

        ref_circ_dict[ps_ref_circ] += [ref_metadata]

        # ipdb.set_trace()

        ref_circ_id_lookup_dict[k] = ps_ref_circ

        

    test_edesign = _FreeformDesign(test_circ_dict)
    ref_edesign = _FreeformDesign(ref_circ_dict)

    ### set default options for circuit mirroring if not provided
    # mirroring_kwargs_dict['num_mcs_per_circ'] = mirroring_kwargs_dict.get('num_mcs_per_circ', 10)
    # mirroring_kwargs_dict['num_ref_per_qubit_subset'] = mirroring_kwargs_dict.get('num_ref_per_qubit_subset', 10)
    # mirroring_kwargs_dict['rand_state'] = mirroring_kwargs_dict.get('rand_state', _np.random.RandomState())

    mirror_edesign = make_mirror_edesign(test_edesign=test_edesign,
                                         ref_edesign=ref_edesign,
                                         ref_id_lookup_dict=ref_circ_id_lookup_dict,
                                         account_for_routing=True,
                                         **mirroring_kwargs_dict
                                         )
    if return_qiskit_time:
        return test_edesign, mirror_edesign, qiskit_time
    else:
        return test_edesign, mirror_edesign

def qiskit_circuits_to_mirror_edesign(qk_circs, # yes transpiled
                                          mirroring_kwargs_dict={}
                                          ):

    """
    Create a noise benchmark from transpiled Qiskit circuits.

    Parameters
    ----------
    qk_circs : List[qiskit.QuantumCircuit] | Dict[qiskit.QuantumCircuit]
        Qiskit QuantumCircuits from which a noise benchmark is to be created.
        If a dictionary is provided, those keys are used.
        If a list is provided, default integer keys are used.

    mirroring_kwargs_dict : dict, optional
        dictionary of keyword arguments to be used in circuit mirroring. If an arg
        is not provided, a default value is used.
        The args are:
            'mirror_circuits_per_circ': default 10. The number of mirror circuits of the
            test-exact and exact-exact varieties to be used for the process fidelity estimation
            of each provided Qiskit circuit.

            'num_ref_per_qubit_subset': default 10. The number of SPAM reference circuits to use
            for each qubit subset that is represented among the provided Qiskit circuits.

            'rand_state': default None. np.random.RandomState to be used for circuit mirroring.

    Returns
    ---------
        Tuple
            pygsti.protocols.FreeformDesign
                Experiment design containing the pyGSTi conversion of all Qiskit circuits that
                were passed in. Does not need executed, but is needed for fidelity calculations.

            pygsti.protocols.CombinedExperimentDesign
                Experiment design containing all mirror circuits that must be executed
                in order to perform mirror circuit fidelity estimation.
                
    """

    try:
        import qiskit
        if qiskit.__version__ != '1.1.1':
            print("warning: 'qiskit_circuits_to_mirror_edesign' is designed for qiskit 1.1.1. Your version is " + qiskit.__version__)

        from qiskit import transpile

    except:
        raise RuntimeError('qiskit is required for this operation, and does not appear to be installed.')


    test_circ_dict = defaultdict(list)
    ref_circ_dict = defaultdict(list)

    ref_circ_id_lookup_dict = {}

    if isinstance(qk_circs, list):
        qk_circs = {i: qk_circ for i, qk_circ in enumerate(qk_circs)}

    for k, qk_test_circ in qk_circs.items():

        # start = time.time()
        qk_ref_circ = transpile(qk_test_circ,
                                basis_gates=['u3', 'cz'],
                                layout_method='trivial',
                                routing_method='none',
                                optimization_level=1,
                                approximation_degree=1.0,
                                seed_transpiler=0)
        
        # elapsed = time.time() - start
        # print(f'transpilation time: {elapsed}')
        

        # from qiskit_aer import AerSimulator

        # sim = AerSimulator()

        # qc_2 = qk_test_circ.compose(qk_ref_circ.inverse())

        # qc_2.measure_active()

        # print('pre-pyGSTi:')
        # print(sim.run(qc_2, shots=64).result().get_counts())

        ps_test_circ, _ = _Circuit.from_qiskit(qk_test_circ, allow_different_gates_in_same_layer=True)
        ps_ref_circ, _ = _Circuit.from_qiskit(qk_ref_circ, allow_different_gates_in_same_layer=False)

        # R = ps_ref_circ
        # T = ps_test_circ

        # R_inv = compute_inverse(circ=R, gate_set='u3_cx', inverse=None, inv_kwargs=None)

        # TR_inv = T + R_inv
        # TR_qk = TR_inv.convert_to_qiskit(qubit_conversion='remove-Q', qubits_to_measure='active')

        # print('post-pyGSTi:')
        # print(sim.run(TR_qk, shots=64).result().get_counts())
        

        ps_test_circ = ps_test_circ.delete_idling_lines()
        # ps_test_circ = ps_test_circ.reorder_lines(sorted(ps_test_circ.line_labels))

        ps_ref_circ = ps_ref_circ.delete_idling_lines()
        # ps_ref_circ = ps_ref_circ.reorder_lines(sorted(ps_ref_circ.line_labels))


        # R = ps_ref_circ
        # T = ps_test_circ

        # R_inv = compute_inverse(circ=R, gate_set='u3_cx', inverse=None, inv_kwargs=None)

        # TR_inv = T + R_inv
        # TR_qk = TR_inv.convert_to_qiskit(qubit_conversion='remove-Q', qubits_to_measure='active')

        # print('post-reordering:')
        # print(sim.run(TR_qk, shots=64).result().get_counts())

        # import ipdb

        # ipdb.set_trace()


        test_circ_metadata = {
            'id': k,
            'width': ps_test_circ.width,
            'depth': ps_test_circ.depth,
            # other information as it becomes necessary
        }

        test_circ_dict[ps_test_circ] += [test_circ_metadata]

        ref_circ_metadata = {
                            'id': k,
                            }
        
        ref_circ_dict[ps_ref_circ] += [ref_circ_metadata]

        ref_circ_id_lookup_dict[k] = ps_ref_circ

        test_edesign = _FreeformDesign(test_circ_dict)
        ref_edesign = _FreeformDesign(ref_circ_dict)

    ### set default values for mirroring kwargs if not provided
    # mirroring_kwargs_dict['num_mcs_per_circ'] = mirroring_kwargs_dict.get('num_mcs_per_circ', 10)
    # mirroring_kwargs_dict['num_ref_per_qubit_subset'] = mirroring_kwargs_dict.get('num_ref_per_qubit_subset', 10)
    # mirroring_kwargs_dict['rand_state'] = mirroring_kwargs_dict.get('rand_state', _np.random.RandomState())
    
    start = time.time()

    mirror_edesign = make_mirror_edesign(test_edesign=test_edesign,
                                         ref_edesign=ref_edesign,
                                         ref_id_lookup_dict=ref_circ_id_lookup_dict,
                                         account_for_routing=False,
                                         **mirroring_kwargs_dict
                                         )

    elapsed = time.time() - start
    print(f'mirroring time: {elapsed}')
    
    return test_edesign, mirror_edesign


def qiskit_circuits_to_svb_mirror_edesign(qk_circs,
                                              aggregate_subcircs,
                                              width_depth_dict,
                                              coupling_map,
                                              instruction_durations,
                                              subcirc_kwargs_dict={},
                                              mirroring_kwargs_dict={}
                                              ): # qk_circs must already be transpiled to the device
    
    """
    Create a subcircuit benchmark from transpiled Qiskit circuits.

    Parameters
    ----------
    qk_circs : List[qiskit.QuantumCircuit] | Dict[qiskit.QuantumCircuit]
        Qiskit QuantumCircuits from which a subcircuit benchmark is to be created.
        If a dictionary is provided, those keys are used.
        If a list is provided, default integer keys are used.

    aggregate_subcircs : bool
        Whether or not the provided Qiskit circuits should be used to create
        one combined subcircuit experiment design or kept separate.
        Circuit aggregration can be useful if the provided circuits are all
        instances of the same 'kind' of the circuit, e.g., all
        Bernstein-Vazirani circuits with different secret keys.

    width_depth_dict : dict[int, list[int]]
        dictionary whose keys are subcircuit widths and whose values are
        lists of depths to snip out for that width.

    coupling_map : str or qiskit.transpiler.CouplingMap
        coupling map for the device the Qiskit circuits were transpiled to.
        If 'all-to-all', an all-to-all coupling map is used.
        If 'linear', a linear topology is used.

    instruction_durations : qiskit.transpiler.InstructionDurations
        instruction durations for each gate in the target device. These
        durations are needed to calculate the appropriate delay time
        when only idling qubits are sampled out from a full circuit layer.


    subcirc_kwargs_dict : dict, optional
        dictionary of keyword arguments to be used in subcircuit selection.
        If an arg is not provided, a default value is used.
        The args are:
            'num_samples_per_width_depth': default 10. number of subcircuits to sample
            from each full circuit. if `aggregate_subcircuits` is set to
            True, `num_samplse_per_width_depth` subcircuits are drawn from
            each full circuit and combined into one experiment design.

            'rand_state': default None. np.random.RandomState to be used for subcircuit selection.

    mirroring_kwargs_dict : dict, optional
        dictionary of keyword arguments to be used in circuit mirroring
        If an arg is not provided, a default value is used.
        The args are:
            'mirror_circuits_per_circ': default 10. The number of mirror circuits of the
            test-exact and exact-exact varieties to be used for the process fidelity estimation
            of each provided Qiskit circuit.

            'num_ref_per_qubit_subset': default 10. The number of SPAM reference circuits to use
            for each qubit subset that is represented among the provided Qiskit circuits.

            'rand_state': default None. np.random.RandomState to be used for circuit mirroring.

    Returns
    ---------
        Tuple
            dict[hashable, pygsti.protocols.FreeformDesign] or pygsti.protocols.FreeformDesign
                Experiment design(s) containing the pyGSTi conversion of all Qiskit circuits that
                were passed in. Does not need executed, but is needed for fidelity calculations.
                A dictionary is returned if `aggregate_subcircs` is False, otherwise a FreeformDesign
                is returned.

            dict[hashable, pygsti.protocols.CombinedExperimentDesign] or pygsti.protocols.CombinedExperimentDesign
                Experiment design(s) containing all mirror circuits that must be executed
                in order to perform mirror circuit fidelity estimation. A dictionary is returned
                if `aggregate_subcircs` is False, otherwise a FreeformDesign is returned.

    """
    
    try:
        import qiskit
        if qiskit.__version__ != '1.1.1':
            print("warning: 'qiskit_circuits_to_svb_mirror_edesign' is designed for qiskit 1.1.1. Your version is " + qiskit.__version__)

        from qiskit import transpile
    except:
        raise RuntimeError('qiskit is required for this operation, and does not appear to be installed.')
    
    """
    Create a subcircuit benchmark from transpiled Qiskit circuits.

    Parameters
    ----------
    qk_circs : List[qiskit.QuantumCircuit] | Dict[qiskit.QuantumCircuit]
        Qiskit QuantumCircuits from which a subcircuit benchmark is to be created.
        If a dictionary is provided, those keys are used.
        If a list is provided, default integer keys are used.

    aggregate_subcircs : bool
        Whether or not the provided Qiskit circuits should be used to create
        one combined subcircuit experiment design or kept separate.
        Circuit aggregration can be useful if the provided circuits are all
        instances of the same 'family' of the circuit, e.g., all
        Bernstein-Vazirani circuits with different secret keys.

    width_depth_dict : dict[int, list[int]]
        dictionary whose keys are subcircuit widths and whose values are
        lists of depths to snip out for that width.

    coupling_map : str or qiskit.transpiler.CouplingMap
        coupling map for the device the Qiskit circuits were transpiled to.
        If 'all-to-all', an all-to-all coupling map is used.
        If 'linear', a linear topology is used.

    instruction_durations : qiskit.transpiler.InstructionDurations
        instruction durations for each gate in the target device. These
        durations are needed to calculate the appropriate delay time
        when only idling qubits are sampled out from a full circuit layer.


    subcirc_kwargs_dict : dict, optional
        dictionary of keyword arguments to be used in subcircuit selection.
        If an arg is not provided, a default value is used.
        The args are:
            'num_samples_per_width_depth': default 10. number of subcircuits to sample
            from each full circuit. if `aggregate_subcircuits` is set to
            True, `num_samplse_per_width_depth` subcircuits are drawn from
            each full circuit and combined into one experiment design.

            'rand_state': default None. np.random.RandomState to be used for subcircuit selection.

    mirroring_kwargs_dict : dict, optional
        dictionary of keyword arguments to be used in circuit mirroring
        If an arg is not provided, a default value is used.
        The args are:
            'mirror_circuits_per_circ': default 10. The number of mirror circuits of the
            test-exact and exact-exact varieties to be used for the process fidelity estimation
            of each provided Qiskit circuit.

            'num_ref_per_qubit_subset': default 10. The number of SPAM reference circuits to use
            for each qubit subset that is represented among the provided Qiskit circuits.

            'rand_state': default None. np.random.RandomState to be used for circuit mirroring.

    Returns
    ---------
        Tuple
            dict[hashable, pygsti.protocols.FreeformDesign] or pygsti.protocols.FreeformDesign
                Experiment design(s) containing the pyGSTi conversion of all Qiskit circuits that
                were passed in. Does not need executed, but is needed for fidelity calculations.
                A dictionary is returned if `aggregate_subcircs` is False, otherwise a FreeformDesign
                is returned.

            dict[hashable, pygsti.protocols.CombinedExperimentDesign] or pygsti.protocols.CombinedExperimentDesign
                Experiment design(s) containing all mirror circuits that must be executed
                in order to perform mirror circuit fidelity estimation. A dictionary is returned
                if `aggregate_subcircs` is False, otherwise a FreeformDesign is returned.

    """

    if isinstance(qk_circs, list):
        qk_circs = {i: qk_circ for i, qk_circ in enumerate(qk_circs)}


    ps_circs = {}
    for k, qk_circ in qk_circs.items():
        ps_circ, _ = _Circuit.from_qiskit(qk_circ, allow_different_gates_in_same_layer=True)
        ps_circ = ps_circ.delete_idling_lines()
        # ps_circ = ps_circ.reorder_lines(sorted(ps_circ.line_labels))

        ps_circs[k] = ps_circ

    # subcirc sampling in device native gate set

    subcirc_kwargs_dict['num_samples_per_width_depth'] = subcirc_kwargs_dict.get('num_samples_per_width_depth', 10)
    
    test_edesigns = {}
    if aggregate_subcircs:
        test_edesign = _subcircsel.sample_subcircuits(full_circs=list(ps_circs.values()),
                                                      width_depths=width_depth_dict,
                                                      instruction_durations=instruction_durations,
                                                      coupling_map=coupling_map,
                                                      **subcirc_kwargs_dict)
        test_edesigns[0] = test_edesign

    else:
        for k, ps_circ in ps_circs.items():
            test_edesign = _subcircsel.sample_subcircuits(full_circs=ps_circ,
                                                          width_depths=width_depth_dict,
                                                          instruction_durations=instruction_durations,
                                                          coupling_map=coupling_map,
                                                          **subcirc_kwargs_dict)
            
            test_edesigns[k] = test_edesign

    mirror_edesigns = {}

    for k, test_edesign in test_edesigns.items():
        ref_circ_dict = defaultdict(list)
        ref_circ_id_lookup_dict = {}

        for ps_test_circ, auxlist in test_edesign.aux_info.items():
            qubit_mapping_dict = {sslbl: i for i, sslbl in enumerate(ps_test_circ.line_labels)} # avoid mapping small circuits to large backends since connectivity is already ensured

            # import ipdb

            # ipdb.set_trace()

            # convert back to qiskit to perform reference transpilations in u3-cz gate set (no layer blocking required, just need logical equivalence)
            qk_test_circ = ps_test_circ.convert_to_qiskit(qubit_conversion=qubit_mapping_dict,
                                                          )
            qk_ref_circ = transpile(qk_test_circ,
                                    basis_gates=['u3', 'cz'],
                                    layout_method='trivial',
                                    routing_method='none',
                                    optimization_level=1,
                                    approximation_degree=1.0,
                                    seed_transpiler=0
                                    )

    

            # convert those reference circuits back to pyGSTi (layer blocking required to separate u3 and cz)
            qubit_mapping_dict_2 = {qk_test_circ.qbit_argument_conversion(i)[0]: sslbl for sslbl, i in qubit_mapping_dict.items()} # now map back
            # in Qiskit 1.1.1, the method is called qbit_argument_conversion. In Qiskit >=1.2 (as far as Noah can tell), the method is called _qbit_argument_conversion.

            ps_ref_circ, _ = _Circuit.from_qiskit(qk_ref_circ,
                                                  qubit_conversion=qubit_mapping_dict_2, allow_different_gates_in_same_layer=False)
            ps_ref_circ = ps_ref_circ.delete_idling_lines()
            # ps_ref_circ = ps_ref_circ.reorder_lines(sorted(ps_ref_circ.line_labels))
            
            for aux in auxlist:
                ref_circ_dict[ps_ref_circ] += [aux]
                ref_circ_id_lookup_dict[aux['id']] = ps_ref_circ

        ref_edesign = _FreeformDesign(ref_circ_dict)

        mirror_edesign = make_mirror_edesign(test_edesign=test_edesign,
                                             ref_edesign=ref_edesign,
                                             ref_id_lookup_dict=ref_circ_id_lookup_dict,
                                             account_for_routing=False,
                                             **mirroring_kwargs_dict
                                             )

        mirror_edesigns[k] = mirror_edesign

    if aggregate_subcircs:
        return test_edesigns[0], mirror_edesigns[0]
    else:
        return test_edesigns, mirror_edesigns



def make_mirror_edesign(test_edesign: _FreeformDesign,
                        account_for_routing: bool, # handles permutation compiler optimization
                        ref_edesign: Optional[_FreeformDesign] = None,
                        ref_id_lookup_dict: Optional[dict] = None,
                        num_mcs_per_circ: int = 10,
                        num_ref_per_qubit_subset: int = 10,
                        mirroring_strategy: Literal['pauli_rc', 'central_pauli'] = 'pauli_rc',
                        gate_set: str = 'u3_cx',
                        inverse: Optional[Callable[[_Circuit], _Circuit]] = None,
                        inv_kwargs: Optional[dict] = None,
                        rc_function: Optional[Callable[[_Circuit], Tuple[_Circuit, str]]] = None,
                        rc_kwargs: Optional[dict] = None,
                        cp_function: Optional[Callable[[_Circuit], Tuple[_Circuit, str]]] = None,
                        cp_kwargs: Optional[dict] = None,
                        state_initialization: Optional[Union[str, Callable[..., _Circuit]]] = None,
                        state_init_kwargs: Optional[dict] = None,
                        rand_state: Optional[_np.random.RandomState] = None) -> _CombinedExperimentDesign:
    # idea for random_compiling and state_initialization is that a power user could supply their own functions if they did not want to use the built-in options.

    # Please read carefully for guidelines on how design your own functions that implement a circuit inverse, a state prep, an RC, and a central Pauli propagation.
    # You may, but are not required to, provide custom implementations for supported gate sets.
    # For unsupported gate sets, you *must* provide a custom implementation.

    # Inverse method
    #   Signature: inverse(circ: pygsti.circuits.Circuit, ...) -> pygsti.circuits.Circuit
    #   You *must* have 'circ' as the circuit parameter name.
    #   Supply any additional kwargs needed by your custom inverse in the inv_kwargs kwarg when calling make_mirror_edesign().

    # State prep method
    #   Signature: state_initialization(qubits, rand_state: Optional[_np.random.RandomState] = None, ...) -> pygsti.circuits.Circuit
    #   You *must* have 'qubits' (list of the qubits being used) and 'rand_state' as parameter names.
    #   Supply any additional kwargs needed by your custom state initialization in the state_init_kwargs kwarg when calling make_mirror_design().

    # RC method
    #   Signature: rc_function(circ: pygsti.circuits.Circuit, rand_state: Optional[_np.random.RandomState] = None, ...) -> Tuple[pygsti.circuits.Circuit, str]
    #   The user-defined function must return the randomized circuit, along with the expected bitstring measurement given the randomization.
    #   This function is called twice:
    #       1) On the reverse half of the circuit, when creating the init-test-ref_inv-init_inv circuit. ref_inv and init_inv are randomized. The bitstring should be the expected measurement *for the full circuit*, *not* ref_inv-init_inv in isolation from the forward half of the circuit. Pass the forward half of the circuit as a kwarg in rc_kwargs if necessary.
    #       2) On the entire circuit, when creating the init-ref-ref_inv-init_inv circuit. All four pieces are randomized. The bitstring should again be the expected measurement for the full circuit, but it is more clear in this case than in the previous.
    #   Supply any additional kwargs needed by your custom RC function in the rc_kwargs kwarg when calling make_mirror_design().

    # CP method
    #   Signature: cp_function(forward_circ: pygsti.circuits.Circuit, reverse_circ: pygsti.circuits.Circuit, rand_state: Optional[_np.random.RandomState] = None, ...) -> Tuple[pygsti.circuits.Circuit, str]
    #   The user-defined function must return the CP circuit, along with the expected bitstring measurement given the central Pauli. The general responsibility can be thought of as creating and then "propagating" a random central Pauli layer.
    #   This function is called once, with init-test passed as the forward circuit and ref_inv-init_inv passed as the reverse circuit.
    #   Supply any additional kwargs needed by your custom CP function in the cp_kwargs kwarg when calling make_mirror_design().



    if rand_state is None:
        rand_state = _np.random.RandomState()

    qubit_subsets = defaultdict(list)

    test_ref_invs = defaultdict(list)
    ref_ref_invs = defaultdict(list)
    spam_refs = defaultdict(list)


    central_pauli_allowed = True

    if ref_edesign is not None:
        assert ref_id_lookup_dict is not None, "when providing separate test and reference compilations, you must provide a lookup dictionary for the reference circuits so they can be matched with the correct test circuits."
        central_pauli_allowed = False # forward and reverse compilations must match for central pauli to be a valid fidelity estimation method.

    else:
        print("using provided edesign for both reference and test compilations")

    start = time.time()

    for c, auxlist in _tqdm.tqdm(test_edesign.aux_info.items(), ascii=True, desc='Sampling mirror circuits'):
        test_aux = auxlist[0]
        # print(f"auxlist length: {len(auxlist)}")
        # add a qubit subset to the pool if there's a circuit that uses it. if multiple identical circuits use it, do not double count; if distinct circuits use the same qubit subset, then do add it again.
        qubits = c.line_labels

        #print(qubits)

        qubit_subsets[test_aux['width']].append(qubits)

        if ref_edesign is not None:
            # find the corresponding exact circuit for the inexact circuit 'c' using the look-up table generated
            circ_id = test_aux['id']

            # assert ref_id_lookup_dict is not None, "when providing separate test and reference compilations, you must provide a lookup dictionary for the reference circuits so they can be matched with the correct test circuits."
            
            exact_circ = ref_id_lookup_dict[circ_id]

            # print(ref_edesign.aux_info[exact_circ])

            valid_test_ids = set(aux['id'] for aux in ref_edesign.aux_info[exact_circ])

            test_id = test_aux['id']

            assert circ_id in valid_test_ids, f"Invalid test ID {circ_id} for ref circuit corresponding to test IDs {valid_test_ids}"

        else:
            # print("using provided edesign for both reference and test compilations")
            exact_circ = c

        # R for "Reference" circuit, T for "Test" circuit
        R = exact_circ
        T = c

        # print(f'test circ depth: {T.depth}')
        # print(f'ref circ depth: {R.depth}')

        R_inv = compute_inverse(circ=R, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)

        # from qiskit_aer import AerSimulator

        # sim = AerSimulator()

        # TR_circ = T + R_inv
        # TR_qk = TR_circ.convert_to_qiskit(qubit_conversion='remove-Q',qubits_to_measure='active')

        # print(sim.run(TR_qk, shots=64).result().get_counts())

        # print(R)
        # print(T)
        for j in range(num_mcs_per_circ):
            
            random_compiler = _rc.RandomCompilation(mirroring_strategy, rand_state)
            
            if mirroring_strategy == 'pauli_rc':
                L_bareref = init_layer(qubits=qubits, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)

                L_refref = init_layer(qubits=qubits, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)
                           
                L_bareref_inv = compute_inverse(circ=L_bareref, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)

                L_refref_inv = compute_inverse(circ=L_refref, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)

                if account_for_routing: # only the bare-ref circuit has non-trivial routing
                    assert ref_edesign is not None, "'account_for_routing' set to True but no ref_edesign has been provided to match routing. If you are not providing a 'ref_edesign' that needs matched with a 'test_edesign', please set 'account_for_routing' to False."

                    T_routing = test_aux['routing_permutation']
                    
                    ref_aux = ref_edesign.aux_info[R][0]

                    R_routing = ref_aux['routing_permutation_inv']

                    mc_routing_perm = {k: R_routing[v] for k,v in T_routing.items()} # compute overall routing permutation for the composition of the test circuit and the inverse of the reference circuit

                    # pygsti_mc_routing_perm = {f'Q{k}': f'Q{v}' for k,v in mc_routing_perm.items()} # pygsti labels are Q0, Q1, etc. instead of 0, 1, etc.

                    L_bareref_inv = L_bareref_inv.map_state_space_labels(mc_routing_perm)



                if rc_function is not None:
                    try:
                        # bare-ref
                        Rinv_Linv, L_T_Rinv_Linv_bs = rc_function(circ=R_inv + L_bareref_inv, rand_state=rand_state, **rc_kwargs)
                        L_T_Rinv_Linv = L_bareref + T + Rinv_Linv

                        # ref-ref
                        L_R_Rinv_Linv, L_R_Rinv_Linv_bs = rc_function(circ=L_refref + R + R_inv + L_refref_inv, rand_state=rand_state, **rc_kwargs)

                        # the assertions below check if the circuit addition in the function calls above have caused a line label reordering. If so, the MCFE code will compare bitstrings incorrectly, which is bad. This issue is fixable with enough Circuit.reorder_lines() calls, but the better approach is simply to ensure that all circuits in test_edesign and ref_edesign obey a lexicographical ordering. This is easily done by using something like 'c = c.reorder_lines(sorted(c.line_labels)) on the circuits *prior* to creating the mirror edesign.

                        assert L_T_Rinv_Linv.line_labels == qubits, f'line labels have been permuted: should be {qubits} but is {L_T_Rinv_Linv.line_labels} instead.'

                        assert L_R_Rinv_Linv.line_labels == qubits, f'line labels have been permuted: should be {qubits} but is {L_R_Rinv_Linv.line_labels} instead.'

                    except:
                        raise RuntimeError(f"User-provided RC function for gate set '{gate_set}' returned an error!")
                elif gate_set == 'u3_cx':

                    # bare-ref

                    # start2 = time.time()

                    Rinv_Linv, L_T_Rinv_Linv_bs = random_compiler.compile(R_inv + L_bareref_inv)
                    L_T_Rinv_Linv = L_bareref + T + Rinv_Linv

                    # print(f'tr time: {time.time() - start2}')

                    # start2 = time.time()

                    # L_T_Rinv_Linv = L_T_Rinv_Linv.reorder_lines(c.line_labels)

                    # ref-ref
                    L_R_Rinv_Linv, L_R_Rinv_Linv_bs = random_compiler.compile(L_refref + R + R_inv + L_refref_inv)

                    # print(f'rr time: {time.time() - start2}')

                    # the assertions below check if the circuit addition in the function calls above have caused a line label reordering. If so, the MCFE code will compare bitstrings incorrectly, which is bad. This issue is fixable with enough Circuit.reorder_lines() calls, but the better approach is simply to ensure that all circuits in test_edesign and ref_edesign obey a lexicographical ordering. This is easily done by using something like 'c = c.reorder_lines(sorted(c.line_labels)) on the circuits *prior* to creating the mirror edesign.

                    assert L_T_Rinv_Linv.line_labels == qubits, f'line labels have been permuted: should be {qubits} but is {L_T_Rinv_Linv.line_labels} instead.'

                    assert L_R_Rinv_Linv.line_labels == qubits, f'line labels have been permuted: should be {qubits} but is {L_R_Rinv_Linv.line_labels} instead.'

                elif gate_set == 'clifford':
                    #TODO: add clifford RC function
                    raise NotImplementedError("Clifford RC is not yet supported!")
                elif gate_set == 'clifford_rz':
                    #TODO: add clifford_rz RC function
                    raise NotImplementedError("Clifford_rz RC is not yet supported!")
                else: #we have no support for this gate set at this point, the user must provide a custom RC function
                    raise RuntimeError(f"No default RC function for gate set '{gate_set}' exists, you must provide your own!")

            elif mirroring_strategy == 'central_pauli':
                assert central_pauli_allowed, "Central Pauli is not allowed when 'ref_edesign' is provided."
                
                #do central pauli mirror circuit
                if cp_function is not None:
                    try:
                        L_T_Rinv_Linv, L_T_Rinv_Linv_bs = cp_function(forward_circ=L_refref+T, reverse_circ=R_inv+L_refref_inv, rand_state=rand_state, **cp_kwargs)

                        # the assertion belows check if the circuit addition in the function call above has caused a line label reordering. If so, the MCFE code will compare bitstrings incorrectly, which is bad. This issue is fixable with enough Circuit.reorder_lines() calls, but the better approach is simply to ensure that all circuits in test_edesign and ref_edesign obey a lexicographical ordering. This is easily done by using something like 'c = c.reorder_lines(sorted(c.line_labels)) on the circuits *prior* to creating the mirror edesign.

                        assert L_T_Rinv_Linv.line_labels == qubits, f'line labels have been permuted: should be {qubits} but is {L_T_Rinv_Linv.line_labels} instead.'

                    except:
                        raise RuntimeError(f"User-provided CP function for gate set '{gate_set}' returned an error!")
                elif gate_set == 'u3_cx':
                    #print(f'ref depth: {R.depth}, ref_inv depth: {R_inv.depth}, L depth: {L_inv.depth}')
                    #reload(cp)
                    # test_rand_state1 = _np.random.RandomState(8675309)

                    CP_Rinv_Linv, L_T_Rinv_Linv_bs = random_compiler.compile(circ=R_inv+L_refref_inv)
                    L_T_Rinv_Linv = L_refref + T + CP_Rinv_Linv
                    # L_T_Rinv_Linv = L_T_Rinv_Linv.reorder_lines(c.line_labels)
                    # print("new function:")
                    # print(L_T_Rinv_Linv)
                    # print(L_T_Rinv_Linv_bs)
                    # test_rand_state2 = _np.random.RandomState(8675309)
                    # L_T_Rinv_Linv1, L_T_Rinv_Linv_bs1 = cp.central_pauli_mirror_circuit(test_circ=L+T, ref_circ=L+R, randomized_state_preparation=False, rand_state=test_rand_state2, new=True)
                    # print("old function:")
                    # print(L_T_Rinv_Linv1)
                    # print(L_T_Rinv_Linv_bs1)
                    # assert L_T_Rinv_Linv_bs == L_T_Rinv_Linv_bs1, "different circuit outcomes predicted!"
                    # assert L_T_Rinv_Linv.__hash__() == L_T_Rinv_Linv1.__hash__(), f"circuit hashes do not match! Circuit 1: {L_T_Rinv_Linv}, Circuit 2: {L_T_Rinv_Linv1}"

                    # L_R_Rinv_Linv, L_R_Rinv_Linv_bs = rc.central_pauli_mirror_circuit(test_circ=L+R, ref_circ=L+R, randomized_state_preparation=False, rand_state=rand_state, new=True)

                    L_T_Rinv_Linv, L_T_Rinv_Linv_bs = cp_function(forward_circ=L_refref+T, reverse_circ=R_inv+L_refref_inv, rand_state=rand_state, **cp_kwargs)

                    # the assertion belows check if the circuit addition in the function call above has caused a line label reordering. If so, the MCFE code will compare bitstrings incorrectly, which is bad. This issue is fixable with enough Circuit.reorder_lines() calls, but the better approach is simply to ensure that all circuits in test_edesign and ref_edesign obey a lexicographical ordering. This is easily done by using something like 'c = c.reorder_lines(sorted(c.line_labels)) on the circuits *prior* to creating the mirror edesign.

                    assert L_T_Rinv_Linv.line_labels == qubits, f'line labels have been permuted: should be {qubits} but is {L_T_Rinv_Linv.line_labels} instead.'

                elif gate_set == 'clifford':
                    #TODO: add clifford CP function
                    raise NotImplementedError("Clifford CP is not yet supported!")
                elif gate_set == 'clifford_rz':
                    #TODO: add clifford_rz CP function
                    raise NotImplementedError("Clifford_rz CP is not yet supported!")
                else: #we have no support for this gate set at this point, the user must provide a custom CP function
                    raise RuntimeError(f"No default CP function for gate set '{gate_set}' exists, you must provide your own!")
                
            else:
                raise RuntimeError("'mirroring_strategy' must be either 'pauli_rc' or 'central_pauli'")


            # #debug shenanigans
            # from qiskit_aer import AerSimulator

            # TR_circ = T + R_inv
            # TR_qk = TR_circ.convert_to_qiskit(qubit_conversion='remove-Q',qubits_to_measure='active')

            # RR_circ = R + R_inv
            # RR_qk = RR_circ.convert_to_qiskit(qubit_conversion='remove-Q',qubits_to_measure='active')

            # M1_qk = L_T_Rinv_Linv.convert_to_qiskit(qubit_conversion='remove-Q',qubits_to_measure='active')
            # M2_qk = L_R_Rinv_Linv.convert_to_qiskit(qubit_conversion='remove-Q',qubits_to_measure='active')

            # sim = AerSimulator()

            # for qk_circ in [TR_qk, RR_qk, M1_qk, M2_qk]:
            #     print(sim.run(qk_circ, shots=64).result().get_counts())

            # import ipdb

            # ipdb.set_trace()


            # ### end debug nonsense

            
            L_T_Rinv_Linv_aux = [{'base_aux': a,
                                  'idealout': L_T_Rinv_Linv_bs,
                                  'qs_to_measure': L_T_Rinv_Linv.line_labels,
                                  'id': j} for a in auxlist]
            
            test_ref_invs[L_T_Rinv_Linv] = L_T_Rinv_Linv_aux

            if mirroring_strategy == 'pauli_rc': #we do not have this class of circuit for central pauli
                L_R_Rinv_Linv_aux = [{'base_aux': a,
                                      'idealout': L_R_Rinv_Linv_bs,
                                      'qs_to_measure': L_R_Rinv_Linv.line_labels,
                                      'id': j} for a in auxlist]
                ref_ref_invs[L_R_Rinv_Linv] = L_R_Rinv_Linv_aux
 
            # could change this to the same structure as described in the comment above if defaultdict(list) is not used

    # print(f'tr+rr time: {time.time() - start}')

    # start = time.time()

    for w, width_subsets in qubit_subsets.items():
        # subset_indices = rand_state.choice(list(range(len(width_subsets))), num_ref_per_qubit_subset)
        # for j in tqdm.tqdm(range(num_ref_per_qubit_subset), ascii=True, desc=f'Sampling width {w} reference circuits'):
        #     subset_idx = subset_indices[j]
        #     spam_qubits = width_subsets[subset_idx]
        #     L = init_layer(qubits=spam_qubits, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)
        #     spam_ref = L + compute_inverse(circ=L, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)
        #     spam_refs[spam_ref].append({'idealout': '0'*w, 'id': j, 'width': w})
        #     # same defaultdict(list) change can be made as earlier
        
        unique_subsets = set(width_subsets)
        print(f'Sampling reference circuits for width {w} with {len(unique_subsets)} subsets')
        for unique_subset in unique_subsets:
            for j in _tqdm.tqdm(range(num_ref_per_qubit_subset), ascii=True, desc=f'Sampling reference circuits for subset {unique_subset}'):
                L = init_layer(qubits=unique_subset, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)
                spam_ref = L + compute_inverse(circ=L, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)
                spam_refs[spam_ref].append({'idealout': '0'*w, 'id': j, 'qs_to_measure': spam_ref.line_labels, 'width': w})

    # print(f'ref time: {time.time() - start}')

    edesigns = {}
    if mirroring_strategy == 'pauli_rc':
        edesigns['br'] = _FreeformDesign(test_ref_invs)
        edesigns['rr'] = _FreeformDesign(ref_ref_invs)
        edesigns['ref'] = _FreeformDesign(spam_refs)
    elif mirroring_strategy == 'central_pauli':
        edesigns['cp'] = _FreeformDesign(test_ref_invs)
        edesigns['cpref'] = _FreeformDesign(spam_refs)
    else:
        raise RuntimeError("unknown mirroring strategy!")

    return _CombinedExperimentDesign(edesigns)


def compute_inverse(circ: _Circuit,
                    gate_set: str,
                    inverse: Optional[Callable[[_Circuit], _Circuit]] = None,
                    inv_kwargs: Optional[dict] = None) -> _Circuit:
    if inverse is not None:
        try:
            circ_inv = inverse(circ=circ, **inv_kwargs)
        except:
            raise RuntimeError(f"User-provided inverse function for gate set '{gate_set}' returned an error!")
    if gate_set == 'u3_cx':
        circ_inv = _rc.u3_cx_inv(circ)
    elif gate_set == 'clifford':
        #TODO: add clifford inverse function to circuit_inverse.py
        raise NotImplementedError("Clifford inversion is not yet supported!")
    elif gate_set == 'clifford_rz':
        #TODO: add clifford_rz inverse function to circuit_inverse.py
        raise NotImplementedError("Clifford_rz inversion is not yet supported!")
    else: #we have no support for this gate set at this point, the user must provide a custom inverse function
        raise RuntimeError(f"No default inverse function for gate set '{gate_set}' exists, you must provide your own!")
    
    return circ_inv

def init_layer(qubits,
               gate_set: str,
               state_initialization: Optional[Union[str, Callable[..., _Circuit]]] = None,
               rand_state: Optional[_np.random.RandomState] = None,
               state_init_kwargs: Optional[dict] = None) -> _Circuit:
    if state_initialization == 'none':
        L = _Circuit([], qubits)
    elif state_initialization is not None:
        try:
            L = state_initialization(qubits=qubits, rand_state=rand_state, **state_init_kwargs)
        except:
            raise RuntimeError(f"User-provided state_initialization function for gate set '{gate_set}' returned an error!")
    elif gate_set == 'u3_cx':
        prep_layer = _rc.haar_random_u3_layer(qubits, rand_state)
        L = _Circuit([prep_layer], qubits)
    elif gate_set == 'clifford':
        #TODO: add clifford default initialization
        raise NotImplementedError("Clifford state initialization is not yet supported!")
    elif gate_set == 'clifford_rz':
        #TODO: add clifford_rz default initialization
        raise NotImplementedError("Clifford_rz state initialization is not yet supported!")
    else: #we have no support for this gate set at this point, the user must provide a custom state initialization
        raise RuntimeError(f"No default state_initialization function for gate set '{gate_set}' exists, you must provide your own!")
    
    return L