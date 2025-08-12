"""
B-Sqale is a software tool for creating scalable, robust benchmarks from any quantum circuit.
Please see <paper link forthcoming> for more information.
"""

# TODO: add copyright assertion

from __future__ import annotations
from typing import Tuple, Optional, Dict, List, Union, Any
import warnings as _warnings

from pygsti.protocols import (
    ProtocolData as _ProtocolData,
    FreeformDesign as _FreeformDesign,
    CombinedExperimentDesign as _CombinedExperimentDesign,
    VBDataFrame as _VBDataFrame,
    mirror_edesign as _mirror
)

import numpy as _np

try:
    import qiskit
    if qiskit.__version__ != '1.1.1':
        _warnings.warn("The B-Sqale functions 'qiskit_circuits_to_mirror_edesign'," \
        "'qiskit_circuits_to_fullstack_mirror_edesign', and" \
        "'qiskit_circuits_to_subcircuit_mirror_edesign are designed for qiskit 1.1.1. Your version is " + qiskit.__version__)
    from qiskit import transpile

except:
    _warnings.warn("qiskit does not appear to be installed, and is required for the B-Sqale functions" \
                   "'qiskit_circuits_to_mirror_edesign'," \
                   "'qiskit_circuits_to_fullstack_mirror_edesign', and" \
                   "'qiskit_circuits_to_subcircuit_mirror_edesign'.")
    

def noise_mirror_benchmark(qk_circs: Union[Dict[Any, qiskit.QuantumCircuit], List[qiskit.QuantumCircuit]],
                         mirroring_kwargs_dict: Dict[str, Any] = {}
                         ) -> Tuple[_FreeformDesign, _CombinedExperimentDesign]:
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

    return _mirror.qiskit_circuits_to_mirror_edesign(qk_circs, mirroring_kwargs_dict)


def fullstack_mirror_benchmark(qk_circs: Union[Dict[Any, qiskit.QuantumCircuit], List[qiskit.QuantumCircuit]],
                             qk_backend: Optional[qiskit.providers.BackendV2] = None,
                             coupling_map: Optional[qiskit.transpiler.CouplingMap] = None,
                             basis_gates: Optional[List[str]] = None,
                             transpiler_kwargs_dict: Dict[str, Any] = {},
                             mirroring_kwargs_dict: Dict[str, Any] = {},
                             num_transpilation_attempts: int = 100,
                             return_qiskit_time: bool = False
                             ) -> Tuple[_FreeformDesign, _CombinedExperimentDesign]:
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

    return _mirror.qiskit_circuits_to_fullstack_mirror_edesign(qk_circs, #not transpiled
                                                    qk_backend,
                                                    coupling_map,
                                                    basis_gates,
                                                    transpiler_kwargs_dict,
                                                    mirroring_kwargs_dict,
                                                    num_transpilation_attempts,
                                                    return_qiskit_time
                                                    )


def subcircuit_mirror_benchmark(qk_circs: Union[Dict[Any, qiskit.QuantumCircuit], List[qiskit.QuantumCircuit]],
                              aggregate_subcircs: bool,
                              width_depth_dict: Dict[int, List[int]],
                              coupling_map: qiskit.transpiler.CouplingMap,
                              instruction_durations: qiskit.transpiler.InstructionDurations,
                              subcirc_kwargs_dict: Dict[str, Any] = {},
                              mirroring_kwargs_dict: Dict[str, Any] = {}
                              ) -> Tuple[_FreeformDesign, _CombinedExperimentDesign]:
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

    return _mirror.qiskit_circuits_to_subcircuit_mirror_edesign(qk_circs,
                                              aggregate_subcircs,
                                              width_depth_dict,
                                              coupling_map,
                                              instruction_durations,
                                              subcirc_kwargs_dict,
                                              mirroring_kwargs_dict
                                              ) # qk_circs must already be transpiled to the device


def calculate_mirror_benchmark_results(unmirrored_design: _FreeformDesign,
                                       mirrored_data: _ProtocolData,
                                       dropped_gates: bool = False,
                                       bootstrap: bool = True,
                                       num_bootstraps: int = 50,
                                       rand_state: _np.random.RandomState = None,
                                       verbose: bool = False,
                                       ) -> _VBDataFrame:
    """
        Create a dataframe from MCFE data and edesigns.

        Parameters
        ----------
        unmirrored_design: pygsti.protocols.protocol.FreeformDesign
            Edesign containing the circuits whose process fidelities are to be estimated.

        mirrored_data: pygsti.protocols.protocol.ProtocolData
            Data object containing the full mirror edesign and the outcome counts for each
            circuit in the full mirror edesign.

        verbose: bool
            Toggle print statements with debug information. If True, print statements are
            turned on. If False, print statements are omitted.

        bootstrap: bool
            Toggle the calculation of error bars from bootstrapped process fidelity calculations. If True,
            error bars are calculated. If False, error bars are not calculated.

        num_bootstraps: int
            Number of samples to draw from the bootstrapped process fidelity calculations. This argument
            is ignored if 'bootstrap' is False.

        rand_state: np.random.RandomState
            random state used to seed bootstrapping. If 'bootstrap' is set to False, this argument is ignored.

        Returns
        ---------
        VBDataFrame
            A VBDataFrame whose dataframe contains calculated MCFE values and circuit statistics.
        """

    return _VBDataFrame.from_mirror_experiment(unmirrored_design, mirrored_data,
                                                          dropped_gates,
                                                          bootstrap,
                                                          num_bootstraps,
                                                          rand_state,
                                                          verbose
                                                          )