"""
Tools for working with ExperimentDesigns
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import math

def calculate_edesign_estimated_runtime(edesign, gate_time_dict=None, gate_time_1Q=None,
                                        gate_time_2Q=None, measure_reset_time=0.0,
                                        interbatch_latency=0.0, total_shots_per_circuit=1000,
                                        shots_per_circuit_per_batch=None, circuits_per_batch=None):
    """Estimate the runtime for an ExperimentDesign from gate times and batch sizes.

    The rough model is that the required circuit shots are split into batches,
    where each batch runs a subset of the circuits for some fraction of the needed shots.
    One round consists of running all batches once, i.e. collecting some shots for all circuits,
    and rounds are repeated until the required number of shots is met for all circuits.

    In addition to gate times, the user can also provide the time at the end of each circuit
    for measurement and/or reset, as well as the latency between batches for classical upload/
    communication of the next set of circuits. Since times are user-provided, this function
    makes no assumption on the units of time, only that a consistent unit is used for all times.

    Parameters
    ----------
    edesign: ExperimentDesign
        An experiment design containing all required circuits.

    gate_time_dict: dict
        Dictionary with keys as either gate names or gate labels (for qubit-specific overrides)
        and values as gate time in user-specified units. All operations in the circuits of
        `edesign` must be specified. Either `gate_time_dict` or both `gate_time_1Q` and `gate_time_2Q`
        must be specified.

    gate_time_1Q: float
        Gate time in user-specified units for all operations acting on one qubit. Either `gate_time_dict`
        or both `gate_time_1Q` and `gate_time_2Q` must be specified.
    
    gate_time_2Q: float
        Gate time in user-specified units for all operations acting on more than one qubit.
        Either `gate_time_dict` or both `gate_time_1Q` and `gate_time_2Q` must be specified.
    
    measure_reset_time: float
        Measurement and/or reset time in user-specified units. This is applied once for every circuit.
    
    interbatch_latency: float
        Time between batches in user-specified units.

    total_shots_per_circuit: int
        Total number of shots per circuit. Together with `shots_per_circuit_per_batch`, this will
        determine the total number of rounds needed.

    shots_per_circuit_per_batch: int
        Number of shots to do for each circuit within a batch. Together with `total_shots_per_circuit`,
        this will determine the total number of rounds needed. If None, this is set to the total shots,
        meaning that only one round is done.
    
    circuits_per_batch: int
        Number of circuits to include in each batch. Together with the number of circuits in `edesign`,
        this will determine the number of batches in each round. If None, this is set to the total number
        of circuits such that only one batch is done.

    Returns
    -------
    float
        The estimated time to run the experiment design. 
    """
    assert gate_time_dict is not None or \
        (gate_time_1Q is not None and gate_time_2Q is not None), \
        "Must either specify a gate_time_dict with entries for every gate name or label, " + \
        "or specify gate_time_1Q and gate_time_2Q for one-qubit and two-qubit gate times, respectively"
        
    def layer_time(layer):
        gate_times = []
        for comp in layer.components:
            if gate_time_dict is not None:
                # Use specific gate times for each gate
                comp_time = gate_time_dict.get(comp, None) # Start with most specific key first
                if comp_time is None:
                    comp_time = gate_time_dict.get(comp.name, None) # Try gate name only next

                assert comp_time is not None, f"Could not look up gate time for {comp}"
            else:
                # Use generic one/two qubit gate times
                comp_qubits = len(comp.sslbls)
                comp_time = gate_time_2Q if comp_qubits > 1 else gate_time_1Q
            
            gate_times.append(comp_time)
        
        if len(gate_times) == 0:
            return 0
        
        return max(gate_times)
    
    total_circ_time = 0.0
    for circ in edesign.all_circuits_needing_data:
        circ_time = measure_reset_time + sum([layer_time(l) for l in circ])
        total_circ_time += circ_time * total_shots_per_circuit
    
    # Default assume all in one batch
    if circuits_per_batch is None: 
        circuits_per_batch = len(edesign.all_circuits_needing_data)
    
    # Default assume all in one round
    if shots_per_circuit_per_batch is None:
            shots_per_circuit_per_batch = total_shots_per_circuit
        
    num_rounds = math.ceil(total_shots_per_circuit / shots_per_circuit_per_batch)
    num_batches = math.ceil(len(edesign.all_circuits_needing_data) / circuits_per_batch)
    
    total_upload_time = interbatch_latency*num_batches*num_rounds
    
    return total_circ_time + total_upload_time