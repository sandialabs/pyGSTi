"""
Defines the WeakForwardSimulator calculator class
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

from .forwardsim import ForwardSimulator as _ForwardSimulator


class WeakForwardSimulator(_ForwardSimulator):
    """
    A calculator of circuit outcome probabilities from a "weak" forward simulator
    (i.e. probabilites taken as average frequencies over a number of "shots").
    """

    def __init__(self, shots, model=None):
        """
        Construct a new WeakForwardSimulator object.

        Parameters
        ----------
        shots: int
            Number of times to run each circuit to obtain an approximate probability
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self.shots = shots
        super().__init__(model)

    def _compute_circuit_outcome_for_shot(self, array_to_fill, spc_circuit, spc_outcomes, resource_alloc, time=None):
        raise NotImplementedError("Derived classes should implement this!")

    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        work_array = _np.zeros_like(array_to_fill)

        expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        for spc, spc_outcomes in expanded_circuit_outcomes.items():  # spc is a SeparatePOVMCircuit
            indices = [outcome_to_index[o] for o in spc_outcomes]

            # TODO: For parallelization, block over this for loop
            for _ in range(self.shots):
                self._compute_circuit_outcome_for_shot(work_array, spc, spc_outcomes, resource_alloc, time)

            array_to_fill[indices[:]] = work_array / self.shots
    
    # If _compute_circuit_outcome_probability_derivatives is not defined, ForwardSimulator will do it by finite difference



