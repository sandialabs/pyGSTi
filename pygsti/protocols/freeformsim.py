"""
ModelTest Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from typing import Optional, Union, Tuple, Any, Dict, TYPE_CHECKING
import numpy as _np
import pandas as pd
if TYPE_CHECKING:
    from pygsti.models.model import Model as _Model

from pygsti.protocols import protocol as _proto
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.data.freedataset import FreeformDataSet as _FreeformDataSet
from pygsti.modelmembers import states as _state

class FreeformDataSimulator(_proto.DataSimulator):
    """
    Computes arbitrary functions of the state data simulator that also computes user-defined functions
    of the final states.
    """

    def __init__(self):
        super().__init__()

    def compute_freeform_data(self, circuit: _Circuit) -> dict:
        """
        Computes the simulated free-form data for a single circuit.

        Parameters
        ----------
        circuit : Circuit
            The circuit to compute data for.

        Returns
        -------
        dict
        """
        raise NotImplementedError("Derived classes should implement this!")

    def run(self, edesign: _proto.ExperimentDesign, memlimit: Optional[int] = None, comm=None) -> _proto.ProtocolData:
        """
        Run this data simulator on an experiment design.

        Parameters
        ----------
        edesign : ExperimentDesign
            The input experiment design.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this data
            simulator in parallel.

        Returns
        -------
        ProtocolData
        """
        dataset = _FreeformDataSet(circuits=edesign.all_circuits_needing_data)
        for c in edesign.all_circuits_needing_data:
            dataset[c] = self.compute_freeform_data(c)
        return _proto.ProtocolData(edesign, dataset)

    def apply_fn(self, series: 'pd.Series') -> 'pd.Series':
        circuit = _Circuit.cast(str(series['Circuit']))  # parse string circuit
        info = self.compute_freeform_data(circuit)
        return pd.Series(info)  # TODO FIX THIS

    def apply(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Apply this data simulator to a data frame having a `Circuit` column.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to apply to.

        Returns
        -------
        pandas.DataFrame
        """
        return pd.concat([df, df.apply(self.apply_fn, axis=1)], axis=1)


class ModelFreeformSimulator(FreeformDataSimulator):
    """
    A base class for data simulators that utilize models (probably most of them!).

    Holds a dictionary of models and provides basic functionality for computing
    probabilities, final states, and process matrices corresponding to circuits
    which make implementing :meth:`compute_freeform_data` easier.

    Parameters
    ----------
    models : dict
        A dictionary whose keys are string labels and values are :class:`Model` objects, specifying
        the models used to compute "simulated" data.
    """

    def __init__(self, models: Dict[str, _Model]):
        self.models = models

    def compute_process_matrix(self, model: _Model, circuit: _Circuit, include_final_state: bool = False, include_probabilities: bool = False) -> Union[_np.ndarray, Tuple[_np.ndarray, _state.StaticState, Any]]:
        prep, circuit_ops, povm = model.split_circuit(circuit)
        mx = model.sim.product(circuit_ops)
        if include_final_state or include_probabilities:
            ret = [mx]
            rho = model.circuit_layer_operator(prep, 'prep')
            final_state = _state.StaticState(_np.dot(mx, rho.to_dense("HilbertSchmidt")),
                                             model.basis, model.evotype, model.state_space)
            if include_final_state:
                ret.append(final_state)
            if include_probabilities:
                M = model.circuit_layer_operator(povm, 'povm')
                probs = M.acton(final_state)
                ret.append(probs)
            return tuple(ret)
        else:
            return mx

    def compute_process_matrices(self, circuit: _Circuit, include_final_state: bool = False, include_probabilities: bool = False) -> dict:
        return {model_lbl: self.compute_process_matrix(model, circuit, include_final_state, include_probabilities)
                for model_lbl, model in self.models.items()}

    def compute_final_state(self, model: _Model, circuit: _Circuit, include_probabilities: bool = False) -> Union[_state.State, Tuple[_state.State, Any]]:
        complete_circuit = model.complete_circuit(circuit).layertup
        rho = model.circuit_layer_operator(complete_circuit[0], 'prep')
        for layer in complete_circuit[1:-1]:
            layerop = model.circuit_layer_operator(layer, 'op')
            rho = layerop.acton(rho)
        if include_probabilities:
            M = model.circuit_layer_operator(complete_circuit[-1], 'povm')
            probs = M.acton(rho)
            return rho, probs
        return rho

    def compute_final_states(self, circuit: _Circuit, include_probabilities: bool = False) -> dict:
        return {model_lbl: self.compute_final_state(model, circuit, include_probabilities)
                for model_lbl, model in self.models.items()}

    def compute_circuit_probabilities(self, model: _Model, circuit: _Circuit) -> _np.ndarray:
        # FUTURE: add a flag in __init__ (?) for computing bulk probabilities at the beginning of
        # run(...) (we'll need to overload run for this) and then this function just indexes the
        # precomputed values.
        return model.probabilities(circuit)

    def compute_probabilities(self, circuit: _Circuit, include_probabilities: bool = False) -> dict:
        return {model_lbl: self.compute_circuit_probabilities(model, circuit)
                for model_lbl, model in self.models.items()}
