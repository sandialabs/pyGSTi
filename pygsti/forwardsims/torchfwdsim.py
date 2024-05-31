"""
Defines the TorchForwardSimulator class
"""
#***************************************************************************************************
# Copyright 2024, National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from typing import Tuple, Optional, Dict, TYPE_CHECKING, TypeAlias
if TYPE_CHECKING:
    from pygsti.baseobjs.label import Label
    from pygsti.models.explicitmodel import ExplicitOpModel
    from pygsti.circuits.circuit import SeparatePOVMCircuit
    from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout
    import torch

import warnings as warnings
from pygsti.modelmembers.torchable import Torchable
from pygsti.forwardsims.forwardsim import ForwardSimulator

try:
    import torch
    TORCH_ENABLED = True
except ImportError:
    TORCH_ENABLED = False
    pass

class StatelessCircuit:
    """
    Helper data structure useful for simulating a specific circuit quantum (including prep,
    applying a sequence of gates, and applying a POVM to the output of the last gate).
    
    The forward simulation can only be done when we have access to a dict that maps
    pyGSTi Labels to certain PyTorch Tensors.
    """

    def __init__(self, spc: SeparatePOVMCircuit):
        self.prep_label = spc.circuit_without_povm[0]
        if len(spc.circuit_without_povm) > 1:
            self.op_labels  = spc.circuit_without_povm[1:]
        else:
            # Importing this at the top of the file would create a circular
            # dependency.
            from pygsti.circuits.circuit import Circuit
            self.op_labels = Circuit(tuple())
        self.povm_label = spc.povm_label
        return


class StatelessModel:
    """
    A container for the information in an ExplicitOpModel that's "stateless"
    in the sense of object-oriented programming.
    
        Currently, that information is just specifications of the model's
        circuits, and model parameter metadata.

    StatelessModels have functions to (1) extract stateful data from an
    ExplicitOpModel, (2) reformat that data into particular PyTorch
    Tensors, and (3) run the forward simulation using that data. There
    is also a function that combines (2) and (3).
    """

    def __init__(self, model: ExplicitOpModel, layout):
        circuits = []
        for _, circuit, outcomes in layout.iter_unique_circuits():
            expanded_circuits = circuit.expand_instruments_and_separate_povm(model, outcomes)
            if len(expanded_circuits) > 1:
                raise NotImplementedError("I don't know what to do with this.")
            spc = next(iter(expanded_circuits))
            c = StatelessCircuit(spc)
            circuits.append(c)
        self.circuits = circuits

        self.param_metadata = []
        for lbl, obj in model._iter_parameterized_objs():
            assert isinstance(obj, Torchable)
            param_type = type(obj)
            param_data = (lbl, param_type) + (obj.stateless_data(),)
            self.param_metadata.append(param_data)
        self.num_parameterized = len(self.param_metadata)
        return
    
    def extract_free_parameters(self, model: ExplicitOpModel) -> Tuple[torch.Tensor]:
        """
        Return a dict mapping pyGSTi Labels to PyTorch Tensors.

            The Labels correspond to parameterized objects in "model".
            The Tensors correspond to the current values of an object's parameters.
        
        For the purposes of forward simulation, we intend that the following 
        equivalence holds:
        
            model == (self, [dict returned by this function]).
            
        That said, the values in this function's returned dict need to be
        formatted by get_torch_bases BEFORE being used in forward simulation.
        """
        free_params = []
        prev_idx = 0
        for i, (lbl, obj) in enumerate(model._iter_parameterized_objs()):
            gpind = obj.gpindices_as_array()
            vec = obj.to_vector()
            vec_size = vec.size
            vec = torch.from_numpy(vec)
            assert gpind[0] == prev_idx and gpind[-1] == prev_idx + vec_size - 1
            # ^ We should have gpind = (prev_idx, prev_idx + 1, ..., prev_idx + vec.size - 1).
            #   That assert checks a cheap necessary condition that this holds.
            prev_idx += vec_size
            assert self.param_metadata[i][0] == lbl
            # ^ This function's output inevitably gets passed to StatelessModel.get_torch_bases.
            #   That function assumes that the keys we're seeing here are the same (and in the
            #   same order!) as those seen when we constructed this StatelessModel.
            free_params.append(vec)
    
        return tuple(free_params)

    def get_torch_bases(self, free_params: Tuple[torch.Tensor], grad: bool) -> Dict[Label, torch.Tensor]:
        """
        Returns a dict that circuit_probs(...) needs for forward simulation.

        Notes
        -----
        If ``grad`` is True, then the values in the returned dict are preparred for use
        in PyTorch's backpropogation functionality. If we want to compute a Jacobian of
        circuit outcome probabilities then such functionality is actually NOT needed.
        Therefore for purposes of computing Jacobians this should be set to False.
        """
        torch_bases = dict()
        for i, val in enumerate(free_params):
            if grad:
                val.requires_grad_(True)

            label, type_handle, stateless_data = self.param_metadata[i]
            param_t = type_handle.torch_base(stateless_data, val)
            torch_bases[label] = param_t
        
        return torch_bases

    def circuit_probs(self, torch_bases: Dict[Label, torch.Tensor]) -> torch.Tensor:
        probs = []
        for c in self.circuits:
            superket = torch_bases[c.prep_label]
            superops = [torch_bases[ol] for ol in c.op_labels]
            povm_mat = torch_bases[c.povm_label]
            for superop in superops:
                superket = superop @ superket
            circuit_probs = povm_mat @ superket
            probs.append(circuit_probs)
        probs = torch.concat(probs)
        return probs
    
    def jac_friendly_circuit_probs(self, *free_params: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        This function combines parameter reformatting and forward simulation.
        It's needed so that we can use PyTorch to compute the Jacobian of
        the map from a model's free parameters to circuit outcome probabilities.
        """
        assert len(free_params) == len(self.param_metadata) == self.num_parameterized
        torch_bases = self.get_torch_bases(free_params, grad=False)
        probs = self.circuit_probs(torch_bases)
        return probs


if TYPE_CHECKING:
    SplitModel: TypeAlias = Tuple[StatelessModel, Dict[Label, torch.Tensor]]


class TorchForwardSimulator(ForwardSimulator):

    ENABLED = TORCH_ENABLED

    """
    A forward simulator that leverages automatic differentiation in PyTorch.
    """
    def __init__(self, model : Optional[ExplicitOpModel] = None):
        if not self.ENABLED:
            raise RuntimeError('PyTorch could not be imported.')
        self.model = model
        super(ForwardSimulator, self).__init__(model)

    @staticmethod
    def get_split_model(model: ExplicitOpModel, layout, grad=False) -> SplitModel:
        slm = StatelessModel(model, layout)
        free_params = slm.extract_free_parameters(model)
        torch_bases = slm.get_torch_bases(free_params, grad)
        return slm, torch_bases

    @staticmethod
    def _check_copa_layout(layout: CircuitOutcomeProbabilityArrayLayout) -> int:
        # I need to verify some assumptions on what layout.iter_unique_circuits()
        # returns. Looking at the implementation of that function, the assumptions
        # can be framed in terms of the "layout._element_indicies" dict.
        eind = layout._element_indices
        assert isinstance(eind, dict)
        items = iter(eind.items())
        k_prev, v_prev = next(items)
        assert k_prev == 0
        assert v_prev.start == 0
        for k, v in items:
            assert k == k_prev + 1
            assert v.start == v_prev.stop
            k_prev = k
            v_prev = v
        return v_prev.stop

    def _bulk_fill_probs(self, array_to_fill, layout, splitm: Optional[SplitModel] = None) -> None:
        if splitm is None:
            slm, torch_bases = TorchForwardSimulator.get_split_model(self.model, layout)
        else:
            slm, torch_bases = splitm

        layout_len = TorchForwardSimulator._check_copa_layout(layout)
        probs = slm.circuit_probs(torch_bases)
        array_to_fill[:layout_len] = probs.cpu().detach().numpy().ravel()
        return

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill) -> None:
        slm = StatelessModel(self.model, layout)
        free_params = slm.extract_free_parameters(self.model)
    
        if pr_array_to_fill is not None:
            torch_bases = slm.get_torch_bases(free_params, grad=False)
            splitm = (slm, torch_bases)
            self._bulk_fill_probs(pr_array_to_fill, layout, splitm)

        argnums = tuple(range(slm.num_parameterized))
        J_func = torch.func.jacfwd(slm.jac_friendly_circuit_probs, argnums=argnums)
        J_val = J_func(*free_params)
        J_val = torch.column_stack(J_val)
        array_to_fill[:] = J_val.cpu().detach().numpy()
        return
