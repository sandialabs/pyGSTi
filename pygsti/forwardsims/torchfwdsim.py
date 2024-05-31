"""
Defines a ForwardSimulator class called "TorchForwardSimulator" that can leverage the automatic
differentation features of PyTorch.

This file also defines two helper classes: StatelessCircuit and StatelessModel.

See also: pyGSTi/modelmembers/torchable.py.
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
from typing import Tuple, Optional, Dict, TYPE_CHECKING
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
    Helper data structure useful for simulating a specific quantum circuit (including prep,
    applying a sequence of gates, and applying a POVM to the output of the last gate).
    
    The forward simulation can only be done when we have access to a dict that maps
    pyGSTi Labels to certain PyTorch Tensors. Construction of that dict is managed by the
    StatelessModel class, defined below.
    """

    def __init__(self, spc: SeparatePOVMCircuit):
        self.prep_label = spc.circuit_without_povm[0]
        self.op_labels  = spc.circuit_without_povm[1:]
        self.povm_label = spc.povm_label
        self.outcome_probs_dim = len(spc.effect_labels)
        return
    
    def outcome_probs(self, torch_bases: Dict[Label, torch.Tensor]) ->  torch.Tensor:
        superket = torch_bases[self.prep_label]
        superops = [torch_bases[ol] for ol in self.op_labels]
        povm_mat = torch_bases[self.povm_label]
        for superop in superops:
            superket = superop @ superket
        probs = povm_mat @ superket
        return probs


class StatelessModel:
    """
    A container for the information in an ExplicitOpModel that's "stateless" in the sense of
    object-oriented programming:
    
        * A list of StatelessCircuits
        * Metadata for parameterized ModelMembers

    StatelessModels have instance functions to facilitate computation of (differentable!)
    circuit outcome probabilities.

    Design notes
    ------------
    Much of this functionality could be packed into the TorchForwardSimulator class.
    Keeping it separate from TorchForwardSimulator helps clarify that it uses none of
    the sophiciated machinery in TorchForwardSimulator's base class.
    """

    def __init__(self, model: ExplicitOpModel, layout: CircuitOutcomeProbabilityArrayLayout):
        circuits = []
        self.outcome_probs_dim = 0
        for _, circuit, outcomes in layout.iter_unique_circuits():
            expanded_circuits = circuit.expand_instruments_and_separate_povm(model, outcomes)
            if len(expanded_circuits) > 1:
                raise NotImplementedError("I don't know what to do with this.")
            spc = next(iter(expanded_circuits))
            c = StatelessCircuit(spc)
            circuits.append(c)
            self.outcome_probs_dim += c.outcome_probs_dim
        self.circuits = circuits

        # We need to verify assumptions on what layout.iter_unique_circuits() returns.
        # Looking at the implementation of that function, the assumptions can be
        # framed in terms of the "layout._element_indicies" dict.
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
        assert self.outcome_probs_dim == v_prev.stop

        self.param_metadata = []
        for lbl, obj in model._iter_parameterized_objs():
            assert isinstance(obj, Torchable)
            param_type = type(obj)
            param_data = (lbl, param_type) + (obj.stateless_data(),)
            self.param_metadata.append(param_data)
        self.params_dim = None 
        # ^ That's set in get_free_params.

        self.default_to_reverse_ad = None
        # ^ That'll be set to a boolean the next time that get_free_params is called.
        return
    
    def get_free_params(self, model: ExplicitOpModel) -> Tuple[torch.Tensor]:
        """
        Return a tuple of Tensors that encode the states of the provided model's ModelMembers
        (where "state" in meant the sense of  object-oriented programming).
        
        We compare the labels of the input model's ModelMembers to those of the model provided
        to StatelessModel.__init__(...). We raise an error if an inconsistency is detected.
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
            if self.param_metadata[i][0] != lbl:
                message = """
                The model passed to get_free_params has a qualitatively different structure from
                the model used to construct this StatelessModel. Specifically, the two models have
                qualitative differences in the output of "model._iter_parameterized_objs()".

                The presence of this structral difference essentially gaurantees that a subsequent
                call to get_torch_bases would silently fail, so we're forced to raise an error here.
                """
                raise ValueError(message)
            free_params.append(vec)
        self.params_dim = prev_idx
        self.default_to_reverse_ad = self.outcome_probs_dim < self.params_dim
        return tuple(free_params)

    def get_torch_bases(self, free_params: Tuple[torch.Tensor], grad: bool) -> Dict[Label, torch.Tensor]:
        """
        Take data of the kind produced by get_free_params and format it in the way required by
        StatelessCircuit.outcome_probs.
        """
        assert len(free_params) == len(self.param_metadata)
         # ^ A sanity check that we're being called with the correct number of arguments.
        torch_bases = dict()
        for i, val in enumerate(free_params):
            if grad:
                val.requires_grad_(True)

            label, type_handle, stateless_data = self.param_metadata[i]
            param_t = type_handle.torch_base(stateless_data, val)
            torch_bases[label] = param_t
        
        return torch_bases

    def circuit_probs_from_torch_bases(self, torch_bases: Dict[Label, torch.Tensor]) -> torch.Tensor:
        """
        Compute the circuit outcome probabilities that result when all of this StatelessModel's
        StatelessCircuits are run with data in torch_bases.
    
        Return the results as a single (vectorized) torch Tensor.
        """
        probs = []
        for c in self.circuits:
            circuit_probs = c.outcome_probs(torch_bases)
            probs.append(circuit_probs)
        probs = torch.concat(probs)
        return probs
    
    def circuit_probs_from_free_params(self, *free_params: Tuple[torch.Tensor], require_reverse_ad=False) -> torch.Tensor:
        """
        This is the basic function we expose to pytorch for automatic differentiation. It returns the circuit
        outcome probabilities resulting when the states of ModelMembers associated with this StatelessModel
        are set according to data in free_params. 

        If (require_reverse_ad or self.default_to_reverse_ad) == True, then the returned Tensor can be used
        in pytorch's reverse-mode automatic differentiation.
        """
        enable_backprop = require_reverse_ad or self.default_to_reverse_ad
        torch_bases = self.get_torch_bases(free_params, grad=enable_backprop)
        probs = self.circuit_probs_from_torch_bases(torch_bases)
        return probs


class TorchForwardSimulator(ForwardSimulator):
    """
    A forward simulator that leverages automatic differentiation in PyTorch.
    """

    ENABLED = TORCH_ENABLED

    def __init__(self, model : Optional[ExplicitOpModel] = None):
        if not self.ENABLED:
            raise RuntimeError('PyTorch could not be imported.')
        self.model = model
        super(ForwardSimulator, self).__init__(model)

    def _bulk_fill_probs(self, array_to_fill, layout, split_model = None) -> None:
        if split_model is None:
            slm = StatelessModel(self.model, layout)
            free_params = slm.get_free_params(self.model)
            torch_bases = slm.get_torch_bases(free_params, grad=False)
        else:
            slm, torch_bases = split_model

        probs = slm.circuit_probs_from_torch_bases(torch_bases)
        array_to_fill[:slm.outcome_probs_dim] = probs.cpu().detach().numpy().ravel()
        return

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill) -> None:
        slm = StatelessModel(self.model, layout)
        # ^ TODO: figure out how to safely recycle StatelessModel objects from one
        #   call to another. The current implementation is wasteful if we need to 
        #   compute many jacobians without structural changes to layout or self.model.
        free_params = slm.get_free_params(self.model)
    
        if pr_array_to_fill is not None:
            torch_bases = slm.get_torch_bases(free_params, grad=False)
            splitm = (slm, torch_bases)
            self._bulk_fill_probs(pr_array_to_fill, layout, splitm)

        argnums = tuple(range(len(slm.param_metadata)))
        if slm.default_to_reverse_ad:
            # Then slm.circuit_probs_from_free_params will automatically construct the
            # torch_base dict to support reverse-mode AD.
            J_func = torch.func.jacrev(slm.circuit_probs_from_free_params, argnums=argnums)
        else:
            # Then slm.circuit_probs_from_free_params will automatically skip the extra
            # steps needed for torch_base to support reverse-mode AD.
            J_func = torch.func.jacfwd(slm.circuit_probs_from_free_params, argnums=argnums)
        # ^ Note that this _bulk_fill_dprobs function doesn't accept parameters that
        #   could be used to override the default behavior of the StatelessModel. If we
        #   have a need to override the default in the future then we'd need to override
        #   the ForwardSimulator function(s) that call self._bulk_fill_dprobs(...).

        J_val = J_func(*free_params)
        J_val = torch.column_stack(J_val)
        array_to_fill[:] = J_val.cpu().detach().numpy()
        return
