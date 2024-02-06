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

from collections import OrderedDict
import warnings as warnings
from typing import Tuple, Optional, TypeVar, Union, List, Dict
import importlib as _importlib
import warnings as _warnings
from pygsti.tools import slicetools as _slct

import numpy as np
import scipy.linalg as la
try:
    import torch
    TORCH_ENABLED = True
except ImportError:
    TORCH_ENABLED = False

from pygsti.forwardsims.forwardsim import ForwardSimulator

# Below: imports only needed for typehints
from pygsti.circuits import Circuit
from pygsti.baseobjs.resourceallocation import ResourceAllocation
Label = TypeVar('Label')
ExplicitOpModel = TypeVar('ExplicitOpModel')
SeparatePOVMCircuit = TypeVar('SeparatePOVMCircuit')
CircuitOutcomeProbabilityArrayLayout = TypeVar('CircuitOutcomeProbabilityArrayLayout')
# ^ declare to avoid circular references


"""
Proposal:
   There are lots of places where we use np.dot in the codebase.
   I think we're much better off replacing with the @ operator
   unless we're using the "out" keyword of np.dot. Reason being:
   different classes of ndarray-like objects (like pytorch Tensors!)
   overload @ in whatever way that they need.
"""

"""Efficiency ideas
 * Compute the jacobian in blocks of rows at a time (iterating over the blocks in parallel). Ideally pytorch
   would recognize how the computations decompose, but we should check to make sure it does.

 * Recycle some of the work in setting up the Jacobian function.
    Calling circuit.expand_instruments_and_separate_povm(model, outcomes) inside the StatelessModel constructor
    might be expensive. It only need to happen once during an iteration of GST.
"""

class StatelessCircuit:
    """
    Helper data structure useful for simulating a specific circuit quantum (including prep,
    applying a sequence of gates, and applying a POVM to the output of the last gate).
    
    The forward simulation can only be done when we have access to a dict that maps
    pyGSTi Labels to certain PyTorch Tensors.
    """

    def __init__(self, spc: SeparatePOVMCircuit):
        self.prep_label = spc.circuit_without_povm[0]
        self.op_labels  = spc.circuit_without_povm[1:]
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
            expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(model, outcomes)
            if len(expanded_circuit_outcomes) > 1:
                raise NotImplementedError("I don't know what to do with this.")
            spc = list(expanded_circuit_outcomes.keys())[0]
            c = StatelessCircuit(spc)
            circuits.append(c)
        self.circuits = circuits

        self.param_metadata = []
        for lbl, obj in model._iter_parameterized_objs():
            param_type = type(obj)
            param_data = (lbl, param_type) + (obj.stateless_data(),)
            self.param_metadata.append(param_data)
        self.num_params = len(self.param_metadata)
        return
    
    def get_free_parameters(self, model: ExplicitOpModel):
        """
        Return an ordered dict that maps pyGSTi Labels to PyTorch Tensors.
        The Labels correspond to parameterized objects in "model".
        The Tensors correspond to the current values of an object's parameters.
        For the purposes of forward simulation, we intend that the following 
        equivalence holds:
        
            model == (self, [dict returned by this function]).
            
        That said, the values in this function's returned dict need to be
        formatted by get_torch_cache BEFORE being used in forward simulation.
        """
        free_params = OrderedDict()
        for i, (lbl, obj) in enumerate(model._iter_parameterized_objs()):
            gpind = obj.gpindices_as_array()
            vec = obj.to_vector()
            vec = torch.from_numpy(vec)
            assert int(gpind.size) == int(np.prod(vec.shape))
            # ^ a sanity check that we're interpreting the results of obj.to_vector()
            #   correctly. Future implementations might need us to also keep track of
            #   the "gpind" variable. Right now we get around NOT using that variable
            #   by using an OrderedDict and by iterating over parameterized objects in
            #   the same way that "model"s does.
            assert self.param_metadata[i][0] == lbl
            # ^ If this check fails then it invalidates our assumptions about how
            #   we're using OrderedDict objects.
            free_params[lbl] = vec
        return free_params

    def get_torch_cache(self, free_params: OrderedDict[Label, torch.Tensor], grad: bool):
        """
        Returns a dict mapping pyGSTi Labels to PyTorch tensors. The dict makes it easy
        to simulate a stateful model implied by (self, free_params). It is obtained by
        applying invertible transformations --- defined in various ModelMember subclasses
        --- on the tensors stored in free_params.

        If ``grad`` is True, then the values in the returned dict are preparred for use
        in PyTorch's backpropogation functionality. If we want to compute a Jacobian of
        circuit outcome probabilities then such functionality is actually NOT needed.
        Therefore for purposes of computing Jacobians this should be set to False.
        """
        torch_cache = dict()
        for i, fp_val in enumerate(free_params.values()):

            if grad: fp_val.requires_grad_(True)
            metadata = self.param_metadata[i]
            
            fp_label = metadata[0]
            fp_type  = metadata[1]
            param_t = fp_type.torch_base(metadata[2], fp_val)
            torch_cache[fp_label] = param_t
        
        return torch_cache

    def circuit_probs(self, torch_cache: Dict[Label, torch.Tensor]):
        probs = []
        for c in self.circuits:
            superket = torch_cache[c.prep_label]
            superops = [torch_cache[ol] for ol in c.op_labels]
            povm_mat = torch_cache[c.povm_label]
            for superop in superops:
                superket = superop @ superket
            circuit_probs = povm_mat @ superket
            probs.append(circuit_probs)
        probs = torch.concat(probs)
        return probs
    
    def jac_friendly_circuit_probs(self, *free_params: Tuple[torch.Tensor]):
        """
        This function combines parameter reformatting and forward simulation.
        It's needed so that we can use PyTorch to compute the Jacobian of
        the map from a model's free parameters to circuit outcome probabilities.
        """
        assert len(free_params) == len(self.param_metadata) == self.num_params
        free_params = {self.param_metadata[i][0] : free_params[i] for i in range(self.num_params)}
        torch_cache = self.get_torch_cache(free_params, grad=False)
        probs = self.circuit_probs(torch_cache)
        return probs


class TorchForwardSimulator(ForwardSimulator):
    """
    A forward simulator that leverages automatic differentiation in PyTorch.
    """
    def __init__(self, model : Optional[ExplicitOpModel] = None):
        if not TORCH_ENABLED:
            raise RuntimeError('PyTorch could not be imported.')
        self.model = model
        super(ForwardSimulator, self).__init__(model)

    @staticmethod
    def separate_state(model: ExplicitOpModel, layout, grad=False):
        slm = StatelessModel(model, layout)
        free_params = slm.get_free_parameters(model)
        torch_cache = slm.get_torch_cache(free_params, grad)
        return slm, torch_cache

    @staticmethod
    def _check_copa_layout(layout: CircuitOutcomeProbabilityArrayLayout):
        # I need to verify some assumptions on what layout.iter_unique_circuits()
        # returns. Looking at the implementation of that function, the assumptions
        # can be framed in terms of the "layout._element_indicies" OrderedDict.
        eind = layout._element_indices
        assert isinstance(eind, OrderedDict)
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

    def _bulk_fill_probs(self, array_to_fill, layout, stripped_abstractions: Optional[tuple] = None):
        if stripped_abstractions is None:
            slm, torch_cache = TorchForwardSimulator.separate_state(self.model, layout)
        else:
            slm, torch_cache = stripped_abstractions

        layout_len = TorchForwardSimulator._check_copa_layout(layout)
        probs = slm.circuit_probs(torch_cache)
        array_to_fill[:layout_len] = probs.cpu().detach().numpy().flatten()
        pass

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill):
        slm = StatelessModel(self.model, layout)
        free_params = slm.get_free_parameters(self.model)
        torch_cache = slm.get_torch_cache(free_params, grad=False)
        if pr_array_to_fill is not None:
            self._bulk_fill_probs(pr_array_to_fill, layout, (slm, torch_cache))

        argnums = tuple(range(slm.num_params))
        J_func = torch.func.jacfwd(slm.jac_friendly_circuit_probs, argnums=argnums)
        free_param_tup = tuple(free_params.values())
        J_val = J_func(*free_param_tup)
        J_val = torch.column_stack(J_val)
        J_np = J_val.cpu().detach().numpy()
        array_to_fill[:] = J_np
        return
