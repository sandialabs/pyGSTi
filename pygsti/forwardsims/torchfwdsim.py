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

    def __init__(self, spc: SeparatePOVMCircuit):
        self.prep_label = spc.circuit_without_povm[0]
        self.op_labels  = spc.circuit_without_povm[1:]
        self.povm_label = spc.povm_label

        # prep = model.circuit_layer_operator(self.prep_label, typ='prep')
        # povm = model.circuit_layer_operator(self.povm_label, 'povm')
        # self.input_dim  = prep.dim
        # self.output_dim = len(povm)

        # self.prep_type = type(prep)
        """ ^
        <class 'pygsti.modelmembers.states.tpstate.TPState'>
        <class 'pygsti.modelmembers.states.densestate.DenseState'>
        <class 'pygsti.modelmembers.states.densestate.DenseStateInterface'>
        <class 'pygsti.modelmembers.states.state.State'>
        <class 'pygsti.modelmembers.modelmember.ModelMember'>
        """
        # self.op_types = OrderedDict()
        # for ol in self.op_labels:
        #     self.op_types[ol] = type(model.circuit_layer_operator(ol, 'op'))
        """ ^ For reasons that I don't understand, this is OFTEN an empty list
        in the first step of iterative GST. When it's nonempty, it contains ...
    
        <class 'pygsti.modelmembers.operations.fulltpop.FullTPOp'>
        <class 'pygsti.modelmembers.operations.denseop.DenseOperator'>
        <class 'pygsti.modelmembers.operations.denseop.DenseOperatorInterface'>
        <class 'pygsti.modelmembers.operations.krausop.KrausOperatorInterface'>
        <class 'pygsti.modelmembers.operations.linearop.LinearOperator'>
        <class 'pygsti.modelmembers.modelmember.ModelMember'>
        """
        # self.povm_type = type(povm)
        """
        <class 'pygsti.modelmembers.povms.tppovm.TPPOVM'>
        <class 'pygsti.modelmembers.povms.basepovm._BasePOVM'>
        <class 'pygsti.modelmembers.povms.povm.POVM'>
            <class 'pygsti.modelmembers.modelmember.ModelMember'>
            <class 'collections.OrderedDict'>
                keyed by effectlabels and ConjugatedStatePOVMEffect-valued
        """
        return


class StatelessModel:

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
            typestr = str(param_type)
            if 'TPPOVM' in typestr:
                param_data = (lbl, param_type, len(obj), obj.dim)
            elif 'FullTPOp' in typestr:
                param_data = (lbl, param_type, obj.dim)
            elif 'TPState' in typestr:
                param_data = (lbl, param_type, obj.dim)
            else:
                raise ValueError()
            self.param_metadata.append(param_data)
        self.num_params = len(self.param_metadata)
        return
    
    def get_free_parameters(self, model: ExplicitOpModel):
        d = OrderedDict()
        for i, (lbl, obj) in enumerate(model._iter_parameterized_objs()):
            gpind = obj.gpindices_as_array()
            vec = obj.to_vector()
            vec = torch.from_numpy(vec)
            assert int(gpind.size) == int(np.prod(vec.shape))
            assert self.param_metadata[i][0] == lbl
            d[lbl] = vec
        return d

    def get_torch_cache(self, free_params: OrderedDict[Label, torch.Tensor], grad: bool):
        torch_cache = dict()
        for i, fp_val in enumerate(free_params.values()):

            if grad: fp_val.requires_grad_(True)
            metadata = self.param_metadata[i]
            fp_label = metadata[0]
            fp_type  = metadata[1]
            fp_tstr  = str(fp_type)

            if ('FullTPOp' in fp_tstr) or ('TPState' in fp_tstr):
                param_t = fp_type.torch_base(metadata[2], fp_val)
            elif 'TPPOVM' in fp_tstr:
                param_t = fp_type.torch_base(metadata[2], metadata[3], fp_val)
            else:
                raise ValueError()
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
    
    def functional_circuit_probs(self, *free_params: Tuple[torch.Tensor]):
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

    def _bulk_fill_probs_block(self, array_to_fill, layout, stripped_abstractions: Optional[tuple] = None):
        if stripped_abstractions is None:
            slm, torch_cache = TorchForwardSimulator.separate_state(self.model, layout)
        else:
            slm, torch_cache = stripped_abstractions

        layout_len = TorchForwardSimulator._check_copa_layout(layout)
        probs = slm.circuit_probs(torch_cache)
        array_to_fill[:layout_len] = probs.cpu().detach().numpy().flatten()
        pass

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill):
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout)
        return self._bulk_fill_dprobs_block(array_to_fill, layout)

    def _bulk_fill_dprobs_block(self, array_to_fill, layout):
        from torch.func import jacfwd

        slm = StatelessModel(self.model, layout)
        argnums = tuple(range(slm.num_params))
        J_func = jacfwd(slm.functional_circuit_probs, argnums=argnums)

        free_params = slm.get_free_parameters(self.model)
        free_param_tup = tuple(free_params.values())

        J_val = J_func(*free_param_tup)
        J_val = torch.column_stack(J_val)
        J_np = J_val.cpu().detach().numpy()
        array_to_fill[:] = J_np
        return


"""
Running GST produces the following traceback if I set a breakpoint inside the
loop over expanded_circuit_outcomes.items() in self._compute_circuit_outcome_probabilities(...).

I think something's happening where accessing the objects here (via the debugger)
makes some object set "self.dirty=True" for the ComplementPOVMEffect.

UPDATE
    The problem shows up when we try to access effect.base for some FullPOVMEffect object "effect".
CONFIRMED
    FullPOVMEffect resolves an attempt to access to .base attribute by a default implementation
    in its DenseEffectInterface subclass. The last thing that function does is set
    self.dirty = True. 

    pyGSTi/pygsti/forwardsims/forwardsim.py:562: in _bulk_fill_probs_block
        self._compute_circuit_outcome_probabilities(array_to_fill[element_indices], circuit,
    pyGSTi/pygsti/forwardsims/torchfwdsim.py:177: in _compute_circuit_outcome_probabilities
        if povmrep is None:
    pyGSTi/pygsti/forwardsims/torchfwdsim.py:177: in <listcomp>
        if povmrep is None:
    pyGSTi/pygsti/models/model.py:1479: in circuit_layer_operator
        self._clean_paramvec()
    pyGSTi/pygsti/models/model.py:679: in _clean_paramvec
        clean_obj(obj, lbl)
    pyGSTi/pygsti/models/model.py:675: in clean_obj
        clean_obj(subm, _Label(lbl.name + ":%d" % i, lbl.sslbls))
    pyGSTi/pygsti/models/model.py:676: in clean_obj
        clean_single_obj(obj, lbl)
    pyGSTi/pygsti/models/model.py:666: in clean_single_obj
        w = obj.to_vector()
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    self = <pygsti.modelmembers.povms.complementeffect.ComplementPOVMEffect object at 0x2a79e31f0>

        def to_vector(self):
            '''<Riley removed comment block>'''
    >       raise ValueError(("ComplementPOVMEffect.to_vector() should never be called"
                            " - use TPPOVM.to_vector() instead"))
    E       ValueError: ComplementPOVMEffect.to_vector() should never be called - use TPPOVM.to_vector() instead

"""
