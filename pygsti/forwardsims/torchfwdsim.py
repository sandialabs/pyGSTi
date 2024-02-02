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
from typing import Tuple, Optional, TypeVar
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


class TorchForwardSimulator(ForwardSimulator):
    """
    A forward simulator that leverages automatic differentiation in PyTorch.
    (The current work-in-progress implementation has no Torch functionality whatsoever.)
    """
    def __init__(self, model : Optional[ExplicitOpModel] = None):
        if not TORCH_ENABLED:
            raise RuntimeError('PyTorch could not be imported.')
        self.model = model
        super(ForwardSimulator, self).__init__(model)

    @staticmethod
    def _build_torch_cache(model: ExplicitOpModel, layout):
        tc = dict()
        for _, circuit, outcomes in layout.iter_unique_circuits():
            expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(model, outcomes)
            # ^ Note, I'm not sure if outcomes needs to be passed to the function above.
            if len(expanded_circuit_outcomes) > 1:
                raise NotImplementedError("I don't know what to do with this.")
            spc = list(expanded_circuit_outcomes.keys())[0]

            prep_label = spc.circuit_without_povm[0]
            op_labels  = spc.circuit_without_povm[1:]
            povm_label = spc.povm_label

            rho = model.circuit_layer_operator(prep_label, typ='prep')
            """ ^
            <class 'pygsti.modelmembers.states.tpstate.TPState'>
            <class 'pygsti.modelmembers.states.densestate.DenseState'>
            <class 'pygsti.modelmembers.states.densestate.DenseStateInterface'>
            <class 'pygsti.modelmembers.states.state.State'>
            <class 'pygsti.modelmembers.modelmember.ModelMember'>
            """
            ops = [model.circuit_layer_operator(ol, 'op') for ol in op_labels]
            """ ^ For reasons that I don't understand, this is OFTEN an empty list
            in the first step of iterative GST. When it's nonempty, it contains ...
        
            <class 'pygsti.modelmembers.operations.fulltpop.FullTPOp'>
            <class 'pygsti.modelmembers.operations.denseop.DenseOperator'>
            <class 'pygsti.modelmembers.operations.denseop.DenseOperatorInterface'>
            <class 'pygsti.modelmembers.operations.krausop.KrausOperatorInterface'>
            <class 'pygsti.modelmembers.operations.linearop.LinearOperator'>
            <class 'pygsti.modelmembers.modelmember.ModelMember'>
            """
            povm = model.circuit_layer_operator(povm_label, 'povm')
            """
            <class 'pygsti.modelmembers.povms.tppovm.TPPOVM'>
            <class 'pygsti.modelmembers.povms.basepovm._BasePOVM'>
            <class 'pygsti.modelmembers.povms.povm.POVM'>
                <class 'pygsti.modelmembers.modelmember.ModelMember'>
                <class 'collections.OrderedDict'>
                    keyed by effectlabels and ConjugatedStatePOVMEffect-valued
            """

            # Get the numerical representations
            #   Right now I have a very awkward switch for gradients used in debugging.
            require_grad = True
            superket_data = rho.torch_base(require_grad,  torch_handle=torch)
            superops_data = [op.torch_base(require_grad,  torch_handle=torch) for op in ops]
            povm_mat_data = povm.torch_base(require_grad, torch_handle=torch)

            tc[prep_label] = superket_data
            for i, ol in enumerate(op_labels):
                tc[ol] = superops_data[i]
            tc[povm_label] = povm_mat_data

        return tc

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
    
    def _all_circuit_probs(self, layout, torch_cache):
        probs = []
        for _, circuit, outcomes in layout.iter_unique_circuits():
            # Step 1. Get the SeparatePOVMCircuit associated with this Circuit object.
            #
            #    Right now I only know how to do this by getting a function that returns
            #    a dict with the SeparatePOVMCircuit as a key. I don't know how I would
            #    be expected to process the dict if it contained more than one key-value
            #    pair, so I'm making some assertions before just unpacking the dict.
            #
            spc_dict = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
            assert len(spc_dict) == 1
            spc, val = next(iter(spc_dict.items()))
            assert val == outcomes
            #
            # Step 2. Use the SeparatePOVMCircuit and torch_cache to compute outcome probabilities.
            #
            circuit_probs = TorchForwardSimulator._circuit_probs(spc, torch_cache)
            probs.append(circuit_probs)
        probs = torch.concat(probs)
        return probs

    def _bulk_fill_probs_block(self, array_to_fill, layout, torch_cache: Optional[dict] = None):
        if torch_cache is None:
            torch_cache = TorchForwardSimulator._build_torch_cache(self.model, layout)
        else:
            assert isinstance(torch_cache, dict)
        layout_len = TorchForwardSimulator._check_copa_layout(layout)
        probs = self._all_circuit_probs(layout, torch_cache)
        array_to_fill[:layout_len] = probs.cpu().detach().numpy().flatten()
        pass

    @staticmethod
    def _circuit_probs(spc: SeparatePOVMCircuit, torch_cache):
        prep_label = spc.circuit_without_povm[0]
        op_labels  = spc.circuit_without_povm[1:]
        povm_label = spc.povm_label

        superket = torch_cache[prep_label][0]
        superops = [torch_cache[ol][0] for ol in op_labels]
        povm_mat = torch_cache[povm_label][0]

        for superop in superops:
            superket = superop @ superket
        probs = povm_mat @ superket
        return probs
    
    @staticmethod
    def _get_jac_bookkeeping_dict(model: ExplicitOpModel, torch_cache):
        d = OrderedDict()
        for lbl, obj in model._iter_parameterized_objs():
            d[lbl] = (obj.gpindices_as_array(), torch_cache[lbl][1])
        return d

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill):
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout)
        return self._bulk_fill_dprobs_block(array_to_fill, layout)

    def _bulk_fill_dprobs_block(self, array_to_fill, layout):
        from torch.func import jacfwd
        probs = np.empty(len(layout), 'd')
        torch_cache = TorchForwardSimulator._build_torch_cache(self.model, layout)
        self._bulk_fill_probs_block(probs, layout, torch_cache)
        
        # jacbook = TorchForwardSimulator._get_jac_bookkeeping_dict(self.model, torch_cache)
        # tprobs = self._all_circuit_probs(layout, torch_cache)

        # num_torch_param_obj = len(jacbook)
        # jac_handle = jacfwd(tprobs, )
        # cur_params = machine.params
        # num_params = len(cur_params)
        # J_func = jacfwd(machine.circuit_outcome_probs, argnums=tuple(range(num_params)))
        # J = J_func(*cur_params)

        probs2 = np.empty(len(layout), 'd')
        orig_vec = self.model.to_vector()
        orig_vec = orig_vec.copy()
        FIN_DIFF_EPS = 1e-7
        for i in range(self.model.num_params):
            vec = orig_vec.copy(); vec[i] += FIN_DIFF_EPS
            self.model.from_vector(vec, close=True)
            self._bulk_fill_probs_block(probs2, layout)
            array_to_fill[:, i] = (probs2 - probs) / FIN_DIFF_EPS

        self.model.from_vector(orig_vec, close=True)

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
