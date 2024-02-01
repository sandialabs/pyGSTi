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
        label_to_gate = dict()
        label_to_povm = dict()
        label_to_state = dict()
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
            superket = rho.torch_base(require_grad,  torch_handle=torch)[0]
            superops = [op.torch_base(require_grad,  torch_handle=torch)[0] for op in ops]
            povm_mat = povm.torch_base(require_grad, torch_handle=torch)[0]

            label_to_state[prep_label] = superket
            for i, ol in enumerate(op_labels):
                label_to_gate[ol] = superops[i]
            label_to_povm[povm_label] = povm_mat

        return label_to_state, label_to_gate, label_to_povm

    def _bulk_fill_probs_block(self, array_to_fill, layout, torch_cache: Optional[Tuple] = None):
        if torch_cache is None:
            torch_cache = TorchForwardSimulator._build_torch_cache(self.model, layout)
        else:
            assert isinstance(torch_cache, tuple)
            assert len(torch_cache) == 3
            assert all(isinstance(d, dict) for d in torch_cache)

        for indices, circuit, outcomes in layout.iter_unique_circuits():
            array = array_to_fill[indices]
            self._circuit_fill_probs_block(array, circuit, outcomes, torch_cache)
        pass

    def _circuit_fill_probs_block(self, array_to_fill, circuit, outcomes, torch_cache):
        l2state, l2gate, l2povm = torch_cache
        spc = next(iter(circuit.expand_instruments_and_separate_povm(self.model, outcomes)))
        prep_label = spc.circuit_without_povm[0]
        op_labels  = spc.circuit_without_povm[1:]
        povm_label = spc.povm_label

        superket = l2state[prep_label]
        superops = [l2gate[ol] for ol in op_labels]
        povm_mat = l2povm[povm_label]

        for superop in superops:
            superket = superop @ superket
        probs = povm_mat @ superket

        probs = probs.cpu().detach().numpy().flatten()
        array_to_fill[:] = probs
        return

    def _compute_circuit_outcome_probabilities(
        self, array_to_fill: np.ndarray, circuit: Circuit,
        outcomes: Tuple[Tuple[str]], resource_alloc: ResourceAllocation, time=None
    ):
        """
        This was originally a helper function, called in a loop inside _bulk_fill_probs_block.
        
        The need for this helper function has been obviated by having
        _bulk_fill_probs_block do initial prep work (via the new 
        _build_torch_cache function), and then calling a new per-circuit helper
        function (specifically, _circuit_fill_probs_block) that takes advantage of
        the prep work.
        """
        raise NotImplementedError()

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill):
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout)
        return self._bulk_fill_dprobs_block(array_to_fill, layout)

    def _bulk_fill_dprobs_block(self, array_to_fill, layout):
        probs = np.empty(len(layout), 'd')
        self._bulk_fill_probs_block(probs, layout)

        probs2 = np.empty(len(layout), 'd')
        orig_vec = self.model.to_vector().copy()
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
