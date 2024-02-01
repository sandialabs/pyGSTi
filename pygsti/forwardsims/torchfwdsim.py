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
    def __init__(self, model = None):
        if not TORCH_ENABLED:
            raise RuntimeError('PyTorch could not be imported.')
        self.model = model
        super(ForwardSimulator, self).__init__(model)

    def _bulk_fill_probs_block(self, array_to_fill, layout):
        l2state, l2gate, l2povm = self._prep_bulk_fill_probs_block(layout)
        for element_indices, circuit, outcomes in layout.iter_unique_circuits():
            self._circuit_fill_probs_block(array_to_fill[element_indices], circuit, outcomes, l2state, l2gate, l2povm)

    def _prep_bulk_fill_probs_block(self, layout):
        label_to_gate = dict()
        label_to_povm = dict()
        label_to_state = dict()
        for _, circuit, outcomes in layout.iter_unique_circuits():
            expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
            # ^ Note, I'm not sure if outcomes needs to be passed to the function above.
            if len(expanded_circuit_outcomes) > 1:
                raise NotImplementedError("I don't know what to do with this.")
            spc = list(expanded_circuit_outcomes.keys())[0]

            prep_label = spc.circuit_without_povm[0]
            op_labels  = spc.circuit_without_povm[1:]
            effect_labels = spc.full_effect_labels

            rho = self.model.circuit_layer_operator(prep_label, typ='prep')
            """ ^
            <class 'pygsti.modelmembers.states.tpstate.TPState'>
            <class 'pygsti.modelmembers.states.densestate.DenseState'>
            <class 'pygsti.modelmembers.states.densestate.DenseStateInterface'>
            <class 'pygsti.modelmembers.states.state.State'>
            <class 'pygsti.modelmembers.modelmember.ModelMember'>
            """
            ops = [self.model.circuit_layer_operator(ol, 'op') for ol in op_labels]
            """ ^ For reasons that I don't understand, this is OFTEN an empty list
            in the first step of iterative GST. When it's nonempty, it contains ...
        
            <class 'pygsti.modelmembers.operations.fulltpop.FullTPOp'>
            <class 'pygsti.modelmembers.operations.denseop.DenseOperator'>
            <class 'pygsti.modelmembers.operations.denseop.DenseOperatorInterface'>
            <class 'pygsti.modelmembers.operations.krausop.KrausOperatorInterface'>
            <class 'pygsti.modelmembers.operations.linearop.LinearOperator'>
            <class 'pygsti.modelmembers.modelmember.ModelMember'>
            """
            effects = [self.model.circuit_layer_operator(el, 'povm') for el in effect_labels]
            """ ^ If we called effect = self.model._circuit_layer_operator(elabel, 'povm')
            then we could skip a call to self.model._cleanparamvec. For some reason
            reaching this code scope in the debugger ends up setting some model member
            to "dirty" and results in an error when we try to clean it. SO, bypassing
            that call to self.model._cleanparamvec, we would see the following class
            inheritance structure of the returned object.
                
            <class 'pygsti.modelmembers.povms.fulleffect.FullPOVMEffect'>
            <class 'pygsti.modelmembers.povms.conjugatedeffect.ConjugatedStatePOVMEffect'>
            <class 'pygsti.modelmembers.povms.conjugatedeffect.DenseEffectInterface'>
            <class 'pygsti.modelmembers.povms.effect.POVMEffect'>
            <class 'pygsti.modelmembers.modelmember.ModelMember'>
            """
        
            rhorep  = rho._rep
            opreps = [op._rep for op in ops]
            effectreps = [effect._rep for effect in effects]
            """ ^ the ._rep fields for states, ops, and effects return
                <class 'pygsti.evotypes.densitymx[_slow].statereps.StateRepDense'>
                <class 'pygsti.evotypes.densitymx[_slow].statereps.StateRep'>
                <class 'pygsti.evotypes.densitymx[_slow].opreps.OpRepDenseSuperop'>
                <class 'pygsti.evotypes.densitymx[_slow].opreps.OpRep'>  
                <class 'pygsti.evotypes.densitymx[_slow].effectreps.EffectRepConjugatedState'>
                <class 'pygsti.evotypes.densitymx[_slow].effectreps.EffectRep'>
            """
        
            # Get the numerical representations
            superket = rhorep.base
            superops = [orep.base for orep in opreps]
            povm_mat = np.row_stack([erep.state_rep.base for erep in effectreps])

            label_to_state[prep_label] = torch.from_numpy(superket)
            for i, ol in enumerate(op_labels):
                label_to_gate[ol] = torch.from_numpy(superops[i])
            label_to_povm[''.join(effect_labels)] = torch.from_numpy(povm_mat)

        return label_to_state, label_to_gate, label_to_povm

    def _circuit_fill_probs_block(self, array_to_fill, circuit, outcomes, l2state, l2gate, l2povm):
        spc = next(iter(circuit.expand_instruments_and_separate_povm(self.model, outcomes)))
        prep_label = spc.circuit_without_povm[0]
        op_labels  = spc.circuit_without_povm[1:]
        povm_label = ''.join(spc.full_effect_labels)

        superket = l2state[prep_label]
        superops = [l2gate[ol] for ol in op_labels]
        povm_mat = l2povm[povm_label]

        for superop in superops:
            superket = superop @ superket
        probs = povm_mat @ superket

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().detach().numpy()
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
        _prep_bulk_probs_block function), and then calling a new per-circuit helper
        function (specifically, _circuit_fill_probs_block) that takes advantage of
        the prep work.
        """
        raise NotImplementedError()



"""
Running GST produces the following traceback if I set a breakpoint inside the
loop over expanded_circuit_outcomes.items() in self._compute_circuit_outcome_probabilities(...).

I think something's happening where accessing the objects here (via the debugger)
makes some object set "self.dirty=True" for the ComplementPOVMEffect.

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
