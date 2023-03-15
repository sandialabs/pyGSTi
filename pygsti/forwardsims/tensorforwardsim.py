"""
ForwardSimulator class for handling tensor network parameterized models
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.forwardsims import ForwardSimulator as _ForwardSimulator
from quimb.tensor import Tensor as _Tensor
from quimb.tensor import TensorNetwork as _TensorNetwork
import numpy as _np




class TensorForwardSimulator(_ForwardSimulator):
    """
    ForwardSimulator for tensor network based simulations.
    """
    
    def __init__(self, model=None):
        """
        Construct a new TensorForwardSimulator object.

        Parameters
        ----------
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        super().__init__(model)
        
        
    def add_time_context(self, TN, t):
        TN_with_time = TN.copy()

        outer_inds = TN_with_time.outer_inds()
        inner_inds = TN_with_time.inner_inds()
        reindex_dict = {ind : ind + f'_{t}' for ind in outer_inds + inner_inds}

        TN_with_time.reindex_(reindex_dict)
        return TN_with_time

    def gate_join_TN(self, t, nQ):
        deltas = [_Tensor(_np.eye(2), inds = [f'bo{q}_{t}', f'bi{q}_{t+1}']) for q in range(nQ)]
        deltas += [_Tensor(_np.eye(2), inds = [f'ko{q}_{t}', f'ki{q}_{t+1}']) for q in range(nQ)]

        return _TensorNetwork(deltas)

    def construct_circuit_tensor_network(self, circuit):
        #map gate indices to have a position in the circuit

        reindexed_TNs = _TensorNetwork([self.add_time_context(self.model[key].LPDO_tensor_network, i + 1) for i, key in enumerate(circuit)])
        index_mapping = _TensorNetwork([self.gate_join_TN(t+1, self.model.state_space.num_qubits) for t in range(len(circuit)-1)])
        
        
        return reindexed_TNs & index_mapping
        
    #define a function that takes as input a circuit and returns the composite tensor network       
        
    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes,resource_alloc, time=None):
        expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        for spc, spc_outcomes in expanded_circuit_outcomes.items():  # spc is a SeparatePOVMCircuit
            # Note: `spc.circuit_without_povm` *always* begins with a prep label.
            indices = [outcome_to_index[o] for o in spc_outcomes]
            if time is None:  # time-independent state propagation

                rho = self.model.circuit_layer_operator(spc.circuit_without_povm[0], 'prep')
                povm = self.model.circuit_layer_operator(spc.povm_label, 'povm')
                
                circuit_tensor_network = self.construct_circuit_tensor_network(spc.circuit_without_povm[1:])
                
                num_qubits= self.model.state_space.num_qubits
                print(rho)
                print(povm)

                if povm is None:
                    effects = [self.model.circuit_layer_operator(elabel, 'povm') for elabel in spc.full_effect_labels]
                    num_qubits= self.model.state_space.num_qubits
                    
                    
                    array_to_fill[indices] = [(circuit_tensor_network & self.gate_join_TN(0, num_qubits) & rho.LPDO_tensor_network & self.gate_join_TN(len(circuit), num_qubits) & effect.LPDO_tensor_network).contract() for effect in effects] # outcome probabilities
                else:
                    #print(spc.effect_labels)
                    #import pdb
                    #pdb.set_trace()
                    #effects = [self.model.circuit_layer_operator(elabel, 'povm') for elabel in spc.effect_labels]
                    effects = [self.model.povms['Mdefault'][elabel] for elabel in spc.effect_labels]
                    num_qubits= self.model.state_space.num_qubits
                    rho_TN = rho.LPDO_tensor_network.copy()
                    rho_TN = self.add_time_context(rho_TN, 0)
                    
                    effect_TNs = [effect.LPDO_tensor_network.copy() for effect in effects]
                    effect_TNs = [self.add_time_context(effect_TN, len(circuit) + 1) for effect_TN in effect_TNs]
                    
                    
                    array_to_fill[indices] = [(circuit_tensor_network & self.gate_join_TN(0, num_qubits) & rho_TN & self.gate_join_TN(len(circuit), num_qubits) & effect_TN).contract() for effect_TN in effect_TNs]  # outcome probabilities
                
            else:
                raise NotImplementedError('Have not done time dependent stuff yet.')
                #t = time  # Note: time in labels == duration
                #rholabel = spc.circuit_without_povm[0]
                #op = self.model.circuit_layer_operator(rholabel, 'prep'); op.set_time(t); t += rholabel.time
                #state = op._rep
                #for ol in spc.circuit_without_povm[1:]:
                #    op = self.model.circuit_layer_operator(ol, 'op'); op.set_time(t); t += ol.time
                #    state = op._rep.acton(state)
                #ps = []
                #for elabel in spc.full_effect_labels:
                #    op = self.model.circuit_layer_operator(elabel, 'povm'); op.set_time(t)
                #    # Note: don't advance time (all effects occur at same time)
                #    ps.append(op._rep.probability(state))
                #array_to_fill[indices] = ps
                
    #def _compute_circuit_outcome_probability_derivatives(self, array_to_fill, circuit, outcomes, param_slice,
    #                                                     resource_alloc):
                                                         
                                                         
                                                            