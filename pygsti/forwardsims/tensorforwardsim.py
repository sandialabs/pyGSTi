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
from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout




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
    #Overide default derivative_dimensions behavior so that it is equal to the number of model parameters
    #down the line we should update this for compatibility with distributable layouts where this may not be
    #the case for MPI compatibility.
    def create_layout(self, circuits, dataset=None, resource_alloc=None,
                      array_types=(), derivative_dimensions=None, verbosity=0):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for `circuits` and `dataset`.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be computed.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        array_types : tuple, optional
            A tuple of string-valued array types, as given by
            :method:`CircuitOutcomeProbabilityArrayLayout.allocate_local_array`.  These types determine
            what types of arrays we anticipate computing using this layout (and forward simulator).  These
            are used to check available memory against the limit (if it exists) within `resource_alloc`.
            The array types also determine the number of derivatives that this layout is able to compute.
            So, for example, if you ever want to compute derivatives or Hessians of element arrays then
            `array_types` must contain at least one `'ep'` or `'epp'` type, respectively or the layout
            will not allocate needed intermediate storage for derivative-containing types.  If you don't
            care about accurate memory limits, use `('e',)` when you only ever compute probabilities and
            never their derivatives, and `('e','ep')` or `('e','ep','epp')` if you need to compute
            Jacobians or Hessians too.

        derivative_dimensions : tuple, optional
            A tuple containing, optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            have minimally 1 or 2 elements when `array_types` contains `'ep'` or `'epp'` types,
            respectively.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.

        Returns
        -------
        CircuitOutcomeProbabilityArrayLayout
        """
        
        if derivative_dimensions is None:
            if 'epp' in array_types:
                derivative_dimensions = (self.model.num_params,self.model.num_params)
            else:
                derivative_dimensions = (self.model.num_params,)
        
        return _CircuitOutcomeProbabilityArrayLayout.create_from(circuits, self.model, dataset, derivative_dimensions,
                                                                 resource_alloc=resource_alloc)
                                                         
                                                         
                                                            