from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy


class GateSetModel(object):
    
    def __init__(self, operators, n, mtype='purestate'):
        
        """
        This is a hacky object that I have introduced in order to remove the imperfect operators
        + simulation things from the DeviceSpec object. This current object is not particularly
        clever. But, I do think that some object along these lines would be useful. What I think
        we need is a flexible way to define imperfect models for gates to then do circuit simulations
        with (this may be something essentially already contained in pyGSTi). E.g., it would be good
        to have a "pauli" model type, which has only Pauli errors, and whereby the model understands
        that it can very efficiently simulate Clifford circuits, and will do so when you pass the 
        model to a simulator.
        
        mtype can be : 'purestate', 'mixedstate', 'pauliclifford'.
        
        operators : a dictionary of imperfect operators. These can be associated with gates, or
        with full circuit layers (perhaps there should be more flexibility than that, but there
        isn't currently).
        
        """
        
        self.fundamental_operators = operators
        self.mtype = mtype
        self.number_of_qubits = n
        self.constructed_operators = {}
    
    def get_gate_operator(self, glabel):
        """
        Gets the operator associated with a glabel, if such an operator exists
        """
        
        try:
            return self.fundamental_operators[glabel]
        except:
            return None
    
    # Todo : change / remove this function.
    def get_circuit_layer_as_operator(self,layer,store=True):
        """
        Todo
        """
        
        try:
            LAO = self.fundamental_operators[tuple(layer)]
        except:
            try:
                LAO = self.constructed__operators[tuple(layer)]
            except:            
                if self.mtype == 'purestate':
                     LAO = self.construct_layer_as_unitary(layer,store)
                if self.mtype == 'mixedstate':
                     LAO =  self.construct_layer_as_superoperator(layer,store) 

        return LAO
    
    
    def construct_layer_as_unitary(self,layer,store=True):
        """
        This function expects a complete circuit layer, whereby every
        qubit has a gate, so it is a list with a length that is the
        number of qubits. This function is also pretty stupid, and should
        be replaced.
        """

        n = self.number_of_qubits
        nn = 2**n
        layer_unitary = _np.zeros((nn,nn),complex)
        
        # This is a list of those qubits for which there is a 1 gate in the layer
        qubits_with_local_gate = []
        # This ....
        qubit_pairs_with_2qubit_gate = []
        
        for i in range(0,n):
            if layer[i].number_of_qubits == 1:
                qubits_with_local_gate.append(layer[i].qubits[0])
            if layer[i].number_of_qubits == 2:
                if (layer[i].qubits[0],layer[i].qubits[1]) not in qubit_pairs_with_2qubit_gate:
                    qubit_pairs_with_2qubit_gate.append((layer[i].qubits[0],layer[i].qubits[1]))
        
        n_local = len(qubits_with_local_gate)
        n_2q = len(qubit_pairs_with_2qubit_gate)
        
        # This contains the superoperators in the layer, to be quickly accessed below. This
        # was here for sensible reasons in a previous version of the code, but is now likely
        # redundent, and could be cut.
        gates_in_layer = {}
        for gate in layer:
            if gate.number_of_qubits == 1:
                gates_in_layer[gate.qubits[0]] = self.fundamental_operators[gate]
            else:
                if (gate.qubits[0],gate.qubits[1]) in qubit_pairs_with_2qubit_gate:
                    gates_in_layer[(gate.qubits[0],gate.qubits[1])] = self.fundamental_operators[gate]
                
        for i in range(0,nn):
            for j in range(0,nn):
                
                ii = [0 for x in range(0,n)]
                base_rep =  [int(x) for x in _np.base_repr(i,2)]
                ii[n-len(base_rep):] = base_rep
                jj = [0 for x in range(0,n)]
                base_rep =  [int(x) for x in _np.base_repr(j,2)]
                jj[n-len(base_rep):] = base_rep
                 
                # To store the values that are multiplied together to get the
                # value of layer_superoperator[i,j].
                elements = _np.zeros(n_local+n_2q,complex)
                
                # Find the values for the 1 qubit gates.
                for k in range(0,n_local):
                    q = qubits_with_local_gate[k]
                    elements[k] = gates_in_layer[q][ii[q],jj[q]]
                    
                # Find the values for the 2 qubit gates
                for k in range(0,n_2q):
                    q1q2 = qubit_pairs_with_2qubit_gate[k]
                    q1 = q1q2[0]
                    q2 = q1q2[1]
                    elements[k+n_local] = gates_in_layer[q1q2][2*ii[q1]+ii[q2],2*jj[q1]+jj[q2]]
                                
                layer_unitary[i,j] =  _np.prod(elements)
        
        if store:
            self.constructed_operators[tuple(layer)] = layer_unitary
        
        return layer_unitary

    def construct_layer_as_superoperator(self,layer,store=True):
        """
        This function expects a complete circuit layer, whereby every
        qubit has a gate, so it is a list with a length that is the
        number of qubits. This function is also pretty stupid, and should
        be replaced.
        """
        
        n = self.number_of_qubits
        nn = 4**n
        layer_superoperator = _np.zeros((nn,nn),float)
        
        # This is a list of those qubits for which there is a 1 gate in the layer
        qubits_with_local_gate = []
        # This ....
        qubit_pairs_with_2qubit_gate = []
        
        for i in range(0,n):
            if layer[i].number_of_qubits == 1:
                qubits_with_local_gate.append(layer[i].qubits[0])
            if layer[i].number_of_qubits == 2:
                if (layer[i].qubits[0],layer[i].qubits[1]) not in qubit_pairs_with_2qubit_gate:
                    qubit_pairs_with_2qubit_gate.append((layer[i].qubits[0],layer[i].qubits[1]))
        
        n_local = len(qubits_with_local_gate)
        n_2q = len(qubit_pairs_with_2qubit_gate)
        #print(n_2q,n_local)
    
        # This contains the superoperators in the layer, to be quickly accessed below. This
        # was here for sensible reasons in a previous version of the code, but is now likely
        # redundent, and could be cut.
        gates_in_layer = {}
        for gate in layer:
            #print(gate)
            if gate.number_of_qubits == 1:
                gates_in_layer[gate.qubits[0]] = self.fundamental_operators[gate]
            else:
                if (gate.qubits[0],gate.qubits[1]) in qubit_pairs_with_2qubit_gate:
                    gates_in_layer[(gate.qubits[0],gate.qubits[1])] = self.fundamental_operators[gate]
                
        #print(gates_in_layer)
        #print(nn)
        for i in range(0,nn):
            for j in range(0,nn):
                
                ii = [0 for x in range(0,n)]
                base_rep =  [int(x) for x in _np.base_repr(i,4)]
                ii[n-len(base_rep):] = base_rep
                jj = [0 for x in range(0,n)]
                base_rep =  [int(x) for x in _np.base_repr(j,4)]
                jj[n-len(base_rep):] = base_rep
                 
                # To store the values that are multiplied together to get the
                # value of layer_superoperator[i,j].
                elements = _np.zeros(n_local+n_2q,float)
                
                # Find the values for the 1 qubit gates.
                for k in range(0,n_local):
                    q = qubits_with_local_gate[k]
                    elements[k] = gates_in_layer[q][ii[q],jj[q]]
                    
                # Find the values for the 2 qubit gates
                for k in range(0,n_2q):
                    q1q2 = qubit_pairs_with_2qubit_gate[k]
                    q1 = q1q2[0]
                    q2 = q1q2[1]
                    elements[k+n_local] = gates_in_layer[q1q2][4*ii[q1]+ii[q2],4*jj[q1]+jj[q2]]
                                
                layer_superoperator[i,j] =  _np.prod(elements)
        
        if store:
            self.constructed_operators[tuple(layer)] = layer_superoperator
        
        return layer_superoperator