#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel as _ElementaryErrorgenLabel, GlobalElementaryErrorgenLabel as _GEEL,\
LocalElementaryErrorgenLabel as _LEEL
try:
    import stim
except ImportError:
    pass
import numpy as _np
from pygsti.tools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen

#TODO: Split this into a parent class and subclass for markovian and non-markovian
#propagation. There is some overhead in instantiating the NM version of these labels
#which we can avoid and make markovian applications much more efficient (label instantiation
#is like a third of runtime when using higher-order BCH, e.g.)
class LocalStimErrorgenLabel(_ElementaryErrorgenLabel):

    """
    `LocalStimErrorgenLabel` is a specialized `ElementaryErrorgenLabel`
    designed to manage the propagation of error generator using Stim primitives for fast Pauli and
    Clifford operations, storing propagation related metadata, and storing metadata relevant to the
    evaluation of non-Markovian error propagators using cumulant expansion based techniques.
    """

    @classmethod
    def cast(cls, obj, sslbls=None):
        """
        Method for casting objects to instances of LocalStimErrorgenLabel.

        Parameters
        ----------
        obj : `LocalStimErrorgenLabel`, ``LocalElementaryErrorgenLabel`, `GlobalElementaryErrorgenLabel`, tuple or list

        sslbls : tuple or list, optional (default None)
            A complete set of state space labels. Used when casting from a GlobalElementaryErrorgenLabel
            or from a tuple of length 3 (wherein the final element is interpreted as the set of ssblbs the error
            generator acts upon).

        Returns
        -------
        `LocalStimErrorgenLabel`
        """
        if isinstance(obj, LocalStimErrorgenLabel):
            return obj
        
        if isinstance(obj, _GEEL):
            #convert to a tuple representation
            assert sslbls is not None, 'Must specify sslbls when casting from `GlobalElementaryErrorgenLabel`.'
            obj = (obj.errorgen_type, obj.basis_element_labels, obj.sslbls)
            initial_label=None
        
        if isinstance(obj, _LEEL):
            #convert to a tuple representation
            initial_label = obj
            obj = (obj.errorgen_type, obj.basis_element_labels)
        
        if isinstance(obj, (tuple, list)):
            #In this case assert that the first element of the tuple is a string corresponding to the
            #error generator type.
            errorgen_type = obj[0]
            initial_label = None

            #two elements for a local label and three for a global one
            #second element should have the basis element labels
            assert len(obj)==2 or len(obj)==3 and isinstance(obj[1], (tuple, list)) 
            
            #if a global label tuple the third element should be a tuple or list.
            if len(obj)==3:
                assert isinstance(obj[2], (tuple, list))
                assert sslbls is not None, 'Must specify sslbls when casting from a tuple or list of length 3. See docstring.'
                #convert to local-style bels.
                indices_to_replace = [sslbls.index(sslbl) for sslbl in obj[2]]
                local_bels = []
                for global_lbl in obj[1]:
                    #start by figure out which initialization to use, either stim
                    #or a string.
                    local_bel = stim.PauliString('I'*len(sslbls))
                    for kk, k in enumerate(indices_to_replace):
                        local_bel[k] = global_lbl[kk]
                    local_bels.append(local_bel)
            else:
                local_bels = obj[1]

        #now build the LocalStimErrorgenLabel
        stim_bels = []
        for bel in local_bels:
            if isinstance(bel, str):
                stim_bels.append(stim.PauliString(bel))
            elif isinstance(bel, stim.PauliString):
                stim_bels.append(bel)
            else:
                raise ValueError('Only str and `stim.PauliString` basis element labels are supported presently.')
            
        return cls(errorgen_type, stim_bels, initial_label=initial_label)


    def __init__(self, errorgen_type, basis_element_labels, circuit_time=None, initial_label=None,
                 label=None, pauli_str_reps=None):
        """
        Create a new instance of  `LocalStimErrorgenLabel`

        Parameters
        ----------
        errorgen_type : str
            A string corresponding to the error generator sector this error generator label is
            an element of. Allowed values are 'H', 'S', 'C' and 'A'.

        basis_element_labels : tuple or list
            A list or tuple of stim.PauliString labeling basis elements used to label this error generator.
            This is either length-1 for 'H' and 'S' type error generators, or length-2 for 'C' and 'A'
            type.

        circuit_time : float, optional (default None)
            An optional value which associates this error generator with a particular circuit time at
            which it arose. This is primarily utilized in the context of non-Markovian simulations and
            estimation where an error generator may notionally be associated with a stochastic process.

        initial_label : `ElementaryErrorgenLabel`, optional (default None)
            If not None, then this `ElementaryErrorgenLabel` is stored within this label and is interpreted
            as being the 'initial' value of this error generator, prior to any propagation or transformation
            during the course of its use. If None, then this is initialized to a `LocalElementaryErrorgenLabel`
            matching the `errorgen_type` and `basis_element_labels` of this label.

        label : str, optional (default None)
            An optional label string which is included when printing the string representation of this
            label.

        pauli_str_reps : tuple of str, optional (default None)
            Optional tuple of python strings corresponding to the stim.PauliStrings in basis_element_labels.
            When specified can speed up construction of hashable label representations.
        """
        self.errorgen_type = errorgen_type
        self.basis_element_labels = tuple(basis_element_labels) 
        self.label = label
        self.circuit_time = circuit_time

        if pauli_str_reps is not None:
            self._hashable_basis_element_labels = pauli_str_reps
            self._hashable_string_rep = self.errorgen_type.join(pauli_str_reps)
        else:
            self._hashable_basis_element_labels = self.bel_to_strings()
            self._hashable_string_rep = self.errorgen_type.join(self._hashable_basis_element_labels)

        #additionally store a copy of the value of the original error generator label which will remain unchanged
        #during the course of propagation for later bookkeeping purposes.
        if initial_label is not None:
            self.initial_label = initial_label
        else:
            self.initial_label = self.to_local_eel()
    #TODO: Update various methods to account for additional metadata that has been added.

    def __hash__(self):
        #return hash((self.errorgen_type, self._hashable_basis_element_labels))
        return hash(self._hashable_string_rep)

    def bel_to_strings(self):
        """
        Convert the elements of `basis_element_labels` to python strings
        (from stim.PauliString(s)) and return as a tuple. 
        """       
        return tuple([str(ps)[1:].replace('_',"I") for ps in self.basis_element_labels])


    def __eq__(self, other):
        """
        Performs equality check by seeing if the two error gen labels have the same `errorgen_type` 
        and `basis_element_labels`.
        """
        return self.errorgen_type == other.errorgen_type and self.basis_element_labels == other.basis_element_labels \
            and isinstance(other, LocalStimErrorgenLabel)
    
 
    def __str__(self):
        if self.label is None:
            return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ")"
        else:
            return self.errorgen_type + " " + str(self.label)+ " " + "(" \
                   + ",".join(map(str, self.basis_element_labels)) + ")"

    def __repr__(self):
        if self.label is None:
            if self.circuit_time is not None:
                return f'({self.errorgen_type}, {self.basis_element_labels}, time={self.circuit_time})'
            else:
                return f'({self.errorgen_type}, {self.basis_element_labels})'
        else:
            if self.circuit_time is not None:
                return f'({self.errorgen_type}, {self.label}, {self.basis_element_labels}, time={self.circuit_time})'
            else:
                return f'({self.errorgen_type}, {self.label}, {self.basis_element_labels})'
    
      
    #TODO: Rework this to not directly modify the weights, and only return the sign modifier.
    def propagate_error_gen_tableau(self, slayer, weight):
        """
        Parameters
        ----------
        slayer : `stim.Tableau`
            `stim.Tableau` object corresponding to an ideal Clifford operations for 
            a circuit layer which we will be propagating this error generator through. 

        weight : float
            Current weight of this error generator.
        
        Returns
        -------
        tuple of consisting of an `LocalStimErrorgenLabel` and an updated error generator
        weight, which may have changed by a sign.
        """
        new_basis_labels = []
        weightmod = 1.0
        if self.errorgen_type == 'S':
            for pauli in self.basis_element_labels:
                temp = slayer(pauli)
                temp = temp*temp.sign
                new_basis_labels.append(temp)
        else:
            for pauli in self.basis_element_labels:
                temp = slayer(pauli)
                temp_sign = temp.sign
                weightmod = temp_sign.real*weightmod
                temp = temp*temp_sign
                new_basis_labels.append(temp)
        
        return (LocalStimErrorgenLabel(self.errorgen_type, new_basis_labels, initial_label=self.initial_label, circuit_time=self.circuit_time), 
                weightmod*weight)
    
    def to_global_eel(self, sslbls = None):
        """
        Returns a `GlobalElementaryErrorgenLabel` equivalent to this `LocalStimErrorgenLabel`.

        sslbls : list (optional, default None)
            A list of state space labels corresponding to the qubits corresponding to each
            of the paulis in the local basis element label. If None this defaults a list of integers
            ranging from 0 to N where N is the number of paulis in the basis element labels.
        """

        #first get the pauli strings corresponding to the stim.PauliString object(s) that are the
        #basis_element_labels.
        pauli_strings = self.bel_to_strings()
        if sslbls is None:
            sslbls = list(range(len(pauli_strings[0]))) #The two pauli strings should be the same length, so take the first.
        #GlobalElementaryErrorgenLabel should have built-in support for casting from a tuple of the error gen type
        #and the paulis for the basis element labels, so long as it is given appropriate sslbls to use.
        return _GEEL.cast((self.errorgen_type,) + pauli_strings, sslbls= sslbls)


    def to_local_eel(self):
        """
        Returns a `LocalElementaryErrorgenLabel` equivalent to this `LocalStimErrorgenLabel`.

        Returns
        -------
        `LocalElementaryErrorgenLabel`
        """
        return _LEEL(self.errorgen_type, self._hashable_basis_element_labels)


