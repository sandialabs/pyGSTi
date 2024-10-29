from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel as _ElementaryErrorgenLabel, GlobalElementaryErrorgenLabel as _GEEL,\
LocalElementaryErrorgenLabel as _LEEL
from .utilspygstistimtranslator import *
import stim
import numpy as _np
from pygsti.tools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen

class LocalStimErrorgenLabel(_ElementaryErrorgenLabel):

    """
    `LocalStimErrorgenLabel` is a specialized `ElementaryErrorgenLabel`
    designed to manage the propagation of error generator using Stim primitives for fast Pauli and
    Clifford operations, storing propagation related metadata, and storing metadata relevant to the
    evaluation of non-Markovian error propagators using cumulant expansion based techniques.
    Inputs:
    ______
    errorgen_type: characture can be set to 'H' Hamiltonian, 'S' Stochastic, 'C' Correlated or 'A' active following the conventions
    of the taxonomy of small markovian errorgens paper.  In addition, a new "error generator" 'Z'  null is added in order for there 
    to be a zero divisor in the set. Thus, N[\rho]=0.  Additionally, a multplicative identity 'I' is added.  These exist to make 
    sure the error generators remain closed under multiplication, commutation, under the influence of measurement with post selection,
    and exponential expansion. Notably, the two non error generators do not have any pauli labels.

    basis_element_labels 

    Outputs:
    Null
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
        
        if isinstance(obj, _LEEL):
            #convert to a tuple representation
            obj = (obj.errorgen_type, obj.basis_element_labels)
        
        if isinstance(obj, (tuple, list)):
            #In this case assert that the first element of the tuple is a string corresponding to the
            #error generator type.
            errorgen_type = obj[0]
            
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
            
        return cls(errorgen_type, stim_bels)


    def __init__(self, errorgen_type, basis_element_labels, circuit_time=None, initial_label=None,
                 label=None):
        """
        Create a new instance of  `LocalStimErrorgenLabel`

        Parameters
        ----------
        errorgen_type : str
            A string corresponding to the error generator sector this error generator label is
            an element of. Allowed values are 'H', 'S', 'C' and 'A'.

        basis_element_labels : tuple or list
            A list or tuple of strings labeling basis elements used to label this error generator.
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

        """
        self.errorgen_type=str(errorgen_type)
        self.basis_element_labels=tuple(basis_element_labels) 
        self.label=label
        self.circuit_time = circuit_time

        #additionally store a copy of the value of the original error generator label which will remain unchanged
        #during the course of propagation for later bookkeeping purposes.
        if initial_label is not None:
            self.initial_label = initial_label
        else:
            self.initial_label = self.to_local_eel()

    #TODO: Update various methods to account for additional metadata that has been added.

    def __hash__(self):
        pauli_hashable = [str(pauli) for pauli in self.basis_element_labels]
        return hash((self.errorgen_type, tuple(pauli_hashable)))
    
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
        return isinstance(other, LocalStimErrorgenLabel) and self.errorgen_type == other.errorgen_type \
               and self.basis_element_labels == other.basis_element_labels
    
 
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
    

    def reduce_label(self,labels):
        paulis=[]
        for el in self.basis_element_labels:
            paulis.append(stimPauli_2_pyGSTiPauli(el))
        new_labels=[]
        for pauli in paulis:
            for idx in labels:
                pauli=pauli[:idx]+'I'+pauli[(idx+1):]
            new_labels.append(stim.PauliString(pauli))
        return LocalStimErrorgenLabel(self.errorgen_type,tuple(new_labels))
        

    '''
    Returns the errorbasis matrix for the associated error generator mulitplied by its error rate

    input: A pygsti defined matrix basis by default can be pauli-product, gellmann 'gm' or then pygsti standard basis 'std'
    functions defaults to pauli product if not specified
    '''
    def toWeightedErrorBasisMatrix(self, weight=1.0, matrix_basis='pp'):
        PauliDict={
            'I' : _np.array([[1.0,0.0],[0.0,1.0]]),
            'X' : _np.array([[0.0j, 1.0+0.0j], [1.0+0.0j, 0.0j]]),
            'Y' : _np.array([[0.0, -1.0j], [1.0j, 0.0]]),
            'Z' : _np.array([[1.0, 0.0j], [0.0j, -1.0]])
        }
        paulis=[]
        for paulistring in self.basis_element_labels:
            for idx,pauli in enumerate(paulistring):
                if idx == 0:
                    pauliMat = PauliDict[pauli]
                else:
                    pauliMat=_np.kron(pauliMat,PauliDict[pauli])
            paulis.append(pauliMat)
        if self.errorgen_type in 'HS':
            return weight*change_basis(create_elementary_errorgen(self.errorgen_type, paulis[0]), 'std', matrix_basis)
        else:
            return weight*change_basis(create_elementary_errorgen(self.errorgen_type, paulis[0], paulis[1]),'std', matrix_basis)  
        
    #TODO: Rework this to not directly modify the weights, and only return the sign modifier.
    def propagate_error_gen_tableau(self, slayer, weight):
        """
        Parameters
        ----------
        slayer : 

        weight : float

        """
        if self.errorgen_type =='Z' or self.errorgen_type=='I':
            return (self, weight)
        new_basis_labels = []
        weightmod = 1
        for pauli in self.basis_element_labels:
            temp = slayer(pauli)
            weightmod=_np.real(temp.sign) * weightmod
            temp=temp*temp.sign
            new_basis_labels.append(temp)
        if self.errorgen_type =='S':
            weightmod=1.0
        
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
        return _LEEL(self.errorgen_type, self.bel_to_strings())


