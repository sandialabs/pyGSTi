from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel
from pygsti.extras.errorgenpropagation.utilspygstistimtranslator import *
import stim
from numpy import array,kron
from pygsti.tools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen

class LocalStimErrorgenLabel(ElementaryErrorgenLabel):

    '''
    Initiates the errorgen object
    Inputs:
    ______
    errorgen_type: characture can be set to 'H' Hamiltonian, 'S' Stochastic, 'C' Correlated or 'A' active following the conventions
    of the taxonomy of small markovian errorgens paper

    basis_element_labels 

    Outputs:
    Null
    '''
    def __init__(self,errorgen_type: str ,basis_element_labels: list):
        self.errorgen_type=str(errorgen_type)
        self.basis_element_labels=tuple(basis_element_labels) 

    '''
    hashes the error gen object
    '''
    def __hash__(self):
        pauli_hashable=[]
        for pauli in self.basis_element_labels:
            pauli_hashable.append(str(pauli))
        return hash((self.errorgen_type,tuple(pauli_hashable)))
    
    def labels_to_strings(self):
        strings=[]
        for paulistring in self.basis_element_labels:
            strings.append(str(paulistring)[1:].replace('_',"I"))
        return tuple(strings)


    '''
    checks and if two error gens have the same type and labels
    '''
    def __eq__(self, other):
        return (self.errorgen_type == other.errorgen_type
                and self.basis_element_labels == other.basis_element_labels)
    
    '''
    displays the errorgens as strings
    '''
    def __str__(self):
        return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ")"
    

    def __repr__(self):
        return str((self.errorgen_type, self.basis_element_labels))
    
    '''
    Returns the errorbasis matrix for the associated errorgenerator mulitplied by its error rate

    input: A pygsti defined matrix basis by default can be pauli-product, gellmann 'gm' or then pygsti standard basis 'std'
    functions defaults to pauli product if not specified
    '''
    def toWeightedErrorBasisMatrix(self,weight=1.0,matrix_basis='pp'):
        PauliDict={
            'I' : array([[1.0,0.0],[0.0,1.0]]),
            'X' : array([[0.0j, 1.0+0.0j], [1.0+0.0j, 0.0j]]),
            'Y' : array([[0.0, -1.0j], [1.0j, 0.0]]),
            'Z' : array([[1.0, 0.0j], [0.0j, -1.0]])
        }
        paulis=[]
        for paulistring in self.basis_element_labels:
            for idx,pauli in enumerate(paulistring):
                if idx == 0:
                    pauliMat = PauliDict[pauli]
                else:
                    pauliMat=kron(pauliMat,PauliDict[pauli])
            paulis.append(pauliMat)
        if self.errorgen_type in 'HS':
            return weight*change_basis(create_elementary_errorgen(self.errorgen_type,paulis[0]),'std',matrix_basis)
        else:
            return weight*change_basis(create_elementary_errorgen(self.errorgen_type,paulis[0],paulis[1]),'std',matrix_basis)  
        
    def propagate_error_gen_tableau(self, slayer,weight):
        new_basis_labels = []
        weightmod = 1
        for pauli in self.basis_element_labels:
            temp = slayer(pauli)
            weightmod=weightmod*temp.sign
            temp=temp*temp.sign
            new_basis_labels.append(temp)
        
        return (LocalStimErrorgenLabel(self.errorgen_type,new_basis_labels),weightmod*weight)

