from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel
from pygsti.extras.errorgenpropagation.utilspygstistimtranslator import *
import stim
from numpy import array,kron
from pygsti.tools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen

class localstimerrorgen(ElementaryErrorgenLabel):


    '''
    Initiates the errorgen object
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
    '''
    def __init__(self,errorgen_type: str ,basis_element_labels: list, label=None):
        self.errorgen_type=str(errorgen_type)
        self.basis_element_labels=tuple(basis_element_labels) 
        self.label=label

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
        if self.label is None:
            return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ")"
        else:
            return self.errorgen_type +" "+str(self.label)+" "+ "(" + ",".join(map(str, self.basis_element_labels)) + ")"

    

    def __repr__(self):
        if self.label is None:
            return str((self.errorgen_type, self.basis_element_labels))
        else:
            return str((self.errorgen_type, self.label, self.basis_element_labels))
    

    def reduce_label(self,labels):
        paulis=[]
        for el in self.basis_element_labels:
            paulis.append(stimPauli_2_pyGSTiPauli(el))
        new_labels=[]
        for pauli in paulis:
            for idx in labels:
                pauli=pauli[:idx]+'I'+pauli[(idx+1):]
            new_labels.append(stim.PauliString(pauli))
        return localstimerrorgen(self.errorgen_type,tuple(new_labels))
        

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
        if self.errorgen_type =='Z' or self.errorgen_type=='I':
            return (self,weight)
        new_basis_labels = []
        weightmod = 1
        for pauli in self.basis_element_labels:
            temp = slayer(pauli)
            weightmod=weightmod*temp.sign
            temp=temp*temp.sign
            new_basis_labels.append(temp)
        if self.errorgen_type =='S':
            weightmod=1.0

        
        return (localstimerrorgen(self.errorgen_type,new_basis_labels),weightmod*weight)

