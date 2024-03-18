from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel
from pygsti.extras.errorgenpropagation.utilspygstistimtranslator import *
import stim
from numpy import array,kron
from pygsti.tools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen
'''
Similar to errorgenlabel but has an errorrate included as well as additional classes
'''
# Create a new pygsti-ish method where we use a modified dictionary and a modified local error generator where the keys are
# stim PauliStrings
class propagatableerrorgen(ElementaryErrorgenLabel):
    '''
    Labels an elementary errorgen by a type, pauli and error rate
    '''

    @classmethod
    def cast(cls, obj, sslbls=None, identity_label='I'):
        raise NotImplementedError("TODO: Implement casts for this method")


    '''
    Initiates the errorgen object
    Inputs
    errorgen_type: charecture can be set to 'H' Hamiltonian, 'S' Stochastic, 'C' Correlated or 'A' active following the conventions
    of the taxonomy of small markovian errorgens paper

    Outputs:
    propagatableerrorgen object
    '''
    def __init__(self,errorgen_type,basis_element_labels,error_rate):
        self.errorgen_type=str(errorgen_type)
        self.basis_element_labels=tuple(basis_element_labels)
        self.error_rate=error_rate

    '''
    hashes the error gen object
    '''
    def __hash__(self):
        return hash((self.errorgen_type,self.basis_element_labels))

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
        return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ")" + ": " + self.error_rate
    

    def __repr__(self):
        return str((self.errorgen_type, self.basis_element_labels, self.error_rate))
    
    '''
    adds the error rates together oftwo error generators of the same type and label
    '''
    def __add__(self,other):
        if self.errorgen_type == other.errorgen_type and self.basis_element_labels == other.basis_element_labels:
            return propagatableerrorgen(self.errorgen_type,self.basis_element_labels,self.error_rate + other.error_rate)
        else:
            raise Exception("ErrorGens are not equal")

    '''
    returns the dictionary representation of the error generator inline with pygsti notation
    '''    
    def to_dict(self):
        return {self: self.error_rate}
    

    '''
    returns the error rate
    '''
    def get_Error_Rate(self):
        return self.error_rate
    
    '''
    returns the string representation of the first pauli label
    '''
    def getP1(self):
        return self.basis_element_labels[0]
    
    '''
    returns the string representation of the second pauli label
    '''
    def getP2(self):
        return self.basis_element_labels[1]

    '''
    propagates a propagatableerrorgen object through a clifford layer, returns the created error gen
    '''
    def propagate_error_gen_inplace(self, player):
        slayer = pyGSTiLayer_to_stimLayer(player)
        new_basis_labels = []
        weightmod = 1
        for pauli in self.basis_element_labels:
            temp=pyGSTiPauli_2_stimPauli(pauli)
            temp = slayer(temp)
            weightmod=weightmod*temp.sign
            new_basis_labels.append(stimPauli_2_pyGSTiPauli(temp))

        if self.errorgen_type in 'HCA':
            self.error_rate=self.error_rate*weightmod
        self.basis_element_labels =tuple(new_basis_labels)

    '''
    using stim propagates the associated pauli labels through a stim tableu object, the object is modified inplace
    '''
    def propagate_error_gen_inplace_tableau(self, slayer):
        new_basis_labels = []
        weightmod = 1
        for pauli in self.basis_element_labels:
            temp=pyGSTiPauli_2_stimPauli(pauli)
            temp = slayer(temp)
            weightmod=weightmod*temp.sign
            new_basis_labels.append(stimPauli_2_pyGSTiPauli(temp))

        if self.errorgen_type in 'HCA':
            self.error_rate=self.error_rate*weightmod
        self.basis_element_labels =tuple(new_basis_labels)

    '''
    returns the strings representing the pauli labels in the pygsti representation of paulis as stim PauliStrings
    '''
    def returnStimPaulis(self):
        paulis_string=[]
        for pauli in self.basis_element_labels:
            paulis_string.append(stim.PauliString(pauli))
        return tuple(paulis_string)

    '''
    Returns the errorbasis matrix for the associated errorgenerator mulitplied by its error rate

    input: A pygsti defined matrix basis by default can be pauli-product, gellmann 'gm' or then pygsti standard basis 'std'
    functions defaults to pauli product if not specified
    '''
    def toWeightedErrorBasisMatrix(self,matrix_basis='pp'):
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
            return self.error_rate*change_basis(create_elementary_errorgen(self.errorgen_type,paulis[0]),'std',matrix_basis)
        else:
            return self.error_rate*change_basis(create_elementary_errorgen(self.errorgen_type,paulis[0],paulis[1]),'std',matrix_basis)     





'''
Returns the Commutator of two errors
'''
def commute_errors(ErG1,ErG2, weightFlip=1.0, BCHweight=1.0):
    def com(p1,p2):
        P1 = pyGSTiPauli_2_stimPauli(p1)
        P2=pyGSTiPauli_2_stimPauli(p2)
        P3=P1*P2-P2*P1
        return (P3.weight,stimPauli_2_pyGSTiPauli(P3))
    
    def acom(p1,p2):
        P1 = pyGSTiPauli_2_stimPauli(p1)
        P2=pyGSTiPauli_2_stimPauli(p2)
        P3=P1*P2+P2*P1
        return (P3.weight,stimPauli_2_pyGSTiPauli(P3))
    
    def labelMultiply(p1,p2):
        P1 = pyGSTiPauli_2_stimPauli(p1)
        P2=pyGSTiPauli_2_stimPauli(p2)
        P3=P1*P2
        return (P3.weight,stimPauli_2_pyGSTiPauli(P3))
    
    errorGens=[]
    
    wT=ErG1.getWeight()*ErG2.getWeight()*weightFlip*BCHweight
    
    if ErG1.getType()=='H' and ErG2.getType()=='H':
        pVec=com(ErG1.getP1() , ErG2.getP2())
        errorGens.append( propagatableerrorgen( 'H' , [pVec[1]] , -1j*wT *pVec[0] ) )
        
    elif ErG1.getType()=='H' and ErG2.getType()=='S':
        pVec=com(ErG2.getP1() , ErG1.getP1())
        errorGens.append( propagatableerrorgen( 'C' , [ErG2.getP1() , pVec[1]] , 1j*wT*pVec[0] ) )
         
    elif ErG1.getType()=='S' and ErG2.getType()=='H':
        pVec=com(ErG2.getP1() , ErG1.getP1())
        errorGens.append( propagatableerrorgen( 'C' , [ErG2.getP1() , pVec[1]] , -1j*wT *pVec[0] ) )
          
    elif ErG1.getType()=='H' and ErG2.getType()=='C':
        pVec1=com(ErG2.getP1() , ErG1.getP1())
        errorGens.append( propagatableerrorgen('C' , [pVec1[1], ErG2.getP2()] , 1j*wT*pVec1[0] ) )
        pVec2=com(ErG2.getP2() , ErG1.getP1())
        errorGens.append( propagatableerrorgen('C' , [pVec2[1] , ErG2.getP1()] , 1j*wT*pVec2[0] ) )
                          
    elif ErG1.getType()=='C' and ErG2.getType()=='H':
        errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
    
    elif ErG1.getType()=='H' and ErG2.getType()=='A':
        pVec1 = com(ErG1.getP1() , ErG2.getP1())
        errorGens.append( propagatableerrorgen('A'  , [pVec1[1] , ErG2.getP2()] , -1j*wT*pVec1[0]) )
        pVec2 = com(ErG1.getP1() , ErG2.getP2())
        errorGens.append( propagatableerrorgen('A'  , [ErG2.getP1(), pVec2[1]] , -1j*wT*pVec2[0] ) )
                          
    elif ErG1.getType()=='A' and ErG2.getType()=='H':
        errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)

    elif ErG1.getType()=='S' and ErG2.getType()=='S':
        errorGens.append( propagatableerrorgen('H', ErG1.getP1(),0 ))
                         
    elif ErG1.getType()=='S' and ErG2.getType()=='C':
       pVec1=labelMultiply(ErG1.getP1() , ErG2.getP1())
       pVec2=labelMultiply(ErG2.getP2() , ErG1.getP1())
       errorGens.append( propagatableerrorgen( 'A' , [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]  ))
       pVec1 = labelMultiply(ErG1.getP1() , ErG2.getP2())
       pVec2 = labelMultiply(ErG2.getP1() , ErG1.getP1())
       errorGens.append( propagatableerrorgen( 'A', [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
       pVec1 =acom(ErG2.getP1(), ErG2.getP2())
       pVec2 = labelMultiply(pVec1[1],ErG1.getP1())
       errorGens.append( propagatableerrorgen( 'A'  ,[pVec2[1], ErG1.getP1()] , -1j*.5*wT*pVec1[0]*pVec2[0]))
       pVec1=acom(ErG2.getP1(), ErG2.getP2())
       pVec2=labelMultiply(ErG1.getP1(),pVec1[1])
       errorGens.append( propagatableerrorgen( 'A',   [ErG1.getP1() ,pVec2[1]],-1j*.5*wT*pVec1[0]*pVec2[0]))
                         
    elif ErG1.getType() == 'C' and ErG2.getType() == 'S':
       errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
                         
    elif ErG1.getType() == 'S' and ErG2.getType() == 'A':
       pVec1 =labelMultiply(ErG1.getP1() , ErG2.getP1())
       pVec2=labelMultiply(ErG2.getP2() , ErG1.getP1())
       errorGens.append( propagatableerrorgen( 'C', [pVec1[1], pVec2[1]] ,1j*wT*pVec1[0]*pVec2[0] ))
       pVec1=labelMultiply(ErG1.getP1() , ErG2.getP2())
       pVec2=labelMultiply(ErG2.getP1() , ErG1.getP1())
       errorGens.append( propagatableerrorgen( 'C', [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
       pVec1 = com(ErG2.getP1() , ErG2.getP2())
       pVec2 = com(ErG1.getP1(),pVec1[1])
       errorGens.append( propagatableerrorgen( 'A', [ErG1.getP1(), pVec2[1]] ,-.5*wT*pVec1[0]*pVec2[0]))
                         
    elif ErG1.getType() == 'A' and ErG1.getType() == 'S':
       errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
                         
    elif ErG1.getType() == 'C' and ErG2.getType() == 'C':
        A=ErG1.getP1()
        B=ErG1.getP2()
        P=ErG2.getP1()
        Q=ErG2.getP2()
        pVec1 = labelMultiply(A,P)
        pVec2 =labelMultiply(Q,B)
        errorGens.append( propagatableerrorgen( 'A', [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0] ))        
        pVec1 = labelMultiply(A,Q)
        pVec2 =labelMultiply(P,B)
        errorGens.append( propagatableerrorgen( 'A'  , [pVec1[1] , pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = labelMultiply(B,P)
        pVec2 =labelMultiply(Q,A)                 
        errorGens.append( propagatableerrorgen( 'A'  , [pVec1[1] , pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = labelMultiply(B,Q)
        pVec2 =labelMultiply(P,A)
        errorGens.append( propagatableerrorgen( 'A'  , [pVec1[1] , pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(A,B)
        pVec2=com(P,pVec1[1])
        errorGens.append( propagatableerrorgen( 'A'  , [pVec2[1] , Q ], -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(A,B)
        pVec2=com(Q,pVec1[1])
        errorGens.append( propagatableerrorgen( 'A'  , [pVec2[1], P] , -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(P,Q)
        pVec2=com(pVec1[1],A)
        errorGens.append( propagatableerrorgen( 'A' , [pVec2[1] , B] , -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(P,Q)
        pVec2=com(pVec1[1],B)
        errorGens.append( propagatableerrorgen( 'A' , [pVec2[1] , A ] , -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(A,B)
        pVec2=acom(P,Q)
        pVec3=com(pVec1[1],pVec2[1])
        errorGens.append( propagatableerrorgen( 'H', [pVec3[1]] ,.25*1j*wT*pVec1[0]*pVec2[0]*pVec3[0]))

    elif ErG1.getType() == 'C' and ErG2.getType() == 'A':
        A=ErG1.getP1()
        B=ErG1.getP2()
        P=ErG2.getP1()
        Q=ErG2.getP2()
        pVec1 = labelMultiply(A,P)
        pVec2 =labelMultiply(Q,B)
        errorGens.append( propagatableerrorgen('C' , [pVec1[1],pVec2[1]] , 1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = labelMultiply(A,Q)
        pVec2 =labelMultiply(P,B)
        errorGens.append( propagatableerrorgen('C' ,[pVec1[1],pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = labelMultiply(B,P)
        pVec2 =labelMultiply(Q,A)
        errorGens.append( propagatableerrorgen('C' , [pVec1[1],pVec2[1]] , 1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = labelMultiply(P,A)
        pVec2 =labelMultiply(B,Q)
        errorGens.append( propagatableerrorgen('C'  ,[pVec1[1],pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = com(P,Q)
        pVec2 =com(A,pVec1[1])
        errorGens.append( propagatableerrorgen('A'  , [pVec2[1] , B] , .5*wT*pVec1[0]*pVec2[0] ))
        pVec1 = com(P,Q)
        pVec2 =com(B,pVec1[1])
        errorGens.append( propagatableerrorgen('A' , [pVec2[1],  A ], .5*wT*pVec1[0]*pVec2[0] ))
        pVec1 = acom(A,B)
        pVec2 =com(P,pVec1[1])
        errorGens.append( propagatableerrorgen('C', [pVec2[1] , Q ], .5*1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = acom(A,B)
        pVec2 =com(Q,pVec1[1])
        errorGens.append( propagatableerrorgen('C',[pVec2[1],P ],-.5*1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = com(P,Q)
        pVec2 =acom(A,B)
        pVec3=com(pVec1[1],pVec2[1])
        errorGens.append( propagatableerrorgen('H',[pVec3[1]],-.25*wT*pVec1[0]*pVec2[0]*pVec3[0]))
    
    elif ErG1.getType() == 'A' and ErG2.getType() == 'C':
        errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
                         
    elif ErG1.getType() == 'A' and ErG2.getType() == 'A':
        A=ErG1.getP1()
        B=ErG1.getP2()
        P=ErG2.getP1()
        Q=ErG2.getP2()
        pVec1=labelMultiply(Q,B)
        pVec2=labelMultiply(A,P)
        errorGens.append(propagatableerrorgen('A',[pVec1[1],pVec2[1]] ,-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=labelMultiply(P,A)
        pVec2=labelMultiply(B,Q)
        errorGens.append(propagatableerrorgen('A',[pVec1[1],pVec2[1]],-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=labelMultiply(B,P)
        pVec2=labelMultiply(Q,A)
        errorGens.append(propagatableerrorgen('A',[pVec1[1],pVec2[1]],-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=labelMultiply(A,Q)
        pVec2=labelMultiply(P,B)
        errorGens.append(propagatableerrorgen('A',[pVec1[1],pVec2[1]],-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=com(P,Q)
        pVec2=com(B,pVec1[1])
        errorGens.append(propagatableerrorgen('C',[pVec2[1],A],.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(P,Q)
        pVec2=com(A,pVec1[1])
        errorGens.append(propagatableerrorgen('C',[pVec2[1],B] ,-.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(A,B)
        pVec2=com(P,pVec1[1])
        errorGens.append(propagatableerrorgen('C', [pVec2[1],Q] ,.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(A,B)
        pVec2=com(Q,pVec1[1])
        errorGens.append(propagatableerrorgen('C', [pVec2[1],P]  ,-.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(P,Q)
        pVec2=com(A,B)
        pVec3=com(pVec1[1],pVec2[1])
        errorGens.append( propagatableerrorgen('H',[pVec3[1]] ,.25*wT*pVec1[0]*pVec2[0]*pVec3[0]))
           
    
    return errorGens


