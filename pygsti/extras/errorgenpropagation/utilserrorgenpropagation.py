
from pygsti.extras.errorgenpropagation.localstimerrorgen import localstimerrorgen
from numpy import conjugate

'''
Returns the Commutator of two errors
'''
def commute_errors(ErG1,ErG2, weightFlip=1.0, BCHweight=1.0):
    def com(P1,P2):
        P3=P1*P2-P2*P1
        return (P3.weight,P3*conjugate(P3.weight))
    
    def acom(P1,P2):
        P3=P1*P2+P2*P1
        return (P3.weight,P3*conjugate(P3.weight))
    
    def labelMultiply(P1,P2):
        P3=P1*P2
        return (P3.weight,P3*conjugate(P3.weight))
    
    errorGens=[]
    
    wT=weightFlip*BCHweight
    
    if ErG1.getType()=='H' and ErG2.getType()=='H':
        pVec=com(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[0])
        errorGens.append( localstimerrorgen( 'H' , [pVec[1]] , -1j*wT *pVec[0] ) )
        
    elif ErG1.getType()=='H' and ErG2.getType()=='S':
        pVec=com(ErG2.basis_element_labels[0] , ErG1.basis_element_labels[0])
        errorGens.append( localstimerrorgen( 'C' , [ErG2.basis_element_labels[0] , pVec[1]] , 1j*wT*pVec[0] ) )
         
    elif ErG1.getType()=='S' and ErG2.getType()=='H':
        pVec=com(ErG2.basis_element_labels[0] , ErG1.basis_element_labels[0])
        errorGens.append( localstimerrorgen( 'C' , [ErG2.basis_element_labels[0] , pVec[1]] , -1j*wT *pVec[0] ) )
          
    elif ErG1.getType()=='H' and ErG2.getType()=='C':
        pVec1=com(ErG2.basis_element_labels[0] , ErG1.basis_element_labels[0])
        errorGens.append( localstimerrorgen('C' , [pVec1[1], ErG2.basis_element_labels[1]] , 1j*wT*pVec1[0] ) )
        pVec2=com(ErG2.basis_element_labels[1] , ErG1.basis_element_labels[0])
        errorGens.append( localstimerrorgen('C' , [pVec2[1] , ErG2.basis_element_labels[0]] , 1j*wT*pVec2[0] ) )
                          
    elif ErG1.getType()=='C' and ErG2.getType()=='H':
        errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
    
    elif ErG1.getType()=='H' and ErG2.getType()=='A':
        pVec1 = com(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[0])
        errorGens.append( localstimerrorgen('A'  , [pVec1[1] , ErG2.basis_element_labels[1]] , -1j*wT*pVec1[0]) )
        pVec2 = com(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[1])
        errorGens.append( localstimerrorgen('A'  , [ErG2.basis_element_labels[0], pVec2[1]] , -1j*wT*pVec2[0] ) )
                          
    elif ErG1.getType()=='A' and ErG2.getType()=='H':
        errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)

    elif ErG1.getType()=='S' and ErG2.getType()=='S':
        errorGens.append( localstimerrorgen('H', ErG1.basis_element_labels[0],0 ))
                         
    elif ErG1.getType()=='S' and ErG2.getType()=='C':
       pVec1=labelMultiply(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[0])
       pVec2=labelMultiply(ErG2.basis_element_labels[1] , ErG1.basis_element_labels[0])
       errorGens.append( localstimerrorgen( 'A' , [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]  ))
       pVec1 = labelMultiply(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[1])
       pVec2 = labelMultiply(ErG2.basis_element_labels[0] , ErG1.basis_element_labels[0])
       errorGens.append( localstimerrorgen( 'A', [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
       pVec1 =acom(ErG2.basis_element_labels[0], ErG2.basis_element_labels[1])
       pVec2 = labelMultiply(pVec1[1],ErG1.basis_element_labels[0])
       errorGens.append( localstimerrorgen( 'A'  ,[pVec2[1], ErG1.basis_element_labels[0]] , -1j*.5*wT*pVec1[0]*pVec2[0]))
       pVec1=acom(ErG2.basis_element_labels[0], ErG2.basis_element_labels[1])
       pVec2=labelMultiply(ErG1.basis_element_labels[0],pVec1[1])
       errorGens.append( localstimerrorgen( 'A',   [ErG1.basis_element_labels[0] ,pVec2[1]],-1j*.5*wT*pVec1[0]*pVec2[0]))
                         
    elif ErG1.getType() == 'C' and ErG2.getType() == 'S':
       errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
                         
    elif ErG1.getType() == 'S' and ErG2.getType() == 'A':
       pVec1 =labelMultiply(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[0])
       pVec2=labelMultiply(ErG2.basis_element_labels[1] , ErG1.basis_element_labels[0])
       errorGens.append( localstimerrorgen( 'C', [pVec1[1], pVec2[1]] ,1j*wT*pVec1[0]*pVec2[0] ))
       pVec1=labelMultiply(ErG1.basis_element_labels[0] , ErG2.basis_element_labels[1])
       pVec2=labelMultiply(ErG2.basis_element_labels[0] , ErG1.basis_element_labels[0])
       errorGens.append( localstimerrorgen( 'C', [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
       pVec1 = com(ErG2.basis_element_labels[0] , ErG2.basis_element_labels[1])
       pVec2 = com(ErG1.basis_element_labels[0],pVec1[1])
       errorGens.append( localstimerrorgen( 'A', [ErG1.basis_element_labels[0], pVec2[1]] ,-.5*wT*pVec1[0]*pVec2[0]))
                         
    elif ErG1.getType() == 'A' and ErG1.getType() == 'S':
       errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
                         
    elif ErG1.getType() == 'C' and ErG2.getType() == 'C':
        A=ErG1.basis_element_labels[0]
        B=ErG1.basis_element_labels[1]
        P=ErG2.basis_element_labels[0]
        Q=ErG2.basis_element_labels[1]
        pVec1 = labelMultiply(A,P)
        pVec2 =labelMultiply(Q,B)
        errorGens.append( localstimerrorgen( 'A', [pVec1[1], pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0] ))        
        pVec1 = labelMultiply(A,Q)
        pVec2 =labelMultiply(P,B)
        errorGens.append( localstimerrorgen( 'A'  , [pVec1[1] , pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = labelMultiply(B,P)
        pVec2 =labelMultiply(Q,A)                 
        errorGens.append( localstimerrorgen( 'A'  , [pVec1[1] , pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = labelMultiply(B,Q)
        pVec2 =labelMultiply(P,A)
        errorGens.append( localstimerrorgen( 'A'  , [pVec1[1] , pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(A,B)
        pVec2=com(P,pVec1[1])
        errorGens.append( localstimerrorgen( 'A'  , [pVec2[1] , Q ], -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(A,B)
        pVec2=com(Q,pVec1[1])
        errorGens.append( localstimerrorgen( 'A'  , [pVec2[1], P] , -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(P,Q)
        pVec2=com(pVec1[1],A)
        errorGens.append( localstimerrorgen( 'A' , [pVec2[1] , B] , -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(P,Q)
        pVec2=com(pVec1[1],B)
        errorGens.append( localstimerrorgen( 'A' , [pVec2[1] , A ] , -.5*1j*wT*pVec1[0]*pVec2[0]))
        pVec1=acom(A,B)
        pVec2=acom(P,Q)
        pVec3=com(pVec1[1],pVec2[1])
        errorGens.append( localstimerrorgen( 'H', [pVec3[1]] ,.25*1j*wT*pVec1[0]*pVec2[0]*pVec3[0]))

    elif ErG1.getType() == 'C' and ErG2.getType() == 'A':
        A=ErG1.basis_element_labels[0]
        B=ErG1.basis_element_labels[1]
        P=ErG2.basis_element_labels[0]
        Q=ErG2.basis_element_labels[1]
        pVec1 = labelMultiply(A,P)
        pVec2 =labelMultiply(Q,B)
        errorGens.append( localstimerrorgen('C' , [pVec1[1],pVec2[1]] , 1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = labelMultiply(A,Q)
        pVec2 =labelMultiply(P,B)
        errorGens.append( localstimerrorgen('C' ,[pVec1[1],pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = labelMultiply(B,P)
        pVec2 =labelMultiply(Q,A)
        errorGens.append( localstimerrorgen('C' , [pVec1[1],pVec2[1]] , 1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = labelMultiply(P,A)
        pVec2 =labelMultiply(B,Q)
        errorGens.append( localstimerrorgen('C'  ,[pVec1[1],pVec2[1]] , -1j*wT*pVec1[0]*pVec2[0]))
        pVec1 = com(P,Q)
        pVec2 =com(A,pVec1[1])
        errorGens.append( localstimerrorgen('A'  , [pVec2[1] , B] , .5*wT*pVec1[0]*pVec2[0] ))
        pVec1 = com(P,Q)
        pVec2 =com(B,pVec1[1])
        errorGens.append( localstimerrorgen('A' , [pVec2[1],  A ], .5*wT*pVec1[0]*pVec2[0] ))
        pVec1 = acom(A,B)
        pVec2 =com(P,pVec1[1])
        errorGens.append( localstimerrorgen('C', [pVec2[1] , Q ], .5*1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = acom(A,B)
        pVec2 =com(Q,pVec1[1])
        errorGens.append( localstimerrorgen('C',[pVec2[1],P ],-.5*1j*wT*pVec1[0]*pVec2[0] ))
        pVec1 = com(P,Q)
        pVec2 =acom(A,B)
        pVec3=com(pVec1[1],pVec2[1])
        errorGens.append( localstimerrorgen('H',[pVec3[1]],-.25*wT*pVec1[0]*pVec2[0]*pVec3[0]))
    
    elif ErG1.getType() == 'A' and ErG2.getType() == 'C':
        errorGens = commute_errors(ErG2,ErG1,weightFlip=-1.0,BCHweight=BCHweight)
                         
    elif ErG1.getType() == 'A' and ErG2.getType() == 'A':
        A=ErG1.basis_element_labels[0]
        B=ErG1.basis_element_labels[1]
        P=ErG2.basis_element_labels[0]
        Q=ErG2.basis_element_labels[1]
        pVec1=labelMultiply(Q,B)
        pVec2=labelMultiply(A,P)
        errorGens.append(localstimerrorgen('A',[pVec1[1],pVec2[1]] ,-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=labelMultiply(P,A)
        pVec2=labelMultiply(B,Q)
        errorGens.append(localstimerrorgen('A',[pVec1[1],pVec2[1]],-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=labelMultiply(B,P)
        pVec2=labelMultiply(Q,A)
        errorGens.append(localstimerrorgen('A',[pVec1[1],pVec2[1]],-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=labelMultiply(A,Q)
        pVec2=labelMultiply(P,B)
        errorGens.append(localstimerrorgen('A',[pVec1[1],pVec2[1]],-1j*wT*pVec1[0]*pVec2[0]))
        pVec1=com(P,Q)
        pVec2=com(B,pVec1[1])
        errorGens.append(localstimerrorgen('C',[pVec2[1],A],.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(P,Q)
        pVec2=com(A,pVec1[1])
        errorGens.append(localstimerrorgen('C',[pVec2[1],B] ,-.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(A,B)
        pVec2=com(P,pVec1[1])
        errorGens.append(localstimerrorgen('C', [pVec2[1],Q] ,.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(A,B)
        pVec2=com(Q,pVec1[1])
        errorGens.append(localstimerrorgen('C', [pVec2[1],P]  ,-.5*wT*pVec1[0]*pVec2[0]))
        pVec1=com(P,Q)
        pVec2=com(A,B)
        pVec3=com(pVec1[1],pVec2[1])
        errorGens.append( localstimerrorgen('H',[pVec3[1]] ,.25*wT*pVec1[0]*pVec2[0]*pVec3[0]))
           
    
    return errorGens