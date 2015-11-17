import GST
from scipy import linalg
import numpy as np

#Note that this has "backwards" convention from our standard gates; there's an overall negative sign on the phase by default

#class myGate():
#    def __init__(self,inputArray):
#        if inputArray.shape[0] != inputArray.shape[1]:
#            raise ValueError("Gate matrix must be square!")
#        self.dim = inputArray.shape[0]
#        self.matrix = inputArray
#    def getNumParams(self, bG0=True):
#        """
#        Get the number of independent parameters which specify this gate.
#
#        Parameters
#        ----------
#        bG0 : bool
#            Whether or not the first row of the gate matrix should be 
#            parameterized.  This is significant in that in the Pauli or 
#            Gell-Mann bases a the first row determines whether the 
#            gate is trace-preserving (TP).
#
#        Returns
#        -------
#        int
#           the number of independent parameters.
#        """
#        # if bG0 == True, need to subtract the number of parameters which *only*
#        #  parameterize the first row of the final gate matrix
#        if bG0:
#            return self.dim**2
#        else:
#            #subtract params for the first row
#            return self.dim**2 - self.dim 
#    def copy(self):
#        return myGate(self.matrix)


sigmaX = np.array([[0,1.],
                   [1.,0]])

sigmaZ = np.array([[1.,0],
                   [0,-1.]])

sigmaN = -0.5 * sigmaZ + np.sqrt(3)/2 * sigmaX

def N1qubit(theta):
    n = linalg.expm(-1j/2. * theta * sigmaN)
    return n

gateNmx = N1qubit(np.pi/2)

gateNSO = np.kron(np.conj(gateNmx),gateNmx)
gateNSOpp = GST.BasisTools.basisChg_StdToPauliProd(gateNSO)

gs_target = GST.buildGateset([2],[('Q0',)], ['Gz','Gn'], 
                              [ "Z(-pi/2,Q0)", "I(Q0)"],
                              rhoExpressions=["0"], EExpressions=["1"], 
                              spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )

gs_target.set_gate('Gn',GST.Gate.FullyParameterizedGate(gateNSOpp))

prepFiducials = GST.GateStringTools.gateStringList([(),
                                               ('Gn',),
                                               ('Gn','Gn'),
                                               ('Gn','Gz','Gn'),
                                               ('Gn','Gn','Gn',),
                                               ('Gn','Gz','Gn','Gn','Gn')]) # for 1Q MUB
                                               
measFiducials = GST.GateStringTools.gateStringList([(),
                                               ('Gn',),
                                               ('Gn','Gn'),
                                               ('Gn','Gz','Gn'),
                                               ('Gn','Gn','Gn',),
                                               ('Gn','Gn','Gn','Gz','Gn')]) # for 1Q MUB

#germs = 