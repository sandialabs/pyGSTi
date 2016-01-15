import GST
from scipy import linalg
import numpy as np

gs_target = GST.build_gateset([2],[('Q0',)], ['Gz','Gn'], 
                             [ "Z(pi/2,Q0)", "N(pi/2, sqrt(3)/2, 0, -0.5, Q0)"],
                             rhoExpressions=["0"], EExpressions=["1"], 
                             spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )

prepFiducials = GST.GateStringTools.gatestring_list([(),
                                                    ('Gn',),
                                                    ('Gn','Gn'),
                                                    ('Gn','Gz','Gn'),
                                                    ('Gn','Gn','Gn',),
                                                    ('Gn','Gz','Gn','Gn','Gn')]) # for 1Q MUB
                                               
measFiducials = GST.GateStringTools.gatestring_list([(),
                                                    ('Gn',),
                                                    ('Gn','Gn'),
                                                    ('Gn','Gz','Gn'),
                                                    ('Gn','Gn','Gn',),
                                                    ('Gn','Gn','Gn','Gz','Gn')]) # for 1Q MUB

germs = GST.GateStringTools.gatestring_list([ ('Gz',),
                                             ('Gn',),
                                             ('Gn','Gn','Gz','Gn','Gz'),
                                             ('Gn','Gz','Gn','Gz','Gz'),
                                             ('Gn','Gz','Gn','Gn','Gz','Gz'),
                                             ('Gn','Gn','Gz','Gn','Gz','Gz'),
                                             ('Gn','Gn','Gn','Gz','Gz','Gz') ])


#bOLD = False 
#if bOLD:  # Kenny's initial implementation of gates
#    #Note that this has "backwards" convention from our standard gates; there's an overall negative sign on the phase by default
#    sigmaN = -0.5 * GST.BT.sigmaz + np.sqrt(3)/2 * GST.BT.sigmax
#
#    def N1qubit(theta):
#        n = linalg.expm(-1j/2. * theta * sigmaN)
#        return n
#
#    gateNmx = N1qubit(np.pi/2)
#
#    gateNSO = np.kron(np.conj(gateNmx),gateNmx) # incorrect superop generation
#    gateNSOpp = GST.BasisTools.std_to_pp(gateNSO)
#
#    gs_target_old = GST.build_gateset([2],[('Q0',)], ['Gz','Gn'], 
#                                     [ "Z(-pi/2,Q0)", "I(Q0)"],
#                                     rhoExpressions=["0"], EExpressions=["1"], 
#                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )
#
#    gs_target_old.set_gate('Gn',GST.Gate.FullyParameterizedGate(gateNSOpp))
#
#    #NOTE: if change gateNSO to correct kron:
#    # gateNSO = np.kron(gateNmx,np.conj(gateNmx))
#    # and also switched Z angle from -pi/2 to the pi/2 the Frobenius
#    # diff check below will be zero
#    # print "CHECK: ", gs_target_old.diff_frobenius(gs_target)
