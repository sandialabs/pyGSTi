""" 
Variables for working with the a gatest containing X(pi/2) and Z(pi/2) gates.
"""
import GST as _GST

description = "X(pi/2) and Z(pi/2) gates"

gates = ['Gx','Gz']
prepFiducials = _GST.GateStringTools.gateStringList([(),
                                               ('Gx',),
                                               ('Gx','Gz'),
                                               ('Gx','Gx'),
                                               ('Gx','Gx','Gx'),
                                               ('Gx','Gz','Gx','Gx')]) # for 1Q MUB
                                               
measFiducials = _GST.GateStringTools.gateStringList([(),
                                               ('Gx',),
                                               ('Gz','Gx'),
                                               ('Gx','Gx'),
                                               ('Gx','Gx','Gx'),
                                               ('Gx','Gx','Gz','Gx')])
                                               
germs = _GST.gateStringList( [('Gx',), ('Gz',), ('Gz','Gx','Gx'), ('Gz','Gz','Gx')] )

#Construct a target gateset:  X(pi/2), Y(pi/2)
gs_target = _GST.buildGateset([2],[('Q0',)], ['Gx','Gz'], 
                              [ "X(pi/2,Q0)", "Z(pi/2,Q0)"],
                              rhoExpressions=["0"], EExpressions=["1"], 
                              spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )
