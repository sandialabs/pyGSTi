""" 
Variables for working with the a gatest containing X(pi/4) and Z(pi/2) gates.
"""
import GST as _GST

description = "X(pi/2) and Z(pi/2) gates"

gates = ['Gx','Gz']
                                               
prepFiducials = _GST.GateStringTools.gatestring_list([(),
                                               ('Gx','Gx',),
                                               ('Gx','Gx','Gz'),
                                               ('Gx','Gx','Gx','Gx'),
                                               ('Gx','Gx','Gx','Gx','Gx','Gx'),
                                               ('Gx','Gx','Gz','Gz','Gz')])


measFiducials = _GST.GateStringTools.gatestring_list([(),
                                               ('Gx','Gx',),
                                               ('Gz','Gx','Gx'),
                                               ('Gx','Gx','Gx','Gx'),
                                               ('Gx','Gx','Gx','Gx','Gx','Gx'),
                                               ('Gz','Gz','Gz','Gx','Gx')])

germs = _GST.gatestring_list( [('Gx',), 
                              ('Gz',), 
                              ('Gz','Gz','Gx'), 
                              ('Gz','Gz','Gx','Gz','Gx','Gx','Gx'),
                              ('Gz','Gx','Gz','Gz','Gx','Gx','Gx')] )

#Construct a target gateset:  X(pi/2), Y(pi/2)
gs_target = _GST.build_gateset( [2], [('Q0',)],['Gx','Gz'], 
                                 ["X(pi/4,Q0)", "Z(pi/2,Q0)"],
                                 rhoExpressions=["0"], EExpressions=["1"], 
                                 spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

