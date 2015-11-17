""" 
Variables for working with the a gatest containing X(pi/2) and Y(pi/2) gates.
"""
import GST as _GST

description = "X(pi/2) and Y(pi/2) gates"

gates = ['Gx','Gy']
fiducials = _GST.gateStringList( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                       ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB
#germs = _GST.gateStringList( [('Gx',), ('Gy',), ('Gx', 'Gy'),
#                              ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')] )
#These germs are not correct

germs = _GST.gateStringList([('Gx',), ('Gy',), ('Gy','Gx','Gx'), ('Gy','Gy','Gx')])

#Construct a target gateset:  X(pi/2), Y(pi/2)
gs_target = _GST.buildGateset([2],[('Q0',)], ['Gx','Gy'],
                              [ "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                              rhoExpressions=["0"], EExpressions=["1"], 
                              spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )
