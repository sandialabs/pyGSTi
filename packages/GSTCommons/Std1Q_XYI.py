""" 
Variables for working with the a gatest containing Idle, X(pi/2) and Y(pi/2) gates.
"""
import GST as _GST

description = "Idle, X(pi/2), and Y(pi/2) gates"

gates = ['Gi','Gx','Gy']
fiducials = _GST.gateStringList( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                       ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB
germs = _GST.gateStringList( [('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy'),
                                  ('Gx', 'Gy', 'Gi'), ('Gx', 'Gi', 'Gy'), ('Gx', 'Gi', 'Gi'), ('Gy', 'Gi', 'Gi'),
                                  ('Gx', 'Gx', 'Gi', 'Gy'), ('Gx', 'Gy', 'Gy', 'Gi'),
                                  ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')] )

#Construct a target gateset: Identity, X(pi/2), Y(pi/2)
gs_target = _GST.buildGateset([2],[('Q0',)], ['Gi','Gx','Gy'], 
                              [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                              rhoExpressions=["0"], EExpressions=["1"], 
                              spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )
