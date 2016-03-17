#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" 
Variables for working with the a gatest containing Idle, X(pi/2) and Y(pi/2) gates.
"""

import gatestringconstruction as _strc
import gatesetconstruction as _setc

description = "Idle, X(pi/2), and Y(pi/2) gates"

gates = ['Gi','Gx','Gy']
fiducials = _strc.gatestring_list( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                     ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB
germs = _strc.gatestring_list( [('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy'),
                                ('Gx', 'Gy', 'Gi'), ('Gx', 'Gi', 'Gy'), ('Gx', 'Gi', 'Gi'), ('Gy', 'Gi', 'Gi'),
                                  ('Gx', 'Gx', 'Gi', 'Gy'), ('Gx', 'Gy', 'Gy', 'Gi'),
                                  ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')] )

#Construct a target gateset: Identity, X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gi','Gx','Gy'], 
                                [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                 rhoLabelList=["rho0"], rhoExpressions=["0"],
                                 ELabelList=["E0"], EExpressions=["1"], 
                                 spamLabelDict={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
