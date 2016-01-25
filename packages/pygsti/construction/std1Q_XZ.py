#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" 
Variables for working with the a gatest containing X(pi/2) and Z(pi/2) gates.
"""

import gatestringconstruction as _strc
import gatesetconstruction as _setc

description = "X(pi/2) and Z(pi/2) gates"

gates = ['Gx','Gz']
prepFiducials = _strc.gatestring_list([(),
                                       ('Gx',),
                                       ('Gx','Gz'),
                                       ('Gx','Gx'),
                                       ('Gx','Gx','Gx'),
                                       ('Gx','Gz','Gx','Gx')]) # for 1Q MUB
                                               
measFiducials = _strc.gatestring_list([(),
                                       ('Gx',),
                                       ('Gz','Gx'),
                                       ('Gx','Gx'),
                                       ('Gx','Gx','Gx'),
                                       ('Gx','Gx','Gz','Gx')])
                                               
germs = _strc.gatestring_list( [('Gx',), ('Gz',), ('Gz','Gx','Gx'), ('Gz','Gz','Gx')] )

#Construct a target gateset:  X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gx','Gz'], 
                                [ "X(pi/2,Q0)", "Z(pi/2,Q0)"],
                                rhoExpressions=["0"], EExpressions=["1"], 
                                spamLabelDict={'plus': (0,0), 'minus': (0,-1) } )
