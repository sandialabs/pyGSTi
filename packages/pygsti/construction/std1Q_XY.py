from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing X(pi/2) and Y(pi/2) gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc

description = "X(pi/2) and Y(pi/2) gates"

gates = ['Gx','Gy']
fiducials = _strc.gatestring_list( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                     ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB

germs = _strc.gatestring_list([('Gy',),
 ('Gy','Gy','Gy','Gx',),
 ('Gy','Gx','Gy','Gx','Gx','Gx',),
 ('Gy','Gx','Gy','Gy','Gx','Gx',),
 ('Gy','Gy','Gy','Gx','Gy','Gx',),
 ('Gx',),
 ('Gx','Gy',),
 ('Gx','Gx','Gy','Gx','Gy','Gy',)])

#Construct a target gateset:  X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gx','Gy'],
                                [ "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                prepLabels=["rho0"], prepExpressions=["0"],
                                effectLabels=["E0"], effectExpressions=["1"],
                                spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
