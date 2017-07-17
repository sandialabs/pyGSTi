from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing Idle, X(pi/2) and Y(pi/2) gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from collections import OrderedDict as _OrderedDict

#Qubit 1 == spectator
description = "Idle, X(pi/2), and Y(pi/2) gates"

gates = ['Gii','Gxi','Gyi']
fiducials = _strc.gatestring_list( [ (), ('Gxi',), ('Gyi',), ('Gxi','Gxi') ] )
#                                     ('Gxi','Gxi','Gxi'), ('Gyi','Gyi','Gyi') ] ) # for 1Q MUB
prepStrs = effectStrs = fiducials

germs = _strc.gatestring_list( [('Gxi',), ('Gyi',), ('Gii',), ('Gxi', 'Gyi'),
                                ('Gxi', 'Gyi', 'Gii'), ('Gxi', 'Gii', 'Gyi'), ('Gxi', 'Gii', 'Gii'), ('Gyi', 'Gii', 'Gii'),
                                  ('Gxi', 'Gxi', 'Gii', 'Gyi'), ('Gxi', 'Gyi', 'Gyi', 'Gii'),
                                  ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi')] )

#Construct a target gateset: Identity, X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gii','Gxi','Gyi'],
                                [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["1"],
                                 spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
