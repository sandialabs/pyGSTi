from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing Idle, Z(pi/2) and rot(X=pi/2, Y=sqrt(3)/2) gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc


gs_target = _setc.build_gateset([2],[('Q0',)], ['Gz','Gn'],
                                [ "Z(pi/2,Q0)", "N(pi/2, sqrt(3)/2, 0, -0.5, Q0)"],
                                prepLabels=["rho0"], prepExpressions=["0"],
                                effectLabels=["E0"], effectExpressions=["1"],
                                spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

prepFiducials = _strc.gatestring_list([(),
                                       ('Gn',),
                                       ('Gn','Gn'),
                                       ('Gn','Gz','Gn'),
                                       ('Gn','Gn','Gn',),
                                       ('Gn','Gz','Gn','Gn','Gn')]) # for 1Q MUB

measFiducials = _strc.gatestring_list([(),
                                       ('Gn',),
                                       ('Gn','Gn'),
                                       ('Gn','Gz','Gn'),
                                       ('Gn','Gn','Gn',),
                                       ('Gn','Gn','Gn','Gz','Gn')]) # for 1Q MUB

germs = _strc.gatestring_list([ ('Gz',),
                                ('Gn',),
                                ('Gn','Gn','Gz','Gn','Gz'),
                                ('Gn','Gz','Gn','Gz','Gz'),
                                ('Gn','Gz','Gn','Gn','Gz','Gz'),
                                ('Gn','Gn','Gz','Gn','Gz','Gz'),
                                ('Gn','Gn','Gn','Gz','Gz','Gz') ])
