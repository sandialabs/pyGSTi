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

description = "Idle, X(pi/2), and Y(pi/2) gates"

gates = ['Gi','Gx','Gy']
fiducials = _strc.gatestring_list( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                     ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB
prepStrs = effectStrs = fiducials

germs = _strc.gatestring_list( [('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy'),
                                ('Gx', 'Gy', 'Gi'), ('Gx', 'Gi', 'Gy'), ('Gx', 'Gi', 'Gi'), ('Gy', 'Gi', 'Gi'),
                                  ('Gx', 'Gx', 'Gi', 'Gy'), ('Gx', 'Gy', 'Gy', 'Gi'),
                                  ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')] )

#Construct a target gateset: Identity, X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gi','Gx','Gy'],
                                [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["1"],
                                 spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') },
                                 basis='pp')

clifford_compilation = _OrderedDict()
clifford_compilation["Gc0"] = ['Gi',]
clifford_compilation["Gc1"] = ['Gy','Gx',]
clifford_compilation["Gc2"] = ['Gx','Gx','Gx','Gy','Gy','Gy',]
clifford_compilation["Gc3"] = ['Gx','Gx',]
clifford_compilation["Gc4"] = ['Gy','Gy','Gy','Gx','Gx','Gx',]
clifford_compilation["Gc5"] = ['Gx','Gy','Gy','Gy',]
clifford_compilation["Gc6"] = ['Gy','Gy',]
clifford_compilation["Gc7"] = ['Gy','Gy','Gy','Gx',]
clifford_compilation["Gc8"] = ['Gx','Gy',]
clifford_compilation["Gc9"] = ['Gx','Gx','Gy','Gy',]
clifford_compilation["Gc10"] = ['Gy','Gx','Gx','Gx',]
clifford_compilation["Gc11"] = ['Gx','Gx','Gx','Gy',]
clifford_compilation["Gc12"] = ['Gy','Gx','Gx',]
clifford_compilation["Gc13"] = ['Gx','Gx','Gx',]
clifford_compilation["Gc14"] = ['Gx','Gy','Gy','Gy','Gx','Gx','Gx',]
clifford_compilation["Gc15"] = ['Gy','Gy','Gy',]
clifford_compilation["Gc16"] = ['Gx',]
clifford_compilation["Gc17"] = ['Gx','Gy','Gx',]
clifford_compilation["Gc18"] = ['Gy','Gy','Gy','Gx','Gx',]
clifford_compilation["Gc19"] = ['Gx','Gy','Gy',]
clifford_compilation["Gc20"] = ['Gx','Gy','Gy','Gy','Gx',]
clifford_compilation["Gc21"] = ['Gy',]
clifford_compilation["Gc22"] = ['Gx','Gx','Gx','Gy','Gy',]
clifford_compilation["Gc23"] = ['Gx','Gy','Gx','Gx','Gx',]
