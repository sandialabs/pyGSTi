from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines canonical 1-qubit quantities used in Randomized Benchmarking"""

from ... import construction as _cnst
from . import rbobjs as _rbobjs

import numpy as _np
from collections import OrderedDict as _OrderedDict

# "canonical" clifford gateset
gs_cliff_canonical = _cnst.build_gateset(
    [2],[('Q0',)], ['Gi','Gxp2','Gxp','Gxmp2','Gyp2','Gyp','Gymp2'], 
    [ "I(Q0)","X(pi/2,Q0)", "X(pi,Q0)", "X(-pi/2,Q0)",
      "Y(pi/2,Q0)", "Y(pi,Q0)", "Y(-pi/2,Q0)"],
    prepLabels=["rho0"], prepExpressions=["0"],
    effectLabels=["E0"], effectExpressions=["1"], 
    spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

#Mapping of all 1Q cliffords onto "canonical" set:
#  Definitions taken from arXiv:1508.06676v1
#  0-indexing used instead of 1-indexing
clifford_to_canonical = _OrderedDict()
clifford_to_canonical["Gc0"] = ['Gi',]
clifford_to_canonical["Gc1"] = ['Gyp2','Gxp2']
clifford_to_canonical["Gc2"] = ['Gxmp2','Gymp2']
clifford_to_canonical["Gc3"] = ['Gxp',]
clifford_to_canonical["Gc4"] = ['Gymp2','Gxmp2']
clifford_to_canonical["Gc5"] = ['Gxp2','Gymp2']
clifford_to_canonical["Gc6"] = ['Gyp',]
clifford_to_canonical["Gc7"] = ['Gymp2','Gxp2']
clifford_to_canonical["Gc8"] = ['Gxp2','Gyp2']
clifford_to_canonical["Gc9"] = ['Gxp','Gyp']
clifford_to_canonical["Gc10"] = ['Gyp2','Gxmp2']
clifford_to_canonical["Gc11"] = ['Gxmp2','Gyp2']
clifford_to_canonical["Gc12"] = ['Gyp2','Gxp']
clifford_to_canonical["Gc13"] = ['Gxmp2']
clifford_to_canonical["Gc14"] = ['Gxp2','Gymp2','Gxmp2']
clifford_to_canonical["Gc15"] = ['Gymp2']
clifford_to_canonical["Gc16"] = ['Gxp2']
clifford_to_canonical["Gc17"] = ['Gxp2','Gyp2','Gxp2']
clifford_to_canonical["Gc18"] = ['Gymp2','Gxp']
clifford_to_canonical["Gc19"] = ['Gxp2','Gyp']
clifford_to_canonical["Gc20"] = ['Gxp2','Gymp2','Gxp2']
clifford_to_canonical["Gc21"] = ['Gyp2']
clifford_to_canonical["Gc22"] = ['Gxmp2','Gyp']
clifford_to_canonical["Gc23"] = ['Gxp2','Gyp2','Gxmp2']

# Mapping the "canonical" gate set onto the "primitive"
# gate set containing Gi, Gx, Gy, using the natural
# mapping
canonical_to_XYI = _OrderedDict()
canonical_to_XYI['Gi'] = ['Gi']
canonical_to_XYI['Gxp2'] = ['Gx']
canonical_to_XYI['Gxp'] = ['Gx','Gx']
canonical_to_XYI['Gxmp2'] = ['Gx','Gx','Gx']
canonical_to_XYI['Gyp2'] = ['Gy']
canonical_to_XYI['Gyp'] = ['Gy','Gy']
canonical_to_XYI['Gymp2'] = ['Gy','Gy','Gy']

# Mapping the Clifford gate set onto the "primitive"
# gate set containing Gi, Gx, Gy, via the composition
# of the clifford -> canonical and canonical ->
# primitive maps
clifford_to_XYI = _cnst.compose_alias_dicts(clifford_to_canonical,
                                                      canonical_to_XYI)

# full 1Q Clifford gateset (24 gates)
gs_clifford_target = _cnst.build_alias_gateset(gs_cliff_canonical,
                                             clifford_to_canonical)

# The single-qubit Clifford group
clifford_group = _rbobjs.MatrixGroup(gs_clifford_target.gates.values(),
                                  gs_clifford_target.gates.keys() )
