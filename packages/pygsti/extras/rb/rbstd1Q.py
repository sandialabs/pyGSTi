""" Defines canonical 1-qubit Clifford gateset and group"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from ... import construction as _cnst
from . import rbobjs as _rbobjs
from collections import OrderedDict as _OrderedDict

# A generating gateset for the 1-qubit the cliffords
gs_clifford_generators = _cnst.build_gateset(
    [2],[('Q0',)], ['Gi','Gxp2','Gxp','Gxmp2','Gyp2','Gyp','Gymp2'], 
    [ "I(Q0)","X(pi/2,Q0)", "X(pi,Q0)", "X(-pi/2,Q0)",
      "Y(pi/2,Q0)", "Y(pi,Q0)", "Y(-pi/2,Q0)"])

# Mapping of all 1Q cliffords onto the generating set given above.
# This uses the compilations in arXiv:1508.06676v1
clifford_to_generators = _OrderedDict()
clifford_to_generators["Gc0"] = ['Gi',]
clifford_to_generators["Gc1"] = ['Gyp2','Gxp2']
clifford_to_generators["Gc2"] = ['Gxmp2','Gymp2']
clifford_to_generators["Gc3"] = ['Gxp',]
clifford_to_generators["Gc4"] = ['Gymp2','Gxmp2']
clifford_to_generators["Gc5"] = ['Gxp2','Gymp2']
clifford_to_generators["Gc6"] = ['Gyp',]
clifford_to_generators["Gc7"] = ['Gymp2','Gxp2']
clifford_to_generators["Gc8"] = ['Gxp2','Gyp2']
clifford_to_generators["Gc9"] = ['Gxp','Gyp']
clifford_to_generators["Gc10"] = ['Gyp2','Gxmp2']
clifford_to_generators["Gc11"] = ['Gxmp2','Gyp2']
clifford_to_generators["Gc12"] = ['Gyp2','Gxp']
clifford_to_generators["Gc13"] = ['Gxmp2']
clifford_to_generators["Gc14"] = ['Gxp2','Gymp2','Gxmp2']
clifford_to_generators["Gc15"] = ['Gymp2']
clifford_to_generators["Gc16"] = ['Gxp2']
clifford_to_generators["Gc17"] = ['Gxp2','Gyp2','Gxp2']
clifford_to_generators["Gc18"] = ['Gymp2','Gxp']
clifford_to_generators["Gc19"] = ['Gxp2','Gyp']
clifford_to_generators["Gc20"] = ['Gxp2','Gymp2','Gxp2']
clifford_to_generators["Gc21"] = ['Gyp2']
clifford_to_generators["Gc22"] = ['Gxmp2','Gyp']
clifford_to_generators["Gc23"] = ['Gxp2','Gyp2','Gxmp2']

# A gateset containing the 1Q Clifford gateset (24 gates)
gs_target = _cnst.build_alias_gateset(gs_clifford_generators,clifford_to_generators)

# The single-qubit Clifford group
clifford_group = _rbobjs.MatrixGroup(gs_target.gates.values(),gs_target.gates.keys() )
