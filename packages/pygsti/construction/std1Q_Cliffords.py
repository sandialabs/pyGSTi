from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing all 24 1-qubit Clifford gates
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from collections import OrderedDict as _OrderedDict

description = "The 1-qubit Clifford group"

gates = ["Gc0","Gc1","Gc2","Gc3","Gc4","Gc5","Gc6","Gc7","Gc8",
              "Gc9","Gc10","Gc11","Gc12","Gc13","Gc14","Gc15","Gc16",
              "Gc17","Gc18","Gc19","Gc20","Gc21","Gc22","Gc23"]

#expressions = ["I(Q0)","Y(pi/2,Q0):X(pi/2,Q0)","X(-pi/2,Q0):Y(-pi/2,Q0)",
#                   "X(pi,Q0)","Y(-pi/2,Q0):X(-pi/2,Q0)","X(pi/2,Q0):Y(-pi/2,Q0)",
#                   "Y(pi,Q0)","Y(-pi/2,Q0):X(pi/2,Q0)","X(pi/2,Q0):Y(pi/2,Q0)",
#                   "X(pi,Q0):Y(pi,Q0)","Y(pi/2,Q0):X(-pi/2,Q0)","X(-pi/2,Q0):Y(pi/2,Q0)",
#                   "Y(pi/2,Q0):X(pi,Q0)","X(-pi/2,Q0)","X(pi/2,Q0):Y(-pi/2,Q0):X(-pi/2,Q0)",
#                   "Y(-pi/2,Q0)","X(pi/2,Q0)","X(pi/2,Q0):Y(pi/2,Q0):X(pi/2,Q0)",
#                   "Y(-pi/2,Q0):X(pi,Q0)","X(pi/2,Q0):Y(pi,Q0)","X(pi/2,Q0):Y(-pi/2,Q0):X(pi/2,Q0)",
#                   "Y(pi/2,Q0)","X(-pi/2,Q0):Y(pi,Q0)","X(pi/2,Q0):Y(pi/2,Q0):X(-pi/2,Q0)"]

expressions = ["I(Q0)","X(pi/2,Q0):Y(pi/2,Q0)","Y(-pi/2,Q0):X(-pi/2,Q0)",
                   "X(pi,Q0)","X(-pi/2,Q0):Y(-pi/2,Q0)","Y(-pi/2,Q0):X(pi/2,Q0)",
                   "Y(pi,Q0)","X(pi/2,Q0):Y(-pi/2,Q0)","Y(pi/2,Q0):X(pi/2,Q0)",
                   "Y(pi,Q0):X(pi,Q0)","X(-pi/2,Q0):Y(pi/2,Q0)","Y(pi/2,Q0):X(-pi/2,Q0)",
                   "X(pi,Q0):Y(pi/2,Q0)","X(-pi/2,Q0)","X(-pi/2,Q0):Y(-pi/2,Q0):X(pi/2,Q0)",
                   "Y(-pi/2,Q0)","X(pi/2,Q0)","X(pi/2,Q0):Y(pi/2,Q0):X(pi/2,Q0)",
                   "X(pi,Q0):Y(-pi/2,Q0)","Y(pi,Q0):X(pi/2,Q0)","X(pi/2,Q0):Y(-pi/2,Q0):X(pi/2,Q0)",
                   "Y(pi/2,Q0)","Y(pi,Q0):X(-pi/2,Q0)","X(-pi/2,Q0):Y(pi/2,Q0):X(pi/2,Q0)"]

gs_target = _setc.build_gateset([2],[('Q0',)], gates, expressions)

clifford_compilation = _OrderedDict()
clifford_compilation["Gc0"] = ["Gc0",]
clifford_compilation["Gc1"] = ["Gc1",]
clifford_compilation["Gc2"] = ["Gc2",]
clifford_compilation["Gc3"] = ["Gc3",]
clifford_compilation["Gc4"] = ["Gc4",]
clifford_compilation["Gc5"] = ["Gc5",]
clifford_compilation["Gc6"] = ["Gc6",]
clifford_compilation["Gc7"] = ["Gc7",]
clifford_compilation["Gc8"] = ["Gc8",]
clifford_compilation["Gc9"] = ["Gc9",]
clifford_compilation["Gc10"] = ["Gc10",]
clifford_compilation["Gc11"] = ["Gc11",]
clifford_compilation["Gc12"] = ["Gc12",]
clifford_compilation["Gc13"] = ["Gc13",]
clifford_compilation["Gc14"] = ["Gc14",]
clifford_compilation["Gc15"] = ["Gc15",]
clifford_compilation["Gc16"] = ["Gc16",]
clifford_compilation["Gc17"] = ["Gc17",]
clifford_compilation["Gc18"] = ["Gc18",]
clifford_compilation["Gc19"] = ["Gc19",]
clifford_compilation["Gc20"] = ["Gc20",]
clifford_compilation["Gc21"] = ["Gc21",]
clifford_compilation["Gc22"] = ["Gc22",]
clifford_compilation["Gc23"] = ["Gc23",]