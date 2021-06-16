#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Variables for working with the a model containing all 24 1-qubit Clifford gates
"""

import sys as _sys
from collections import OrderedDict as _OrderedDict

from ...models import modelconstruction as _setc
from .. import stdtarget as _stdtarget

description = "The 1-qubit Clifford group"

gates = ["Gc0", "Gc1", "Gc2", "Gc3", "Gc4", "Gc5", "Gc6", "Gc7", "Gc8",
         "Gc9", "Gc10", "Gc11", "Gc12", "Gc13", "Gc14", "Gc15", "Gc16",
         "Gc17", "Gc18", "Gc19", "Gc20", "Gc21", "Gc22", "Gc23"]

#expressions = ["I(Q0)","Y(pi/2,Q0):X(pi/2,Q0)","X(-pi/2,Q0):Y(-pi/2,Q0)",
#                   "X(pi,Q0)","Y(-pi/2,Q0):X(-pi/2,Q0)","X(pi/2,Q0):Y(-pi/2,Q0)",
#                   "Y(pi,Q0)","Y(-pi/2,Q0):X(pi/2,Q0)","X(pi/2,Q0):Y(pi/2,Q0)",
#                   "X(pi,Q0):Y(pi,Q0)","Y(pi/2,Q0):X(-pi/2,Q0)","X(-pi/2,Q0):Y(pi/2,Q0)",
#                   "Y(pi/2,Q0):X(pi,Q0)","X(-pi/2,Q0)","X(pi/2,Q0):Y(-pi/2,Q0):X(-pi/2,Q0)",
#                   "Y(-pi/2,Q0)","X(pi/2,Q0)","X(pi/2,Q0):Y(pi/2,Q0):X(pi/2,Q0)",
#                   "Y(-pi/2,Q0):X(pi,Q0)","X(pi/2,Q0):Y(pi,Q0)","X(pi/2,Q0):Y(-pi/2,Q0):X(pi/2,Q0)",
#                   "Y(pi/2,Q0)","X(-pi/2,Q0):Y(pi,Q0)","X(pi/2,Q0):Y(pi/2,Q0):X(-pi/2,Q0)"]

expressions = ["I(Q0)", "X(pi/2,Q0):Y(pi/2,Q0)", "Y(-pi/2,Q0):X(-pi/2,Q0)",
               "X(pi,Q0)", "X(-pi/2,Q0):Y(-pi/2,Q0)", "Y(-pi/2,Q0):X(pi/2,Q0)",
               "Y(pi,Q0)", "X(pi/2,Q0):Y(-pi/2,Q0)", "Y(pi/2,Q0):X(pi/2,Q0)",
               "Y(pi,Q0):X(pi,Q0)", "X(-pi/2,Q0):Y(pi/2,Q0)", "Y(pi/2,Q0):X(-pi/2,Q0)",
               "X(pi,Q0):Y(pi/2,Q0)", "X(-pi/2,Q0)", "X(-pi/2,Q0):Y(-pi/2,Q0):X(pi/2,Q0)",
               "Y(-pi/2,Q0)", "X(pi/2,Q0)", "X(pi/2,Q0):Y(pi/2,Q0):X(pi/2,Q0)",
               "X(pi,Q0):Y(-pi/2,Q0)", "Y(pi,Q0):X(pi/2,Q0)", "X(pi/2,Q0):Y(-pi/2,Q0):X(pi/2,Q0)",
               "Y(pi/2,Q0)", "Y(pi,Q0):X(-pi/2,Q0)", "X(-pi/2,Q0):Y(pi/2,Q0):X(pi/2,Q0)"]

_target_model = _setc.create_explicit_model([('Q0',)], gates, expressions)

_gscache = {("full", "auto"): _target_model}


def target_model(parameterization_type="full", sim_type="auto"):
    """
    Returns a copy of the target model in the given parameterization.

    Parameters
    ----------
    parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
        The gate and SPAM vector parameterization type. See
        :function:`Model.set_all_parameterizations` for all allowed values.

    sim_type : {"auto", "matrix", "map", "termorder:X" }
        The simulator type to be used for model calculations (leave as
        "auto" if you're not sure what this is).

    Returns
    -------
    Model
    """
    return _stdtarget._copy_target(_sys.modules[__name__], parameterization_type,
                                   sim_type, _gscache)


clifford_compilation = _OrderedDict()
clifford_compilation["Gc0"] = ["Gc0", ]
clifford_compilation["Gc1"] = ["Gc1", ]
clifford_compilation["Gc2"] = ["Gc2", ]
clifford_compilation["Gc3"] = ["Gc3", ]
clifford_compilation["Gc4"] = ["Gc4", ]
clifford_compilation["Gc5"] = ["Gc5", ]
clifford_compilation["Gc6"] = ["Gc6", ]
clifford_compilation["Gc7"] = ["Gc7", ]
clifford_compilation["Gc8"] = ["Gc8", ]
clifford_compilation["Gc9"] = ["Gc9", ]
clifford_compilation["Gc10"] = ["Gc10", ]
clifford_compilation["Gc11"] = ["Gc11", ]
clifford_compilation["Gc12"] = ["Gc12", ]
clifford_compilation["Gc13"] = ["Gc13", ]
clifford_compilation["Gc14"] = ["Gc14", ]
clifford_compilation["Gc15"] = ["Gc15", ]
clifford_compilation["Gc16"] = ["Gc16", ]
clifford_compilation["Gc17"] = ["Gc17", ]
clifford_compilation["Gc18"] = ["Gc18", ]
clifford_compilation["Gc19"] = ["Gc19", ]
clifford_compilation["Gc20"] = ["Gc20", ]
clifford_compilation["Gc21"] = ["Gc21", ]
clifford_compilation["Gc22"] = ["Gc22", ]
clifford_compilation["Gc23"] = ["Gc23", ]
