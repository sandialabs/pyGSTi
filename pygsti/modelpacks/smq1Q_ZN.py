"""
Variables for working with the a model containing Idle, Z(pi/2) and rot(X=pi/2, Y=sqrt(3)/2) gates.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict
from pygsti.construction import circuitconstruction as _strc
from pygsti.construction import modelconstruction as _setc

from pygsti.modelpacks._modelpack import SMQModelPack


class _Module(SMQModelPack):
    description = "None"

    gates = None

    germs = _strc.circuit_list([(('Gz', 0), ), (('Gn', 0), ), (('Gz', 0), ('Gn', 0)), (('Gz', 0), ('Gz', 0), ('Gn', 0)),
                                (('Gz', 0), ('Gn', 0), ('Gn', 0)),
                                (('Gz', 0), ('Gz', 0), ('Gn', 0), ('Gz', 0), ('Gn', 0), ('Gn', 0))],
                               line_labels=[0])

    germs_lite = germs[:]

    fiducials = None

    prepStrs = _strc.circuit_list([(), (('Gn', 0), ), (('Gn', 0), ('Gn', 0)), (('Gn', 0), ('Gz', 0), ('Gn', 0)),
                                   (('Gn', 0), ('Gn', 0), ('Gn', 0)),
                                   (('Gn', 0), ('Gz', 0), ('Gn', 0), ('Gn', 0), ('Gn', 0))],
                                  line_labels=[0])

    effectStrs = _strc.circuit_list([(), (('Gn', 0), ), (('Gn', 0), ('Gn', 0)), (('Gn', 0), ('Gz', 0), ('Gn', 0)),
                                     (('Gn', 0), ('Gn', 0), ('Gn', 0)),
                                     (('Gn', 0), ('Gn', 0), ('Gn', 0), ('Gz', 0), ('Gn', 0))],
                                    line_labels=[0])

    clifford_compilation = None
    global_fidPairs = [(0, 0), (2, 3), (5, 2), (5, 4)]
    pergerm_fidPairsDict = {
        ('Gz', ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gn', ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gn'): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gn', 'Gn'): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gz', 'Gn'): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gz', 'Gn', 'Gz', 'Gn', 'Gn'): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)]
    }
    global_fidPairs_lite = [(0, 0), (2, 3), (5, 2), (5, 4)]
    pergerm_fidPairsDict_lite = {
        ('Gz', ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gn', ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gn'): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gn', 'Gn'): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gz', 'Gn'): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gz', 'Gn', 'Gz', 'Gn', 'Gn'): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)]
    }

    @property
    def _target_model(self):
        return _setc.build_explicit_model([0], [('Gz', 0), ('Gn', 0)], ['Z(pi/2,0)', 'N(pi/2, sqrt(3)/2, 0, -0.5, 0)'])


import sys
sys.modules[__name__] = _Module()
