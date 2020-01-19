"""
Variables for working with the a model containing X(pi/2) and Z(pi/2) gates.
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
    description = "X(pi/2) and Z(pi/2) gates"

    gates = [('Gx', 0), ('Gz', 0)]

    germs = _strc.circuit_list([(('Gx', 0), ), (('Gz', 0), ), (('Gz', 0), ('Gx', 0), ('Gx', 0)),
                                (('Gz', 0), ('Gz', 0), ('Gx', 0))],
                               line_labels=[0])

    germs_lite = _strc.circuit_list([(('Gx', 0), ), (('Gz', 0), ), (('Gx', 0), ('Gz', 0)),
                                     (('Gx', 0), ('Gx', 0), ('Gz', 0))],
                                    line_labels=[0])

    fiducials = None

    prepStrs = _strc.circuit_list([(), (('Gx', 0), ), (('Gx', 0), ('Gz', 0)), (('Gx', 0), ('Gx', 0)),
                                   (('Gx', 0), ('Gx', 0), ('Gx', 0)), (('Gx', 0), ('Gz', 0), ('Gx', 0), ('Gx', 0))],
                                  line_labels=[0])

    effectStrs = _strc.circuit_list([(), (('Gx', 0), ), (('Gz', 0), ('Gx', 0)), (('Gx', 0), ('Gx', 0)),
                                     (('Gx', 0), ('Gx', 0), ('Gx', 0)), (('Gx', 0), ('Gx', 0), ('Gz', 0), ('Gx', 0))],
                                    line_labels=[0])

    clifford_compilation = None
    global_fidPairs = [(0, 1), (1, 2), (4, 3), (4, 4)]
    pergerm_fidPairsDict = {
        ('Gx', ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ('Gz', ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gz', 'Gz', 'Gx'): [(0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
        ('Gz', 'Gx', 'Gx'): [(0, 3), (0, 4), (1, 0), (1, 4), (2, 1), (4, 5)]
    }
    global_fidPairs_lite = [(0, 1), (1, 2), (4, 3), (4, 4)]
    pergerm_fidPairsDict_lite = {
        ('Gx', ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ('Gz', ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ('Gx', 'Gz'): [(0, 3), (3, 2), (4, 0), (5, 3)],
        ('Gx', 'Gx', 'Gz'): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)]
    }

    @property
    def _target_model(self):
        return _setc.build_explicit_model([0], [('Gx', 0), ('Gz', 0)], ['X(pi/2,0)', 'Z(pi/2,0)'])


import sys
sys.modules[__name__] = _Module()
