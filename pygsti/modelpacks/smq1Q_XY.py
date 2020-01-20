"""
Variables for working with the a model containing X(pi/2) and Y(pi/2) gates.
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
    description = "X(pi/2) and Y(pi/2) gates"

    gates = [('Gx', 0), ('Gy', 0)]

    _sslbls = [0]

    _germs = [(('Gx', 0), ), (('Gy', 0), ), (('Gx', 0), ('Gy', 0)), (('Gx', 0), ('Gx', 0), ('Gy', 0)),
              (('Gx', 0), ('Gy', 0), ('Gy', 0)), (('Gx', 0), ('Gx', 0), ('Gy', 0), ('Gx', 0), ('Gy', 0), ('Gy', 0))]

    _germs_lite = [(('Gx', 0), ), (('Gy', 0), ), (('Gx', 0), ('Gy', 0)), (('Gx', 0), ('Gx', 0), ('Gy', 0))]

    _fiducials = [(), (('Gx', 0), ), (('Gy', 0), ), (('Gx', 0), ('Gx', 0)), (('Gx', 0), ('Gx', 0), ('Gx', 0)),
                  (('Gy', 0), ('Gy', 0), ('Gy', 0))]

    _prepStrs = [(), (('Gx', 0), ), (('Gy', 0), ), (('Gx', 0), ('Gx', 0)), (('Gx', 0), ('Gx', 0), ('Gx', 0)),
                 (('Gy', 0), ('Gy', 0), ('Gy', 0))]

    _effectStrs = [(), (('Gx', 0), ), (('Gy', 0), ), (('Gx', 0), ('Gx', 0)), (('Gx', 0), ('Gx', 0), ('Gx', 0)),
                   (('Gy', 0), ('Gy', 0), ('Gy', 0))]

    clifford_compilation = OrderedDict([('Gc0', []), ('Gc1', [('Gy', 0), ('Gx', 0)]),
                                        ('Gc2', [('Gx', 0), ('Gx', 0), ('Gx', 0), ('Gy', 0), ('Gy', 0), ('Gy', 0)]),
                                        ('Gc3', [('Gx', 0), ('Gx', 0)]),
                                        ('Gc4', [('Gy', 0), ('Gy', 0), ('Gy', 0), ('Gx', 0), ('Gx', 0), ('Gx', 0)]),
                                        ('Gc5', [('Gx', 0), ('Gy', 0), ('Gy', 0), ('Gy', 0)]),
                                        ('Gc6', [('Gy', 0), ('Gy', 0)]),
                                        ('Gc7', [('Gy', 0), ('Gy', 0), ('Gy', 0), ('Gx', 0)]),
                                        ('Gc8', [('Gx', 0), ('Gy', 0)]),
                                        ('Gc9', [('Gx', 0), ('Gx', 0), ('Gy', 0), ('Gy', 0)]),
                                        ('Gc10', [('Gy', 0), ('Gx', 0), ('Gx', 0), ('Gx', 0)]),
                                        ('Gc11', [('Gx', 0), ('Gx', 0), ('Gx', 0), ('Gy', 0)]),
                                        ('Gc12', [('Gy', 0), ('Gx', 0), ('Gx', 0)]),
                                        ('Gc13', [('Gx', 0), ('Gx', 0), ('Gx', 0)]),
                                        ('Gc14', [('Gx', 0), ('Gy', 0), ('Gy', 0), ('Gy', 0), ('Gx', 0), ('Gx', 0),
                                                  ('Gx', 0)]), ('Gc15', [('Gy', 0), ('Gy', 0), ('Gy', 0)]),
                                        ('Gc16', [('Gx', 0)]), ('Gc17', [('Gx', 0), ('Gy', 0), ('Gx', 0)]),
                                        ('Gc18', [('Gy', 0), ('Gy', 0), ('Gy', 0), ('Gx', 0), ('Gx', 0)]),
                                        ('Gc19', [('Gx', 0), ('Gy', 0), ('Gy', 0)]),
                                        ('Gc20', [('Gx', 0), ('Gy', 0), ('Gy', 0), ('Gy', 0), ('Gx', 0)]),
                                        ('Gc21', [('Gy', 0)]),
                                        ('Gc22', [('Gx', 0), ('Gx', 0), ('Gx', 0), ('Gy', 0), ('Gy', 0)]),
                                        ('Gc23', [('Gx', 0), ('Gy', 0), ('Gx', 0), ('Gx', 0), ('Gx', 0)])])

    global_fidPairs = [(0, 0), (2, 3), (5, 2), (5, 4)]

    pergerm_fidPairsDict = {
        ('Gx', ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ('Gy', ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        ('Gx', 'Gy'): [(0, 0), (0, 4), (2, 5), (5, 4)],
        ('Gx', 'Gx', 'Gy'): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
        ('Gx', 'Gy', 'Gy'): [(0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
        ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy'): [(0, 0), (2, 3), (5, 2), (5, 4)]
    }

    global_fidPairs_lite = [(0, 2), (2, 4), (3, 1), (3, 3)]

    pergerm_fidPairsDict_lite = {
        ('Gx', ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ('Gy', ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        ('Gx', 'Gy'): [(0, 0), (0, 4), (2, 5), (5, 4)],
        ('Gx', 'Gx', 'Gy'): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)]
    }

    @property
    def _target_model(self):
        return _setc.build_explicit_model([(0, )], [('Gx', 0), ('Gy', 0)], ['X(pi/2,0)', 'Y(pi/2,0)'])


import sys
sys.modules[__name__] = _Module()
