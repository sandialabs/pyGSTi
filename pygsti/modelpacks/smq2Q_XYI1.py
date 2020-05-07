"""
A standard multi-qubit gate set module.

Variables for working with the a model containing Idle, X(pi/2) and Y(pi/2) gates.
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

from pygsti.modelpacks._modelpack import GSTModelPack, RBModelPack


class _Module(GSTModelPack, RBModelPack):
    description = "Idle, X(pi/2), and Y(pi/2) gates"

    gates = [(), ('Gxpi2', 0), ('Gypi2', 0)]

    _sslbls = (0, 1)

    _germs = [((), ), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gypi2', 0)), (('Gxpi2', 0), ('Gypi2', 0), ()), (('Gxpi2', 0), (), ('Gypi2', 0)),
              (('Gxpi2', 0), (), ()), (('Gypi2', 0), (), ()), (('Gxpi2', 0), ('Gxpi2', 0), (), ('Gypi2', 0)), (('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()),
              (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _germs_lite = [((), ), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gypi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0))]

    _fiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0))]

    _prepfiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0))]

    _measfiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0))]

    _clifford_compilation = OrderedDict([('Gc1c0', [('Gypi2', 0), ('Gxpi2', 0), (), (), (), (), ()]),
                                        ('Gc2c0', [('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()]),
                                        ('Gc3c0', [('Gxpi2', 0), ('Gxpi2', 0), (), (), (), (), ()]),
                                        ('Gc4c0', [('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ()]),
                                        ('Gc5c0', [('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), (), (), ()]),
                                        ('Gc6c0', [('Gypi2', 0), ('Gypi2', 0), (), (), (), (), ()]),
                                        ('Gc7c0', [('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gxpi2', 0), (), (), ()]),
                                        ('Gc8c0', [('Gxpi2', 0), ('Gypi2', 0), (), (), (), (), ()]),
                                        ('Gc9c0', [('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), (), (), ()]),
                                        ('Gc10c0', [('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), (), (), ()]),
                                        ('Gc11c0', [('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), (), (), ()]),
                                        ('Gc12c0', [('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), (), (), (), ()]),
                                        ('Gc13c0', [('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), (), (), (), ()]),
                                        ('Gc14c0', [('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)]),
                                        ('Gc15c0', [('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), (), (), (), ()]),
                                        ('Gc16c0', [('Gxpi2', 0), (), (), (), (), (), ()]),
                                        ('Gc17c0', [('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), (), (), (), ()]),
                                        ('Gc18c0', [('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), (), ()]),
                                        ('Gc19c0', [('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), (), (), (), ()]),
                                        ('Gc20c0', [('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0), ('Gxpi2', 0), (), ()]),
                                        ('Gc21c0', [('Gypi2', 0), (), (), (), (), (), ()]),
                                        ('Gc22c0', [('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), (), ()]),
                                        ('Gc23c0', [('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), (), ()])])

    global_fidpairs = [(0, 1), (2, 0), (2, 1), (3, 3)]

    _pergerm_fidpairsdict = {
        ((), ): [(1, 1), (2, 2), (3, 3)],
        (('Gxpi2', 0), ): [(1, 2), (2, 2), (3, 1), (3, 3)],
        (('Gypi2', 0), ): [(0, 1), (1, 1), (2, 0), (3, 0)],
        (('Gxpi2', 0), ('Gypi2', 0)): [(0, 1), (2, 0), (2, 1), (3, 3)],
        (('Gxpi2', 0), ('Gypi2', 0), ()): [(0, 1), (2, 0), (2, 1), (3, 3)],
        (('Gxpi2', 0), (), ('Gypi2', 0)): [(0, 1), (2, 0), (2, 1), (3, 3)],
        (('Gypi2', 0), (), ()): [(0, 1), (1, 1), (2, 0), (3, 0)],
        (('Gxpi2', 0), (), ()): [(1, 2), (2, 2), (3, 1), (3, 3)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()): [(0, 2), (1, 0), (1, 1), (2, 0), (2, 2), (3, 3)],
        (('Gxpi2', 0), ('Gxpi2', 0), (), ('Gypi2', 0)): [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2), (3, 3)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0)): [(0, 0), (0, 1), (0, 2), (1, 2)]
    }

    global_fidpairs_lite = [(0, 4), (0, 5), (1, 0), (2, 0), (2, 4), (2, 5), (3, 0), (4, 2), (4, 4), (5, 1), (5, 2),
                            (5, 3)]

    _pergerm_fidpairsdict_lite = {
        (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ((), ): [(0, 3), (1, 1), (5, 5)],
        (('Gypi2', 0), ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)]
    }

    def _target_model(self, sslbls):
        return self._build_explicit_target_model(
            sslbls, [(), ('Gxpi2', 0), ('Gypi2', 0)],
            ['I({0})', 'X(pi/2,{0})', 'Y(pi/2,{0})'],
            effect_labels=['0', '1'], effect_expressions=['0', '1'])


import sys
sys.modules[__name__] = _Module()
