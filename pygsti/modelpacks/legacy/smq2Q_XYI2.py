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

from pygsti.modelpacks.legacy._modelpack import GSTModelPack, RBModelPack


class _Module(GSTModelPack, RBModelPack):
    description = "Idle, X(pi/2), and Y(pi/2) gates"

    gates = [(), ('Gxpi2', 1), ('Gypi2', 1)]

    _sslbls = (0, 1)

    _germs = [((), ), (('Gxpi2', 1), ), (('Gypi2', 1), ), (('Gxpi2', 1), ('Gypi2', 1)), (('Gxpi2', 1), ('Gypi2', 1), ()), (('Gxpi2', 1), (), ('Gypi2', 1)),
              (('Gxpi2', 1), (), ()), (('Gypi2', 1), (), ()), (('Gxpi2', 1), ('Gxpi2', 1), (), ('Gypi2', 1)), (('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), ()),
              (('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1))]

    _germs_lite = [((), ), (('Gxpi2', 1), ), (('Gypi2', 1), ), (('Gxpi2', 1), ('Gypi2', 1)), (('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1))]

    _fiducials = [(), (('Gxpi2', 1), ), (('Gypi2', 1), ), (('Gxpi2', 1), ('Gxpi2', 1))]

    _prepfiducials = [(), (('Gxpi2', 1), ), (('Gypi2', 1), ), (('Gxpi2', 1), ('Gxpi2', 1))]

    _measfiducials = [(), (('Gxpi2', 1), ), (('Gypi2', 1), ), (('Gxpi2', 1), ('Gxpi2', 1))]

    _clifford_compilation = OrderedDict([('Gc0c0', [(), (), (), (), (), (), ()]),
                                        ('Gc0c1', [('Gypi2', 1), ('Gxpi2', 1), (), (), (), (), ()]),
                                        ('Gc0c2', [('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), ()]),
                                        ('Gc0c3', [('Gxpi2', 1), ('Gxpi2', 1), (), (), (), (), ()]),
                                        ('Gc0c4', [('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ()]),
                                        ('Gc0c5', [('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), (), (), ()]),
                                        ('Gc0c6', [('Gypi2', 1), ('Gypi2', 1), (), (), (), (), ()]),
                                        ('Gc0c7', [('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gxpi2', 1), (), (), ()]),
                                        ('Gc0c8', [('Gxpi2', 1), ('Gypi2', 1), (), (), (), (), ()]),
                                        ('Gc0c9', [('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), (), (), ()]),
                                        ('Gc0c10', [('Gypi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), (), (), ()]),
                                        ('Gc0c11', [('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1), (), (), ()]),
                                        ('Gc0c12', [('Gypi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), (), (), (), ()]),
                                        ('Gc0c13', [('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), (), (), (), ()]),
                                        ('Gc0c14', [('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1)]),
                                        ('Gc0c15', [('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), (), (), (), ()]),
                                        ('Gc0c16', [('Gxpi2', 1), (), (), (), (), (), ()]),
                                        ('Gc0c17', [('Gxpi2', 1), ('Gypi2', 1), ('Gxpi2', 1), (), (), (), ()]),
                                        ('Gc0c18', [('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), (), ()]),
                                        ('Gc0c19', [('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), (), (), (), ()]),
                                        ('Gc0c20', [('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gypi2', 1), ('Gxpi2', 1), (), ()]),
                                        ('Gc0c21', [('Gypi2', 1), (), (), (), (), (), ()]),
                                        ('Gc0c22', [('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), (), ()]),
                                        ('Gc0c23', [('Gxpi2', 1), ('Gypi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), ('Gxpi2', 1), (), ()])])

    global_fidpairs = [(0, 1), (2, 0), (2, 1), (3, 3)]

    _pergerm_fidpairsdict = {
        (('Gxpi2', 1), ): [(1, 2), (2, 2), (3, 1), (3, 3)],
        ((), ): [(1, 1), (2, 2), (3, 3)],
        (('Gypi2', 1), ): [(0, 1), (1, 1), (2, 0), (3, 0)],
        (('Gxpi2', 1), ('Gypi2', 1)): [(0, 1), (2, 0), (2, 1), (3, 3)],
        (('Gypi2', 1), (), ()): [(0, 1), (1, 1), (2, 0), (3, 0)],
        (('Gxpi2', 1), (), ('Gypi2', 1)): [(0, 1), (2, 0), (2, 1), (3, 3)],
        (('Gxpi2', 1), ('Gypi2', 1), ()): [(0, 1), (2, 0), (2, 1), (3, 3)],
        (('Gxpi2', 1), (), ()): [(1, 2), (2, 2), (3, 1), (3, 3)],
        (('Gxpi2', 1), ('Gxpi2', 1), (), ('Gypi2', 1)): [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2), (3, 3)],
        (('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1), ()): [(0, 2), (1, 0), (1, 1), (2, 0), (2, 2), (3, 3)],
        (('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gxpi2', 1), ('Gypi2', 1), ('Gypi2', 1)): [(0, 0), (0, 1), (0, 2), (1, 2)]
    }

    global_fidpairs_lite = [(0, 4), (0, 5), (1, 0), (2, 0), (2, 4), (2, 5), (3, 0), (4, 2), (4, 4), (5, 1), (5, 2),
                            (5, 3)]

    _pergerm_fidpairsdict_lite = {
        (('Gxpi2', 1), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ((), ): [(0, 3), (1, 1), (5, 5)],
        (('Gypi2', 1), ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 1), ('Gypi2', 1)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gxpi2', 1), ('Gxpi2', 1), ('Gypi2', 1)): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)]
    }

    def _target_model(self, sslbls, **kwargs):  # Note: same as smq2Q_XYI1 -- (this entire module may be redundant)
        return self._build_explicit_target_model(
            sslbls, [(), ('Gxpi2', 1), ('Gypi2', 1)],
            ['I({0})', 'X(pi/2,{0})', 'Y(pi/2,{0})'],
            effect_labels=['0', '1'], effect_expressions=['0', '1'], **kwargs)


import sys
sys.modules[__name__] = _Module()
