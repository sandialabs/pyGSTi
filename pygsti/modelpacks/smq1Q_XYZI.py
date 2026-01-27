"""
A standard multi-qubit gate set module.

Variables for working with the a model containing Idle, X(pi/2), Y(pi/2), and Z(pi/2) gates.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelpacks._modelpack import GSTModelPack


class _Module(GSTModelPack):
    description = "Idle, X(pi/2), Y(pi/2), Z(pi/2) gates"

    gates = [(), ('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)]

    _sslbls = (0,)

    _germs = [((), ), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gzpi2', 0), ), (('Gxpi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0)),
              (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)), (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)),
              (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0), ()), (('Gxpi2', 0), (), ('Gypi2', 0)),
              (('Gxpi2', 0), (), ()), (('Gypi2', 0), (), ()), ((), ('Gxpi2', 0), ('Gzpi2', 0)), ((), ('Gypi2', 0), ('Gzpi2', 0)),
              (('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()), (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _germs_lite = [((), ), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gzpi2', 0), ), (('Gxpi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0)),
                   (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)),
                   (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0))]

    _fiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                  (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _prepfiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                 (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _measfiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                   (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    global_fidpairs = [(0, 0), (2, 3), (5, 2), (5, 4)]

    _pergerm_fidpairsdict = {
        (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ((), ): [(0, 3), (1, 1), (5, 5)],
        (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gypi2', 0), ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gxpi2', 0), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gypi2', 0), (), ()): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
        (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
        (('Gxpi2', 0), ('Gypi2', 0), ()): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), (), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
        ((), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ((), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 3), (3, 2), (4, 0), (5, 3)],
        (('Gxpi2', 0), (), ()): [(1, 1), (3, 4), (4, 2), (5, 5)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()): [(0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)]
    }

    global_fidpairs_lite = [(0, 4), (0, 5), (1, 0), (2, 0), (2, 4), (2, 5), (3, 0), (4, 2), (4, 4), (5, 1), (5, 2),
                            (5, 3)]

    _pergerm_fidpairsdict_lite = {
        (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ((), ): [(0, 3), (1, 1), (5, 5)],
        (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gypi2', 0), ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gxpi2', 0), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)]
    }

    def _target_model(self, sslbls, **kwargs):
        return self._build_explicit_target_model(
            sslbls, [(), ('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)],
            ['I({0})', 'X(pi/2,{0})', 'Y(pi/2,{0})', 'Z(pi/2,{0})'], **kwargs)


import sys
sys.modules[__name__] = _Module()
