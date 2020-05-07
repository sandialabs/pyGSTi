"""
A standard multi-qubit gate set module.

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

from pygsti.modelpacks._modelpack import GSTModelPack


class _Module(GSTModelPack):
    description = "None"

    gates = None

    _sslbls = (0,)

    _germs = [(('Gzpi2', 0), ), (('Gn', 0), ), (('Gzpi2', 0), ('Gn', 0)), (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0)),
              (('Gzpi2', 0), ('Gn', 0), ('Gn', 0)), (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0), ('Gzpi2', 0), ('Gn', 0), ('Gn', 0))]

    _germs_lite = [(('Gzpi2', 0), ), (('Gn', 0), ), (('Gzpi2', 0), ('Gn', 0)), (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0)),
                   (('Gzpi2', 0), ('Gn', 0), ('Gn', 0)),
                   (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0), ('Gzpi2', 0), ('Gn', 0), ('Gn', 0))]

    _fiducials = None

    _prepfiducials = [(), (('Gn', 0), ), (('Gn', 0), ('Gn', 0)), (('Gn', 0), ('Gzpi2', 0), ('Gn', 0)),
                 (('Gn', 0), ('Gn', 0), ('Gn', 0)), (('Gn', 0), ('Gzpi2', 0), ('Gn', 0), ('Gn', 0), ('Gn', 0))]

    _measfiducials = [(), (('Gn', 0), ), (('Gn', 0), ('Gn', 0)), (('Gn', 0), ('Gzpi2', 0), ('Gn', 0)),
                   (('Gn', 0), ('Gn', 0), ('Gn', 0)), (('Gn', 0), ('Gn', 0), ('Gn', 0), ('Gzpi2', 0), ('Gn', 0))]

    global_fidpairs = [(0, 0), (2, 3), (5, 2), (5, 4)]

    _pergerm_fidpairsdict = {
        (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gn', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gn', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gn', 0), ('Gn', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0), ('Gzpi2', 0), ('Gn', 0), ('Gn', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)]
    }

    global_fidpairs_lite = [(0, 0), (2, 3), (5, 2), (5, 4)]

    _pergerm_fidpairsdict_lite = {
        (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gn', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gn', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gn', 0), ('Gn', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gzpi2', 0), ('Gzpi2', 0), ('Gn', 0), ('Gzpi2', 0), ('Gn', 0), ('Gn', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)]
    }

    def _target_model(self, sslbls):
        return self._build_explicit_target_model(
            sslbls, [('Gzpi2', 0), ('Gn', 0)],
            ['Z(pi/2,{0})', 'N(pi/2, sqrt(3)/2, 0, -0.5, {0})'])


import sys
sys.modules[__name__] = _Module()
