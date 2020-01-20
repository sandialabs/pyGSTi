"""
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

from pygsti.modelpacks._modelpack import SMQModelPack


class _Module(SMQModelPack):
    description = "Idle, X(pi/2), and Y(pi/2) gates"

    gates = ['Gii', 'Gix', 'Giy']

    _sslbls = [0]

    _germs = [('Gii', ), ('Gix', ), ('Giy', ), ('Gix', 'Giy'), ('Gix', 'Giy', 'Gii'), ('Gix', 'Gii', 'Giy'),
              ('Gix', 'Gii', 'Gii'), ('Giy', 'Gii', 'Gii'), ('Gix', 'Gix', 'Gii', 'Giy'), ('Gix', 'Giy', 'Giy', 'Gii'),
              ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy')]

    _germs_lite = None

    _fiducials = [(), ('Gix', ), ('Giy', ), ('Gix', 'Gix')]

    _prepStrs = [(), ('Gix', ), ('Giy', ), ('Gix', 'Gix')]

    _effectStrs = [(), ('Gix', ), ('Giy', ), ('Gix', 'Gix')]

    clifford_compilation = OrderedDict([('Gc0c0', ['Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c1', ['Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c2', ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']),
                                        ('Gc0c3', ['Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c4', ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']),
                                        ('Gc0c5', ['Gix', 'Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c6', ['Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c7', ['Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c8', ['Gix', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c9', ['Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c10', ['Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c11', ['Gix', 'Gix', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c12', ['Giy', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c13', ['Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c14', ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']),
                                        ('Gc0c15', ['Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c16', ['Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c17', ['Gix', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c18', ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']),
                                        ('Gc0c19', ['Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c20', ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']),
                                        ('Gc0c21', ['Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc0c22', ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']),
                                        ('Gc0c23', ['Gix', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii'])])

    global_fidPairs = [(0, 1), (2, 0), (2, 1), (3, 3)]

    pergerm_fidPairsDict = {
        ('Gix', ): [(1, 2), (2, 2), (3, 1), (3, 3)],
        ('Gii', ): [(1, 1), (2, 2), (3, 3)],
        ('Giy', ): [(0, 1), (1, 1), (2, 0), (3, 0)],
        ('Gix', 'Giy'): [(0, 1), (2, 0), (2, 1), (3, 3)],
        ('Giy', 'Gii', 'Gii'): [(0, 1), (1, 1), (2, 0), (3, 0)],
        ('Gix', 'Gii', 'Giy'): [(0, 1), (2, 0), (2, 1), (3, 3)],
        ('Gix', 'Giy', 'Gii'): [(0, 1), (2, 0), (2, 1), (3, 3)],
        ('Gix', 'Gii', 'Gii'): [(1, 2), (2, 2), (3, 1), (3, 3)],
        ('Gix', 'Gix', 'Gii', 'Giy'): [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2), (3, 3)],
        ('Gix', 'Giy', 'Giy', 'Gii'): [(0, 2), (1, 0), (1, 1), (2, 0), (2, 2), (3, 3)],
        ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'): [(0, 0), (0, 1), (0, 2), (1, 2)]
    }

    global_fidPairs_lite = None

    pergerm_fidPairsDict_lite = None

    @property
    def _target_model(self):
        return _setc.build_explicit_model([(0, )], ['Gii', 'Gix', 'Giy'], ['I(0)', 'X(pi/2,0)', 'Y(pi/2,0)'],
                                          effectLabels=['0', '1'],
                                          effectExpressions=['0', '1'])


import sys
sys.modules[__name__] = _Module()
