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

    gates = ['Gii', 'Gxi', 'Gyi']

    germs = _strc.circuit_list([('Gii', ), ('Gxi', ), ('Gyi', ), ('Gxi', 'Gyi'), ('Gxi', 'Gyi', 'Gii'),
                                ('Gxi', 'Gii', 'Gyi'), ('Gxi', 'Gii', 'Gii'), ('Gyi', 'Gii', 'Gii'),
                                ('Gxi', 'Gxi', 'Gii', 'Gyi'), ('Gxi', 'Gyi', 'Gyi', 'Gii'),
                                ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi')],
                               line_labels=[0])

    germs_lite = None

    fiducials = _strc.circuit_list([(), ('Gxi', ), ('Gyi', ), ('Gxi', 'Gxi')], line_labels=[0])

    prepStrs = fiducials

    effectStrs = fiducials

    clifford_compilation = OrderedDict([('Gc1c0', ['Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc2c0', ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']),
                                        ('Gc3c0', ['Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc4c0', ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']),
                                        ('Gc5c0', ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']),
                                        ('Gc6c0', ['Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc7c0', ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']),
                                        ('Gc8c0', ['Gxi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc9c0', ['Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']),
                                        ('Gc10c0', ['Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']),
                                        ('Gc11c0', ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']),
                                        ('Gc12c0', ['Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc13c0', ['Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc14c0', ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']),
                                        ('Gc15c0', ['Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc16c0', ['Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc17c0', ['Gxi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc18c0', ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']),
                                        ('Gc19c0', ['Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc20c0', ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']),
                                        ('Gc21c0', ['Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']),
                                        ('Gc22c0', ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']),
                                        ('Gc23c0', ['Gxi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii'])])
    global_fidPairs = [(0, 1), (2, 0), (2, 1), (3, 3)]
    pergerm_fidPairsDict = {
        ('Gii', ): [(1, 1), (2, 2), (3, 3)],
        ('Gxi', ): [(1, 2), (2, 2), (3, 1), (3, 3)],
        ('Gyi', ): [(0, 1), (1, 1), (2, 0), (3, 0)],
        ('Gxi', 'Gyi'): [(0, 1), (2, 0), (2, 1), (3, 3)],
        ('Gxi', 'Gyi', 'Gii'): [(0, 1), (2, 0), (2, 1), (3, 3)],
        ('Gxi', 'Gii', 'Gyi'): [(0, 1), (2, 0), (2, 1), (3, 3)],
        ('Gyi', 'Gii', 'Gii'): [(0, 1), (1, 1), (2, 0), (3, 0)],
        ('Gxi', 'Gii', 'Gii'): [(1, 2), (2, 2), (3, 1), (3, 3)],
        ('Gxi', 'Gyi', 'Gyi', 'Gii'): [(0, 2), (1, 0), (1, 1), (2, 0), (2, 2), (3, 3)],
        ('Gxi', 'Gxi', 'Gii', 'Gyi'): [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2), (3, 3)],
        ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'): [(0, 0), (0, 1), (0, 2), (1, 2)]
    }
    global_fidPairs_lite = None
    pergerm_fidPairsDict_lite = None

    @property
    def _target_model(self):
        return _setc.build_explicit_model([(0, )], ['Gii', 'Gxi', 'Gyi'], ['I(0)', 'X(pi/2,0)', 'Y(pi/2,0)'],
                                          effectLabels=['0', '1'],
                                          effectExpressions=['0', '1'])


import sys
sys.modules[__name__] = _Module()
