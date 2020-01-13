""" Variables for working with the a model containing X(pi/2) and Y(pi/2) gates. """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict

from ..construction import circuitconstruction as strc
from ..construction import modelconstruction as setc

from ._modelpack import SMQModelPack


# TODO update to SMQ
class SMQ1Q_XY(SMQModelPack):
    description = "X(pi/2) and Y(pi/2) gates"

    gates = ['Gx', 'Gy']
    fiducials = strc.circuit_list([(), ('Gx',), ('Gy',), ('Gx', 'Gx'),
                                   ('Gx', 'Gx', 'Gx'), ('Gy', 'Gy', 'Gy')])  # for 1Q MUB
    prepStrs = effectStrs = fiducials
    germs = strc.circuit_list(
        [('Gx',),
         ('Gy',),
         ('Gx', 'Gy',),
         ('Gx', 'Gx', 'Gy'),
         ('Gx', 'Gy', 'Gy'),
         ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy',)])

    germs_lite = germs[0:4]

    @property
    def _target_model(self):
        #Construct a target model:  X(pi/2), Y(pi/2)
        return setc.build_explicit_model([('Q0',)], ['Gx', 'Gy'],
                                         ["X(pi/2,Q0)", "Y(pi/2,Q0)"])

    clifford_compilation = OrderedDict([
        ("Gc0", []),
        ("Gc1", ['Gy', 'Gx', ]),
        ("Gc2", ['Gx', 'Gx', 'Gx', 'Gy', 'Gy', 'Gy', ]),
        ("Gc3", ['Gx', 'Gx', ]),
        ("Gc4", ['Gy', 'Gy', 'Gy', 'Gx', 'Gx', 'Gx', ]),
        ("Gc5", ['Gx', 'Gy', 'Gy', 'Gy', ]),
        ("Gc6", ['Gy', 'Gy', ]),
        ("Gc7", ['Gy', 'Gy', 'Gy', 'Gx', ]),
        ("Gc8", ['Gx', 'Gy', ]),
        ("Gc9", ['Gx', 'Gx', 'Gy', 'Gy', ]),
        ("Gc10", ['Gy', 'Gx', 'Gx', 'Gx', ]),
        ("Gc11", ['Gx', 'Gx', 'Gx', 'Gy', ]),
        ("Gc12", ['Gy', 'Gx', 'Gx', ]),
        ("Gc13", ['Gx', 'Gx', 'Gx', ]),
        ("Gc14", ['Gx', 'Gy', 'Gy', 'Gy', 'Gx', 'Gx', 'Gx', ]),
        ("Gc15", ['Gy', 'Gy', 'Gy', ]),
        ("Gc16", ['Gx', ]),
        ("Gc17", ['Gx', 'Gy', 'Gx', ]),
        ("Gc18", ['Gy', 'Gy', 'Gy', 'Gx', 'Gx', ]),
        ("Gc19", ['Gx', 'Gy', 'Gy', ]),
        ("Gc20", ['Gx', 'Gy', 'Gy', 'Gy', 'Gx', ]),
        ("Gc21", ['Gy', ]),
        ("Gc22", ['Gx', 'Gx', 'Gx', 'Gy', 'Gy', ]),
        ("Gc23", ['Gx', 'Gy', 'Gx', 'Gx', 'Gx', ])
    ])

    global_fidPairs = [
        (0, 0), (2, 3), (5, 2), (5, 4)]

    pergerm_fidPairsDict = {
        ('Gx',): [
            (1, 1), (3, 4), (4, 2), (5, 5)],
        ('Gy',): [
            (0, 2), (2, 2), (2, 4), (4, 4)],
        ('Gx', 'Gy'): [
            (0, 0), (0, 4), (2, 5), (5, 4)],
        ('Gx', 'Gx', 'Gy'): [
            (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
        ('Gx', 'Gy', 'Gy'): [
            (0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
        ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy'): [
            (0, 0), (2, 3), (5, 2), (5, 4)],
    }

    global_fidPairs_lite = [
        (0, 2), (2, 4), (3, 1), (3, 3)]

    pergerm_fidPairsDict_lite = {
        ('Gx',): [
            (1, 1), (3, 4), (4, 2), (5, 5)],
        ('Gy',): [
            (0, 2), (2, 2), (2, 4), (4, 4)],
        ('Gx', 'Gy'): [
            (0, 0), (0, 4), (2, 5), (5, 4)],
        ('Gx', 'Gx', 'Gy'): [
            (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
    }

import sys
sys.modules[__name__] = SMQ1Q_XY()
