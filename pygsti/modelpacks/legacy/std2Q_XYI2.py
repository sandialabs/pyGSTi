#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Variables for working with the a model containing Idle, X(pi/2) and Y(pi/2) gates.
"""

import sys as _sys
from collections import OrderedDict as _OrderedDict

from ...circuits import circuitconstruction as _strc
from ...models import modelconstruction as _setc
from .. import stdtarget as _stdtarget

description = "Idle, X(pi/2), and Y(pi/2) gates"

gates = ['Gii', 'Gix', 'Giy']
fiducials = _strc.to_circuits([(), ('Gix',), ('Giy',), ('Gix', 'Gix')], line_labels=('*',))
#                                     ('Gix','Gix','Gix'), ('Giy','Giy','Giy') ] ) # for 1Q MUB
prepStrs = effectStrs = fiducials

germs = _strc.to_circuits([('Gii',), ('Gix',), ('Giy',), ('Gix', 'Giy'),
                            ('Gix', 'Giy', 'Gii'), ('Gix', 'Gii', 'Giy'), ('Gix', 'Gii', 'Gii'), ('Giy', 'Gii', 'Gii'),
                            ('Gix', 'Gix', 'Gii', 'Giy'), ('Gix', 'Giy', 'Giy', 'Gii'),
                            ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy')], line_labels=('*',))

#Construct a target model: Identity, X(pi/2), Y(pi/2)
_target_model = _setc.create_explicit_model_from_expressions([('Q0',)], ['Gii', 'Gix', 'Giy'],
                                                             ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                                             effect_labels=['0', '1'], effect_expressions=["0", "1"])

_gscache = {("full", "auto"): _target_model}


def target_model(parameterization_type="full", sim_type="auto"):
    """
    Returns a copy of the target model in the given parameterization.

    Parameters
    ----------
    parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
        The gate and SPAM vector parameterization type. See
        :function:`Model.set_all_parameterizations` for all allowed values.

    sim_type : {"auto", "matrix", "map", "termorder:X" }
        The simulator type to be used for model calculations (leave as
        "auto" if you're not sure what this is).

    Returns
    -------
    Model
    """
    return _stdtarget._copy_target(_sys.modules[__name__], parameterization_type,
                                   sim_type, _gscache)


clifford_compilation = _OrderedDict()
clifford_compilation['Gc0c0'] = ['Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c1'] = ['Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c2'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc0c3'] = ['Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c4'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc0c5'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c6'] = ['Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c7'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c8'] = ['Gix', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c9'] = ['Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c10'] = ['Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c11'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c12'] = ['Giy', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c13'] = ['Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c14'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc0c15'] = ['Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c16'] = ['Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c17'] = ['Gix', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c18'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc0c19'] = ['Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c20'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc0c21'] = ['Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc0c22'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc0c23'] = ['Gix', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']


global_fidPairs = [
    (0, 1), (2, 0), (2, 1), (3, 3)]

pergerm_fidPairsDict = {
    ('Gix',): [
        (1, 2), (2, 2), (3, 1), (3, 3)],
    ('Gii',): [
        (1, 1), (2, 2), (3, 3)],
    ('Giy',): [
        (0, 1), (1, 1), (2, 0), (3, 0)],
    ('Gix', 'Giy'): [
        (0, 1), (2, 0), (2, 1), (3, 3)],
    ('Giy', 'Gii', 'Gii'): [
        (0, 1), (1, 1), (2, 0), (3, 0)],
    ('Gix', 'Gii', 'Giy'): [
        (0, 1), (2, 0), (2, 1), (3, 3)],
    ('Gix', 'Giy', 'Gii'): [
        (0, 1), (2, 0), (2, 1), (3, 3)],
    ('Gix', 'Gii', 'Gii'): [
        (1, 2), (2, 2), (3, 1), (3, 3)],
    ('Gix', 'Gix', 'Gii', 'Giy'): [
        (0, 0), (1, 0), (1, 1), (2, 1), (3, 2), (3, 3)],
    ('Gix', 'Giy', 'Giy', 'Gii'): [
        (0, 2), (1, 0), (1, 1), (2, 0), (2, 2), (3, 3)],
    ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'): [
        (0, 0), (0, 1), (0, 2), (1, 2)],
}
