#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Variables for working with the 2-qubit model containing the gates
I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, and CPHASE.
"""

import sys as _sys
from collections import OrderedDict as _OrderedDict

from ...circuits import circuitconstruction as _strc
from ...models import modelconstruction as _setc
from .. import stdtarget as _stdtarget

description = ("I*I, I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, X(pi/2)*X(pi/2), Y(pi/2)*Y(pi/2), X(pi/2)*Y(pi/2), "
               "and Y(pi/2)*X(pi/2) gates")

gates = ['Gii', 'Gix', 'Giy', 'Gxi', 'Gyi', 'Gxx', 'Gyy', 'Gxy', 'Gyx']

fiducials16 = _strc.to_circuits(
    [(), ('Gix',), ('Giy',), ('Gix', 'Gix'),
     ('Gxi',), ('Gxi', 'Gix'), ('Gxi', 'Giy'), ('Gxi', 'Gix', 'Gix'),
     ('Gyi',), ('Gyi', 'Gix'), ('Gyi', 'Giy'), ('Gyi', 'Gix', 'Gix'),
     ('Gxi', 'Gxi'), ('Gxi', 'Gxi', 'Gix'), ('Gxi', 'Gxi', 'Giy'), ('Gxi', 'Gxi', 'Gix', 'Gix')], line_labels=('*',))

fiducials36 = _strc.to_circuits(
    [(),
     ('Gix',),
     ('Giy',),
     ('Gix', 'Gix'),
     ('Gix', 'Gix', 'Gix'),
     ('Giy', 'Giy', 'Giy'),
     ('Gxi',),
     ('Gxi', 'Gix'),
     ('Gxi', 'Giy'),
     ('Gxi', 'Gix', 'Gix'),
     ('Gxi', 'Gix', 'Gix', 'Gix'),
     ('Gxi', 'Giy', 'Giy', 'Giy'),
     ('Gyi',),
     ('Gyi', 'Gix'),
     ('Gyi', 'Giy'),
     ('Gyi', 'Gix', 'Gix'),
     ('Gyi', 'Gix', 'Gix', 'Gix'),
     ('Gyi', 'Giy', 'Giy', 'Giy'),
     ('Gxi', 'Gxi'),
     ('Gxi', 'Gxi', 'Gix'),
     ('Gxi', 'Gxi', 'Giy'),
     ('Gxi', 'Gxi', 'Gix', 'Gix'),
     ('Gxi', 'Gxi', 'Gix', 'Gix', 'Gix'),
     ('Gxi', 'Gxi', 'Giy', 'Giy', 'Giy'),
     ('Gxi', 'Gxi', 'Gxi'),
     ('Gxi', 'Gxi', 'Gxi', 'Gix'),
     ('Gxi', 'Gxi', 'Gxi', 'Giy'),
     ('Gxi', 'Gxi', 'Gxi', 'Gix', 'Gix'),
     ('Gxi', 'Gxi', 'Gxi', 'Gix', 'Gix', 'Gix'),
     ('Gxi', 'Gxi', 'Gxi', 'Giy', 'Giy', 'Giy'),
     ('Gyi', 'Gyi', 'Gyi'),
     ('Gyi', 'Gyi', 'Gyi', 'Gix'),
     ('Gyi', 'Gyi', 'Gyi', 'Giy'),
     ('Gyi', 'Gyi', 'Gyi', 'Gix', 'Gix'),
     ('Gyi', 'Gyi', 'Gyi', 'Gix', 'Gix', 'Gix'),
     ('Gyi', 'Gyi', 'Gyi', 'Giy', 'Giy', 'Giy')], line_labels=('*',))

fiducials = fiducials16
prepStrs = fiducials16

effectStrs = _strc.to_circuits(
    [(), ('Gix',), ('Giy',),
     ('Gix', 'Gix'), ('Gxi',),
     ('Gyi',), ('Gxi', 'Gxi'),
     ('Gxi', 'Gix'), ('Gxi', 'Giy'),
     ('Gyi', 'Gix'), ('Gyi', 'Giy')], line_labels=('*',))

germs = _strc.to_circuits(
    [('Gii',),
     ('Gxi',),
     ('Gyi',),
     ('Gix',),
     ('Giy',),
     ('Gxx',),
     ('Gxy',),
     ('Gyx',),
     ('Gyy',),
     ('Gxi', 'Gyi'),
     ('Gix', 'Giy'),
     ('Giy', 'Gyi'),
     ('Gix', 'Gxi'),
     ('Gix', 'Gyi'),
     ('Giy', 'Gxi'),
     ('Gii', 'Gix'),
     ('Gii', 'Giy'),
     ('Gii', 'Gyi'),
     ('Giy', 'Gxx'),
     ('Gyi', 'Gxx'),
     ('Gxy', 'Gyx'),
     ('Gxx', 'Gxy'),
     ('Gxx', 'Gyx'),
     ('Gyy', 'Gxy'),
     ('Gxx', 'Gyy'),
     ('Gix', 'Gyy'),
     ('Gyi', 'Gxy'),
     ('Gxi', 'Gxi', 'Gyi'),
     ('Gix', 'Gix', 'Giy'),
     ('Gyy', 'Gxy', 'Gyx'),
     ('Gxx', 'Gxy', 'Gyy'),
     ('Gyy', 'Gyx', 'Gxy'),
     ('Gxi', 'Gyi', 'Gyi'),
     ('Gix', 'Giy', 'Giy'),
     ('Giy', 'Gxi', 'Gxi'),
     ('Giy', 'Gxi', 'Gyi'),
     ('Gix', 'Gxi', 'Giy'),
     ('Gix', 'Gyi', 'Gxi'),
     ('Gix', 'Gyi', 'Giy'),
     ('Gix', 'Giy', 'Gyi'),
     ('Gix', 'Giy', 'Gxi'),
     ('Giy', 'Gyi', 'Gxi'),
     ('Gxi', 'Gyi', 'Gii'),
     ('Gxi', 'Gii', 'Gyi'),
     ('Gxi', 'Gii', 'Gii'),
     ('Gyi', 'Gii', 'Gii'),
     ('Gix', 'Giy', 'Gii'),
     ('Gix', 'Gii', 'Giy'),
     ('Gix', 'Gii', 'Gii'),
     ('Giy', 'Gii', 'Gii'),
     ('Gii', 'Gix', 'Gyi'),
     ('Gii', 'Giy', 'Gyi'),
     ('Gii', 'Gyi', 'Gix'),
     ('Gxi', 'Gyi', 'Gxx'),
     ('Gyi', 'Gxx', 'Gxx'),
     ('Giy', 'Gxx', 'Gxx'),
     ('Giy', 'Gyi', 'Gxx'),
     ('Gix', 'Gix', 'Gxx'),
     ('Gix', 'Giy', 'Gxx'),
     ('Gix', 'Gxx', 'Gxi'),
     ('Gii', 'Gxx', 'Gxx'),
     ('Gix', 'Gxx', 'Giy'),
     ('Giy', 'Gxx', 'Gyi'),
     ('Gxi', 'Gxi', 'Gxx'),
     ('Gxi', 'Gyx', 'Gyy'),
     ('Gyi', 'Gxx', 'Gyx'),
     ('Gix', 'Gxy', 'Gyx'),
     ('Gyi', 'Gxy', 'Gyy'),
     ('Giy', 'Gyy', 'Gyx'),
     ('Gix', 'Gyy', 'Gyx'),
     ('Gyi', 'Gxy', 'Gxy'),
     ('Gxy', 'Gxy', 'Gyx'),
     ('Giy', 'Gyx', 'Gyy'),
     ('Gyi', 'Gyy', 'Gxy'),
     ('Giy', 'Gyx', 'Gyx'),
     ('Gxy', 'Gyx', 'Gyx'),
     ('Gxx', 'Gyy', 'Gxy'),
     ('Gxx', 'Gyx', 'Gyy'),
     ('Gxx', 'Gyx', 'Gxy'),
     ('Gix', 'Gxy', 'Gxx'),
     ('Gxi', 'Gyy', 'Gxy'),
     ('Gxx', 'Gyy', 'Gyy'),
     ('Gxx', 'Gxy', 'Gyx'),
     ('Giy', 'Gxx', 'Gxy'),
     ('Gxx', 'Gyy', 'Gyx'),
     ('Gix', 'Gxy', 'Gyi'),
     ('Gxi', 'Gyy', 'Gyy'),
     ('Gyy', 'Gyx', 'Gyx'),
     ('Gyi', 'Gyx', 'Gxx'),
     ('Giy', 'Gxy', 'Gxx'),
     ('Gxi', 'Gxy', 'Gyy'),
     ('Gxx', 'Gxx', 'Gyy'),
     ('Gix', 'Gyx', 'Gxy'),
     ('Giy', 'Gyx', 'Gxy'),
     ('Gxi', 'Gyx', 'Gxx'),
     ('Gyi', 'Gyx', 'Gxy'),
     ('Gyi', 'Gyi', 'Gyy'),
     ('Gix', 'Gxy', 'Gxy'),
     ('Gyx', 'Gix', 'Gxy', 'Gxi'),
     ('Gyi', 'Gix', 'Gxi', 'Giy'),
     ('Gix', 'Giy', 'Gxi', 'Gyi'),
     ('Gix', 'Gix', 'Gix', 'Giy'),
     ('Gxi', 'Gyi', 'Gyi', 'Gyi'),
     ('Gyi', 'Gyi', 'Giy', 'Gyi'),
     ('Gyi', 'Gix', 'Gix', 'Gix'),
     ('Gxi', 'Gyi', 'Gix', 'Gix'),
     ('Gxi', 'Gyi', 'Gyi', 'Gii'),
     ('Gix', 'Giy', 'Giy', 'Gii'),
     ('Giy', 'Gxi', 'Gii', 'Gii'),
     ('Gix', 'Gii', 'Gii', 'Gxi'),
     ('Gxx', 'Gxx', 'Gyi', 'Gyi'),
     ('Giy', 'Gxx', 'Gxi', 'Giy'),
     ('Gyx', 'Gyx', 'Gyy', 'Gyy'),
     ('Gxx', 'Gxy', 'Gyx', 'Gyx'),
     ('Giy', 'Gxy', 'Gxy', 'Gyx'),
     ('Gxx', 'Gxx', 'Gxx', 'Gix', 'Gxi'),
     ('Gxi', 'Gyy', 'Gyi', 'Gxx', 'Gxx'),
     ('Giy', 'Gyi', 'Gxi', 'Gxi', 'Giy'),
     ('Gxi', 'Gxi', 'Giy', 'Gyi', 'Giy'),
     ('Giy', 'Gix', 'Gxi', 'Gix', 'Gxi'),
     ('Gyi', 'Giy', 'Gyi', 'Gix', 'Gix'),
     ('Giy', 'Gxi', 'Gix', 'Giy', 'Gyi'),
     ('Giy', 'Giy', 'Gxi', 'Gyi', 'Gxi'),
     ('Gyi', 'Gyi', 'Giy', 'Gxx', 'Giy'),
     ('Gii', 'Gyy', 'Gxy', 'Gyy', 'Gyy'),
     ('Gyx', 'Gxy', 'Gxi', 'Gxy', 'Gxy'),
     ('Gxi', 'Gix', 'Giy', 'Gxi', 'Giy', 'Gyi'),
     ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gix'),
     ('Gxy', 'Gyx', 'Gix', 'Giy', 'Gxx', 'Gix'),
     ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'),
     ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'),
     ('Gyi', 'Gxi', 'Gix', 'Giy', 'Gxi', 'Gix'),
     ('Gyi', 'Gxi', 'Gix', 'Gxi', 'Gix', 'Giy'),
     ('Gxi', 'Gix', 'Giy', 'Giy', 'Gxi', 'Gyi'),
     ('Gix', 'Giy', 'Giy', 'Gix', 'Gxi', 'Gxi'),
     ('Gyi', 'Giy', 'Gxi', 'Giy', 'Giy', 'Giy'),
     ('Gyi', 'Gyi', 'Gyi', 'Giy', 'Gyi', 'Gix'),
     ('Giy', 'Giy', 'Gxi', 'Giy', 'Gix', 'Giy'),
     ('Gyi', 'Gxi', 'Giy', 'Gyi', 'Gxx', 'Gyi'),
     ('Giy', 'Gix', 'Gyi', 'Gyi', 'Gix', 'Gxi', 'Giy'),
     ('Gii', 'Gyi', 'Gxi', 'Gxx', 'Gxx', 'Gix', 'Gxx'),
     ('Gxx', 'Gxx', 'Gxx', 'Gxi', 'Gix', 'Gii', 'Giy'),
     ('Gyi', 'Gxi', 'Giy', 'Gxi', 'Gix', 'Gxi', 'Gyi', 'Giy'),
     ('Gix', 'Gix', 'Gyi', 'Gxi', 'Giy', 'Gxi', 'Giy', 'Gyi')
     ], line_labels=('*',))

germs_lite = _strc.to_circuits(
    [('Gii',),
     ('Gxi',),
     ('Gyi',),
     ('Gix',),
     ('Giy',),
     ('Gxx',),
     ('Gxy',),
     ('Gyx',),
     ('Gyy',),
     ('Gxi', 'Gyi'),
     ('Gix', 'Giy'),
     ('Gxi', 'Gxi', 'Gyi'),
     ('Gix', 'Gix', 'Giy'),
     ('Gyy', 'Gxy', 'Gyx'),
     ('Gxx', 'Gxy', 'Gyy'),
     ('Gyy', 'Gyx', 'Gxy'),
     ('Gyx', 'Gix', 'Gxy', 'Gxi'),
     ('Gxx', 'Gxx', 'Gxx', 'Gix', 'Gxi'),
     ('Gxi', 'Gyy', 'Gyi', 'Gxx', 'Gxx'),
     ('Gxi', 'Gix', 'Giy', 'Gxi', 'Giy', 'Gyi'),
     ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gix'),
     ('Gxy', 'Gyx', 'Gix', 'Giy', 'Gxx', 'Gix'),
     ('Gyi', 'Gxi', 'Giy', 'Gxi', 'Gix', 'Gxi', 'Gyi', 'Giy')
     ], line_labels=('*',))

#Construct the target model
_target_model = _setc.create_explicit_model_from_expressions(
    [('Q0', 'Q1')], ['Gii', 'Gix', 'Giy', 'Gxi', 'Gyi', 'Gxx', 'Gyy', 'Gxy', 'Gyx'],
    ["I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)",
     "Y(pi/2,Q0):I(Q1)", "X(pi/2,Q0):X(pi/2,Q1)", "Y(pi/2,Q0):Y(pi/2,Q1)",
     "X(pi/2,Q0):Y(pi/2,Q1)", "Y(pi/2,Q0):X(pi/2,Q1)"],
    effect_labels=['00', '01', '10', '11'], effect_expressions=["0", "1", "2", "3"])

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
clifford_compilation['Gc1c0'] = ['Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c1'] = ['Gyy', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c2'] = ['Gyx', 'Gxx', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc1c3'] = ['Gyx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c4'] = ['Gyy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc1c5'] = ['Gyx', 'Gxy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c6'] = ['Gyy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c7'] = ['Gyy', 'Gxy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c8'] = ['Gyx', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c9'] = ['Gyx', 'Gxx', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c10'] = ['Gyy', 'Gxx', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c11'] = ['Gyx', 'Gxx', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c12'] = ['Gyy', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c13'] = ['Gyx', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c14'] = ['Gyx', 'Gxy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc1c15'] = ['Gyy', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c16'] = ['Gyx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c17'] = ['Gyx', 'Gxy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c18'] = ['Gyy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc1c19'] = ['Gyx', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c20'] = ['Gyx', 'Gxy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc1c21'] = ['Gyy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc1c22'] = ['Gyx', 'Gxx', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc1c23'] = ['Gyx', 'Gxy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc2c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c1'] = ['Gxy', 'Gxx', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c2'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gii']
clifford_compilation['Gc2c3'] = ['Gxx', 'Gxx', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c4'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gyx', 'Gyx', 'Gii']
clifford_compilation['Gc2c5'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c6'] = ['Gxy', 'Gxy', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c7'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c8'] = ['Gxx', 'Gxy', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c9'] = ['Gxx', 'Gxx', 'Gxy', 'Gyy', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c10'] = ['Gxy', 'Gxx', 'Gxx', 'Gyx', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c11'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c12'] = ['Gxy', 'Gxx', 'Gxx', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c13'] = ['Gxx', 'Gxx', 'Gxx', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c14'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gyx', 'Gix']
clifford_compilation['Gc2c15'] = ['Gxy', 'Gxy', 'Gxy', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c16'] = ['Gxx', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c17'] = ['Gxx', 'Gxy', 'Gxx', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c18'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gyx', 'Gyi', 'Gii']
clifford_compilation['Gc2c19'] = ['Gxx', 'Gxy', 'Gxy', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c20'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gyi', 'Gii']
clifford_compilation['Gc2c21'] = ['Gxy', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']
clifford_compilation['Gc2c22'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gyy', 'Gyi', 'Gii']
clifford_compilation['Gc2c23'] = ['Gxx', 'Gxy', 'Gxx', 'Gyx', 'Gyx', 'Gyi', 'Gii']
clifford_compilation['Gc3c0'] = ['Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c1'] = ['Gxy', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c2'] = ['Gxx', 'Gxx', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc3c3'] = ['Gxx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c4'] = ['Gxy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc3c5'] = ['Gxx', 'Gxy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c6'] = ['Gxy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c7'] = ['Gxy', 'Gxy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c8'] = ['Gxx', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c9'] = ['Gxx', 'Gxx', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c10'] = ['Gxy', 'Gxx', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c11'] = ['Gxx', 'Gxx', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c12'] = ['Gxy', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c13'] = ['Gxx', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c14'] = ['Gxx', 'Gxy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc3c15'] = ['Gxy', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c16'] = ['Gxx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c17'] = ['Gxx', 'Gxy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c18'] = ['Gxy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc3c19'] = ['Gxx', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c20'] = ['Gxx', 'Gxy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc3c21'] = ['Gxy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc3c22'] = ['Gxx', 'Gxx', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc3c23'] = ['Gxx', 'Gxy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc4c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c1'] = ['Gyy', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c2'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gxy', 'Gxy', 'Gii']
clifford_compilation['Gc4c3'] = ['Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c4'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxx', 'Gxx', 'Gii']
clifford_compilation['Gc4c5'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c6'] = ['Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c7'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c8'] = ['Gyx', 'Gyy', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c9'] = ['Gyx', 'Gyx', 'Gyy', 'Gxy', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c10'] = ['Gyy', 'Gyx', 'Gyx', 'Gxx', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c11'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c12'] = ['Gyy', 'Gyx', 'Gyx', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c13'] = ['Gyx', 'Gyx', 'Gyx', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c14'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gxx', 'Gxx', 'Gix']
clifford_compilation['Gc4c15'] = ['Gyy', 'Gyy', 'Gyy', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c16'] = ['Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c17'] = ['Gyx', 'Gyy', 'Gyx', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c18'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxx', 'Gxi', 'Gii']
clifford_compilation['Gc4c19'] = ['Gyx', 'Gyy', 'Gyy', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c20'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gxx', 'Gxi', 'Gii']
clifford_compilation['Gc4c21'] = ['Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']
clifford_compilation['Gc4c22'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gxy', 'Gxi', 'Gii']
clifford_compilation['Gc4c23'] = ['Gyx', 'Gyy', 'Gyx', 'Gxx', 'Gxx', 'Gxi', 'Gii']
clifford_compilation['Gc5c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c1'] = ['Gxy', 'Gyx', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c2'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc5c3'] = ['Gxx', 'Gyx', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c4'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc5c5'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c6'] = ['Gxy', 'Gyy', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c7'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c8'] = ['Gxx', 'Gyy', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c9'] = ['Gxx', 'Gyx', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c10'] = ['Gxy', 'Gyx', 'Gyx', 'Gyx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c11'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c12'] = ['Gxy', 'Gyx', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c13'] = ['Gxx', 'Gyx', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c14'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc5c15'] = ['Gxy', 'Gyy', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c16'] = ['Gxx', 'Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c17'] = ['Gxx', 'Gyy', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c18'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc5c19'] = ['Gxx', 'Gyy', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c20'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc5c21'] = ['Gxy', 'Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc5c22'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc5c23'] = ['Gxx', 'Gyy', 'Gyx', 'Gyx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc6c0'] = ['Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c1'] = ['Gyy', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c2'] = ['Gyx', 'Gyx', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc6c3'] = ['Gyx', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c4'] = ['Gyy', 'Gyy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc6c5'] = ['Gyx', 'Gyy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c6'] = ['Gyy', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c7'] = ['Gyy', 'Gyy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c8'] = ['Gyx', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c9'] = ['Gyx', 'Gyx', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c10'] = ['Gyy', 'Gyx', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c11'] = ['Gyx', 'Gyx', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c12'] = ['Gyy', 'Gyx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c13'] = ['Gyx', 'Gyx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c14'] = ['Gyx', 'Gyy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc6c15'] = ['Gyy', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c16'] = ['Gyx', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c17'] = ['Gyx', 'Gyy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c18'] = ['Gyy', 'Gyy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc6c19'] = ['Gyx', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c20'] = ['Gyx', 'Gyy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc6c21'] = ['Gyy', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c22'] = ['Gyx', 'Gyx', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc6c23'] = ['Gyx', 'Gyy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc7c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c1'] = ['Gyy', 'Gyx', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c2'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc7c3'] = ['Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c4'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc7c5'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c6'] = ['Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c7'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c8'] = ['Gyx', 'Gyy', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c9'] = ['Gyx', 'Gyx', 'Gyy', 'Gxy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c10'] = ['Gyy', 'Gyx', 'Gyx', 'Gxx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c11'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c12'] = ['Gyy', 'Gyx', 'Gyx', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c13'] = ['Gyx', 'Gyx', 'Gyx', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c14'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc7c15'] = ['Gyy', 'Gyy', 'Gyy', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c16'] = ['Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c17'] = ['Gyx', 'Gyy', 'Gyx', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c18'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc7c19'] = ['Gyx', 'Gyy', 'Gyy', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c20'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc7c21'] = ['Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc7c22'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc7c23'] = ['Gyx', 'Gyy', 'Gyx', 'Gxx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc8c0'] = ['Gxi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c1'] = ['Gxy', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c2'] = ['Gxx', 'Gyx', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc8c3'] = ['Gxx', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c4'] = ['Gxy', 'Gyy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc8c5'] = ['Gxx', 'Gyy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c6'] = ['Gxy', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c7'] = ['Gxy', 'Gyy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c8'] = ['Gxx', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c9'] = ['Gxx', 'Gyx', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c10'] = ['Gxy', 'Gyx', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c11'] = ['Gxx', 'Gyx', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c12'] = ['Gxy', 'Gyx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c13'] = ['Gxx', 'Gyx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c14'] = ['Gxx', 'Gyy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc8c15'] = ['Gxy', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c16'] = ['Gxx', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c17'] = ['Gxx', 'Gyy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c18'] = ['Gxy', 'Gyy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc8c19'] = ['Gxx', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c20'] = ['Gxx', 'Gyy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc8c21'] = ['Gxy', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc8c22'] = ['Gxx', 'Gyx', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc8c23'] = ['Gxx', 'Gyy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc9c0'] = ['Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c1'] = ['Gxy', 'Gxx', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c2'] = ['Gxx', 'Gxx', 'Gyx', 'Gyy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc9c3'] = ['Gxx', 'Gxx', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c4'] = ['Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc9c5'] = ['Gxx', 'Gxy', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c6'] = ['Gxy', 'Gxy', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c7'] = ['Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c8'] = ['Gxx', 'Gxy', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c9'] = ['Gxx', 'Gxx', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c10'] = ['Gxy', 'Gxx', 'Gyx', 'Gyx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c11'] = ['Gxx', 'Gxx', 'Gyx', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c12'] = ['Gxy', 'Gxx', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c13'] = ['Gxx', 'Gxx', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c14'] = ['Gxx', 'Gxy', 'Gyy', 'Gyy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc9c15'] = ['Gxy', 'Gxy', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c16'] = ['Gxx', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c17'] = ['Gxx', 'Gxy', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c18'] = ['Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc9c19'] = ['Gxx', 'Gxy', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c20'] = ['Gxx', 'Gxy', 'Gyy', 'Gyy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc9c21'] = ['Gxy', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc9c22'] = ['Gxx', 'Gxx', 'Gyx', 'Gyy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc9c23'] = ['Gxx', 'Gxy', 'Gyx', 'Gyx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc10c0'] = ['Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c1'] = ['Gyy', 'Gxx', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c2'] = ['Gyx', 'Gxx', 'Gxx', 'Gxy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc10c3'] = ['Gyx', 'Gxx', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c4'] = ['Gyy', 'Gxy', 'Gxy', 'Gxx', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc10c5'] = ['Gyx', 'Gxy', 'Gxy', 'Gxy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c6'] = ['Gyy', 'Gxy', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c7'] = ['Gyy', 'Gxy', 'Gxy', 'Gxx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c8'] = ['Gyx', 'Gxy', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c9'] = ['Gyx', 'Gxx', 'Gxy', 'Gxy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c10'] = ['Gyy', 'Gxx', 'Gxx', 'Gxx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c11'] = ['Gyx', 'Gxx', 'Gxx', 'Gxy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c12'] = ['Gyy', 'Gxx', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c13'] = ['Gyx', 'Gxx', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c14'] = ['Gyx', 'Gxy', 'Gxy', 'Gxy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc10c15'] = ['Gyy', 'Gxy', 'Gxy', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c16'] = ['Gyx', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c17'] = ['Gyx', 'Gxy', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c18'] = ['Gyy', 'Gxy', 'Gxy', 'Gxx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc10c19'] = ['Gyx', 'Gxy', 'Gxy', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c20'] = ['Gyx', 'Gxy', 'Gxy', 'Gxy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc10c21'] = ['Gyy', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc10c22'] = ['Gyx', 'Gxx', 'Gxx', 'Gxy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc10c23'] = ['Gyx', 'Gxy', 'Gxx', 'Gxx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc11c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c1'] = ['Gxy', 'Gxx', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c2'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc11c3'] = ['Gxx', 'Gxx', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c4'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc11c5'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c6'] = ['Gxy', 'Gxy', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c7'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c8'] = ['Gxx', 'Gxy', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c9'] = ['Gxx', 'Gxx', 'Gxy', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c10'] = ['Gxy', 'Gxx', 'Gxx', 'Gyx', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c11'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c12'] = ['Gxy', 'Gxx', 'Gxx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c13'] = ['Gxx', 'Gxx', 'Gxx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c14'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc11c15'] = ['Gxy', 'Gxy', 'Gxy', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c16'] = ['Gxx', 'Gxi', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c17'] = ['Gxx', 'Gxy', 'Gxx', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c18'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc11c19'] = ['Gxx', 'Gxy', 'Gxy', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c20'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc11c21'] = ['Gxy', 'Gxi', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc11c22'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc11c23'] = ['Gxx', 'Gxy', 'Gxx', 'Gyx', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc12c0'] = ['Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c1'] = ['Gyy', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c2'] = ['Gyx', 'Gxx', 'Gxx', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc12c3'] = ['Gyx', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c4'] = ['Gyy', 'Gxy', 'Gxy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc12c5'] = ['Gyx', 'Gxy', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c6'] = ['Gyy', 'Gxy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c7'] = ['Gyy', 'Gxy', 'Gxy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c8'] = ['Gyx', 'Gxy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c9'] = ['Gyx', 'Gxx', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c10'] = ['Gyy', 'Gxx', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c11'] = ['Gyx', 'Gxx', 'Gxx', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c12'] = ['Gyy', 'Gxx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c13'] = ['Gyx', 'Gxx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c14'] = ['Gyx', 'Gxy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc12c15'] = ['Gyy', 'Gxy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c16'] = ['Gyx', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c17'] = ['Gyx', 'Gxy', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c18'] = ['Gyy', 'Gxy', 'Gxy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc12c19'] = ['Gyx', 'Gxy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c20'] = ['Gyx', 'Gxy', 'Gxy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc12c21'] = ['Gyy', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc12c22'] = ['Gyx', 'Gxx', 'Gxx', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc12c23'] = ['Gyx', 'Gxy', 'Gxx', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc13c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c1'] = ['Gxy', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c2'] = ['Gxx', 'Gxx', 'Gxx', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc13c3'] = ['Gxx', 'Gxx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c4'] = ['Gxy', 'Gxy', 'Gxy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc13c5'] = ['Gxx', 'Gxy', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c6'] = ['Gxy', 'Gxy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c7'] = ['Gxy', 'Gxy', 'Gxy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c8'] = ['Gxx', 'Gxy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c9'] = ['Gxx', 'Gxx', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c10'] = ['Gxy', 'Gxx', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c11'] = ['Gxx', 'Gxx', 'Gxx', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c12'] = ['Gxy', 'Gxx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c13'] = ['Gxx', 'Gxx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c14'] = ['Gxx', 'Gxy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc13c15'] = ['Gxy', 'Gxy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c16'] = ['Gxx', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c17'] = ['Gxx', 'Gxy', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c18'] = ['Gxy', 'Gxy', 'Gxy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc13c19'] = ['Gxx', 'Gxy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c20'] = ['Gxx', 'Gxy', 'Gxy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc13c21'] = ['Gxy', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc13c22'] = ['Gxx', 'Gxx', 'Gxx', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc13c23'] = ['Gxx', 'Gxy', 'Gxx', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc14c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c1'] = ['Gxy', 'Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c2'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gxy', 'Gxy', 'Gxi']
clifford_compilation['Gc14c3'] = ['Gxx', 'Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c4'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gxx', 'Gxx', 'Gxi']
clifford_compilation['Gc14c5'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c6'] = ['Gxy', 'Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c7'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c8'] = ['Gxx', 'Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c9'] = ['Gxx', 'Gyx', 'Gyy', 'Gyy', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c10'] = ['Gxy', 'Gyx', 'Gyx', 'Gyx', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c11'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c12'] = ['Gxy', 'Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c13'] = ['Gxx', 'Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c14'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxx', 'Gxx']
clifford_compilation['Gc14c15'] = ['Gxy', 'Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c16'] = ['Gxx', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c17'] = ['Gxx', 'Gyy', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c18'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gxx', 'Gxi', 'Gxi']
clifford_compilation['Gc14c19'] = ['Gxx', 'Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c20'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxi', 'Gxi']
clifford_compilation['Gc14c21'] = ['Gxy', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']
clifford_compilation['Gc14c22'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gxy', 'Gxi', 'Gxi']
clifford_compilation['Gc14c23'] = ['Gxx', 'Gyy', 'Gyx', 'Gyx', 'Gxx', 'Gxi', 'Gxi']
clifford_compilation['Gc15c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c1'] = ['Gyy', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c2'] = ['Gyx', 'Gyx', 'Gyx', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc15c3'] = ['Gyx', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c4'] = ['Gyy', 'Gyy', 'Gyy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc15c5'] = ['Gyx', 'Gyy', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c6'] = ['Gyy', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c7'] = ['Gyy', 'Gyy', 'Gyy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c8'] = ['Gyx', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c9'] = ['Gyx', 'Gyx', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c10'] = ['Gyy', 'Gyx', 'Gyx', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c11'] = ['Gyx', 'Gyx', 'Gyx', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c12'] = ['Gyy', 'Gyx', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c13'] = ['Gyx', 'Gyx', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c14'] = ['Gyx', 'Gyy', 'Gyy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc15c15'] = ['Gyy', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c16'] = ['Gyx', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c17'] = ['Gyx', 'Gyy', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c18'] = ['Gyy', 'Gyy', 'Gyy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc15c19'] = ['Gyx', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c20'] = ['Gyx', 'Gyy', 'Gyy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc15c21'] = ['Gyy', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc15c22'] = ['Gyx', 'Gyx', 'Gyx', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc15c23'] = ['Gyx', 'Gyy', 'Gyx', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc16c0'] = ['Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c1'] = ['Gxy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c2'] = ['Gxx', 'Gix', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc16c3'] = ['Gxx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c4'] = ['Gxy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc16c5'] = ['Gxx', 'Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c6'] = ['Gxy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c7'] = ['Gxy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c8'] = ['Gxx', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c9'] = ['Gxx', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c10'] = ['Gxy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c11'] = ['Gxx', 'Gix', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c12'] = ['Gxy', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c13'] = ['Gxx', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c14'] = ['Gxx', 'Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc16c15'] = ['Gxy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c16'] = ['Gxx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c17'] = ['Gxx', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c18'] = ['Gxy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc16c19'] = ['Gxx', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c20'] = ['Gxx', 'Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc16c21'] = ['Gxy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc16c22'] = ['Gxx', 'Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc16c23'] = ['Gxx', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc17c0'] = ['Gxi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c1'] = ['Gxy', 'Gyx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c2'] = ['Gxx', 'Gyx', 'Gxx', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc17c3'] = ['Gxx', 'Gyx', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c4'] = ['Gxy', 'Gyy', 'Gxy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc17c5'] = ['Gxx', 'Gyy', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c6'] = ['Gxy', 'Gyy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c7'] = ['Gxy', 'Gyy', 'Gxy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c8'] = ['Gxx', 'Gyy', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c9'] = ['Gxx', 'Gyx', 'Gxy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c10'] = ['Gxy', 'Gyx', 'Gxx', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c11'] = ['Gxx', 'Gyx', 'Gxx', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c12'] = ['Gxy', 'Gyx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c13'] = ['Gxx', 'Gyx', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c14'] = ['Gxx', 'Gyy', 'Gxy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc17c15'] = ['Gxy', 'Gyy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c16'] = ['Gxx', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c17'] = ['Gxx', 'Gyy', 'Gxx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c18'] = ['Gxy', 'Gyy', 'Gxy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc17c19'] = ['Gxx', 'Gyy', 'Gxy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c20'] = ['Gxx', 'Gyy', 'Gxy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc17c21'] = ['Gxy', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc17c22'] = ['Gxx', 'Gyx', 'Gxx', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc17c23'] = ['Gxx', 'Gyy', 'Gxx', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc18c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c1'] = ['Gyy', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c2'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gxy', 'Giy', 'Gii']
clifford_compilation['Gc18c3'] = ['Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c4'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxx', 'Gix', 'Gii']
clifford_compilation['Gc18c5'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c6'] = ['Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c7'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c8'] = ['Gyx', 'Gyy', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c9'] = ['Gyx', 'Gyx', 'Gyy', 'Gxy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c10'] = ['Gyy', 'Gyx', 'Gyx', 'Gxx', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c11'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c12'] = ['Gyy', 'Gyx', 'Gyx', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c13'] = ['Gyx', 'Gyx', 'Gyx', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c14'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gxx', 'Gix', 'Gix']
clifford_compilation['Gc18c15'] = ['Gyy', 'Gyy', 'Gyy', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c16'] = ['Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c17'] = ['Gyx', 'Gyy', 'Gyx', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c18'] = ['Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc18c19'] = ['Gyx', 'Gyy', 'Gyy', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c20'] = ['Gyx', 'Gyy', 'Gyy', 'Gxy', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc18c21'] = ['Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc18c22'] = ['Gyx', 'Gyx', 'Gyx', 'Gxy', 'Gxy', 'Gii', 'Gii']
clifford_compilation['Gc18c23'] = ['Gyx', 'Gyy', 'Gyx', 'Gxx', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc19c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c1'] = ['Gxy', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c2'] = ['Gxx', 'Gyx', 'Gyx', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc19c3'] = ['Gxx', 'Gyx', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c4'] = ['Gxy', 'Gyy', 'Gyy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc19c5'] = ['Gxx', 'Gyy', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c6'] = ['Gxy', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c7'] = ['Gxy', 'Gyy', 'Gyy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c8'] = ['Gxx', 'Gyy', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c9'] = ['Gxx', 'Gyx', 'Gyy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c10'] = ['Gxy', 'Gyx', 'Gyx', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c11'] = ['Gxx', 'Gyx', 'Gyx', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c12'] = ['Gxy', 'Gyx', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c13'] = ['Gxx', 'Gyx', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c14'] = ['Gxx', 'Gyy', 'Gyy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc19c15'] = ['Gxy', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c16'] = ['Gxx', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c17'] = ['Gxx', 'Gyy', 'Gyx', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c18'] = ['Gxy', 'Gyy', 'Gyy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc19c19'] = ['Gxx', 'Gyy', 'Gyy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c20'] = ['Gxx', 'Gyy', 'Gyy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc19c21'] = ['Gxy', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc19c22'] = ['Gxx', 'Gyx', 'Gyx', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc19c23'] = ['Gxx', 'Gyy', 'Gyx', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc20c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c1'] = ['Gxy', 'Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c2'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gxy', 'Giy', 'Gii']
clifford_compilation['Gc20c3'] = ['Gxx', 'Gyx', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c4'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gxx', 'Gix', 'Gii']
clifford_compilation['Gc20c5'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c6'] = ['Gxy', 'Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c7'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c8'] = ['Gxx', 'Gyy', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c9'] = ['Gxx', 'Gyx', 'Gyy', 'Gyy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c10'] = ['Gxy', 'Gyx', 'Gyx', 'Gyx', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c11'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c12'] = ['Gxy', 'Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c13'] = ['Gxx', 'Gyx', 'Gyx', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c14'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gix', 'Gix']
clifford_compilation['Gc20c15'] = ['Gxy', 'Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c16'] = ['Gxx', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c17'] = ['Gxx', 'Gyy', 'Gyx', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c18'] = ['Gxy', 'Gyy', 'Gyy', 'Gyx', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc20c19'] = ['Gxx', 'Gyy', 'Gyy', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c20'] = ['Gxx', 'Gyy', 'Gyy', 'Gyy', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc20c21'] = ['Gxy', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc20c22'] = ['Gxx', 'Gyx', 'Gyx', 'Gyy', 'Gxy', 'Gii', 'Gii']
clifford_compilation['Gc20c23'] = ['Gxx', 'Gyy', 'Gyx', 'Gyx', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc21c0'] = ['Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c1'] = ['Gyy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c2'] = ['Gyx', 'Gix', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']
clifford_compilation['Gc21c3'] = ['Gyx', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c4'] = ['Gyy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']
clifford_compilation['Gc21c5'] = ['Gyx', 'Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c6'] = ['Gyy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c7'] = ['Gyy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c8'] = ['Gyx', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c9'] = ['Gyx', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c10'] = ['Gyy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c11'] = ['Gyx', 'Gix', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c12'] = ['Gyy', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c13'] = ['Gyx', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c14'] = ['Gyx', 'Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']
clifford_compilation['Gc21c15'] = ['Gyy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c16'] = ['Gyx', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c17'] = ['Gyx', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c18'] = ['Gyy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc21c19'] = ['Gyx', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c20'] = ['Gyx', 'Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc21c21'] = ['Gyy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc21c22'] = ['Gyx', 'Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']
clifford_compilation['Gc21c23'] = ['Gyx', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']
clifford_compilation['Gc22c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c1'] = ['Gxy', 'Gxx', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c2'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gyy', 'Giy', 'Gii']
clifford_compilation['Gc22c3'] = ['Gxx', 'Gxx', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c4'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gyx', 'Gix', 'Gii']
clifford_compilation['Gc22c5'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c6'] = ['Gxy', 'Gxy', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c7'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c8'] = ['Gxx', 'Gxy', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c9'] = ['Gxx', 'Gxx', 'Gxy', 'Gyy', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c10'] = ['Gxy', 'Gxx', 'Gxx', 'Gyx', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c11'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c12'] = ['Gxy', 'Gxx', 'Gxx', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c13'] = ['Gxx', 'Gxx', 'Gxx', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c14'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gix', 'Gix']
clifford_compilation['Gc22c15'] = ['Gxy', 'Gxy', 'Gxy', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c16'] = ['Gxx', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c17'] = ['Gxx', 'Gxy', 'Gxx', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c18'] = ['Gxy', 'Gxy', 'Gxy', 'Gyx', 'Gyx', 'Gii', 'Gii']
clifford_compilation['Gc22c19'] = ['Gxx', 'Gxy', 'Gxy', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c20'] = ['Gxx', 'Gxy', 'Gxy', 'Gyy', 'Gyx', 'Gii', 'Gii']
clifford_compilation['Gc22c21'] = ['Gxy', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']
clifford_compilation['Gc22c22'] = ['Gxx', 'Gxx', 'Gxx', 'Gyy', 'Gyy', 'Gii', 'Gii']
clifford_compilation['Gc22c23'] = ['Gxx', 'Gxy', 'Gxx', 'Gyx', 'Gyx', 'Gii', 'Gii']
clifford_compilation['Gc23c0'] = ['Gxi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c1'] = ['Gxy', 'Gyx', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c2'] = ['Gxx', 'Gyx', 'Gxx', 'Gxy', 'Gxy', 'Giy', 'Gii']
clifford_compilation['Gc23c3'] = ['Gxx', 'Gyx', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c4'] = ['Gxy', 'Gyy', 'Gxy', 'Gxx', 'Gxx', 'Gix', 'Gii']
clifford_compilation['Gc23c5'] = ['Gxx', 'Gyy', 'Gxy', 'Gxy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c6'] = ['Gxy', 'Gyy', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c7'] = ['Gxy', 'Gyy', 'Gxy', 'Gxx', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c8'] = ['Gxx', 'Gyy', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c9'] = ['Gxx', 'Gyx', 'Gxy', 'Gxy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c10'] = ['Gxy', 'Gyx', 'Gxx', 'Gxx', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c11'] = ['Gxx', 'Gyx', 'Gxx', 'Gxy', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c12'] = ['Gxy', 'Gyx', 'Gxx', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c13'] = ['Gxx', 'Gyx', 'Gxx', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c14'] = ['Gxx', 'Gyy', 'Gxy', 'Gxy', 'Gxx', 'Gix', 'Gix']
clifford_compilation['Gc23c15'] = ['Gxy', 'Gyy', 'Gxy', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c16'] = ['Gxx', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c17'] = ['Gxx', 'Gyy', 'Gxx', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c18'] = ['Gxy', 'Gyy', 'Gxy', 'Gxx', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc23c19'] = ['Gxx', 'Gyy', 'Gxy', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c20'] = ['Gxx', 'Gyy', 'Gxy', 'Gxy', 'Gxx', 'Gii', 'Gii']
clifford_compilation['Gc23c21'] = ['Gxy', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']
clifford_compilation['Gc23c22'] = ['Gxx', 'Gyx', 'Gxx', 'Gxy', 'Gxy', 'Gii', 'Gii']
clifford_compilation['Gc23c23'] = ['Gxx', 'Gyy', 'Gxx', 'Gxx', 'Gxx', 'Gii', 'Gii']


global_fidPairs = [
    (0, 4), (0, 5), (1, 6), (2, 0), (2, 4), (2, 10), (3, 1),
    (3, 3), (3, 4), (3, 10), (4, 3), (4, 4), (4, 5), (4, 6),
    (4, 10), (5, 2), (6, 5), (7, 1), (7, 2), (7, 5), (7, 7),
    (7, 8), (7, 10), (8, 6), (8, 7), (8, 10), (9, 8), (10, 6),
    (10, 9), (10, 10), (11, 1), (11, 8), (12, 3), (13, 5), (13, 6),
    (14, 4), (14, 7)]

pergerm_fidPairsDict = {
    ('Gix',): [
        (0, 5), (1, 0), (1, 1), (2, 2), (2, 5), (2, 9), (3, 3),
        (3, 4), (3, 8), (4, 0), (4, 2), (4, 7), (4, 8), (4, 10),
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 8), (6, 7), (6, 8),
        (6, 9), (7, 0), (7, 4), (8, 5), (8, 9), (9, 5), (10, 8),
        (10, 10), (12, 2), (12, 4), (12, 7), (13, 2), (13, 3),
        (13, 9), (14, 0), (14, 5), (14, 6), (15, 5), (15, 8),
        (15, 9)],
    ('Gyx',): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4),
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4),
        (14, 9), (15, 5), (15, 7)],
    ('Gyi',): [
        (3, 1), (4, 1), (4, 2), (5, 0), (5, 1), (5, 7), (6, 0),
        (6, 8), (7, 2), (7, 4), (7, 9), (8, 0), (8, 7), (9, 2),
        (9, 3), (10, 9), (10, 10), (14, 7), (14, 9), (15, 10)],
    ('Giy',): [
        (0, 0), (0, 7), (1, 1), (3, 5), (3, 6), (4, 2), (4, 4),
        (4, 5), (5, 3), (5, 7), (7, 1), (7, 8), (8, 5), (9, 4),
        (9, 5), (9, 9), (10, 5), (11, 5), (11, 6), (11, 8), (11, 10),
        (12, 0), (12, 3), (13, 10), (14, 0), (14, 5), (14, 6),
        (14, 7), (15, 0), (15, 6), (15, 9)],
    ('Gyy',): [
        (0, 6), (0, 8), (0, 10), (1, 0), (1, 1), (1, 3), (2, 9),
        (3, 8), (4, 4), (4, 7), (5, 7), (6, 1), (7, 0), (7, 8),
        (9, 10), (10, 5), (11, 5), (12, 5), (12, 6), (14, 0),
        (15, 0), (15, 6), (15, 8)],
    ('Gxx',): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1),
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10),
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Gii',): [
        (0, 8), (1, 0), (1, 1), (1, 3), (1, 10), (2, 5), (2, 9),
        (3, 3), (3, 9), (4, 3), (4, 8), (5, 0), (5, 5), (5, 7),
        (6, 4), (6, 6), (6, 8), (6, 10), (7, 0), (7, 2), (7, 3),
        (7, 4), (7, 6), (7, 10), (8, 3), (8, 5), (9, 3), (9, 4),
        (9, 5), (9, 6), (9, 8), (9, 9), (10, 3), (10, 9), (10, 10),
        (11, 1), (11, 5), (12, 5), (12, 7), (12, 9), (13, 0),
        (13, 10), (14, 0), (14, 1), (14, 2), (14, 6), (15, 0),
        (15, 5), (15, 6), (15, 7), (15, 8)],
    ('Gxy',): [
        (1, 1), (2, 8), (3, 0), (3, 2), (3, 6), (4, 7), (7, 2),
        (8, 6), (9, 1), (9, 7), (9, 9), (10, 2), (10, 10), (11, 8),
        (12, 6), (13, 2), (13, 7), (14, 2), (15, 5)],
    ('Gxi',): [
        (0, 7), (1, 1), (1, 7), (2, 7), (3, 3), (4, 9), (5, 4),
        (7, 2), (7, 10), (8, 2), (9, 2), (9, 8), (9, 9), (10, 1),
        (10, 10), (11, 2), (11, 5), (11, 6), (13, 2), (14, 7),
        (15, 2), (15, 3)],
    ('Giy', 'Gyi'): [
        (0, 6), (0, 8), (0, 10), (1, 0), (1, 1), (1, 3), (2, 9),
        (3, 8), (4, 4), (4, 7), (5, 7), (6, 1), (7, 0), (7, 8),
        (9, 10), (10, 5), (11, 5), (12, 5), (12, 6), (14, 0),
        (15, 0), (15, 6), (15, 8)],
    ('Gxx', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxy', 'Gyx'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Giy', 'Gxi'): [
        (1, 1), (2, 8), (3, 0), (3, 2), (3, 6), (4, 7), (7, 2),
        (8, 6), (9, 1), (9, 7), (9, 9), (10, 2), (10, 10), (11, 8),
        (12, 6), (13, 2), (13, 7), (14, 2), (15, 5)],
    ('Gyi', 'Gxx'): [
        (1, 10), (2, 10), (4, 8), (5, 5), (5, 6), (6, 10), (7, 0),
        (7, 5), (7, 6), (7, 8), (8, 5), (12, 5), (13, 0), (13, 2),
        (14, 1)],
    ('Gii', 'Giy'): [
        (0, 0), (0, 7), (1, 1), (3, 5), (3, 6), (4, 2), (4, 4),
        (4, 5), (5, 3), (5, 7), (7, 1), (7, 8), (8, 5), (9, 4),
        (9, 5), (9, 9), (10, 5), (11, 5), (11, 6), (11, 8), (11, 10),
        (12, 0), (12, 3), (13, 10), (14, 0), (14, 5), (14, 6),
        (14, 7), (15, 0), (15, 6), (15, 9)],
    ('Gxx', 'Gxy'): [
        (1, 1), (2, 5), (4, 3), (5, 5), (6, 3), (7, 1), (10, 2),
        (10, 5), (11, 2), (11, 5), (12, 7), (12, 10), (13, 0),
        (13, 4), (14, 5)],
    ('Gyy', 'Gxy'): [
        (0, 9), (1, 1), (1, 9), (2, 7), (3, 4), (4, 4), (4, 10),
        (6, 0), (6, 3), (7, 0), (9, 4), (11, 5), (12, 4), (13, 7),
        (14, 0)],
    ('Gix', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5),
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6),
        (12, 9), (13, 9), (15, 1)],
    ('Gix', 'Gyy'): [
        (3, 0), (4, 4), (5, 1), (5, 8), (6, 5), (7, 3), (8, 6),
        (8, 7), (9, 5), (10, 3), (11, 4), (14, 0), (14, 6), (14, 9),
        (15, 5)],
    ('Gix', 'Gxi'): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1),
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10),
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Gii', 'Gyi'): [
        (3, 1), (4, 1), (4, 2), (5, 0), (5, 1), (5, 7), (6, 0),
        (6, 8), (7, 2), (7, 4), (7, 9), (8, 0), (8, 7), (9, 2),
        (9, 3), (10, 9), (10, 10), (14, 7), (14, 9), (15, 10)],
    ('Gix', 'Gyi'): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4),
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4),
        (14, 9), (15, 5), (15, 7)],
    ('Gxx', 'Gyx'): [
        (1, 10), (2, 10), (4, 8), (5, 5), (5, 6), (6, 10), (7, 0),
        (7, 5), (7, 6), (7, 8), (8, 5), (12, 5), (13, 0), (13, 2),
        (14, 1)],
    ('Gyi', 'Gxy'): [
        (0, 9), (1, 1), (1, 9), (2, 7), (3, 4), (4, 4), (4, 10),
        (6, 0), (6, 3), (7, 0), (9, 4), (11, 5), (12, 4), (13, 7),
        (14, 0)],
    ('Gii', 'Gix'): [
        (0, 5), (1, 0), (1, 1), (2, 2), (2, 5), (2, 9), (3, 3),
        (3, 4), (3, 8), (4, 0), (4, 2), (4, 7), (4, 8), (4, 10),
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 8), (6, 7), (6, 8),
        (6, 9), (7, 0), (7, 4), (8, 5), (8, 9), (9, 5), (10, 8),
        (10, 10), (12, 2), (12, 4), (12, 7), (13, 2), (13, 3),
        (13, 9), (14, 0), (14, 5), (14, 6), (15, 5), (15, 8),
        (15, 9)],
    ('Gxi', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Giy', 'Gxx'): [
        (0, 6), (3, 0), (5, 0), (6, 7), (7, 1), (8, 3), (9, 9),
        (10, 4), (10, 9), (12, 9), (13, 2), (14, 5), (14, 8),
        (14, 10), (15, 6)],
    ('Gix', 'Gxy', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyy', 'Gyx', 'Gyx'): [
        (0, 1), (0, 3), (0, 9), (2, 3), (2, 6), (3, 10), (5, 7),
        (6, 0), (7, 2), (7, 6), (7, 7), (8, 1), (8, 5), (9, 4),
        (14, 10)],
    ('Gix', 'Gxy', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gxy', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gxi', 'Gxx'): [
        (0, 6), (1, 3), (1, 7), (1, 10), (2, 10), (4, 1), (5, 1),
        (5, 5), (7, 3), (8, 2), (8, 3), (9, 8), (10, 1), (10, 6),
        (10, 10), (11, 7), (15, 3)],
    ('Gxx', 'Gyx', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gxx', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gyy', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gyi', 'Gxi'): [
        (1, 10), (2, 10), (4, 8), (5, 5), (5, 6), (6, 10), (7, 0),
        (7, 5), (7, 6), (7, 8), (8, 5), (12, 5), (13, 0), (13, 2),
        (14, 1)],
    ('Gxx', 'Gyx', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gxi', 'Giy'): [
        (0, 6), (3, 0), (5, 0), (6, 7), (7, 1), (8, 3), (9, 9),
        (10, 4), (10, 9), (12, 9), (13, 2), (14, 5), (14, 8),
        (14, 10), (15, 6)],
    ('Gxi', 'Gyx', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxx', 'Gxy', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gxx', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gii', 'Gii'): [
        (0, 7), (1, 1), (1, 7), (2, 7), (3, 3), (4, 9), (5, 4),
        (7, 2), (7, 10), (8, 2), (9, 2), (9, 8), (9, 9), (10, 1),
        (10, 10), (11, 2), (11, 5), (11, 6), (13, 2), (14, 7),
        (15, 2), (15, 3)],
    ('Gii', 'Giy', 'Gyi'): [
        (0, 6), (0, 8), (0, 10), (1, 0), (1, 1), (1, 3), (2, 9),
        (3, 8), (4, 4), (4, 7), (5, 7), (6, 1), (7, 0), (7, 8),
        (9, 10), (10, 5), (11, 5), (12, 5), (12, 6), (14, 0),
        (15, 0), (15, 6), (15, 8)],
    ('Giy', 'Gyx', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gyx', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Giy', 'Gyi'): [
        (0, 1), (4, 2), (4, 7), (6, 7), (8, 3), (9, 5), (9, 7),
        (10, 0), (10, 4), (10, 5), (11, 2), (11, 9), (14, 6),
        (14, 8), (15, 3)],
    ('Giy', 'Gii', 'Gii'): [
        (0, 0), (0, 7), (1, 1), (3, 5), (3, 6), (4, 2), (4, 4),
        (4, 5), (5, 3), (5, 7), (7, 1), (7, 8), (8, 5), (9, 4),
        (9, 5), (9, 9), (10, 5), (11, 5), (11, 6), (11, 8), (11, 10),
        (12, 0), (12, 3), (13, 10), (14, 0), (14, 5), (14, 6),
        (14, 7), (15, 0), (15, 6), (15, 9)],
    ('Giy', 'Gyx', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gyi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gxx', 'Gxi'): [
        (1, 5), (3, 3), (4, 1), (6, 1), (6, 6), (6, 8), (8, 6),
        (10, 10), (11, 8), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Gxx', 'Gyy', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gyx', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gii', 'Gii'): [
        (3, 1), (4, 1), (4, 2), (5, 0), (5, 1), (5, 7), (6, 0),
        (6, 8), (7, 2), (7, 4), (7, 9), (8, 0), (8, 7), (9, 2),
        (9, 3), (10, 9), (10, 10), (14, 7), (14, 9), (15, 10)],
    ('Gii', 'Gix', 'Gyi'): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4),
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4),
        (14, 9), (15, 5), (15, 7)],
    ('Gix', 'Gxy', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gix', 'Gxx', 'Giy'): [
        (0, 3), (1, 0), (1, 4), (3, 10), (4, 3), (5, 7), (7, 2),
        (7, 4), (7, 7), (7, 8), (8, 1), (8, 5), (8, 7), (8, 9),
        (9, 2), (9, 6), (10, 3), (14, 10), (15, 4)],
    ('Giy', 'Gxx', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gyy', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Giy', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gyi', 'Giy'): [
        (3, 0), (4, 4), (5, 1), (5, 8), (6, 5), (7, 3), (8, 6),
        (8, 7), (9, 5), (10, 3), (11, 4), (14, 0), (14, 6), (14, 9),
        (15, 5)],
    ('Giy', 'Gyi', 'Gxx'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gxi', 'Gii', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Giy', 'Gxi', 'Gxi'): [
        (1, 7), (2, 2), (4, 8), (7, 2), (7, 10), (8, 6), (9, 8),
        (9, 9), (10, 1), (11, 4), (11, 9), (12, 8), (12, 9),
        (13, 0), (13, 1), (13, 9)],
    ('Gxi', 'Gyy', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyy', 'Gxy', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxx', 'Gyy', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gyy', 'Gyx'): [
        (0, 2), (1, 0), (1, 4), (1, 9), (2, 4), (2, 10), (4, 3),
        (7, 4), (7, 8), (8, 7), (8, 9), (9, 2), (9, 6), (10, 3),
        (15, 4)],
    ('Gix', 'Gxy', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Giy', 'Giy'): [
        (0, 4), (0, 5), (0, 7), (1, 1), (1, 6), (2, 3), (4, 10),
        (5, 4), (6, 8), (7, 4), (7, 10), (8, 8), (8, 9), (10, 5),
        (11, 5), (11, 6), (11, 9), (13, 10), (14, 1), (14, 9)],
    ('Gxx', 'Gxx', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gyi', 'Gii'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gix', 'Gix', 'Giy'): [
        (0, 0), (0, 6), (1, 0), (1, 10), (4, 0), (4, 4), (4, 7),
        (4, 8), (5, 5), (6, 7), (7, 6), (8, 9), (9, 9), (10, 2),
        (10, 8), (11, 10), (12, 6), (12, 9), (13, 1), (13, 9),
        (15, 1)],
    ('Giy', 'Gxx', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gii', 'Gxx', 'Gxx'): [
        (1, 5), (3, 3), (4, 1), (6, 1), (6, 6), (6, 8), (8, 6),
        (10, 10), (11, 8), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Gyi', 'Gyi', 'Gyy'): [
        (0, 2), (1, 1), (1, 4), (2, 1), (2, 10), (3, 10), (4, 0),
        (5, 3), (5, 7), (6, 4), (6, 10), (8, 2), (8, 3), (9, 0),
        (10, 8), (11, 1), (11, 7), (13, 1), (13, 8)],
    ('Giy', 'Gxi', 'Gyi'): [
        (0, 9), (1, 1), (1, 9), (2, 7), (3, 4), (4, 4), (4, 10),
        (6, 0), (6, 3), (7, 0), (9, 4), (11, 5), (12, 4), (13, 7),
        (14, 0)],
    ('Gxx', 'Gxy', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gyi', 'Gxi'): [
        (0, 9), (1, 1), (1, 9), (2, 7), (3, 4), (4, 4), (4, 10),
        (6, 0), (6, 3), (7, 0), (9, 4), (11, 5), (12, 4), (13, 7),
        (14, 0)],
    ('Gii', 'Gyi', 'Gix'): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4),
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4),
        (14, 9), (15, 5), (15, 7)],
    ('Giy', 'Gyx', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Giy', 'Gxi'): [
        (0, 6), (3, 0), (5, 0), (6, 7), (7, 1), (8, 3), (9, 9),
        (10, 4), (10, 9), (12, 9), (13, 2), (14, 5), (14, 8),
        (14, 10), (15, 6)],
    ('Gix', 'Gix', 'Gxx'): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1),
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10),
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Gxi', 'Gyi', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxy', 'Gxy', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gyx', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxx', 'Gyy', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxy', 'Gyx', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gyy', 'Gyx'): [
        (0, 2), (1, 0), (1, 4), (1, 9), (2, 4), (2, 10), (4, 3),
        (7, 4), (7, 8), (8, 7), (8, 9), (9, 2), (9, 6), (10, 3),
        (15, 4)],
    ('Gyi', 'Gxx', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gxy', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Giy', 'Gii'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5),
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6),
        (12, 9), (13, 9), (15, 1)],
    ('Gyy', 'Gyx', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gii', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5),
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6),
        (12, 9), (13, 9), (15, 1)],
    ('Gix', 'Gii', 'Gii'): [
        (0, 5), (1, 0), (1, 1), (2, 2), (2, 5), (2, 9), (3, 3),
        (3, 4), (3, 8), (4, 0), (4, 2), (4, 7), (4, 8), (4, 10),
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 8), (6, 7), (6, 8),
        (6, 9), (7, 0), (7, 4), (8, 5), (8, 9), (9, 5), (10, 8),
        (10, 10), (12, 2), (12, 4), (12, 7), (13, 2), (13, 3),
        (13, 9), (14, 0), (14, 5), (14, 6), (15, 5), (15, 8),
        (15, 9)],
    ('Gyi', 'Gxy', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gxi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gyx', 'Gxy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gxy', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gxy', 'Gxy', 'Gyx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gix', 'Gxi', 'Giy'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gix', 'Gii', 'Gii', 'Gxi'): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1),
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10),
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Gix', 'Giy', 'Gxi', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gxx', 'Gxy', 'Gyx', 'Gyx'): [
        (1, 10), (2, 10), (4, 8), (5, 5), (5, 6), (6, 10), (7, 0),
        (7, 5), (7, 6), (7, 8), (8, 5), (12, 5), (13, 0), (13, 2),
        (14, 1)],
    ('Giy', 'Gxi', 'Gii', 'Gii'): [
        (1, 1), (2, 8), (3, 0), (3, 2), (3, 6), (4, 7), (7, 2),
        (8, 6), (9, 1), (9, 7), (9, 9), (10, 2), (10, 10), (11, 8),
        (12, 6), (13, 2), (13, 7), (14, 2), (15, 5)],
    ('Gxx', 'Gxx', 'Gyi', 'Gyi'): [
        (0, 1), (0, 3), (0, 9), (2, 3), (2, 6), (3, 10), (5, 7),
        (6, 0), (7, 2), (7, 6), (7, 7), (8, 1), (8, 5), (9, 4),
        (14, 10)],
    ('Gyi', 'Gix', 'Gix', 'Gix'): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4),
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4),
        (14, 9), (15, 5), (15, 7)],
    ('Gyi', 'Gyi', 'Giy', 'Gyi'): [
        (0, 2), (1, 1), (1, 4), (2, 1), (2, 10), (3, 10), (4, 0),
        (5, 3), (5, 7), (6, 4), (6, 10), (8, 2), (8, 3), (9, 0),
        (10, 8), (11, 1), (11, 7), (13, 1), (13, 8)],
    ('Gyx', 'Gyx', 'Gyy', 'Gyy'): [
        (1, 10), (2, 10), (4, 8), (5, 5), (5, 6), (6, 10), (7, 0),
        (7, 5), (7, 6), (7, 8), (8, 5), (12, 5), (13, 0), (13, 2),
        (14, 1)],
    ('Gix', 'Giy', 'Giy', 'Gii'): [
        (0, 4), (0, 5), (0, 7), (1, 1), (1, 6), (2, 3), (4, 10),
        (5, 4), (6, 8), (7, 4), (7, 10), (8, 8), (8, 9), (10, 5),
        (11, 5), (11, 6), (11, 9), (13, 10), (14, 1), (14, 9)],
    ('Gxi', 'Gyi', 'Gix', 'Gix'): [
        (1, 10), (2, 10), (4, 8), (5, 5), (5, 6), (6, 10), (7, 0),
        (7, 5), (7, 6), (7, 8), (8, 5), (12, 5), (13, 0), (13, 2),
        (14, 1)],
    ('Gyx', 'Gix', 'Gxy', 'Gxi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gyi', 'Gyi', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Giy', 'Gxx', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gix', 'Gix', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5),
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6),
        (12, 9), (13, 9), (15, 1)],
    ('Gxi', 'Gyi', 'Gyi', 'Gii'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gyy', 'Gyi', 'Gxx', 'Gxx'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gii', 'Gyy', 'Gxy', 'Gyy', 'Gyy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Giy', 'Gxi', 'Gyi', 'Gxi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyx', 'Gxy', 'Gxi', 'Gxy', 'Gxy'): [
        (3, 0), (4, 4), (5, 1), (5, 8), (6, 5), (7, 3), (8, 6),
        (8, 7), (9, 5), (10, 3), (11, 4), (14, 0), (14, 6), (14, 9),
        (15, 5)],
    ('Gyi', 'Gyi', 'Giy', 'Gxx', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Giy', 'Gyi', 'Gix', 'Gix'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gyi', 'Gxi', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gxi', 'Giy', 'Gyi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxx', 'Gxx', 'Gxx', 'Gix', 'Gxi'): [
        (1, 5), (3, 3), (4, 1), (6, 1), (6, 6), (6, 8), (8, 6),
        (10, 10), (11, 8), (13, 1), (13, 4), (13, 6), (13, 10),
        (14, 8), (15, 3)],
    ('Giy', 'Gxi', 'Gix', 'Giy', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Gix', 'Gxi', 'Gix', 'Gxi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Giy', 'Giy', 'Gix', 'Gxi', 'Gxi'): [
        (1, 1), (2, 5), (4, 3), (5, 5), (6, 3), (7, 1), (10, 2),
        (10, 5), (11, 2), (11, 5), (12, 7), (12, 10), (13, 0),
        (13, 4), (14, 5)],
    ('Gyi', 'Giy', 'Gxi', 'Giy', 'Giy', 'Giy'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gix'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gxi', 'Gix', 'Giy', 'Gxi', 'Giy', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gix', 'Giy', 'Giy', 'Gxi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxy', 'Gyx', 'Gix', 'Giy', 'Gxx', 'Gix'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Giy', 'Giy', 'Gxi', 'Giy', 'Gix', 'Giy'): [
        (0, 4), (0, 6), (1, 1), (2, 2), (4, 1), (4, 3), (5, 1),
        (5, 3), (6, 10), (8, 2), (8, 8), (9, 4), (10, 7), (12, 1),
        (13, 2), (15, 6), (15, 9)],
    ('Gyi', 'Gxi', 'Giy', 'Gyi', 'Gxx', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Gyi', 'Gxi', 'Gix', 'Giy', 'Gxi', 'Gix'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gyi', 'Gyi', 'Giy', 'Gyi', 'Gix'): [
        (0, 3), (1, 0), (1, 4), (3, 10), (4, 3), (5, 7), (7, 2),
        (7, 4), (7, 7), (7, 8), (8, 1), (8, 5), (8, 7), (8, 9),
        (9, 2), (9, 6), (10, 3), (14, 10), (15, 4)],
    ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5),
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6),
        (12, 9), (13, 9), (15, 1)],
    ('Gyi', 'Gxi', 'Gix', 'Gxi', 'Gix', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gxx', 'Gxx', 'Gxx', 'Gxi', 'Gix', 'Gii', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gii', 'Gyi', 'Gxi', 'Gxx', 'Gxx', 'Gix', 'Gxx'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10),
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8),
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
    ('Giy', 'Gix', 'Gyi', 'Gyi', 'Gix', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gyi', 'Gxi', 'Giy', 'Gxi', 'Gix', 'Gxi', 'Gyi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3),
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6),
        (15, 0), (15, 5)],
    ('Gix', 'Gix', 'Gyi', 'Gxi', 'Giy', 'Gxi', 'Giy', 'Gyi'): [
        (1, 1), (2, 5), (4, 3), (5, 5), (6, 3), (7, 1), (10, 2),
        (10, 5), (11, 2), (11, 5), (12, 7), (12, 10), (13, 0),
        (13, 4), (14, 5)],
}
