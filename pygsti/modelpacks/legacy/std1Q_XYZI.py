#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Variables for working with the a model containing Idle, X(pi/2), Y(pi/2), and Z(pi/2) gates.
"""

import sys as _sys

from ...circuits import circuitconstruction as _strc
from ...models import modelconstruction as _setc
from .. import stdtarget as _stdtarget

description = "Idle, X(pi/2), Y(pi/2), Z(pi/2) gates"

gates = ['Gi', 'Gx', 'Gy', 'Gz']
fiducials = _strc.to_circuits([(), ('Gx',), ('Gy',), ('Gx', 'Gx'),
                                ('Gx', 'Gx', 'Gx'), ('Gy', 'Gy', 'Gy')], line_labels=('*',))  # for 1Q MUB
prepStrs = effectStrs = fiducials

germs = _strc.to_circuits(
    [('Gi',), ('Gx',), ('Gy',), ('Gz',),
     ('Gx', 'Gz'), ('Gx', 'Gy'),
     ('Gx', 'Gx', 'Gy'), ('Gx', 'Gx', 'Gz'),
     ('Gy', 'Gy', 'Gz'), ('Gx', 'Gy', 'Gz'),
     ('Gx', 'Gy', 'Gi'), ('Gx', 'Gi', 'Gy'),
     ('Gx', 'Gi', 'Gi'), ('Gy', 'Gi', 'Gi'),
     ('Gi', 'Gx', 'Gz'), ('Gi', 'Gy', 'Gz'),
     ('Gx', 'Gy', 'Gy', 'Gi'), ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy')], line_labels=('*',))
germs_lite = germs[0:10]

#Construct a target model: Identity, X(pi/2), Y(pi/2)
_target_model = _setc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy', 'Gz'],
                                                             ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)", "Z(pi/2,Q0)"])

_gscache = {("full", "auto"): _target_model}


def processor_spec():
    return target_model('static').create_processor_spec(None)


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


global_fidPairs = [
    (0, 0), (2, 3), (5, 2), (5, 4)]

pergerm_fidPairsDict = {
    ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
    ('Gi',): [
        (0, 3), (1, 1), (5, 5)],
    ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
    ('Gy',): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
    ('Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
    ('Gx', 'Gy'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
    ('Gy', 'Gi', 'Gi'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
    ('Gx', 'Gx', 'Gz'): [
        (0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
    ('Gy', 'Gy', 'Gz'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
    ('Gx', 'Gy', 'Gi'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
    ('Gx', 'Gy', 'Gz'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
    ('Gx', 'Gi', 'Gy'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
    ('Gx', 'Gx', 'Gy'): [
        (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
    ('Gi', 'Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
    ('Gi', 'Gy', 'Gz'): [
        (0, 3), (3, 2), (4, 0), (5, 3)],
    ('Gx', 'Gi', 'Gi'): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
    ('Gx', 'Gy', 'Gy', 'Gi'): [
        (0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
    ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
}


global_fidPairs_lite = [
    (0, 4), (0, 5), (1, 0), (2, 0), (2, 4), (2, 5), (3, 0), (4, 2),
    (4, 4), (5, 1), (5, 2), (5, 3)]

pergerm_fidPairsDict_lite = {
    ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
    ('Gi',): [
        (0, 3), (1, 1), (5, 5)],
    ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
    ('Gy',): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
    ('Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
    ('Gx', 'Gy'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
    ('Gy', 'Gy', 'Gz'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
    ('Gx', 'Gy', 'Gz'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
    ('Gx', 'Gx', 'Gy'): [
        (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
    ('Gx', 'Gx', 'Gz'): [
        (0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
}
