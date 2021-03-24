"""Defines Python-version calculation "representation" objects for external simulators"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import sys
import time as _time
import math as _math
import numpy as _np
from numpy.random import RandomState as _RandomState
import scipy.sparse as _sps
import itertools as _itertools
import functools as _functools

from ...tools import mpitools as _mpit
from ...tools import slicetools as _slct
from ...tools import matrixtools as _mt
from ...tools import listtools as _lt
from ...tools import optools as _gt
from ...tools.matrixtools import _fas

from scipy.sparse.linalg import LinearOperator

# TODO: Shift to base class (either ExternalOpRep or pushed up for all OpReps)
# Also maybe check we don't see a performance hit with the inheritance
# Representations for external simulators do not need to have acton methods
# as we do not always have access to the internal state of simulator
class ExternalOpRep(object):
    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()


class StochasticCHPOpRep(ExternalOpRep):
    class CHPOpRep(object):
        def __init__(self, ops, qubit_templates):
            self.ops = ops
            self.qubit_templates = qubit_templates
            self.dim = 4**len(self.qubit_templates)

    def __init__(self, ops_list, qubit_templates, probs):
        self.chp_ops = [self.CHPOpRep(ops, qubit_templates) for ops in ops_list]
        self.qubit_templates = qubit_templates
        self.probs = probs
        self.dim = 2**len(self.qubit_templates)

    

