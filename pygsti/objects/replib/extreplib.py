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


## SS TODO: Finish flushing this out
class CHPOpRep(object):
    def __init__(self, chp_list):
        self.chp_list = chp_list
        self.dim = len(chp_list)

    # Representations for external simulators do not need to have acton methods
    # as we do not always have access to the internal state of simulator

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()