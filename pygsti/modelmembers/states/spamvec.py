"""
Defines classes with represent SPAM operations, along with supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.sparse as _sps
import collections as _collections
import numbers as _numbers
import itertools as _itertools
import functools as _functools
import copy as _copy

from .. import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools import symplectic as _symp
from .basis import Basis as _Basis
from .protectedarray import ProtectedArray as _ProtectedArray
from . import modelmember as _modelmember
from .errorgencontainer import ErrorMapContainer as _ErrorMapContainer

from . import term as _term
from . import stabilizer as _stabilizer
from .polynomial import Polynomial as _Polynomial
from . import replib
from .opcalc import bulk_eval_compact_polynomials_complex as _bulk_eval_compact_polynomials_complex

try:
    from ..tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


IMAG_TOL = 1e-8  # tolerance for imaginary part being considered zero


def optimize_spamvec(vec_to_optimize, target_vec):
    """
    Optimize the parameters of vec_to_optimize.

    The optimization is performed so that the the resulting SPAM vector is as
    close as possible to target_vec.

    This is trivial for the case of FullSPAMVec instances, but for other types
    of parameterization this involves an iterative optimization over all the
    parameters of vec_to_optimize.

    Parameters
    ----------
    vec_to_optimize : SPAMVec
        The vector to optimize. This object gets altered.

    target_vec : SPAMVec
        The SPAM vector used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(vec_to_optimize, StaticSPAMVec):
        return  # nothing to optimize

    if isinstance(vec_to_optimize, FullSPAMVec):
        if(target_vec.dim != vec_to_optimize.dim):  # special case: gates can have different overall dimension
            vec_to_optimize.dim = target_vec.dim  # this is a HACK to allow model selection code to work correctly
        vec_to_optimize.set_dense(target_vec)  # just copy entire overall matrix since fully parameterized
        return

    assert(target_vec.dim == vec_to_optimize.dim)  # vectors must have the same overall dimension
    targetVector = _np.asarray(target_vec)

    def _objective_func(param_vec):
        vec_to_optimize.from_vector(param_vec)
        return _mt.frobeniusnorm(vec_to_optimize - targetVector)

    x0 = vec_to_optimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    vec_to_optimize.from_vector(minSol.x)
    #print("DEBUG: optimized vector to min frobenius distance %g" % _mt.frobeniusnorm(vec_to_optimize-targetVector))




























