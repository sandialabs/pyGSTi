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


def convert(spamvec, to_type, basis, extra=None):
    """
    Convert SPAM vector to a new type of parameterization.

    This potentially creates a new SPAMVec object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    to_type : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `spamvec`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.

    Returns
    -------
    SPAMVec
        The converted SPAM vector, usually a distinct
        object from the object passed as input.
    """
    if to_type == "full":
        if isinstance(spamvec, FullSPAMVec):
            return spamvec  # no conversion necessary
        else:
            typ = spamvec._prep_or_effect if isinstance(spamvec, SPAMVec) else "prep"
            return FullSPAMVec(spamvec.to_dense(), typ=typ)

    elif to_type == "TP":
        if isinstance(spamvec, TPSPAMVec):
            return spamvec  # no conversion necessary
        else:
            return TPSPAMVec(spamvec.to_dense())
            # above will raise ValueError if conversion cannot be done

    elif to_type == "TrueCPTP":  # a non-lindbladian CPTP spamvec that hasn't worked well...
        if isinstance(spamvec, CPTPSPAMVec):
            return spamvec  # no conversion necessary
        else:
            return CPTPSPAMVec(spamvec, basis)
            # above will raise ValueError if conversion cannot be done

    elif to_type == "static":
        if isinstance(spamvec, StaticSPAMVec):
            return spamvec  # no conversion necessary
        else:
            typ = spamvec._prep_or_effect if isinstance(spamvec, SPAMVec) else "prep"
            return StaticSPAMVec(spamvec, typ=typ)

    elif to_type == "static unitary":
        dmvec = _bt.change_basis(spamvec.to_dense(), basis, 'std')
        purevec = _gt.dmvec_to_state(dmvec)
        return StaticSPAMVec(purevec, "statevec", spamvec._prep_or_effect)

    elif _gt.is_valid_lindblad_paramtype(to_type):

        if extra is None:
            purevec = spamvec  # right now, we don't try to extract a "closest pure vec"
            # to spamvec - below will fail if spamvec isn't pure.
        else:
            purevec = extra  # assume extra info is a pure vector

        nQubits = _np.log2(spamvec.dim) / 2.0
        bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis
        typ = spamvec._prep_or_effect if isinstance(spamvec, SPAMVec) else "prep"

        return LindbladSPAMVec._from_spamvec_obj(
            spamvec, typ, to_type, None, proj_basis, basis,
            truncate=True, lazy=True)

    elif to_type == "clifford":
        if isinstance(spamvec, StabilizerSPAMVec):
            return spamvec  # no conversion necessary

        purevec = spamvec.flatten()  # assume a pure state (otherwise would
        # need to change Model dim)
        return StabilizerSPAMVec.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


def _convert_to_lindblad_base(vec, typ, new_evotype, mx_basis="pp"):
    """
    Attempts to convert `vec` to a static (0 params) SPAMVec with
    evoution type `new_evotype`.  Used to convert spam vecs to
    being LindbladSPAMVec objects.
    """
    if vec._evotype == new_evotype and vec.num_params == 0:
        return vec  # no conversion necessary
    if new_evotype == "densitymx":
        return StaticSPAMVec(vec.to_dense(), "densitymx", typ)
    if new_evotype in ("svterm", "cterm"):
        if isinstance(vec, ComputationalSPAMVec):  # special case when conversion is easy
            return ComputationalSPAMVec(vec._zvals, new_evotype, typ)
        elif vec._evotype == "densitymx":
            # then try to extract a (static) pure state from vec wth
            # evotype 'statevec' or 'stabilizer' <=> 'svterm', 'cterm'
            if isinstance(vec, DenseSPAMVec):
                dmvec = _bt.change_basis(vec, mx_basis, 'std')
                purestate = StaticSPAMVec(_gt.dmvec_to_state(dmvec), 'statevec', typ)
            elif isinstance(vec, PureStateSPAMVec):
                purestate = vec.pure_state_vec  # evotype 'statevec'
            else:
                raise ValueError("Unable to obtain pure state from density matrix type %s!" % type(vec))

            if new_evotype == "cterm":  # then purestate 'statevec' => 'stabilizer' (if possible)
                if typ == "prep":
                    purestate = StabilizerSPAMVec.from_dense_purevec(purestate.to_dense())
                else:  # type == "effect"
                    purestate = StabilizerEffectVec.from_dense_purevec(purestate.to_dense())

            return PureStateSPAMVec(purestate, new_evotype, mx_basis, typ)

    raise ValueError("Could not convert %s (evotype %s) to %s w/0 params!" %
                     (str(type(vec)), vec._evotype, new_evotype))


def finite_difference_deriv_wrt_params(spamvec, wrt_filter=None, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a SPAMVec object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the spam vector with respect to a single
    parameter, matching the format expected from the spam vectors's
    `deriv_wrt_params` method.

    Parameters
    ----------
    spamvec : SPAMVec
        The spam vector object to compute a Jacobian for.

    wrt_filter : list or numpy.ndarray
        List of parameter indices to take derivative with respect to.
        (None means to use all the this operation's parameters.)

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    numpy.ndarray
        An M by N matrix where M is the number of gate elements and
        N is the number of gate parameters.
    """
    dim = spamvec.dim
    spamvec2 = spamvec.copy()
    p = spamvec.to_vector()
    fd_deriv = _np.empty((dim, spamvec.num_params), 'd')  # assume real (?)

    for i in range(spamvec.num_params):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        spamvec2.from_vector(p_plus_dp, close=True)
        fd_deriv[:, i:i + 1] = (spamvec2 - spamvec) / eps

    fd_deriv.shape = [dim, spamvec.num_params]
    if wrt_filter is None:
        return fd_deriv
    else:
        return _np.take(fd_deriv, wrt_filter, axis=1)


def check_deriv_wrt_params(spamvec, deriv_to_check=None, wrt_filter=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a SPAMVec object.

    This routine is meant to be used as an aid in testing and debugging
    SPAMVec classes by comparing the finite-difference Jacobian that
    *should* be returned by `spamvec.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    spamvec : SPAMVec
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `spamvec.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a LinearOperator
        class's `deriv_wrt_params()` method itself as a part of testing.

    wrt_filter : list or numpy.ndarray
        List of parameter indices to take derivative with respect to.
        (None means to use all the this operation's parameters.)

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    None
    """
    fd_deriv = finite_difference_deriv_wrt_params(spamvec, wrt_filter, eps)
    if deriv_to_check is None:
        deriv_to_check = spamvec.deriv_wrt_params()

    #print("Deriv shapes = %s and %s" % (str(fd_deriv.shape),
    #                                    str(deriv_to_check.shape)))
    #print("finite difference deriv = \n",fd_deriv)
    #print("deriv_wrt_params deriv = \n",deriv_to_check)
    #print("deriv_wrt_params - finite diff deriv = \n",
    #      deriv_to_check - fd_deriv)

    for i in range(deriv_to_check.shape[0]):
        for j in range(deriv_to_check.shape[1]):
            diff = abs(deriv_to_check[i, j] - fd_deriv[i, j])
            if diff > 5 * eps:
                print("deriv_chk_mismatch: (%d,%d): %g (comp) - %g (fd) = %g" %
                      (i, j, deriv_to_check[i, j], fd_deriv[i, j], diff))

    if _np.linalg.norm(fd_deriv - deriv_to_check) > 100 * eps:
        raise ValueError("Failed check of deriv_wrt_params:\n"
                         " norm diff = %g" %
                         _np.linalg.norm(fd_deriv - deriv_to_check))


























