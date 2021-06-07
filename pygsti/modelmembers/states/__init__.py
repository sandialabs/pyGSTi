"""
Sub-package holding model state preparation objects.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .composedstate import ComposedState
from .computationalstate import ComputationalBasisState
from .cptpstate import CPTPState
#from .densestate  REMOVE?
from .fullpurestate import FullPureState
from .fullstate import FullState
#from .purestate #TODO??
from .state import State
from .staticstate import StaticState
from .staticpurestate import StaticPureState
from .tensorprodstate import TensorProductState
from .tpstate import TPState


import numpy as _np
from ...tools import optools as _ot
from ...tools import basistools as _bt


def convert(state, to_type, basis, extra=None):
    """
    TODO: update docstring
    Convert SPAM vector to a new type of parameterization.

    This potentially creates a new SPAMVec object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    state : SPAMVec
        SPAM vector to convert

    to_type : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `state`.  Allowed values are Matrix-unit (std),
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
        if isinstance(state, FullState):
            return state  # no conversion necessary
        else:
            return FullState(state.to_dense(), state.evotype, state.state_space)

    elif to_type == "TP":
        if isinstance(state, TPState):
            return state  # no conversion necessary
        else:
            return TPState(state.to_dense(), state.evotype, state.state_space)
            # above will raise ValueError if conversion cannot be done

    elif to_type == "TrueCPTP":  # a non-lindbladian CPTP state that hasn't worked well...
        if isinstance(state, CPTPState):
            return state  # no conversion necessary
        else:
            truncate = False
            return CPTPState(state.to_dense(), basis, truncate, state.evotype, state.state_space)
            # above will raise ValueError if conversion cannot be done

    elif to_type == "static":
        if isinstance(state, StaticState):
            return state  # no conversion necessary
        else:
            return StaticState(state.to_dense(), state.evotype, state.state_space)

    elif to_type == "static unitary":
        dmvec = _bt.change_basis(state.to_dense(), basis, 'std')
        purevec = _ot.dmvec_to_state(dmvec)
        return StaticPureState(purevec, basis, state.evotype, state.state_space)

    elif _ot.is_valid_lindblad_paramtype(to_type):

        from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
        purevec = None
        if isinstance(state, (FullState, TPState, StaticState)):
            try:
                dmvec = _bt.change_basis(state.to_dense(), basis, 'std')
                purevec = _ot.dmvec_to_state(dmvec)  # raises error if dmvec does not correspond to a pure state
            except ValueError:
                purevec = None

        if purevec is not None:
            static_state = StaticPureState(purevec, basis, state.evotype, state.state_space)
        elif state.num_params > 0:  # then we need to convert to a static state
            static_state = StaticState(state.to_dense(), state.evotype, state.state_space)
        else:  # state.num_params == 0 so it's already static
            static_state = state

        proj_basis = 'pp' if state.state_space.is_entirely_qubits else basis
        nonham_mode, param_mode, use_ham_basis, use_nonham_basis = \
            _LindbladErrorgen.decomp_paramtype(to_type)
        ham_basis = proj_basis if use_ham_basis else None
        nonham_basis = proj_basis if use_nonham_basis else None

        errorgen = _LindbladErrorgen.from_error_generator(_np.zeros((state.state_space.dim,
                                                                     state.state_space.dim), 'd'),
                                                          ham_basis, nonham_basis, param_mode, nonham_mode,
                                                          basis, truncate=True, evotype=state.evotype)
        return ComposedState(static_state, _ExpErrorgenOp(errorgen))

    elif to_type == "static clifford":
        if isinstance(state, ComputationalBasisState):
            return state  # no conversion necessary

        purevec = state.to_dense().flatten()  # assume a pure state (otherwise would need to change Model dim)
        return ComputationalBasisState.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


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


def optimize_state(vec_to_optimize, target_vec):
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
    if isinstance(vec_to_optimize, StaticState):
        return  # nothing to optimize

    if isinstance(vec_to_optimize, FullState):
        if(target_vec.dim != vec_to_optimize.dim):  # special case: gates can have different overall dimension
            vec_to_optimize.dim = target_vec.dim  # this is a HACK to allow model selection code to work correctly
        vec_to_optimize.set_dense(target_vec.to_dense())  # just copy entire overall matrix since fully parameterized
        return

    from ... import optimize as _opt
    from ...tools import matrixtools as _mt
    assert(target_vec.dim == vec_to_optimize.dim)  # vectors must have the same overall dimension
    targetVector = target_vec.to_dense() if isinstance(target_vec, State) else target_vec

    def _objective_func(param_vec):
        vec_to_optimize.from_vector(param_vec)
        return _mt.frobeniusnorm(vec_to_optimize.to_dense() - targetVector)

    x0 = vec_to_optimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    vec_to_optimize.from_vector(minSol.x)
    #print("DEBUG: optimized vector to min frobenius distance %g" % _mt.frobeniusnorm(vec_to_optimize-targetVector))
