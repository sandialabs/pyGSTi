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

import numpy as _np

from .composedstate import ComposedState
from .computationalstate import ComputationalBasisState
from .cptpstate import CPTPState
# from .densestate  REMOVE?
from .fullpurestate import FullPureState
from .fullstate import FullState
# from .purestate #TODO??
from .state import State
from .staticpurestate import StaticPureState
from .staticstate import StaticState
from .tensorprodstate import TensorProductState
from .tpstate import TPState
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot


def create_from_pure_vector(pure_vector, state_type, basis='pp', evotype='default', state_space=None):
    """ TODO: docstring -- create a State from a state vector """
    state_type_preferences = (state_type,) if isinstance(state_type, str) else state_type
    if state_space is None:
        state_space = _statespace.default_space_for_udim(len(pure_vector))

    for typ in state_type_preferences:
        try:
            if typ in ('computational', 'static standard'):
                st = ComputationalBasisState.from_pure_vector(pure_vector, basis, evotype, state_space)
            #elif typ == ('static stabilizer', 'static clifford'):
            #    st = StaticStabilizerState(...)  # TODO
            elif typ == ('static pure', 'static unitary'):
                st = StaticPureState(pure_vector, basis, evotype, state_space)
            elif typ == ('full pure', 'full unitary'):
                st = FullPureState(pure_vector, basis, evotype, state_space)
            elif typ in ('static', 'full', 'full TP', 'TrueCPTP'):
                superket = _bt.change_basis(_ot.state_to_dmvec(pure_vector), 'std', basis)
                st = create_from_dmvec(superket, typ, basis, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                static_state = create_from_pure_vector(pure_vector, ('computational', 'static pure'),
                                                       basis, evotype, state_space)

                proj_basis = 'pp' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, typ, proj_basis, basis,
                                                                  truncate=True, evotype=evotype,
                                                                  state_space=state_space)
                st = ComposedState(static_state, _ExpErrorgenOp(errorgen))
            else:
                raise ValueError("Unknown state type '%s'!" % str(typ))

            return st  # if we get to here, then we've successfully created a state to return
        except (ValueError, AssertionError):
            pass  # move on to next type

    raise ValueError("Could not create a state of type(s) %s from the given pure vector!" % (str(state_type)))


def create_from_dmvec(superket_vector, state_type, basis='pp', evotype='default', state_space=None):
    state_type_preferences = (state_type,) if isinstance(state_type, str) else state_type

    for typ in state_type_preferences:
        try:
            if typ == "static":
                st = StaticState(superket_vector, evotype, state_space)
            elif typ == "full":
                st = FullState(superket_vector, evotype, state_space)
            elif typ == "full TP":
                st = TPState(superket_vector, evotype, state_space)
            elif typ == "TrueCPTP":  # a non-lindbladian CPTP state that hasn't worked well...
                truncate = False
                st = CPTPState(superket_vector, basis, truncate, evotype, state_space)
            else:
                # Anything else we try to convert to a pure vector and convert the pure state vector
                dmvec = _bt.change_basis(state.to_dense(), basis, 'std')
                purevec = _ot.dmvec_to_state(dmvec)
                st = create_from_pure_vector(purevec, typ, basis, evotype, state_space)
            return st
        except (ValueError, AssertionError):
            pass  # move on to next type

    raise ValueError("Could not create a state of type(s) %s from the given superket vector!" % (str(state_type)))


def convert(state, to_type, basis, extra=None):
    """
    TODO: update docstring
    Convert SPAM vector to a new type of parameterization.

    This potentially creates a new State object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    state : State
        State vector to convert

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
    State
        The converted State vector, usually a distinct
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
        errorgen = _LindbladErrorgen.from_error_generator(state.state_space.dim, to_type, proj_basis,
                                                          basis, truncate=True, evotype=state.evotype)
        return ComposedState(static_state, _ExpErrorgenOp(errorgen))

    elif to_type == "static clifford":
        if isinstance(state, ComputationalBasisState):
            return state  # no conversion necessary

        purevec = state.to_dense().flatten()  # assume a pure state (otherwise would need to change Model dim)
        return ComputationalBasisState.from_pure_vector(purevec)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


def finite_difference_deriv_wrt_params(state, wrt_filter=None, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a State object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the spam vector with respect to a single
    parameter, matching the format expected from the spam vectors's
    `deriv_wrt_params` method.

    Parameters
    ----------
    state : State
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
    dim = state.dim
    state2 = state.copy()
    p = state.to_vector()
    fd_deriv = _np.empty((dim, state.num_params), 'd')  # assume real (?)

    for i in range(state.num_params):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        state2.from_vector(p_plus_dp, close=True)
        fd_deriv[:, i:i + 1] = (state2 - state) / eps

    fd_deriv.shape = [dim, state.num_params]
    if wrt_filter is None:
        return fd_deriv
    else:
        return _np.take(fd_deriv, wrt_filter, axis=1)


def check_deriv_wrt_params(state, deriv_to_check=None, wrt_filter=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a State object.

    This routine is meant to be used as an aid in testing and debugging
    State classes by comparing the finite-difference Jacobian that
    *should* be returned by `state.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    state : State
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `state.deriv_wrt_parms()` is used.  Setting this
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
    fd_deriv = finite_difference_deriv_wrt_params(state, wrt_filter, eps)
    if deriv_to_check is None:
        deriv_to_check = state.deriv_wrt_params()

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

    The optimization is performed so that the the resulting State vector is as
    close as possible to target_vec.

    This is trivial for the case of FullState instances, but for other types
    of parameterization this involves an iterative optimization over all the
    parameters of vec_to_optimize.

    Parameters
    ----------
    vec_to_optimize : State
        The state vector to optimize. This object gets altered.

    target_vec : State
        The state vector used as the target.

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

    from pygsti import optimize as _opt
    from pygsti.tools import matrixtools as _mt
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
