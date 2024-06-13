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
import scipy.linalg as _spl
import scipy.optimize as _spo
import warnings as _warnings

from numpy.lib.arraysetops import isin

from pygsti.modelmembers.povms.computationalpovm import ComputationalBasisPOVM

from .composedstate import ComposedState
from .computationalstate import ComputationalBasisState
from .cptpstate import CPTPState
from .fullpurestate import FullPureState
from .fullstate import FullState
from .state import State
from .staticpurestate import StaticPureState
from .staticstate import StaticState
from .tensorprodstate import TensorProductState
from .tpstate import TPState
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot

# Avoid circular import
import pygsti.modelmembers as _mm


def create_from_pure_vector(pure_vector, state_type, basis='pp', evotype='default', state_space=None,
                            on_construction_error='warn'):
    """ TODO: docstring -- create a State from a state vector """
    state_type_preferences = (state_type,) if isinstance(state_type, str) else state_type
    if state_space is None:
        state_space = _statespace.default_space_for_udim(len(pure_vector))

    for typ in state_type_preferences:
        try:
            if typ == 'computational':
                st = ComputationalBasisState.from_pure_vector(pure_vector, basis, evotype, state_space)
            #elif typ == ('static stabilizer', 'static clifford'):
            #    st = StaticStabilizerState(...)  # TODO
            elif typ == 'static pure':
                st = StaticPureState(pure_vector, basis, evotype, state_space)
            elif typ == 'full pure':
                st = FullPureState(pure_vector, basis, evotype, state_space)
            elif typ in ('static', 'full', 'full TP', 'full CPTP'):
                superket = _bt.change_basis(_ot.state_to_dmvec(pure_vector), 'std', basis)
                st = create_from_dmvec(superket, typ, basis, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(typ)

                static_state = create_from_pure_vector(pure_vector, ('computational', 'static pure'),
                                                       basis, evotype, state_space)

                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, typ, proj_basis, basis,
                                                                  truncate=True, evotype=evotype,
                                                                  state_space=state_space)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                st = ComposedState(static_state, EffectiveExpErrorgen(errorgen))
            else:
                raise ValueError("Unknown state type '%s'!" % str(typ))

            return st  # if we get to here, then we've successfully created a state to return
        except (ValueError, AssertionError) as err:
            if on_construction_error == 'raise':
                raise err
            elif on_construction_error == 'warn':
                print('Failed to construct state with type "{}" with error: {}'.format(typ, str(err)))
            pass  # move on to next type

    raise ValueError("Could not create a state of type(s) %s from the given pure vector!" % (str(state_type)))


def create_from_dmvec(superket_vector, state_type, basis='pp', evotype='default', state_space=None):
    state_type_preferences = (state_type,) if isinstance(state_type, str) else state_type
    if state_space is None:
        state_space = _statespace.default_space_for_dim(len(superket_vector))

    for typ in state_type_preferences:
        try:
            if typ == "static":
                st = StaticState(superket_vector, basis, evotype, state_space)
            elif typ == "full":
                st = FullState(superket_vector, basis, evotype, state_space)
            elif typ == "full TP":
                st = TPState(superket_vector, basis, evotype, state_space)
            elif typ == "full CPTP":  # a non-lindbladian CPTP state that hasn't worked well...
                truncate = False
                st = CPTPState(superket_vector, basis, truncate, evotype, state_space)

            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(typ)

                try:
                    dmvec = _bt.change_basis(superket_vector, basis, 'std')
                    purevec = _ot.dmvec_to_state(dmvec)  # raises error if dmvec does not correspond to a pure state
                    static_state = StaticPureState(purevec, basis, evotype, state_space)
                except ValueError:
                    static_state = StaticState(superket_vector, basis, evotype, state_space)

                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, typ, proj_basis,
                                                                  basis, truncate=True, evotype=evotype)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                return ComposedState(static_state, EffectiveExpErrorgen(errorgen))

            else:
                # Anything else we try to convert to a pure vector and convert the pure state vector
                dmvec = _bt.change_basis(superket_vector, basis, 'std')
                purevec = _ot.dmvec_to_state(dmvec)
                st = create_from_pure_vector(purevec, typ, basis, evotype, state_space)
            return st
        except (ValueError, AssertionError):
            pass  # move on to next type

    raise ValueError("Could not create a state of type(s) %s from the given superket vector!" % (str(state_type)))


def state_type_from_op_type(op_type):
    """Decode an op type into an appropriate state type.

    Parameters:
    -----------
    op_type: str or list of str
        Operation parameterization type (or list of preferences)

    Returns
    -------
    str
        State parameterization type
    """
    op_type_preferences = _mm.operations.verbose_type_from_op_type(op_type)

    state_conversion = {
        'auto': 'computational',
        'static standard': 'computational',
        'static clifford': 'computational',
        'static unitary': 'static pure',
        'full unitary': 'full pure',
        'static': 'static',
        'full': 'full',
        'full TP': 'full TP',
        'full CPTP': 'full CPTP',
        'linear': 'full',
    }

    state_type_preferences = []
    for typ in op_type_preferences:
        state_type = None
        if _ot.is_valid_lindblad_paramtype(typ):
            # Lindblad types are passed through
            state_type = typ
        else:
            state_type = state_conversion.get(typ, None)

        if state_type is None:
            continue

        if state_type not in state_type_preferences:
            state_type_preferences.append(state_type)

    if len(state_type_preferences) == 0:
        raise ValueError(
            'Could not convert any op types from {}.\n'.format(op_type_preferences)
            + '\tKnown op_types: Lindblad types or {}\n'.format(sorted(list(state_conversion.keys())))
            + '\tValid state_types: Lindblad types or {}'.format(sorted(list(set(state_conversion.values()))))
        )

    return state_type_preferences


def convert(state, to_type, basis, ideal_state=None, flatten_structure=False):
    """
    TODO: update docstring
    Convert SPAM vector to a new type of parameterization.

    This potentially creates a new State object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    state : State
        State vector to convert

    to_type : {"full","full TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :meth:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `state`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    ideal_state : State, optional
        The ideal (usually pure) version of `state`,
        potentially used when converting to an error-generator type.

    flatten_structure : bool, optional
        When `False`, the sub-members of composed and embedded operations
        are separately converted, leaving the original state's structure
        unchanged.  When `True`, composed and embedded operations are "flattened"
        into a single state of the requested `to_type`.

    Returns
    -------
    State
        The converted State vector, usually a distinct
        object from the object passed as input.
    """
    to_types = to_type if isinstance(to_type, (tuple, list)) else (to_type,)  # HACK to support multiple to_type values
    error_msgs = {}

    destination_types = {'full': FullState,
                         'full TP': TPState,
                         'full CPTP': CPTPState,
                         'static': StaticState,
                         'static unitary': StaticPureState,
                         'static clifford': ComputationalBasisState}
    NoneType = type(None)

    for to_type in to_types:
        try:
            if isinstance(state, destination_types.get(to_type, NoneType)):
                return state

            if not flatten_structure and isinstance(state, ComposedState):
                return ComposedState(state.state_vec.copy(),  # don't convert (usually static) state vec
                                     _mm.operations.convert(state.error_map, to_type, basis, "identity",
                                                            flatten_structure))

            elif _ot.is_valid_lindblad_paramtype(to_type) and (ideal_state is not None or state.num_params == 0):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(to_type)

                st = ideal_state if (ideal_state is not None) else state
                proj_basis = 'PP' if state.state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state.state_space.dim, to_type, proj_basis,
                                                                  basis, truncate=True, evotype=state.evotype)
                if st is not state and not _np.allclose(st.to_dense(), state.to_dense()):
                    #Need to set errorgen so exp(errorgen)|st> == |state>
                    dense_st = st.to_dense()
                    dense_state = state.to_dense()

                    def _objfn(v):
                        errorgen.from_vector(v)
                        return _np.linalg.norm(_spl.expm(errorgen.to_dense()) @ dense_st - dense_state)
                    #def callback(x): print("callbk: ",_np.linalg.norm(x),_objfn(x))  # REMOVE
                    soln = _spo.minimize(_objfn, _np.zeros(errorgen.num_params, 'd'), method="CG", options={},
                                         tol=1e-8)  # , callback=callback)
                    #print("DEBUG: opt done: ",soln.success, soln.fun, soln.x)  # REMOVE
                    if not soln.success and soln.fun > 1e-6:  # not "or" because success is often not set correctly
                        raise ValueError("Failed to find an errorgen such that exp(errorgen)|ideal> = |state>")
                    errorgen.from_vector(soln.x)

                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                return ComposedState(st, EffectiveExpErrorgen(errorgen))
            else:
                min_space = state.evotype.minimal_space
                vec = state.to_dense(min_space)
                if min_space == 'Hilbert':
                    return create_from_pure_vector(vec, to_type, basis, state.evotype, state.state_space,
                                                   on_construction_error='raise')
                else:
                    return create_from_dmvec(vec, to_type, basis, state.evotype, state.state_space)

        except ValueError as e:
            #_warnings.warn('Failed to convert state to type %s with error: %s' % (to_type, e))
            error_msgs[to_type] = str(e)  # try next to_type

    raise ValueError("Could not convert state to to type(s): %s\n%s" % (str(to_types), str(error_msgs)))


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
    assert(target_vec.dim == vec_to_optimize.dim)  # vectors must have the same overall dimension
    targetVector = target_vec.to_dense() if isinstance(target_vec, State) else target_vec

    def _objective_func(param_vec):
        vec_to_optimize.from_vector(param_vec)
        return _np.linalg.norm(vec_to_optimize.to_dense() - targetVector)

    x0 = vec_to_optimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    vec_to_optimize.from_vector(minSol.x)
    return
