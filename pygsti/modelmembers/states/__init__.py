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
            return FullState(state.to_dense())

    elif to_type == "TP":
        if isinstance(state, TPState):
            return state  # no conversion necessary
        else:
            return TPState(state.to_dense())
            # above will raise ValueError if conversion cannot be done

    elif to_type == "TrueCPTP":  # a non-lindbladian CPTP state that hasn't worked well...
        if isinstance(state, CPTPState):
            return state  # no conversion necessary
        else:
            return CPTPState(state, basis)
            # above will raise ValueError if conversion cannot be done

    elif to_type == "static":
        if isinstance(state, StaticState):
            return state  # no conversion necessary
        else:
            return StaticState(state)

    elif to_type == "static unitary":
        dmvec = _bt.change_basis(state.to_dense(), basis, 'std')
        purevec = _ot.dmvec_to_state(dmvec)
        return StaticPureState(purevec, "statevec", state._prep_or_effect)

    elif _ot.is_valid_lindblad_paramtype(to_type):

        if extra is None:
            purevec = state  # right now, we don't try to extract a "closest pure vec"
            # to state - below will fail if state isn't pure.
        else:
            purevec = extra  # assume extra info is a pure vector

        nQubits = _np.log2(state.dim) / 2.0
        bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis
        typ = state._prep_or_effect if isinstance(state, State) else "prep"

        return LindbladState._from_state_obj(
            state, typ, to_type, None, proj_basis, basis,
            truncate=True, lazy=True)

    elif to_type == "clifford":
        if isinstance(state, StabilizerState):
            return state  # no conversion necessary

        purevec = state.flatten()  # assume a pure state (otherwise would
        # need to change Model dim)
        return StabilizerState.from_dense_purevec(purevec)

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
