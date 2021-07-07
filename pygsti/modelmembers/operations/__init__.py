"""
Sub-package holding model operation objects.
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

from .composederrorgen import ComposedErrorgen
from .composedop import ComposedOp
from .denseop import DenseOperator, DenseOperatorInterface
from .depolarizeop import DepolarizeOp
from .eigpdenseop import EigenvalueParamDenseOp
from .embeddederrorgen import EmbeddedErrorgen
from .embeddedop import EmbeddedOp
from .experrorgenop import ExpErrorgenOp
from .fullarbitraryop import FullArbitraryOp
from .fulltpop import FullTPOp
from .fullunitaryop import FullUnitaryOp
from .lindbladerrorgen import LindbladErrorgen, LindbladParameterization
from .linearop import LinearOperator
from .linearop import finite_difference_deriv_wrt_params, finite_difference_hessian_wrt_params
from .lpdenseop import LinearlyParamArbitraryOp
from .opfactory import OpFactory
from .staticarbitraryop import StaticArbitraryOp
from .staticcliffordop import StaticCliffordOp
from .staticstdop import StaticStandardOp
from .staticunitaryop import StaticUnitaryOp
from .stochasticop import StochasticNoiseOp
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot



def create_from_unitary_mx(unitary_mx, op_type, basis='pp', stdname=None, evotype='default', state_space=None):
    """ TODO: docstring - note that op_type can be a list/tuple of types in order of precedence """
    op_type_preferences = (op_type,) if isinstance(op_type, str) else op_type
    U = unitary_mx
    if state_space is None:
        state_space = _statespace.default_space_for_udim(U.shape[0])

    for typ in op_type_preferences:
        try:
            if typ == 'static standard' and stdname is not None:
                op = StaticStandardOp(stdname, basis, evotype, state_space)
            elif typ == 'static clifford':
                op = StaticCliffordOp(U, None, basis, evotype, state_space)
            elif typ == 'static unitary':
                op = StaticUnitaryOp(U, basis, evotype, state_space)
            elif typ == "full unitary":
                op = FullUnitaryOp(U, basis, evotype, state_space)
            elif typ in ('static', 'full', 'full TP', 'linear'):
                superop_mx = _bt.change_basis(_ot.unitary_to_process_mx(U), 'std', basis)
                op = create_from_superop_mx(superop_mx, op_type, basis, stdname, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):  # maybe "lindblad XXX" where XXX is a valid lindblad type?
                if _np.allclose(U, _np.identity(U.shape[0], 'd')):
                    unitary_postfactor = None
                else:
                    unitary_postfactor = create_from_unitary_mx(
                        U, ('static standard', 'static clifford', 'static unitary'),
                        basis, stdname, evotype, state_space)
                    
                proj_basis = 'pp' if state_space.is_entirely_qubits else basis
                errorgen = LindbladErrorgen.from_error_generator(state_space.dim, typ, proj_basis, basis,
                                                                 truncate=True, evotype=evotype,
                                                                 state_space=state_space)

                op = ExpErrorgenOp(errorgen) if (unitary_postfactor is None) \
                    else ComposedOp([unitary_postfactor, ExpErrorgenOp(errorgen)])

                if op.dim <= 16:  # only do this for up to 2Q operations, otherwise to_dense is too expensive
                    expected_superop_mx = _bt.change_basis(_ot.unitary_to_process_mx(U), 'std', basis)
                    assert (_np.linalg.norm(op.to_dense() - expected_superop_mx) < 1e-6), \
                        "Failure to create Lindblad operation (maybe due the complex log's branch cut?)"
            else:
                raise ValueError("Unknown operation type '%s'!" % str(typ))

            return op  # if we get to here, then we've successfully created an op to return
        except (ValueError, AssertionError, AttributeError):
            pass  # move on to next type

    raise ValueError("Could not create an operator of type(s) %s from the given unitary op!" % (str(op_type)))


def create_from_superop_mx(superop_mx, op_type, basis='pp', stdname=None, evotype='default', state_space=None):
    op_type_preferences = (op_type,) if isinstance(op_type, str) else op_type

    for typ in op_type_preferences:
        try:
            if typ == "static":  # "static arbitrary"?
                op = StaticArbitraryOp(superop_mx, evotype, state_space)
            elif typ == "full":  # "full arbitrary"?
                op = FullArbitraryOp(superop_mx, evotype, state_space)
            elif typ == "full TP":
                op = FullTPOp(superop_mx, evotype, state_space)
            elif typ == "linear":  # "linear arbitrary"?
                real = _np.isclose(_np.linalg.norm(superop_mx.imag), 0)
                op = LinearlyParamArbitraryOp(superop_mx, _np.array([]), {}, real, evotype, state_space)
            else:
                #Anything else we try to convert to a unitary and convert the unitary
                from pygsti.tools import jamiolkowski as _jt; RANK_TOL = 1e-6
                J = _jt.fast_jamiolkowski_iso_std(superop_mx, op_mx_basis=basis)  # Choi mx basis doesn't matter
                if _np.linalg.matrix_rank(J, RANK_TOL) > 1:
                    raise ValueError("`superop_mx` is not a unitary action!")

                std_superop_mx = _bt.change_basis(superop_mx, basis, 'std')
                unitary_mx = _ot.process_mx_to_unitary(std_superop_mx)
                op = create_from_unitary_mx(unitary_mx, typ, basis, stdname, evotype, state_space)

            return op  # if we get to here, then we've successfully created an op to return
        except (ValueError, AssertionError):
            pass  # move on to next type

    raise ValueError("Could not create an operator of type(s) %s from the given superop!" % (str(op_type)))


def convert(operation, to_type, basis, extra=None):
    """
    Convert operation to a new type of parameterization.

    This potentially creates a new LinearOperator object, and
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    operation : LinearOperator
        LinearOperator to convert

    to_type : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `operation`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.

    Returns
    -------
    LinearOperator
        The converted operation, usually a distinct
        object from the operation object passed as input.
    """
    if to_type == "full":
        if isinstance(operation, FullArbitraryOp):
            return operation  # no conversion necessary
        else:
            return FullArbitraryOp(operation.to_dense(), operation.evotype, operation.state_space)

    elif to_type == "TP":
        if isinstance(operation, FullTPOp):
            return operation  # no conversion necessary
        else:
            return FullTPOp(operation.to_dense(), operation.evotype, operation.state_space)
            # above will raise ValueError if conversion cannot be done

    elif to_type == "linear":
        if isinstance(operation, LinearlyParamArbitraryOp):
            return operation  # no conversion necessary
        elif isinstance(operation, StaticArbitraryOp):
            real = _np.isclose(_np.linalg.norm(operation.imag), 0)
            return LinearlyParamArbitraryOp(operation.to_dense(), _np.array([]), {}, real,
                                            operation.evotype, operation.state_space)
        else:
            raise ValueError("Cannot convert type %s to LinearlyParamArbitraryOp"
                             % type(operation))

    elif to_type == "static":
        if isinstance(operation, StaticArbitraryOp):
            return operation  # no conversion necessary
        else:
            return StaticArbitraryOp(operation.to_dense(), operation.evotype, operation.state_space)

    elif to_type == "static unitary":
        op_std = _bt.change_basis(operation, basis, 'std')
        unitary = _ot.process_mx_to_unitary(op_std)
        return StaticUnitaryOp(unitary, basis, operation.evotype, operation.state_space)

    elif _ot.is_valid_lindblad_paramtype(to_type):
        # e.g. "H+S terms","H+S clifford terms"

        #REMOVE
        #_, evotype = _ot.split_lindblad_paramtype(to_type)
        #LindbladOpType = LindbladOp
        #if evotype in ("svterm", "cterm") else \
        #    LindbladDenseOp

        unitary_postfactor = None
        if isinstance(operation, (FullArbitraryOp, FullTPOp, StaticArbitraryOp)):
            from pygsti.tools import jamiolkowski as _jt
            RANK_TOL = 1e-6
            J = _jt.fast_jamiolkowski_iso_std(operation.to_dense(), op_mx_basis=basis)  # Choi mx basis doesn't matter
            if _np.linalg.matrix_rank(J, RANK_TOL) == 1:  # when 'operation' is unitary, separate it
                unitary_op = _ot.process_mx_to_unitary(_bt.change_basis(operation.to_dense(), basis, 'std'))
                unitary_postfactor = StaticUnitaryOp(unitary_op, basis, operation.evotype, operation.state_space)

        proj_basis = 'pp' if operation.state_space.is_entirely_qubits else basis
        if unitary_postfactor is not None:
            errorgen = LindbladErrorgen.from_error_generator(operation.state_space.dim, to_type, proj_basis,
                                                             basis, truncate=True, evotype=operation.evotype,
                                                             state_space=operation.state_space)
            ret = ComposedOp([unitary_postfactor, ExpErrorgenOp(errorgen)])
        else:
            errorgen = LindbladErrorgen.from_operation_matrix(operation.to_dense(), to_type, proj_basis,
                                                              mx_basis=basis, truncate=True, evotype=operation.evotype,
                                                              state_space=operation.state_space)
            ret = ExpErrorgenOp(errorgen)

        if ret.dim <= 16:  # only do this for up to 2Q operations, otherwise to_dense is too expensive
            assert(_np.linalg.norm(operation.to_dense() - ret.to_dense()) < 1e-6), \
                "Failure to create CPTP operation (maybe due the complex log's branch cut?)"
        return ret

    elif to_type == "static clifford":
        if isinstance(operation, StaticCliffordOp):
            return operation  # no conversion necessary

        # assume operation represents a unitary op (otherwise
        #  would need to change Model dim, which isn't allowed)
        return StaticCliffordOp(operation)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


def check_deriv_wrt_params(operation, deriv_to_check=None, wrt_filter=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a LinearOperator object.

    This routine is meant to be used as an aid in testing and debugging
    operation classes by comparing the finite-difference Jacobian that
    *should* be returned by `operation.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    operation : LinearOperator
        The operation object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `operation.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a LinearOperator
        class's `deriv_wrt_params()` method itself as a part of testing.

    wrt_filter : list or numpy.ndarray
        List of parameter indices to filter the result by (as though
        derivative is only taken with respect to these parameters).

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    None
    """
    fd_deriv = finite_difference_deriv_wrt_params(operation, wrt_filter, eps)
    if deriv_to_check is None:
        deriv_to_check = operation.deriv_wrt_params()

    #print("Deriv shapes = %s and %s" % (str(fd_deriv.shape),
    #                                    str(deriv_to_check.shape)))
    #print("finite difference deriv = \n",fd_deriv)
    #print("deriv_wrt_params deriv = \n",deriv_to_check)
    #print("deriv_wrt_params - finite diff deriv = \n",
    #      deriv_to_check - fd_deriv)
    for i in range(deriv_to_check.shape[0]):
        for j in range(deriv_to_check.shape[1]):
            diff = abs(deriv_to_check[i, j] - fd_deriv[i, j])
            if diff > 10 * eps:
                print("deriv_chk_mismatch: (%d,%d): %g (comp) - %g (fd) = %g" %
                      (i, j, deriv_to_check[i, j], fd_deriv[i, j], diff))  # pragma: no cover

    if _np.linalg.norm(fd_deriv - deriv_to_check) / fd_deriv.size > 10 * eps:
        raise ValueError("Failed check of deriv_wrt_params:\n"
                         " norm diff = %g" %
                         _np.linalg.norm(fd_deriv - deriv_to_check))  # pragma: no cover


def optimize_operation(op_to_optimize, target_op):
    """
    Optimize the parameters of `op_to_optimize`.

    Optimization is performed so that the the resulting operation matrix
    is as close as possible to target_op's matrix.

    This is trivial for the case of FullArbitraryOp
    instances, but for other types of parameterization
    this involves an iterative optimization over all the
    parameters of op_to_optimize.

    Parameters
    ----------
    op_to_optimize : LinearOperator
        The operation to optimize.  This object gets altered.

    target_op : LinearOperator
        The operation whose matrix is used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(op_to_optimize, StaticArbitraryOp):
        return  # nothing to optimize

    if isinstance(op_to_optimize, FullArbitraryOp):
        if(target_op.dim != op_to_optimize.dim):  # special case: operations can have different overall dimension
            op_to_optimize.dim = target_op.dim  # this is a HACK to allow model selection code to work correctly
        op_to_optimize.set_dense(target_op.to_dense())  # just copy entire overall matrix since fully parameterized
        return

    from pygsti import optimize as _opt
    from pygsti.tools import matrixtools as _mt
    assert(target_op.dim == op_to_optimize.dim)  # operations must have the same overall dimension
    targetMatrix = target_op.to_dense() if isinstance(target_op, LinearOperator) else target_op

    def _objective_func(param_vec):
        op_to_optimize.from_vector(param_vec)
        return _mt.frobeniusnorm(op_to_optimize.to_dense() - targetMatrix)

    x0 = op_to_optimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    op_to_optimize.from_vector(minSol.x)
    #print("DEBUG: optimized operation to min frobenius distance %g" %
    #      _mt.frobeniusnorm(op_to_optimize-targetMatrix))
