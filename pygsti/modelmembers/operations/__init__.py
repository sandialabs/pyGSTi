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

from .cliffordop import CliffordOp
from .composederrorgen import ComposedErrorgen
from .composedop import ComposedOp, ComposedDenseOp
from .denseop import DenseOperator, DenseOperatorInterface
from .depolarizeop import DepolarizeOp
from .eigpdenseop import EigenvalueParamDenseOp
from .embeddederrorgen import EmbeddedErrorgen
from .embeddedop import EmbeddedOp, EmbeddedDenseOp
from .experrorgenop import ExpErrorgenOp, ExpErrorgenDenseOp
from .fulldenseop import FullDenseOp
from .fullunitaryop import FullUnitaryOp
from .lindbladerrorgen import LindbladErrorgen
from .linearop import LinearOperator
from .lpdenseop import LinearlyParamDenseOp
from .staticdenseop import StaticDenseOp
from .staticstdop import StaticStandardOp
from .staticunitaryop import StaticUnitaryOp
from .stochasticop import StochasticNoiseOp
from .tpdenseop import TPDenseOp

from .opfactory import OpFactory

from .linearop import finite_difference_deriv_wrt_params, finite_difference_hessian_wrt_params


import numpy as _np
from ...tools import basistools as _bt
from ...tools import optools as _ot


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
        if isinstance(operation, FullDenseOp):
            return operation  # no conversion necessary
        else:
            return FullDenseOp(operation.to_dense(), operation.evotype, operation.state_space)

    elif to_type == "TP":
        if isinstance(operation, TPDenseOp):
            return operation  # no conversion necessary
        else:
            return TPDenseOp(operation.to_dense(), operation.evotype, operation.state_space)
            # above will raise ValueError if conversion cannot be done

    elif to_type == "linear":
        if isinstance(operation, LinearlyParamDenseOp):
            return operation  # no conversion necessary
        elif isinstance(operation, StaticDenseOp):
            real = _np.isclose(_np.linalg.norm(operation.imag), 0)
            return LinearlyParamDenseOp(operation.to_dense(), _np.array([]), {}, real,
                                        operation.evotype, operation.state_space)
        else:
            raise ValueError("Cannot convert type %s to LinearlyParamDenseOp"
                             % type(operation))

    elif to_type == "static":
        if isinstance(operation, StaticDenseOp):
            return operation  # no conversion necessary
        else:
            return StaticDenseOp(operation.to_dense(), operation.evotype, operation.state_space)

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
        if isinstance(operation, (FullDenseOp, TPDenseOp, StaticDenseOp)):
            from ...tools import jamiolkowski as _jt
            RANK_TOL = 1e-6
            J = _jt.fast_jamiolkowski_iso_std(operation.to_dense(), op_mx_basis=basis)  # Choi mx basis doesn't matter
            if _np.linalg.matrix_rank(J, RANK_TOL) == 1:  # when 'operation' is unitary, separate it
                unitary_op = _ot.process_mx_to_unitary(_bt.change_basis(operation.to_dense(), basis, 'std'))
                unitary_postfactor = StaticUnitaryOp(unitary_op, basis, operation.evotype, operation.state_space)

        proj_basis = 'pp' if operation.state_space.is_entirely_qubits else basis
        nonham_mode, param_mode, use_ham_basis, use_nonham_basis = \
            LindbladErrorgen.decomp_paramtype(to_type)
        ham_basis = proj_basis if use_ham_basis else None
        nonham_basis = proj_basis if use_nonham_basis else None

        if unitary_postfactor is not None:
            errorgen = LindbladErrorgen.from_error_generator(_np.zeros((operation.state_space.dim,
                                                                        operation.state_space.dim), 'd'),
                                                             ham_basis, nonham_basis, param_mode, nonham_mode,
                                                             basis, truncate=True, evotype=operation.evotype)
            ret = ComposedOp([unitary_postfactor, ExpErrorgenOp(errorgen)])
        else:
            errorgen = LindbladErrorgen.from_operation_matrix(operation.to_dense(), ham_basis, nonham_basis,
                                                              param_mode, nonham_mode, truncate=True, mx_basis=basis,
                                                              evotype=operation.evotype)
            ret = ExpErrorgenOp(errorgen)

        if ret.dim <= 16:  # only do this for up to 2Q operations, otherwise to_dense is too expensive
            assert(_np.linalg.norm(operation.to_dense() - ret.to_dense()) < 1e-6), \
                "Failure to create CPTP operation (maybe due the complex log's branch cut?)"
        return ret

    elif to_type == "static clifford":
        if isinstance(operation, CliffordOp):
            return operation  # no conversion necessary

        # assume operation represents a unitary op (otherwise
        #  would need to change Model dim, which isn't allowed)
        return CliffordOp(operation)

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

    This is trivial for the case of FullDenseOp
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
    if isinstance(op_to_optimize, StaticDenseOp):
        return  # nothing to optimize

    if isinstance(op_to_optimize, FullDenseOp):
        if(target_op.dim != op_to_optimize.dim):  # special case: operations can have different overall dimension
            op_to_optimize.dim = target_op.dim  # this is a HACK to allow model selection code to work correctly
        op_to_optimize.set_dense(target_op.to_dense())  # just copy entire overall matrix since fully parameterized
        return

    from ... import optimize as _opt
    from ...tools import matrixtools as _mt
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
