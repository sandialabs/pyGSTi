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
import warnings as _warnings

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
from .fullcptpop import FullCPTPOp
from .lindbladerrorgen import LindbladErrorgen, LindbladParameterization
from .linearop import LinearOperator
from .linearop import finite_difference_deriv_wrt_params, finite_difference_hessian_wrt_params
from .lpdenseop import LinearlyParamArbitraryOp
from .opfactory import OpFactory, EmbeddedOpFactory, EmbeddingOpFactory, ComposedOpFactory
from .repeatedop import RepeatedOp
from .staticarbitraryop import StaticArbitraryOp
from .staticcliffordop import StaticCliffordOp
from .staticstdop import StaticStandardOp
from .staticunitaryop import StaticUnitaryOp
from .stochasticop import StochasticNoiseOp
from .lindbladcoefficients import LindbladCoefficientBlock as _LindbladCoefficientBlock
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot


def create_from_unitary_mx(unitary_mx, op_type, basis='pp', stdname=None, evotype='default', state_space=None):
    """ TODO: docstring - note that op_type can be a list/tuple of types in order of precedence """
    op_type_preferences = verbose_type_from_op_type(op_type)
    error_msgs = {}

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
            elif typ in ('static', 'full', 'full TP', 'linear', 'full CPTP'):
                superop_mx = _ot.unitary_to_superop(U, basis)
                op = create_from_superop_mx(superop_mx, op_type, basis, stdname, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):  # maybe "lindblad XXX" where XXX is a valid lindblad type?
                if _np.allclose(U, _np.identity(U.shape[0], 'd')):
                    unitary_postfactor = None
                else:
                    unitary_postfactor = create_from_unitary_mx(
                        U, ('static standard', 'static clifford', 'static unitary'),
                        basis, stdname, evotype, state_space)

                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = LindbladErrorgen.from_error_generator(state_space.dim, typ, proj_basis, basis,
                                                                 truncate=True, evotype=evotype,
                                                                 state_space=state_space)

                op = ExpErrorgenOp(errorgen) if (unitary_postfactor is None) \
                    else ComposedOp([unitary_postfactor, ExpErrorgenOp(errorgen)])

                if op.dim <= 16:  # only do this for up to 2Q operations, otherwise to_dense is too expensive
                    expected_superop_mx = _ot.unitary_to_superop(U, basis)
                    assert (_np.linalg.norm(op.to_dense('HilbertSchmidt') - expected_superop_mx) < 1e-6), \
                        "Failure to create Lindblad operation (maybe due the complex log's branch cut?)"
            else:
                raise ValueError("Unknown operation type '%s'!" % str(typ))

            return op  # if we get to here, then we've successfully created an op to return
        except (ValueError, AssertionError, AttributeError) as e:
            #_warnings.warn('Failed to create operator with type %s with error: %s' % (typ, e))
            error_msgs[typ] = str(e)  # # move on to text type

    raise ValueError("Could not create an operator of type(s) %s from the given unitary op:\n%s"
                     % (str(op_type), str(error_msgs)))


def create_from_superop_mx(superop_mx, op_type, basis='pp', stdname=None, evotype='default', state_space=None):
    op_type_preferences = (op_type,) if isinstance(op_type, str) else op_type
    error_msgs = {}

    for typ in op_type_preferences:
        try:
            if typ == "static":  # "static arbitrary"?
                op = StaticArbitraryOp(superop_mx, basis, evotype, state_space)
            elif typ == "full":  # "full arbitrary"?
                op = FullArbitraryOp(superop_mx, basis, evotype, state_space)
            elif typ in ["TP", "full TP"]:
                op = FullTPOp(superop_mx, basis, evotype, state_space)
            elif typ == "full CPTP":
                op = FullCPTPOp.from_superop_matrix(superop_mx, basis, evotype, state_space)
            elif typ == "linear":  # "linear arbitrary"?
                real = _np.isclose(_np.linalg.norm(superop_mx.imag), 0)
                op = LinearlyParamArbitraryOp(superop_mx, _np.array([]), {}, None, None, real, basis,
                                              evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):  # maybe "lindblad XXX" where XXX is a valid lindblad type?
                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                if _ot.superop_is_unitary(superop_mx, basis):
                    unitary_postfactor = StaticUnitaryOp(_ot.superop_to_unitary(superop_mx, basis, False),
                                                         basis, evotype, state_space)
                    errorgen = LindbladErrorgen.from_error_generator(state_space.dim, typ, proj_basis,
                                                                     basis, truncate=True, evotype=evotype,
                                                                     state_space=state_space)
                    ret = ComposedOp([unitary_postfactor, ExpErrorgenOp(errorgen)])
                else:
                    errorgen = LindbladErrorgen.from_operation_matrix(superop_mx, typ,
                                                                      proj_basis, mx_basis=basis, truncate=True,
                                                                      evotype=evotype, state_space=state_space)
                    ret = ExpErrorgenOp(errorgen)

                if ret.dim <= 16:  # only do this for up to 2Q operations, otherwise to_dense is too expensive
                    assert(_np.linalg.norm(superop_mx - ret.to_dense('HilbertSchmidt'))
                           < 1e-6), "Failure to create CPTP operation (maybe due the complex log's branch cut?)"
                return ret

            else:
                #Anything else we try to convert to a unitary and convert the unitary
                unitary_mx = _ot.superop_to_unitary(superop_mx, basis)  # raises ValueError if superop isn't unitary
                op = create_from_unitary_mx(unitary_mx, typ, basis, stdname, evotype, state_space)

            return op  # if we get to here, then we've successfully created an op to return
        except (ValueError, AssertionError) as e:
            error_msgs[typ] = str(e)  # # move on to text type

    raise ValueError("Could not create an operator of type(s) %s from the given superop:\n%s"
                     % (str(op_type), str(error_msgs)))


def verbose_type_from_op_type(op_type):
    """Decode an op type into the "canonical", more verbose op type.

    Parameters:
    -----------
    op_type: str or list of str
        Operation parameterization type (or list of preferences)

    Returns
    -------
    povm_type_preferences: tuple of str
        POVM parameterization types
    """
    op_type_preferences = (op_type,) if isinstance(op_type, str) else op_type

    verbose_conversion = {
        'static standard': 'static standard',
        'static clifford': 'static clifford',
        'static unitary': 'static unitary',
        'full unitary': 'full unitary',
        'static': 'static',
        'full': 'full',
        'full TP': 'full TP',
        'full CPTP': 'full CPTP',
        'TP': 'full TP',
        'linear': 'linear',
    }

    verbose_type_preferences = []
    for typ in op_type_preferences:
        verbose_type = None
        if _ot.is_valid_lindblad_paramtype(typ):
            # TODO: DO we want to prepend with lindblad?
            verbose_type = typ
        else:
            verbose_type = verbose_conversion.get(typ, None)

        if verbose_type is None:
            continue

        if verbose_type not in verbose_type_preferences:
            verbose_type_preferences.append(verbose_type)

    if len(verbose_type_preferences) == 0:
        raise ValueError(
            'Could not convert any op types from {}.\n'.format(op_type_preferences)
            + '\tKnown op_types: Lindblad types or {}\n'.format(sorted(list(verbose_conversion.keys())))
            + '\tValid povm_types: Lindblad types or {}'.format(sorted(list(set(verbose_conversion.values()))))
        )

    return tuple(verbose_type_preferences)


def convert_errorgen(errorgen, to_type, basis, flatten_structure=False):
    """ TODO: docstring """
    to_types = verbose_type_from_op_type(to_type)
    error_msgs = {}

    for to_type in to_types:
        try:
            if not flatten_structure and isinstance(errorgen, (ComposedErrorgen, EmbeddedErrorgen)):
                def convert_errorgen_structure(eg):
                    if isinstance(eg, ComposedErrorgen):
                        return ComposedErrorgen([convert_errorgen_structure(f) for f in eg.factors],
                                                eg.evotype, eg.state_space)
                    elif isinstance(eg, EmbeddedErrorgen):
                        return EmbeddedErrorgen(eg.state_space, eg.target_labels,
                                                convert_errorgen_structure(eg.embedded_op))
                    else:
                        return convert_errorgen(eg)
                return convert_errorgen_structure(errorgen)

            elif isinstance(errorgen, LindbladErrorgen) and _ot.is_valid_lindblad_paramtype(to_type):
                # Convert the *parameterizations* of block only -- just use to_type as a lookup for
                # how to convert any blocks of each type (don't require all objects being converted have
                # all & exactly the Lindbald coefficient block types in `to_type`)
                to_type = LindbladParameterization.cast(to_type)
                block_param_conversion = {blk_type: param_mode for blk_type, param_mode
                                          in zip(to_type.block_types, to_type.param_modes)}
                converted_blocks = [blk.convert(block_param_conversion.get(blk._block_type, blk._param_mode))
                                    for blk in errorgen.coefficient_blocks]
                return LindbladErrorgen(converted_blocks, 'auto', errorgen.matrix_basis,
                                        errorgen.evotype, errorgen.state_space)
            else:
                raise ValueError("%s is not a valid error generator type!" % str(to_type))

        except Exception as e:
            error_msgs[to_type] = str(e)  # try next to_type

    raise ValueError("Could not convert error generator to type(s): %s\n%s" % (str(to_types), str(error_msgs)))


def convert(operation, to_type, basis, ideal_operation=None, flatten_structure=False):
    """
    Convert operation to a new type of parameterization.

    This potentially creates a new LinearOperator object, and
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    operation : LinearOperator
        LinearOperator to convert

    to_type : {"full","full TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `operation`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    ideal_operation : LinearOperator or "identity", optional
        The ideal (usually unitary) version of `operation`,
        potentially used when converting to an error-generator type.
        The special value `"identity"` can be used to indicate that the
        ideal operation is the identity.

    flatten_structure : bool, optional
        When `False`, the sub-members of composed and embedded operations
        are separately converted, leaving the original operation's structure
        unchanged.  When `True`, composed and embedded operations are "flattened"
        into a single operation of the requested `to_type`.

    Returns
    -------
    LinearOperator
        The converted operation, usually a distinct
        object from the operation object passed as input.
    """
    to_types = verbose_type_from_op_type(to_type)
    error_msgs = {}

    destination_types = {'full': FullArbitraryOp,
                         'full TP': FullTPOp,
                         'full CPTP': FullCPTPOp,
                         'linear': LinearlyParamArbitraryOp,
                         'static': StaticArbitraryOp,
                         'static unitary': StaticUnitaryOp,
                         'static clifford': StaticCliffordOp}
    NoneType = type(None)

    for to_type in to_types:
        is_errorgen_type = _ot.is_valid_lindblad_paramtype(to_type)  # whether to_type specifies a type of errorgen
        try:
            if isinstance(operation, destination_types.get(to_type, NoneType)):
                return operation

            if not flatten_structure and isinstance(operation, (ComposedOp, EmbeddedOp)):
                def convert_structure(op):
                    if isinstance(op, ComposedOp):
                        return ComposedOp([convert_structure(f) for f in op.factorops], op.evotype, op.state_space)
                    elif isinstance(op, EmbeddedOp):
                        return EmbeddedOp(op.state_space, op.target_labels, convert_structure(op.embedded_op))
                    else:
                        return convert(op, to_type, basis, None, flatten_structure)
                return convert_structure(operation)

            elif isinstance(operation, ExpErrorgenOp) and is_errorgen_type:
                # Just an error generator type conversion
                return ExpErrorgenOp(convert_errorgen(operation.errorgen, to_type, basis, flatten_structure))

            elif (_ot.is_valid_lindblad_paramtype(to_type)
                  and (ideal_operation is not None or operation.num_params == 0)):  # e.g. TP -> Lindblad
                # Above: consider "isinstance(operation, StaticUnitaryOp)" instead of num_params == 0?
                #Convert a non-exp(errorgen) op to  exp(errorgen) * ideal
                proj_basis = 'PP' if operation.state_space.is_entirely_qubits else basis
                if ideal_operation == "identity":  # special value
                    postfactor_op = None
                    error_map_mx = operation.to_dense('HilbertSchmidt')  # error generators are only in HS space
                else:
                    postfactor_op = ideal_operation if (ideal_operation is not None) else operation
                    error_map_mx = _np.dot(operation.to_dense('HilbertSchmidt'),
                                           _np.linalg.inv(postfactor_op.to_dense('HilbertSchmidt')))

                errorgen = LindbladErrorgen.from_operation_matrix(error_map_mx, to_type,
                                                                  proj_basis, mx_basis=basis, truncate=True,
                                                                  evotype=operation.evotype,
                                                                  state_space=operation.state_space)
                ret = ComposedOp([postfactor_op.copy(), ExpErrorgenOp(errorgen)]) \
                    if (postfactor_op is not None) else ExpErrorgenOp(errorgen)

                if ret.dim <= 16:  # only do this for up to 2Q operations, otherwise to_dense is too expensive
                    assert(_np.linalg.norm(operation.to_dense('HilbertSchmidt') - ret.to_dense('HilbertSchmidt'))
                           < 1e-6), "Failure to create CPTP operation (maybe due the complex log's branch cut?)"
                return ret

            else:
                min_space = operation.evotype.minimal_space
                mx = operation.to_dense(min_space)
                if min_space == 'Hilbert':
                    return create_from_unitary_mx(mx, to_type, basis, None, operation.evotype, operation.state_space)
                else:
                    return create_from_superop_mx(mx, to_type, basis, None, operation.evotype, operation.state_space)

        except Exception as e:
            error_msgs[to_type] = str(e)  # try next to_type

    raise ValueError("Could not convert operation to to type(s): %s\n%s" % (str(to_types), str(error_msgs)))


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
