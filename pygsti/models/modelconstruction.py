"""
Functions for the construction of new models.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools
from os import stat
from pygsti.modelmembers.instruments.instrument import Instrument

import numpy as _np
import scipy as _scipy
import scipy.linalg as _spl

from pygsti.evotypes import Evotype as _Evotype
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state
from pygsti.modelmembers import instruments as _instrument
from pygsti.modelmembers.operations import opfactory as _opfactory
from pygsti.models import stencillabel as _stencil
from pygsti.models.modelnoise import OpModelNoise as _OpModelNoise
from pygsti.models.modelnoise import OpModelPerOpNoise as _OpModelPerOpNoise
from pygsti.models.modelnoise import ComposedOpModelNoise as _ComposedOpModelNoise
from pygsti.models.modelnoise import LindbladNoise as _LindbladNoise
from pygsti.models.modelnoise import StochasticNoise as _StochasticNoise
from pygsti.models.modelnoise import DepolarizationNoise as _DepolarizationNoise
from pygsti.models import explicitmodel as _emdl
from pygsti.models import gaugegroup as _gg
from pygsti.models.localnoisemodel import LocalNoiseModel as _LocalNoiseModel
from pygsti.models.cloudnoisemodel import CloudNoiseModel as _CloudNoiseModel
from pygsti.baseobjs import label as _label
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.basis import ExplicitBasis as _ExplicitBasis
from pygsti.baseobjs.basis import DirectSumBasis as _DirectSumBasis
from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph
from pygsti.tools import basistools as _bt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import optools as _ot
from pygsti.tools import listtools as _lt
from pygsti.baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.tools.legacytools import deprecate as _deprecated_fn


#############################################
# Build gates based on "standard" gate names
############################################
def _basis_create_spam_vector(vec_expr, basis):
    """
    Build a rho or E vector from an expression.

    Parameters
    ----------
    vec_expr : string
        the expression which determines which vector to build.  Currenlty, only
        integers are allowed, which specify a the vector for the pure state of
        that index.  For example, "1" means return vectorize(``|1><1|``).  The
        index labels the absolute index of the state within the entire state
        space, and is independent of the direct-sum decomposition of density
        matrix space.

    basis : Basis object
        The basis of the returned vector.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        The vector specified by vec_expr in the desired basis.
    """
    #So far just allow integer prep_expressions that give the index of state (within the state space) that we
    #prep/measure
    try:
        index = int(vec_expr)
    except:
        raise ValueError("Expression must be the index of a state (as a string)")

    #standard basis that has the same direct-sum structure as `basis`:
    std_basis = basis.create_equivalent('std')
    vecInSimpleStdBasis = _np.zeros(std_basis.elshape, 'd')  # a matrix, but flattened it is our spamvec
    vecInSimpleStdBasis[index, index] = 1.0  # now a matrix with just a single 1 on the diag
    vecInReducedStdBasis = _np.dot(std_basis.from_elementstd_transform_matrix, vecInSimpleStdBasis.flatten())
    # translates the density matrix / state vector to the std basis with our desired block structure

    vec = _bt.change_basis(vecInReducedStdBasis, std_basis, basis)
    return vec.reshape(-1, 1)


@_deprecated_fn('_basis_create_spam_vector(...)')
def _create_spam_vector(state_space_dims, state_space_labels, vec_expr, basis="gm"):
    """
    DEPRECATED: use :func:`_basis_create_spam_vector` instead.
    """
    return _basis_create_spam_vector(vec_expr, _Basis.cast(basis, state_space_dims))


def _basis_create_identity_vec(basis):
    """
    Build a the identity vector for a given space and basis.

    Parameters
    ----------
    basis : Basis object
        The basis of the returned vector.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        The identity vector in the desired basis.
    """
    opDim = basis.dim
    if isinstance(basis, _DirectSumBasis):
        blockDims = [c.dim for c in basis.component_bases]
    else: blockDims = [opDim]

    # assume index given as vec_expr refers to a Hilbert-space state index, so "reduced-std" basis
    vecInReducedStdBasis = _np.zeros((opDim, 1), 'd')

    #set all diagonal elements of density matrix to 1.0 (end result = identity density mx)
    start = 0; vecIndex = 0
    for blockVecDim in blockDims:
        blockDim = int(_np.sqrt(blockVecDim))  # vec -> matrix dim
        for i in range(start, start + blockDim):
            for j in range(start, start + blockDim):
                if i == j: vecInReducedStdBasis[vecIndex, 0] = 1.0  # set diagonal element of density matrix
                vecIndex += 1
        start += blockDim
    return _bt.change_basis(vecInReducedStdBasis, "std", basis)


def _create_identity_vec(state_space_dims, basis="gm"):
    """
    Build the identity vector given a certain density matrix struture.

    Parameters
    ----------
    state_space_dims : list
        A list of integers specifying the dimension of each block
        of a block-diagonal the density matrix.

    basis : str, optional
        The string abbreviation of the basis of the returned vector.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt).

    Returns
    -------
    numpy array
    """
    return _basis_create_identity_vec(_Basis.cast(basis, state_space_dims))


def _basis_create_operation(state_space, op_expr, basis="gm", parameterization="full", evotype='default'):
    """
    Build an operation object from an expression.

    Parameters
    ----------
    state_space : StateSpace
        The state space that the created operation should act upon.

    op_expr : string
        expression for the gate to build.  String is first split into parts
        delimited by the colon (:) character, which are composed together to
        create the final gate.  Each part takes on of the allowed forms:

        - I(ssl_0, ...) = identity operation on one or more state space labels
          (ssl_i)
        - X(theta, ssl) = x-rotation by theta radians of qubit labeled by ssl
        - Y(theta, ssl) = y-rotation by theta radians of qubit labeled by ssl
        - Z(theta, ssl) = z-rotation by theta radians of qubit labeled by ssl
        - CX(theta, ssl0, ssl1) = controlled x-rotation by theta radians.  Acts
          on qubit labeled by ssl1 with ssl0 being the control.
        - CY(theta, ssl0, ssl1) = controlled y-rotation by theta radians.  Acts
          on qubit labeled by ssl1 with ssl0 being the control.
        - CZ(theta, ssl0, ssl1) = controlled z-rotation by theta radians.  Acts
          on qubit labeled by ssl1 with ssl0 being the control.
        - CNOT(ssl0, ssl1) = standard controlled-not gate.  Acts on qubit
          labeled by ssl1 with ssl0 being the control.
        - CPHASE(ssl0, ssl1) = standard controlled-phase gate.  Acts on qubit
          labeled by ssl1 with ssl0 being the control.
        - LX(theta, i0, i1) = leakage between states i0 and i1.  Implemented as
          an x-rotation between states with integer indices i0 and i1 followed
          by complete decoherence between the states.

    basis : Basis object
        The basis the returned operation should be represented in.

    parameterization : {"full","TP","static"}, optional
        How to parameterize the resulting gate.

        - "full" = return a FullArbitraryOp.
        - "TP" = return a FullTPOp.
        - "static" = return a StaticArbitraryOp.

    evotype : Evotype or str, optional
        The evolution type of this operation, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    Returns
    -------
    LinearOperator
        A gate object representing the gate given by op_expr in the desired
        basis.
    """
    # op_expr can contain single qubit ops: X(theta) ,Y(theta) ,Z(theta)
    #                      two qubit ops: CNOT
    #                      clevel qubit ops: Leak
    #                      two clevel opts: Flip
    #  each of which is given additional parameters specifying which indices it acts upon

    #Working with a StateSpaceLabels object gives us access to all the info we'll need later
    state_space = _statespace.StateSpace.cast(state_space)
    if isinstance(basis, str):
        basis = _Basis.cast(basis, state_space)
    assert(state_space.dim == basis.dim), \
        "State space labels dim (%s) != basis dim (%s)" % (state_space.dim, basis.dim)

    # ------------------------------------------------------------------------------------------------------------------
    # -- Helper Functions ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def to_label(lbl):
        """ Convert integer-strings to integers in state space label """
        try: return int(lbl)
        except: return lbl.strip()

    def to_labels(lbls):
        """ Convert integer-strings to integers in state space labels """
        return [to_label(lbl) for lbl in lbls]

    # ------------------------------------------------------------------------------------------------------------------
    # -- End Helper Functions ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    #FUTURE?: type_preferences = ('static standard', 'static clifford', 'static unitary')
    build_evotype = 'default'
    superop_mxs_in_basis = []
    exprTerms = op_expr.split(':')
    for exprTerm in exprTerms:

        l = exprTerm.index('('); r = exprTerm.rindex(')')
        opName = exprTerm[0:l]
        argsStr = exprTerm[l + 1:r]
        args = argsStr.split(',')

        if opName == "I":
            # qubit labels (TODO: what about 'L' labels? -- not sure if they work with this...)
            labels = to_labels(args)
            stateSpaceUDim = int(_np.product([state_space.label_udimension(l) for l in labels]))
            # a complex 2x2 mx unitary for the identity in Pauli-product basis
            Uop = _op.StaticUnitaryOp(_np.identity(stateSpaceUDim, 'complex'), 'pp', build_evotype)

            #FUTURE?:
            # stdname = 'Gi' if (stateSpaceUDim == 2) else None
            # Uop = _op.create_from_unitary_mx(_np.identity(stateSpaceUDim, complex), type_preferences, 'pp',
            #                                  stdname=stdname, evotype=evotype)

            # a complex 2*num_qubits x 2*num_qubits mx unitary on full space in Pauli-product basis
            Uop_embed = _op.EmbeddedOp(state_space, labels, Uop)
            # a real 4*num_qubits x 4*num_qubits mx superoperator in final basis
            superop_mx_pp = Uop_embed.to_dense(on_space='HilbertSchmidt')
            # a real 4*num_qubits x 4*num_qubits mx superoperator in final basis
            superop_mx_in_basis = _bt.change_basis(superop_mx_pp, 'pp', basis)

        elif opName == "D":
            # like 'I', but only parameterize the diagonal elements - so can be a depolarization-type map
            raise NotImplementedError("Removed temporarily - need to update using embedded gates")
            # # qubit labels (TODO: what about 'L' labels? -- not sure if they work with this...)
            # labels = to_labels(args)
            # stateSpaceDim = sslbls.product_dim(labels)

            # if parameterization not in ("linear","linearTP"):
            #     raise ValueError("'D' gate only makes sense to use when and parameterization == 'linear'")

            # if defaultI2P == "TP":
            #     # parameterize only the diagonals els after the first
            #     indicesToParameterize = [ (i,i) for i in range(1,stateSpaceDim**2) ]
            # else:
            #     # parameterize only the diagonals els
            #     indicesToParameterize = [ (i,i) for i in range(0,stateSpaceDim**2) ]
            # # *real* 4x4 mx in Pauli-product basis -- still just the identity!
            # pp_opMx = _np.identity(stateSpaceDim**2, 'd')
            # # pp_opMx assumed to be in the Pauli-product basis
            # opTermInFinalBasis = embed_operation(pp_opMx, tuple(labels), indicesToParameterize)

        elif opName in ('X', 'Y', 'Z'):  # single-qubit gate names
            assert(len(args) == 2)  # theta, qubit-index
            theta = eval(args[0], {"__builtins__": None}, {'pi': _np.pi})
            label = to_label(args[1])
            assert(state_space.label_dimension(label) == 4), "%s gate must act on qubits!" % opName

            if opName == 'X': ex = -1j * theta * sigmax / 2
            elif opName == 'Y': ex = -1j * theta * sigmay / 2
            elif opName == 'Z': ex = -1j * theta * sigmaz / 2

            # complex 2x2 unitary matrix operating on single qubit in Pauli-product basis
            Uop = _op.StaticUnitaryOp(_spl.expm(ex), 'pp', build_evotype)

            #FUTURE?:
            #stdname = None
            #if _np.isclose(theta, _np.pi): stdname = 'G%spi' % opName.lower()
            #elif _np.isclose(theta, _np.pi/2): stdname = 'G%spi2' % opName.lower()
            # Uop = _op.create_from_unitary_mx(_spl.expm(ex), type_preferences, 'pp', stdname=stdname, evotype=evotype)

            # a complex 2*num_qubits x 2*num_qubits mx unitary on full space in Pauli-product basis
            Uop_embed = _op.EmbeddedOp(state_space, (label,), Uop)
            # a real 4*num_qubits x 4*num_qubits mx superoperator in Pauli-product basis
            superop_mx_pp = Uop_embed.to_dense(on_space='HilbertSchmidt')
            # a real 4*num_qubits x 4*num_qubits mx superoperator in final basis
            superop_mx_in_basis = _bt.change_basis(superop_mx_pp, 'pp', basis)

        elif opName == 'N':  # more general single-qubit gate
            assert(len(args) == 5)  # theta, sigmaX-coeff, sigmaY-coeff, sigmaZ-coeff, qubit-index
            theta = eval(args[0], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            sxCoeff = eval(args[1], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            syCoeff = eval(args[2], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            szCoeff = eval(args[3], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            label = to_label(args[4])
            assert(state_space.label_dimension(label) == 4), "%s gate must act on qubits!" % opName

            ex = -1j * theta * (sxCoeff * sigmax / 2. + syCoeff * sigmay / 2. + szCoeff * sigmaz / 2.)
            # complex 2x2 unitary matrix operating on single qubit in Pauli-product basis
            Uop = _op.StaticUnitaryOp(_spl.expm(ex), 'pp', evotype=build_evotype)
            #FUTURE?: Uop = _op.create_from_unitary_mx(_spl.expm(ex), type_preferences, 'pp', evotype=evotype)
            # a complex 2*num_qubits x 2*num_qubits mx unitary on full space in Pauli-product basis
            Uop_embed = _op.EmbeddedOp(state_space, (label,), Uop)
            # a real 4*num_qubits x 4*num_qubits mx superoperator in Pauli-product basis
            superop_mx_pp = Uop_embed.to_dense(on_space='HilbertSchmidt')
            # a real 4*num_qubits x 4*num_qubits mx superoperator in final basis
            superop_mx_in_basis = _bt.change_basis(superop_mx_pp, 'pp', basis)

        elif opName in ('CX', 'CY', 'CZ', 'CNOT', 'CPHASE'):  # two-qubit gate names

            if opName in ('CX', 'CY', 'CZ'):
                assert(len(args) == 3)  # theta, qubit-label1, qubit-label2
                theta = eval(args[0], {"__builtins__": None}, {'pi': _np.pi})
                label1 = to_label(args[1]); label2 = to_label(args[2])

                if opName == 'CX': ex = -1j * theta * sigmax / 2
                elif opName == 'CY': ex = -1j * theta * sigmay / 2
                elif opName == 'CZ': ex = -1j * theta * sigmaz / 2
                Utarget = _spl.expm(ex)  # 2x2 unitary matrix operating on target qubit

            else:  # opName in ('CNOT','CPHASE')
                assert(len(args) == 2)  # qubit-label1, qubit-label2
                label1 = to_label(args[0]); label2 = to_label(args[1])

                if opName == 'CNOT':
                    Utarget = _np.array([[0, 1],
                                         [1, 0]], 'd')
                elif opName == 'CPHASE':
                    Utarget = _np.array([[1, 0],
                                         [0, -1]], 'd')

            # 4x4 unitary matrix operating on isolated two-qubit space
            U = _np.identity(4, 'complex'); U[2:, 2:] = Utarget
            assert(state_space.label_dimension(label1) == 4 and state_space.label_dimension(label2) == 4), \
                "%s gate must act on qubits!" % opName
            # complex 4x4 unitary matrix operating on two-qubit in Pauli-product basis
            Uop = _op.StaticUnitaryOp(U, 'pp', build_evotype)

            #FUTURE?:
            # if opName == "CNOT": stdname = "Gcnot"
            # elif opName == "CPHASE": stdname = "Gcphase"
            # else: stdname = None
            # Uop = _op.create_from_unitary_mx(U, type_preferences, 'pp', stdname=stdname, evotype=evotype)

            # a complex 2*num_qubits x 2*num_qubits mx unitary on full space
            Uop_embed = _op.EmbeddedOp(state_space, [label1, label2], Uop)
            # a real 4*num_qubits x 4*num_qubits mx superoperator in Pauli-product basis
            superop_mx_pp = Uop_embed.to_dense(on_space='HilbertSchmidt')
            # a real 4*num_qubits x 4*num_qubits mx superoperator in final basis
            superop_mx_in_basis = _bt.change_basis(superop_mx_pp, 'pp', basis)

        elif opName == "LX":  # TODO - better way to describe leakage?
            assert(len(args) == 3)  # theta, dmIndex1, dmIndex2 - X rotation between any two density matrix basis states
            theta = eval(args[0], {"__builtins__": None}, {'pi': _np.pi})
            i1 = int(args[1])  # row/column index of a single *state* within the density matrix
            i2 = int(args[2])  # row/column index of a single *state* within the density matrix
            ex = -1j * theta * sigmax / 2
            Uop = _spl.expm(ex)  # 2x2 unitary matrix operating on the i1-th and i2-th states of the state space basis

            opDim = basis.dim
            dmDim = int(_np.sqrt(basis.elsize))  # matrix dim of the "embedding space"
            if isinstance(basis, _DirectSumBasis):
                blockDims = [c.dim for c in basis.component_bases]
            else: blockDims = [opDim]

            Utot = _np.identity(dmDim, 'complex')
            Utot[i1, i1] = Uop[0, 0]
            Utot[i1, i2] = Uop[0, 1]
            Utot[i2, i1] = Uop[1, 0]
            Utot[i2, i2] = Uop[1, 1]

            # dmDim^2 x dmDim^2 mx operating on vectorized total densty matrix
            opTermInStdBasis = _ot.unitary_to_process_mx(Utot)

            # contract [3] to [2, 1]
            embedded_std_basis = _Basis.cast('std', 9)  # [2]
            std_basis = _Basis.cast('std', blockDims)  # std basis w/blockdim structure, i.e. [4,1]
            opTermInReducedStdBasis = _bt.resize_std_mx(opTermInStdBasis, 'contract',
                                                        embedded_std_basis, std_basis)

            superop_mx_in_basis = _bt.change_basis(opTermInReducedStdBasis, std_basis, basis)

        else: raise ValueError("Invalid gate name: %s" % opName)

        superop_mxs_in_basis.append(superop_mx_in_basis)

    #Note: expressions are listed in "matrix composition order"
    final_superop_mx = superop_mxs_in_basis[0]
    for mx in superop_mxs_in_basis[1:]:
        final_superop_mx = _np.dot(final_superop_mx, mx)

    if basis.real:
        assert(_np.linalg.norm(final_superop_mx.imag) < 1e-6), "Operation matrix should be real but isn't!"
        final_superop_mx = _np.real(final_superop_mx)

    return _op.create_from_superop_mx(final_superop_mx, parameterization, basis,
                                      evotype=evotype, state_space=state_space)


@_deprecated_fn('_basis_create_operation(...)')
def _create_operation(state_space_dims, state_space_labels, op_expr, basis="gm", parameterization="full"):
    """
    DEPRECATED: use :func:`_basis_create_operation` instead.
    """
    udims = []
    for tpbdims in state_space_dims:
        udims.append(tuple([int(_np.sqrt(d)) for d in tpbdims]))
    sslbls = _statespace.ExplicitStateSpace(state_space_labels, udims)
    return _basis_create_operation(sslbls, op_expr, _Basis.cast(basis, state_space_dims),
                                   parameterization, evotype='default')


def _create_explicit_model_from_expressions(state_space, basis,
                                            op_labels, op_expressions,
                                            prep_labels=('rho0',), prep_expressions=('0',),
                                            effect_labels='standard', effect_expressions='standard',
                                            povm_labels='Mdefault', gate_type="full", prep_type="auto",
                                            povm_type="auto", instrument_type="auto", evotype='default'):
    """
    Build a new Model given lists of operation labels and expressions.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this model.

    basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    op_labels : list of strings
        A list of labels for each created gate in the final model.  To
         conform with text file parsing conventions these names should begin
         with a capital G and can be followed by any number of lowercase
         characters, numbers, or the underscore character.

    op_expressions : list of strings
        A list of gate expressions, each corresponding to a operation label in
        op_labels, which determine what operation each gate performs (see
        documentation for :meth:`_basis_create_operation`).

    prep_labels : list of string, optional
        A list of labels for each created state preparation in the final
        model.  To conform with conventions these labels should begin with
        "rho".

    prep_expressions : list of strings, optional
        A list of vector expressions for each state preparation vector (see
        documentation for :meth:`_create_spam_vector`).

    effect_labels : list, optional
        If `povm_labels` is a string, then this is just a list of the effect
        (outcome) labels for the single POVM.  If `povm_labels` is a tuple,
        then `effect_labels` must be a list of lists of effect labels, each
        list corresponding to a POVM.  If set to the special string `"standard"`
        then the length-n binary strings are used when the state space consists
        of n qubits (e.g. `"000"`, `"001"`, ... `"111"` for 3 qubits) and
        the labels `"0"`, `"1"`, ... `"<dim>"` are used, where `<dim>`
        is the dimension of the state space, in all non-qubit cases.

    effect_expressions : list, optional
        A list or list-of-lists of (string) vector expressions for each POVM
        effect vector (see documentation for :meth:`_create_spam_vector`).  Expressions
        correspond to labels in `effect_labels`.  If set to the special string
        `"standard"`, then the expressions `"0"`, `"1"`, ... `"<dim>"` are used,
        where `<dim>` is the dimension of the state space.

    povm_labels : list or string, optional
        A list of POVM labels, or a single (string) label.  In the latter case,
        only a single POVM is created and the format of `effect_labels` and
        `effect_expressions` is simplified (see above).

    parameterization : {"full","TP","static"}, optional
        How to parameterize the gates of the resulting Model (see
        documentation for :meth:`_basis_create_operation`).

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    Returns
    -------
    Model
        The created model.
    """
    #defP = "TP" if (parameterization in ("TP","linearTP")) else "full"
    state_space = _statespace.StateSpace.cast(state_space)

    ret = _emdl.ExplicitOpModel(state_space, basis.copy(), default_gate_type=gate_type,
                                default_prep_type=prep_type, default_povm_type=povm_type,
                                default_instrument_type=instrument_type, evotype=evotype)
    #prep_prefix="rho", effect_prefix="E", gate_prefix="G")

    if prep_type == "auto":
        prep_type = _state.get_state_type_from_op_type(gate_type)
    if povm_type == "auto":
        povm_type = _povm.get_povm_type_from_op_type(gate_type)
    if instrument_type == "auto":
        instrument_type = _instrument.get_instrument_type_from_op_type(gate_type)

    for label, rhoExpr in zip(prep_labels, prep_expressions):
        vec = _basis_create_spam_vector(rhoExpr, basis)
        ret.preps[label] = _state.create_from_dmvec(vec, prep_type, basis, evotype, state_space)

    if isinstance(povm_labels, str):
        povm_labels = [povm_labels]
        effect_labels = [effect_labels]
        effect_expressions = [effect_expressions]

    dmDim = int(_np.sqrt(basis.dim))  # "densitymx" evotype assumed... FIX?
    for povmLbl, ELbls, EExprs in zip(povm_labels,
                                      effect_labels, effect_expressions):
        effect_vecs = {}

        if ELbls == "standard":
            qubit_dim = 4
            if state_space.num_tensor_product_blocks == 1 and \
               all([ldim == qubit_dim for ldim in state_space.tensor_product_block_dimensions(0)]):
                # a single tensor product block comprised of qubits: '000', '001', etc.
                nQubits = len(state_space.tensor_product_block_dimensions(0))
                ELbls = [''.join(t) for t in _itertools.product(('0', '1'), repeat=nQubits)]
            else:
                ELbls = list(map(str, range(dmDim)))  # standard = 0,1,...,dmDim
        if EExprs == "standard":
            EExprs = list(map(str, range(dmDim)))  # standard = 0,1,...,dmDim

        effect_vecs = {label: _basis_create_spam_vector(expr, basis) for label, expr in zip(ELbls, EExprs)}

        if len(effect_vecs) > 0:  # don't add POVMs with 0 effects
            ret.povms[povmLbl] = _povm.create_from_dmvecs(effect_vecs, povm_type, basis, evotype, state_space)

    for (opLabel, opExpr) in zip(op_labels, op_expressions):
        ret.operations[opLabel] = _basis_create_operation(state_space, opExpr,
                                                          basis, gate_type, evotype)

    if gate_type == "full":
        ret.default_gauge_group = _gg.FullGaugeGroup(ret.state_space, evotype)
    elif gate_type == "full TP":
        ret.default_gauge_group = _gg.TPGaugeGroup(ret.state_space, evotype)
    elif gate_type == 'CPTP':
        ret.default_gauge_group = _gg.UnitaryGaugeGroup(ret.state_space, basis, evotype)
    else:
        ret.default_gauge_group = _gg.TrivialGaugeGroup(ret.state_space)

    return ret


def create_explicit_model_from_expressions(state_space,
                                           op_labels, op_expressions,
                                           prep_labels=('rho0',), prep_expressions=('0',),
                                           effect_labels='standard', effect_expressions='standard',
                                           povm_labels='Mdefault', basis="auto", gate_type="full",
                                           prep_type="auto", povm_type="auto", instrument_type="auto",
                                           evotype='default'):
    """
    Build a new :class:`ExplicitOpModel` given lists of labels and expressions.

    Parameters
    ----------
    state_space : StateSpace
        the state space for the model.

    op_labels : list of strings
        A list of labels for each created gate in the final model.  To
         conform with text file parsing conventions these names should begin
         with a capital G and can be followed by any number of lowercase
         characters, numbers, or the underscore character.

    op_expressions : list of strings
        A list of gate expressions, each corresponding to a operation label in
        op_labels, which determine what operation each gate performs (see
        documentation for :meth:`_basis_create_operation`).

    prep_labels : list of string
        A list of labels for each created state preparation in the final
        model.  To conform with conventions these labels should begin with
        "rho".

    prep_expressions : list of strings
        A list of vector expressions for each state preparation vector (see
        documentation for :meth:`_create_spam_vector`).

    effect_labels : list, optional
        If `povm_labels` is a string, then this is just a list of the effect
        (outcome) labels for the single POVM.  If `povm_labels` is a tuple,
        then `effect_labels` must be a list of lists of effect labels, each
        list corresponding to a POVM.  If set to the special string `"standard"`
        then the length-n binary strings are used when the state space consists
        of n qubits (e.g. `"000"`, `"001"`, ... `"111"` for 3 qubits) and
        the labels `"0"`, `"1"`, ... `"<dim>"` are used, where `<dim>`
        is the dimension of the state space, in all non-qubit cases.

    effect_expressions : list, optional
        A list or list-of-lists of (string) vector expressions for each POVM
        effect vector (see documentation for :meth:`_create_spam_vector`).  Expressions
        correspond to labels in `effect_labels`.  If set to the special string
        `"standard"`, then the expressions `"0"`, `"1"`, ... `"<dim>"` are used,
        where `<dim>` is the dimension of the state space.

    povm_labels : list or string, optional
        A list of POVM labels, or a single (string) label.  In the latter case,
        only a single POVM is created and the format of `effect_labels` and
        `effect_expressions` is simplified (see above).

    basis : {'gm','pp','std','qt','auto'}, optional
        the basis of the matrices in the returned Model

        - "std" = operation matrix operates on density mx expressed as sum of matrix
          units
        - "gm"  = operation matrix operates on dentity mx expressed as sum of
          normalized Gell-Mann matrices
        - "pp"  = operation matrix operates on density mx expresses as sum of
          tensor-product of Pauli matrices
        - "qt"  = operation matrix operates on density mx expressed as sum of
          Qutrit basis matrices
        - "auto" = "pp" if possible (integer num of qubits), "qt" if density
          matrix dim == 3, and "gm" otherwise.

    parameterization : {"full","TP"}, optional
        How to parameterize the gates of the resulting Model (see
        documentation for :meth:`_basis_create_operation`).

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    Returns
    -------
    ExplicitOpModel
        The created model.
    """

    #Note: so far, all allowed `parameterization` values => densitymx evotype
    state_space = _statespace.StateSpace.cast(state_space)
    stateSpaceDim = state_space.dim
    # Note: what about state_space_labels.tpb_dims?

    if basis == "auto":
        if _np.isclose(_np.log2(stateSpaceDim) / 2,
                       round(_np.log2(stateSpaceDim) / 2)):
            basis = "pp"
        elif stateSpaceDim == 9:
            basis = "qt"
        else: basis = "gm"

    return _create_explicit_model_from_expressions(state_space,
                                                   _Basis.cast(basis, state_space),
                                                   op_labels, op_expressions,
                                                   prep_labels, prep_expressions,
                                                   effect_labels, effect_expressions,
                                                   povm_labels, gate_type=gate_type,
                                                   prep_type=prep_type, povm_type=povm_type,
                                                   instrument_type=instrument_type, evotype=evotype)


def create_explicit_alias_model(mdl_primitives, alias_dict):
    """
    Creates a model by applying aliases to an existing model.

    The new model is created by composing the gates of an existing `Model`,
    `mdl_primitives`, according to a dictionary of `Circuit`s, `alias_dict`.
    The keys of `alias_dict` are the operation labels of the returned `Model`.
    state preparations and POVMs are unaltered, and simply copied from `mdl_primitives`.

    Parameters
    ----------
    mdl_primitives : Model
        A Model containing the "primitive" gates (those used to compose
        the gates of the returned model).

    alias_dict : dictionary
        A dictionary whose keys are strings and values are Circuit objects
        specifying sequences of primitive gates.  Each key,value pair specifies
        the composition rule for a creating a gate in the returned model.

    Returns
    -------
    Model
        A model whose gates are compositions of primitive gates and whose
        spam operations are the same as those of `mdl_primitives`.
    """
    mdl_new = mdl_primitives.copy()
    for gl in mdl_primitives.operations.keys():
        del mdl_new.operations[gl]  # remove all gates from mdl_new

    for gl, opstr in alias_dict.items():
        mdl_new.operations[gl] = mdl_primitives.sim.product(opstr)
        #Creates fully parameterized gates by default...
    return mdl_new


def create_explicit_model(processor_spec, custom_gates=None,
                          depolarization_strengths=None, stochastic_error_probs=None, lindblad_error_coeffs=None,
                          depolarization_parameterization='depolarize', stochastic_parameterization='stochastic',
                          lindblad_parameterization='auto',
                          evotype="default", simulator="auto",
                          ideal_gate_type='auto', ideal_spam_type='computational',
                          embed_gates=False, basis='pp'):
    modelnoises = []
    if depolarization_strengths is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _DepolarizationNoise(val, depolarization_parameterization)
                                               for lbl, val in depolarization_strengths.items()}))
    if stochastic_error_probs is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _StochasticNoise(val, stochastic_parameterization)
                                               for lbl, val in stochastic_error_probs.items()}))
    if lindblad_error_coeffs is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _LindbladNoise(val, lindblad_parameterization)
                                               for lbl, val in lindblad_error_coeffs.items()}))

    return _create_explicit_model(processor_spec, _ComposedOpModelNoise(modelnoises), custom_gates, evotype,
                                  simulator, ideal_gate_type, ideal_spam_type, embed_gates, basis)


def _create_explicit_model(processor_spec, modelnoise, custom_gates=None, evotype="default", simulator="auto",
                           ideal_gate_type='auto', ideal_prep_type='auto', ideal_povm_type='auto',
                           embed_gates=False, basis='pp'):
    qubit_labels = processor_spec.qubit_labels
    state_space = _statespace.QubitSpace(qubit_labels)
    evotype = _Evotype.cast(evotype)
    modelnoise = _OpModelNoise.cast(modelnoise)
    modelnoise.reset_access_counters()

    if custom_gates is None:
        custom_gates = {}

    if ideal_gate_type == "auto":
        ideal_gate_type = ('static standard', 'static clifford', 'static unitary')
    if ideal_prep_type == "auto":
        ideal_prep_type = _state.get_state_type_from_op_type(ideal_gate_type)
    if ideal_povm_type == "auto":
        ideal_povm_type = _povm.get_povm_type_from_op_type(ideal_gate_type)

    def _embed_unitary(statespace, target_labels, unitary):
        dummyop = _op.EmbeddedOp(statespace, target_labels,
                                 _op.StaticUnitaryOp(unitary, basis='pp', evotype="statevec_slow"))  # basis hardcode?
        return dummyop.to_dense("Hilbert")

    local_gates = _setup_local_gates(processor_spec, evotype, None, {}, ideal_gate_type)  # no custom *local* gates
    ret = _emdl.ExplicitOpModel(state_space, basis, default_gate_type=ideal_gate_type, evotype=evotype,
                                simulator=simulator)

    # Special rule: when initializng an explicit model, if the processor spec has an implied global idle
    #  gate (e.g. "(idle)", then the created model instead has a empty-tuple Label as the key for this op.
    global_idle_name = processor_spec.global_idle_gate_name
    if (global_idle_name is not None) and global_idle_name.startswith('(') and global_idle_name.endswith(')'):
        gn_to_make_emptytup = global_idle_name
    else:
        gn_to_make_emptytup = None

    for gn, gate_unitary in processor_spec.gate_unitaries.items():

        gate_is_factory = callable(gate_unitary)
        resolved_avail = processor_spec.resolved_availability(gn)

        if callable(resolved_avail) or resolved_avail == '*':
            assert (embed_gates), "Cannot create factories with `embed_gates=False` yet!"
            key = _label.Label(gn) if (gn != gn_to_make_emptytup) else _label.Label(())
            allowed_sslbls_fn = resolved_avail if callable(resolved_avail) else None
            gate_nQubits = processor_spec.gate_num_qubits(gn)
            ideal_factory = _opfactory.EmbeddingOpFactory(
                state_space, local_gates[gn], num_target_labels=gate_nQubits, allowed_sslbls_fn=allowed_sslbls_fn)
            noiseop = modelnoise.create_errormap(key, evotype, state_space)  # No target indices... just local errs?
            factory = ideal_factory if (noiseop is None) else _op.ComposedOpFactory([ideal_factory, noiseop])
            ret.factories[key] = factory

        else:  # resolved_avail is a list/tuple of available sslbls for the current gate/factory
            for inds in resolved_avail:  # inds are target qubit labels
                key = _label.Label(()) if (inds is None and gn == gn_to_make_emptytup) else _label.Label(gn, inds)

                if key in custom_gates:  # allow custom_gates to specify gate elements directly
                    if isinstance(custom_gates[key], _opfactory.OpFactory):
                        ret.factories[key] = custom_gates[key]
                    elif isinstance(custom_gates[key], _op.LinearOperator):
                        ret.operations[key] = custom_gates[key]
                    else:  # presumably a numpy array or something like it.
                        ret.operations[key] = _op.StaticArbitraryOp(custom_gates[key], evotype,
                                                                    state_space)  # static gates by default
                    continue

                if gate_is_factory:
                    assert(embed_gates), "Cannot create factories with `embed_gates=False` yet!"
                    # TODO: check for modelnoise on *local* factory, i.e. create_errormap(gn, ...)??
                    if inds is None or inds == tuple(qubit_labels):  # then no need to embed
                        ideal_factory = local_gates[gn]
                    else:
                        ideal_factory = _opfactory.EmbeddedOpFactory(state_space, inds, local_gates[gn])
                    noiseop = modelnoise.create_errormap(key, evotype, state_space, target_labels=inds)
                    factory = ideal_factory if (noiseop is None) else _op.ComposedOpFactory([ideal_factory, noiseop])
                    ret.factories[key] = factory
                else:
                    if inds is None or inds == tuple(qubit_labels):  # then no need to embed
                        if isinstance(gate_unitary, (int, _np.int64)):  # interpret gate_unitary as identity
                            assert(gate_unitary == len(qubit_labels)), \
                                "Idle unitary as int should be on all qubits for %s" % (str(gn))
                            ideal_gate = _op.ComposedOp([], evotype, state_space)  # (identity gate on *all* qubits)
                        else:
                            ideal_gate = _op.create_from_unitary_mx(gate_unitary, ideal_gate_type, 'pp',
                                                                    None, evotype, state_space)
                    else:
                        if embed_gates:
                            ideal_gate = local_gates[gn]
                            ideal_gate = _op.EmbeddedOp(state_space, inds, ideal_gate)
                        else:
                            if isinstance(gate_unitary, (int, _np.int64)):  # interpret gate_unitary as identity
                                gate_unitary = _np.identity(2**gate_unitary, 'd')  # turn into explicit identity op
                            embedded_unitary = _embed_unitary(state_space, inds, gate_unitary)
                            ideal_gate = _op.create_from_unitary_mx(embedded_unitary, ideal_gate_type, 'pp',
                                                                    None, evotype, state_space)

                    #TODO: check for modelnoise on *local* gate, i.e. create_errormap(gn, ...)??
                    noiseop = modelnoise.create_errormap(key, evotype, state_space, target_labels=inds)
                    layer = _op.ComposedOp([ideal_gate, noiseop]) if (noiseop is not None) else ideal_gate
                    ret.operations[key] = layer

    # SPAM:
    local_noise = False; independent_gates = True; independent_spam = True
    prep_layers, povm_layers = _create_spam_layers(processor_spec, modelnoise, local_noise,
                                                   ideal_prep_type, ideal_povm_type, evotype,
                                                   state_space, independent_gates, independent_spam)
    for k, v in prep_layers.items():
        ret.preps[k] = v
    for k, v in povm_layers.items():
        ret.povms[k] = v

    modelnoise.warn_about_zero_counters()
    return ret


def _create_spam_layers(processor_spec, modelnoise, local_noise,
                        ideal_prep_type, ideal_povm_type, evotype, state_space, independent_gates, independent_spam):
    """ local_noise=True creates lindblad ops that are embedded & composed 1Q ops, and assumes
        that modelnoise specifies 1Q noise.  local_noise=False assumes modelnoise specifies n-qubit noise"""
    qubit_labels = processor_spec.qubit_labels
    num_qubits = processor_spec.num_qubits
    singleQ_state_space = _statespace.default_space_for_udim(2)  # single qubit state space

    #  Step 1 -- get the ideal prep and POVM, created as the types we want
    #  Step 2 -- add noise, by composing ideal with a noise operation (if desired)
    prep_layers = {}
    povm_layers = {}

    def _add_prep_noise(prep_ops):
        """ Adds one or more noise ops to prep_ops lists (to compose later) """
        if local_noise:  # then assume modelnoise specifies 1Q errors
            prep_noiseop1Q = modelnoise.create_errormap('prep', evotype, singleQ_state_space, target_labels=None)
            if prep_noiseop1Q is not None:
                err_gates = [prep_noiseop1Q.copy() for i in range(num_qubits)] \
                    if independent_gates else [prep_noiseop1Q] * num_qubits
                prep_ops.extend([_op.EmbeddedOp(state_space, [qubit_labels[i]], err_gates[i])
                                 for i in range(num_qubits)])
        else:  # use modelnoise to construct n-qubit noise
            prepNoiseMap = modelnoise.create_errormap('prep', evotype, state_space, target_labels=None,
                                                      qubit_graph=processor_spec.qubit_graph)
            if prepNoiseMap is not None: prep_ops.append(prepNoiseMap)

    def _add_povm_noise(povm_ops):
        """ Adds one or more noise ops to prep_ops lists (to compose later) """
        if local_noise:  # then assume modelnoise specifies 1Q errors
            povm_noiseop1Q = modelnoise.create_errormap('povm', evotype, singleQ_state_space, target_labels=None)
            if povm_noiseop1Q is not None:
                err_gates = [povm_noiseop1Q.copy() for i in range(num_qubits)] \
                    if independent_gates else [povm_noiseop1Q] * num_qubits
                povm_ops.extend([_op.EmbeddedOp(state_space, [qubit_labels[i]], err_gates[i])
                                 for i in range(num_qubits)])
        else:  # use modelnoise to construct n-qubit noise
            povmNoiseMap = modelnoise.create_errormap('povm', evotype, state_space, target_labels=None,
                                                      qubit_graph=processor_spec.qubit_graph)
            if povmNoiseMap is not None: povm_ops.append(povmNoiseMap)

    def _add_to_prep_layers(ideal_prep, prep_ops):
        """ Adds noise elements to prep_layers """
        if len(prep_ops_to_compose) == 0:
            prep_layers['rho0'] = ideal_prep
        elif len(prep_ops_to_compose) == 1:
            prep_layers['rho0'] = _state.ComposedState(ideal_prep, prep_ops[0])
        else:
            prep_layers['rho0'] = _state.ComposedState(ideal_prep, _op.ComposedOp(prep_ops))

    def _add_to_povm_layers(ideal_povm, povm_ops):
        """ Adds noise elements to povm_layers """
        if len(povm_ops_to_compose) == 0:
            povm_layers['Mdefault'] = ideal_povm
        elif len(povm_ops_to_compose) == 1:
            povm_layers['Mdefault'] = _povm.ComposedPOVM(povm_ops[0], ideal_povm, 'pp')
        else:
            povm_layers['Mdefault'] = _povm.ComposedPOVM(_op.ComposedOp(povm_ops), ideal_povm, 'pp')

    def _create_nq_noise(lndtype):
        if local_noise:
            # create a 1-qubit exp(errorgen) that is applied to each qubit independently
            errgen_1Q = _op.LindbladErrorgen.from_error_generator(singleQ_state_space.dim, lndtype, 'pp', 'pp',
                                                                  truncate=True, evotype=evotype, state_space=None)
            err_gateNQ = _op.ComposedOp([_op.EmbeddedOp(state_space, [qubit_labels[i]],
                                                        _op.ExpErrorgenOp(errgen_1Q.copy()))
                                         for i in range(num_qubits)], evotype, state_space)
        else:
            # create an n-qubit exp(errorgen)
            errgen_NQ = _op.LindbladErrorgen.from_error_generator(state_space.dim, lndtype, 'pp', 'pp',
                                                                  truncate=True, evotype=evotype,
                                                                  state_space=state_space)
            err_gateNQ = _op.ExpErrorgenOp(errgen_NQ)
        return err_gateNQ

    # Here's where the actual logic starts.  The above functions avoid repeated blocks within the different
    # cases below.

    # Prep logic
    if isinstance(ideal_prep_type, (tuple, list)): ideal_prep_type = ideal_prep_type[0]  # HACK to support multiple vals
    if ideal_prep_type == 'computational' or ideal_prep_type.startswith('lindblad '):
        ideal_prep = _state.ComputationalBasisState([0] * num_qubits, 'pp', evotype, state_space)

        prep_ops_to_compose = []
        if ideal_prep_type.startswith('lindblad '):  # then add a composed exp(errorgen) to computational SPAM
            lndtype = ideal_prep_type[len('lindblad '):]

            err_gateNQ = _create_nq_noise(lndtype)

            prep_ops_to_compose.append(err_gateNQ)

        # Add noise
        _add_prep_noise(prep_ops_to_compose)

        #Add final ops to returned dictionaries  (Note: None -> ComputationPOVM within ComposedPOVM)
        _add_to_prep_layers(ideal_prep, prep_ops_to_compose)

    elif ideal_prep_type.startswith('tensor product '):
        #Note: with "tensor product <X>" types, e.g. "tensor product static", we assume modelnoise specifies just
        # a 1Q noise operation, even when `local_noise=False`
        vectype = ideal_prep_type[len('tensor product '):]

        v0, v1 = _np.array([1, 0], 'd'), _np.array([0, 1], 'd')
        ideal_prep1Q = _state.create_from_pure_vector(v0, vectype, 'pp', evotype, state_space=None)
        prep_factors = [ideal_prep1Q.copy() for i in range(num_qubits)]

        # Add noise
        prep_noiseop1Q = modelnoise.create_errormap('prep', evotype, singleQ_state_space, target_labels=None)
        if prep_noiseop1Q is not None:
            prep_factors = [_state.ComposedState(
                factor, (prep_noiseop1Q.copy() if independent_spam else prep_noiseop1Q)) for factor in prep_factors]

        prep_layers['rho0'] = _state.TensorProductState(prep_factors, state_space)

    else:  # assume ideal_spam_type is a valid 'vectype' for creating n-qubit state vectors & POVMs

        vectype = ideal_prep_type
        vecs = []  # all the basis vectors for num_qubits
        for i in range(2**num_qubits):
            v = _np.zeros(2**num_qubits, 'd'); v[i] = 1.0
            vecs.append(v)

        ideal_prep = _state.create_from_pure_vector(vecs[0], vectype, 'pp', evotype, state_space=state_space)

        # Add noise
        prep_ops_to_compose = []
        _add_prep_noise(prep_ops_to_compose)

        # Add final ops to returned dictionaries
        _add_to_prep_layers(ideal_prep, prep_ops_to_compose)

    # Povm logic
    if isinstance(ideal_povm_type, (tuple, list)): ideal_povm_type = ideal_povm_type[0]  # HACK to support multiple vals
    if ideal_povm_type == 'computational' or ideal_povm_type.startswith('lindblad '):
        ideal_povm = _povm.ComputationalBasisPOVM(num_qubits, evotype, state_space=state_space)

        povm_ops_to_compose = []
        if ideal_povm_type.startswith('lindblad '):  # then add a composed exp(errorgen) to computational SPAM
            lndtype = ideal_povm_type[len('lindblad '):]

            err_gateNQ = _create_nq_noise(lndtype)

            povm_ops_to_compose.append(err_gateNQ.copy())  # .copy() => POVM errors independent

        # Add noise
        _add_povm_noise(povm_ops_to_compose)

        #Add final ops to returned dictionaries  (Note: None -> ComputationPOVM within ComposedPOVM)
        effective_ideal_povm = None if len(povm_ops_to_compose) > 0 else ideal_povm
        _add_to_povm_layers(effective_ideal_povm, povm_ops_to_compose)

    elif ideal_povm_type.startswith('tensor product '):
        #Note: with "tensor product <X>" types, e.g. "tensor product static", we assume modelnoise specifies just
        # a 1Q noise operation, even when `local_noise=False`
        vectype = ideal_povm_type[len('tensor product '):]

        v0, v1 = _np.array([1, 0], 'd'), _np.array([0, 1], 'd')
        ideal_povm1Q = _povm.create_from_pure_vectors([('0', v0), ('1', v1)], vectype, 'pp',
                                                      evotype, state_space=None)
        povm_factors = [ideal_povm1Q.copy() for i in range(num_qubits)]

        # Add noise
        povm_noiseop1Q = modelnoise.create_errormap('povm', evotype, singleQ_state_space, target_labels=None)
        if povm_noiseop1Q is not None:
            povm_factors = [_povm.ComposedPOVM(
                (povm_noiseop1Q.copy() if independent_spam else povm_noiseop1Q), factor, 'pp')
                for factor in povm_factors]

        povm_layers['Mdefault'] = _povm.TensorProductPOVM(povm_factors, evotype, state_space)

    else:  # assume ideal_spam_type is a valid 'vectype' for creating n-qubit state vectors & POVMs

        vectype = ideal_povm_type
        vecs = []  # all the basis vectors for num_qubits
        for i in range(2**num_qubits):
            v = _np.zeros(2**num_qubits, 'd'); v[i] = 1.0
            vecs.append(v)

        ideal_povm = _povm.create_from_pure_vectors(
            [(format(i, 'b').zfill(num_qubits), v) for i, v in enumerate(vecs)],
            vectype, 'pp', evotype, state_space=state_space)

        # Add noise
        povm_ops_to_compose = []
        _add_povm_noise(povm_ops_to_compose)

        # Add final ops to returned dictionaries
        _add_to_povm_layers(ideal_povm, povm_ops_to_compose)

    return prep_layers, povm_layers


def _setup_local_gates(processor_spec, evotype, modelnoise=None, custom_gates=None,
                       ideal_gate_type=('static standard', 'static clifford', 'static unitary')):
    """
    Construct a dictionary of potentially noisy gates that act only on their target qubits.

    These gates are "local" because they act only on their intended target qubits.  The gates
    consist of an ideal gate (obviously local, and crosstalk free) of the type given by
    `ideal_gate_type` composed with a noise operation given by `modelnoise`, if one exists.
    The returned dictionary contains keys for all the gate names in `processor_spec`.  Custom
    gate objects can be given by `custom_gates`, which override the normal gate construction.

    Parameters
    ----------
    processor_spec : ProcessorSpec
        The processor to create gate operations for.  This object specifies the
        gate names and unitaries for the processor, among other things.

    evotype : Evotype
        Create gate objects with this evolution type.

    modelnoise : ModelNoise, optional
        Noise that should be applied after the ideal gates.  This noise must
        be *local* to each gate (i.e. acting on its target qubits).  See the
        :class:`ModelNoise` object documentation for details regarding how
        to specify different types of noise.  If `None`, then no noise is added .

    custom_gates : dict, optional
        A dictionary of gate objects that should be placed in the returned
        dictionary in lieu of objects that would normally be constructed.
        Keys are gate names and values are gates.

    ideal_gate_type : str or tuple, optional
        A gate type or tuple of gate types (listed in order of priority) which
        is used to construct the ideal gates.  A gate type usually specifies the
        Python class that will be created, which determines 1) the parameterization
        of the gate and 2) the class/category of the gate (e.g. a :class:`StaticClifford`
        operation has no parameters and is a Clifford operation).

    Returns
    -------
    gatedict : dict
        A dictionary mapping gate names to local gate operations.
    """
    std_gate_unitaries = _itgs.standard_gatename_unitaries()
    if custom_gates is None: custom_gates = {}
    if modelnoise is None: modelnoise = _OpModelPerOpNoise({})

    # All possible entries into the upcoming gate dictionary
    # Not just gatenames as it is possible to override in qubit-specific operations
    all_keys = _lt.remove_duplicates(list(processor_spec.gate_names)
                                     + list(custom_gates.keys())
                                     + list(modelnoise.keys()))

    gatedict = _collections.OrderedDict()
    for key in all_keys:
        # Use custom gate directly as error gate
        if key in custom_gates:
            gatedict[key] = custom_gates[key]
            continue

        # Skip prep, and povm here, just do gates
        if key in ['prep', 'povm']:
            continue

        # If key has qubits, get base name for lookup
        label = _label.Label(key)
        name = label.name

        U = processor_spec.gate_unitaries[name]  # all gate names must be in the processorspec
        if ((name not in processor_spec.nonstd_gate_unitaries)
                or (not callable(processor_spec.nonstd_gate_unitaries[name]) and (name in std_gate_unitaries)
                    and processor_spec.nonstd_gate_unitaries[name].shape == std_gate_unitaries[name].shape
                    and _np.allclose(processor_spec.nonstd_gate_unitaries[name], std_gate_unitaries[name]))):
            stdname = name  # setting `stdname` != None means we can try to create a StaticStandardOp below
        else:
            stdname = None

        if isinstance(U, (int, _np.int64)):  # signals that the gate is an identity on `U` qubits and shouldn't be built
            # A key is added to the returned gatedict only if there is any noise on this operation
            ideal_gate_state_space = _statespace.default_space_for_num_qubits(U)
            noiseop = modelnoise.create_errormap(key, evotype, ideal_gate_state_space, target_labels=None)
            if noiseop is not None:
                gatedict[key] = noiseop

        elif not callable(U):  # normal operation (not a factory)
            ideal_gate = _op.create_from_unitary_mx(U, ideal_gate_type, 'pp', stdname, evotype, state_space=None)
            noiseop = modelnoise.create_errormap(key, evotype, ideal_gate.state_space, target_labels=None)
            # Note: above line creates a *local* noise op, working entirely in the ideal gate's target space.
            #   This means it will fail to create error maps with a given (non-local/stencil) set of sslbls, as desired

            if noiseop is None:
                gatedict[key] = ideal_gate
            else:
                if isinstance(noiseop, _op.ComposedOp):  # avoid additional nested ComposedOp if we already have one
                    noiseop.insert(0, ideal_gate)
                    gatedict[key] = noiseop
                else:
                    gatedict[key] = _op.ComposedOp([ideal_gate, noiseop])

        else:  # a factory, given by the unitary-valued function U: args -> unitary
            local_state_space = _statespace.default_space_for_udim(U(None).shape[0])
            ideal_factory = _opfactory.UnitaryOpFactory(U, local_state_space, 'pp', evotype)
            noiseop = modelnoise.create_errormap(key, evotype, ideal_factory.state_space, target_labels=None)
            gatedict[key] = _opfactory.ComposedOpFactory([ideal_factory, noiseop]) \
                if (noiseop is not None) else ideal_factory
    return gatedict


def create_crosstalk_free_model(processor_spec, custom_gates=None,
                                depolarization_strengths=None, stochastic_error_probs=None, lindblad_error_coeffs=None,
                                depolarization_parameterization='depolarize', stochastic_parameterization='stochastic',
                                lindblad_parameterization='auto',
                                evotype="default", simulator="auto", on_construction_error='raise',
                                independent_gates=False, independent_spam=True, ensure_composed_gates=False,
                                ideal_gate_type='auto', ideal_spam_type='computational'):
    """
    Create a n-qubit "crosstalk-free" model.

    By virtue of being crosstalk-free, this model's operations only
    act nontrivially on their target qubits.  Gates consist of an ideal gate
    operation possibly followed by an error operation.

    Errors can be specified using any combination of the 4 error rate/coeff arguments,
    but each gate name must be provided exclusively to one type of specification.
    Each specification results in a different type of operation, depending on the parameterization:
        - `depolarization_strengths`    -> DepolarizeOp, StochasticNoiseOp, or exp(LindbladErrorgen)
        - `stochastic_error_probs`      -> StochasticNoiseOp or exp(LindbladErrorgen)
        - `lindblad_error_coeffs`       -> exp(LindbladErrorgen)

    In addition to the gate names, the special values `"prep"` and `"povm"` may be
    used as keys to specify the error on the state preparation, measurement, respectively.

    Parameters
    ----------
    processor_spec : ProcessorSpec
        The processor specification to create a model for.  This object specifies the
        gate names and unitaries for the processor, and their availability on the
        processor.

    custom_gates : dict, optional
        A dictionary that associates with gate labels
        :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
        objects.  These objects override any other behavior for constructing
        their designated operations.  Keys of this dictionary may
        be string-type gate *names* or labels that include target qubits.

    depolarization_strengths : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are floats that specify the strength of uniform depolarization.

    stochastic_error_probs : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are tuples that specify Pauli-stochastic rates for each of the non-trivial
        Paulis (so a 3-tuple would be expected for a 1Q gate and a 15-tuple for a 2Q gate).

    lindblad_error_coeffs : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are dictionaries corresponding to the `lindblad_term_dict` kwarg taken
        by `LindbladErrorgen`. Keys are `(termType, basisLabel1, <basisLabel2>)`
        tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
        (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
        have a single basis label (so key is a 2-tuple) whereas Stochastic
        tuples with 1 basis label indicate a *diagonal* term, and are the
        only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
        Stochastic term tuples can include 2 basis labels to specify
        "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
        strings or integers.  Values are complex coefficients.

    depolarization_parameterization : str of {"depolarize", "stochastic", or "lindblad"}
        Determines whether a DepolarizeOp, StochasticNoiseOp, or LindbladErrorgen
        is used to parameterize the depolarization noise, respectively.
        When "depolarize" (the default), a DepolarizeOp is created with the strength given
        in `depolarization_strengths`. When "stochastic", the depolarization strength is split
        evenly among the stochastic channels of a StochasticOp. When "lindblad", the depolarization
        strength is split evenly among the coefficients of the stochastic error generators
        (which are exponentiated to form a LindbladErrorgen with the "depol" parameterization).

    stochastic_parameterization : str of {"stochastic", or "lindblad"}
        Determines whether a StochasticNoiseOp or LindbladErrorgen is used to parameterize the
        stochastic noise, respectively. When "stochastic", elements of `stochastic_error_probs`
        are used as coefficients in a linear combination of stochastic channels (the default).
        When "lindblad", the elements of `stochastic_error_probs` are coefficients of
        stochastic error generators (which are exponentiated to form a LindbladErrorgen with the
        "cptp" parameterization).

    lindblad_parameterization : "auto" or a LindbladErrorgen paramtype
        Determines the parameterization of the LindbladErrorgen. When "auto" (the default), the parameterization
        is inferred from the types of error generators specified in the `lindblad_error_coeffs` dictionaries.
        When not "auto", the parameterization type is passed through to the LindbladErrorgen.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The simulator used to compute predicted probabilities for the
        resulting :class:`Model`.  Using `"auto"` selects `"matrix"` when there
        are 2 qubits or less, and otherwise selects `"map"`.

    on_construction_error : {'raise','warn',ignore'}
        What to do when the creation of a gate with the given
        `parameterization` fails.  Usually you'll want to `"raise"` the error.
        In some cases, for example when converting as many gates as you can
        into `parameterization="clifford"` gates, `"warn"` or even `"ignore"`
        may be useful.

    independent_gates : bool, optional
        Whether gates are allowed independent local noise or not.  If False,
        then all gates with the same name (e.g. "Gx") will have the *same*
        (local) noise (e.g. an overrotation by 1 degree), and the
        `operation_bks['gates']` dictionary contains a single key per gate
        name.  If True, then gates with the same name acting on different
        qubits may have different local noise, and so the
        `operation_bks['gates']` dictionary contains a key for each gate
         available gate placement.

    ensure_composed_gates : bool, optional
        If True then the elements of the `operation_bks['gates']` will always
        be :class:`ComposedOp` objects.  The purpose of this is to
        facilitate modifying the gate operations after the model is created.
        If False, then the appropriately parameterized gate objects (often
        dense gates) are used directly.

    ideal_gate_type : str or tuple, optional
        A gate type or tuple of gate types (listed in order of priority) which
        is used to construct the ideal gates.  A gate type usually specifies the
        Python class that will be created, which determines 1) the parameterization
        of the gate and 2) the class/category of the gate (e.g. a :class:`StaticClifford`
        operation has no parameters and is a Clifford operation).

    ideal_spam_type : str or tuple, optional
        Similar to `ideal_gate_type` but for SPAM elements (state preparations
        and POVMs).

    Returns
    -------
    LocalNoiseModel
        A model with `"rho0"` prep, `"Mdefault"` POVM, and gates labeled by
        the gate names and qubit labels (as specified by `processor_spec`).
        For instance, the operation label for the `"Gx"` gate on the second
        qubit might be `Label("Gx",1)`.
    """
    modelnoises = []
    if depolarization_strengths is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _DepolarizationNoise(val, depolarization_parameterization)
                                               for lbl, val in depolarization_strengths.items()}))
    if stochastic_error_probs is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _StochasticNoise(val, stochastic_parameterization)
                                               for lbl, val in stochastic_error_probs.items()}))
    if lindblad_error_coeffs is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _LindbladNoise(val, lindblad_parameterization)
                                               for lbl, val in lindblad_error_coeffs.items()}))

    return _create_crosstalk_free_model(processor_spec, _ComposedOpModelNoise(modelnoises), custom_gates, evotype,
                                        simulator, on_construction_error, independent_gates, independent_spam,
                                        ensure_composed_gates, ideal_gate_type, ideal_spam_type, ideal_spam_type)


def _create_crosstalk_free_model(processor_spec, modelnoise, custom_gates=None, evotype="default", simulator="auto",
                                 on_construction_error='raise', independent_gates=False, independent_spam=True,
                                 ensure_composed_gates=False, ideal_gate_type='auto', ideal_prep_type='auto',
                                 ideal_povm_type='auto'):
    """
    Create a n-qubit "crosstalk-free" model.

    Similar to :method:`create_crosstalk_free_model` but the noise is input more generally,
    as a :class:`ModelNoise` object.  Arguments are the same as this function except that
    `modelnoise` is given instead of several more specific noise-describing arguments.

    Returns
    -------
    LocalNoiseModel
    """
    qubit_labels = processor_spec.qubit_labels
    state_space = _statespace.QubitSpace(qubit_labels)
    evotype = _Evotype.cast(evotype)
    modelnoise = _OpModelNoise.cast(modelnoise)
    modelnoise.reset_access_counters()

    if ideal_gate_type == "auto":
        ideal_gate_type = ('static standard', 'static clifford', 'static unitary')
    if ideal_prep_type == "auto":
        ideal_prep_type = _state.get_state_type_from_op_type(ideal_gate_type)
    if ideal_povm_type == "auto":
        ideal_povm_type = _povm.get_povm_type_from_op_type(ideal_gate_type)

    gatedict = _setup_local_gates(processor_spec, evotype, modelnoise, custom_gates, ideal_gate_type)

    # (Note: global idle is now handled through processor-spec processing)

    # SPAM:
    local_noise = True
    prep_layers, povm_layers = _create_spam_layers(processor_spec, modelnoise, local_noise,
                                                   ideal_prep_type, ideal_povm_type, evotype,
                                                   state_space, independent_gates, independent_spam)

    modelnoise.warn_about_zero_counters()
    return _LocalNoiseModel(processor_spec, gatedict, prep_layers, povm_layers,
                            evotype, simulator, on_construction_error,
                            independent_gates, ensure_composed_gates)


def create_cloud_crosstalk_model(processor_spec, custom_gates=None,
                                 depolarization_strengths=None, stochastic_error_probs=None, lindblad_error_coeffs=None,
                                 depolarization_parameterization='depolarize', stochastic_parameterization='stochastic',
                                 lindblad_parameterization='auto', evotype="default", simulator="auto",
                                 independent_gates=False, independent_spam=True, errcomp_type="errorgens",
                                 implicit_idle_mode="add_global", verbosity=0):
    """
    Create a n-qubit "cloud-crosstalk" model.

    In a cloud crosstalk model, gates consist of a (local) ideal gates followed
    by an error operation that can act nontrivially on *any* of the processor's qubits
    (not just a gate's target qubits).  Typically a gate's errors are specified
    relative to the gate's target qubits, forming a "cloud" of errors around the
    target qubits using some notion of locality (that may not be spatial, e.g.
    local in frequency).  Currently, the "ideal" portion of each gate can only be
    created as a *static* (parameterless) object -- all gate parameters come from
    the error operation.

    Errors can be specified using any combination of the 4 error rate/coeff arguments,
    but each gate name must be provided exclusively to one type of specification.
    Each specification results in a different type of operation, depending on the parameterization:
        - `depolarization_strengths`    -> DepolarizeOp, StochasticNoiseOp, or exp(LindbladErrorgen)
        - `stochastic_error_probs`      -> StochasticNoiseOp or exp(LindbladErrorgen)
        - `lindblad_error_coeffs`       -> exp(LindbladErrorgen)

    In addition to the gate names, the special values `"prep"` and `"povm"` may be
    used as keys to specify the error on the state preparation, measurement, respectively.

    Parameters
    ----------
    processor_spec : ProcessorSpec
        The processor specification to create a model for.  This object specifies the
        gate names and unitaries for the processor, and their availability on the
        processor.

    custom_gates : dict, optional
        A dictionary that associates with gate labels
        :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
        objects.  These objects override any other behavior for constructing
        their designated operations.  Keys of this dictionary may
        be string-type gate *names* or labels that include target qubits.

    depolarization_strengths : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are floats that specify the strength of uniform depolarization.

    stochastic_error_probs : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are tuples that specify Pauli-stochastic rates for each of the non-trivial
        Paulis (so a 3-tuple would be expected for a 1Q gate and a 15-tuple for a 2Q gate).

    lindblad_error_coeffs : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are dictionaries corresponding to the `lindblad_term_dict` kwarg taken
        by `LindbladErrorgen`. Keys are `(termType, basisLabel1, <basisLabel2>)`
        tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
        (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
        have a single basis label (so key is a 2-tuple) whereas Stochastic
        tuples with 1 basis label indicate a *diagonal* term, and are the
        only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
        Stochastic term tuples can include 2 basis labels to specify
        "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
        strings or integers.  Values are complex coefficients.

    depolarization_parameterization : str of {"depolarize", "stochastic", or "lindblad"}
        Determines whether a DepolarizeOp, StochasticNoiseOp, or LindbladErrorgen
        is used to parameterize the depolarization noise, respectively.
        When "depolarize" (the default), a DepolarizeOp is created with the strength given
        in `depolarization_strengths`. When "stochastic", the depolarization strength is split
        evenly among the stochastic channels of a StochasticOp. When "lindblad", the depolarization
        strength is split evenly among the coefficients of the stochastic error generators
        (which are exponentiated to form a LindbladErrorgen with the "depol" parameterization).

    stochastic_parameterization : str of {"stochastic", or "lindblad"}
        Determines whether a StochasticNoiseOp or LindbladErrorgen is used to parameterize the
        stochastic noise, respectively. When "stochastic", elements of `stochastic_error_probs`
        are used as coefficients in a linear combination of stochastic channels (the default).
        When "lindblad", the elements of `stochastic_error_probs` are coefficients of
        stochastic error generators (which are exponentiated to form a LindbladErrorgen with the
        "cptp" parameterization).

    lindblad_parameterization : "auto" or a LindbladErrorgen paramtype
        Determines the parameterization of the LindbladErrorgen. When "auto" (the default), the parameterization
        is inferred from the types of error generators specified in the `lindblad_error_coeffs` dictionaries.
        When not "auto", the parameterization type is passed through to the LindbladErrorgen.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The simulator used to compute predicted probabilities for the
        resulting :class:`Model`.  Using `"auto"` selects `"matrix"` when there
        are 2 qubits or less, and otherwise selects `"map"`.

    independent_gates : bool, optional
        Whether gates are allowed independent noise or not.  If False,
        then all gates with the same name (e.g. "Gx") will have the *same*
        noise (e.g. an overrotation by 1 degree), and the
        `operation_bks['cloudnoise']` dictionary will contains a single key per gate
        name.  If True, then gates with the same name acting on different
        qubits may have different local noise, and so the
        `operation_bks['cloudnoise']` dictionary contains a key for each gate
         available gate placement.

    independent_spam : bool, optional
        Similar to `indepenent_gates` but for SPAM operations.

    errcomp_type : {'gates', 'errorgens'}
        Whether errors should be combined by composing error maps (`gates`) or by
        exponentiating the sum of error generators (composing the error generators,
        `errorgens`).  The latter is only an option when the noise is given solely
        in terms of Lindblad error coefficients.

    implicit_idle_mode : {'none', 'add_global'}
        The way idel operations are added implicitly within the created model. `"nonw"`
        doesn't add any "extra" idle operations when there is a layer that contains some
        gates but not gates on all the qubits.  `"add_global"` adds the global idle operation,
        i.e., the operation for a global idle layer (zero gates - a completely empty layer),
        to every layer that is simulated, using the global idle as a background idle that always
        occurs regardless of the operation.

    verbosity : int or VerbosityPrinter, optional
        Amount of detail to print to stdout.

    Returns
    -------
    CloudNoiseModel
    """
    modelnoises = []
    if depolarization_strengths is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _DepolarizationNoise(val, depolarization_parameterization)
                                               for lbl, val in depolarization_strengths.items()}))
    if stochastic_error_probs is not None:
        modelnoises.append(_OpModelPerOpNoise({lbl: _StochasticNoise(val, stochastic_parameterization)
                                               for lbl, val in stochastic_error_probs.items()}))
    if lindblad_error_coeffs is not None:

        def process_stencil_labels(flat_lindblad_errs):
            nonlocal_errors = _collections.OrderedDict()
            local_errors = _collections.OrderedDict()

            for nm, val in flat_lindblad_errs.items():
                if isinstance(nm, str): nm = (nm[0], nm[1:])  # e.g. "HXX" => ('H','XX')
                err_typ, basisEls = nm[0], nm[1:]
                sslbls = None
                local_nm = [err_typ]
                for bel in basisEls:  # e.g. bel could be "X:Q0" or "XX:Q0,Q1"
                    # OR "X:<n>" where n indexes a target qubit or "X:<dir>" where dir indicates
                    # a graph *direction*, e.g. "up"
                    if ':' in bel:
                        bel_name, bel_sslbls = bel.split(':')  # should have form <name>:<comma-separated-sslbls>
                        bel_sslbls = bel_sslbls.split(',')  # e.g. ('Q0','Q1')
                        integerized_sslbls = []
                        for ssl in bel_sslbls:
                            try: integerized_sslbls.append(int(ssl))
                            except: integerized_sslbls.append(ssl)
                        bel_sslbls = tuple(integerized_sslbls)
                    else:
                        bel_name = bel
                        bel_sslbls = None

                    if sslbls is None:
                        sslbls = bel_sslbls
                    else:
                        #Note: sslbls should always be the same if there are multiple basisEls,
                        #  i.e for nm == ('S',bel1,bel2)
                        assert(sslbls is bel_sslbls or sslbls == bel_sslbls), \
                            "All basis elements of the same error term must operate on the *same* state!"
                    local_nm.append(bel_name)  # drop the state space labels, e.g. "XY:Q0,Q1" => "XY"

                # keep track of errors by the qubits they act on, as only each such
                # set will have it's own LindbladErrorgen
                local_nm = tuple(local_nm)  # so it's hashable
                if sslbls is not None:
                    sslbls = tuple(sorted(sslbls))
                    if sslbls not in nonlocal_errors:
                        nonlocal_errors[sslbls] = _collections.OrderedDict()
                    if local_nm in nonlocal_errors[sslbls]:
                        nonlocal_errors[sslbls][local_nm] += val
                    else:
                        nonlocal_errors[sslbls][local_nm] = val
                else:
                    if local_nm in local_errors:
                        local_errors[local_nm] += val
                    else:
                        local_errors[local_nm] = val

            if len(nonlocal_errors) == 0:
                return _LindbladNoise(local_errors, lindblad_parameterization)
            else:
                all_errors = []
                if len(local_errors) > 0:
                    all_errors.append((None, _LindbladNoise(local_errors, lindblad_parameterization)))
                for sslbls, errdict in nonlocal_errors.items():
                    all_errors.append((sslbls, _LindbladNoise(errdict, lindblad_parameterization)))
                return _collections.OrderedDict(all_errors)

        modelnoises.append(_OpModelPerOpNoise({lbl: process_stencil_labels(val)
                                               for lbl, val in lindblad_error_coeffs.items()}))

    return _create_cloud_crosstalk_model(processor_spec, _ComposedOpModelNoise(modelnoises), custom_gates, evotype,
                                         simulator, independent_gates, independent_spam, errcomp_type,
                                         implicit_idle_mode, verbosity)


def _create_cloud_crosstalk_model(processor_spec, modelnoise, custom_gates=None,
                                  evotype="default", simulator="auto", independent_gates=False,
                                  independent_spam=True, errcomp_type="errorgens",
                                  implicit_idle_mode="add_global", verbosity=0):
    """
    Create a n-qubit "cloud-crosstalk" model.

    Similar to :method:`create_cloud_crosstalk_model` but the noise is input more generally,
    as a :class:`ModelNoise` object.  Arguments are the same as this function except that
    `modelnoise` is given instead of several more specific noise-describing arguments.

    Returns
    -------
    CloudNoiseModel
    """
    qubit_labels = processor_spec.qubit_labels
    state_space = _statespace.QubitSpace(qubit_labels)  # FUTURE: allow other types of state spaces somehow?
    evotype = _Evotype.cast(evotype)
    modelnoise = _OpModelNoise.cast(modelnoise)
    modelnoise.reset_access_counters()
    printer = _VerbosityPrinter.create_printer(verbosity)

    #Create static ideal gates without any noise (we use `modelnoise` further down)
    gatedict = _setup_local_gates(processor_spec, evotype, None, custom_gates,
                                  ideal_gate_type=('static standard', 'static clifford', 'static unitary'))
    stencils = _collections.OrderedDict()

    # (Note: global idle is now processed with other processorspec gates)

    # SPAM
    local_noise = False
    prep_layers, povm_layers = _create_spam_layers(processor_spec, modelnoise, local_noise,
                                                   'computational', 'computational', evotype, state_space,
                                                   independent_gates, independent_spam)

    if errcomp_type == 'gates':
        create_stencil_fn = modelnoise.create_errormap_stencil
        apply_stencil_fn = modelnoise.apply_errormap_stencil
    elif errcomp_type == 'errorgens':
        create_stencil_fn = modelnoise.create_errorgen_stencil
        apply_stencil_fn = modelnoise.apply_errorgen_stencil
    else:
        raise ValueError("Invalid `errcomp_type` value: %s" % str(errcomp_type))

    def build_cloudnoise_fn(lbl):
        # lbl will be for a particular gate and target qubits.  If we have error rates for this specific gate
        # and target qubits (i.e this primitive layer op) then we should build it directly (and independently,
        # regardless of the value of `independent_gates`) using these rates.  Otherwise, if we have a stencil
        # for this gate, then we should use it to construct the output, using a copy when gates are independent
        # and a reference to the *same* stencil operations when `independent_gates==False`.

        num_sslbls = len(lbl.sslbls) if (lbl.sslbls is not None) else None
        if lbl in modelnoise:
            stencil = create_stencil_fn(lbl, evotype, state_space, num_target_labels=num_sslbls)
        elif lbl.name in stencils:
            stencil = stencils[lbl.name]
        elif lbl.name in modelnoise:
            stencils[lbl.name] = create_stencil_fn(lbl.name, evotype, state_space, num_target_labels=num_sslbls)
            stencil = stencils[lbl.name]
        else:
            return None  # no cloudnoise error for this label

        return apply_stencil_fn(stencil, evotype, state_space, target_labels=lbl.sslbls,
                                qubit_graph=processor_spec.qubit_graph,
                                copy=independent_gates and (lbl not in modelnoise))  # no need to copy if first case

    def build_cloudkey_fn(lbl):
        num_sslbls = len(lbl.sslbls) if (lbl.sslbls is not None) else None
        if lbl in modelnoise:
            stencil = create_stencil_fn(lbl, evotype, state_space, num_target_labels=num_sslbls)
        elif lbl.name in stencils:
            stencil = stencils[lbl.name]
        elif lbl.name in modelnoise:
            stencils[lbl.name] = create_stencil_fn(lbl.name, evotype, state_space, num_target_labels=num_sslbls)
            stencil = stencils[lbl.name]
        else:
            # simple cloud-key when there is no cloud noise
            return tuple(lbl.sslbls) if (lbl.sslbls is not None) else qubit_labels

        #Otherwise, process stencil to get a list of all the qubit labels `lbl`'s cloudnoise error
        # touches and form this into a key
        cloud_sslbls = modelnoise.compute_stencil_absolute_sslbls(stencil, state_space, lbl.sslbls,
                                                                  processor_spec.qubit_graph)
        hashable_sslbls = tuple(lbl.sslbls) if (lbl.sslbls is not None) else qubit_labels
        cloud_key = (hashable_sslbls, tuple(sorted(cloud_sslbls)))  # (sets are unhashable)
        return cloud_key

    ret = _CloudNoiseModel(processor_spec, gatedict, prep_layers, povm_layers,
                           build_cloudnoise_fn, build_cloudkey_fn,
                           simulator, evotype, errcomp_type,
                           implicit_idle_mode, printer)
    modelnoise.warn_about_zero_counters()  # must do this after model creation so build_ fns have been run
    return ret


def create_cloud_crosstalk_model_from_hops_and_weights(
        processor_spec, custom_gates=None,
        max_idle_weight=1, max_spam_weight=1,
        maxhops=0, extra_weight_1_hops=0, extra_gate_weight=0,
        simulator="auto", evotype='default',
        gate_type="H+S", spam_type="H+S",
        implicit_idle_mode="add_global", errcomp_type="gates",
        independent_gates=True, independent_spam=True,
        connected_highweight_errors=True,
        verbosity=0):
    """
    Create a "cloud crosstalk" model based on maximum error weights and hops along the processor's qubit graph.

    This function provides a convenient way to construct cloud crosstalk models whose gate errors
    consist of Pauli elementary error generators (i.e. that correspond to Lindblad error coefficients)
    that are limited in weight (number of non-identity Paulis) and support (which qubits have non-trivial
    Paulis on them).  Errors are taken to be approximately local, meaning they are concentrated near the
    target qubits of a gate, with the notion of locality taken from the processor specification's qubit graph.
    The caller provides maximum-weight, maximum-hop (a "hop" is the movement along a single graph edge), and
    gate type arguments to specify the set of possible errors on a gate.

    - The global idle gate (corresponding to an empty circuit layer) has errors that are limited only by
      a maximum weight, `max_idle_weight`.
    - State preparation and POVM errors are constructed similarly, with a global-idle-like error following
      or preceding the preparation or measurement, respectively.
    - Gate errors are placed on all the qubits that can be reached with at most `maxhops` hops from (any of)
      the gate's target qubits.  Elementary error generators up to weight `W`, where `W` equals the number
      of target qubits (e.g., 2 for a CNOT gate) plus `extra_gate_weight` are allowed.  Weight-1 terms
      are a special case, and the `extra_weight_1_hops` argument adds to the usual `maxhops` in this case
      to allow weight-1 errors on a possibly larger region of qubits around the target qubits.

    Parameters
    ----------
    processor_spec : ProcessorSpec
        The processor specification to create a model for.  This object specifies the
        gate names and unitaries for the processor, and their availability on the
        processor.

    custom_gates : dict
        A dictionary that associates with gate labels
        :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
        objects.  These objects describe the full action of the gate or
        primitive-layer they're labeled by (so if the model represents
        states by density matrices these objects are superoperators, not
        unitaries), and override any standard construction based on builtin
        gate names or `nonstd_gate_unitaries`.  Keys of this dictionary must
        be string-type gate *names* -- they cannot include state space labels
        -- and they must be *static* (have zero parameters) because they
        represent only the ideal behavior of each gate -- the cloudnoise
        operations represent the parameterized noise.  To fine-tune how this
        noise is parameterized, call the :class:`CloudNoiseModel` constructor
        directly.

    max_idle_weight : int, optional
        The maximum-weight for errors on the global idle gate.

    max_spam_weight : int, optional
        The maximum-weight for state preparation and measurement (SPAM) errors.

    maxhops : int
        The locality constraint: for a gate, errors (of weight up to the
        maximum weight for the gate) are allowed to occur on the gate's
        target qubits and those reachable by hopping at most `maxhops` times
        from a target qubit along nearest-neighbor links (defined by the
        `geometry`).

    extra_weight_1_hops : int, optional
        Additional hops (adds to `maxhops`) for weight-1 errors.  A value > 0
        can be useful for allowing just weight-1 errors (of which there are
        relatively few) to be dispersed farther from a gate's target qubits.
        For example, a crosstalk-detecting model might use this.

    extra_gate_weight : int, optional
        Addtional weight, beyond the number of target qubits (taken as a "base
        weight" - i.e. weight 2 for a 2Q gate), allowed for gate errors.  If
        this equals 1, for instance, then 1-qubit gates can have up to weight-2
        errors and 2-qubit gates can have up to weight-3 errors.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The circuit simulator used to compute any
        requested probabilities, e.g. from :method:`probs` or
        :method:`bulk_probs`.  Using `"auto"` selects `"matrix"` when there
        are 2 qubits or less, and otherwise selects `"map"`.

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    gate_type : str, optional
        The Lindblad-error parameterization type used for gate operations.  This
        may be expanded in the future, but currently the gate errors *must* be of
        the Lindblad error-generator coefficients type, and this argument specifies
        what elementary error-generator coefficients are initially allowed (and linked to
        model parameters), before maximum-weight and locality constraints are imposed.
        In addition to the usual Lindblad error types, (e.g. `"H"`, `"H+S"`) the special
        values `"none"` is allowed to indicate that there should be no errors on the gates
        (useful if you only want errors on the SPAM, for instance).

    spam_type : str, optional
        Similar to `gate_type` but for SPAM elements (state preparations
        and POVMs).  This specifies the Lindblad-error parameterization for the
        state prepearation and POVM.

    implicit_idle_mode : {'none', 'add_global'}
        The way idel operations are added implicitly within the created model. `"nonw"`
        doesn't add any "extra" idle operations when there is a layer that contains some
        gates but not gates on all the qubits.  `"add_global"` adds the global idle operation,
        i.e., the operation for a global idle layer (zero gates - a completely empty layer),
        to every layer that is simulated, using the global idle as a background idle that always
        occurs regardless of the operation.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the created
        model.  `"gates"` means that the errors on multiple gates in a single
        layer are composed as separate and subsequent processes.  Specifically,
        the layer operation has the form `Composed(target,idleErr,cloudErr)`
        where `target` is a composition of all the ideal gate operations in the
        layer, `idleErr` is the global idle error if `implicit_idle_mode == 'add_global'`,
        and `cloudErr` is the composition (ordered as layer-label) of cloud-
        noise contributions, i.e. a map that acts as the product of exponentiated
        error-generator matrices.  `"errorgens"` means that layer operations
        have the form `Composed(target, error)` where `target` is as above and
        `error` results from composing (summing) the idle and cloud-noise error
        *generators*, i.e. a map that acts as the exponentiated sum of error
        generators (ordering is irrelevant in this case).

    independent_gates : bool, optional
        Whether the noise added to a gate when it acts on one set of target
        qubits is independent of its noise on a different set of target qubits.
        If False, then all gates with the same name (e.g. "Gx") will be constrained
        to having the *same* noise on the cloud around the target qubits (even though
        the target qubits and cloud are different).  If True, then gate noise operations
        for different sets of target qubits are independent.

    independent_spam : bool, optional
        Similar to `independent_gates` but for state preparation and measurement operations.
        When `False`, the noise applied to each set (individual or pair or triple etc.) of
        qubits must be the same, e.g., if the state preparation is a perfect preparation followed
        by a single-qubit rotation then this rotation must be by the *same* angle on all of
        the qubits.

    connected_highweight_errors : bool, optional
        An additional constraint regarding high-weight errors.  When `True`, only high weight
        (weight 2+) elementary error generators whose non-trivial Paulis occupy a *connected*
        portion of the qubit graph are allowed.  For example, if the qubit graph is a 1D chain
        of 4 qubits, 1-2-3-4, and weight-2 errors are allowed on a single-qubit gate with
        target = qubit-2, then weight-2 errors on 1-2 and 2-3 would be allowed, but errors on
        1-3 would be forbidden.  When `False`, no constraint is imposed.

    verbosity : int or VerbosityPrinter, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    CloudNoiseModel
    """

    # construct noise specifications for the cloudnoise model
    modelnoise = {}
    all_qubit_labels = processor_spec.qubit_labels
    conn = connected_highweight_errors  # shorthand: whether high-weight errors must be connected on the graph
    global_idle_name = processor_spec.global_idle_gate_name

    # Global Idle
    if max_idle_weight > 0:
        assert(global_idle_name is not None), \
            "`max_idle_weight` must equal 0 for processor specs without a global idle gate!"
        #printer.log("Creating Idle:")
        wt_maxhop_tuples = [(i, None) for i in range(1, max_idle_weight + 1)]
        modelnoise[global_idle_name] = _build_weight_maxhops_modelnoise(all_qubit_labels, wt_maxhop_tuples,
                                                                        gate_type, conn)

    # SPAM
    if max_spam_weight > 0:
        wt_maxhop_tuples = [(i, None) for i in range(1, max_spam_weight + 1)]
        modelnoise['prep'] = _build_weight_maxhops_modelnoise(all_qubit_labels, wt_maxhop_tuples, spam_type, conn)
        modelnoise['povm'] = _build_weight_maxhops_modelnoise(all_qubit_labels, wt_maxhop_tuples, spam_type, conn)

    # Gates
    weight_maxhops_tuples_1Q = [(1, maxhops + extra_weight_1_hops)] + \
                               [(1 + x, maxhops) for x in range(1, extra_gate_weight + 1)]

    weight_maxhops_tuples_2Q = [(1, maxhops + extra_weight_1_hops), (2, maxhops)] + \
                               [(2 + x, maxhops) for x in range(1, extra_gate_weight + 1)]

    for gatenm, gate_unitary in processor_spec.gate_unitaries.items():
        if gatenm == global_idle_name: continue  # processed above
        gate_nQubits = int(gate_unitary) if isinstance(gate_unitary, (int, _np.int64)) \
            else int(round(_np.log2(gate_unitary.shape[0])))  # NOTE: integer gate_unitary => idle on n qubits
        if gate_nQubits not in (1, 2):
            raise ValueError("Only 1- and 2-qubit gates are supported.  %s acts on %d qubits!"
                             % (str(gatenm), gate_nQubits))
        weight_maxhops_tuples = weight_maxhops_tuples_1Q if gate_nQubits == 1 else weight_maxhops_tuples_2Q
        target_sslbls = ('@0',) if gate_nQubits == 1 else ('@0', '@1')
        modelnoise[gatenm] = _build_weight_maxhops_modelnoise(target_sslbls, weight_maxhops_tuples,
                                                              gate_type, conn)

    return _create_cloud_crosstalk_model(processor_spec, modelnoise, custom_gates,
                                         evotype, simulator, independent_gates, independent_spam,
                                         errcomp_type, implicit_idle_mode, verbosity)


def _iter_basis_inds(weight):
    """ Iterate over product of `weight` non-identity Pauli 1Q basis indices """
    basisIndList = [[1, 2, 3]] * weight  # assume pauli 1Q basis, and only iterate over non-identity els
    for basisInds in _itertools.product(*basisIndList):
        yield basisInds


def _pauli_product_matrix(sigma_inds):
    """
    Construct the Pauli product matrix from the given `sigma_inds`

    Parameters
    ----------
    sigma_inds : iterable
        A sequence of integers in the range [0,3] corresponding to the
        I, X, Y, Z Pauli basis matrices.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
    """
    sigmaVec = (id2x2 / sqrt2, sigmax / sqrt2, sigmay / sqrt2, sigmaz / sqrt2)
    M = _np.identity(1, 'complex')
    for i in sigma_inds:
        M = _np.kron(M, sigmaVec[i])
    return M


def _construct_restricted_weight_pauli_basis(wt, sparse=False):
    basisEl_Id = _pauli_product_matrix(_np.zeros(wt, _np.int64))
    errbasis = [basisEl_Id]
    errbasis_lbls = ['I']
    for err_basis_inds in _iter_basis_inds(wt):
        error = _np.array(err_basis_inds, _np.int64)  # length == wt
        basisEl = _pauli_product_matrix(error)
        errbasis.append(basisEl)
        errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))

    #printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds, len(errbasis)), 3)
    return _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse)


def _build_weight_maxhops_modelnoise(target_sslbls, weight_maxhops_tuples, lnd_parameterization, connected=True):

    # This function:
    # loop over all size-`wt` *connected* combinations, `err_qubit_inds`, of the qubit indices in
    #   `possible_err_qubit_inds`
    #   - construct a local weight-`wt` Pauli basis & corresponding LindbladErrorgen on `wt` qubits
    #       => replace with: opnoise.create_errorgen(evotype, state_space=None)  where opnoise is for a wt-qubit op
    #   - embed this constructed local error onto `err_qubit_inds`
    #   - append embedded error onto running list
    #
    # Noise object structure:
    #  OpModelPerOpNoise( { op_key/'idle': { sslbls : opnoise } } )
    #     where sslbls can be absolute labels or stencil labels
    # -- could have a fn that spreads a single opnoise onto all the sslbls
    #      given by size-`wt` connected combos of `possible_err_qubit_inds` - this would work for independent clouds
    # -- have LindbladNoiseDict and another LindbladPauliAtWeight (?) noise objects,
    #     since we want to specify a lindblad noise by giving a weight and an initial basis (Pauli here)

    # To build a cloudnoise model from hops & weights:
    modelnoise_dict = {}
    if lnd_parameterization == 'none' or lnd_parameterization is None:
        return {}  # special case when we don't want any error parameterization

    for wt, max_hops in weight_maxhops_tuples:
        if max_hops is None or max_hops == 0:  # Note: maxHops not used in this case
            stencil_lbl = _stencil.StencilLabelAllCombos(target_sslbls, wt, connected)
        else:
            stencil_lbl = _stencil.StencilLabelRadiusCombos(target_sslbls, max_hops, wt, connected)

        local_state_space = _statespace.default_space_for_num_qubits(wt)
        modelnoise_dict[stencil_lbl] = _LindbladNoise.from_basis_coefficients(
            lnd_parameterization, _construct_restricted_weight_pauli_basis(wt),
            local_state_space)
    return modelnoise_dict


@_deprecated_fn("This function is overly specific and will be removed soon.")
def _nparams_xycnot_cloudnoise_model(num_qubits, geometry="line", max_idle_weight=1, maxhops=0,
                                     extra_weight_1_hops=0, extra_gate_weight=0, require_connected=False,
                                     independent_1q_gates=True, zz_only=False, bidirectional_cnots=True, verbosity=0):
    """
    Compute the number of parameters in a particular :class:`CloudNoiseModel`.

    Returns the number of parameters in the :class:`CloudNoiseModel` containing
    X(pi/2), Y(pi/2) and CNOT gates using the specified arguments without
    actually constructing the model (useful for considering parameter-count
    scaling).

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object may be passed directly.

    max_idle_weight : int, optional
        The maximum-weight for errors on the global idle gate.

    maxhops : int
        The locality constraint: for a gate, errors (of weight up to the
        maximum weight for the gate) are allowed to occur on the gate's
        target qubits and those reachable by hopping at most `maxhops` times
        from a target qubit along nearest-neighbor links (defined by the
        `geometry`).

    extra_weight_1_hops : int, optional
        Additional hops (adds to `maxhops`) for weight-1 errors.  A value > 0
        can be useful for allowing just weight-1 errors (of which there are
        relatively few) to be dispersed farther from a gate's target qubits.
        For example, a crosstalk-detecting model might use this.

    extra_gate_weight : int, optional
        Addtional weight, beyond the number of target qubits (taken as a "base
        weight" - i.e. weight 2 for a 2Q gate), allowed for gate errors.  If
        this equals 1, for instance, then 1-qubit gates can have up to weight-2
        errors and 2-qubit gates can have up to weight-3 errors.

    require_connected : bool, optional
        If True, then high-weight errors only occur on connected (via `geometry`) qubits.
        For example in a line of qubits there would not be weight-2 errors on qubits 1 and 3.

    independent_1q_gates : bool, optional
        If True, 1Q gates on different qubits have separate (distinct) parameters.  If
        False, the 1Q gates of each type (e.g. an pi/2 X gate) for different qubits share
        the same set of parameters.

    zz_only : bool, optional
        If True, the only high-weight errors allowed are of "Z^n" type.

    bidirectional_cnots : bool
        Whether CNOT gates can be performed in either direction (and each direction should
        be treated as an indepedent gate)

    verbosity : int, optional
        An integer >= 0 dictating how much output to send to stdout.

    Returns
    -------
    int
    """
    # noise can be either a seed or a random array that is long enough to use

    printer = _VerbosityPrinter.create_printer(verbosity)
    printer.log("Computing parameters for a %d-qubit %s model" % (num_qubits, geometry))

    qubitGraph = _QubitGraph.common_graph(num_qubits, geometry, directed=True, all_directions=True)
    #printer.log("Created qubit graph:\n"+str(qubitGraph))

    def idle_count_nparams(max_weight):
        """Parameter count of a `build_nqn_global_idle`-constructed gate"""
        ret = 0
        possible_err_qubit_inds = _np.arange(num_qubits)
        for wt in range(1, max_weight + 1):
            nErrTargetLocations = qubitGraph.connected_combos(possible_err_qubit_inds, wt)
            if zz_only and wt > 1: basisSizeWoutId = 1**wt  # ( == 1)
            else: basisSizeWoutId = 3**wt  # (X,Y,Z)^wt
            nErrParams = 2 * basisSizeWoutId  # H+S terms
            ret += nErrTargetLocations * nErrParams
        return ret

    def op_count_nparams(target_qubit_inds, weight_maxhops_tuples, debug=False):
        """Parameter count of a `build_nqn_composed_gate`-constructed gate"""
        ret = 0
        #Note: no contrib from idle noise (already parameterized)
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops), _np.int64)
            if require_connected:
                nErrTargetLocations = qubitGraph.connected_combos(possible_err_qubit_inds, wt)
            else:
                nErrTargetLocations = _scipy.special.comb(len(possible_err_qubit_inds), wt)
            if zz_only and wt > 1: basisSizeWoutId = 1**wt  # ( == 1)
            else: basisSizeWoutId = 3**wt  # (X,Y,Z)^wt
            nErrParams = 2 * basisSizeWoutId  # H+S terms
            if debug:
                print(" -- wt%d, hops%d: inds=%s locs = %d, eparams=%d, total contrib = %d" %
                      (wt, maxHops, str(possible_err_qubit_inds), nErrTargetLocations,
                       nErrParams, nErrTargetLocations * nErrParams))
            ret += nErrTargetLocations * nErrParams
        return ret

    nParams = _collections.OrderedDict()

    printer.log("Creating Idle:")
    nParams[_label.Label('Gi')] = idle_count_nparams(max_idle_weight)

    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    weight_maxhops_tuples_1Q = [(1, maxhops + extra_weight_1_hops)] + \
                               [(1 + x, maxhops) for x in range(1, extra_gate_weight + 1)]

    if independent_1q_gates:
        for i in range(num_qubits):
            printer.log("Creating 1Q X(pi/2) and Y(pi/2) gates on qubit %d!!" % i)
            nParams[_label.Label("Gx", i)] = op_count_nparams((i,), weight_maxhops_tuples_1Q)
            nParams[_label.Label("Gy", i)] = op_count_nparams((i,), weight_maxhops_tuples_1Q)
    else:
        printer.log("Creating common 1Q X(pi/2) and Y(pi/2) gates")
        rep = int(num_qubits / 2)
        nParams[_label.Label("Gxrep")] = op_count_nparams((rep,), weight_maxhops_tuples_1Q)
        nParams[_label.Label("Gyrep")] = op_count_nparams((rep,), weight_maxhops_tuples_1Q)

    #2Q gates: CNOT gates along each graph edge
    weight_maxhops_tuples_2Q = [(1, maxhops + extra_weight_1_hops), (2, maxhops)] + \
                               [(2 + x, maxhops) for x in range(1, extra_gate_weight + 1)]
    seen_pairs = set()
    for i, j in qubitGraph.edges():  # note: all edges have i<j so "control" of CNOT is always lower index (arbitrary)
        if bidirectional_cnots is False:
            ordered_tup = (i, j) if i <= j else (j, i)
            if ordered_tup in seen_pairs: continue
            else: seen_pairs.add(ordered_tup)

        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i, j))
        nParams[_label.Label("Gcnot", (i, j))] = op_count_nparams((i, j), weight_maxhops_tuples_2Q)

    #SPAM
    nPOVM_1Q = 4  # params for a single 1Q POVM
    nParams[_label.Label('rho0')] = 3 * num_qubits  # 3 b/c each component is TP
    nParams[_label.Label('Mdefault')] = nPOVM_1Q * num_qubits  # num_qubits 1Q-POVMs

    return nParams, sum(nParams.values())
