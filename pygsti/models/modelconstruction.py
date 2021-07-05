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

import numpy as _np
import scipy.linalg as _spl
import scipy.sparse as _sps

from pygsti.evotypes import Evotype as _Evotype
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state
from pygsti.modelmembers.operations import opfactory as _opfactory
from pygsti.models import explicitmodel as _emdl
from pygsti.models import gaugegroup as _gg
from pygsti.models.localnoisemodel import LocalNoiseModel as _LocalNoiseModel
from pygsti.models.cloudnoisemodel import CloudNoiseModel as _CloudNoiseModel
from pygsti.baseobjs import label as _label, statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.basis import ExplicitBasis as _ExplicitBasis
from pygsti.baseobjs.basis import DirectSumBasis as _DirectSumBasis
from pygsti.tools import basistools as _bt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import optools as _ot
from pygsti.tools import listtools as _lt
from pygsti.baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.tools.legacytools import deprecate as _deprecated_fn
from pygsti.circuits.circuitparser import CircuitParser as _CircuitParser


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
    #TODO REMOVE
    #opDim = basis.dim
    #vecInReducedStdBasis = _np.zeros( (opDim,1), 'd' ) # assume index given as vec_expr refers to a
    #                                                     #Hilbert-space state index, so "reduced-std" basis
    #blockDims = [int(_np.sqrt(opDim))] # FIX - get block dims from basis?

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

    #TODO REMOVE
    #start = 0; vecIndex = 0
    #for blockDim in blockDims:
    #    for i in range(start,start+blockDim):
    #        for j in range(start,start+blockDim):
    #            if (i,j) == (index,index):
    #                vecInReducedStdBasis[ vecIndex, 0 ] = 1.0  #set diagonal element of density matrix
    #                break
    #            vecIndex += 1
    #    start += blockDim
    #from ..baseobjs.basis import BuiltinBasis
    #hackstd = BuiltinBasis('std',opDim)
    #return _bt.change_basis(vecInReducedStdBasis, hackstd, basis)

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
    #dmDim, opDim, blockDims = basis.dim REMOVE
    #fullOpDim = dmDim**2

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

    #print "DB: dim = ",dim, " dmDim = ",dmDim
    opInFinalBasis = None  # what will become the final operation matrix
    # defaultI2P = "all" if parameterization != "linearTP" else "TP"
    #default indices to parameterize (I2P) - used only when
    # creating parameterized gates

    opTermsInFinalBasis = []
    exprTerms = op_expr.split(':')
    for exprTerm in exprTerms:

        l = exprTerm.index('('); r = exprTerm.rindex(')')
        opName = exprTerm[0:l]
        argsStr = exprTerm[l + 1:r]
        args = argsStr.split(',')

        if opName == "I":
            # qubit labels (TODO: what about 'L' labels? -- not sure if they work with this...)
            labels = to_labels(args)
            stateSpaceDim = int(_np.product([state_space.label_dimension(l) for l in labels]))
            # *real* 4x4 mx in Pauli-product basis -- still just the identity!
            pp_opMx = _op.StaticArbitraryOp(_np.identity(stateSpaceDim, 'd'), evotype=evotype)
            opTermInFinalBasis = _op.EmbeddedOp(state_space, labels, pp_opMx)

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

            Uop = _spl.expm(ex)  # 2x2 unitary matrix operating on single qubit in [0,1] basis
            # complex 4x4 mx operating on vectorized 1Q densty matrix in std basis
            operationMx = _ot.unitary_to_process_mx(Uop)
            # *real* 4x4 mx in Pauli-product basis -- better for parameterization
            pp_opMx = _op.StaticArbitraryOp(_bt.change_basis(operationMx, 'std', 'pp'), evotype, state_space=None)
            opTermInFinalBasis = _op.EmbeddedOp(state_space, [label], pp_opMx)

        elif opName == 'N':  # more general single-qubit gate
            assert(len(args) == 5)  # theta, sigmaX-coeff, sigmaY-coeff, sigmaZ-coeff, qubit-index
            theta = eval(args[0], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            sxCoeff = eval(args[1], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            syCoeff = eval(args[2], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            szCoeff = eval(args[3], {"__builtins__": None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            label = to_label(args[4])
            assert(state_space.label_dimension(label) == 4), "%s gate must act on qubits!" % opName

            ex = -1j * theta * (sxCoeff * sigmax / 2. + syCoeff * sigmay / 2. + szCoeff * sigmaz / 2.)
            Uop = _spl.expm(ex)  # 2x2 unitary matrix operating on single qubit in [0,1] basis
            # complex 4x4 mx operating on vectorized 1Q densty matrix in std basis
            operationMx = _ot.unitary_to_process_mx(Uop)
            # *real* 4x4 mx in Pauli-product basis -- better for parameterization
            pp_opMx = _op.StaticArbitraryOp(_bt.change_basis(operationMx, 'std', 'pp'), evotype, state_space=None)
            opTermInFinalBasis = _op.EmbeddedOp(state_space, [label], pp_opMx)

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
            Uop = _np.identity(4, 'complex'); Uop[2:, 2:] = Utarget
            assert(state_space.label_dimension(label1) == 4 and state_space.label_dimension(label2) == 4), \
                "%s gate must act on qubits!" % opName

            # complex 16x16 mx operating on vectorized 2Q densty matrix in std basis
            operationMx = _ot.unitary_to_process_mx(Uop)
            # *real* 16x16 mx in Pauli-product basis -- better for parameterization
            pp_opMx = _op.StaticArbitraryOp(_bt.change_basis(operationMx, 'std', 'pp'), evotype, state_space=None)
            opTermInFinalBasis = _op.EmbeddedOp(state_space, [label1, label2], pp_opMx)

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

            opMxInFinalBasis = _bt.change_basis(opTermInReducedStdBasis, std_basis, basis)
            opTermInFinalBasis = _op.FullArbitraryOp(opMxInFinalBasis, evotype, state_space)

        else: raise ValueError("Invalid gate name: %s" % opName)

        opTermsInFinalBasis.append(opTermInFinalBasis)

    opInFinalBasis = opTermsInFinalBasis[0] if len(opTermsInFinalBasis) == 1 \
        else _op.ComposedOp(list(reversed(opTermsInFinalBasis)))
    #Note: expressions are listed in "matrix composition order" (reverse for ComposedOp)

    finalOpMx = opInFinalBasis.to_dense(on_space='HilbertSchmidt')
    if basis.real:
        assert(_np.linalg.norm(finalOpMx.imag) < 1e-6), "Operation matrix should be real but isn't!"
        finalOpMx = _np.real(finalOpMx)

    if parameterization == "full":
        return _op.FullArbitraryOp(finalOpMx, evotype, state_space)
    if parameterization == "static":
        return _op.StaticArbitraryOp(finalOpMx, evotype, state_space)
    if parameterization == "TP":
        return _op.FullTPOp(finalOpMx, evotype, state_space)

    raise ValueError("Invalid 'parameterization' parameter: "
                     "%s (must by 'full', 'TP', 'static')"
                     % parameterization)


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


def _create_explicit_model_from_expessions(state_space, basis,
                                           op_labels, op_expressions,
                                           prep_labels=('rho0',), prep_expressions=('0',),
                                           effect_labels='standard', effect_expressions='standard',
                                           povm_labels='Mdefault', parameterization="full", evotype='default'):
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

    ret = _emdl.ExplicitOpModel(state_space, basis.copy(), default_param=parameterization,
                                evotype=evotype)
    #prep_prefix="rho", effect_prefix="E", gate_prefix="G")

    for label, rhoExpr in zip(prep_labels, prep_expressions):
        vec = _basis_create_spam_vector(rhoExpr, basis)
        if parameterization == "full":
            ret.preps[label] = _state.FullState(vec, evotype, state_space)
        elif parameterization == "TP":
            ret.preps[label] = _state.TPState(vec, evotype, state_space)
        elif parameterization == "static":
            ret.preps[label] = _state.StaticState(vec, evotype, state_space)
        else:
            raise ValueError("Invalid parameterization: %s" % parameterization)

    if isinstance(povm_labels, str):
        povm_labels = [povm_labels]
        effect_labels = [effect_labels]
        effect_expressions = [effect_expressions]

    dmDim = int(_np.sqrt(basis.dim))  # "densitymx" evotype assumed... FIX?
    for povmLbl, ELbls, EExprs in zip(povm_labels,
                                      effect_labels, effect_expressions):
        effects = []

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

        for label, EExpr in zip(ELbls, EExprs):
            evec = _basis_create_spam_vector(EExpr, basis)
            if parameterization == "static":
                effects.append((label, _povm.StaticPOVMEffect(evec, evotype)))
            else:
                effects.append((label, _povm.FullPOVMEffect(evec, evotype)))

        if len(effects) > 0:  # don't add POVMs with 0 effects
            if parameterization == "TP":
                ret.povms[povmLbl] = _povm.TPPOVM(effects)
            else:
                ret.povms[povmLbl] = _povm.UnconstrainedPOVM(effects)

    for (opLabel, opExpr) in zip(op_labels, op_expressions):
        ret.operations[opLabel] = _basis_create_operation(state_space, opExpr,
                                                          basis, parameterization, evotype)

    if parameterization == "full":
        ret.default_gauge_group = _gg.FullGaugeGroup(ret.state_space, evotype)
    elif parameterization == "TP":
        ret.default_gauge_group = _gg.TPGaugeGroup(ret.state_space, evotype)
    else:
        ret.default_gauge_group = None  # assume no gauge freedom

    return ret


def create_explicit_model_from_expressions(state_space,
                                           op_labels, op_expressions,
                                           prep_labels=('rho0',), prep_expressions=('0',),
                                           effect_labels='standard', effect_expressions='standard',
                                           povm_labels='Mdefault', basis="auto", parameterization="full",
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

    return _create_explicit_model_from_expessions(state_space,
                                                  _Basis.cast(basis, state_space),
                                                  op_labels, op_expressions,
                                                  prep_labels, prep_expressions,
                                                  effect_labels, effect_expressions,
                                                  povm_labels, parameterization=parameterization,
                                                  evotype=evotype)


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
        modelnoises.append(OpModelPerOpNoise({lbl: DepolarizationNoise(val, depolarization_parameterization)
                                              for lbl, val in depolarization_strengths.items()}))
    if stochastic_error_probs is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: StochasticNoise(val, stochastic_parameterization)
                                              for lbl, val in stochastic_error_probs.items()}))
    if lindblad_error_coeffs is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: LindbladNoise(val, lindblad_parameterization)
                                              for lbl, val in lindblad_error_coeffs.items()}))

    return _create_explicit_model(processor_spec, ComposedOpModelNoise(modelnoises), custom_gates, evotype,
                                  simulator, ideal_gate_type, ideal_spam_type, embed_gates, basis)


def _create_explicit_model(processor_spec, modelnoise, custom_gates=None, evotype="default", simulator="auto",
                           ideal_gate_type='auto', ideal_spam_type='computational', embed_gates=False, basis='pp'):
    qubit_labels = processor_spec.qubit_labels
    state_space = _statespace.QubitSpace(qubit_labels)
    evotype = _Evotype.cast(evotype)
    modelnoise = OpModelNoise.cast(modelnoise)

    if custom_gates is None:
        custom_gates = {}

    if ideal_gate_type == "auto":
        ideal_gate_type = ('static standard', 'static clifford', 'static unitary')

    def _embed_unitary(statespace, target_labels, unitary):
        dummyop = _op.EmbeddedOp(statespace, target_labels,
                                 _op.StaticUnitaryOp(unitary, basis='pp', evotype="statevec_slow"))  # basis hardcode?
        return dummyop.to_dense("Hilbert")

    local_gates = _setup_local_gates(processor_spec, evotype, None, {}, ideal_gate_type)  # no custom *local* gates
    ret = _emdl.ExplicitOpModel(state_space, basis, default_param=ideal_gate_type, evotype=evotype,
                                simulator=simulator)

    for gn, gate_unitary in processor_spec.gate_unitaries.items():

        gate_is_factory = callable(gate_unitary)
        resolved_avail = processor_spec.resolved_availability(gn)

        if callable(resolved_avail) or resolved_avail == '*':
            assert (embed_gates), "Cannot create factories with `embed_gates=False` yet!"
            key = _label.Label(gn)
            allowed_sslbls_fn = resolved_avail if callable(resolved_avail) else None
            gate_nQubits = processor_spec.gate_number_of_qubits(gn)
            ideal_factory = _opfactory.EmbeddingOpFactory(
                state_space, local_gates[gn], num_target_labels=gate_nQubits, allowed_sslbls_fn=allowed_sslbls_fn)
            noiseop = modelnoise.create_errormap(key, evotype, state_space)  # No target indices... just local errs?
            factory = ideal_factory if (noiseop is None) else _op.ComposedOpFactory([ideal_factory, noiseop])
            ret.factories[key] = factory

        else:  # resolved_avail is a list/tuple of available sslbls for the current gate/factory
            for inds in resolved_avail:  # inds are target qubit labels
                key = _label.Label(gn, inds)

                if key in custom_gates:  # allow custom_gates to specify gate elements directly
                    if isinstance(custom_gates[key], _opfactory.OpFactory):
                        ret.factories[key] = custom_gates[key]
                    else:
                        ret.operations[key] = custom_gates[key]
                    continue

                if gate_is_factory:
                    assert(embed_gates), "Cannot create factories with `embed_gates=False` yet!"
                    # TODO: check for modelnoise on *local* factory, i.e. create_errormap(gn, ...)??
                    ideal_factory = _opfactory.EmbeddedOpFactory(state_space, inds, local_gates[gn])
                    noiseop = modelnoise.create_errormap(key, evotype, state_space, target_labels=inds)
                    factory = ideal_factory if (noiseop is None) else _op.ComposedOpFactory([ideal_factory, noiseop])
                    ret.factories[key] = factory
                else:
                    if embed_gates:
                        ideal_gate = local_gates[gn]
                        ideal_gate = _op.EmbeddedOp(state_space, inds, ideal_gate)
                    else:
                        embedded_unitary = _embed_unitary(state_space, inds, gate_unitary)
                        ideal_gate = _op.create_from_unitary_mx(embedded_unitary, ideal_gate_type, 'pp',
                                                                None, evotype, state_space)

                    #TODO: check for modelnoise on *local* gate, i.e. create_errormap(gn, ...)??
                    noiseop = modelnoise.create_errormap(key, evotype, state_space, target_labels=inds)
                    layer = _op.ComposedOp([ideal_gate, noiseop]) if (noiseop is not None) else ideal_gate
                    ret.operations[key] = layer

    # SPAM:
    local_noise = False; independent_gates=True
    prep_layers, povm_layers = _create_spam_layers(processor_spec, modelnoise, local_noise,
                                                   ideal_spam_type, evotype, state_space, independent_gates)
    for k, v in prep_layers.items():
        ret.preps[k] = v
    for k, v in povm_layers.items():
        ret.povms[k] = v

    return ret



def _create_spam_layers(processor_spec, modelnoise, local_noise,
                        ideal_spam_type, evotype, state_space, independent_gates):
    """ local_noise=True creates lindblad ops that are embedded & composed 1Q ops, and assumes
        that modelnoise specifies 1Q noise.  local_noise=False assumes modelnoise specifies n-qubit noise"""
    qubit_labels = processor_spec.qubit_labels
    num_qubits = processor_spec.number_of_qubits
    singleQ_state_space = _statespace.default_space_for_udim(2)  # single qubit state space

    #  Step 1 -- get the ideal prep and POVM, created as the types we want
    #  Step 2 -- add noise, by composing ideal with a noise operation (if desired)
    prep_layers = {}
    povm_layers = {}

    def _add_noise(prep_ops, povm_ops):
        """ Adds one or more noise ops to prep_ops and povm_ops lists (to compose later) """
        if local_noise:  # then assume modelnoise specifies 1Q errors
            prep_noiseop1Q = modelnoise.create_errormap('prep', evotype, singleQ_state_space, target_labels=None)
            if prep_noiseop1Q is not None:
                err_gates = [prep_noiseop1Q.copy() for i in range(num_qubits)] \
                    if independent_gates else [prep_noiseop1Q] * num_qubits
                prep_ops.extend([_op.EmbeddedOp(state_space, [qubit_labels[i]], err_gates[i])
                                 for i in range(num_qubits)])

            povm_noiseop1Q = modelnoise.create_errormap('povm', evotype, singleQ_state_space, target_labels=None)
            if povm_noiseop1Q is not None:
                err_gates = [povm_noiseop1Q.copy() for i in range(num_qubits)] \
                    if independent_gates else [povm_noiseop1Q] * num_qubits
                povm_ops.extend([_op.EmbeddedOp(state_space, [qubit_labels[i]], err_gates[i])
                                 for i in range(num_qubits)])

        else:  # use modelnoise to construct n-qubit noise
            prepNoiseMap = modelnoise.create_errormap('prep', evotype, state_space, target_labels=None,
                                                      qubit_graph=processor_spec.qubit_graph)
            povmNoiseMap = modelnoise.create_errormap('povm', evotype, state_space, target_labels=None,
                                                      qubit_graph=processor_spec.qubit_graph)
            if prepNoiseMap is not None: prep_ops.append(prepNoiseMap)
            if povmNoiseMap is not None: povm_ops.append(povmNoiseMap)

    def _add_to_layers(ideal_prep, prep_ops, ideal_povm, povm_ops):
        """ Adds noise elements to prep_layers and povm_layers """
        if len(prep_ops_to_compose) == 0:
            prep_layers['rho0'] = ideal_prep
        elif len(prep_ops_to_compose) == 1:
            prep_layers['rho0'] = _state.ComposedState(ideal_prep, prep_ops[0])
        else:
            prep_layers['rho0'] = _state.ComposedState(ideal_prep, _op.ComposedOp(prep_ops))

        if len(povm_ops_to_compose) == 0:
            povm_layers['Mdefault'] = ideal_povm
        elif len(povm_ops_to_compose) == 1:
            povm_layers['Mdefault'] = _povm.ComposedPOVM(povm_ops[0], ideal_povm, 'pp')
        else:
            povm_layers['Mdefault'] = _povm.ComposedPOVM(_op.ComposedOp(povm_ops), ideal_povm, 'pp')

    # Here's where the actual logic starts.  The above functions avoid repeated blocks within the different
    # cases below.
    if ideal_spam_type == 'computational' or ideal_spam_type.startswith('lindblad '):
        ideal_prep = _state.ComputationalBasisState([0] * num_qubits, 'pp', evotype, state_space)
        ideal_povm = _povm.ComputationalBasisPOVM(num_qubits, evotype, state_space=state_space)

        prep_ops_to_compose = []
        povm_ops_to_compose = []
        if ideal_spam_type.startswith('lindblad '):  # then add a composed exp(errorgen) to computational SPAM
            lndtype = ideal_spam_type[len('lindblad '):]

            if local_noise:
                # create a 1-qubit exp(errorgen) that is applied to each qubit independently
                err_gate = _op.LindbladErrorgen.from_error_generator(singleQ_state_space.dim, lndtype, 'pp', 'pp',
                                                                     truncate=True, evotype=evotype, state_space=None)
                err_gateNQ = _op.ComposedOp([_op.EmbeddedOp(state_space, [qubit_labels[i]], err_gate.copy())
                                             for i in range(num_qubits)], evotype, state_space)
            else:
                # create an n-qubit exp(errorgen)
                err_gateNQ = _op.LindbladErrorgen.from_error_generator(state_space.dim, lndtype, 'pp', 'pp',
                                                                       truncate=True, evotype=evotype,
                                                                       state_space=state_space)
            prep_ops_to_compose.append(err_gateNQ)
            povm_ops_to_compose.append(err_gateNQ.copy())  # .copy() => POVM errors independent

        # Add noise
        _add_noise(prep_ops_to_compose, povm_ops_to_compose)

        #Add final ops to returned dictionaries  (Note: None -> ComputationPOVM within ComposedPOVM)
        effective_ideal_povm = None if len(povm_ops_to_compose) > 0 else ideal_povm
        _add_to_layers(ideal_prep, prep_ops_to_compose, effective_ideal_povm, povm_ops_to_compose)

    elif ideal_spam_type.startswith('tensor product '):
        #Note: with "tensor product <X>" types, e.g. "tensor product static", we assume modelnoise specifies just
        # a 1Q noise operation, even when `local_noise=False`
        vectype = ideal_spam_type[len('tensor product '):]

        v0, v1 = _np.array([1, 0], 'd'), _np.array([0, 1], 'd')
        ideal_prep1Q = _state.create_from_pure_vector(v0, vectype, 'pp', evotype, state_space=None)
        ideal_povm1Q = _povm.create_from_pure_vectors([('0', v0), ('1', v1)], vectype, 'pp',
                                                      evotype, state_space=None)
        prep_factors = [ideal_prep1Q.copy() for i in range(num_qubits)]
        povm_factors = [ideal_povm1Q.copy() for i in range(num_qubits)]

        # Add noise
        prep_noiseop1Q = modelnoise.create_errormap('prep', evotype, singleQ_state_space, target_labels=None)
        if prep_noiseop1Q is not None:
            prep_factors = [_state.ComposedState(
                factor, (prep_noiseop1Q.copy() if independent_gates else prep_noiseop1Q)) for factor in prep_factors]

        povm_noiseop1Q = modelnoise.create_errormap('povm', evotype, singleQ_state_space, target_labels=None)
        if povm_noiseop1Q is not None:
            pov_factors = [_povm.ComposedPOVM(
                (povm_noiseop1Q.copy() if independent_gates else povm_noiseop1Q), factor) for factor in povm_factors]

        prep_layers['rho0'] = _state.TensorProductState(prep_factors, state_space)
        povm_layers['Mdefault'] = _povm.TensorProductPOVM(povm_factors, evotype, state_space)

        #    [('0', _povm.create_effect_from_state_vector(v0, vectype, 'pp', evotype, state_space=None)),
        #     ('1', _povm.create_effect_from_state_vector(v1, vectype, 'pp', evotype, state_space=None))],

    else:  # assume ideal_spam_type is a valid 'vectype' for creating n-qubit state vectors & POVMs

        vectype = ideal_spam_type
        vecs = []  # all the basis vectors for num_qubits
        for i in range(num_qubits):
            v = _np.zeros(num_qubits, 'd'); v[i] = 1.0
            vecs.append(v)

        ideal_prep = _state.create_from_pure_vector(v[0], vectype, 'pp', evotype, state_space=state_space)
        ideal_povm = _povm.create_from_pure_vectors(
            [(format(i, 'b').zfill(num_qubits), v) for i, v in enumerate(vecs)],
            vectype, 'pp', evotype, state_space=state_space)

        # Add noise
        prep_ops_to_compose = []
        povm_ops_to_compose = []
        _add_noise(prep_ops_to_compose, povm_ops_to_compose)

        # Add final ops to returned dictionaries
        _add_to_layers(ideal_prep, prep_ops_to_compose, ideal_povm, povm_ops_to_compose)

    #else:
    #    raise ValueError("Invalid `ideal_spam_type`: %s" % str(ideal_spam_type))

    return prep_layers, povm_layers


def create_localnoise_model(num_qubits, gate_names, nonstd_gate_unitaries=None, custom_gates=None,
                            availability=None, qubit_labels=None, geometry="line", parameterization='static',
                            evotype="default", simulator="auto", on_construction_error='raise',
                            independent_gates=False, ensure_composed_gates=False, global_idle=None):
    """
    Creates a "standard" n-qubit local-noise model, usually of ideal gates.

    The returned model is "standard", in that the following standard gate
    names may be specified as elements to `gate_names` without the need to
    supply their corresponding unitaries (as one must when calling
    the constructor directly):

    - 'Gi' : the 1Q idle operation
    - 'Gx','Gy','Gz' : 1Q pi/2 rotations
    - 'Gxpi','Gypi','Gzpi' : 1Q pi rotations
    - 'Gh' : Hadamard
    - 'Gp' : phase
    - 'Gcphase','Gcnot','Gswap' : standard 2Q gates

    Furthermore, if additional "non-standard" gates are needed,
    they are specified by their *unitary* gate action, even if
    the final model propagates density matrices (as opposed
    to state vectors).

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

    gate_names : list
        A list of string-type gate names (e.g. `"Gx"`) either taken from
        the list of builtin "standard" gate names given above or from the
        keys of `nonstd_gate_unitaries`.  These are the typically 1- and 2-qubit
        gates that are repeatedly embedded (based on `availability`) to form
        the resulting model.

    nonstd_gate_unitaries : dict, optional
        A dictionary of numpy arrays which specifies the unitary gate action
        of the gate names given by the dictionary's keys.  As an advanced
        behavior, a unitary-matrix-returning function which takes a single
        argument - a tuple of label arguments - may be given instead of a
        single matrix to create an operation *factory* which allows
        continuously-parameterized gates.  This function must also return
        an empty/dummy unitary when `None` is given as it's argument.

    custom_gates : dict, optional
        A dictionary that associates with gate labels
        :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
        objects.  These objects describe the full action of the gate or
        primitive-layer they're labeled by (so if the model represents
        states by density matrices these objects are superoperators, not
        unitaries), and override any standard construction based on builtin
        gate names or `nonstd_gate_unitaries`.  Keys of this dictionary may
        be string-type gate *names*, which will be embedded according to
        `availability`, or labels that include target qubits,
        e.g. `("Gx",0)`, which override this default embedding behavior.
        Furthermore, :class:`OpFactory` objects may be used in place of
        `LinearOperator` objects to allow the evaluation of labels with
        arguments.

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gatedict` and whose values are lists of qubit-label-tuples.  Each
        qubit-label-tuple must have length equal to the number of qubits
        the corresponding gate acts upon, and causes that gate to be
        embedded to act on the specified qubits.  For example,
        `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
        the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
        0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
        acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
        values of `availability` may take the special values:

        - `"all-permutations"` and `"all-combinations"` equate to all possible
        permutations and combinations of the appropriate number of qubit labels
        (deterined by the gate's dimension).
        - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (a key of `gatedict`) is not present in `availability`,
        the default is `"all-edges"`.

    qubit_labels : tuple, optional
        The circuit-line labels for each of the qubits, which can be integers
        and/or strings.  Must be of length `num_qubits`.  If None, then the
        integers from 0 to `num_qubits-1` are used.

    geometry : {"line","ring","grid","torus"} or QubitGraph, optional
        The type of connectivity among the qubits, specifying a graph used to
        define neighbor relationships.  Alternatively, a :class:`QubitGraph`
        object with `qubit_labels` as the node labels may be passed directly.
        This argument is only used as a convenient way of specifying gate
        availability (edge connections are used for gates whose availability
        is unspecified by `availability` or whose value there is `"all-edges"`).

    parameterization : {"full", "TP", "CPTP", "H+S", "S", "static", "H+S terms",
        "H+S clifford terms", "clifford"}
        The type of parameterizaton to use for each gate value before it is
        embedded. See :method:`Model.set_all_parameterizations` for more
        details.

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The simulator used to compute predicted probabilities for the
        resulting :class:`Model`.  Usually `"auto"` is fine, the default for
        each `evotype` is usually what you want.  Setting this to something
        else is expert-level tuning.

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

    global_idle : LinearOperator, optional
        A global idle operation, which is performed once at the beginning
        of every circuit layer.  If `None`, no such operation is performed.
        If a 1-qubit operator is given and `num_qubits > 1` the global idle
        is the parallel application of this operator on each qubit line.
        Otherwise the given operator must act on all `num_qubits` qubits.

    Returns
    -------
    Model
        A model with `"rho0"` prep, `"Mdefault"` POVM, and gates labeled by
        gate name (keys of `gatedict`) and qubit labels (from within
        `availability`).  For instance, the operation label for the `"Gx"` gate on
        qubit 2 might be `Label("Gx",1)`.
    """
    return _LocalNoiseModel.from_parameterization(
        num_qubits, gate_names, nonstd_gate_unitaries, custom_gates,
        availability, qubit_labels, geometry, parameterization, evotype,
        simulator, on_construction_error, independent_gates,
        ensure_composed_gates, global_idle)

class StencilLabel(object):

    @classmethod
    def cast(cls, obj):
        if obj is None: return StencilLabelTuple(None)
        if isinstance(obj, StencilLabel): return obj
        if isinstance(obj, tuple): return StencilLabelTuple(obj)
        if isinstance(obj, list): return StencilLabelSet(obj)
        raise ValueError("Cannot cast %s to a StencilLabel" % str(type(obj)))

    def __init__(self, local_state_space=None):
        self.local_state_space = local_state_space

    def _resolve_single_sslbls_tuple(self, sslbls, qubit_graph, state_space, target_lbls):
        if qubit_graph is None:  # without a graph, we need to ensure all the stencil_sslbls are valid
            assert (state_space.contains_labels(sslbls))
            return sslbls
        else:
            ret = [qubit_graph.resolve_relative_nodelabel(s, target_lbls) for s in sslbls]
            if any([x is None for x in ret]): return None  # signals there is a non-present dirs, e.g. end of chain
            return tuple(ret)

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        raise NotImplementedError("Derived classes should implement this!")

    def create_local_state_space(self, entire_state_space):
        """ Creates a local state space for an operator indexed using this stencil label """
        if self.local_state_space is not None:
            return self.local_state_space  # so the user can always override this space if needed
        else:
            return self._create_local_state_space(entire_state_space)

    def _create_local_state_space_for_sslbls(self, sslbls, entire_state_space):
        if entire_state_space.contains_labels(sslbls):  # absolute sslbls - get space directly
            return entire_state_space.create_subspace(sslbls)
        else:
            return entire_state_space.create_stencil_subspace(sslbls)
            # only works when state space has a common label dimension

    
class StencilLabelTuple(StencilLabel):
    def __init__(self, stencil_sslbls):
        self.sslbls = stencil_sslbls
        super(StencilLabelTuple, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        # Return a *list* of sslbls, since some stencil labels may resolve into multiple absolute sslbls
        if self.sslbls is None:
            return [None]  # sslbls=None is resolved to `None`
        return [self._resolve_single_sslbls_tuple(self.sslbls, qubit_graph, state_space, target_lbls)]

    def _create_local_state_space(self, entire_state_space):
        return self._create_local_state_space_for_sslbls(self.sslbls, entire_state_space)

    def __str__(self):
        return "StencilLabel(" + str(self.sslbls) + ")"


class StencilLabelSet(StencilLabel):
    def __init__(self, stencil_sslbls_set):
        self.sslbls_set = stencil_sslbls_set
        super(StencilLabelSet, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        # return a *list* of sslbls, since some stencil labels may resolve into multiple absolute sslbls
        return [self._resolve_single_sslbls_tuple(sslbls, qubit_graph, state_space, target_lbls)
                for sslbls in self.sslbls_set]

    def _create_local_state_space(self, entire_state_space):
        if len(self.sslbls_set) == 0: return None  # or an empty space?
        return self._create_local_state_space_for_sslbls(self.sslbls_set[0], entire_state_space)

    def __str__(self):
        return "StencilLabel{" + str(self.sslbls_set) + "}"



class StencilLabelAllCombos(StencilLabel):
    def __init__(self, possible_sslbls, num_to_choose, connected=False):
        self.possible_sslbls = possible_sslbls
        self.num_to_choose = num_to_choose
        self.connected = connected
        super(StencilLabelAllCombos, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        ret = []
        for chosen_sslbls in _itertools.combinations(self.possible_sslbls, self.num_to_choose):
            if self.connected and len(chosen_sslbls) == 2 \
               and not qubit_graph.is_directly_connected(chosen_sslbls[0], chosen_sslbls[1]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph
            ret.append(self._resolve_single_sslbls_tuple(chosen_sslbls, qubit_graph, state_space, target_lbls))
        return ret  # return a *list* of

    def _create_local_state_space(self, entire_state_space):
        common_udim = entire_state_space.common_udimension
        if common_udim is None:
            raise ValueError(("All-combos stencil labels can only be used with state spaces that"
                              " have a common label dimension"))
        lbls = tuple(range(self.num_to_choose)); udims = (common_udim,) * self.num_to_choose
        return _statespace.ExplicitStateSpace(lbls, udims)

    def __str__(self):
        return ("StencilCombos(" + str(self.possible_sslbls) + (" connected-" if self.connected else " ")
                +  "choose %d" % self.num_to_choose + ")")


class StencilLabelRadiusCombos(StencilLabel):
    def __init__(self, base_sslbls, radius, num_to_choose, connected=False):
        self.base_sslbls = base_sslbls
        self.radius = radius  # in "hops" along graph
        self.num_to_choose = num_to_choose
        self.connected = connected
        super(StencilLabelRadiusCombos, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        ret = []
        assert(qubit_graph is not None), "A qubit graph is required by StencilLabelRadiusCombos!"
        abs_base_sslbls = self._resolve_single_sslbls_tuple(self.base_sslbls, qubit_graph, state_space, target_lbls)
        radius_nodes = qubit_graph.radius(abs_base_sslbls, self.radius)

        for chosen_sslbls in _itertools.combinations(radius_nodes, self.num_to_choose):
            if self.connected and len(chosen_sslbls) == 2 \
               and not qubit_graph.is_directly_connected(chosen_sslbls[0], chosen_sslbls[1]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph
            ret.append(self._resolve_single_sslbls_tuple(chosen_sslbls, qubit_graph, state_space, target_lbls))
        return ret  # return a *list* of sslbls

    def _create_local_state_space(self, entire_state_space):
        # A duplicate of StencilLabelAllCombos._create_local_state_space
        common_udim = entire_state_space.common_udimension
        if common_udim is None:
            raise ValueError(("All-combos stencil labels can only be used with state spaces that"
                              " have a common label dimension"))
        lbls = tuple(range(self.num_to_choose));
        udims = (common_udim,) * self.num_to_choose
        return _statespace.ExplicitStateSpace(lbls, udims)

    def __str__(self):
        return ("StencilRadius(%schoose %d within %d hops from %s)" % (("connected-" if self.connected else ""),
                                                                       self.num_to_choose, self.radius,
                                                                       str(self.base_sslbls)))


class ModelNoise(object):
    """ TODO: docstring -- lots of docstrings to do from here downward!"""
    pass


class OpModelNoise(ModelNoise):

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, OpModelNoise):
            return obj
        elif isinstance(obj, (list, tuple)):  # assume obj == list of OpModelNoise objects
            return ComposedOpModelNoise([cls.cast(el) for el in obj])
        elif isinstance(obj, dict):  # assume obj = dict of per-op noise
            return OpModelPerOpNoise(obj)
        else:
            raise ValueError("Cannot convert type %s to an OpModelNoise object!" % str(type(obj)))

    def keys(self):
        raise NotImplementedError("Derived classes should implement this!")

    def __contains__(self, key):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errorgen_stencil(self, opkey, evotype, state_space, target_labels=None):
        raise NotImplementedError("Derived classes should implement this!")

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errorgen(self, opkey, evotype, state_space, target_labels=None, qubit_graph=None):
        stencil = self.create_errorgen_stencil(opkey, evotype, state_space, target_labels)
        return self.apply_errorgen_stencil(stencil, evotype, state_space, target_labels, qubit_graph)

    def create_errormap_stencil(self, opkey, evotype, state_space, target_labels=None):
        raise NotImplementedError("Derived classes should implement this!")

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errormap(self, opkey, evotype, state_space, target_labels=None, qubit_graph=None):
        stencil = self.create_errormap_stencil(opkey, evotype, state_space, target_labels)
        return self.apply_errormap_stencil(stencil, evotype, state_space, target_labels, qubit_graph)        


class OpModelPerOpNoise(OpModelNoise):

    def __init__(self, per_op_noise):
        # a dictionary mapping operation keys -> OpNoise objects
        #                                  OR -> {dict mapping sslbls -> OpNoise objects}
        self.per_op_noise = per_op_noise.copy()

        #Update any label-string format keys to actual Labels (convenience for users)
        cparser = _CircuitParser()
        cparser.lookup = None  # lookup - functionality removed as it wasn't used
        for k, v in per_op_noise.items():
            if isinstance(k, str) and ":" in k:  # then parse this to get a label, allowing, e.g. "Gx:0"
                lbls, _, _ = cparser.parse(k)
                assert (len(lbls) == 1), "Only single primitive-gate labels allowed as keys! (not %s)" % str(k)
                del self.per_op_noise[k]
                self.per_op_noise[lbls[0]] = v

        super(OpModelPerOpNoise, self).__init__()

    def keys(self):
        return self.per_op_noise.keys()

    def __contains__(self, key):
        return key in self.per_op_noise

    def create_errorgen_stencil(self, opkey, evotype, state_space, target_labels=None):
        if opkey not in self.per_op_noise: return {}  # an empty stencil
        opnoise = self.per_op_noise[opkey]
        errgens_to_embed_then_compose = _collections.OrderedDict()  # the "stencil" we return
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise
            for sslbls, opnoise in opnoise_dict.items():
                local_state_space = StencilLabel.cast(sslbls).create_local_state_space(state_space)
                local_errorgen = opnoise.create_errorgen(evotype, local_state_space)
                errgens_to_embed_then_compose[sslbls] = local_errorgen
        else:  # assume opnoise is an OpNoise object
            local_errorgen = opnoise.create_errorgen(evotype, state_space)
            errgens_to_embed_then_compose[target_labels] = local_errorgen
        return errgens_to_embed_then_compose

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        embedded_errgens = []
        for stencil_sslbls, local_errorgen in stencil.items():
            sslbls_list = self._map_stencil_sslbls(stencil_sslbls, qubit_graph, state_space, target_labels)
            if sslbls_list is None and stencil_sslbls is not None:
                continue  # signals not all directions present => skip this term
            for sslbls in sslbls_list:
                op_to_embed = local_errorgen if (sslbls is None or state_space.is_entire_space(sslbls)) \
                    else _op.EmbeddedErrorgen(state_space, sslbls, local_errorgen)
                embedded_errgens.append(op_to_embed.copy() if copy else op_to_embed)

        if len(embedded_errgens) == 0:
            return None  # ==> no errorgen (could return an empty ComposedOp instead?)
        else:
            return _op.ComposedErrorgen(embedded_errgens, evotype, state_space) \
                if len(embedded_errgens) > 1 else embedded_errgens[0]

    def create_errormap_stencil(self, opkey, evotype, state_space, target_labels=None):
        if opkey not in self.per_op_noise: return {}  # an empty stencil
        opnoise = self.per_op_noise[opkey]
        errmaps_to_embed_then_compose = _collections.OrderedDict()  # the "stencil" we return
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise
            for sslbls, opnoise in opnoise_dict.items():
                local_state_space = StencilLabel.cast(sslbls).create_local_state_space(state_space)
                local_errormap = opnoise.create_errormap(evotype, local_state_space)
                errmaps_to_embed_then_compose[sslbls] = local_errormap
        else:  # assume opnoise is an OpNoise object
            local_errormap = opnoise.create_errormap(evotype, state_space)
            errmaps_to_embed_then_compose[target_labels] = local_errormap
        return errmaps_to_embed_then_compose

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        embedded_errmaps = []
        for stencil_sslbls, local_errormap in stencil.items():
            sslbls_list = self._map_stencil_sslbls(stencil_sslbls, qubit_graph, state_space, target_labels)
            if sslbls_list is None and stencil_sslbls is not None:
                continue  # signals not all directions present => skip this term
            for sslbls in sslbls_list:
                op_to_embed = local_errormap if (sslbls is None or state_space.is_entire_space(sslbls)) \
                    else _op.EmbeddedOp(state_space, sslbls, local_errormap)
                embedded_errmaps.append(op_to_embed.copy() if copy else op_to_embed)

        if len(embedded_errmaps) == 0:
            return None  # ==> no errormap (could return an empty ComposedOp instead?)
        else:
            return _op.ComposedOp(embedded_errmaps, evotype, state_space) \
                if len(embedded_errmaps) > 1 else embedded_errmaps[0]

    def _map_stencil_sslbls(self, stencil_sslbls, qubit_graph, state_space, target_lbls):  # deals with graph directions
        stencil_sslbls = StencilLabel.cast(stencil_sslbls)
        return stencil_sslbls.compute_absolute_sslbls(qubit_graph, state_space, target_lbls)

    def _key_to_str(self, key, prefix=""):
        opnoise = self.per_op_noise.get(key, None)
        if opnoise is None:
            return prefix + str(key) + ": <missing>"
        
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise; val_str = ''
            for sslbls, opnoise in opnoise_dict.items():
                val_str += prefix + "  " + str(sslbls) + ": " + str(opnoise) + '\n'
        else:
            val_str = prefix + "  " + str(opnoise) + '\n'
        return prefix + str(key) + ":\n" + val_str

    def __str__(self):
        return '\n'.join([self._key_to_str(k) for k in self.keys()])


class ComposedOpModelNoise(OpModelNoise):
    def __init__(self, opmodelnoises):
        self.opmodelnoises = tuple(opmodelnoises)  # elements == OpModelNoise objects
        #self.ensure_no_duplicates()  # not actually needed; we just compose errors
        super(ComposedOpModelNoise, self).__init__()

    def ensure_no_duplicates(self):
        running_keys = set()
        for modelnoise in self.opmodelnoises:
            duplicate_keys = running_keys.intersection(modelnoise.keys())
            assert (len(duplicate_keys) == 0), \
                "Duplicate keys not allowed in model noise specifications: %s" % ','.join(duplicate_keys)
            running_keys = running_keys.union(modelnoise.keys())

    def keys(self):
        # Use remove_duplicates rather than set(.) to preserve ordering (but this is slower!)
        return _lt.remove_duplicates(_itertools.chain(*[modelnoise.keys() for modelnoise in self.opmodelnoises]))

    def __contains__(self, key):
        return any([(key in modelnoise) for modelnoise in self.opmodelnoises])

    def create_errorgen_stencil(self, opkey, evotype, state_space, target_labels=None):
        return tuple([modelnoise.create_errorgen_stencil(opkey, evotype, state_space, target_labels)
                      for modelnoise in self.opmodelnoises])

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        noise_errgens = [modelnoise.apply_errorgen_stencil(s, evotype, state_space, target_labels, qubit_graph, copy)
                         for s, modelnoise in zip(stencil, self.opmodelnoises)]
        noise_errgens = list(filter(lambda x: x is not None, noise_errgens))
        return _op.ComposedErrorgen(noise_errgens) if len(noise_errgens) > 0 else None

    def create_errormap_stencil(self, opkey, evotype, state_space, target_labels=None):
        return tuple([modelnoise.create_errormap_stencil(opkey, evotype, state_space, target_labels)
                      for modelnoise in self.opmodelnoises])

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        noise_ops = [modelnoise.apply_errormap_stencil(s, evotype, state_space, target_labels, qubit_graph, copy)
                         for s, modelnoise in zip(stencil, self.opmodelnoises)]
        noise_ops = list(filter(lambda x: x is not None, noise_ops))
        return _op.ComposedOp(noise_ops) if len(noise_ops) > 0 else None

    def _key_to_str(self, key, prefix=''):
        val_str = ''
        for i, modelnoise in enumerate(self.opmodelnoises):
            if key in modelnoise:
              val_str += prefix + ("  [%d]:\n" % i) + modelnoise._key_to_str(key, prefix + "    ")
        if len(val_str) > 0:
            return prefix + str(key) + ":\n" + val_str
        else:
            return prefix + str(key) + ": <missing>"

    def __str__(self):
        return '\n'.join([self._key_to_str(k) for k in self.keys()])


class OpNoise(object):

    def __str__(self):
        return self.__class__.__name__ + "(" + ", ".join(["%s=%s" % (str(k), str(v))
                                                          for k, v in self.__dict__.items()]) + ")"


class DepolarizationNoise(OpNoise):
    def __init__(self, depolarization_rate, parameterization='depolarize'):
        self.depolarization_rate = depolarization_rate
        self.parameterization = parameterization

        valid_depol_params = ['depolarize', 'stochastic', 'lindblad']
        assert (self.parameterization in valid_depol_params), \
            "The depolarization parameterization must be one of %s, not %s" \
            % (valid_depol_params, self.parameterization)

    def create_errorgen(self, evotype, state_space):
        if self.parameterization != 'lindblad':
            raise ValueError("Cannot only construct error generators for 'lindblad' parameterization")

        # LindbladErrorgen with "depol" or "diagonal" param
        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        rate_per_pauli =  self.depolarization_rate / (basis_size - 1)
        errdict = {('S', bl): rate_per_pauli for bl in basis.labels[1:]}
        return _op.LindbladErrorgen(errdict, "D", basis, mx_basis='pp',
                                    truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        basis_size = state_space.dim  # e.g. 4 for a single qubit

        if self.parameterization == "depolarize":  # DepolarizeOp
            return _op.DepolarizeOp(state_space, basis="pp", evotype=evotype,
                                    initial_rate=self.depolarization_rate)

        elif self.parameterization == "stochastic":  # StochasticNoiseOp
            rate_per_pauli = self.depolarization_rate / (basis_size - 1)
            rates = [rate_per_pauli] * (basis_size - 1)
            return _op.StochasticNoiseOp(state_space, basis="pp", evotype=evotype, initial_rates=rates)

        elif self.parameterization == "lindblad":
            errgen = self.create_errorgen(evotype, state_space)
            return _op.ExpErrorgenOp(errgen)

        else:
            raise ValueError("Unknown parameterization %s for depolarizing error specification"
                             % self.parameterization)


class StochasticNoise(OpNoise):
    def __init__(self, error_probs, parameterization='stochastic'):
        self.error_probs = error_probs
        self.parameterization = parameterization

        valid_sto_params = ['stochastic', 'lindblad']
        assert (self.parameterization in valid_sto_params), \
            "The stochastic parameterization must be one of %s, not %s" \
            % (valid_sto_params, self.parameterization)

    def create_errorgen(self, evotype, state_space):
        sto_rates = self.error_probs

        if self.parameterization != 'lindblad':
            raise ValueError("Cannot only construct error generators for 'lindblad' parameterization")

        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        errdict = {('S', bl): rate for bl, rate in zip(basis.labels[1:], sto_rates)}
        return _op.LindbladErrorgen(errdict, "S", basis, mx_basis='pp',
                                    truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        sto_rates = self.error_probs

        if self.parameterization == "stochastic":  # StochasticNoiseOp
            return _op.StochasticNoiseOp(state_space, basis="pp", evotype=evotype, initial_rates=sto_rates)

        elif self.parameterization  == "lindblad":  # LindbladErrorgen with "cptp", "diagonal" parameterization
            errgen = self.create_errorgen(evotype, state_space)
            return _op.ExpErrorgenOp(errgen)
        else:
            raise ValueError("Unknown parameterization %s for stochastic error specification"
                             % self.parameterization )


class LindbladNoise(OpNoise):
    @classmethod
    def from_basis_coefficients(cls, parameterization, lindblad_basis, state_space, ham_coefficients=None,
                                nonham_coefficients=None):
        """ TODO: docstring - None coefficients mean zeros"""
        dim = state_space.dim
        lindblad_basis = _Basis.cast(lindblad_basis, dim)

        parameterization = _op.LindbladParameterization.cast(parameterization)
        ham_basis = lindblad_basis if parameterization.ham_params_allowed else None
        nonham_basis = lindblad_basis if parameterization.nonham_params_allowed else None

        if ham_coefficients is None and ham_basis is not None:
            ham_coefficients = _np.zeros(len(ham_basis) - 1, 'd')
        if nonham_coefficients is None and nonham_basis is not None:
            d = len(ham_basis) - 1
            nonham_coefficients = _np.zeros((d,d), complex) if parameterization.nonham_mode == 'all' \
                else _np.zeros(d, 'd')

        # coeffs + bases => Ltermdict, basis
        Ltermdict, _ = _ot.projections_to_lindblad_terms(
            ham_coefficients, nonham_coefficients, ham_basis, nonham_basis, parameterization.nonham_mode)
        return cls(Ltermdict, parameterization)

    def __init__(self, error_coeffs, parameterization='auto'):
        self.error_coeffs = error_coeffs
        self.parameterization = parameterization

    def create_errorgen(self, evotype, state_space):
        # Build LindbladErrorgen directly to have control over which parameters are set (leads to lower param counts)
        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        return _op.LindbladErrorgen(self.error_coeffs, self.parameterization, basis, mx_basis='pp',
                                    truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        errgen = self.create_errorgen(evotype, state_space)
        return _op.ExpErrorgenOp(errgen)


def _setup_local_gates(processor_spec, evotype, modelnoise=None, custom_gates=None,
                       ideal_gate_type=('static standard', 'static clifford', 'static unitary')):
    """ TODO: docstring"""
    std_gate_unitaries = _itgs.standard_gatename_unitaries()
    if custom_gates is None: custom_gates = {}
    if modelnoise is None: modelnoise = OpModelPerOpNoise({})

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

        # Skip idle, prep, and povm here, just do gates
        if key in ['idle', 'prep', 'povm']:
            continue

        # If key has qubits, get base name for lookup
        label = _label.Label(key)
        name = label.name

        U = processor_spec.gate_unitaries[name]  # all gate names must be in the processorspec
        if ((name not in processor_spec.nonstd_gate_unitaries)
                or (not callable(processor_spec.nonstd_gate_unitaries[name])
                    and processor_spec.nonstd_gate_unitaries[name].shape == std_gate_unitaries[name].shape
                    and _np.allclose(processor_spec.nonstd_gate_unitaries[name], std_gate_unitaries[name]))):
            stdname = name  # setting `stdname` != None means we can try to create a StaticStandardOp below
        else:
            stdname = None

        if not callable(U):  # normal operation (not a factory)
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
            local_state_space = _statespace.default_space_for_udim(U.udim)
            ideal_factory = _opfactory.UnitaryOpFactory(U, local_state_space, 'pp', evotype)
            noiseop = modelnoise.create_errormap(key, evotype, ideal_gate.state_space, target_labels=None)
            gatedict[key] = _opfactory.ComposedOpFactory([ideal_factory, noiseop]) \
                if (noiseop is not None) else ideal_factory
    return gatedict


def create_crosstalk_free_model(processor_spec, custom_gates=None,
                                depolarization_strengths=None, stochastic_error_probs=None, lindblad_error_coeffs=None,
                                depolarization_parameterization='depolarize', stochastic_parameterization='stochastic',
                                lindblad_parameterization='auto',
                                evotype="default", simulator="auto", on_construction_error='raise',
                                independent_gates=False, ensure_composed_gates=False,
                                ideal_gate_type='auto', ideal_spam_type='computational'):
    """
    TODO: update docstring
    Create a n-qubit "crosstalk-free" model.

    By virtue of being crosstalk-free, this model's operations only
    act nontrivially on their target qubits.

    Errors can be specified using any combination of the 4 error rate/coeff arguments,
    but each gate name must be provided exclusively to one type of specification.
    Each specification results in a different type of operation, depending on the parameterization:
        - `depolarization_strengths`    -> DepolarizeOp, StochasticNoiseOp, or exp(LindbladErrorgen)
        - `stochastic_error_probs`      -> StochasticNoiseOp or exp(LindbladErrorgen)
        - `lindblad_error_coeffs`       -> exp(LindbladErrorgen)

    In addition to the gate names, the special values `"prep"`, `"povm"`, `"idle"`,
    may be used as keys to specify the error on the state preparation, measurement, and global idle,
    respectively.

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

    gate_names : list
        A list of string-type gate names (e.g. `"Gx"`) either taken from
        the list of builtin "standard" gate names or from the
        keys of `nonstd_gate_unitaries`.  These are the typically 1- and 2-qubit
        gates that are repeatedly embedded (based on `availability`) to form
        the resulting model.

    nonstd_gate_unitaries : dict, optional
        A dictionary of numpy arrays which specifies the unitary gate action
        of the gate names given by the dictionary's keys.

    custom_gates : dict, optional
        A dictionary that associates with gate labels
        :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
        objects.  These objects override any other behavior for constructing
        their designated operations (e.g. from `error_rates` or
        `nonstd_gate_unitaries`).  Keys of this dictionary may
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

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gatedict` and whose values are lists of qubit-label-tuples.  Each
        qubit-label-tuple must have length equal to the number of qubits
        the corresponding gate acts upon, and causes that gate to be
        embedded to act on the specified qubits.  For example,
        `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
        the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
        0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
        acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
        values of `availability` may take the special values:

        - `"all-permutations"` and `"all-combinations"` equate to all possible
        permutations and combinations of the appropriate number of qubit labels
        (deterined by the gate's dimension).
        - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (a key of `gatedict`) is not present in `availability`,
        the default is `"all-edges"`.

    qubit_labels : tuple, optional
        The circuit-line labels for each of the qubits, which can be integers
        and/or strings.  Must be of length `num_qubits`.  If None, then the
        integers from 0 to `num_qubits-1` are used.

    geometry : {"line","ring","grid","torus"} or QubitGraph, optional
        The type of connectivity among the qubits, specifying a graph used to
        define neighbor relationships.  Alternatively, a :class:`QubitGraph`
        object with `qubit_labels` as the node labels may be passed directly.
        This argument is only used as a convenient way of specifying gate
        availability (edge connections are used for gates whose availability
        is unspecified by `availability` or whose value there is `"all-edges"`).

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


    Returns
    -------
    Model
        A model with `"rho0"` prep, `"Mdefault"` POVM, and gates labeled by
        gate name (keys of `gatedict`) and qubit labels (from within
        `availability`).  For instance, the operation label for the `"Gx"` gate on
        qubit 2 might be `Label("Gx",1)`.
    """
    modelnoises = []
    if depolarization_strengths is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: DepolarizationNoise(val, depolarization_parameterization)
                                              for lbl, val in depolarization_strengths.items()}))
    if stochastic_error_probs is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: StochasticNoise(val, stochastic_parameterization)
                                              for lbl, val in stochastic_error_probs.items()}))
    if lindblad_error_coeffs is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: LindbladNoise(val, lindblad_parameterization)
                                              for lbl, val in lindblad_error_coeffs.items()}))

    return _create_crosstalk_free_model(processor_spec, ComposedOpModelNoise(modelnoises), custom_gates, evotype,
                                        simulator, on_construction_error, independent_gates, ensure_composed_gates,
                                        ideal_gate_type, ideal_spam_type)


# num_qubits, gate_names, nonstd_gate_unitaries={}, availability=None, qubit_labels=None, geometry="line"
def _create_crosstalk_free_model(processor_spec, modelnoise, custom_gates=None, evotype="default", simulator="auto",
                                 on_construction_error='raise', independent_gates=False, ensure_composed_gates=False,
                                 ideal_gate_type='auto', ideal_spam_type='computational'):
    qubit_labels = processor_spec.qubit_labels
    state_space = _statespace.QubitSpace(qubit_labels)
    evotype = _Evotype.cast(evotype)
    modelnoise = OpModelNoise.cast(modelnoise)
    num_qubits = len(qubit_labels)

    if ideal_gate_type == "auto":
        ideal_gate_type = ('static standard', 'static clifford', 'static unitary')

    gatedict = _setup_local_gates(processor_spec, evotype, modelnoise, custom_gates, ideal_gate_type)

    # GLOBAL IDLE:
    ideal_global_idle = _op.create_from_unitary_mx(_np.identity(2), ideal_gate_type,
                                                   'pp', 'Gi', evotype, state_space=None)
    global_idle_op = ideal_global_idle if (ideal_global_idle.num_params > 0) else None
    if 'idle' in modelnoise:
        idle_state_space = _statespace.default_space_for_udim(2)  # single qubit state space
        global_idle_error = modelnoise.create_errormap('idle', evotype, idle_state_space,
                                                    target_labels=None)  # 1-qubit idle op, can be None
        global_idle_op = global_idle_error if (global_idle_op is None) \
            else _op.ComposedOp((global_idle_op, global_idle_error))

    # SPAM:
    local_noise = True
    prep_layers, povm_layers = _create_spam_layers(processor_spec, modelnoise, local_noise,
                                                   ideal_spam_type, evotype, state_space, independent_gates)

    return _LocalNoiseModel(processor_spec, gatedict, prep_layers, povm_layers,
                            evotype, simulator, on_construction_error,
                            independent_gates, ensure_composed_gates, global_idle_op)


def create_cloud_crosstalk_model(processor_spec, custom_gates=None,
                                 depolarization_strengths={}, stochastic_error_probs={}, lindblad_error_coeffs={},
                                 depolarization_parameterization='depolarize', stochastic_parameterization='stochastic',
                                 lindblad_parameterization='auto', evotype="default", simulator="auto",
                                 independent_gates=False, errcomp_type="errorgens", add_idle_noise_to_all_gates=True,
                                 verbosity=0):
    """
    TODO: update docstring - see long docstring in nqnoiseconstruction.py
    """
    modelnoises = []
    if depolarization_strengths is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: DepolarizationNoise(val, depolarization_parameterization)
                                              for lbl, val in depolarization_strengths.items()}))
    if stochastic_error_probs is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: StochasticNoise(val, stochastic_parameterization)
                                              for lbl, val in stochastic_error_probs.items()}))
    if lindblad_error_coeffs is not None:
        modelnoises.append(OpModelPerOpNoise({lbl: LindbladNoise(val, lindblad_parameterization)
                                              for lbl, val in lindblad_error_coeffs.items()}))

    return _create_cloud_crosstalk_model(processor_spec, ComposedOpModelNoise(modelnoises), custom_gates, evotype,
                                         simulator, simulator, independent_gates, errcomp_type,
                                         add_idle_noise_to_all_gates, verbosity)


def _create_cloud_crosstalk_model(processor_spec, modelnoise, custom_gates=None,
                                  evotype="default", simulator="auto", independent_gates=False,
                                  errcomp_type="errorgens", add_idle_noise_to_all_gates=True, verbosity=0):
    qubit_labels = processor_spec.qubit_labels
    state_space = _statespace.QubitSpace(qubit_labels)  # FUTURE: allow other types of state spaces somehow?
    evotype = _Evotype.cast(evotype)
    modelnoise = OpModelNoise.cast(modelnoise)
    num_qubits = len(qubit_labels)
    printer = _VerbosityPrinter.create_printer(verbosity)

    #Create static ideal gates without any noise (we use `modelnoise` further down)
    gatedict = _setup_local_gates(processor_spec, evotype, None, custom_gates,
                                  ideal_gate_type=('static standard', 'static clifford', 'static unitary'))
    stencils = _collections.OrderedDict()

    # Global Idle
    if 'idle' in modelnoise:
        printer.log("Creating Idle:")
        global_idle_layer = modelnoise.create_errormap('idle', evotype, state_space, target_labels=None,
                                                       qubit_graph=processor_spec.qubit_graph)
    else:
        global_idle_layer = None

    # SPAM
    local_noise = False
    prep_layers, povm_layers = _create_spam_layers(processor_spec, modelnoise, local_noise,
                                                   'computational', evotype, state_space, independent_gates)

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

        if lbl in modelnoise:
            stencil = create_stencil_fn(lbl, evotype, state_space, target_labels=lbl.sslbls)
        elif lbl.name in stencils:
            stencil = stencils[lbl.name]
        elif lbl.name in modelnoise:
            stencils[lbl.name] = create_stencil_fn(lbl.name, evotype, state_space, target_labels=lbl.sslbls)
            stencil = stencils[lbl.name]
        else:
            return None  # no cloudnoise error for this label

        # REMOVE
        # elif build_modelnoise_fn is not None:
        #     opnoise_dict = build_modelnoise_fn(lbl)
        #     if opnoise_dict is not None:
        #         modelnoise_lazy = OpModelPerOpNoise({lbl: opnoise_dict})
        #
        #         if errcomp_type == 'gates':
        #             return modelnoise_lazy.create_errormap(lbl, evotype, state_space, target_labels=lbl.sslbls)
        #         elif errcomp_type == 'errorgens':
        #             return modelnoise_lazy.create_errorgen(lbl, evotype, state_space, target_labels=lbl.sslbls)
        #         else:
        #             raise ValueError("Invalid `errcomp_type` value: %s" % str(errcomp_type))
        return apply_stencil_fn(stencil, evotype, state_space, target_labels=lbl.sslbls,
                                qubit_graph=processor_spec.qubit_graph,
                                copy=independent_gates and (lbl not in modelnoise))  # no need to copy if first case

    def build_cloudkey_fn(lbl):
        if lbl in modelnoise:
            stencil = create_stencil_fn(lbl, evotype, state_space, target_labels=lbl.sslbls)
        elif lbl.name in stencils:
            stencil = stencils[lbl.name]
        elif lbl.name in modelnoise:
            stencils[lbl.name] = create_stencil_fn(lbl.name, evotype, state_space, target_labels=lbl.sslbls)
            stencil = stencils[lbl.name]
        else:
            return tuple(lbl.sslbls)  # simple cloud-key when there is no cloud noise

        #Otherwise, process stencil to get a list of all the qubit labels `lbl`'s cloudnoise error
        # touches and form this into a key
        cloud_lbls = set()
        for stencil_sslbls, local_errorgen in stencil.items():
            sslbls_list = StencilLabel.cast(stencil_sslbls).compute_absolute_sslbls(
                processor_spec.qubit_graph, state_space, lbl.sslbls)
            for sslbls in sslbls_list: cloud_lbls.update(sslbls if (sslbls is not None) else {})

        cloud_key = (tuple(lbl.sslbls), tuple(sorted(cloud_lbls)))  # (sets are unhashable)
        return cloud_key

    return _CloudNoiseModel(processor_spec, gatedict, global_idle_layer, prep_layers, povm_layers,
                            build_cloudnoise_fn, build_cloudkey_fn,
                            simulator, evotype, errcomp_type,
                            add_idle_noise_to_all_gates, printer)


def create_cloud_crosstalk_model_from_hops_and_weights(
        processor_spec, custom_gates=None,
        max_idle_weight=1, max_spam_weight=1,
        maxhops=0, extra_weight_1_hops=0, extra_gate_weight=0,
        simulator="auto", evotype='default',
        gate_type="H+S", spam_type="H+S",
        add_idle_noise_to_all_gates=True, errcomp_type="gates",
        independent_gates=True, connected_highweight_errors=True,
        verbosity=0):
        """
        TODO: update docstring
        Create a :class:`CloudNoiseModel` from hopping rules.

        Parameters
        ----------
        num_qubits : int
            The number of qubits

        gate_names : list
            A list of string-type gate names (e.g. `"Gx"`) either taken from
            the list of builtin "standard" gate names given above or from the
            keys of `nonstd_gate_unitaries`.  These are the typically 1- and 2-qubit
            gates that are repeatedly embedded (based on `availability`) to form
            the resulting model.

        nonstd_gate_unitaries : dict, optional
            A dictionary of numpy arrays which specifies the unitary gate action
            of the gate names given by the dictionary's keys.  As an advanced
            behavior, a unitary-matrix-returning function which takes a single
            argument - a tuple of label arguments - may be given instead of a
            single matrix to create an operation *factory* which allows
            continuously-parameterized gates.  This function must also return
            an empty/dummy unitary when `None` is given as it's argument.

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

        availability : dict, optional
            A dictionary whose keys are the same gate names as in
            `gatedict` and whose values are lists of qubit-label-tuples.  Each
            qubit-label-tuple must have length equal to the number of qubits
            the corresponding gate acts upon, and causes that gate to be
            embedded to act on the specified qubits.  For example,
            `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
            the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
            0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
            acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
            values of `availability` may take the special values:

            - `"all-permutations"` and `"all-combinations"` equate to all possible
            permutations and combinations of the appropriate number of qubit labels
            (deterined by the gate's dimension).
            - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
            edges, for 2Q gates of the geometry.
            - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
            on any target qubits via an :class:`EmbeddingOpFactory` (uses less
            memory but slower than `"all-permutations"`.

            If a gate name (a key of `gatedict`) is not present in `availability`,
            the default is `"all-edges"`.

        qubit_labels : tuple, optional
            The circuit-line labels for each of the qubits, which can be integers
            and/or strings.  Must be of length `num_qubits`.  If None, then the
            integers from 0 to `num_qubits-1` are used.

        geometry : {"line","ring","grid","torus"} or QubitGraph
            The type of connectivity among the qubits, specifying a
            graph used to define neighbor relationships.  Alternatively,
            a :class:`QubitGraph` object with node labels equal to
            `qubit_labels` may be passed directly.

        max_idle_weight : int, optional
            The maximum-weight for errors on the global idle gate.

        max_spam_weight : int, optional
            The maximum-weight for SPAM errors when `spamtype == "linblad"`.

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

        parameterization : str, optional
            Can be any Lindblad parameterization base type (e.g. CPTP,
            H+S+A, H+S, S, D, etc.) This is the type of parameterizaton to use in
            the constructed model.

        evotype : Evotype or str, optional
            The evolution type of this model, describing how states are
            represented.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        spamtype : { "static", "lindblad", "tensorproduct" }
            Specifies how the SPAM elements of the returned `Model` are formed.
            Static elements are ideal (perfect) operations with no parameters, i.e.
            no possibility for noise.  Lindblad SPAM operations are the "normal"
            way to allow SPAM noise, in which case error terms up to weight
            `max_spam_weight` are included.  Tensor-product operations require that
            the state prep and POVM effects have a tensor-product structure; the
            "tensorproduct" mode exists for historical reasons and is *deprecated*
            in favor of `"lindblad"`; use it only if you know what you're doing.

        add_idle_noise_to_all_gates : bool, optional
            Whether the global idle should be added as a factor following the
            ideal action of each of the non-idle gates.

        errcomp_type : {"gates","errorgens"}
            How errors are composed when creating layer operations in the created
            model.  `"gates"` means that the errors on multiple gates in a single
            layer are composed as separate and subsequent processes.  Specifically,
            the layer operation has the form `Composed(target,idleErr,cloudErr)`
            where `target` is a composition of all the ideal gate operations in the
            layer, `idleErr` is idle error (`.operation_blks['layers']['globalIdle']`),
            and `cloudErr` is the composition (ordered as layer-label) of cloud-
            noise contributions, i.e. a map that acts as the product of exponentiated
            error-generator matrices.  `"errorgens"` means that layer operations
            have the form `Composed(target, error)` where `target` is as above and
            `error` results from composing the idle and cloud-noise error
            *generators*, i.e. a map that acts as the exponentiated sum of error
            generators (ordering is irrelevant in this case).

        independent_clouds : bool, optional
            Currently this must be set to True.  In a future version, setting to
            true will allow all the clouds of a given gate name to have a similar
            cloud-noise process, mapped to the full qubit graph via a stencil.

        verbosity : int, optional
            An integer >= 0 dictating how must output to send to stdout.

        Returns
        -------
        CloudNoiseModel
        """

        # construct noise specifications for the cloudnoise model
        modelnoise = {}
        all_qubit_labels = processor_spec.qubit_labels
        conn = connected_highweight_errors  # shorthand: whether high-weight errors must be connected on the graph

        # Global Idle
        if max_idle_weight > 0:
            #printer.log("Creating Idle:")
            wt_maxhop_tuples = [(i, None) for i in range(1, max_idle_weight + 1)]
            modelnoise['idle'] = _build_weight_maxhops_modelnoise(all_qubit_labels, wt_maxhop_tuples, gate_type, conn)

        # SPAM
        if max_spam_weight > 0:
            wt_maxhop_tuples = [(i, None) for i in range(1, max_spam_weight + 1)]
            modelnoise['prep'] = _build_weight_maxhops_modelnoise(all_qubit_labels, wt_maxhop_tuples, spam_type, conn)
            modelnoise['povm'] = _build_weight_maxhops_modelnoise(all_qubit_labels, wt_maxhop_tuples, spam_type, conn)

        # Gates
        weight_maxhops_tuples_1Q = [(1, maxhops + extra_weight_1_hops)] + \
                                   [(1 + x, maxhops) for x in range(1, extra_gate_weight + 1)]
        # REMOVE cloud_maxhops_1Q = max([mx for wt, mx in weight_maxhops_tuples_1Q])  # max of max-hops

        weight_maxhops_tuples_2Q = [(1, maxhops + extra_weight_1_hops), (2, maxhops)] + \
                                   [(2 + x, maxhops) for x in range(1, extra_gate_weight + 1)]
        # REMOVE cloud_maxhops_2Q = max([mx for wt, mx in weight_maxhops_tuples_2Q])  # max of max-hops

        for gatenm, gate_unitary in processor_spec.gate_unitaries.items():
            gate_nQubits = int(round(_np.log2(gate_unitary.shape[0])))
            if gate_nQubits not in (1, 2):
                raise ValueError("Only 1- and 2-qubit gates are supported.  %s acts on %d qubits!"
                                 % (str(gatenm), gate_nQubits))
            weight_maxhops_tuples = weight_maxhops_tuples_1Q if gate_nQubits == 1 else weight_maxhops_tuples_2Q
            target_sslbls = ('@0',) if gate_nQubits == 1 else ('@0', '@1')
            modelnoise[gatenm] = _build_weight_maxhops_modelnoise(target_sslbls, weight_maxhops_tuples,
                                                                  gate_type, conn)

        # def build_modelnoise_fn(lbl):
        #     gate_nQubits = len(lbl.sslbls)
        #     if gate_nQubits not in (1, 2):
        #         raise ValueError("Only 1- and 2-qubit gates are supported.  %s acts on %d qubits!"
        #                          % (str(lbl.name), gate_nQubits))
        #     weight_maxhops_tuples = weight_maxhops_tuples_1Q if len(lbl.sslbls) == 1 else weight_maxhops_tuples_2Q
        #     return _build_weight_maxhops_modelnoise(lbl.sslbls, weight_maxhops_tuples, gate_type, conn)

        # def build_cloudkey_fn(lbl):
        #     cloud_maxhops = cloud_maxhops_1Q if len(lbl.sslbls) == 1 else cloud_maxhops_2Q
        #     cloud_inds = tuple(qubitGraph.radius(lbl.sslbls, cloud_maxhops))
        #     cloud_key = (tuple(lbl.sslbls), tuple(sorted(cloud_inds)))  # (sets are unhashable)
        #     return cloud_key

        return _create_cloud_crosstalk_model(processor_spec, modelnoise, custom_gates,
                                             evotype, simulator, independent_gates,
                                             errcomp_type, add_idle_noise_to_all_gates, verbosity)


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
    # loop over all size-`wt` *connected* combinations, `err_qubit_inds`, of the qubit indices in `possible_err_qubit_inds`
    #   - construct a local weight-`wt` Pauli basis & corresponding LindbladErrorgen on `wt` qubits
    #       => replace with: opnoise.create_errorgen(evotype, state_space=None)  where opnoise is for a wt-qubit op
    #   - embed this constructed local error onto `err_qubit_inds`
    #   - append embedded error onto running list
    #
    # Noise object structure:
    #  OpModelPerOpNoise( { op_key/'idle': { sslbls : opnoise } } )
    #     where sslbls can be absolute labels or stencil labels
    # -- could have a fn that spreads a single opnoise onto all the sslbls
    #      given by size-`wt` connected combos of `possible_err_qubit_inds` - this would work for independent clouds/gates
    # -- have LindbladNoiseDict and another LindbladPauliAtWeight (?) noise objects,
    #     since we want to specify a lindblad noise by giving a weight and an initial basis (Pauli here)

    # To build a cloudnoise model from hops & weights:
    modelnoise_dict = {}
    for wt, max_hops in weight_maxhops_tuples:
        if max_hops is None or max_hops == 0:  # Note: maxHops not used in this case
            stencil_lbl = StencilLabelAllCombos(target_sslbls, wt, connected)
        else:
            stencil_lbl = StencilLabelRadiusCombos(target_sslbls, max_hops, wt, connected)

        local_state_space = _statespace.default_space_for_num_qubits(wt)
        modelnoise_dict[stencil_lbl] = LindbladNoise.from_basis_coefficients(
            lnd_parameterization, _construct_restricted_weight_pauli_basis(wt),
            local_state_space)
    return modelnoise_dict


# def _build_nqn_global_noise(qubit_graph, max_weight, sparse_lindblad_basis=False, sparse_lindblad_reps=False,
#                             simulator=None, parameterization="H+S", evotype='default', errcomp_type="gates",
#                             verbosity=0):
#     """
#     Create a "global" idle gate, meaning one that acts on all the qubits in
#     `qubit_graph`.  The gate will have up to `max_weight` errors on *connected*
#     (via the graph) sets of qubits.
#
#     Parameters
#     ----------
#     qubit_graph : QubitGraph
#         A graph giving the geometry (nearest-neighbor relations) of the qubits.
#
#     max_weight : int
#         The maximum weight errors to include in the resulting gate.
#
#     sparse_lindblad_basis : bool, optional
#         Whether the embedded Lindblad-parameterized gates within the constructed
#         gate are represented as sparse or dense matrices.  (This is determied by
#         whether they are constructed using sparse basis matrices.)
#
#     sparse_lindblad_reps : bool, optional
#         Whether created Lindblad operations use sparse (more memory efficient but
#         slower action) or dense representations.
#
#     simulator : ForwardSimulator
#         The forward simulation (probability computation) being used by
#         the model this gate is destined for. `None` means a :class:`MatrixForwardSimulator`.
#
#     parameterization : str
#         The type of parameterizaton for the constructed gate. E.g. "H+S",
#         "CPTP", etc.
#
#     evotype : Evotype or str, optional
#         The evolution type.  The special value `"default"` is equivalent
#         to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
#
#     errcomp_type : {"gates","errorgens"}
#         How errors are composed when creating layer operations in the associated
#         model.  See :method:`CloudnoiseModel.__init__` for details.
#
#     verbosity : int, optional
#         An integer >= 0 dictating how must output to send to stdout.
#
#     Returns
#     -------
#     LinearOperator
#     """
#     assert(max_weight <= 2), "Only `max_weight` equal to 0, 1, or 2 is supported"
#     if simulator is None: simulator = _MatrixFSim()
#
#     prefer_dense_reps = isinstance(simulator, _MatrixFSim)
#     evotype = _Evotype.cast(evotype, prefer_dense_reps)
#
#     if errcomp_type == "gates":
#         Composed = _op.ComposedOp
#         Embedded = _op.EmbeddedOp
#     elif errcomp_type == "errorgens":
#         Composed = _op.ComposedErrorgen
#         Embedded = _op.EmbeddedErrorgen
#     else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
#     ExpErrorgen = _get_experrgen_factory(simulator, parameterization, errcomp_type, evotype)
#     #constructs a gate or errorgen based on value of errcomp_type
#
#     printer = _VerbosityPrinter.create_printer(verbosity)
#     printer.log("*** Creating global idle ***")
#
#     termops = []  # gates or error generators to compose
#     qubit_labels = qubit_graph.node_names
#     state_space = _statespace.QubitSpace(qubit_labels)
#
#     nQubits = qubit_graph.nqubits
#     possible_err_qubit_inds = _np.arange(nQubits)
#     nPossible = nQubits
#     for wt in range(1, max_weight + 1):
#         printer.log("Weight %d: %d possible qubits" % (wt, nPossible), 2)
#         basisEl_Id = basis_product_matrix(_np.zeros(wt, _np.int64), sparse_lindblad_basis)
#         if errcomp_type == "gates":
#             wtNoErr = _sps.identity(4**wt, 'd', 'csr') if sparse_lindblad_basis else _np.identity(4**wt, 'd')
#         elif errcomp_type == "errorgens":
#             wtNoErr = _sps.csr_matrix((4**wt, 4**wt)) if sparse_lindblad_basis else _np.zeros((4**wt, 4**wt), 'd')
#         else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
#         wtBasis = _BuiltinBasis('pp', 4**wt, sparse=sparse_lindblad_basis)
#
#         for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
#             if len(err_qubit_inds) == 2 and not qubit_graph.is_directly_connected(qubit_labels[err_qubit_inds[0]],
#                                                                                   qubit_labels[err_qubit_inds[1]]):
#                 continue  # TO UPDATE - check whether all wt indices are a connected subgraph
#
#             errbasis = [basisEl_Id]
#             errbasis_lbls = ['I']
#             for err_basis_inds in _iter_basis_inds(wt):
#                 error = _np.array(err_basis_inds, _np.int64)  # length == wt
#                 basisEl = basis_product_matrix(error, sparse_lindblad_basis)
#                 errbasis.append(basisEl)
#                 errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))
#
#             printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds, len(errbasis)), 3)
#             errbasis = _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse_lindblad_basis)
#             termErr = ExpErrorgen(wtNoErr, proj_basis=errbasis, mx_basis=wtBasis)
#
#             err_qubit_global_inds = err_qubit_inds
#             fullTermErr = Embedded(state_space, [qubit_labels[i] for i in err_qubit_global_inds], termErr)
#             assert(fullTermErr.num_params == termErr.num_params)
#             printer.log("Exp(errgen) gate w/nqubits=%d and %d params -> embedded to gate w/nqubits=%d" %
#                         (termErr.state_space.num_qubits, termErr.num_params, fullTermErr.state_space.num_qubits))
#
#             termops.append(fullTermErr)
#
#     if errcomp_type == "gates":
#         return Composed(termops)
#     elif errcomp_type == "errorgens":
#         errgen = Composed(termops)
#         #assert(not(sparse_lindblad_reps and isinstance(simulator, _MatrixFSim))), \
#         #    "Cannot use sparse ExpErrorgen-op reps with a MatrixForwardSimulator!"
#         return _op.ExpErrorgenOp(None, errgen)
#     else: assert(False)
#
#
# def _build_nqn_cloud_noise(target_qubit_inds, qubit_graph, weight_maxhops_tuples,
#                            errcomp_type="gates", sparse_lindblad_basis=False, sparse_lindblad_reps=False,
#                            simulator=None, parameterization="H+S", evotype='default', verbosity=0):
#     """
#     Create an n-qubit gate that is a composition of:
#
#     `target_op(target_qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)`
#
#     where `idle_noise` is given by the `idle_noise` argument and `loc_noise` is
#     given by the rest of the arguments.  `loc_noise` can be implemented either
#     by a single (n-qubit) embedded exp(errorgen) gate with all relevant error
#     generators, or as a composition of embedded single-errorgenerator exp(errorgen) gates
#     (see param `errcomp_type`).
#
#     The local noise consists terms up to a maximum weight acting on the qubits
#     given reachable by a given maximum number of hops (along the neareset-
#     neighbor edges of `qubit_graph`) from the target qubits.
#
#
#     Parameters
#     ----------
#     target_qubit_inds : list
#         The indices of the target qubits.
#
#     qubit_graph : QubitGraph
#         A graph giving the geometry (nearest-neighbor relations) of the qubits.
#
#     weight_maxhops_tuples : iterable
#         A list of `(weight,maxhops)` 2-tuples specifying which error weights
#         should be included and what region of the graph (as a `maxhops` from
#         the set of target qubits) should have errors of the given weight applied
#         to it.
#
#     errcomp_type : {"gates","errorgens"}
#         How errors are composed when creating layer operations in the associated
#         model.  See :method:`CloudnoiseModel.__init__` for details.
#
#     sparse_lindblad_basis : bool, optional
#         TODO - update docstring and probabaly rename this and arg below
#         Whether the embedded Lindblad-parameterized gates within the constructed
#         gate are represented as sparse or dense matrices.  (This is determied by
#         whether they are constructed using sparse basis matrices.)
#
#     sparse_lindblad_reps : bool, optional
#         Whether created Lindblad operations use sparse (more memory efficient but
#         slower action) or dense representations.
#
#     simulator : ForwardSimulator
#         The forward simulation (probability computation) being used by
#         the model this gate is destined for. `None` means a :class:`MatrixForwardSimulator`.
#
#     parameterization : str
#         The type of parameterizaton for the constructed gate. E.g. "H+S",
#         "CPTP", etc.
#
#     evotype : Evotype or str, optional
#         The evolution type.  The special value `"default"` is equivalent
#         to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
#
#     verbosity : int, optional
#         An integer >= 0 dictating how must output to send to stdout.
#
#     Returns
#     -------
#     LinearOperator
#     """
#     if simulator is None: simulator = _MatrixFSim()
#
#     prefer_dense_reps = isinstance(simulator, _MatrixFSim)
#     evotype = _Evotype.cast(evotype, prefer_dense_reps)
#
#     if errcomp_type == "gates":
#         Composed = _op.ComposedOp
#         Embedded = _op.EmbeddedOp
#     elif errcomp_type == "errorgens":
#         Composed = _op.ComposedErrorgen
#         Embedded = _op.EmbeddedErrorgen
#     else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
#     ExpErrorgen = _get_experrgen_factory(simulator, parameterization, errcomp_type, evotype)
#     #constructs a gate or errorgen based on value of errcomp_type
#
#     printer = _VerbosityPrinter.create_printer(verbosity)
#     printer.log("Creating local-noise error factor (%s)" % errcomp_type)
#
#     # make a composed-gate of embedded single-elementary-errogen exp(errogen)-gates or -errorgens,
#     #  one for each specified error term
#
#     loc_noise_termops = []  # list of gates to compose
#     qubit_labels = qubit_graph.node_names
#     state_space = _statespace.QubitSpace(qubit_labels)
#
#     for wt, maxHops in weight_maxhops_tuples:
#
#         ## loc_noise_errinds = [] # list of basis indices for all local-error terms
#         radius_nodes = qubit_graph.radius([qubit_labels[i] for i in target_qubit_inds], maxHops)
#         possible_err_qubit_inds = _np.array([qubit_labels.index(nn) for nn in radius_nodes], _np.int64)
#         nPossible = len(possible_err_qubit_inds)  # also == "nLocal" in this case
#         basisEl_Id = basis_product_matrix(_np.zeros(wt, _np.int64), sparse_lindblad_basis)  # identity basis el
#
#         if errcomp_type == "gates":
#             wtNoErr = _sps.identity(4**wt, 'd', 'csr') if sparse_lindblad_basis else _np.identity(4**wt, 'd')
#         elif errcomp_type == "errorgens":
#             wtNoErr = _sps.csr_matrix((4**wt, 4**wt)) if sparse_lindblad_basis else _np.zeros((4**wt, 4**wt), 'd')
#         else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
#         wtBasis = _BuiltinBasis('pp', 4**wt, sparse=sparse_lindblad_basis)
#
#         printer.log("Weight %d, max-hops %d: %d possible qubits" % (wt, maxHops, nPossible), 3)
#         # print("DB: possible qubits = ", possible_err_qubit_inds,
#         #       " (radius of %d around %s)" % (maxHops,str(target_qubit_inds)))
#
#         for err_qubit_local_inds in _itertools.combinations(list(range(nPossible)), wt):
#             # err_qubit_inds are in range [0,nPossible-1] qubit indices
#             #Future: check that err_qubit_inds marks qubits that are connected
#
#             errbasis = [basisEl_Id]
#             errbasis_lbls = ['I']
#             for err_basis_inds in _iter_basis_inds(wt):
#                 error = _np.array(err_basis_inds, _np.int64)  # length == wt
#                 basisEl = basis_product_matrix(error, sparse_lindblad_basis)
#                 errbasis.append(basisEl)
#                 errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))
#
#             err_qubit_global_inds = possible_err_qubit_inds[list(err_qubit_local_inds)]
#             printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_global_inds, len(errbasis)), 4)
#             errbasis = _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse_lindblad_basis)
#             termErr = ExpErrorgen(wtNoErr, proj_basis=errbasis, mx_basis=wtBasis, relative=True)
#
#             fullTermErr = Embedded(state_space, [qubit_labels[i] for i in err_qubit_global_inds], termErr)
#             assert(fullTermErr.num_params == termErr.num_params)
#             printer.log("Exp(errorgen) gate w/nqubits=%d and %d params -> embedded to gate w/nqubits=%d" %
#                         (termErr.state_space.num_qubits, termErr.num_params, fullTermErr.state_space.num_qubits))
#
#             loc_noise_termops.append(fullTermErr)
#
#     fullCloudErr = Composed(loc_noise_termops)
#     return fullCloudErr
