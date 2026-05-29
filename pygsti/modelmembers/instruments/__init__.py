"""
Sub-package holding model instrument objects.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from typing import Optional, Union

from .instrument import Instrument
from .tpinstrument import TPInstrument
from .tpinstrumentop import TPInstrumentOp

import warnings as _warnings
import scipy.linalg as _la
import numpy as _np
from pygsti.tools import optools as _ot, basistools as _bt
from pygsti.tools.exceptions import DubiousTargetWarning as _DubiousTargetWarning
from pygsti.baseobjs.label import Label
from pygsti.baseobjs.basis import Basis, BasisLike
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _pv
from pygsti.modelmembers.povms.basepovm import _BasePOVM


def instrument_type_from_op_type(op_type):
    """Decode an op type into an appropriate instrument type.

    Parameters:
    -----------
    op_type: str or list of str
        Operation parameterization type (or list of preferences)

    Returns
    -------
    instr_type_preferences: tuple of str
        POVM parameterization types
    """
    op_type_preferences = _op.verbose_type_from_op_type(op_type)
    return op_type_preferences[0]


def convert(instrument, to_type, basis, ideal_instrument=None, flatten_structure=False):
    """
    TODO: update docstring
    Convert intrument to a new type of parameterization.

    This potentially creates a new object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    instrument : Instrument
        Instrument to convert

    to_type : {"full","TP","static","static unitary"}
        The type of parameterizaton to convert to.  See
        :meth:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    ideal_instrument : Instrument, optional
        The ideal version of `instrument`, potentially used when
        converting to an error-generator type.

    flatten_structure : bool, optional
        When `False`, the sub-members of composed and embedded operations
        are separately converted, leaving the original instrument's structure
        unchanged.  When `True`, composed and embedded operations are "flattened"
        into a single instrument of the requested `to_type`.

    Returns
    -------
    Instrument
        The converted instrument, usually a distinct
        object from the object passed as input.
    """
    if ideal_instrument is None:
        ideal_instrument = dict()

    if not isinstance(to_type, str):
        if len(to_type) > 1:
            raise ValueError(f"Expected to_type to be a string, but got {to_type}")
        to_type = to_type[0]
        assert isinstance(to_type, str)
    
    if to_type == "full TP":
        if isinstance(instrument, TPInstrument):
            return instrument
        inst_arrays = dict()
        for k, v in instrument.items():
            if hasattr(v, 'to_dense'):
                inst_arrays[k] = v.to_dense('HilbertSchmidt')
            else:
                inst_arrays[k] = v
        members = list(inst_arrays.items())
        return TPInstrument(members, instrument.evotype, instrument.state_space)
    
    if to_type in ("full", "static", "static unitary"):
        members = []
        for k, g in instrument.items():
            g_ideal = ideal_instrument.get(k, None)
            g_conv  = _op.convert(g, to_type, basis, g_ideal, flatten_structure)
            members.append((k, g_conv))
        return Instrument(members, instrument.evotype, instrument.state_space)

    # Else, we're falling back on the operations.convert(...) function inside kraus_polar_instrument,
    # which will be called on StaticUnitary channels.
    op_arrays          = {k: v.to_dense('HilbertSchmidt') for (k,v) in instrument.items()}
    post_unitary_error = to_type
    povm_errormap      = _op.LindbladParameterization.minimal_cp_paramtype(to_type)
    inst = kraus_polar_instrument(
        op_arrays, basis,
        post_unitary_error=post_unitary_error,
        povm_errormap=povm_errormap
    )
    return inst



ErrorMapSpec = str


def kraus_polar_instrument(
        op_arrays: dict[str, _np.ndarray], basis: BasisLike,
        error_tol: float = 1e-6, trunc_tol: float = 1e-7,
        post_unitary_error: ErrorMapSpec='CPTPLND',
        povm_errormap: Optional[Union[_op.LinearOperator, str]]='CPTPLND'  # type: ignore
    ) -> Instrument:
    """
    Construct an appropriately parameterized Instrument from the `op_arrays` dict,
    which holds CPTR maps as dense arrays with respect to `basis`.

    Our approach is as follows.

        For a given CPTR operator, we compute its Kraus representation and
        then polar-decompose its Kraus operators into their unitary and psd
        parts. Each unitary is then promoted to a noisy (parameterized)
        channel and each psd part is cast to a static POVMEffect.
        
        The condition that the CPTR operators sum to a TP map can be enforced
        by saying these POVM effects sum to the identity. Therefore we can
        encode this constraint in *parameterized* CPTR operators by saying that
        the effects in their polar-decomposed Kraus operators all belong to a 
        single parameterized POVM. We use a ComposedPOVM for that purpose,
        with an error channel specified by `povm_errormap`. It's important that
        the parameterization be CPTP by construction.

        The Instrument class requires that the CPTR operators are LinearOperator
        objects. We do this with ComposedOp and possibily SummedOperator. The
        latter class only comes into play if an input CPTR operator has Kraus
        rank > 1, which is unusual but not forbidden.

    Parameters
    ----------
    op_arrays : dict[str, numpy.ndarray]
        Mapping from outcome label to dense superoperator matrix (in the given basis).
        Each matrix must represent a CPTR map; the function raises or warns if the
        Kraus decomposition is inconsistent with this.

    basis : BasisLike
        A Basis object (or string identifier for a built-in basis) specifying how
        op_arrays are represented.

    error_tol : float, optional
        Tolerance for eigenvalue errors in the minimal Kraus decomposition.  Eigenvalues
        below ``-error_tol`` cause an error to be raised.  Default ``1e-6``.

    trunc_tol : float, optional
        Truncation tolerance for the minimal Kraus decomposition.  Eigenvalues between
        ``-error_tol`` and ``trunc_tol`` are silently set to zero.  Default ``1e-7``.

    post_unitary_error : ErrorMapSpec
        Specifies how we parameterize post-unitary error channels. We do not require 
        that the parameterization ensures completely positivity.

    povm_errormap : LinearOperator or str
        Either a LinearOperator that's CPTP by construction or a string specification
        for a such an operator in the error generator formalism. 

    Returns
    -------
    Instrument
        An ``Instrument`` whose member operations are CPTR maps sharing a common set
        of CPTPLND parameters.  The instrument is not itself a ``TPInstrument`` because
        the completeness constraint is enforced implicitly through the shared POVM.
    """
    udim = round(basis.dim ** 0.5)
    I_hilbert = _np.eye(udim, dtype='complex')

    if not isinstance(povm_errormap, _op.LinearOperator):
        assert isinstance(povm_errormap, str)
        I_static = _op.StaticUnitaryOp(I_hilbert, basis)          # type: ignore
        I_param  = _op.convert(I_static, povm_errormap, basis)
        povm_errormap : _op.LinearOperator = I_param.factorops[1] # type: ignore  

    per_cptr_unitaries = dict()
    shared_effects     = dict()

    # Kraus decompose each CPTR operator and polar decompose each Kraus operator.
    # Define the parameterized unitaries and static POVM effects as we go.
    
    effects_sum = _np.zeros((udim, udim), dtype='complex')
    for oplbl, op in op_arrays.items():
        krausops = _ot.minimal_kraus_decomposition(op, basis, error_tol, trunc_tol)

        if len(krausops) > 1:
            msg = f"Target CPTR operator {oplbl} has Kraus rank {len(krausops)} > 1."
            _warnings.warn(msg, _DubiousTargetWarning)

        per_cptr_unitaries[oplbl] = []
        for i, K in enumerate(krausops):
            u, root_p = _la.polar(K, side='right')
            u_static = _op.StaticUnitaryOp(u, basis)  # type: ignore
            u_param  = _op.convert(u_static, post_unitary_error, basis)
            per_cptr_unitaries[oplbl].append(u_param)
            p = root_p @ root_p
            effects_sum += p
            E_superket = _bt.stdmx_to_vec(p, basis)
            E_static   = _pv.StaticPOVMEffect(E_superket)
            shared_effects[Label((oplbl, i))] = E_static

    if not _np.allclose(effects_sum, I_hilbert):
        raise ValueError('Values in op_arrays dict do not sum to a TP channel.')

    M_static = _BasePOVM(shared_effects)
    M_param  = _pv.ComposedPOVM(povm_errormap, M_static)

    # Stitch the unitaries and root-conj channels together into instrument operations.
    inst_ops : dict[str, Union[_op.ComposedOp, _op.SummedOperator]] = dict()
    for lbl in op_arrays:
        op_unitaries = per_cptr_unitaries[lbl]
        op_summands  = []
        for i, U_param in enumerate(op_unitaries):
            E_param = M_param[Label((lbl, i))]
            summand = _op.ComposedOp([ _op.RootConjOperator(E_param, basis), U_param ])
            op_summands.append(summand)
        if len(op_summands) == 1:
            inst_ops[lbl] = op_summands[0]
        else:
            inst_ops[lbl] = _op.SummedOperator(op_summands, basis)

    inst = Instrument(inst_ops)

    return inst


def instrument_superops_from_effects(members, basis: BasisLike,
                                     atol: float = 1e-6) -> dict[str, _np.ndarray]:
    """
    Build instrument member superoperators from a ``{label: (effect, gate)}`` mapping.

    This is a convenience for the common case in which an (ideal) instrument is
    described as a collection of (effect, gate, label) triples.  Each member is
    the composition

        I_k : rho  |->  G_k( E_k^{1/2} rho E_k^{1/2} ),

    i.e. a "soft" measurement of the POVM effect E_k followed by the
    trace-preserving post-measurement map G_k.

    This function assembles the member superoperators `G_k @ rootconj_superop(E_k)`
    for you, accepting a variety of representations for the effect and gate
    (including omitting the gate entirely) and performing conversions as needed.
    The returned mapping is exactly the ``op_arrays`` dict consumed by
    :func:`kraus_polar_instrument`, and can also be handed directly to the
    :class:`Instrument` constructor::

        superops  = instrument_superops_from_effects({'p0': E0, 'p1': E1}, basis)
        inst      = Instrument(superops)                     # plain / dense
        inst_cptp = kraus_polar_instrument(superops, basis)  # CP-constrained

    Parameters
    ----------
    members : dict

        Maps each outcome label to either an ``(effect, gate)`` pair or a bare
        ``effect`` (in which case the gate defaults to the identity).

        - ``effect`` may be a POVM-effect superket (a length-``d**2`` vector in ``basis``),
          a Hermitian ``d x d`` matrix, or any object with a ``to_dense()`` method
          (e.g., a :class:`POVMEffect`).  We require ``0 <= E_k <= I``.

        - ``gate`` may be a ``d**2 x d**2`` superoperator (in ``basis``), a
          ``d x d`` unitary, ``None`` (the identity), or any object with a
          ``to_dense('HilbertSchmidt')`` method (e.g., a :class:`LinearOperator`).

    basis : Basis or str
        The basis in which dense arrays are represented.

    atol : float, optional
        Absolute tolerance for the per-gate trace-preservation check and the
        completeness check ``sum_k E_k == I``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from each outcome label to its dense member superoperator (in ``basis``).
    """
    def _effect_of(val):
        return val[0] if isinstance(val, tuple) else val

    def _gate_of(val):
        return val[1] if isinstance(val, tuple) and len(val) > 1 else None

    if isinstance(basis, str):
        # Infer the Hilbert-Schmidt dimension from the first effect.
        first = _effect_of(next(iter(members.values())))
        arr = _np.asarray(first.to_dense() if hasattr(first, 'to_dense') else first)
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            dim = arr.shape[0]
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            dim = arr.shape[0] ** 2   # a Hermitian (udim x udim) matrix
        else:
            raise ValueError("Could not infer the dimension from the effects; "
                             "pass `basis` as a Basis object instead of a string.")
        basis = Basis.cast(basis, dim)
    else:
        basis = Basis.cast(basis)  # makes linter happy (view `basis` a Basis object)

    dim = basis.dim
    udim = round(dim ** 0.5)
    I_hilbert = _np.eye(udim)
    I_superket = _bt.stdmx_to_vec(I_hilbert, basis).reshape(-1)  # <<I| extracts the trace

    def as_superket(effect):
        arr = _np.asarray(effect.to_dense() if hasattr(effect, 'to_dense') else effect)
        if arr.ndim == 2 and arr.shape == (udim, udim):
            arr = _bt.stdmx_to_vec(arr, basis)   # Hermitian d x d matrix -> superket
        return arr.reshape(-1)

    def as_superop(gate):
        if gate is None:
            return _np.eye(dim)
        if hasattr(gate, 'to_dense'):
            return _np.asarray(gate.to_dense('HilbertSchmidt'))
        arr = _np.asarray(gate)
        if arr.shape == (udim, udim):
            return _ot.unitary_to_superop(arr, basis)   # type: ignore  # unitary -> superop
        if arr.shape == (dim, dim):
            return arr
        raise ValueError(f"Gate has shape {arr.shape}; expected a ({dim}, {dim}) "
                         f"superoperator or a ({udim}, {udim}) unitary.")

    inst_arrays = dict()
    effect_sum = _np.zeros((udim, udim), dtype=complex)
    for label, val in members.items():
        E_superket = as_superket(_effect_of(val))
        G_superop = as_superop(_gate_of(val))
        if not _np.allclose(I_superket @ G_superop, I_superket, atol=atol):
            raise ValueError(f"The post-measurement gate for outcome {label!r} is not TP.")
        # rootconj_superop validates 0 <= E_k <= I and raises/warns otherwise.
        inst_arrays[label] = G_superop @ _ot.rootconj_superop(E_superket, basis)
        effect_sum += _bt.vec_to_stdmx(E_superket, basis, keep_complex=True)

    if not _np.allclose(effect_sum, I_hilbert, atol=atol):
        raise ValueError("The provided effects do not sum to the identity; an "
                         "instrument's effects must satisfy sum_k E_k == I.")

    return inst_arrays
