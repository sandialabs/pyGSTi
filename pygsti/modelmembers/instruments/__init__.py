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
from types import NoneType
from pygsti.baseobjs.label import Label
from pygsti.baseobjs.basis import BasisLike
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _pv
from pygsti.modelmembers.povms.basepovm import _BasePOVM

# Avoid circular import
import pygsti.modelmembers as _mm


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
    op_type_preferences = _mm.operations.verbose_type_from_op_type(op_type)

    # Limited set (only matching what is in convert)
    instr_conversion = {
        'auto': 'full TP',
        'static unitary': 'static unitary',
        'static clifford': 'static clifford',
        'static': 'static',
        'full': 'full',
        'full TP': 'full TP',
        'full CPTP': 'full CPTP',
        'full unitary': 'full unitary',
        'GLND': 'full TP',
        # ^ It's pretty harmless to associate GLND operations with "full TP"
        #   instruments. In both cases we're relaxing positivity constraints.
        'CPTPLND': 'CPTPLND'
    }

    instr_type_preferences = []
    for typ in op_type_preferences:
        instr_type = instr_conversion.get(typ, None)

        if instr_type is None and _ot.is_valid_lindblad_paramtype(typ):
            # NOTE: need to update the message below if more lindblad
            # types are added as keys to the instr_conversion dict.
            msg = \
            f"""
            Operation type {typ} is a Lindblad parameterization, but
            is neither 'GLND' or 'CPTPLND'. That means you might be
            asking for a reduced-order model that's a subset of all
            CPTP models. We don't support that parameterization at this
            time. We're falling back to a 'full TP' parameterization!
            """
            _warnings.warn(msg)
            instr_type = 'full TP' # non-CPTPLND falls back to full TP.

        if instr_type not in instr_type_preferences:
            instr_type_preferences.append(instr_type)

    if len(instr_type_preferences) == 0:
        raise ValueError(
            'Could not convert any op types from {}.\n'.format(op_type_preferences)
            + '\tKnown op_types: Lindblad types or {}\n'.format(sorted(list(instr_conversion.keys())))
            + '\tValid instrument_types: Lindblad types or {}'.format(sorted(list(set(instr_conversion.values()))))
        )

    return instr_type_preferences


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
    if not isinstance(to_type, str):
        if len(to_type) > 1:
            raise ValueError(f"Expected to_type to be a string, but got {to_type}")
        to_type = to_type[0]
        assert isinstance(to_type, str)

    destination_types = {'full TP': TPInstrument}
    
    if isinstance(instrument, destination_types.get( to_type, NoneType ) ):
        return instrument
    
    if to_type == "full TP":
        inst_arrays = dict()
        for k, v in instrument.items():
            if hasattr(v, 'to_dense'):
                inst_arrays[k] = v.to_dense('HilbertSchmidt')
            else:
                inst_arrays[k] = v
        return TPInstrument(list(inst_arrays.items()), instrument.evotype, instrument.state_space)
    
    if to_type in ("full", "static", "static unitary"):
        from ..operations import convert as _op_convert
        ideal_items = dict(ideal_instrument.items()) if (ideal_instrument is not None) else {}
        members = [(k, _op_convert(g, to_type, basis, ideal_items.get(k, None), flatten_structure))
                    for k, g in instrument.items()]
        return Instrument(members, instrument.evotype, instrument.state_space)

    if to_type == 'CPTPLND':
        op_arrays = {k: v.to_dense('HilbertSchmidt') for (k,v) in instrument.items()}
        inst = kraus_polar_instrument(op_arrays, basis)
        return inst

    raise ValueError("Cannot convert an instrument to type %s" % to_type)


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
