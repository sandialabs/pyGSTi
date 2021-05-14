"""
Sub-package holding model POVM and POVM effect objects.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .complementeffect import ComplementPOVMEffect
from .composedeffect import ComposedPOVMEffect
from .computationaleffect import ComputationalBasisPOVMEffect
from .conjugatedeffect import ConjugatedStatePOVMEffect
#from .denseeffect.py  # REMOVE
from .effect import POVMEffect
from .fulleffect import FullPOVMEffect
from .fullpureeffect import FullPOVMPureEffect
from .staticeffect import StaticPOVMEffect
from .tensorprodeffect import TensorProductPOVMEffect

from .composedpovm import ComposedPOVM
from .computationalpovm import ComputationalBasisPOVM
from .marginalizedpovm import MarginalizedPOVM
from .povm import POVM
from .tensorprodpovm import TensorProductPOVM
from .tppovm import TPPOVM
from .unconstrainedpovm import UnconstrainedPOVM


import numpy as _np
import itertools as _itertools
import functools as _functools
from ...tools import optools as _ot
from ...tools import basistools as _bt


def convert(povm, to_type, basis, extra=None):
    """
    Convert a POVM to a new type of parameterization.

    This potentially creates a new object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    povm : POVM
        POVM to convert

    to_type : {"full","TP","static","static unitary","H+S terms",
        "H+S clifford terms","clifford"}
        The type of parameterizaton to convert to.  See
        :method:`Model.set_all_parameterizations` for more details.
        TODO docstring: update the options here.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Unused.

    Returns
    -------
    POVM
        The converted POVM vector, usually a distinct
        object from the object passed as input.
    """
    if to_type in ("full", "static", "static unitary"):
        converted_effects = [(lbl, convert_effect(vec, to_type, basis))
                             for lbl, vec in povm.items()]
        return UnconstrainedPOVM(converted_effects)

    elif to_type == "TP":
        if isinstance(povm, TPPOVM):
            return povm  # no conversion necessary
        else:
            converted_effects = [(lbl, convert_effect(vec, "full", basis))
                                 for lbl, vec in povm.items()]
            return TPPOVM(converted_effects)

    elif _ot.is_valid_lindblad_paramtype(to_type):

        # A LindbladPOVM needs a *static* base/reference POVM
        #  with the appropriate evotype.  If we can convert `povm` to such a
        #  thing we win.  (the error generator is initialized as just the identity below)

        nQubits = int(round(_np.log2(povm.dim) / 2.0))  # Linblad ops always work on density-matrices, never states
        bQubits = bool(_np.isclose(nQubits, _np.log2(povm.dim) / 2.0))  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis

        _, evotype = _ot.split_lindblad_paramtype(to_type)

        if isinstance(povm, ComputationalBasisPOVM):  # special easy case
            assert(povm.nqubits == nQubits)
            base_povm = ComputationalBasisPOVM(nQubits, evotype)
        else:
            base_items = [(lbl, _convert_to_static_effect(Evec, evotype, basis))
                          for lbl, Evec in povm.items()]
            base_povm = UnconstrainedPOVM(base_items)

        # purevecs = extra if (extra is not None) else None # UNUSED
        cls = _op.LindbladDenseOp if (povm.dim <= 64 and evotype == "densitymx") \
            else _op.LindbladOp
        povmNoiseMap = cls.from_operation_obj(_np.identity(povm.dim, 'd'), to_type,
                                              None, proj_basis, basis, truncate=True)
        return ComposedPOVM(povmNoiseMap, base_povm, basis)

    elif to_type == "clifford":
        if isinstance(povm, ComputationalBasisPOVM) and povm._evotype == "stabilizer":
            return povm

        #OLD
        ##Try to figure out whether this POVM acts on states or density matrices
        #if any([ (isinstance(Evec,DenseSPAMVec) and _np.iscomplexobj(Evec.base)) # PURE STATE?
        #         for Evec in povm.values()]):
        #    nqubits = int(round(_np.log2(povm.dim)))
        #else:
        #    nqubits = int(round(_np.log2(povm.dim))) // 2

        #Assume `povm` already represents state-vec ops, since otherwise we'd
        # need to change dimension
        nqubits = int(round(_np.log2(povm.dim)))

        #Check if `povm` happens to be a Z-basis POVM on `nqubits`
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1) - eigenstates of sigma_z
        for zvals, lbl in zip(_itertools.product(*([(0, 1)] * nqubits)), povm.keys()):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if not _np.allclose(testvec, povm[lbl].to_dense()):
                raise ValueError("Cannot convert POVM into a Z-basis stabilizer state POVM")

        #If no errors, then return a stabilizer POVM
        return ComputationalBasisPOVM(nqubits, 'stabilizer')

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


def _convert_to_static_effect(effect, new_evotype, mx_basis="pp"):
    """
    Attempts to convert `vec` to a static (0 params) SPAMVec with
    evoution type `new_evotype`.  Used to convert spam vecs to
    being LindbladSPAMVec objects.
    """
    if effect.evotype == new_evotype and effect.num_params == 0:
        return effect  # no conversion necessary

    #First, check if it's a computational effect, which is easy to convert evotypes of:
    if isinstance(effect, ComputationalBasisPOVMEffect):
        return ComputationalBasisPOVMEffect(effect._zvals, new_evotype)

    #Next, try to construct a pure-state if possible
    #if isinstance(effect, ConjugatedStatePOVMEffect):
    try:
        dmvec = _bt.change_basis(effect.to_dense(), mx_basis, 'std')
        purevec = _ot.dmvec_to_state(dmvec)
        return StaticPOVMPureEffect(purevec, new_evotype)
    except:
        return StaticPOVMEffect(effect.to_dense(), new_evotype)


def convert_effect(effect, to_type, basis, extra=None):
    """
    TODO: update docstring
    Convert SPAM vector to a new type of parameterization.

    This potentially creates a new SPAMVec object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    to_type : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `spamvec`.  Allowed values are Matrix-unit (std),
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
        if isinstance(effect, FullPOVMEffect):
            return effect  # no conversion necessary
        else:
            return FullPOVMEffect(effect.to_dense())

    elif to_type == "static":
        if isinstance(effect, StaticPOVMEffect):
            return effect  # no conversion necessary
        else:
            return StaticPOVMEffect(effect.to_dense())

    elif to_type == "static unitary":
        dmvec = _bt.change_basis(effect.to_dense(), basis, 'std')
        purevec = _ot.dmvec_to_state(dmvec)
        return StaticPOVMPureEffect(purevec)

    elif _ot.is_valid_lindblad_paramtype(to_type):

        if extra is None:
            purevec = effect  # right now, we don't try to extract a "closest pure vec"
            # to effect - below will fail if effect isn't pure.
        else:
            purevec = extra  # assume extra info is a pure vector

        nQubits = _np.log2(effect.dim) / 2.0
        bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis
        typ = effect._prep_or_effect if isinstance(effect, POVMEffect) else "prep"

        return ComposedPOVMEffect._from_effect_obj(
            effect, typ, to_type, None, proj_basis, basis,
            truncate=True, lazy=True)  # TODO ----------------------------------------------------------------------------------

    # "clifford" is more of an evotype than a parameterization...
    #elif to_type == "clifford":
    #    if isinstance(effect, StabilizerEffect):
    #        return effect  # no conversion necessary
    #
    #    purevec = effect.flatten()  # assume a pure state (otherwise would
    #    # need to change Model dim)
    #    return StabilizerEffect.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)
