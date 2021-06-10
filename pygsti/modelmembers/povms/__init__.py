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

import functools as _functools
import itertools as _itertools

import numpy as _np

from .complementeffect import ComplementPOVMEffect
from .composedeffect import ComposedPOVMEffect
from .composedpovm import ComposedPOVM
from .computationaleffect import ComputationalBasisPOVMEffect
from .computationalpovm import ComputationalBasisPOVM
from .conjugatedeffect import ConjugatedStatePOVMEffect
# from .denseeffect.py  # REMOVE
from .effect import POVMEffect
from .fulleffect import FullPOVMEffect
from .fullpureeffect import FullPOVMPureEffect
from .marginalizedpovm import MarginalizedPOVM
from .povm import POVM
from .staticeffect import StaticPOVMEffect
from .staticpureeffect import StaticPOVMPureEffect
from .tensorprodeffect import TensorProductPOVMEffect
from .tensorprodpovm import TensorProductPOVM
from .tppovm import TPPOVM
from .unconstrainedpovm import UnconstrainedPOVM
from ...tools import basistools as _bt
from ...tools import optools as _ot


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
        return UnconstrainedPOVM(converted_effects, povm.evotype, povm.state_space)

    elif to_type == "TP":
        if isinstance(povm, TPPOVM):
            return povm  # no conversion necessary
        else:
            converted_effects = [(lbl, convert_effect(vec, "full", basis))
                                 for lbl, vec in povm.items()]
            return TPPOVM(converted_effects, povm.evotype, povm.state_space)

    elif _ot.is_valid_lindblad_paramtype(to_type):
        from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp

        #Construct a static "base" POVM
        if isinstance(povm, ComputationalBasisPOVM):  # special easy case
            base_povm = ComputationalBasisPOVM(povm.state_space.num_qubits, povm.evotype)  # just copy it?
        else:
            base_items = [(lbl, convert_effect(vec, 'static', basis)) for lbl, vec in povm.items()]
            base_povm = UnconstrainedPOVM(base_items, povm.evotype, povm.state_space)

        proj_basis = 'pp' if povm.state_space.is_entirely_qubits else basis
        nonham_mode, param_mode, use_ham_basis, use_nonham_basis = \
            _LindbladErrorgen.decomp_paramtype(to_type)
        ham_basis = proj_basis if use_ham_basis else None
        nonham_basis = proj_basis if use_nonham_basis else None

        errorgen = _LindbladErrorgen.from_error_generator(_np.zeros((povm.state_space.dim,
                                                                     povm.state_space.dim), 'd'),
                                                          ham_basis, nonham_basis, param_mode, nonham_mode,
                                                          basis, truncate=True, evotype=povm.evotype)
        return ComposedPOVM(_ExpErrorgenOp(errorgen), base_povm, mx_basis=basis)

    elif to_type == "static clifford":
        if isinstance(povm, ComputationalBasisPOVM):
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


#REMOVE (UNUSED) - but an example of evotype-changing code...
#def _convert_to_static_effect(effect, new_evotype, mx_basis="pp"):
#    """
#    Attempts to convert `vec` to a static (0 params) SPAMVec with
#    evoution type `new_evotype`.  Used to convert spam vecs to
#    being LindbladSPAMVec objects.
#    """
#    if effect.evotype == new_evotype and effect.num_params == 0:
#        return effect  # no conversion necessary
#
#    #First, check if it's a computational effect, which is easy to convert evotypes of:
#    if isinstance(effect, ComputationalBasisPOVMEffect):
#        return ComputationalBasisPOVMEffect(effect._zvals, new_evotype)
#
#    #Next, try to construct a pure-state if possible
#    #if isinstance(effect, ConjugatedStatePOVMEffect):
#    try:
#        dmvec = _bt.change_basis(effect.to_dense(), mx_basis, 'std')
#        purevec = _ot.dmvec_to_state(dmvec)
#        return StaticPOVMPureEffect(purevec, new_evotype)
#    except:
#        return StaticPOVMEffect(effect.to_dense(), new_evotype)


def convert_effect(effect, to_type, basis, extra=None):
    """
    TODO: update docstring
    Convert POVM effect vector to a new type of parameterization.

    This potentially creates a new POVMEffect object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    effect : POVMEffect
        POVM effect vector to convert

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
    POVMEffect
        The converted POVM effect vector, usually a distinct
        object from the object passed as input.
    """
    if to_type == "full":
        if isinstance(effect, FullPOVMEffect):
            return effect  # no conversion necessary
        else:
            return FullPOVMEffect(effect.to_dense(), effect.evotype, effect.state_space)

    elif to_type == "static":
        if isinstance(effect, StaticPOVMEffect):
            return effect  # no conversion necessary
        else:
            return StaticPOVMEffect(effect.to_dense(), effect.evotype, effect.state_space)

    elif to_type == "static unitary":
        dmvec = _bt.change_basis(effect.to_dense(), basis, 'std')
        purevec = _ot.dmvec_to_state(dmvec)
        return StaticPOVMPureEffect(purevec, basis, effect.evotype, effect.state_space)

    elif _ot.is_valid_lindblad_paramtype(to_type):

        from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
        from ..states import FullState as _FullState, TPState as _TPState, StaticState as _StaticState
        from ..states import StaticPureState as _StaticPureState

        pureeffect = None
        if isinstance(effect, ConjugatedStatePOVMEffect) \
           and isinstance(effect.state, (_FullState, _TPState, _StaticState)):
            #Similar to conversion for states
            state = effect.state
            try:
                dmvec = _bt.change_basis(state.to_dense(), basis, 'std')
                purevec = _ot.dmvec_to_state(dmvec)  # raises error if dmvec does not correspond to a pure state
                pureeffect = ConjugatedStatePOVMEffect(_StaticPureState(purevec, basis,
                                                                        state.evotype, state.state_space))
            except ValueError:
                pureeffect = None

        if pureeffect is not None:
            static_effect = pureeffect
        elif effect.num_params > 0:  # then we need to convert to a static state
            static_effect = StaticPOVMEffect(effect.to_dense(), effect.evotype, effect.state_space)
        else:  # state.num_params == 0 so it's already static
            static_effect = effect

        proj_basis = 'pp' if state.state_space.is_entirely_qubits else basis
        nonham_mode, param_mode, use_ham_basis, use_nonham_basis = \
            _LindbladErrorgen.decomp_paramtype(to_type)
        ham_basis = proj_basis if use_ham_basis else None
        nonham_basis = proj_basis if use_nonham_basis else None

        errorgen = _LindbladErrorgen.from_error_generator(_np.zeros((effect.state_space.dim,
                                                                     effect.state_space.dim), 'd'),
                                                          ham_basis, nonham_basis, param_mode, nonham_mode,
                                                          basis, truncate=True, evotype=effect.evotype)
        return ComposedPOVMEffect(static_effect, _ExpErrorgenOp(errorgen))

    elif to_type == "static clifford":
        if isinstance(effect, ComputationalBasisPOVMEffect):
            return effect  # no conversion necessary

        purevec = effect.to_dense().flatten()  # assume a pure state (otherwise would need to change Model dim)
        return ComputationalBasisPOVMEffect.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


def optimize_effect(vec_to_optimize, target_vec):
    """
    Optimize the parameters of vec_to_optimize.

    The optimization is performed so that the the resulting POVM effect is as
    close as possible to target_vec.

    Parameters
    ----------
    vec_to_optimize : POVMEffect
        The effect vector to optimize. This object gets altered.

    target_vec : POVMEffect
        The effect vector used as the target.

    Returns
    -------
    None
    """

    if not isinstance(vec_to_optimize, ConjugatedStatePOVMEffect):
        return  # we don't know how to deal with anything but a conjuated state effect...

    from ..states import optimize_state as _optimize_state
    _optimize_state(vec_to_optimize.state, target_vec)
    vec_to_optimize.from_vector(vec_to_optimize.state.to_vector())  # make sure effect is updated
