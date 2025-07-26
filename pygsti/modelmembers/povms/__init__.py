"""
Sub-package holding model POVM and POVM effect objects.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import functools as _functools
import itertools as _itertools
import warnings
import numpy as _np
import scipy.linalg as _spl
import scipy.optimize as _spo

from .complementeffect import ComplementPOVMEffect
from .composedeffect import ComposedPOVMEffect
from .composedpovm import ComposedPOVM
from .computationaleffect import ComputationalBasisPOVMEffect
from .computationalpovm import ComputationalBasisPOVM
from .conjugatedeffect import ConjugatedStatePOVMEffect
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
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot
from pygsti.tools import sum_of_negative_choi_eigenvalues_gate
from pygsti.baseobjs import Basis

# Avoid circular import
import pygsti.modelmembers as _mm


def create_from_pure_vectors(pure_vectors, povm_type, basis='pp', evotype='default', state_space=None,
                             on_construction_error='warn'):
    """
    Creates a Positive Operator-Valued Measure (POVM) from a list or dictionary of (key, pure-vector) pairs.

    Parameters
    ----------
    pure_vectors : list or dict
        A list of (key, pure-vector) pairs or a dictionary where keys are labels and values are pure state vectors.
        
    povm_type : str or tuple
        The type of POVM to create. This can be a single string or a tuple of strings indicating the preferred types.
        Supported types include 'computational', 'static pure', 'full pure', 'static', 'full', 'full TP', and any valid
        Lindblad parameterization type.

    basis : str, optional
        The basis in which the pure vectors are expressed. Default is 'pp'.

    evotype : str, optional
        The evolution type. Default is 'default'.

    state_space : StateSpace, optional
        The state space in which the POVM operates. Default is None.

    on_construction_error : str, optional
        Specifies the behavior when an error occurs during POVM construction. Options are 'raise' to raise the error,
        'warn' to print a warning message, or any other value to silently ignore the error. Default is 'warn'.

    Returns
    -------
    POVM
        The constructed POVM object.
    """
    povm_type_preferences = (povm_type,) if isinstance(povm_type, str) else povm_type
    if not isinstance(pure_vectors, dict):  # then assume it's a list of (key, value) pairs
        pure_vectors = dict(pure_vectors)
    if state_space is None:
        state_space = _statespace.default_space_for_udim(len(next(iter(pure_vectors.values()))))

    for typ in povm_type_preferences:
        try:
            if typ == 'computational':
                povm = ComputationalBasisPOVM.from_pure_vectors(pure_vectors, evotype, state_space)
            #elif typ in ('static stabilizer', 'static clifford'):
            #    povm = ComputationalBasisPOVM(...evotype='stabilizer') ??
            elif typ in ('static pure', 'full pure', 'static', 'full'):
                effects = [(lbl, create_effect_from_pure_vector(vec, typ, basis, evotype, state_space))
                           for lbl, vec in pure_vectors.items()]
                povm = UnconstrainedPOVM(effects, evotype, state_space)
            elif typ == 'full TP':
                effects = [(lbl, create_effect_from_pure_vector(vec, "full", basis, evotype, state_space))
                           for lbl, vec in pure_vectors.items()]
                povm = TPPOVM(effects, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(typ)

                base_povm = create_from_pure_vectors(pure_vectors, ('computational', 'static pure'),
                                                     basis, evotype, state_space)

                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, lndtype, proj_basis, basis,
                                                                  truncate=True, evotype=evotype,
                                                                  state_space=state_space)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                povm = ComposedPOVM(EffectiveExpErrorgen(errorgen), base_povm, mx_basis=basis)
            else:
                raise ValueError("Unknown POVM type '%s'!" % str(typ))

            return povm  # if we get to here, then we've successfully created a state to return
        except (ValueError, AssertionError) as err:
            if on_construction_error == 'raise':
                raise err
            elif on_construction_error == 'warn':
                print('Failed to construct povm with type "{}" with error: {}'.format(typ, str(err)))
            pass  # move on to next type

    raise ValueError("Could not create a POVM of type(s) %s from the given pure vectors!" % (str(povm_type)))


def create_from_dmvecs(superket_vectors, povm_type, basis='pp', evotype='default', state_space=None,
                       on_construction_error='warn'):
    """ 
    Creates a Positive Operator-Valued Measure (POVM) from a list or dictionary of (key, superket) pairs.

    Parameters
    ----------
    superket_vectors : list or dict
        A list of (key, pure-vector) pairs or a dictionary where keys are labels and values are superket vectors.
        i.e. vectorized density matrices.
        
    povm_type : str or tuple
        The type of POVM to create. This can be a single string or a tuple of strings indicating the preferred types.
        Supported types include 'full', 'static', 'full TP', 'computational', 'static pure', 'full pure', and any valid
        Lindblad parameterization type.

    basis : str or `Basis`, optional
        The basis in which the density matrix vectors are expressed. Default is 'pp'.

    evotype : str, optional
        The evolution type. Default is 'default'.

    state_space : StateSpace, optional
        The state space in which the POVM operates. Default is None.

    on_construction_error : str, optional
        Specifies the behavior when an error occurs during POVM construction. Options are 'raise' to raise the error,
        'warn' to print a warning message, or any other value to silently ignore the error. Default is 'warn'.

    Returns
    -------
    POVM
        The constructed POVM object. 
    """
    povm_type_preferences = (povm_type,) if isinstance(povm_type, str) else povm_type
    if not isinstance(superket_vectors, dict):  # then assume it's a list of (key, value) pairs
        superket_vectors = dict(superket_vectors)

    for typ in povm_type_preferences:
        try:
            if typ in ("full", "static"):
                effects = [(lbl, create_effect_from_dmvec(dmvec, typ, basis, evotype, state_space))
                           for lbl, dmvec in superket_vectors.items()]
                povm = UnconstrainedPOVM(effects, evotype, state_space)
            elif typ == 'full TP':
                effects = [(lbl, create_effect_from_dmvec(dmvec, 'full', basis, evotype, state_space))
                           for lbl, dmvec in superket_vectors.items()]
                povm = TPPOVM(effects, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(typ)

                base_povm = create_from_dmvecs(superket_vectors, ('computational', 'static'),
                                               basis, evotype, state_space)

                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, lndtype, proj_basis, basis,
                                                                  truncate=True, evotype=evotype,
                                                                  state_space=state_space)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                povm = ComposedPOVM(EffectiveExpErrorgen(errorgen), base_povm, mx_basis=basis)
            elif typ in ('computational', 'static pure', 'full pure'):
                # RESHAPE NOTE: .ravel() added to line below (to convert pure *col* vec -> 1D) to fix unit tests
                pure_vectors = {k: _ot.dmvec_to_state(_bt.change_basis(superket, basis, 'std')).ravel()
                                for k, superket in superket_vectors.items()}
                povm = create_from_pure_vectors(pure_vectors, typ, basis, evotype, state_space)
            else:
                raise ValueError("Unknown POVM type '%s'!" % str(typ))

            return povm  # if we get to here, then we've successfully created a state to return
        except (ValueError, AssertionError) as err:
            if on_construction_error == 'raise':
                raise err
            elif on_construction_error == 'warn':
                print('Failed to construct povm with type "{}" with error: {}'.format(typ, str(err)))
            pass  # move on to next type

    raise ValueError("Could not create a POVM of type(s) %s from the given density matrix vectors!" % (str(povm_type)))


def create_effect_from_pure_vector(pure_vector, effect_type, basis='pp', evotype='default', state_space=None,
                                   on_construction_error='warn'):
    """
    Creates a POVM effect from a pure state vector.

    Parameters
    ----------
    pure_vector : array-like
        The pure state vector from which to create the POVM effect.
        
    effect_type : str or tuple
        The type of effect to create. This can be a single string or a tuple of strings indicating the preferred types.
        Supported types include 'computational', 'static pure', 'full pure', 'static', 'full', 'static clifford', and
        any valid Lindblad parameterization type.

    basis : str or `Basis` optional
        The basis in which the pure vector is expressed. Default is 'pp'.

    evotype : str, optional
        The evolution type. Default is 'default'.

    state_space : StateSpace, optional
        The state space in which the effect operates. Default is None.

    on_construction_error : str, optional
        Specifies the behavior when an error occurs during effect construction. Options are 'raise' to raise the error,
        'warn' to print a warning message, or any other value to silently ignore the error. Default is 'warn'.

    Returns
    -------
    POVMEffect
        The constructed POVM effect object.
    """
    effect_type_preferences = (effect_type,) if isinstance(effect_type, str) else effect_type
    if state_space is None:
        state_space = _statespace.default_space_for_udim(len(pure_vector))

    for typ in effect_type_preferences:
        try:
            if typ == 'computational':
                ef = ComputationalBasisPOVMEffect.from_pure_vector(pure_vector, basis, evotype, state_space)
            #elif typ == ('static stabilizer', 'static clifford'):
            #    ef = StaticStabilizerEffect(...)  # TODO
            elif typ == 'static pure':
                ef = StaticPOVMPureEffect(pure_vector, basis, evotype, state_space)
            elif typ == 'full pure':
                ef = FullPOVMPureEffect(pure_vector, basis, evotype, state_space)
            elif typ in ('static', 'full'):
                superket = _bt.change_basis(_ot.state_to_dmvec(pure_vector), 'std', basis)
                ef = create_effect_from_dmvec(superket, typ, basis, evotype, state_space)
            elif typ == 'static clifford':
                ef = ComputationalBasisPOVMEffect.from_pure_vector(pure_vector.ravel())
            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(typ)

                static_effect = create_effect_from_pure_vector(
                    pure_vector, ('computational', 'static pure'), basis, evotype, state_space)

                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, lndtype, proj_basis, basis,
                                                                  truncate=True, evotype=evotype,
                                                                  state_space=state_space)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                ef = ComposedPOVMEffect(static_effect, EffectiveExpErrorgen(errorgen))
            else:
                raise ValueError("Unknown effect type '%s'!" % str(typ))

            return ef  # if we get to here, then we've successfully created a state to return
        except (ValueError, AssertionError) as err:
            if on_construction_error == 'raise':
                raise err
            elif on_construction_error == 'warn':
                print('Failed to construct effect with type "{}" with error: {}'.format(typ, str(err)))
            pass  # move on to next type

    raise ValueError("Could not create an effect of type(s) %s from the given pure vector!" % (str(effect_type)))


def create_effect_from_dmvec(superket_vector, effect_type, basis='pp', evotype='default', state_space=None,
                             on_construction_error='warn'):
    """
    Creates a POVM effect from a density matrix vector (superket).

    Parameters
    ----------
    superket_vector : array-like
        The density matrix vector (superket) from which to create the POVM effect.
        
    effect_type : str or tuple
        The type of effect to create. This can be a single string or a tuple of strings indicating the preferred types.
        Supported types include 'static', 'full', and any valid Lindblad parameterization type. For other types
        we first try to convert to a pure state vector and then utilize `create_effect_from_pure_vector`

    basis : str or `Basis` optional
        The basis in which the superket vector is expressed. Default is 'pp'.

    evotype : str, optional
        The evolution type. Default is 'default'.

    state_space : StateSpace, optional
        The state space in which the effect operates. Default is None.

    on_construction_error : str, optional
        Specifies the behavior when an error occurs during effect construction. Options are 'raise' to raise the error,
        'warn' to print a warning message, or any other value to silently ignore the error. Default is 'warn'.

    Returns
    -------
    POVMEffect
        The constructed POVM effect object.
    """

    effect_type_preferences = (effect_type,) if isinstance(effect_type, str) else effect_type
    if state_space is None:
        state_space = _statespace.default_space_for_dim(len(superket_vector))

    for typ in effect_type_preferences:
        try:
            if typ == "static":
                ef = StaticPOVMEffect(superket_vector, basis, evotype, state_space)
            elif typ == "full":
                ef = FullPOVMEffect(superket_vector, basis, evotype, state_space)
            elif _ot.is_valid_lindblad_paramtype(typ):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(typ)

                try:
                    dmvec = _bt.change_basis(superket_vector, basis, 'std')
                    purevec = _ot.dmvec_to_state(dmvec)  # raises error if dmvec does not correspond to a pure state
                    static_effect = StaticPOVMPureEffect(purevec, basis, evotype, state_space)
                except ValueError:
                    static_effect = StaticPOVMEffect(superket_vector, basis, evotype, state_space)
                proj_basis = 'PP' if state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(state_space.dim, lndtype, proj_basis,
                                                                  basis, truncate=True, evotype=evotype)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                ef = ComposedPOVMEffect(static_effect, EffectiveExpErrorgen(errorgen))
            else:
                # Anything else we try to convert to a pure vector and convert the pure state vector
                dmvec = _bt.change_basis(superket_vector, basis, 'std')
                purevec = _ot.dmvec_to_state(dmvec)  # raises error if dmvec does not correspond to a pure state

                ef = create_effect_from_pure_vector(purevec, typ, basis, evotype, state_space)
            return ef
        except (ValueError, AssertionError) as err:
            if on_construction_error == 'raise':
                raise err
            elif on_construction_error == 'warn':
                print('Failed to construct effect with type "{}" with error: {}'.format(typ, str(err)))
            pass  # move on to next type

    raise ValueError("Could not create an effect of type(s) %s from the given superket vector!" % (str(effect_type)))


def povm_type_from_op_type(op_type):
    """
    Decode an op type into an appropriate povm type.

    Parameters:
    -----------
    op_type: str or list of str
        Operation parameterization type (or list of preferences)

    Returns
    -------
    povm_type_preferences: tuple of str
        POVM parameterization types
    """
    op_type_preferences = _mm.operations.verbose_type_from_op_type(op_type)

    # computational and TP are directly constructed as POVMS
    # All others pass through to the effects
    povm_conversion = {
        'auto': 'computational',
        'static standard': 'computational',
        'static clifford': 'computational',
        'static unitary': 'static pure',
        'full unitary': 'full pure',
        'static': 'static',
        'full': 'full',
        'full TP': 'full TP',
        'full CPTP': 'computational',  # TEMPORARY HACK until we create a legit option here
        'linear': 'full',
    }

    povm_type_preferences = []
    for typ in op_type_preferences:
        povm_type = None
        if _ot.is_valid_lindblad_paramtype(typ):
            # Lindblad types are passed through
            povm_type = typ
        else:
            povm_type = povm_conversion.get(typ, None)

        if povm_type is None:
            continue

        if povm_type not in povm_type_preferences:
            povm_type_preferences.append(povm_type)

    if len(povm_type_preferences) == 0:
        raise ValueError(
            'Could not convert any op types from {}.\n'.format(op_type_preferences)
            + '\tKnown op_types: Lindblad types or {}\n'.format(sorted(list(povm_conversion.keys())))
            + '\tValid povm_types: Lindblad types or {}'.format(sorted(list(set(povm_conversion.values()))))
        )

    return povm_type_preferences


def convert(povm, to_type, basis, ideal_povm=None, flatten_structure=False, cp_penalty=1e-7):
    """
    TODO: update docstring
    Convert a POVM to a new type of parameterization.

    This potentially creates a new object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    povm : POVM
        POVM to convert

    to_type : {"full","full TP","static","static pure","H+S terms",
        "H+S clifford terms","clifford"}
        The type of parameterizaton to convert to.  See
        :meth:`Model.set_all_parameterizations` for more details.
        TODO docstring: update the options here.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    ideal_povm : POVM, optional
        The ideal version of `povm`, potentially used when
        converting to an error-generator type.

    flatten_structure : bool, optional
        When `False`, the sub-members of composed and embedded operations
        are separately converted, leaving the original POVM's structure
        unchanged.  When `True`, composed and embedded operations are "flattened"
        into a single POVM of the requested `to_type`.
    
    cp_penalty : float, optional (default 1e-7)
            Converting SPAM operations to an error generator representation may 
            introduce trivial gauge degrees of freedom. These gauge degrees of freedom 
            are called trivial because they quite literally do not change the dense representation 
            (i.e. Hilbert-Schmidt vectors) at all. Despite being trivial, error generators along 
            this trivial gauge orbit may be non-CP, so this cptp penalty is used to favor channels 
            within this gauge orbit which are CPTP.

    Returns
    -------
    POVM
        The converted POVM vector, usually a distinct
        object from the object passed as input.
    """

    to_types = to_type if isinstance(to_type, (tuple, list)) else (to_type,)  # HACK to support multiple to_type values
    error_msgs = {}

    destination_types = {'full TP': TPPOVM,
                         'static clifford': ComputationalBasisPOVM}
    NoneType = type(None)

    for to_type in to_types:
        try:
            if isinstance(povm, destination_types.get(to_type, NoneType)):
                return povm

            idl = dict(ideal_povm.items()) if ideal_povm is not None else {}  # ideal effects

            if to_type in ("full", "static", "static pure"):
                converted_effects = [(lbl, convert_effect(vec, to_type, basis, idl.get(lbl, None), flatten_structure))
                                     for lbl, vec in povm.items()]
                return UnconstrainedPOVM(converted_effects, povm.evotype, povm.state_space)

            elif to_type == "full TP":
                converted_effects = [(lbl, convert_effect(vec, "full", basis, idl.get(lbl, None), flatten_structure))
                                     for lbl, vec in povm.items()]
                return TPPOVM(converted_effects, povm.evotype, povm.state_space)

            elif _ot.is_valid_lindblad_paramtype(to_type):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(to_type)

                
                #Construct a static "base" POVM
                if isinstance(povm, ComputationalBasisPOVM):  # special easy case
                    base_povm = ComputationalBasisPOVM(povm.state_space.num_qubits, povm.evotype)  # just copy it?
                else:
                    try:
                        if povm.evotype.minimal_space == 'Hilbert':
                            base_items = [(lbl, convert_effect(vec, 'static pure', basis,
                                                               idl.get(lbl, None), flatten_structure))
                                          for lbl, vec in povm.items()]
                        else:
                            raise RuntimeError('Evotype must be compatible with Hilbert ops to use pure effects')
                    except RuntimeError:  # try static mixed states next:
                        #if idl.get(lbl,None) is not None:
                        
                        base_items = []
                        for lbl, vec in povm.items():
                            ideal_effect = idl.get(lbl,None)
                            if ideal_effect is not None:
                                base_items.append((lbl, convert_effect(ideal_effect, 'static', basis, ideal_effect, flatten_structure)))
                            else:
                                base_items.append((lbl, convert_effect(vec, 'static', basis, idl.get(lbl, None), flatten_structure)))
                    base_povm = UnconstrainedPOVM(base_items, povm.evotype, povm.state_space)

                proj_basis = 'PP' if povm.state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(povm.state_space.dim, lndtype, proj_basis,
                                                                  basis, truncate=True, evotype=povm.evotype)
                    
                #Collect all ideal effects
                base_dense_effects = []
                for item in base_items:
                    dense_effect = item[1].to_dense()
                    base_dense_effects.append(dense_effect.reshape((1,len(dense_effect))))

                dense_ideal_povm = _np.concatenate(base_dense_effects, axis=0)

                #Collect all noisy effects
                dense_effects = []
                for effect in povm.values():
                    dense_effect = effect.to_dense()
                    dense_effects.append(dense_effect.reshape((1,len(dense_effect))))

                dense_povm = _np.concatenate(dense_effects, axis=0)
                
                #It is often the case that there are more error generators than physical degrees of freedom in the POVM
                #We define a function which finds linear comb. of errgens that span these degrees of freedom.
                #This has been called "the trivial gauge", and this function is meant to avoid it
                def calc_physical_subspace(dense_ideal_povm, epsilon = 1e-4):

                    degrees_of_freedom = (dense_ideal_povm.shape[0] - 1) * dense_ideal_povm.shape[1]
                    errgen = _LindbladErrorgen.from_error_generator(povm.state_space.dim, parameterization=to_type)

                    if degrees_of_freedom > errgen.num_params:
                        warnings.warn("POVM has more degrees of freedom than the available number of parameters, representation in this parameterization is not guaranteed")
                    exp_errgen = _ExpErrorgenOp(errgen)
                    
                    num_errgens = errgen.num_params
                    #TODO: Maybe we can use the num of params instead of number of matrix entries, as some of them are linearly dependent.
                    #i.e E0 completely determines E1 if those are the only two povm elements (E0 + E1 = Identity)
                    num_entries = dense_ideal_povm.size

                    #Compute the jacobian with respect to the error generators. This will allow us to see which
                    #error generators change the POVM entries
                    J = _np.zeros((num_entries,num_errgens))
                    new_vec = _np.zeros(num_errgens)
                    for i in range(num_errgens):
                        
                        new_vec[i] = epsilon
                        exp_errgen.from_vector(new_vec)
                        new_vec[i] = 0
                        vectorized_povm = _np.zeros(num_entries)
                        perturbed_povm = (dense_ideal_povm @ exp_errgen.to_dense() - dense_ideal_povm)/epsilon 

                        vectorized_povm = perturbed_povm.flatten(order='F')
                        
                        J[:,i] = vectorized_povm

                    _,S,Vt = _np.linalg.svd(J, full_matrices=False)

                    #Only return nontrivial singular vectors
                    Vt = Vt[S > 1e-13, :].reshape((-1, Vt.shape[1]))
                    return Vt
                    
                
                phys_directions = calc_physical_subspace(dense_ideal_povm)

                #We use optimization to find the best error generator representation
                #we only vary physical directions, not independent error generators
                def _objfn(v):

                    #For some reason adding the sum_of_negative_choi_eigenvalues_gate term
                    #resulted in minimize() sometimes choosing NaN values for v. There are
                    #two stack exchange issues showing this problem with no solution.
                    if _np.isnan(v).any():
                        v = _np.zeros(len(v))

                    L_vec = _np.zeros(len(phys_directions[0]))
                    for coeff, phys_direction in zip(v,phys_directions):
                        L_vec += coeff * phys_direction
                    errorgen.from_vector(L_vec)
                    proc_matrix = _spl.expm(errorgen.to_dense())
                    
                    return _np.linalg.norm(dense_povm - dense_ideal_povm @ proc_matrix) + cp_penalty * sum_of_negative_choi_eigenvalues_gate(proc_matrix, basis)
                
                soln = _spo.minimize(_objfn, _np.zeros(len(phys_directions), 'd'), method="Nelder-Mead", options={},
                                        tol=1e-13) 
                if not soln.success and soln.fun > 1e-6:  # not "or" because success is often not set correctly
                    raise ValueError("Failed to find an errorgen such that <ideal|exp(errorgen) = <effect|")
                errgen_vec = _np.linalg.lstsq(phys_directions, soln.x)[0]
                errorgen.from_vector(errgen_vec)
                
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                return ComposedPOVM(EffectiveExpErrorgen(errorgen), base_povm, mx_basis=basis)

            elif to_type == "static clifford":
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
        except Exception as e:
            error_msgs[to_type] = str(e)  # try next to_type

    raise ValueError("Could not convert POVM to to type(s): %s\n%s" % (str(to_types), str(error_msgs)))


def convert_effect(effect, to_type, basis, ideal_effect=None, flatten_structure=False):
    """
    TODO: update docstring
    Convert POVM effect vector to a new type of parameterization.

    This potentially creates a new POVMEffect object.
    Raises ValueError for invalid conversions.

    Parameters
    ----------
    effect : POVMEffect
        POVM effect vector to convert

    to_type : {"full","TP","static","static pure","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :meth:`Model.set_all_parameterizations` for more details.

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
    to_types = to_type if isinstance(to_type, (tuple, list)) else (to_type,)  # HACK to support multiple to_type values
    destination_types = {'full': FullPOVMEffect,
                         'static': StaticPOVMEffect,
                         #'static pure': StaticPOVMPureEffect,
                         'static clifford': ComputationalBasisPOVMEffect}
    NoneType = type(None)

    for to_type in to_types:
        try:
            if isinstance(effect, destination_types.get(to_type, NoneType)):
                return effect

            if not flatten_structure and isinstance(effect, ComposedPOVMEffect):
                return ComposedPOVMEffect(effect.effect_vec.copy(),  # don't convert (usually static) effect vec
                                          _mm.operations.convert(effect.error_map, to_type, basis, "identity",
                                                                 flatten_structure))

            elif _ot.is_valid_lindblad_paramtype(to_type) and (ideal_effect is not None or effect.num_params == 0):
                from ..operations import LindbladErrorgen as _LindbladErrorgen, ExpErrorgenOp as _ExpErrorgenOp
                from ..operations import IdentityPlusErrorgenOp as _IdentityPlusErrorgenOp
                from ..operations import LindbladParameterization as _LindbladParameterization
                lndtype = _LindbladParameterization.cast(to_type)

                ef = ideal_effect if (ideal_effect is not None) else effect
                if ef is not effect and not _np.allclose(ef.to_dense(), effect.to_dense()):
                    raise NotImplementedError("Must supply ideal or a static effect to convert to a Lindblad type!")

                proj_basis = 'PP' if effect.state_space.is_entirely_qubits else basis
                errorgen = _LindbladErrorgen.from_error_generator(effect.state_space.dim, lndtype, proj_basis,
                                                                  basis, truncate=True, evotype=effect.evotype)
                EffectiveExpErrorgen = _IdentityPlusErrorgenOp if lndtype.meta == '1+' else _ExpErrorgenOp
                return ComposedPOVMEffect(ef, EffectiveExpErrorgen(errorgen))

            else:
                min_space = effect.evotype.minimal_space
                vec = effect.to_dense(min_space)
                if min_space == 'Hilbert':
                    return create_effect_from_pure_vector(vec, to_type, basis, effect.evotype, effect.state_space,
                                                          on_construction_error='raise')
                else:
                    return create_effect_from_dmvec(vec, to_type, basis, effect.evotype, effect.state_space)
        except ValueError:
            pass

    raise ValueError("Could not convert effect to type(s): %s" % str(to_types))


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
