#***************************************************************************************************
# Copyright 2015, 2019, 2025, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations

import collections as _collections
import numpy as _np

from pygsti.modelmembers import modelmember as _mm
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import states as _state
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs import BasisLike as _BasisLike
from pygsti.tools import matrixtools as _mt
from pygsti.tools import slicetools as _slct
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.statespace import StateSpace as _StateSpace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pygsti.modelmembers.povms.effect import POVMEffect
    from pygsti.modelmembers.states.state import State
    from pygsti.models.gaugegroup import GaugeGroupElement
    # Type vocabulary for the effect-then-gate construction API:
    EffectSpec = _np.ndarray | POVMEffect                     # a POVM effect (superket, Hermitian matrix, or object)
    GateSpec   = _np.ndarray | _op.LinearOperator | None      # a post-measurement gate (None means the identity)
    MemberSpec = EffectSpec | tuple[EffectSpec, GateSpec]     # one outcome's (effect[, gate]) spec
    MemberOps = (dict[str, _op.LinearOperator | _np.ndarray]  # member ops as a dict or ordered (label, op) pairs
                 | list[tuple[str, _op.LinearOperator | _np.ndarray]])


class Instrument(_mm.ModelMember, _collections.OrderedDict):
    """
    A generalized quantum instrument.

    Meant to correspond to a quantum instrument in theory, this class
    generalizes that notion slightly to include a collection of gates that may
    or may not have all of the properties associated by a mathematical quantum
    instrument.

    Parameters
    ----------
    member_ops : dict of LinearOperator objects
        A dict (or list of key,value pairs) of the gates.

    evotype : Evotype or str, optional
        The evolution type.  If `None`, the evotype is inferred
        from the first instrument member.  If `len(member_ops) == 0` in this case,
        an error is raised.

    state_space : StateSpace, optional
        The state space for this POVM.  If `None`, the space is inferred
        from the first instrument member.  If `len(member_ops) == 0` in this case,
        an error is raised.

    items : list or dict, optional
        Initial values.  This should only be used internally in de-serialization.
    """

    def __init__(self, member_ops: MemberOps | None, evotype: _Evotype | str | None = None,
                 state_space: _StateSpace | None = None, called_from_reduce: bool = False,
                 items: list | None = None):
        if items is None:
            items = []
        self._readonly = False  # until init is done
        if len(items) > 0:
            assert(member_ops is None), "`items` was given when op_matrices != None"

        if member_ops is not None:
            if isinstance(member_ops, dict):
                member_list = [(k, v) for k, v in member_ops.items()]  # gives definite ordering
            elif isinstance(member_ops, list):
                member_list = member_ops  # assume it's is already an ordered (key,value) list
            else:
                raise ValueError("Invalid `member_ops` arg of type %s" % type(member_ops))

            #Special case when we're given matrices: infer a default state space and evotype:
            if len(member_list) > 0 and not isinstance(member_list[0][1], _op.LinearOperator):
                if state_space is None:
                    state_space = _statespace.default_space_for_dim(member_list[0][1].shape[0])
                if evotype is None:
                    evotype = _Evotype.cast('default', state_space=state_space)
                member_list = [(k, v if isinstance(v, _op.LinearOperator) else
                                _op.FullArbitraryOp(v, None, evotype, state_space)) for k, v in member_list]

            assert(len(member_list) > 0 or state_space is not None), \
                "Must specify `state_space` when there are no instrument members!"
            assert(len(member_list) > 0 or evotype is not None), \
                "Must specify `evotype` when there are no instrument members!"
            state_space = member_list[0][1].state_space if (state_space is None) \
                else _statespace.StateSpace.cast(state_space)
            evotype = _Evotype.cast(evotype, state_space=state_space) if (evotype is not None)\
                else member_list[0][1].evotype
            items = []
            for k, member in member_list:
                assert(evotype == member.evotype), \
                    "All instrument members must have the same evolution type"
                assert(state_space.is_compatible_with(member.state_space)), \
                    "All instrument members must have compatible state spaces!"
                items.append((k, member))
        else:
            if len(items) > 0:  # HACK so that OrderedDict.copy() works, which creates a new object with only items...
                if state_space is None: state_space = items[0][1].state_space
                if evotype is None: evotype = items[0][1].evotype

            assert(state_space is not None), "`state_space` cannot be `None` when there are no members!"
            assert(evotype is not None), "`evotype` cannot be `None` when there are no members!"

        _collections.OrderedDict.__init__(self, items)
        _mm.ModelMember.__init__(self, state_space, evotype)
        if not called_from_reduce:  # if called from reduce, gpindices are already initialized
            self.init_gpindices()  # initialize our gpindices based on sub-members
        self._readonly = True

    @staticmethod
    def from_effects(members: dict[str, MemberSpec], basis: _BasisLike, gate_parameterization: str = 'CPTPLND',
                     povm_errormap: _op.LinearOperator | str = 'CPTPLND', atol: float = 1e-6) -> Instrument:
        r"""
        Construct a parameterized instrument from measurement effects (and optional
        post-measurement gates).

        Each member is built in the canonical *measure-then-gate* form

            I_k(rho) = G_k( E_k^{1/2} rho E_k^{1/2} ),

        a soft measurement of the POVM effect `E_k` followed by the
        post-measurement (CP)TP gate `G_k`.  The effects `{E_k}` are gathered
        into a single shared :class:`ComposedPOVM` whose CP-constrained error map
        keeps every `E_k` positive and makes them sum to the identity. Each
        gate `G_k` is parameterized independently.

        The constraints decouple cleanly: POVM completeness gives joint trace
        preservation, and a CP-constrained `gate_parameterization` makes each
        member completely positive.  An `n`-outcome instrument needs only `n`
        effects and `n` gates.

        Parameters
        ----------
        members : dict
            Maps each outcome label to either an `(effect, gate)` pair or a bare
            `effect` (in which case the gate defaults to the identity).
            
            Each `effect` may be a length-`d**2` superket in `basis`, or a `d x d`
            Hermitian matrix, or a :class:`POVMEffect`. The effects must comprise
            a physically meaningful POVM.

            Each `gate` may be a `d**2 x d**2` superoperator, a `d x d` unitary,
            a :class:`LinearOperator`, or None. The last of these is equivalent
            to setting `gate` to the identity matrix. Each gate must be TP.

        basis : _BasisLike
            The basis in which dense arrays are represented.

        gate_parameterization : str, optional
            A TP parameterization for the gates `{G_k}`. CP-constrained Lindblad
            types (`'CPTPLND'`, `'H+S'`) make each member completely positive;
            non-CP-constrained Lindblad types (`'GLND'`, `'H+s'`, `'full TP'`)
            keep the instrument TP but allow non-CP members. `'static'` freezes
            the gates. `'full'` is rejected (it is not TP).

        povm_errormap : LinearOperator or str, optional
            A CP-by-construction :class:`LinearOperator`, or a string spec for one,
            used as the shared error map of the effects' :class:`ComposedPOVM`.

        atol : float, optional
            Absolute tolerance for the per-gate TP check and the completeness check.

        Returns
        -------
        Instrument
        """
        from pygsti.modelmembers.instruments._construction import (
            _normalize_effects_and_gates, _parameterized_instrument
        )
        basis, superkets, superops = _normalize_effects_and_gates(members, basis, atol)
        member_ops = _parameterized_instrument(
            basis, superkets, superops, gate_parameterization, povm_errormap
        )
        return Instrument(member_ops)

    @staticmethod
    def from_cptr_superops(
            op_arrays: dict[str, _np.ndarray], basis: _BasisLike,
            gate_parameterization: str = 'CPTPLND',
            povm_errormap: _op.LinearOperator | str = 'CPTPLND',
            error_tol: float = 1e-6, trunc_tol: float = 1e-7
        ) -> Instrument:
        r"""
        Construct a parameterized instrument from arbitrary dense CPTR member superops.

        Every completely-positive instrument member factors as a measurement effect
        followed by a post-measurement CPTP gate,

            I_k(rho) = G_k( E_k^{1/2} rho E_k^{1/2} ),   E_k = I_k^dagger(I),

        and this constructor recovers that decomposition directly from each dense
        superop. The effect is the Heisenberg-dual applied to the identity; the gate
        is the CPTP completion `G_k = Q_k + P_k` with `Q_k = I_k . pinv(rootconj(E_k))`
        and `P_k` a conjugation by the projector onto `ker(E_k)`.  The effects are
        gathered into a single shared :class:`ComposedPOVM` and each gate is
        parameterized independently, so an `n`-outcome instrument needs only `n`
        effects and `n` gates regardless of any member's Kraus rank.

        Parameters
        ----------
        op_arrays : dict[str, np.ndarray]
            Maps each outcome label to a dense `d**2 x d**2` CPTR superoperator in
            `basis`.  Each must be completely positive, and together they must sum
            to a trace-preserving channel (`sum_k E_k == I`).

        basis : BasisLike
            The basis in which the superoperators are represented.

        gate_parameterization : str, optional
            See :meth:`from_effects`.

        povm_errormap : LinearOperator or str, optional
            See :meth:`from_effects`.

        error_tol : float, optional
            Tolerance for the per-member complete-positivity check (negative Choi
            eigenvalues below `-error_tol` raise an error).

        trunc_tol : float, optional
            Cutoff below which an effect eigenvalue is treated as zero when forming
            the `ker(E_k)` projector that completes each gate.

        Returns
        -------
        Instrument
        """
        from pygsti.modelmembers.instruments._construction import (
            _decompose_cptr, _check_effects_complete,
            _parameterized_instrument
        )
        dim = next(iter(op_arrays.values())).shape[0]
        basis = _Basis.cast(basis, dim)
        superkets = dict()
        superops  = dict()
        for lbl, I_k in op_arrays.items():
            E_k, G_k = _decompose_cptr(I_k, basis, error_tol, trunc_tol)
            superkets[lbl] = E_k 
            superops[lbl]  = G_k
        _check_effects_complete(superkets, basis, error_tol)
        member_ops = _parameterized_instrument(
            basis, superkets, superops, gate_parameterization, povm_errormap
        )
        return Instrument(member_ops)

    def submembers(self) -> list:
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return list(self.values())

    def to_memoized_dict(self, mmg_memo: dict) -> dict:
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['member_labels'] = list(self.keys())  # labels of the submember effects

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict: dict, serial_memo: dict) -> Instrument:
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])
        members = [(lbl, serial_memo[subm_serial_id])
                   for lbl, subm_serial_id in zip(mm_dict['member_labels'], mm_dict['submembers'])]
        return cls(members, mm_dict['evotype'], state_space)

    def _is_similar(self, other: Instrument, rtol: float, atol: float) -> bool:
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return list(self.keys()) == list(other.keys())

    def __setitem__(self, key, value) -> None:
        if self._readonly: raise ValueError("Cannot alter Instrument elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)

    def __reduce__(self) -> tuple:
        """ Needed for OrderedDict-derived classes (to set dict items) """
        #need to *not* pickle parent, as __reduce__ bypasses ModelMember.__getstate__
        dict_to_pickle = self.__dict__.copy()
        dict_to_pickle['_parent'] = None

        #Note: must *copy* elements for pickling/copying
        return (Instrument, (None, self.evotype, self.state_space, True,
                             [(key, gate.copy()) for key, gate in self.items()]),
                dict_to_pickle)

    def __pygsti_reduce__(self) -> tuple:
        return self.__reduce__()

    def simplify_operations(self, prefix: str | _Label = "") -> _collections.OrderedDict:
        """
        Creates a dictionary of simplified instrument operations.

        Returns a dictionary of operations that belong to the Instrument's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this instruments's gpindices.  These are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this instrument, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of Gates
        """
        #Create a "simplified" (Model-referencing) set of element gates
        simplified = _collections.OrderedDict()
        if isinstance(prefix, _Label):  # Deal with case when prefix isn't just a string
            for k, g in self.items():
                simplified[_Label(prefix.name + "_" + k, prefix.sslbls)] = g
        else:
            if prefix: prefix += "_"
            for k, g in self.items():
                simplified[prefix + k] = g
        return simplified

    @property
    def parameter_labels(self) -> _np.ndarray:
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        plabels_per_local_index = _collections.defaultdict(list)
        for operation, factorgate_local_inds in zip(self.submembers(), self._submember_rpindices):
            for i, plbl in zip(_slct.to_array(factorgate_local_inds), operation.parameter_labels):
                plabels_per_local_index[i].append(plbl)

        vl = _np.empty(self.num_params, dtype=object)
        for i in range(self.num_params):
            vl[i] = ', '.join(plabels_per_local_index[i])
        return vl

    @property
    def num_elements(self) -> int:
        """
        Return the number of total gate elements in this instrument.

        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the matrix *elements*.

        Returns
        -------
        int
        """
        return sum([g.size for g in self.values()])

    @property
    def num_params(self) -> int:
        """
        Get the number of independent parameters which specify this Instrument.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self) -> _np.ndarray:
        """
        Extract a vector of the underlying gate parameters from this Instrument.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        assert(self.gpindices is not None), "Must set an Instrument's .gpindices before calling to_vector"
        v = _np.empty(self.num_params, 'd')
        for operation, factor_local_inds in zip(self.values(), self._submember_rpindices):
            v[factor_local_inds] = operation.to_vector()
        return v

    def from_vector(self, v: _np.ndarray, close: bool = False, dirty_value: bool = True) -> None:
        """
        Initialize the Instrument using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this Instrument's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        assert(self.gpindices is not None), "Must set an Instrument's .gpindices before calling from_vector"
        for operation, factor_local_inds in zip(self.values(), self._submember_rpindices):
            operation.from_vector(v[factor_local_inds], close, dirty_value)
        self.dirty = dirty_value

    def transform_inplace(self, s: GaugeGroupElement) -> None:
        """
        Update each Instrument element matrix `O` with `inv(s) * O * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # transform the MT and Di (self.param_ops) and re-init the elements.
        for gate in self.values():
            gate.transform_inplace(s)
        self.dirty = True

    def depolarize(self, amount: float | tuple) -> None:
        """
        Depolarize this Instrument by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # depolarize the MT and Di (self.param_ops) and re-init the elements.
        for gate in self.values():
            gate.depolarize(amount)
        self.dirty = True

    def rotate(self, amount: tuple, mx_basis: _BasisLike = 'gm') -> None:
        """
        Rotate this instrument by the given `amount`.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # rotate the MT and Di (self.param_ops) and re-init the elements.
        for gate in self.values():
            gate.rotate(amount, mx_basis)
        self.dirty = True

    def acton(self, state: State) -> _collections.OrderedDict[str, tuple[float, State]]:
        """
        Act with this instrument upon `state`

        Parameters
        ----------
        state : State
            The state to act on

        Returns
        -------
        OrderedDict
            A dictionary whose keys are the outcome labels (strings)
            and whose values are `(prob, normalized_state)` tuples
            giving the probability of seeing the given outcome and
            the resulting state that would be obtained if and when
            that outcome is observed.
        """
        assert(state._evotype == self._evotype), "Evolution type mismatch: %s != %s" % (self._evotype, state._evotype)

        staterep = state._rep
        outcome_probs_and_states = _collections.OrderedDict()

        for lbl, element in self.items():
            output_rep = element._rep.acton(staterep)
            unnormalized_state_array = output_rep.to_dense()
            prob = _ot.superket_trace(unnormalized_state_array, element.basis)
            output_state_array = unnormalized_state_array / prob
            output_state = _state.StaticState(output_state_array, self.evotype, self.state_space)
            outcome_probs_and_states[lbl] = (prob, output_state)

        return outcome_probs_and_states

    def __str__(self) -> str:
        s = "Instrument with elements:\n"
        for lbl, element in self.items():
            s += "%s:\n%s\n" % (lbl, _mt.mx_to_string(element.to_dense(), width=4, prec=2))
        return s
