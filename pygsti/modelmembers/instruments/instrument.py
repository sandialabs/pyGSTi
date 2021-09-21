"""
Defines the Instrument class
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
import numpy as _np

from pygsti.modelmembers import modelmember as _mm
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import states as _state
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import matrixtools as _mt
from pygsti.tools import slicetools as _slct
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.statespace import StateSpace as _StateSpace


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

    def __init__(self, member_ops, evotype=None, state_space=None, items=[]):
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
                    evotype = _Evotype.cast('default')
                member_list = [(k, v if isinstance(v, _op.LinearOperator) else
                                _op.FullArbitraryOp(v, evotype, state_space)) for k, v in member_list]

            assert(len(member_list) > 0 or state_space is not None), \
                "Must specify `state_space` when there are no instrument members!"
            assert(len(member_list) > 0 or evotype is not None), \
                "Must specify `evotype` when there are no instrument members!"
            evotype = _Evotype.cast(evotype) if (evotype is not None) else member_list[0][1].evotype
            state_space = member_list[0][1].state_space if (state_space is None) \
                else _statespace.StateSpace.cast(state_space)

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
        #REMOVE self._paramvec, self._paramlbls = self._build_paramvec()
        self.init_gpindices()
        self._readonly = True

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return list(self.values())

    def to_memoized_dict(self, mmg_memo):
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
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])
        members = [(lbl, serial_memo[subm_serial_id])
                   for lbl, subm_serial_id in zip(mm_dict['member_labels'], mm_dict['submembers'])]
        return cls(members, mm_dict['evotype'], state_space)

    #REMOVE
    #No good way to update Instrument on the fly yet...
    #def _update_paramvec(self, modified_obj=None):
    #    """Updates self._paramvec after a member of this Model is modified"""
    #    for obj in self.values():
    #        assert(obj.gpindices is self), "Cannot add/adjust parameter vector!"
    #
    #    #update parameters changed by modified_obj
    #    self._paramvec[modified_obj.gpindices] = modified_obj.to_vector()
    #
    #    #re-initialze any members that also depend on the updated parameters
    #    modified_indices = set(modified_obj.gpindices_as_array())
    #    for obj in self.values()
    #        if obj is modified_obj: continue
    #        if modified_indices.intersection(obj.gpindices_as_array()):
    #            obj.from_vector(self._paramvec[obj.gpindices])

    #def _build_paramvec(self):
    #    """ Resizes self._paramvec and updates gpindices & parent members as needed,
    #        and will initialize new elements of _paramvec, but does NOT change
    #        existing elements of _paramvec (use _clean_paramvec for this)"""
    #    v = _np.empty(0, 'd'); off = 0
    #    vl = _np.empty(0, dtype=object)
    #
    #    # Step 2: add parameters that don't exist yet
    #    for lbl, obj in self.items():
    #        if obj.gpindices is None or obj.parent is not self:
    #            #Assume all parameters of obj are new independent parameters
    #            v = _np.insert(v, off, obj.to_vector())
    #            vl = _np.insert(vl, off, ["%s: %s" % (str(lbl), obj_plbl) for obj_plbl in obj.parameter_labels])
    #            num_new_params = obj.allocate_gpindices(off, self)
    #            off += num_new_params
    #        else:
    #            inds = obj.gpindices_as_array()
    #            M = max(inds) if len(inds) > 0 else -1; L = len(v)
    #            if M >= L:
    #                #Some indices specified by obj are absent, and must be created.
    #                w = obj.to_vector()
    #                wl = _np.array(["%s: %s" % (str(lbl), obj_plbl) for obj_plbl in obj.parameter_labels])
    #                v = _np.concatenate((v, _np.empty(M + 1 - L, 'd')), axis=0)  # [v.resize(M+1) doesn't work]
    #                vl = _np.concatenate((vl, _np.empty(M + 1 - L, dtype=object)), axis=0)
    #                for ii, i in enumerate(inds):
    #                    if i >= L:
    #                        v[i] = w[ii]
    #                        vl[i] = wl[ii]
    #            off = M + 1
    #    return v, vl

    #def _clean_paramvec(self):
    #    """ Updates _paramvec corresponding to any "dirty" elements, which may
    #        have been modified without out knowing, leaving _paramvec out of
    #        sync with the element's internal data.  It *may* be necessary
    #        to resolve conflicts where multiple dirty elements want different
    #        values for a single parameter.  This method is used as a safety net
    #        that tries to insure _paramvec & Instrument elements are consistent
    #        before their use."""
    #
    #    #Currently there's not "need-to-rebuild" flag because we don't let the user change
    #    # the elements of an Instrument after it's created.
    #    #if self._need_to_rebuild:
    #    #    self._build_paramvec()
    #    #    self._need_to_rebuild = False
    #
    #    # This closely parallels the _clean_paramvec method of a Model (TODO: consolidate?)
    #    if self.dirty:  # if any member object is dirty (ModelMember.dirty setter should set this value)
    #        TOL = 1e-8
    #
    #        #Note: lbl args used *just* for potential debugging - could strip out once
    #        # we're confident this code always works.
    #        def clean_single_obj(obj, lbl):  # sync an object's to_vector result w/_paramvec
    #            if obj.dirty:
    #                w = obj.to_vector()
    #                chk_norm = _np.linalg.norm(self._paramvec[obj.gpindices] - w)
    #                #print(lbl, " is dirty! vec = ", w, "  chk_norm = ",chk_norm)
    #                if (not _np.isfinite(chk_norm)) or chk_norm > TOL:
    #                    self._paramvec[obj.gpindices] = w
    #                obj.dirty = False
    #
    #        def clean_obj(obj, lbl):  # recursive so works with objects that have sub-members
    #            for i, subm in enumerate(obj.submembers()):
    #                clean_obj(subm, _Label(lbl.name + ":%d" % i, lbl.sslbls))
    #            clean_single_obj(obj, lbl)
    #
    #        for lbl, obj in self.items():
    #            clean_obj(obj, lbl)
    #
    #        #re-update everything to ensure consistency ~ self.from_vector(self._paramvec)
    #        #print("DEBUG: non-trivially CLEANED paramvec due to dirty elements")
    #        for obj in self.values():
    #            obj.from_vector(self._paramvec[obj.gpindices], dirty_value=False)
    #            #object is known to be consistent with _paramvec
    #
    #        self.dirty = False

    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter Instrument elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        #need to *not* pickle parent, as __reduce__ bypasses ModelMember.__getstate__
        dict_to_pickle = self.__dict__.copy()
        dict_to_pickle['_parent'] = None

        #Note: must *copy* elements for pickling/copying
        return (Instrument, (None, self.evotype, self.state_space, [(key, gate.copy()) for key, gate in self.items()]),
                dict_to_pickle)

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def simplify_operations(self, prefix=""):
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
                comp = g #.copy()
                #REMOVE (components now hold global model indices)
                #comp.set_gpindices(_mm._compose_gpindices(self.gpindices,
                #                                          g.gpindices), self.parent)
                simplified[_Label(prefix.name + "_" + k, prefix.sslbls)] = comp
        else:
            if prefix: prefix += "_"
            for k, g in self.items():
                comp = g #.copy()
                #REMOVE (components now hold global model indices)
                #comp.set_gpindices(_mm._compose_gpindices(self.gpindices,
                #                                          g.gpindices), self.parent)
                simplified[prefix + k] = comp
        return simplified

    @property
    def parameter_labels(self):
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
    def num_elements(self):
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
    def num_params(self):
        """
        Get the number of independent parameters which specify this Instrument.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
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

    def from_vector(self, v, close=False, dirty_value=True):
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

    def transform_inplace(self, s):
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
            #REMOVE self._paramvec[gate.gpindices] = gate.to_vector()
        self.dirty = True

    def depolarize(self, amount):
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
            #REMOVE self._paramvec[gate.gpindices] = gate.to_vector()
        self.dirty = True

    def rotate(self, amount, mx_basis='gm'):
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
            #REMOVE self._paramvec[gate.gpindices] = gate.to_vector()
        self.dirty = True

    def acton(self, state):
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
            output_unnormalized_state = output_rep.to_dense()
            prob = output_unnormalized_state[0] * state.dim**0.25
            output_normalized_state = output_unnormalized_state / prob  # so [0]th == 1/state_dim**0.25
            outcome_probs_and_states[lbl] = (prob, _state.StaticState(output_normalized_state, self.evotype,
                                                                      self.state_space))

        return outcome_probs_and_states

    def __str__(self):
        s = "Instrument with elements:\n"
        for lbl, element in self.items():
            s += "%s:\n%s\n" % (lbl, _mt.mx_to_string(element.to_dense(), width=4, prec=2))
        return s
