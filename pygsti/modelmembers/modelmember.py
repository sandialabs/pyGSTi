"""
Defines the ModelChild and ModelMember classes, which represent Model members
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict
import copy as _copy

import numpy as _np

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.tools import listtools as _lt
from pygsti.tools import slicetools as _slct


class ModelChild(object):
    """
    Base class for all objects contained in a Model that hold a `parent` reference to their parent Model.

    Parameters
    ----------
    parent : Model, optional
        The parent model.

    Attributes
    ----------
    parent : Model
        The parent of this object.
    """

    def __init__(self, parent=None):
        self._parent = parent  # parent Model used to determine how to process
        # a LinearOperator's gpindices when inserted into a Model

    def copy(self, parent=None, memo=None):
        """
        Copy this object. Resets parent to None or `parent`.

        Parameters
        ----------
        parent : Model, optional
            The parent of the new, copied, object.

        Returns
        -------
        ModelChild
            A copy of this object.
        """
        #Copying resets or updates the parent of a ModelChild
        if memo is None:
            memo = {id(self.parent): None}  # so deepcopy uses None instead of copying parent
        else:
            memo[id(self.parent)] = None  # so deepcopy uses None instead of copying parent

        copyOfMe = _copy.deepcopy(self, memo)  # parent == None now
        copyOfMe.parent = parent
        return copyOfMe

    @property
    def parent(self):
        """
        Gets the parent of this object.

        Returns
        -------
        Model
        """
        return self._parent

    @parent.setter
    def parent(self, value):
        """
        Sets the parent of this object.

        Returns
        -------
        None
        """
        self._parent = value

    def __getstate__(self):
        """ Don't pickle parent """
        d = self.__dict__.copy()
        d['_parent'] = None
        return d


class ModelMember(ModelChild, _NicelySerializable):
    """
    Base class for Model member objects that possess a definite state space, parameters count, and evolution type.

    A ModelMember can be vectorized into/onto a portion of their parent Model's
    (or other ModelMember's) parameter vector.  They therefore contain a
    `gpindices` reference to the global Model indices "owned" by this member.
    Note that GateSetMembers may contain other GateSetMembers (may be nested).

    Parameters
    ----------
    state_space : StateSpace
        The state space, which should match the parent model if/when one exists.

    evotype : EvoType or str
        The evolution type, which should match the parent model if/when one exists.

    gpindices : slice or numpy.ndarray, optional
        The indices of this member's local parameters into the parent Model's
        parameter vector.

    parent : Model, optional
        The parent model.

    Attributes
    ----------
    dirty : bool
        Whether this member's local parameters may have been updated without
        its parent's knowledge.  The parent model can check this flag and perform
        re-synchronization of it's parameter vector when needed.

    gpindices : slice or numpy.ndarray
        The indices of this member's local parameters into the parent Model's
        parameter vector.

    parent : Model, optional
        The parent model.
    """

    def __init__(self, state_space, evotype, gpindices=None, parent=None):
        """ Initialize a new ModelMember """
        self._state_space = state_space
        self._evotype = evotype
        self._gpindices = gpindices
        self._submember_rpindices = ()  # parameter indices relative to this object's parameters
        self._paramlbls = None  # signals auto-generation of "unknown" parameter labels
        self._param_bounds = None  # either None or a (num_params,2)-shaped array of (lower,upper) bounds
        self._dirty = False  # True when there's any *possibility* that this
        # gate's parameters have been changed since the
        # last setting of dirty=False
        ModelChild.__init__(self, parent)
        _NicelySerializable.__init__(self)

    @property
    def state_space(self):
        return self._state_space

    # Need to work on this, since submembers shouldn't necessarily be updated to the same state space -- maybe a
    # replace_state_space_labels(...) member would be better?
    #@state_space.setter
    #def state_space(self, state_space):
    #    assert(self._state_space.is_compatible_with(state_space), "Cannot change to an incompatible state space!"
    #    for subm in self.submembers():
    #        subm.state_space = state_space
    #    return self._state_space = state_space

    @property
    def evotype(self):
        return self._evotype

    @property
    def dirty(self):
        """
        Flag indicating whether this member's local parameters may have been updated without its parent's knowledge.
        """
        return self._dirty

    @dirty.setter
    def dirty(self, value):
        """
        Flag indicating whether this member's local parameters may have been updated without its parent's knowledge.
        """
        self._dirty = value
        if value and self.parent:  # propagate "True" dirty flag to parent (usually a Model)
            self.parent.dirty = value

    @property
    def gpindices(self):
        """
        The indices of this member's local parameters into the parent Model's parameter vector.

        Returns
        -------
        slice or numpy.ndarray
        """
        return self._gpindices

    @gpindices.setter
    def gpindices(self, value):
        """
        The indices of this member's local parameters into the parent Model's parameter vector.
        """
        raise ValueError(("Use set_gpindices(...) to set the gpindices member"
                          " of a ModelMember object"))

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        if self._paramlbls is None:
            return _np.array(["Unknown param %d" % i for i in range(self.num_params)], dtype=object)
        return self._paramlbls

    @property
    def parameter_bounds(self):
        """ Upper and lower bounds on the values of each parameter, utilized by optimization routines """
        if self._param_bounds is not None:
            return self._param_bounds

        if len(self.submembers()) > 0:
            param_bounds = _np.empty((self.num_params, 2), 'd')
            param_bounds[:, 0] = -_np.inf
            param_bounds[:, 1] = +_np.inf
            for subm, local_inds in zip(self.submembers(), self._submember_rpindices):
                subm_bounds = subm.parameter_bounds
                if subm_bounds is not None:
                    param_bounds[local_inds] = subm_bounds

            # reset bounds to None if all are trivial
            if _np.all(param_bounds[:, 0] == -_np.inf) and _np.all(param_bounds[:, 1] == _np.inf):
                param_bounds = None
        else:
            param_bounds = self._param_bounds  # may be None

        return param_bounds

    @parameter_bounds.setter
    def parameter_bounds(self, val):
        """ Upper and lower bounds on the values of each parameter, utilized by optimization routines """
        if val is not None:
            assert(val.shape == (self.num_params, 2)), \
                "`parameter_bounds` can only be set to None or a (num_params, 2)-shaped array!"
        self._param_bounds = val
        if self.parent:
            self.parent._mark_for_rebuild(self)

    @property
    def parent(self):
        """
        The parent of this object.

        Returns
        -------
        Model
        """
        return self._parent

    @parent.setter
    def parent(self, value):
        """
        The parent of this object.
        """
        raise ValueError(("Use set_gpindices(...) to set the parent"
                          " of a ModelMember object"))

    def submembers(self):
        """
        Returns a sequence of any sub-ModelMember objects contained in this one.

        Sub-members are processed by other :class:`ModelMember` methods
        (e.g. `unlink_parent` and `set_gpindices`) as though the parent
        object is *just* a container for these sub-members and has no
        parameters of its own.  Member objects that contain other members
        *and* possess their own independent parameters should implement
        the appropriate `ModelMember` functions (usually just
        `allocate_gpindices`, using the base implementation as a reference).

        Returns
        -------
        list or tuple
        """
        return ()

    def relink_parent(self, parent):
        """
        Sets the parent of this object *without* altering its gpindices.

        This operation is appropriate to do when "re-linking" a parent with
        its children after the parent and child have been serialized.
        (the parent is *not* saved in serialization - see
         ModelChild.__getstate__ -- and so must be manually re-linked
         upon de-serialization).

        In addition to setting the parent of this object, this method
        sets the parent of any objects this object contains (i.e.
        depends upon) - much like allocate_gpindices.  To ensure a valid
        parent is not overwritten, the existing parent *must be None*
        prior to this call.

        Parameters
        ----------
        parent : Model
            The model to (re-)set as the parent of this member.

        Returns
        -------
        None
        """
        for subm in self.submembers():
            subm.relink_parent(parent)

        if self._parent is parent: return  # OK to relink multiple times
        assert(self._parent is None), "Cannot relink parent: parent is not None!"
        self._parent = parent  # assume no dependent objects

    def unlink_parent(self, force=False):
        """
        Remove the parent-link of this member.

        Called when at least one reference (via `key`) to this object is being
        disassociated with `parent`.   If *all* references are to this object
        are now gone, set parent to None, invalidating any gpindices.

        Parameters
        ----------
        force : bool, optional
            If `True`, then resets parent to `None`, effectively de-allocating
            the model member's parameters from the parent model, even if the
            parent still contains references to it.  If `False`, the parent
            is only set to `None` when its parent contains no reference to it.

        Returns
        -------
        None
        """
        for subm in self.submembers():
            subm.unlink_parent()

        if (self.parent is not None) and (force or self.parent._obj_refcount(self) == 0):
            self._parent = None

    # UNUSED - as this doesn't mark parameter for reallocation like it used to
    #def clear_gpindices(self):
    #    """
    #    Sets gpindices to None, along with any submembers' gpindices.
    #
    #    This essentially marks these members for parameter re-allocation
    #    (e.g. if the number - not just the value - of parameters they have
    #    changes).
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    for subm in self.submembers():
    #        subm.clear_gpindices()
    #    self._gpindices = None

    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that are used by this ModelMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : set, optional
            A set keeping track of the object ids that have had their indices
            set in a root `set_gpindices` call.  Used to prevent duplicate
            calls and self-referencing loops.  If `memo` contains an object's
            id (`id(self)`) then this routine will exit immediately.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        #must set the gpindices of sub-members based on new
        my_old_gpindices = self.gpindices
        if my_old_gpindices is not None:
            # update submembers if our gpindices were previously
            # set and are getting reset to something else.
            for i, (subm, subm_rpindices) in enumerate(zip(self.submembers(), self._submember_rpindices)):
                if id(subm) in memo: continue  # already processed
                new_subm_gpindices = _compose_gpindices(
                    gpindices, subm_rpindices)
                subm.set_gpindices(new_subm_gpindices, parent, memo)

        self._set_only_my_gpindices(gpindices, parent)

    def shift_gpindices(self, above, amount, parent_filter=None, memo=None):
        """
        Shifts this member's gpindices by the given amount.

        Usually called by the parent model when it shifts parameter
        indices around in its parameter vector.

        Parameters
        ----------
        above : int
            The "insertion point" within the range of indices.  All indices
            greater than or equal to this index are shifted.

        amount : int
            The amount to shift indices greater than or equal to `above` by.

        parent_filter : Model or None
            If a :class:`Model` object, then only those members with indices
            allocated to this model will be shifted.  It usually makes sense
            to specify this argument, supplying the parent model whose parameter
            vector is being shifted.

        memo : set, optional
            A set keeping track of the object ids that have had their indices
            set in a root `set_gpindices` call.  Used to prevent duplicate
            calls and self-referencing loops.  If `memo` contains an object's
            id (`id(self)`) then this routine will exit immediately.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        for subm in self.submembers():
            subm.shift_gpindices(above, amount, parent_filter, memo)

        if amount != 0 and (parent_filter is None or self.parent is parent_filter) and self.gpindices is not None:
            if isinstance(self.gpindices, slice):
                if self.gpindices.start >= above:
                    self._set_only_my_gpindices(_slct.shift(self.gpindices, amount), self.parent)
            else:
                shifted = _np.where(self.gpindices >= above, self.gpindices + amount, self.gpindices)
                self._set_only_my_gpindices(shifted, self.parent)  # works for integer arrays

    def _set_only_my_gpindices(self, gpindices, parent):
        self._parent = parent
        self._gpindices = gpindices

    def _collect_parents(self, set_to_fill=None, memo=None):
        """ Traverse sub-member tree and record all distinct parent Models. Useful for finding
            the "anticipated parent" model when initializing a member with sub-members """
        if set_to_fill is None: set_to_fill = set()
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        if len(self.submembers()) > 0:
            for subm in self.submembers():
                subm._collect_parents(set_to_fill, memo)
        if self.parent is not None:
            set_to_fill.add(self.parent)
        return set_to_fill

    def gpindices_are_allocated(self, model, memo=None):
        """
        Whether or not this model member's parameter indices are allocated within a potential parent model.

        This is used to infer when the model member needs to have its parameter indices
        reallocated (by its parent model).

        Parameters
        ----------
        model : Model
            Test for parameter allocation with respect to this model.

        memo : dict, optional
            Used to prevent duplicate calls and self-referencing loops.  If
            `memo` contains an object's id (`id(self)`) then this routine
            will exit immediately with a cached value.

        Returns
        -------
        bool
            True if this member's `.gpindices` and those of its sub-members are not None
            and refer to parameter indices of `model`.
        """
        # Note: maybe this routine should always return False if model is None (?)
        if memo is None: memo = {}
        elif id(self) in memo: return memo[id(self)]

        allocated = bool(self.gpindices is not None and model is self.parent)  # whether 'self' is allocated
        if allocated and len(self.submembers()) > 0:  # then also check that submembers are allocated too
            allocated = all([subm.gpindices_are_allocated(model, memo) for subm in self.submembers()])

        memo[id(self)] = allocated
        return allocated

    def preallocate_gpindices(self, parent, memo=None):
        """
        Computes two key pieces of information for a model preparing to allocate this member.

        These pieces of information are:

        1. the total number of new parameters that would need to be allocated to `parent`
           in order to have all this model member's parameters allocated to `parent`.
        2. the largest parameter index from all the parameters already allocated to
           `parent`.  This is useful for providing an ideal insertion point for the
           new parameters, once the model has made space for them.

        Note that this function does not update this model member's `.gpindces` or
        other attributes at all - it just serves as a pre-allocation probe so that
        the allocating model knows how much space in its parameter vector is needed/
        requested by this perhaps-not-fully-allocated member.

        Parameters
        ----------
        parent : Model
            The model that parameter allocation is being considered with respect to.

        memo : set, optional
            Used to prevent duplicate calls and self-referencing loops.  If
            `memo` contains an object's id (`id(self)`) then this routine
            will exit immediately..

        Returns
        -------
        num_new_params : int
            the number of parameters that aren't currently allocated to `parent`
        max_index : int
            the maximum index of the parameters currently allocated to `parent`
        """
        assert(parent is not None), \
            "Parent model cannot be `None`. Just call allocate_gpindices if there is no parent."

        if memo is None: memo = set()
        elif id(self) in memo:
            return (0, -1)  # if already processes then don't need to do so again, and
            # returning this is saying "don't tally any more new-params or adjust the max index"

        memo.add(id(self))  # don't call this again

        if len(self.submembers()) > 0:
            #Allocate sub-members
            max_existing_index = -1; tot_new_params = 0
            for subm in self.submembers():
                num_new_params, maxindex = subm.preallocate_gpindices(parent, memo)
                tot_new_params += num_new_params
                max_existing_index = max(max_existing_index, maxindex)
            return tot_new_params, max_existing_index

        else:  # no sub-members

            if self.gpindices is None or parent is not self.parent:
                # Note: parent == None has the special meaning of "no parent model".  When
                # a modelmember is called upon to allocate indices to this non-model it
                # *always* allocates the parameters even if it's self.parent is currently None.

                #default behavior: assume num_params() works even with
                # gpindices == None and indicate all our parameters should be allocated as "new"
                return (self.num_params, -1)  # all params new, max existing index = -1 (no existing inds)
            else:  # assume gpindices is good & everything's allocated already
                max_existing_index = (self.gpindices.stop - 1) if isinstance(self.gpindices, slice) \
                    else max(self.gpidices)  # an array
                return 0, max_existing_index

    def allocate_gpindices(self, starting_index, parent, memo=None, submembers_already_allocated=False):
        """
        Sets gpindices array for this object or any objects it contains (i.e. depends upon).

        Indices may be obtained from contained objects which have already been
        initialized (e.g. if a contained object is shared with other top-level
        objects), or given new indices starting with `starting_index`.

        Parameters
        ----------
        starting_index : int
            The starting index for un-allocated parameters.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : set, optional
            Used to prevent duplicate calls and self-referencing loops.  If
            `memo` contains an object's id (`id(self)`) then this routine
            will exit immediately.

        submembers_already_allocated : bool, optional
            Whether submembers of this object are known to already have their
            parameter indices allocated to `parent`.  Leave this as `False`
            unless you know what you're doing.

        Returns
        -------
        num_new : int
            The number of *new* allocated parameters (so
            the parent should mark as allocated parameter
            indices `starting_index` to `starting_index + new_new`).
        """
        if memo is None: memo = set()
        elif id(self) in memo: return 0

        if len(self.submembers()) > 0:

            if submembers_already_allocated:
                tot_new_params = 0
                submembers_with_params = [subm for subm in self.submembers() if
                                          ((isinstance(subm.gpindices, slice) and subm.gpindices != slice(0, 0))
                                           or (not isinstance(subm.gpindices, slice) and len(subm.gpindices) > 0))]
                num_submembers_with_params = len(submembers_with_params)
                if num_submembers_with_params == 0:  # Special case (for speed)
                    gpindices_slice_if_possible = slice(0, 0)
                elif num_submembers_with_params == 1:  # Special case (for speed)
                    gpindices_slice_if_possible = submembers_with_params[0].gpindices
                else:
                    gpindices_slice_if_possible = _merge_indices(
                        [subm.gpindices for subm in submembers_with_params], submembers_with_params)
            else:
                #print("ALLOC SUBMEMBERS!")
                #Allocate sub-members
                tot_new_params = 0
                for subm in self.submembers():
                    num_new_params = subm.allocate_gpindices(starting_index, parent, memo)  # *same* parent as self
                    starting_index += num_new_params
                    tot_new_params += num_new_params

                submembers_with_params = [subm for subm in self.submembers() if
                                          ((isinstance(subm.gpindices, slice) and subm.gpindices != slice(0, 0))
                                           or (not isinstance(subm.gpindices, slice) and len(subm.gpindices) > 0))]
                gpindices_slice_if_possible = _merge_indices(
                    [subm.gpindices for subm in submembers_with_params], submembers_with_params)

            #Then just set the gpindices of this member to those used by
            # its submembers - assume this object doesn't need to allocate any
            # indices of its own. (otherwise derived class should override this!)
            #Note: don't call self.set_gpindices here as this is used to shift
            # or change an already allocated ._gpindices slice/array.  Here we
            # need to set (really, "allocate") *just* the ._gpindices of this
            # object, not the submembers as this is already done above.
            memo.add(id(self))  # would have been called in a proper set_gpindices call
            self._set_only_my_gpindices(gpindices_slice_if_possible, parent)

            #parameter index allocation also freezes the relative indices
            # between this object's parameter indices and those of its submembers
            self._submember_rpindices = tuple([_decompose_gpindices(
                self.gpindices, subm.gpindices) for subm in self.submembers()])

            return tot_new_params

        else:  # no sub-members

            #DEBUG def pp(x): return id(x) if (x is not None) else x
            #DEBUG
            # print(" >>> DB DEFAULT %d ALLOCATING: " % id(self),
            #       self.gpindices, " parents:",
            #       pp(self.parent), pp(parent))
            if self.gpindices is None or parent is None or parent is not self.parent:
                # Note: parent == None has the special meaning of "no parent model".  When
                # a modelmember is called upon to allocate indices to this non-model it
                # *always* allocates the parameters even if it's self.parent is currently None.

                #default behavior: assume num_params() works even with
                # gpindices == None and allocate all our parameters as "new"
                Np = self.num_params
                slc = slice(starting_index, starting_index + Np) \
                    if Np > 0 else slice(0, 0, None)  # special "null slice" for zero params
                self.set_gpindices(slc, parent, memo)
                #print(" -- allocated %d indices: %s" % (Np,str(slc)))
                return Np
            else:  # assume gpindices is good & everything's allocated already
                #print(" -- no need to allocate anything")
                return 0

    def _obj_refcount(self, obj):
        """ Number of references to `obj` contained within this object """
        cnt = 1 if (obj is self) else 0
        for subm in self.submembers():
            cnt += subm._obj_refcount(obj)
        return cnt

    def init_gpindices(self, allocated_to_parent=None):
        """
        Initializes this model member's parameter indices by allocating them to an "anticipated" parent model.

        Objects with submembers often rely on having valid `.gpindices` and `.subm_rpindices`
        attributes, but these aren't set until the object is allocated to a parent model.  This
        method initializes these attributes in the best way possible *before* receiving the
        actual parent model.  Typically model members (containing sub-members) are build in two ways:

        1. The sub-members are all un-allocated, i.e. their `.parent` model is `None`
        2. The sub-members are all allocated to the *same* parent model.

        This method computes an "anticipated parent" model as the common parent of all
        the submembers (if one exists) or `None`, and calls :method:`allocate_gpindices`
        using this parent model and a starting index of 0.  This has the desired behavior
        in the two cases above.  In case 1, parameter indices are set (allocated) but the
        parent is set to `None`, so that the to-be parent model will see this member as
        being unallocated.  In case 2, the parent model, if it is the anticipated one,
        will see that this member's indices are already allocated to it, and won't need
        to re-allocate them.
        """
        if allocated_to_parent is not None:
            #Shortcut - assume that all submembers are already allocated to *this* (common) parent
            num_new_params = self.allocate_gpindices(starting_index=0, parent=allocated_to_parent,
                                                     submembers_already_allocated=True)
            assert(num_new_params == 0), \
                "No new parameters should have needed to be added (all sub-members were already allocated)!"
            return

        all_parents = self._collect_parents()
        if len(all_parents) == 1:  # all sub-members have a common (non-None) parent model
            # Next, check that all the sub-members are also *allocated* (have non-None gpindices) to this common parent
            common_parent = all_parents.__iter__().__next__()
            if all([subm.gpindices_are_allocated(common_parent) for subm in self.submembers()]):
                # If so, then we can allocate our gpindices using this common parent
                num_new_params = self.allocate_gpindices(starting_index=0, parent=common_parent)
                assert(num_new_params == 0), \
                    "No new parameters should have needed to be added (all sub-members were already allocated)!"
                return

        # Otherwise, we'll re-allocate all the indices using parent=None so that gpindices
        #  and subm_rpindices get set for now, but will trigger a re-allocation when added
        #  to an actual parent model.  When this re-allocation occurs, the basic properties
        #  of this member (num_params, etc) should be the same as they are now (with parent=None).
        self.allocate_gpindices(starting_index=0, parent=None)

    def gpindices_as_array(self):
        """
        Returns gpindices as a `numpy.ndarray` of integers.

        The underlying `.gpindices` attribute itself can be None, a slice,
        or an integer array.  If gpindices is None, an empty array is returned.

        Returns
        -------
        numpy.ndarray
        """
        if self._gpindices is None:
            return _np.empty(0, _np.int64)
        elif isinstance(self._gpindices, slice):
            return _np.array(_slct.indices(self._gpindices), _np.int64)
        else:
            return self._gpindices  # it's already an array

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this object.

        Returns
        -------
        int
        """
        return 0  # by default, object has no parameters

    def to_vector(self):
        """
        Get this object's parameters as a 1D array of values.

        Returns
        -------
        numpy.ndarray
        """
        return _np.array([], 'd')  # no parameters

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this object using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to the current parameter vector.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        assert(len(v) == 0)  # should be no parameters, and nothing to do

    def copy(self, parent=None, memo=None):
        """
        Copy this object.

        Parameters
        ----------
        parent : Model, optional
            The parent of the returned copy.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # A default for derived classes - deep copy everything except the parent,
        #  which will get reset by _copy_gpindices anyway.
        if memo is None:
            memo = {id(self.parent): None}  # so deepcopy uses None instead of copying parent
        else:
            if id(self) in memo: return memo[id(self)]
            memo[id(self.parent)] = None  # so deepcopy uses None instead of copying parent
        return self._copy_gpindices(_copy.deepcopy(self, memo), parent, memo)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return True  # default is to have no additional checks for similarity

    def is_similar(self, other, rtol=1e-5, atol=1e-8):
        """
        Comparator returning whether two ModelMembers are the same type
        and parameterization.

        ModelMembers with internal parameterization information (e.g.
        LindbladErrorgen) should overload this function to account for that.

        Parameters
        ---------
        other: ModelMember
            ModelMember to compare to
        rtol: float
            Relative tolerance for floating poing comparisons (passed to numpy.isclose)
        atol: float
            Absolute tolerance for floating point comparisons (passed to numpy.isclose)

        Returns
        -------
        bool
            True if structure (all but parameter *values*) matches
        """
        if type(self) != type(other): return False
        #if str(self.evotype) != str(other.evotype): return False  # allow models of different evotypes to be similar
        if not self._is_similar(other, rtol, atol): return False

        # Recursive check on submembers
        if len(self.submembers()) != len(other.submembers()): return False
        for sm1, sm2 in zip(self.submembers(), other.submembers()):
            if not sm1.is_similar(sm2): return False

        return True

    def is_equivalent(self, other, rtol=1e-5, atol=1e-8):
        """
        Comparator returning whether two ModelMembers are equivalent.

        This uses is_similar for type checking and NumPy allclose for parameter
        checking, so is unlikely to be needed to overload.

        Note that this only checks for NUMERICAL equivalence, not whether the objects
        are the same.

        Parameters
        ---------
        other: ModelMember
            ModelMember to compare to
        rtol: float
            Relative tolerance for floating point comparisons (passed to numpy.isclose)
        atol: float
            Absolute tolerance for floating point comparisons (passed to numpy.isclose)

        Returns
        -------
        bool
            True if structure AND parameter vectors match
        """
        if not self.is_similar(other): return False

        if not _np.allclose(self.to_vector(), other.to_vector(), rtol=rtol, atol=atol):
            return False

        # Recursive check on submembers
        if len(self.submembers()) != len(other.submembers()): return False
        for sm1, sm2 in zip(self.submembers(), other.submembers()):
            # Technically calling is_equivalent here is extra type check work,
            # but this is safer in case is_equivalent is overloaded in derived classes
            if not sm1.is_equivalent(sm2): return False

        return True

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
        mm_dict = OrderedDict()
        mm_dict['module'] = self.__module__
        mm_dict['class'] = self.__class__.__name__
        mm_dict['submembers'] = []
        mm_dict['state_space'] = self.state_space.to_nice_serialization()
        mm_dict['evotype'] = str(self.evotype)
        mm_dict['model_parameter_indices'] = self._encodemx(self.gpindices_as_array())
        mm_dict['relative_submember_parameter_indices'] = [list(map(int, _slct.to_array(inds)))
                                                           for inds in self._submember_rpindices]
        assert(self._paramlbls is None or isinstance(self._paramlbls, _np.ndarray))
        mm_dict['parameter_labels'] = self._paramlbls.tolist() if (self._paramlbls is not None) else None
        mm_dict['parameter_bounds'] = self._encodemx(self._param_bounds)

        # Dereference submembers
        for sm in self.submembers():
            try:
                mm_node = mmg_memo[id(sm)]
                mm_dict['submembers'].append(mm_node.serialize_id)
            except KeyError:
                print('Each submember must be in the memo.')
                print('Submember Id: ', id(sm), ', Submember: \n', sm)
                print('Memo:\n')
                for k, v in mmg_memo.items():
                    print('  Id: ', k, ', Serialize Id: ', v.serialize_id, end=', ')
                    print('  ModelMember:\n', v.mm)

        return mm_dict

    @classmethod
    def _check_memoized_dict(cls, mm_dict, serial_memo):
        """Performs simple checks to ensure that `mm_dict` corresponds to the
           actual class( `cls`) being created, and that all submembers are present in `serial_memo` """
        needed_tags = ['module', 'class', 'submembers', 'state_space', 'evotype']
        assert all([tag in mm_dict.keys() for tag in needed_tags]), 'Must provide all needed tags: %s' % needed_tags

        assert mm_dict['module'] == cls.__module__, "Module must match"
        assert mm_dict['class'] == cls.__name__, "Class must match"
        assert all([(sub_id in serial_memo) for sub_id in mm_dict['submembers']]), "Not all sub-members exist!"

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        """
        For subclasses to implement.  Submember-existence checks are performed,
        and the gpindices of the return value is set, by the non-underscored
        :method:`from_memoized_dict` implemented in this class.
        """
        #E.g.:
        # assert len(mm_dict['submembers']) == 0, 'ModelMember base class has no submembers'
        # return cls(mm_dict['state_space'], mm_dict['evotype'])
        raise NotImplementedError("Derived classes should implement this!")

    @classmethod
    def from_memoized_dict(cls, mm_dict, serial_memo, parent_model):
        """Deserialize a ModelMember object and relink submembers from a memo.

        Parameters
        ----------
        mm_dict: dict
            A dict representation of this ModelMember ready for deserialization
            This must have at least the following fields:
                module, class, submembers, state_space, evotype

        serial_memo: dict
            Keys are serialize_ids and values are ModelMembers. This is NOT the same as
            other memos in ModelMember, (e.g. copy(), allocate_gpindices(), etc.).
            This is similar but not the same as mmg_memo in to_memoized_dict(),
            as we do not need to build a ModelMemberGraph for deserialization.

        parent_model: Model
            The parent model that being build that will eventually hold this ModelMember object.
            It's important to set this so that the Model considers the set gpindices *valid* and
            doesn't just wipe them out when cleaning its parameter vector.

        Returns
        -------
        ModelMember
            An initialized object
        """
        cls._check_memoized_dict(mm_dict, serial_memo)
        obj = cls._from_memoized_dict(mm_dict, serial_memo)

        my_gpindices = _slct.list_to_slice(cls._decodemx(mm_dict['model_parameter_indices']), array_ok=True)
        obj._set_only_my_gpindices(my_gpindices, parent=parent_model)
        obj._submember_rpindices = tuple([_slct.list_to_slice(inds)
                                          for inds in mm_dict['relative_submember_parameter_indices']])
        if mm_dict['parameter_labels'] is not None:
            obj._paramlbls = _np.empty(len(mm_dict['parameter_labels']), dtype=object)
            obj._paramlbls[:] = mm_dict['parameter_labels']  # 2-step init because otherwise we end up with a 2D array
        else:
            obj._paramlbls = None
        obj._param_bounds = cls._decodemx(mm_dict['parameter_bounds'])

        return obj

    def _copy_gpindices(self, op_obj, parent, memo):
        """ Helper function for implementing copy in derived classes """
        gpindices_copy = None
        if isinstance(self.gpindices, slice):
            gpindices_copy = self.gpindices  # slices are immutable
        elif self.gpindices is not None:
            gpindices_copy = self.gpindices.copy()  # arrays are not

        #Call base class implementation here because
        # "advanced" implementations containing sub-members assume that the
        # gpindices has already been set and is just being updated (so it compares
        # the "old" (existing) gpindices with the value being set).  Here,
        # we just want to copy any existing gpindices from "self" to op_obj
        # and *not* cause op_obj to shift any underlying indices (they'll
        # be copied separately. -- FUTURE: make separate "update_gpindices" and
        # "copy_gpindices" functions?
        op_obj._set_only_my_gpindices(gpindices_copy, parent)
        #op_obj.set_gpindices(gpindices_copy, parent) #don't do this, as
        # this routines doesn't copy sub-member indices yet -- copy(...) methods
        # of derived classes do this.

        #copy the relative indices between this object's parameter indices and those of its submembers
        op_obj._submember_rpindices = tuple([rinds if isinstance(rinds, slice) else rinds.copy()
                                             for rinds in self._submember_rpindices])

        #For convenience, also perform the memo update here, so derived classes don't need to repeat logic
        if memo is not None:
            memo[id(self)] = op_obj

        return op_obj

    def _print_gpindices(self, prefix="", member_label=None, param_labels=None, max_depth=0):
        nsub = len(self.submembers())
        print(prefix
              + ">>> " + (str(member_label) if (member_label is not None) else "")
              + " [%s]: %d params, indices=%s" % (type(self).__name__, self.num_params, str(self.gpindices))
              + ((", %d sub-members" % nsub) if (nsub > 0 and max_depth == 0) else ""))
        # Note: we only print # of sub-members if they're not shown below.

        if param_labels is not None:
            for i, plbl in zip(self.gpindices_as_array(), self.parameter_labels):
                gplabel = param_labels.get(i, "(no label)")
                if gplabel == "(no label)" or gplabel == (member_label, plbl):
                    # this parameter corresponds to the default model label for this object,
                    # so don't print the member label for brevity
                    lbl = str(plbl)
                else:
                    lbl = str(plbl) + " --> " + str(gplabel)
                print(prefix + "   %d: %s" % (i, lbl))
        #print(self.gpindices, " [%s]" % str(type(self)))
        if max_depth > 0:
            for i, sub in enumerate(self.submembers()):
                #print(prefix, "  Sub%d: " % i, end='')
                sub._print_gpindices(prefix + "  ", "Sub%d" % i, param_labels, max_depth - 1)

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "(contents not available)"


def _compose_gpindices(parent_gpindices, child_gpindices):
    """
    Maps `child_gpindices`, which index `parent_gpindices` into a new slice
    or array of indices that is the subset of parent indices.

    Essentially:
    `return parent_gpindices[child_gpindices]`
    """
    if parent_gpindices is None or child_gpindices is None: return None
    if isinstance(child_gpindices, slice) and child_gpindices == slice(0, 0, None):
        return slice(0, 0, None)  # "null slice" ==> "null slice" convention
    if isinstance(parent_gpindices, slice):
        start = parent_gpindices.start
        assert(parent_gpindices.step is None), "No support for nontrivial step size yet"

        if isinstance(child_gpindices, slice):
            return _slct.shift(child_gpindices, start)
        else:  # child_gpindices is an index array
            return child_gpindices + start  # numpy "shift"

    else:  # parent_gpindices is an index array, so index with child_gpindices
        return parent_gpindices[child_gpindices]


def _decompose_gpindices(parent_gpindices, sibling_gpindices):
    """
    Maps `sibling_gpindices`, which index the same space as `parent_gpindices`,
    into a new slice or array of indices that gives the indices into
    `parent_gpindices` which result in `sibling_gpindices` (this requires that
    `sibling_indices` lies within `parent_gpindices`.

    Essentially:
    `sibling_gpindices = parent_gpindices[returned_gpindices]`
    """
    if parent_gpindices is None or sibling_gpindices is None: return None
    if isinstance(parent_gpindices, slice):
        start, stop = parent_gpindices.start, parent_gpindices.stop
        assert(parent_gpindices.step is None), "No support for nontrivial step size yet"

        if isinstance(sibling_gpindices, slice):
            if sibling_gpindices.start == sibling_gpindices.stop == 0:  # "null slice"
                return slice(0, 0, None)  # ==> just return null slice
            assert(start <= sibling_gpindices.start and sibling_gpindices.stop <= stop), \
                "Sibling indices (%s) must be a sub-slice of parent indices (%s)!" % (
                    str(sibling_gpindices), str(parent_gpindices))
            return _slct.shift(sibling_gpindices, -start)
        else:  # child_gpindices is an index array
            return sibling_gpindices - start  # numpy "shift"

    else:  # parent_gpindices is an index array
        sibInds = _slct.indices(sibling_gpindices) \
            if isinstance(sibling_gpindices, slice) else sibling_gpindices
        parent_lookup = {j: i for i, j in enumerate(parent_gpindices)}
        return _np.array([parent_lookup[j] for j in sibInds], _np.int64)
        #Note: this will work even for "null array" when sibling_gpindices is empty


_merge_indices_cache = {}


def _merge_indices(gpindices_list, member_list):
    cache_key = tuple([(('slc', gpi.start, gpi.stop) if isinstance(gpi, slice) else tuple(gpi))
                       for gpi in gpindices_list])

    if cache_key not in _merge_indices_cache:
        all_gpindices = []
        for m in member_list:
            all_gpindices.extend(m.gpindices_as_array())
        _lt.remove_duplicates_in_place(all_gpindices)  # in case multiple members have the same param
        all_gpindices.sort()
        _merge_indices_cache[cache_key] = _slct.list_to_slice(all_gpindices, array_ok=True,
                                                              require_contiguous=True)
    return _merge_indices_cache[cache_key]
