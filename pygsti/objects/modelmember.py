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

import numpy as _np
import copy as _copy
from ..tools import slicetools as _slct
from ..tools import listtools as _lt


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

    def copy(self, parent=None):
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
        memo = {id(self.parent): None}  # so deepcopy uses None instead of copying parent
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


class ModelMember(ModelChild):
    """
    Base class for Model member objects that possess a definite dimension, parameters count, and evolution type.

    A ModelMember can be vectorized into/onto a portion of their parent Model's
    (or other ModelMember's) parameter vector.  They therefore contain a
    `gpindices` reference to the global Model indices "owned" by this member.
    Note that GateSetMembers may contain other GateSetMembers (may be nested).

    Parameters
    ----------
    dim : int
        The dimension.

    evotype : str
        The evolution type.

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

    def __init__(self, dim, evotype, gpindices=None, parent=None):
        """ Initialize a new ModelMember """
        self.dim = dim
        self._evotype = evotype
        self._gpindices = gpindices
        self._gplabels = None  # a placeholder for FUTURE features
        self._dirty = False  # True when there's any *possibility* that this
        # gate's parameters have been changed since the
        # last setting of dirty=False
        super(ModelMember, self).__init__(parent)

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

    def unlink_parent(self):
        """
        Remove the parent-link of this member.

        Called when at least one reference (via `key`) to this object is being
        disassociated with `parent`.   If *all* references are to this object
        are now gone, set parent to None, invalidating any gpindices.
        `starting_index`.

        Returns
        -------
        None
        """
        for subm in self.submembers():
            subm.unlink_parent()

        if (self.parent is not None) and (self.parent._obj_refcount(self) == 0):
            self._parent = None

    def clear_gpindices(self):
        """
        Sets gpindices to None, along with any submembers' gpindices.

        This essentially marks these members for parameter re-allocation
        (e.g. if the number - not just the value - of parameters they have
        changes).

        Returns
        -------
        None
        """
        for subm in self.submembers():
            subm.clear_gpindices()
        self._gpindices = None

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
            for i, subm in enumerate(self.submembers()):
                if id(subm) in memo: continue  # already processed
                rel_subm_gpindices = _decompose_gpindices(
                    my_old_gpindices, subm.gpindices)
                new_subm_gpindices = _compose_gpindices(
                    gpindices, rel_subm_gpindices)
                subm.set_gpindices(new_subm_gpindices, parent, memo)

        self._set_only_my_gpindices(gpindices, parent)

    def _set_only_my_gpindices(self, gpindices, parent):
        self._parent = parent
        self._gpindices = gpindices

    def allocate_gpindices(self, starting_index, parent, memo=None):
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
            #Allocate sub-members
            tot_new_params = 0; all_gpindices = []
            for subm in self.submembers():
                num_new_params = subm.allocate_gpindices(starting_index, parent, memo)  # *same* parent as this member
                starting_index += num_new_params
                tot_new_params += num_new_params
                all_gpindices.extend(subm.gpindices_as_array())

            _lt.remove_duplicates_in_place(all_gpindices)  # in case multiple submembers have the same params
            all_gpindices.sort()

            #Then just set the gpindices of this member to those used by
            # its submembers - assume this object doesn't need to allocate any
            # indices of its own. (otherwise derived class should override this!)
            #Note: don't call self.set_gpindices here as this is used to shift
            # or change an already allocated ._gpindices slice/array.  Here we
            # need to set (really, "allocate") *just* the ._gpindices of this
            # object, not the submembers as this is already done above.
            memo.add(id(self))  # would have been called in a proper set_gpindices call
            self._set_only_my_gpindices(
                _slct.list_to_slice(all_gpindices, array_ok=True, require_contiguous=True),
                parent)
            return tot_new_params

        else:  # no sub-members

            #DEBUG def pp(x): return id(x) if (x is not None) else x
            #DEBUG
            # print(" >>> DB DEFAULT %d ALLOCATING: " % id(self),
            #       self.gpindices, " parents:",
            #       pp(self.parent), pp(parent))
            if self.gpindices is None or parent is not self.parent:
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

    def copy(self, parent=None):
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
        memo = {id(self.parent): None}  # so deepcopy uses None instead of copying parent
        return self._copy_gpindices(_copy.deepcopy(self, memo), parent)

    def _copy_gpindices(self, op_obj, parent):
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

        return op_obj

    def _print_gpindices(self, prefix=""):
        print(self.gpindices, " [%s]" % str(type(self)))
        for i, sub in enumerate(self.submembers()):
            print(prefix, "  Sub%d: " % i, end='')
            sub._print_gpindices(prefix + "  ")


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
