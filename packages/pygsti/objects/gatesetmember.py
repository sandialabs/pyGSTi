""" Defines the GatesetObject class, which represents GateSet members """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy
from ..tools import slicetools as _slct

class GateSetChild(object):
    """
    Base class for all objects contained in a GateSet that
    hold a `parent` reference to their parent GateSet.
    """
    def __init__(self, parent=None):
        self._parent = parent # parent GateSet used to determine how to process
                              # a Gate's gpindices when inserted into a GateSet

    @property
    def parent(self):
        """ Gets the parent of this object."""
        return self._parent

    @parent.setter
    def parent(self, value):
        """ Sets the parent of this object."""
        self._parent = value

    def __getstate__(self):
        """ Don't pickle parent """
        d = self.__dict__.copy()
        d['_parent'] = None
        return d


        

class GateSetMember(GateSetChild):
    """ 
    Base class for all GateSet member objects which possess a definite
    dimension, number of parmeters, and evolution type (_evotype).  A 
    GateSetMember can be vectorized into/onto a portion of their parent
    GateSet's (or other GateSetMember's) parameter vector.  They therefore
    contain a `gpindices` reference to the global GateSet indices "owned" by
    this member.  Note that GateSetMembers may contain other GateSetMembers (may
    be nested).
    """
    def __init__(self, dim, evotype, gpindices=None, parent=None):
        """ Initialize a new GateSetMember """
        self.dim = dim
        self._evotype = evotype
        self._gpindices = gpindices
        self.dirty = False # True when there's any *possibility* that this
                           # gate's parameters have been changed since the
                           # last setting of dirty=False
        super(GateSetMember,self).__init__(parent)
        
    def get_dimension(self):
        """ Return the dimension of this object. """
        return self.dim

    @property
    def gpindices(self):
        """ 
        Gets the gateset parameter indices of this object.
        """
        return self._gpindices

    @gpindices.setter
    def gpindices(self, value):
        raise ValueError(("Use set_gpindices(...) to set the gpindices member"
                          " of a GateSetMember object"))

    @property
    def parent(self):
        """ 
        Gets the parent of this object.
        """
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ValueError(("Use set_gpindices(...) to set the parent"
                          " of a GateSetMember object"))


    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this GateSetMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : GateSet or GateSetMember
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
        if memo is not None:
            if id(self) in memo: return
            memo.add(id(self))
        self._parent = parent
        self._gpindices = gpindices

    def allocate_gpindices(self, startingIndex, parent):
        """
        Sets gpindices array for this object or any objects it
        contains (i.e. depends upon).  Indices may be obtained
        from contained objects which have already been initialized
        (e.g. if a contained object is shared with other
         top-level objects), or given new indices starting with
        `startingIndex`.

        Parameters
        ----------
        startingIndex : int
            The starting index for un-allocated parameters.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        num_new: int
            The number of *new* allocated parameters (so 
            the parent should mark as allocated parameter
            indices `startingIndex` to `startingIndex + new_new`).
        """
        #DEBUG def pp(x): return id(x) if (x is not None) else x
        #DEBUG print(" >>> DB DEFAULT %d ALLOCATING: " % id(self), self.gpindices, " parents:", pp(self.parent), pp(parent))
        if self.gpindices is None or parent is not self.parent:
            #default behavior: assume num_params() works even with
            # gpindices == None and allocate all our parameters as "new"
            Np = self.num_params()
            self.set_gpindices( slice(startingIndex,
                                      startingIndex+Np), parent )
            #print(" -- allocated %d indices" % Np)
            return Np
        else: #assume gpindices is good & everything's allocated already
            #print(" -- no need to allocate anything")
            return 0

        
    def gpindices_as_array(self):
        """ 
        Returns gpindices as a `numpy.ndarray` of integers (gpindices itself
        can be None, a slice, or an integer array).  If gpindices is None, an
        empty array is returned.

        Returns
        -------
        numpy.ndarray
        """
        if self._gpindices is None:
            return _np.empty(0,_np.int64)
        elif isinstance(self._gpindices, slice):
            return _np.array(_slct.indices(self._gpindices),_np.int64)
        else:
            return self._gpindices #it's already an array

    
    def num_params(self):
        """
        Get the number of independent parameters which specify this object.
        """
        return 0 # by default, object has no parameters

    def to_vector(self):
        """
        Get this object's parameters as an array of values.
        """
        return _np.array([], 'd') # no parameters

    def from_vector(self, v):
        """
        Initialize this object using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(len(v) == 0) #should be no parameters, and nothing to do
    
    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        Gate
            A copy of this object.
        """
        # A default for derived classes - deep copy everything except the parent,
        #  which will get reset by _copy_gpindices anyway.
        memo = {id(self.parent): None} # so deepcopy uses None instead of copying parent
        return self._copy_gpindices(_copy.deepcopy(self,memo), parent)

    
    def _copy_gpindices(self, gateObj, parent):
        """ Helper function for implementing copy in derived classes """
        gpindices_copy = None
        if isinstance(self.gpindices, slice):
            gpindices_copy = self.gpindices #slices are immutable
        elif self.gpindices is not None:
            gpindices_copy = self.gpindices.copy() #arrays are not

        #Call base class implementation here because
        # "advanced" implementations containing sub-members assume that the
        # gpindices has already been set and is just being updated (so it compares
        # the "old" (existing) gpindices with the value being set).  Here,
        # we just want to copy any existing gpindices from "self" to gateObj
        # and *not* cause gateObj to shift any underlying indices (they'll
        # be copied separately. -- FUTURE: make separate "update_gpindices" and
        # "copy_gpindices" functions?
        GateSetMember.set_gpindices(gateObj, gpindices_copy, parent)
        #gateObj.set_gpindices(gpindices_copy, parent) #don't do this
        
        return gateObj


def _compose_gpindices(parent_gpindices, child_gpindices):
    """ 
    Maps `child_gpindices`, which index `parent_gpindices` into a new slice
    or array of indices that is the subset of parent indices.

    Essentially:
    `return parent_gpindices[child_gpindices]`
    """
    if parent_gpindices is None or child_gpindices is None: return None
    if isinstance(parent_gpindices, slice):
        start,stop = parent_gpindices.start, parent_gpindices.stop
        assert(parent_gpindices.step is None),"No support for nontrivial step size yet"
        
        if isinstance(child_gpindices, slice):
            return _slct.shift( child_gpindices, start )
        else: # child_gpindices is an index array
            return child_gpindices + start # numpy "shift"
        
    else: #parent_gpindices is an index array, so index with child_gpindices
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
        start,stop = parent_gpindices.start, parent_gpindices.stop
        assert(parent_gpindices.step is None),"No support for nontrivial step size yet"
        
        if isinstance(sibling_gpindices, slice):
            assert(start <= sibling_gpindices.start and sibling_gpindices.stop <= stop), \
                "Sibling indices must be a sub-slice of parent indices!"
            return _slct.shift( sibling_gpindices, -start )
        else: # child_gpindices is an index array
            return sibling_gpindices - start # numpy "shift"
        
    else: #parent_gpindices is an index array
        sibInds = _slct.indices(sibling_gpindices) \
                  if isinstance(sibling_gpindices, slice) else sibling_gpindices
        parent_lookup = { j: i for i,j in enumerate(parent_gpindices) }
        return _np.array( [ parent_lookup[j] for j in sibInds ], _np.int64)
