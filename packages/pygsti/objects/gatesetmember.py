""" Defines the GatesetObject class, which represents GateSet members """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from ..tools import slicetools as _slct

class GateSetChild(object):
    """
    Base class for all objects contained in a GateSet that
    hold a `parent` refernce to their parent GateSet.
    """
    def __init__(self, parent=None):
        self._parent = parent # parent GateSet used to determine how to process
                              # a Gate's gpindices when inserted into a GateSet
                              # Note that this is *not* pickled by virtue of all
                              # the Gate classes implementing a __reduce__ which
                              # sets parent==None via this constructor.

    @property
    def parent(self):
        """ Gets the parent of this object."""
        return self._parent

    @parent.setter
    def parent(self, value):
        """ Sets the parent of this object."""
        self._parent = value

        

class GateSetMember(GateSetChild):
    """ 
    Base class for all GateSet member objects which possess a definite
    dimension and number of parmeters, can be vectorized into/onto a portion of
    their paren GateSet's parameter vector.  They therefore contain a
    `gpindices` reference to the global GateSet indices "owned" by this member.
    Note that GateSetMembers may contain other GateSetMembers (may be nested).
    """
    def __init__(self, dim, gpindices=None, parent=None):
        """ Initialize a new GateSetMember """
        self.dim = dim
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


    def set_gpindices(self, gpindices, parent):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this GateSetMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
        parent : GateSet or GateSetMember

        Returns
        -------
        None
        """
        self._parent = parent
        self._gpindices = gpindices
             
    def gpindices_as_array(self, asarray=False):
        """ 
        Returns gpindices as a `numpy.ndarray` of integers (gpindices itself
        can be None, a slice, or an integer array).  If gpindices is None, an
        empty array is returned.

        Returns
        -------
        numpy.ndarray
        """
        if self._gpindices is None:
            return _np.empty(0,'i')
        elif isinstance(self._gpindices, slice):
            return _np.array(_slct.indices(self._gpindices),'i')
        else:
            return self._gpindices #it's already an array

    
    def num_params(self):
        """
        Get the number of independent parameters which specify this object.
        """
        raise NotImplementedError("num_params not implemented!")

    def to_vector(self):
        """
        Get this object's parameters as an array of values.
        """
        raise NotImplementedError("to_vector not implemented!")

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
        raise NotImplementedError("from_vector not implemented!")
    
    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        Gate
            A copy of this object.
        """
        raise NotImplementedError("copy not implemented!")

    def _copy_gpindices(self, gateObj, parent):
        """ Helper function for implementing copy in derived classes """
        gpindices_copy = None
        if isinstance(self.gpindices, slice):
            gpindices_copy = self.gpindices #slices are immutable
        elif gateObj.gpindices is not None:
            gpindices_copy = self.gpindices.copy() #arrays are not
        gateObj.set_gpindices(gpindices_copy, parent)
        return gateObj

    def __getstate__(self):
        """ Don't pickle parent """
        d = self.__dict__.copy()
        d['_parent'] = None
        return d


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
        return _np.array( [ parent_lookup[j] for j in sibInds ], 'i')
