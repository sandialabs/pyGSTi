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
        self.parent = parent # parent GateSet used to determine how to process
                             # a Gate's gpindices when inserted into a GateSet
                             # Note that this is *not* pickled by virtue of all
                             # the Gate classes implementing a __reduce__ which
                             # sets parent==None via this constructor.

        

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
        self._gpindices = value
        self._post_set_gpindices()

    def _post_set_gpindices(self):
        """ 
        Called after self.gpindices is set, so that if this member contains
        *other* (nested) members, it can (re)set the indices of those members.
        """
        pass
    
    def gpindices_as_array(self, asarray=False):
        """ 
        Returns gpindices as a `numpy.ndarray` of integers (gpindices itself
        can be None, a slice, or an integer array).  If gpindices is None, an
        empty array is returned.

        Returns
        -------
        numpy.ndarray
        """
        if self.gpindices is None:
            return _np.empty(0,'i')
        elif isinstance(self.gpindices, slice):
            return _np.array(_slct.indices(self.gpindices),'i')
        else:
            return self.gpindices #it's already an array

    
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
        gateObj.parent = parent
        if isinstance(self.gpindices, slice):
            gateObj.gpindices = self.gpindices #slices are immutable
        elif gateObj.gpindices is not None:
            gateObj.gpindices = self.gpindices.copy() #arrays are not
        return gateObj

    def __getstate__(self):
        """ Don't pickle parent """
        d = self.__dict__.copy()
        d['parent'] = None
        return d


