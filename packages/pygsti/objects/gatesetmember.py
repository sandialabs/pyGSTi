""" Defines the GatesetObject class, which represents GateSet members """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from ..tools import slicetools as _slct

class GateSetMember(object):
    """ Base class for all GateSet member objects (possibly nested) """
    
    def __init__(self, dim):
        """ Initialize a new GatesetObject """
        self.dim = dim
        self.gpindices = None
        self.parent = None # parent GateSet used to determine how to process
                           # a Gate's gpindices when inserted into a GateSet
                           # Note that this is *not* pickled by virtue of all
                           # the Gate classes implementing a __reduce__ which
                           # sets parent==None via this constructor.
        self.dirty = False # True when there's any *possibility* that this
                           # gate's parameters have been changed since the
                           # last setting of dirty=False
        
    def get_dimension(self):
        """ Return the dimension of this object. """
        return self.dim
    
    def get_gpindices(self, asarray=False):
        """ 
        Returns the indices of the parent's parameters that are used
        by this object.

        Parameters
        ----------
        asarray : bool, optional
            if True, then the returned value will always be a `numpy.ndarray`
            of integers.  If False, then the raw `gpindices` member will be
            returned, which can be either an array or a slice.

        Returns
        -------
        numpy.ndarray or slice
        """
        if asarray:
            if self.gpindices is None:
                return _np.empty(0,'i')
            elif isinstance(self.gpindices, slice):
                return _np.array(_slct.indices(self.gpindices),'i')
        return self.gpindices

    
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

    def _reduce_dict(self):
        """ Helper function that returns a dict suitable for __reduce__ call """
        d = self.__dict__.copy()
        d['parent'] = None
        return d

