"""Defines OrderedDict-derived classes used to store specific pyGSTi objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
import collections as _collections
import numpy as _np
import copy as _copy
import warnings as _warnings

from . import spamvec as _sv
from . import gate as _gate
from . import gatesetmember as _gm
from ..tools import compattools as _compat

class PrefixOrderedDict(_collections.OrderedDict):
    """ 
    A base class for an ordered dictionary whose keys *must* be strings
    which begin with a given prefix.
    """
    def __init__(self, prefix, items=[]):
        """ Creates a new PrefixOrderedDict whose keys must begin
            with the string `prefix`."""
        #** Note: if change __init__ signature, update __reduce__ below
        self._prefix = prefix
        super(PrefixOrderedDict,self).__init__(items)

    def __setitem__(self, key, val):
        if not key.startswith(self._prefix):
            raise KeyError("All keys must be strings, " +
                           "beginning with the prefix '%s'" % self._prefix)
        super(PrefixOrderedDict,self).__setitem__(key, val)

    #Handled by derived classes
    #def __reduce__(self):
    #    items = [(k,v) for k,v in self.iteritems()]
    #    return (PrefixOrderedDict, (self._prefix, items), None)


    """ 
    An ordered dictionary whose keys must begin with a given prefix,
    and which holds Gate objects.  This class ensures that every value is a
    :class:`Gate`-derived object by converting any non-`Gate` values into
    `Gate`s upon assignment and raising an error if this is not possible.
    """

    

class OrderedMemberDict(PrefixOrderedDict, _gm.GateSetChild):
    """ 
    An ordered dictionary whose keys must begin with a given prefix.

    This class also ensure that every value is an object of the appropriate GateSet
    member type (e.g. :class:`SPAMVec`- or :class:`Gate`-derived object) by converting any
    values into that type upon assignment and raising an error if this is not possible.
    """
    def __init__(self, parent, default_param, prefix, typ, items=[]):
        """
        Creates a new OrderedMemberDict.

        Parameters
        ----------
        parent : GateSet
            The parent gate set, needed to obtain the dimension and handle
            updates to parameters.
        
        default_param : {"TP","full"}
            The default parameterization used when creating a `SPAMVec`-derived
            object from a key assignment.

        prefix : str
            The required prefix of all keys (which must be strings).

        typ : {"gate","spamvec","povm","instrument"}
            The type of objects that this dictionary holds.  This is 
            needed for automatic object creation and for validation. 

        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        #** Note: if change __init__ signature, update __reduce__ below
        
        # Note: we *don't* want to be calling parent's "rebuild" function here,
        # when we're creating a new list, as this behavior is only intented for
        # explicit insertions.  Since calling the base class __init__ will
        # call this class's __setitem__ we set parent to None for this step.
        self.parent = None # so __init__ below doesn't call _rebuild_paramvec
        self.default_param = default_param  # "TP", "full", or "static"
        self.typ = typ

        PrefixOrderedDict.__init__(self, prefix, items)
        _gm.GateSetChild.__init__(self, parent) # set's self.parent

        #Set parent our elements, now that the list has been initialized
        # (done for un-pickling b/c reduce => __init__ is called to construct
        #  unpickled object)
        if self.parent is not None:
            for el in self.values(): el.set_gpindices(el.gpindices, self.parent)
              #sets parent and retains any existing indices in elements
            

    def _check_dim(self, obj):
        if isinstance(obj, _gm.GateSetMember):
            dim = obj.dim
        elif self.typ == "spamvec":
            dim = len(obj)
        elif self.typ == "gate":
            try:
                d1 = len(obj)
                d2 = len(obj[0]) #pylint: disable=unused-variable
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % obj)
            if any([len(row) != d1 for row in obj]):
                raise ValueError("%s is not a *square* 2D array" % obj)
            dim = d1
        else:
            raise ValueError("Cannot obtain dimension!")

        if self.parent is None: return
        if self.parent.dim is None:
            self.parent._dim = dim
        elif self.parent.dim != dim:
            raise ValueError("Cannot add object with dimension " +
                             "%s to gateset of dimension %d"
                             % (dim,self.parent.dim))

    def __getitem__(self, key):
        #if self.parent is not None:
        #    #print("DEBUG: cleaning paramvec before getting ", key)
        #    self.parent._clean_paramvec()
        return super(OrderedMemberDict,self).__getitem__(key)


    def __setitem__(self, key, value):
        self._check_dim(value)

        if isinstance(value, _gm.GateSetMember):  #if we're given an object, just replace
            #When self has a valid parent (as it usually does, except when first initializing)
            # we copy and reset the gpindices & parent of GateSetMember values which either:
            # 1) belong to a different parent (indices would be inapplicable if the exist)
            # 2) have indices but no parent (indices are inapplicable to us)
            # Note that we don't copy and reset the case when a value's parent and gpindices
            #  are both None, as gpindices==None indicates that the value may not have had
            #  its gpindices allocated yet and so *might* have "latent" gpindices that do
            #  belong to our parent (self.parent) (and copying a value will reset all
            #  the parents to None).
            wrongParent = (value.parent is not None) and (value.parent is not self.parent)
            inappInds = (value.parent is None) and (value.gpindices is not None)
            
            if self.parent is not None and (wrongParent or inappInds):
                value = value.copy()  # copy value (so we don't mess up other parent) and
                value.set_gpindices(None, self.parent) # erase gindices don't apply to us
            super(OrderedMemberDict,self).__setitem__(key, value)

        elif key in self: #if a object already exists...
            #try to set its value
            super(OrderedMemberDict,self).__getitem__(key).set_value(value)

        else:
            #otherwise, we've been given a non-GateSetMember-object that doesn't
            # exist yet, so use default creation flags to make one:
            obj = None
            if self.default_param == "TP":
                if self.typ == "spamvec": obj = _sv.TPParameterizedSPAMVec(value)
                if self.typ == "gate": obj = _gate.TPParameterizedGate(value)
            elif self.default_param == "full":
                if self.typ == "spamvec":  obj = _sv.FullyParameterizedSPAMVec(value)
                if self.typ == "gate":  obj = _gate.FullyParameterizedGate(value)
            elif self.default_param == "static":
                if self.typ == "spamvec":  obj = _sv.StaticSPAMVec(value)
                if self.typ == "gate":  obj = _gate.StaticGate(value)
            else:
                raise ValueError("Invalid default_param: %s" % self.default_param)
            
            if obj is None:
                raise ValueError("Cannot set a value of type: ",type(value))

            if self.parent is not None: obj.set_gpindices(None, self.parent)
            super(OrderedMemberDict,self).__setitem__(key, obj)

            
        #rebuild GateSet's parameter vector (params may need to be added)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after inserting ", key, " : ", list(self.keys()))
            self.parent._update_paramvec( super(OrderedMemberDict,self).__getitem__(key) )

    def __delitem__(self, key):
        """Implements `del self[key]`"""
        super(OrderedMemberDict,self).__delitem__(key)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after deleting ", key, " : ", list(self.keys()))
            self.parent._rebuild_paramvec()


    def copy(self, parent=None):
        """
        Returns a copy of this OrderedMemberDict.

        Parameters
        ----------
        parent : GateSet
            The new parent GateSet, if one exists.  Typically, when copying
            an OrderedMemberDict you want to reset the parent.

        Returns
        -------
        OrderedMemberDict
        """
        return OrderedMemberDict(parent, self.default_param,
                                 self._prefix, self.typ,
                                 [(lbl,val.copy(parent)) for lbl,val in self.items()])
    
    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        return (OrderedMemberDict,
                (None, self.default_param, self._prefix, self.typ, list(self.items())), None)

    def __pygsti_reduce__(self):
        return self.__reduce__()




class OutcomeLabelDict(_collections.OrderedDict):
    """
    An ordered dictionary of outcome labels, whose keys are tuple-valued
    outcome labels.  This class extends an ordinary OrderedDict by
    implements mapping string-values single-outcome labels to 1-tuples
    containing that label (and vice versa), allowing the use of strings
    as outcomes labels from the user's perspective.
    """

    #Whether mapping from strings to 1-tuples is performed
    _strict = False

    def __init__(self, items=[]):
        """
        Creates a new OutcomeLabelDict.

        Parameters
        ----------
        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """  
        #** Note: if change __init__ signature, update __reduce__ below
        super(OutcomeLabelDict,self).__init__(items)
        

    def __getitem__(self, key):
        if not OutcomeLabelDict._strict:
            key = (key,) if _compat.isstr(key) else tuple(key)
        return super(OutcomeLabelDict,self).__getitem__(key)
        
    def __setitem__(self, key, val):
        if not OutcomeLabelDict._strict:
            key = (key,) if _compat.isstr(key) else tuple(key)
        super(OutcomeLabelDict,self).__setitem__(key,val)

    def __contains__(self, key):
        if not OutcomeLabelDict._strict:
            key = (key,) if _compat.isstr(key) else tuple(key)
        return key in super(OutcomeLabelDict,self).keys()

    def copy(self):
        """ Return a copy of this OutcomeLabelDict. """
        return OutcomeLabelDict([(lbl,_copy.deepcopy(val))
                                 for lbl,val in self.items()])

    def __pygsti_reduce__(self):
        items = [(k,v) for k,v in self.items()]
        return (OutcomeLabelDict, (items,), None)

    def __reduce__(self):
        items = [(k,v) for k,v in self.items()]
        return (OutcomeLabelDict, (items,), None)
