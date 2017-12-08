"""Defines OrderedDict-derived classes used to store specific pyGSTi objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
import collections as _collections
import numpy as _np
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

    

class OrderedMemberDict(PrefixOrderedDict):
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

        typ : {"gate","spamvec"}
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
        super(OrderedMemberDict,self).__init__(prefix, items)

        #Set parent of this dict and it's elements, not that it's been initialized
        self.parent = parent # dimension == parent.dim
        if self.parent is not None:
            for el in self.values(): el.parent = self.parent
            
        

    def _check_dim(self, obj):
        if self.typ == "spamvec":
            dim = len(obj)
        if self.typ == "gate":
            if isinstance(obj, _gate.Gate):
                dim = obj.dim
            else:
                try:
                    d1 = len(obj)
                    d2 = len(obj[0]) #pylint: disable=unused-variable
                except:
                    raise ValueError("%s doesn't look like a 2D array/list" % obj)
                if any([len(row) != d1 for row in obj]):
                    raise ValueError("%s is not a *square* 2D array" % obj)
                dim = d1

        if self.typ in ('gate','spamvec'):
            if self.parent is None: return
            if self.parent.dim is None:
                self.parent._dim = dim
            elif self.parent.dim != dim:
                raise ValueError("Cannot add object with dimension" +
                                 "%d to gateset of dimension %d"
                                 % (dim,self.parent.dim))

    def __getitem__(self, key):
        #if self.parent is not None:
        #    #print("DEBUG: cleaning paramvec before getting ", key)
        #    self.parent._clean_paramvec()
        return super(OrderedMemberDict,self).__getitem__(key)


    def __setitem__(self, key, value):
        self._check_dim(value)

        if isinstance(value, _gm.GateSetMember):  #if we're given an object, just replace
            if self.parent is not None and value.parent is not self.parent:
                value = value.copy(self.parent); value.gpindices=None
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

            if self.parent is not None: obj.parent=self.parent
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


    def copy(self, parent):
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

    def __pygsti_reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v.copy(None)) for k,v in self.items()] #store items with parent=None
        return (OrderedMemberDict,
                (None, self.default_param, self._prefix, self.typ, items), None)
    
    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v.copy(None)) for k,v in self.items()] #store items with parent=None
        return (OrderedMemberDict,
                (None, self.default_param, self._prefix, self.typ, items), None)



class OrderedSPAMLabelDict(_collections.OrderedDict):
    """
    An ordered dictionary of SPAM (outcome) labels, which associates string-type
    keys with 2-tuple values of the form `(prepLabel,effectLabel)`.  It also 
    allows a special "remainder label" which should downstream generate 
    probabilities equal to `1-(probabilities_of_all_other_outcomes)`.
    """

    def __init__(self, remainderLabel, items=[]):
        """
        Creates a new OrderedSPAMLabelDict.
        Parameters
        ----------
        remainderLabel : str
            If not None, the remainder label that will signify the special
            computation of probabilities described above, and is not associated
            with a particular `(prepLabel,effectLabel)` pair.

        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """  
        #** Note: if change __init__ signature, update __reduce__ below
        self.remainderLabel = remainderLabel
        super(OrderedSPAMLabelDict,self).__init__(items)
        

    def __setitem__(self, key, val):
        if not _compat.isstr(key):
            key = str(key)
        if type(val) != tuple or len(val) != 2:
            raise KeyError("SPAM label values must be 2-tuples!")

        prepLabel, effectLabel = val
        if prepLabel == self.remainderLabel:
            if effectLabel != self.remainderLabel:
                raise ValueError("POVM label must always == %s" % self.remainderLabel +
                                 "when preparation label does")

            # This spam label generates probabilities which equal
            #  1 - sum(other spam label probs)
            _warnings.warn( "You have choosen to use a gate set in a sum-to-one"
                            + "mode in which the spam label %s " % key
                            + "will generate probabilities equal to 1.0 minus "
                            + "the sum of all the probabilities of the other "
                            + "spam labels.")

        # if inserted *value* already exists, clobber so values are unique
        iToDel = None
        for k,v in self.items():
            if val == v:
                iToDel = k; break
        if iToDel is not None:
            del self[iToDel] #can't do within loop in Python3

        #TODO: perhaps add checks that value == (prepLabel,effectLabel) labels exist
        # (would need to add a "parent" member to access the GateSet)

        super(OrderedSPAMLabelDict,self).__setitem__(key,val)

    def copy(self):
        """ Return a copy of this OrderedSPAMLabelDict. """
        return OrderedSPAMLabelDict(self.remainderLabel, [(lbl,val) for lbl,val in self.items()])


    #def __pygsti_getstate__(self):
    #    #Use '__pygsti_getstate__' instead of '__getstate__' because we
    #    # don't want this json-serializer to interfere with the '__reduce__'
    #    # function, which is needed b/c OrderedDicts use __reduce__ when pickling.
    #    d = self.__dict__.copy()
    #    d['parent'] = None #reset parent when saving
    #    return d

    def __pygsti_reduce__(self):
        items = [(k,v) for k,v in self.items()]
        return (OrderedSPAMLabelDict, (self.remainderLabel, items), None)

    def __reduce__(self):
        items = [(k,v) for k,v in self.items()]
        return (OrderedSPAMLabelDict, (self.remainderLabel, items), None)
