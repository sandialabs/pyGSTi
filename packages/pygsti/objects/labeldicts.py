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



class OrderedSPAMVecDict(PrefixOrderedDict):
    """ 
    An ordered dictionary whose keys must begin with a given prefix.

    This class also ensure that every value is a :class:`SPAMVec`-derived object
    by converting any non-`SPAMVec` values into `SPAMVec`s upon assignment and
    raising an error if this is not possible.
    """
    def __init__(self, parent, default_param, prefix, items=[]):
        """
        Creates a new OrderedSPAMVecDict.

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

        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        #** Note: if change __init__ signature, update __reduce__ below
        
        # Note: we *don't* want to be calling parent's "rebuild" function here,
        # when we're creating a new list, as this behavior is only intented for
        # explicit insertions.  Since calling the base class __init__ will
        # call this class's __setitem__ we set parent to None for this step.
        self.parent = None # so __init__ below doesn't call _rebuild_paramvec
        self.default_param = default_param  # "TP" or "full"
        super(OrderedSPAMVecDict,self).__init__(prefix, items) #l

        #Set parent of this dict and it's elements, not that it's been initialized
        self.parent = parent # dimension == parent.dim
        if self.parent is not None:
            for el in self.values(): el.parent = self.parent
            
        

    def _check_dim(self, vec):
        if self.parent is None: return
        if self.parent.dim is None:
            self.parent._dim = len(vec) # use _dim to set
        elif self.parent.dim != len(vec):
            raise ValueError("Cannot add vector with dimension" +
                             "%d to gateset of dimension %d"
                             % (len(vec),self.parent.dim))

    def __getitem__(self, key):
        #if self.parent is not None:
        #    #print("DEBUG: cleaning paramvec before getting ", key)
        #    self.parent._clean_paramvec()
        return super(OrderedSPAMVecDict,self).__getitem__(key)


    def __setitem__(self, key, vec):
        self._check_dim(vec)

        if isinstance(vec, _sv.SPAMVec):  #if we're given a SPAMVec object...
            #just replace or create vector
            if self.parent is not None and vec.parent is not self.parent:
                vec = vec.copy(self.parent); vec.gpindices=None
            super(OrderedSPAMVecDict,self).__setitem__(key, vec)

        elif key in self: #if a SPAMVec object already exists...
            #try to set its value
            super(OrderedSPAMVecDict,self).__getitem__(key).set_vector(vec)

        else:
            #otherwise, we've been given a non-SPAMVec-object that doesn't
            # exist yet, so use default creation flags to make one:
            if self.default_param == "TP":
                vecObj = _sv.TPParameterizedSPAMVec(vec)
            elif self.default_param == "full":
                vecObj = _sv.FullyParameterizedSPAMVec(vec)
            elif self.default_param == "static":
                vecObj = _sv.StaticSPAMVec(vec)
            else:
                raise ValueError("Invalid default_param: %s" % self.default_param)

            if self.parent is not None: vecObj.parent=self.parent
            super(OrderedSPAMVecDict,self).__setitem__(key, vecObj)

        #rebuild GateSet's parameter vector (params may need to be added)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after inserting ", key, " : ", list(self.keys()))
            self.parent._update_paramvec( super(OrderedSPAMVecDict,self).__getitem__(key) )

    def __delitem__(self, key):
        """Implements `del self[key]`"""
        super(OrderedSPAMVecDict,self).__delitem__(key)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after deleting ", key, " : ", list(self.keys()))
            self.parent._rebuild_paramvec()


    def copy(self, parent):
        """
        Returns a copy of this OrderedSPAMVecDict.

        Parameters
        ----------
        parent : GateSet
            The new parent GateSet, if one exists.  Typically, when copying
            an OrderedSPAMVecDict you want to reset the parent.

        Returns
        -------
        OrderedSPAMVecDict
        """
        return OrderedSPAMVecDict(parent, self.default_param,
                           self._prefix, [(lbl,val.copy(parent)) for lbl,val in self.items()])

    #def __pygsti_getstate__(self):
    #    #Use '__pygsti_getstate__' instead of '__getstate__' because we
    #    # don't want this json-serializer to interfere with the '__reduce__'
    #    # function, which is needed b/c OrderedDicts use __reduce__ when pickling.
    #    d = self.__dict__.copy()
    #    d['parent'] = None #reset parent when saving
    #    return d

    def __pygsti_reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v.copy(None)) for k,v in self.items()] #store items with parent=None
        return (OrderedSPAMVecDict,
                (None, self.default_param, self._prefix, items), None)
    
    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v.copy(None)) for k,v in self.items()] #store items with parent=None
        return (OrderedSPAMVecDict,
                (None, self.default_param, self._prefix, items), None)





class OrderedGateDict(PrefixOrderedDict):
    """ 
    An ordered dictionary whose keys must begin with a given prefix,
    and which holds Gate objects.  This class ensures that every value is a
    :class:`Gate`-derived object by converting any non-`Gate` values into
    `Gate`s upon assignment and raising an error if this is not possible.
    """

    def __init__(self, parent, default_param, prefix, items=[]):
        """
        Creates a new OrderedGateDict.

        Parameters
        ----------
        parent : GateSet
            The parent gate set, needed to obtain the dimension.
        
        default_param : {"TP","full","static"}
            The default parameterization used when creating a `Gate`-derived
            object from a key assignment.

        prefix : str
            The required prefix of all keys (which must be strings).

        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        #** Note: if change __init__ signature, update __reduce__ below

        # Note: we *don't* want to be calling parent's "rebuild" function here,
        # when we're creating a new list, as this behavior is only intented for
        # explicit insertions.  Since calling the base class __init__ will
        # call this class's __setitem__ we set parent to None for this step.
        self.parent = None # so __init__ below doesn't call _rebuild_paramvec
        self.default_param = default_param  # "TP" or "full" or "static"
        super(OrderedGateDict,self).__init__(prefix, items)

        #Set parent of this dict and it's elements, not that it's been initialized
        self.parent = parent # dimension == parent.dim
        if self.parent is not None:
            for el in self.values(): el.parent = self.parent


    def _check_dim(self, M):
        if isinstance(M, _gate.Gate):
            gate_dim = M.dim
        else:
            try:
                d1 = len(M)
                d2 = len(M[0]) #pylint: disable=unused-variable
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % M)
            if any([len(row) != d1 for row in M]):
                raise ValueError("%s is not a *square* 2D array" % M)
            gate_dim = d1

        #Dimension check
        if self.parent is None: return
        if self.parent.dim is None:
            self.parent._dim = gate_dim #use _dim to set
        elif self.parent.dim != gate_dim:
            raise ValueError("Cannot add gate with dimension " +
                             "%d to gateset of dimension %d"
                             % (gate_dim,self.parent.dim))

    def __getitem__(self, key):
        #if self.parent is not None:
        #    #print("DEBUG: cleaning paramvec before getting ", key)
        #    self.parent._clean_paramvec()
        return super(OrderedGateDict,self).__getitem__(key)

    
    def __setitem__(self, key, M):
        self._check_dim(M)
        #if 'Gi' in self: print("__setitem__ with Gi = ",super(OrderedGateDict,self).__getitem__('Gi').gpindices)

        if isinstance(M, _gate.Gate):  #if we're given a Gate object...
            #just replace or create vector
            if self.parent is not None and M.parent is not self.parent:
                M = M.copy(self.parent); M.gpindices=None
            super(OrderedGateDict,self).__setitem__(key, M)

        elif key in self: #if a Gate object already exists...
            #try to set its value
            super(OrderedGateDict,self).__getitem__(key).set_matrix(_np.asarray(M))

        else:
            #otherwise, we've been given a non-Gate-object that doesn't
            # exist yet, so use default creation flags to make one:
            if self.default_param == "TP":
                gateObj = _gate.TPParameterizedGate(M)
            elif self.default_param == "full":
                gateObj = _gate.FullyParameterizedGate(M)
            elif self.default_param == "static":
                gateObj = _gate.StaticGate(M)
            else:
                raise ValueError("Invalid default_param: %s" %
                                 self.default_param)

            if self.parent is not None: gateObj.parent=self.parent
            super(OrderedGateDict,self).__setitem__(key, gateObj)

        #rebuild GateSet's parameter vector (params may need to be added)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after inserting ", key, " : ", list(self.keys()))
            self.parent._update_paramvec( super(OrderedGateDict,self).__getitem__(key) )
            
    def __delitem__(self, key):
        """Implements `del self[key]`"""
        super(OrderedGateDict,self).__delitem__(key)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after deleting ", key, " : ", list(self.keys()))
            self.parent._rebuild_paramvec()


    def copy(self, parent):
        """
        Returns a copy of this OrderedGateDict.

        Parameters
        ----------
        parent : GateSet, optional
            The new parent GateSet, if one exists.  Typically, when copying
            an OrderedGateDict you want to reset the parent.

        Returns
        -------
        OrderedGateDict
        """
        return OrderedGateDict(parent, self.default_param, self._prefix,
                           [(lbl,val.copy(parent)) for lbl,val in self.items()])

    
    #def __pygsti_getstate__(self):
    #    #Use '__pygsti_getstate__' instead of '__getstate__' because we
    #    # don't want this json-serializer to interfere with the '__reduce__'
    #    # function, which is needed b/c OrderedDicts use __reduce__ when pickling.
    #    d = self.__dict__.copy()
    #    d['parent'] = None #reset parent when saving
    #    return d

    def __pygsti_reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v.copy(None)) for k,v in self.items()] #store items with parent=None
        return (OrderedGateDict,
                (None, self.default_param, self._prefix, items), None)

    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v.copy(None)) for k,v in self.items()] #store items with parent=None
        return (OrderedGateDict,
                (None, self.default_param, self._prefix, items), None)




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
