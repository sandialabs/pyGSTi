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
import numbers as _numbers
import warnings as _warnings

from . import spamvec as _sv
from . import gate as _gate
from . import gatesetmember as _gm
from ..tools import compattools as _compat
from ..baseobjs import Dim as _Dim
from ..baseobjs import Label as _Label


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
        """ Assumes key is a Label object """
        if not key.has_prefix(self._prefix):
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
        
        default_param : {"TP","full",...}
            The default parameterization used when creating an
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
        self.default_param = default_param  # "TP", "full", "static", "clifford",...
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
            if self.parent._sim_type == "auto":
                self.parent.set_simtype("auto") # run deferred auto-simtype now that _dim is set
        elif self.parent.dim != dim:
            raise ValueError("Cannot add object with dimension " +
                             "%s to gateset of dimension %d"
                             % (dim,self.parent.dim))


    def _check_evotype(self, evotype):
        if self.parent is None: return
        if self.parent._evotype is None:
            self.parent._evotype = evotype
        elif self.parent._evotype != evotype:
            raise ValueError(("Cannot add an object with evolution type"
                              " '%s' to a gateset with one of '%s'") %
                             (evotype,self.parent._evotype))

    def __contains__(self, key):
        if not isinstance(key, _Label): key = _Label(key,None)
        return super(OrderedMemberDict,self).__contains__(key)

    def __getitem__(self, key):
        #if self.parent is not None:
        #    #print("DEBUG: cleaning paramvec before getting ", key)
        #    self.parent._clean_paramvec()
        if not isinstance(key, _Label): key = _Label(key,None)
        return super(OrderedMemberDict,self).__getitem__(key)
    

    def _auto_embed(self, key_label, value):
        if self.parent is not None and key_label.sslbls is not None:
            if self.typ == "gate":
                return self.parent._embedGate(key_label.sslbls, self.cast_to_obj(value))
            else:
                raise NotImplementedError("Cannot auto-embed objects other than gates yet.")
        else:
            return value

    def cast_to_obj(self, value):
        """
        Creates an object from `value` with a type given by the the
        default parameterization if it isn't a :class:`GateSetMember`.

        Parameters
        ----------
        value : object

        Returns
        -------
        object
        """
        if isinstance(value, _gm.GateSetMember): return value

        basis = self.parent.basis if self.parent else None
        obj = None; 
        if self.typ == "spamvec":
            typ = self.default_param
            rtyp = "TP" if typ in ("CPTP","H+S","S") else typ
            obj = _sv.StaticSPAMVec(value)
            obj = _sv.convert(obj, rtyp, basis)
        elif self.typ == "gate":
            obj = _gate.StaticGate(value)
            obj = _gate.convert(obj, self.default_param, basis)
        return obj


    def __setitem__(self, key, value):
        if not isinstance(key, _Label): key = _Label(key)
        value = self._auto_embed(key,value) # automatically create an embedded gate if needed
        self._check_dim(value)

        if isinstance(value, _gm.GateSetMember):  #if we're given an object, just replace
            #When self has a valid parent (as it usually does, except when first initializing)
            # we copy and reset the gpindices & parent of GateSetMember values which either:
            # 1) belong to a different parent (indices would be inapplicable if they exist)
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

            if not hasattr(value, "_evotype"): value._evotype = "densitymx" # for backward compatibility
            self._check_evotype(value._evotype)
            super(OrderedMemberDict,self).__setitem__(key, value)

        elif key in self: #if a object already exists...
            #try to set its value
            super(OrderedMemberDict,self).__getitem__(key).set_value(value)

        else:
            #otherwise, we've been given a non-GateSetMember-object that doesn't
            # exist yet, so use default creation flags to make one:
            obj = self.cast_to_obj(value)

            if obj is None:
                raise ValueError("Cannot set a value of type: ",type(value))

            self._check_evotype(obj._evotype)
            if self.parent is not None: obj.set_gpindices(None, self.parent)
            super(OrderedMemberDict,self).__setitem__(key, obj)

            
        #rebuild GateSet's parameter vector (params may need to be added)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after inserting ", key, " : ", list(self.keys()))
            self.parent._update_paramvec( super(OrderedMemberDict,self).__getitem__(key) )

    def __delitem__(self, key):
        """Implements `del self[key]`"""
        if not isinstance(key, _Label): key = _Label(key,None)
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


class StateSpaceLabels(object):
    """
    A labeling of the decomposition of a state space (Hilbert Space).

    A StateSpaceLabels object describes, using string/int labels, how an entire
    Hilbert state space is decomposed into the direct sum of terms which
    themselves are tensor products of smaller (typically qubit-sized) Hilber
    spaces.
    """

    def __init__(self, labelList, dims=None):
        """
        Creates a new StateSpaceLabels object.

        Parameters
        ----------
        labelList : str or int or iterable
            Most generally, this can be a list of tuples, where each tuple
            contains the state-space labels (which can be strings or integers)
            for a single "tensor product block" formed by taking the tensor
            product of the spaces asociated with the labels.  The full state
            space is the direct sum of all the tensor product blocks.
            E.g. `[('Q0','Q1'), ('Q2',)]`.
            
            If just an iterable of labels is given, e.g. `('Q0','Q1')`, it is
            assumed to specify the first and only tensor product block.

            If a single state space label is given, e.g. `'Q2'`, then it is
            assumed to completely specify the first and only tensor product
            block.

        dims : int or iterable, optional
            The dimension of each state space label as an integer, tuple of
            integers, or list or tuples of integers to match the structure
            of `labelList` (i.e., if `labelList=('Q0','Q1')` then `dims` should
            be a tuple of 2 integers).  Values specify state-space dimensions: 2
            for a qubit, 3 for a qutrit, etc.  If None, then the dimensions are
            inferred, if possible, from the following naming rules:
        
            - if the label starts with 'L', dim=1 (a single Level)
            - if the label starts with 'Q' OR is an int, dim=2 (a Qubit)
            - if the label starts with 'T', dim=3 (a quTrit)
        """

        #Allow initialization via another StateSpaceLabels object
        if isinstance(labelList, StateSpaceLabels):
            labelList = labelList.labels

        #Step1: convert labelList (and dims, if given) to a list of 
        # elements describing each "tensor product block" - each of
        # which is a tuple of string labels.
        def isLabel(x):
            """ Return whether x is a valid space-label """
            return _compat.isstr(x) or isinstance(x,_numbers.Integral)
        
        if isLabel(labelList):
            labelList = [ (labelList,) ]
            if dims is not None: dims = [ (dims,) ]
        else:
            #labelList must be iterable if it's not a string
            labelList = list(labelList)
                
        if len(labelList) > 0 and isLabel(labelList[0]):
            # assume we've just been give the labels for a single tensor-prod-block 
            labelList = [ labelList ]
            if dims is not None: dims = [ dims ]
            
        self.labels = tuple([ tuple(tpbLabels) for tpbLabels in labelList])

        #Type check - labels must be strings or ints
        for tpbLabels in self.labels: #loop over tensor-prod-blocks
            for lbl in tpbLabels:
                if not isLabel(lbl):
                    raise ValueError("'%s' is an invalid state-space label (must be a string or integer)" % lbl)

        # Get the dimension of each labeled space
        self.labeldims = {} 
        if dims is None: # try to determine dims from label naming conventions
            for tpbLabels in self.labels: #loop over tensor-prod-blocks
                for lbl in tpbLabels:
                    if isinstance(lbl,_numbers.Integral): d = 2 # ints = qubits
                    elif lbl.startswith('T'): d = 3 # qutrit
                    elif lbl.startswith('Q'): d = 2 # qubits
                    elif lbl.startswith('L'): d = 1 # single level
                    else: raise ValueError("Cannot determine state-space dimension from '%s'" % lbl)
                    self.labeldims[lbl] = d
        else:
            for tpbLabels,tpbDims in zip(self.labels,dims):
                for lbl,dim in zip(tpbLabels,tpbDims):
                    assert(isinstance(lbl,_numbers.Integral)), "Dimensions must be integers!"
                    self.labeldims[lbl] = dim

        # Store the starting index (within the density matrix / state vec) of
        # each tensor-product-block (TPB), and which labels belong to which TPB
        self.tpb_index = {}

        tpb_dims = []
        for iTPB, tpbLabels in enumerate(self.labels):
            tpb_dims.append(_np.product( [ self.labeldims[lbl] for lbl in tpbLabels ] ))
            self.tpb_index.update( { lbl: iTPB for lbl in tpbLabels } )

        self.dim = _Dim(tpb_dims) #Note: access tensor-prod-block dims via self.dim.blockDims

    def product_dim(self, labels):
        """
        Computes the product of the state-space dimensions associated with each
        label in `labels`.

        Parameters
        ----------
        labels : list
            A list of state space labels (strings or integers).

        Returns
        -------
        int
        """
        return int( _np.product([self.labeldims[l] for l in labels]) )

    def __str__(self):
        if len(self.labels) == 0: return "(Null state space)"
        elif len(self.labels) == 1: return str(self.labels[0])
        else: return str(self.labels)
