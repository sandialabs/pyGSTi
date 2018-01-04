"""Defines the POVM class"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
import collections as _collections
import numpy as _np
import warnings as _warnings

#from . import labeldicts as _ld
from . import gatesetmember as _gm
from . import spamvec as _sv
from ..tools import matrixtools as _mt


#Thoughts:
# what are POVM objs needed for?
# - construction of Effect vectors: allocating a pool of
#    shared parameters that multiple SPAMVecs use
#    - how should GateSet add items?
#      "default allocator" inserts new params into _paramvec when gpindices is None
#       (or is made None b/c parent is different) and sets gpindices accordingly
#      Could an alternate allocator allocate a POVM, which asks for/presensts a
#      block of indices, and after receiving this block adds effect vec to GateSet
#      which use the indices in this block? - maybe when GateSet inserts a POVM
#      it rebuilds paramvec as usual but doesn't insert it's effects into GateSet
#      (maybe not really inserting but "allocating/integrating" it - meaning it's
#       gpindices is set) until after the POVM's block of indices is allocated?
#    - maybe concept of "allocation" is a good one - meaning when an objects
#       gpindices and parent are set, and there's room in the GateSet's _paramvec
#       for the parameters.
#    - currently, a gates are "allocated" by _rebuild_paramvec when their
#       gpindices is None (if gpindices is not None, the indices can get
#       "shifted" but not "allocated" (check this!)
#    - maybe good to alert an object when it has be "allocated" to a GateSet;
#       a Gate may do nothing, but a POVM might then allocate its member effects.
#       E.G:  POVM created = creates objects all with None gpindices
#             POVM assigned to a GateSet => GateSet allocates POVM & calls POVM.allocated_callback()
#             POVM.allocated_callback() allocates (on behalf of GateSet b/c POVM owns those indices?) its member effects - maybe needs to
#               add them to GateSet.effects so they're accounted for later & calls SPAMVec.allocated_callback()
#             SPAMVec.allocated_callback() does nothing.
#    - it seems good for GateSet to keep track directly of allocated preps, gates, & effects OR else
#      it will need to alert objects when they're allocated indices shift so they can shift their member's indices... (POVM.shifted_callback())
#    - at this point, could just add set_gpindices and shift_gpindices members to GateSetMember, though not all indices necessarily shift by same amt...
# - grouping a set of effect vectors together for iterating
#    over (just holding the names seems sufficient)

# Conclusions/philosphy: 12/8/2017
# - povms and instruments will hold their members, but member SPAMVec or Gate objects
#   will have the GateSet as their parent, and have gpindices which reference the GateSet.
# - it is the parent object's (e.g. a GateSet, POVM, or Instrument) which is responsible
#   for setting the gpindices of its members.  The gpindices is set via a property or method
#   call, and parent objects will thereby set the gpindices of their contained elements.

#

def convert(povm, toType, basis):
    """
    Convert POVM to a new type of parameterization, potentially
    creating a new object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    povm : BasePOVM
        POVM to convert

    toType : {"full","TP","static"}
        The type of parameterizaton to convert to.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `povm`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    Returns
    -------
    POVM
       The converted POVM vector, usually a distinct
       object from the object passed as input.
    """
    if toType in ("full","static"):
        converted_effects = [ (lbl,_sv.convert(vec, toType, basis))
                              for lbl,vec in povm.items() ]
        return POVM(converted_effects)

    elif toType == "TP":
        if isinstance(povm, TPPOVM):
            return povm # no conversion necessary
        else:
            converted_effects = [ (lbl,_sv.convert(vec, "full", basis))
                                  for lbl,vec in povm.items() ]
            return TPPOVM(converted_effects)
    else:
        raise ValueError("Invalid toType argument: %s" % toType)



class BasePOVM(_gm.GateSetMember, _collections.OrderedDict):
    """ 
    Meant to correspond to a  positive operator-valued measure,
    in theory, this class generalizes that notion slightly to
    include a collection of effect vectors that may or may not
    have all of the properties associated by a mathematical POVM.
    """
    def __init__(self, effects, preserve_sum=False):
        """
        Creates a new BasePOVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.

        preserve_sum : bool, optional
            If true, the sum of `effects` is taken to be a constraint
            and so the final effect vector is made into a 
            :class:`ComplementSPAMVec`.
        """
        dim = None
        self.Np = 0
        self._readonly = False #until init is done
        
        if isinstance(effects,dict):
            items = [(k,v) for k,v in effects.items()] #gives definite ordering of effects
        elif isinstance(effects,list):
            items = effects # assume effects is already an ordered (key,value) list
        else:
            raise ValueError("Invalid `effects` arg of type %s" % type(effects))

        if preserve_sum:
            assert(len(items) > 1), "Cannot create a TP-POVM with < 2 effects!"
            self.complement_label = items[-1][0]
            comp_val = _np.array(items[-1][1]) # current value of complement vec
        else:
            self.complement_label = None

        #Copy each effect vector and set it's parent and gpindices.
        # Assume each given effect vector's parameters are independent.
        copied_items = []
        for k,v in items:
            if k == self.complement_label: continue
            effect = v.copy() if isinstance(v,_sv.SPAMVec) else \
                     _sv.FullyParameterizedSPAMVec(v)

            if dim is None: dim = effect.dim
            assert(dim == effect.dim),"All effect vectors must have the same dimension"

            N = effect.num_params()
            effect.set_gpindices(slice(self.Np,self.Np+N),self); self.Np += N
            copied_items.append( (k,effect) )
        items = copied_items

        #Add a complement effect if desired
        if self.complement_label is not None:  # len(items) > 0 by assert
            non_comp_effects = [v for k,v in items]
            identity_for_complement = _np.array(sum(non_comp_effects) +
                                                comp_val, 'd')
            complement_effect = _sv.ComplementSPAMVec(
                identity_for_complement, non_comp_effects)
            complement_effect.set_gpindices(slice(0,self.Np), self) #all parameters
            items.append( (self.complement_label, complement_effect) )

        _collections.OrderedDict.__init__(self, items)
        _gm.GateSetMember.__init__(self, dim)
        self._readonly = True
        assert(self.dim == dim)

        
    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter POVM elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)

        
    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        effects = [ (lbl,effect) for lbl,effect in self.items()
                    if lbl != self.complement_label ]
        
        if self.complement_label is not None:
            #add complement effect as a std numpy array - it will get
            # re-created correctly by __init__ w/preserve_sum == True
            effects.append( (self.complement_label,
                             _np.array(self[self.complement_label])) )
            preserve_sum = True
        else:
            preserve_sum = False
            
        return (BasePOVM, (effects, preserve_sum),
                {'_gpindices': self._gpindices} ) #preserve gpindices (but not parent)

    def __pygsti_reduce__(self):
        return self.__reduce__()


    def compile_effects(self, prefix=""):
        """
        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `GateSet` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `GateSet`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the compiled gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        if prefix: prefix += "_"
        compiled = _collections.OrderedDict()
        for lbl,effect in self.items():
            if lbl == self.complement_label: continue
            compiled[prefix+lbl] = effect.copy()
            compiled[prefix+lbl].set_gpindices(
                _gm._compose_gpindices(self.gpindices, effect.gpindices),
                self.parent )
            
        if self.complement_label:
            lbl = self.complement_label
            compiled[prefix+lbl] = _sv.ComplementSPAMVec(
                self[lbl].identity, [v for k,v in compiled.items()])
            self._copy_gpindices(compiled[prefix+lbl], self.parent) #set gpindices
              # of complement vector to the same as POVM (it uses *all* params)
        return compiled
    

    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.Np


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params(),'d')
        for lbl,effect in self.items():
            if lbl == self.complement_label: continue
            v[effect.gpindices] = effect.to_vector()
        return v


    def from_vector(self, v):
        for lbl,effect in self.items():
            if lbl == self.complement_label: continue
            effect.from_vector( v[effect.gpindices] )
        if self.complement_label: #re-init Ec
            self[self.complement_label]._construct_vector()

    def transform(self, S):
        """
        Update each POVM effect E as S^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * S`.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.            
        """
        for lbl,effect in self.items():
            if lbl == self.complement_label: continue
            effect.transform(S,'effect')

        if self.complement_label:
            #Other effects being transformed transforms the complement,
            # so just check that the transform preserves the identity.
            TOL = 1e-6
            identityVec = _np.array(self[self.complement_label].identity)
            SmxT = _np.transpose(S.get_transform_matrix())
            assert(_np.linalg.norm(identityVec-_np.dot(SmxT,identityVec))<TOL),\
                ("Cannot transform complement effect in a way that doesn't"
                 " preservee the identity!")
            self[self.complement_label]._construct_vector()
            
        self.dirty = True


    def depolarize(self, amount):
        """
        Depolarize this POVM by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the 
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        for lbl,effect in self.items():
            if lbl == self.complement_label:
                #Don't depolarize complements since this will depol the
                # other effects via their shared params - cleanup will update
                # any complement vectors
                continue
            effect.depolarize(amount)

        if self.complement_label:
            # depolarization of other effects "depolarizes" the complement
            #self[self.complement_label].depolarize(amount) # I don't think this is desired - still want probs to sum to 1!
            self[self.complement_label]._construct_vector()
        self.dirty = True


    def num_elements(self):
        """
        Return the number of total spam vector elements in this povm.
        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the vector *elements*.

        Returns
        -------
        int
        """
        return sum([ E.size for E in self.values() ])
        
    def copy(self, parent=None):
        """
        Copy this POVM.

        Returns
        -------
        BasePOVM
            A copy of this POVM
        """
        effects = [ (k,v.copy()) for k,v in self.items() if k != self.complement_label]
        if self.complement_label is not None: #don't copy ComplementSPAMVec
            effects.append( (self.complement_label, _np.array(self[self.complement_label])) )
        return self._copy_gpindices(
            BasePOVM(effects, bool(self.complement_label is not None)), parent)

    def __str__(self):
        s = "POVM with effect vectors:\n"
        for lbl,effect in self.items():
            s += "%s:\n%s\n" % (lbl, _mt.mx_to_string(effect.base, width=4, prec=2))
        return s


class POVM(BasePOVM):
    """ 
    An unconstrained POVM that just holds a set of effect vectors,
    parameterized individually however you want.
    """
    def __init__(self, effects):
        """
        Creates a new POVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.
        """
        super(POVM,self).__init__(effects, preserve_sum=False)


    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is None)
        effects = [ (lbl,effect) for lbl,effect in self.items()]
        return (POVM, (effects,), {'_gpindices': self._gpindices} )

    def __pygsti_reduce__(self):
        return self.__reduce__()


    def copy(self, parent=None):
        """
        Copy this POVM.

        Returns
        -------
        POVM
            A copy of this POVM
        """
        effects = [ (k,v.copy()) for k,v in self.items() ]
        return self._copy_gpindices(POVM(effects), parent)



class TPPOVM(BasePOVM):
    """ 
    An POVM whose sum-of-effects is constrained to what, by definition,
    we call the "identity".
    """
    def __init__(self, effects):
        """
        Creates a new POVM object.

        Parameters
        ----------
        effects : dict of SPAMVecs or array-like
            A dict (or list of key,value pairs) of the effect vectors.  The 
            final effect vector will be stripped of any existing
            parameterization and turned into a ComplementSPAMVec which has
            no additional parameters and is always equal to 
            `identity - sum(other_effects`, where `identity` is the sum of
            `effects` when this __init__ call is made.
        """
        super(TPPOVM,self).__init__(effects, preserve_sum=True)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is not None)
        effects = [ (lbl,effect) for lbl,effect in self.items()
                    if lbl != self.complement_label ]

        #add complement effect as a std numpy array - it will get
        # re-created correctly by __init__ w/preserve_sum == True
        effects.append( (self.complement_label,
                         _np.array(self[self.complement_label])) )
            
        return (TPPOVM, (effects,), {'_gpindices': self._gpindices} )

    def __pygsti_reduce__(self):
        return self.__reduce__()

    
    def copy(self, parent=None):
        """
        Copy this POVM.

        Returns
        -------
        TPPOVM
            A copy of this POVM
        """
        assert(self.complement_label is not None)
        effects = [ (k,v.copy()) for k,v in self.items() if k != self.complement_label]
        effects.append( (self.complement_label, _np.array(self[self.complement_label])) )
        return self._copy_gpindices(TPPOVM(effects), parent)
