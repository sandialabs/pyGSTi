"""
Defines the POVM class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import collections as _collections
import numpy as _np

from pygsti.modelmembers import modelmember as _mm

#Thoughts:
# what are POVM objs needed for?
# - construction of Effect vectors: allocating a pool of
#    shared parameters that multiple POVMEffects use
#    - how should Model add items?
#      "default allocator" inserts new params into _paramvec when gpindices is None
#       (or is made None b/c parent is different) and sets gpindices accordingly
#      Could an alternate allocator allocate a POVM, which asks for/presensts a
#      block of indices, and after receiving this block adds effect vec to Model
#      which use the indices in this block? - maybe when Model inserts a POVM
#      it rebuilds paramvec as usual but doesn't insert it's effects into Model
#      (maybe not really inserting but "allocating/integrating" it - meaning it's
#       gpindices is set) until after the POVM's block of indices is allocated?
#    - maybe concept of "allocation" is a good one - meaning when an objects
#       gpindices and parent are set, and there's room in the Model's _paramvec
#       for the parameters.
#    - currently, a gates are "allocated" by _rebuild_paramvec when their
#       gpindices is None (if gpindices is not None, the indices can get
#       "shifted" but not "allocated" (check this!)
#    - maybe good to alert an object when it has be "allocated" to a Model;
#       a LinearOperator may do nothing, but a POVM might then allocate its member effects.
#       E.G:  POVM created = creates objects all with None gpindices
#             POVM assigned to a Model => Model allocates POVM & calls POVM.allocated_callback()
#             POVM.allocated_callback() allocates (on behalf of Model b/c POVM owns those indices?) its member effects -
#               maybe needs to add them to Model.effects so they're accounted for later & calls
#               POVMEffect.allocated_callback()
#             POVMEffect.allocated_callback() does nothing.
#    - it seems good for Model to keep track directly of allocated preps, gates, & effects OR else it will need to alert
#      objects when they're allocated indices shift so they can shift their member's
#      indices... (POVM.shifted_callback())
#    - at this point, could just add set_gpindices and shift_gpindices members to ModelMember, though not all indices
#      necessarily shift by same amt...
# - grouping a set of effect vectors together for iterating
#    over (just holding the names seems sufficient)

# Conclusions/philosphy: 12/8/2017
# - povms and instruments will hold their members, but member POVMEffect or LinearOperator objects
#   will have the Model as their parent, and have gpindices which reference the Model.
# - it is the parent object's (e.g. a Model, POVM, or Instrument) which is responsible
#   for setting the gpindices of its members.  The gpindices is set via a property or method
#   call, and parent objects will thereby set the gpindices of their contained elements.


class POVM(_mm.ModelMember, _collections.OrderedDict):
    """
    A generalized positive operator-valued measure (POVM).

    Meant to correspond to a  positive operator-valued measure,
    in theory, this class generalizes that notion slightly to
    include a collection of effect vectors that may or may not
    have all of the properties associated by a mathematical POVM.

    Parameters
    ----------
    state_space : StateSpace
        The state space of this POVM (and of the effect vectors).

    evotype : Evotype
        The evolution type.

    items : list or dict, optional
        Initial values.  This should only be used internally in de-serialization.
    """

    def __init__(self, state_space, evotype, items=[]):
        self._readonly = False  # until init is done
        _collections.OrderedDict.__init__(self, items)
        _mm.ModelMember.__init__(self, state_space, evotype)
        self._readonly = True

    def __setitem__(self, key, value):
        if self._readonly: raise ValueError("Cannot alter POVM elements")
        else: return _collections.OrderedDict.__setitem__(self, key, value)

    def __pygsti_reduce__(self):
        return self.__reduce__()

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 0  # default == no parameters

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return _np.array([], 'd')  # no parameters

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        assert(len(v) == 0)  # should be no parameters

    def set_time(self, t):
        """
        Sets the current time for a time-dependent operator.

        For time-independent operators (the default), this function does absolutely nothing.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        None
        """
        pass

    def transform_inplace(self, s):
        """
        Update each POVM effect E as s^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        raise ValueError("Cannot transform a %s object" % self.__class__.__name__)
        #self.dirty = True

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
        raise ValueError("Cannot depolarize a %s object" % self.__class__.__name__)
        #self.dirty = True

    @property
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
        return sum([E.size for E in self.values()])

    def acton(self, state):
        """
        Compute the outcome probabilities the result from acting on `state` with this POVM.

        Parameters
        ----------
        state : State
            The state to act on

        Returns
        -------
        OrderedDict
            A dictionary whose keys are the outcome labels (strings)
            and whose values are the probabilities of seeing each outcome.
        """
        assert(self._evotype in ('densitymx', 'statevec', 'stabilizer')), \
            "probabilities(...) cannot be used with the %s evolution type!" % self._evotype
        assert(state._evotype == self._evotype), "Evolution type mismatch: %s != %s" % (self._evotype, state._evotype)

        staterep = state._rep
        outcome_probs = _collections.OrderedDict()
        for lbl, E in self.items():
            outcome_probs[lbl] = E._rep.probability(staterep)
        return outcome_probs

    def __str__(self):
        s = "%s with effect vectors:\n" % self.__class__.__name__
        for lbl, effect in self.items():
            s += "%s: %s\n" % (lbl, str(effect))
        return s
