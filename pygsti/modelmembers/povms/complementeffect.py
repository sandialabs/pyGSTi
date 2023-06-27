"""
The ComplementPOVMEffect class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.modelmembers.povms.conjugatedeffect import ConjugatedStatePOVMEffect as _ConjugatedStatePOVMEffect
from pygsti.modelmembers import modelmember as _modelmember
from pygsti.modelmembers.states.fullstate import FullState as _FullState
from pygsti.modelmembers.states.state import State as _State


class ComplementPOVMEffect(_ConjugatedStatePOVMEffect):
    """
    TODO: docstring
    A POVM effect vector that ensures that all the effects of a POVM sum to the identity.

    This POVM effect vector is paramterized as `I - sum(other_spam_vecs)` where `I` is
    a (static) identity element and `other_param_vecs` is a list of other spam
    vectors in the same parent :class:`POVM`.  This only *partially* implements
    the model-member interface (some methods such as `to_vector` and `from_vector`
    will thunk down to base class versions which raise `NotImplementedError`),
    as instances are meant to be contained within a :class:`POVM` which takes
    care of vectorization.

    Parameters
    ----------
    identity : array_like or POVMEffect
        a 1D numpy array representing the static identity operation from
        which the sum of the other vectors is subtracted.

    other_spamvecs : list of POVMEffects
        A list of the "other" parameterized POVM effect vectors which are
        subtracted from `identity` to compute the final value of this
        "complement" POVM effect vector.
    """

    def __init__(self, identity, other_effects):
        evotype = other_effects[0]._evotype
        state_space = other_effects[0].state_space

        # UNSPECIFIED BASIS -- should be able to use _rep.basis without guarding once we get std attribute setup
        try:  # get a basis for the below identity state if possible, otherwise try to continue without one
            id_basis = other_effects[0]._rep.basis
        except AttributeError:
            id_basis = None

        self.identity = _FullState(_State._to_vector(identity), id_basis, evotype, state_space)
        # an effect so easy to transform or depolarize by parent POVM (need basis in FUTURE?)

        self.other_effects = other_effects
        #Note: we assume that our parent will do the following:
        # 1) set our gpindices to indicate how many parameters we have
        # 2) set the gpindices of the elements of other_spamvecs so
        #    that they index into our local parameter vector.

        _ConjugatedStatePOVMEffect.__init__(self, self.identity.copy())
        self.init_gpindices()  # initialize our gpindices based on sub-members
        self._construct_vector()  # reset's self.base

    def _construct_vector(self):
        #Note: assumes other effects are also ConjugatedStatePOVMEffect objects
        base1d = self.state._ptr
        base1d.flags.writeable = True
        base1d[:] = self.identity.to_dense() - sum([vec.to_dense() for vec in self.other_effects])
        base1d.flags.writeable = False
        self._ptr_has_changed()

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)
        mm_dict['identity_vector'] = self._encodemx(self.identity.to_dense())
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        identity = cls._decodemx(mm_dict['identity_vector'])
        other_effects = [serial_memo[i] for i in mm_dict['submembers']]
        return cls(identity, other_effects)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return (self.identity.shape == other.identity.shape
                and _np.allclose(self.identity.to_dense(), other.identity.to_dense(), rtol=rtol, atol=atol))

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        # Note: don't include [self.state] because its params aren't ComplementPOVMEffect params
        return self.other_effects

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM effect vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the POVM effect vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        raise ValueError(("ComplementPOVMEffect.to_vector() should never be called"
                          " - use TPPOVM.to_vector() instead"))

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the POVM effect vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM effect vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this POVM effect vector's current
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
        #Rely on prior .from_vector initialization of self.other_effects, so
        # we just construct our vector based on them.
        #Note: this is needed for finite-differencing in map-based calculator
        self._construct_vector()
        self.dirty = False  # dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this POVM effect vector.

        Construct a matrix whose columns are the derivatives of the POVM effect vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per POVM effect vector parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        if len(self.other_effects) == 0: return _np.zeros((self.dim, 0), 'd')  # Complement vecs assumed real
        Np = len(self.gpindices_as_array())
        neg_deriv = _np.zeros((self.dim, Np), 'd')
        for ovec in self.other_effects:
            local_inds = _modelmember._decompose_gpindices(
                self.gpindices, ovec.gpindices)
            #Note: other_vecs are not copies but other *sibling* effect vecs
            # so their gpindices index the same space as this complement vec's
            # does - so we need to "_decompose_gpindices"
            neg_deriv[:, local_inds] += ovec.deriv_wrt_params()
        derivMx = -neg_deriv

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this POVM effect vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return False
