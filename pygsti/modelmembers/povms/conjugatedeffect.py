"""
The ConjugatedStatePOVMEffect class and supporting functionality.
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
import copy as _copy

from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect
from pygsti.modelmembers import term as _term
from pygsti.tools import matrixtools as _mt


class DenseEffectInterface(object):
    """
    Adds a numpy-array-mimicing interface onto a POVM effect object.
    """
    # Note: this class may not really be necessary, and maybe methods should just be
    # placed within ConjugatedStatePOVMEffect?

    @property
    def _ptr(self):
        raise NotImplementedError("Derived classes must implement the _ptr property!")

    def _ptr_has_changed(self):
        """ Derived classes should override this function to handle rep updates
            when the `_ptr` property is changed. """
        pass

    @property
    def columnvec(self):
        """
        Direct access the the underlying data as column vector, i.e, a (dim,1)-shaped array.
        """
        bv = self._ptr.view()
        bv.shape = (bv.size, 1)  # 'base' is by convention a (N,1)-shaped array
        return bv

    def __copy__(self):
        # We need to implement __copy__ because we defer all non-existing
        # attributes to self.columnvec (a numpy array) which *has* a __copy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def __deepcopy__(self, memo):
        # We need to implement __deepcopy__ because we defer all non-existing
        # attributes to self.columnvec (a numpy array) which *has* a __deepcopy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, _copy.deepcopy(v, memo))
        return cpy

    #Access to underlying array
    def __getitem__(self, key):
        if not isinstance(key, (int, _np.int64)):  # don't set dirty flag if returning a single element
            self.dirty = True
        return self.columnvec.__getitem__(key)

    def __getslice__(self, i, j):
        self.dirty = True
        return self.__getitem__(slice(i, j))  # Called for A[:]

    def __setitem__(self, key, val):
        self.dirty = True
        ret = self.columnvec.__setitem__(key, val)
        self._ptr_has_changed()
        return ret

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        if '_rep' in self.__dict__:  # sometimes in loading __getattr__ gets called before the instance is loaded
            ret = getattr(self.columnvec, attr)
        else:
            raise AttributeError("No attribute:", attr)
        self.dirty = True
        return ret

    #Mimic array
    def __pos__(self): return self.columnvec
    def __neg__(self): return -self.columnvec
    def __abs__(self): return abs(self.columnvec)
    def __add__(self, x): return self.columnvec + x
    def __radd__(self, x): return x + self.columnvec
    def __sub__(self, x): return self.columnvec - x
    def __rsub__(self, x): return x - self.columnvec
    def __mul__(self, x): return self.columnvec * x
    def __rmul__(self, x): return x * self.columnvec
    def __truediv__(self, x): return self.columnvec / x
    def __rtruediv__(self, x): return x / self.columnvec
    def __floordiv__(self, x): return self.columnvec // x
    def __rfloordiv__(self, x): return x // self.columnvec
    def __pow__(self, x): return self.columnvec ** x
    def __eq__(self, x): return self.columnvec == x
    def __len__(self): return len(self.columnvec)
    def __int__(self): return int(self.columnvec)
    def __long__(self): return int(self.columnvec)
    def __float__(self): return float(self.columnvec)
    def __complex__(self): return complex(self.columnvec)


class ConjugatedStatePOVMEffect(DenseEffectInterface, _POVMEffect):
    """
    TODO: update docstring
    A POVM effect vector that behaves like a numpy array.

    This class is the common base class for parameterizations of an effect vector
    that have a dense representation and can be accessed like a numpy array.

    Parameters
    ----------
    vec : numpy.ndarray
        The POVM effect vector as a dense numpy array.

    evotype : {"statevec", "densitymx"}
        The evolution type.

    Attributes
    ----------
    _base_1d : numpy.ndarray
        Direct access to the underlying 1D array.

    base : numpy.ndarray
        Direct access the the underlying data as column vector,
        i.e, a (dim,1)-shaped array.
    """

    def __init__(self, state):
        self.state = state
        evotype = state._evotype
        rep = evotype.create_conjugatedstate_effect_rep(state._rep)
        _POVMEffect.__init__(self, rep, evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.state.parameter_labels

    def to_dense(self, on_space='minimal', scratch=None):
        """
        Return this POVM effect vector as a (dense) numpy array.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        #don't use scratch since we already have memory allocated
        return self._rep.to_dense(on_space)  # conjugate?

    @property
    def _ptr(self):
        return self.state._ptr

    def _ptr_has_changed(self):
        """ Derived classes should override this function to handle rep updates
            when the `_ptr` property is changed. """
        self.state._ptr_has_changed()

    @property
    def size(self):
        return self.state.size

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__, self.dim)
        s += _mt.mx_to_string(self.to_dense(on_space='minimal'), width=4, prec=2)
        return s

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.state]

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM effect vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.state.num_params

    def to_vector(self):
        """
        Get the POVM effect vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.state.to_vector()

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
        self.state.from_vector(v, close, dirty_value)
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this POVM effect vector.

        Construct a matrix whose columns are the derivatives of the POVM effect vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per POVM effect parameter.

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
        return self.state.deriv_wrt_params(wrt_filter)

    def has_nonzero_hessian(self):
        """
        Whether this POVM effect vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.state.has_nonzero_hessian()

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this POVM effect vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1 : list or numpy.ndarray
            List of parameter indices to take 1st derivatives with respect to.
            (None means to use all the this operation's parameters.)

        wrt_filter2 : list or numpy.ndarray
            List of parameter indices to take 2nd derivatives with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        return self.state.hessian_wrt_params(wrt_filter1, wrt_filter2)

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this state vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        State's parameters, where the polynomial's variable indices index the
        *global* parameters of the State's parent (usually a :class:`Model`)
        , not the State's local parameter array (i.e. that returned from
        `to_vector`).

        Parameters
        ----------
        order : int
            The order of terms to get.

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        return_coeff_polys : bool
            Whether a parallel list of locally-indexed (using variable indices
            corresponding to *this* object's parameters rather than its parent's)
            polynomial coefficients should be returned as well.

        Returns
        -------
        terms : list
            A list of :class:`RankOneTerm` objects.
        coefficients : list
            Only present when `return_coeff_polys == True`.
            A list of *compact* polynomial objects, meaning that each element
            is a `(vtape,ctape)` 2-tuple formed by concatenating together the
            output of :method:`Polynomial.compact`.
        """
        ret = self.state.taylor_order_terms(order, max_polynomial_vars, return_coeff_polys)
        state_terms = ret[0] if return_coeff_polys else ret

        evotype = self.evotype
        effect_terms = []
        for state_term in state_terms:
            assert(isinstance(state_term, _term.RankOnePolynomialPrepTerm))
            effect_term = _term.RankOnePolynomialEffectTerm(evotype.conjugate_state_term_rep(state_term._rep), evotype)
            effect_terms.append(effect_term)

        return (effect_terms, ret[1]) if return_coeff_polys else effect_terms

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

        #TEMPORARY HACK - for now, we manually serialize this objects submembers (just
        # the conjugated state) first, since this to_memoized_dict is called manually from
        # a POVM's to_memoized_dict.  Once POVM effects are "submembers" this should be
        # done automatically.
        from pygsti.modelmembers.modelmembergraph import MMGNode as _MMGNode
        mmg_memo[id(self.state)] = _MMGNode(self.state, {})

        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['conjugated_state'] = self.state.to_memoized_dict({})  # TEMPORARY!!!!!!!!

        return mm_dict

    @classmethod
    def from_memoized_dict(cls, mm_dict, serial_memo):
        """Deserialize a ModelMember object and relink submembers from a memo.

        Parameters
        ----------
        mm_dict: dict
            A dict representation of this ModelMember ready for deserialization
            This must have at least the following fields:
                module, class, submembers, state_space, evotype

        serial_memo: dict
            Keys are serialize_ids and values are ModelMembers. This is NOT the same as
            other memos in ModelMember, (e.g. copy(), allocate_gpindices(), etc.).
            This is similar but not the same as mmg_memo in to_memoized_dict(),
            as we do not need to build a ModelMemberGraph for deserialization.

        Returns
        -------
        ModelMember
            An initialized object
        """
        from pygsti.modelmembers.states import State as _State
        serial_memo_hack = serial_memo.copy()
        serial_memo_hack[0] = None  # HACK to behave as if the submember were loaded
        cls._check_memoized_dict(mm_dict, serial_memo_hack)

        #Special init that circumvents derived class __init__ method to set state
        # (works when derived class just holds a conjugated state and nothing else)
        ret = cls.__new__(cls)
        state_class = _State._state_class(mm_dict['conjugated_state'])
        state = state_class.from_memoized_dict(mm_dict['conjugated_state'], serial_memo)
        ConjugatedStatePOVMEffect.__init__(ret, state)  # just call our init function
        return ret
