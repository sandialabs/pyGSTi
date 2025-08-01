"""
The ConjugatedStatePOVMEffect class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

class ConjugatedStatePOVMEffect(_POVMEffect):
    """
    This class is the common base class for parameterizations of an effect vector
    that have a dense representation.



    Attributes
    ----------
    _base_1d : numpy.ndarray
        Direct access to the underlying 1D array.

    base : numpy.ndarray
        Direct access the the underlying data as column vector,
        i.e, a (dim,1)-shaped array.
    """

    def __init__(self, state, called_from_reduce=False):
        """
        Parameters
        ----------
        vec : numpy.ndarray
            The POVM effect vector as a dense numpy array.

        evotype : {"statevec", "densitymx"}
            The evolution type.
        
        called_from_reduce : bool, optional (default False)
            Special flag used when pickling. Users should not need to
            interact with this flag directly.
        """

        self.state = state
        evotype = state._evotype
        rep = evotype.create_conjugatedstate_effect_rep(state._rep)
        _POVMEffect.__init__(self, rep, evotype)
        if not called_from_reduce:
            self.init_gpindices()  # initialize our gpindices based on sub-members
        else:
            self.allocate_gpindices(10000, None, submembers_already_allocated=True)

    @property
    def _basis(self):
        # UNSPECIFIED BASIS -- rename this as needed when setting up std rep attribute
        return self.state._basis  # try to access contained state's basis

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
    def hilbert_schmidt_size(self):
        return self.state.hilbert_schmidt_size

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
            output of :meth:`Polynomial.compact`.
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

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        # This method is meant to function for derived classes whose __init__
        # methods just construct a specific type of state and pass this to
        # ConjugatedStatePOVMEffect.__init__ (and don't add any other attributes).
        # This includes FullPOVMEffect, FullPOVMPureEffect, etc.  As such, we need
        # to construct the object is a more complex way:
        ret = cls.__new__(cls)  # create a new object of the correct type
        ConjugatedStatePOVMEffect.__init__(ret, serial_memo[mm_dict['submembers'][0]])  # init via this class's __init__
        return ret
