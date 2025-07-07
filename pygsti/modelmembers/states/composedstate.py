"""
The ComposedState class and supporting functionality.
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

from pygsti.modelmembers.states.state import State as _State
from pygsti.modelmembers.states.staticstate import StaticState as _StaticState
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.modelmembers.errorgencontainer import ErrorMapContainer as _ErrorMapContainer
from pygsti.tools import SpaceConversionType

class ComposedState(_State):  # , _ErrorMapContainer
    """
    TODO: update docstring
    A Lindblad-parameterized State (that is also expandable into terms).

    Parameters
    ----------
    pure_vec : numpy array or State
        An array or State in the *full* density-matrix space (this
        vector will have dimension 4 in the case of a single qubit) which
        represents a pure-state preparation or projection.  This is used as
        the "base" preparation or projection that is followed or preceded
        by, respectively, the parameterized Lindblad-form error generator.
        (This argument is *not* copied if it is a State.  A numpy array
        is converted to a new StaticState.)

    errormap : MapOperator
        The error generator action and parameterization, encapsulated in
        a gate object.  Usually a :class:`LindbladOp`
        or :class:`ComposedOp` object.  (This argument is *not* copied,
        to allow ComposedStates to share error generator
        parameters with other gates and spam vectors.)
    """

    def __init__(self, static_state, errormap):
        evotype = errormap._evotype
        #from .operation import LindbladOp as _LPGMap
        #assert(evotype in ("densitymx", "svterm", "cterm")), \
        #    "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        if not isinstance(static_state, _State):
            # UNSPECIFIED BASIS - change None to static_state.basis once we have a std attribute
            static_state = _StaticState(static_state, None, evotype)  # assume spamvec is just a vector

        assert(static_state._evotype == evotype), \
            "`static_state` evotype must match `errormap` ('%s' != '%s')" % (static_state._evotype, evotype)
        assert(static_state.num_params == 0), "`static_state` 'reference' must have *zero* parameters!"

        #d2 = static_state.dim
        self.state_vec = static_state
        self.error_map = errormap
        self.terms = {}
        self.local_term_poly_coeffs = {}

        #Create representation
        rep = evotype.create_composed_state_rep(self.state_vec._rep, self.error_map._rep, static_state.state_space)

        _State.__init__(self, rep, evotype)
        _ErrorMapContainer.__init__(self, self.error_map)
        self.init_gpindices()  # initialize our gpindices based on sub-members

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        error_map = serial_memo[mm_dict['submembers'][0]]
        static_state = serial_memo[mm_dict['submembers'][1]]
        return cls(static_state, error_map)

    def _update_rep(self):
        self._rep.reps_have_changed()
        #stateRep = self.state_vec._rep
        #errmapRep = self.error_map._rep
        #self._rep.copy_from(errmapRep.acton(stateRep))

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.error_map, self.state_vec]

    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that are used by this ModelMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : dict, optional
            A memo dict used to avoid circular references.

        Returns
        -------
        None
        """
        self.terms = {}  # clear terms cache since param indices have changed now
        self.local_term_poly_coeffs = {}
        super().set_gpindices(gpindices, parent, memo)

    def to_dense(self, on_space='minimal', scratch=None):
        """
        Return this state vector as a (dense) numpy array.

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
        #error map acts on dmVec
        return _np.dot(self.error_map.to_dense(on_space), self.state_vec.to_dense(on_space))

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
        if order not in self.terms:
            assert(self.gpindices is not None), "ComposedSstate must be added to a Model before use!"

            state_terms = self.state_vec.taylor_order_terms(0, max_polynomial_vars); assert(len(state_terms) == 1)
            stateTerm = state_terms[0]
            err_terms, cpolys = self.error_map.taylor_order_terms(order, max_polynomial_vars, True)
            terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's

            #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen
            self.local_term_poly_coeffs[order] = cpolys
            self.terms[order] = terms

        if return_coeff_polys:
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def taylor_order_terms_above_mag(self, order, max_polynomial_vars, min_term_mag):
        """
        Get the `order`-th order Taylor-expansion terms of this state vector that have magnitude above `min_term_mag`.

        This function constructs the terms at the given order which have a magnitude (given by
        the absolute value of their coefficient) that is greater than or equal to `min_term_mag`.
        It calls :meth:`taylor_order_terms` internally, so that all the terms at order `order`
        are typically cached for future calls.

        Parameters
        ----------
        order : int
            The order of terms to get.

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        min_term_mag : float
            the minimum term magnitude.

        Returns
        -------
        list
        """
        state_terms = self.state_vec.taylor_order_terms(0, max_polynomial_vars); assert(len(state_terms) == 1)
        stateTerm = state_terms[0]
        stateTerm = stateTerm.copy_with_magnitude(1.0)
        #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen

        err_terms = self.error_map.taylor_order_terms_above_mag(
            order, max_polynomial_vars, min_term_mag / stateTerm.magnitude)

        #This gives the appropriate logic, but *both* prep or effect results in *same* expression, so just run it:
        #if self._prep_or_effect == "prep":
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        #else:  # "effect"
        #    # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
        #    # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
        #    # too - the difference comes when this State is used as an effect rather than a prep.
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        terms = [_term.compose_terms_with_mag((stateTerm, t), stateTerm.magnitude * t.magnitude)
                 for t in err_terms]  # t ops occur *after* stateTerm's
        return terms

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this state vector's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this state vector in a Taylor series.

        Returns
        -------
        float
        """
        # return (sum of absvals of *all* term coeffs)
        return self.error_map.total_term_magnitude  # error map is only part with terms

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this state vector's terms.

        Get the derivative of the total (sum) of the magnitudes of all this
        state vector's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        return self.error_map.total_term_magnitude_deriv

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this state vector.

        Construct a matrix whose columns are the derivatives of the state vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per state vector parameter.

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
        dmVec = self.state_vec.to_dense(SpaceConversionType.Minimal)

        derrgen = self.error_map.deriv_wrt_params(wrt_filter)  # shape (dim*dim, n_params)
        derrgen.shape = (self.dim, self.dim, derrgen.shape[1])  # => (dim,dim,n_params)

        #derror map acts on dmVec
        #return _np.einsum("ijk,j->ik", derrgen, dmVec) # return shape = (dim,n_params)
        return _np.tensordot(derrgen, dmVec, (1, 0))  # return shape = (dim,n_params)

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this state vector with respect to its parameters.

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
        dmVec = self.state_vec.to_dense(SpaceConversionType.Minimal)

        herrgen = self.error_map.hessian_wrt_params(wrt_filter1, wrt_filter2)  # shape (dim*dim, nParams1, nParams2)
        herrgen.shape = (self.dim, self.dim, herrgen.shape[1], herrgen.shape[2])  # => (dim,dim,nParams1, nParams2)

        #derror map acts on dmVec
        #return _np.einsum("ijkl,j->ikl", herrgen, dmVec) # return shape = (dim,n_params)
        return _np.tensordot(herrgen, dmVec, (1, 0))  # return shape = (dim,n_params)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.error_map.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.error_map.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.error_map.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the state vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of state vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this state vector's current
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
        self.error_map.from_vector(v, close, dirty_value)
        self._update_rep()
        self.dirty = dirty_value

    def transform_inplace(self, s):
        """
        Update state (column) vector V as inv(s) * V or s^T * V for preparation or  effect state vectors, respectively.

        Note that this is equivalent to state preparation vectors getting
        mapped: `rho -> inv(s) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * s`.

        Generally, the transform function updates the *parameters* of
        the state vector such that the resulting vector is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        #Defer everything to LindbladOp's
        # `spam_tranform` function, which applies either
        # error_map -> inv(s) * error_map ("prep" case) OR
        # error_map -> error_map * s      ("effect" case)
        self.error_map.spam_transform_inplace(s, 'prep')
        self._update_rep()
        self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this state vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the State such that the resulting vector is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the spam vector. All but
            the first element of the spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        self.error_map.depolarize(amount)
        self._update_rep()

    def errorgen_coefficient_labels(self, label_type='global'):
        """
        The elementary error-generator labels corresponding to the elements of :meth:`errorgen_coefficients_array`.

        Parameters
        ----------
        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.
        
        Returns
        -------
        tuple
            A tuple of (<type>, <basisEl1> [,<basisEl2]) elements identifying the elementary error
            generators of this gate.
        """
        return self.error_map.errorgen_coefficient_labels(label_type)

    def errorgen_coefficients_array(self):
        """
        The weighted coefficients of this state prep's error generator in terms of "standard" error generators.

        Constructs a 1D array of all the coefficients returned by :meth:`errorgen_coefficients`,
        weighted so that different error generators can be weighted differently when a
        `errorgen_penalty_factor` is used in an objective function.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to the number of coefficients in the linear combination
            of standard error generators that is this state preparation's error generator.
        """
        return self.error_map.errorgen_coefficients_array()

    def errorgen_coefficients(self, return_basis=False, logscale_nonham=False, label_type='global'):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients of this state.

        Note that these are not necessarily the parameter values, as these
        coefficients are generally functions of the parameters (so as to keep
        the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool, optional
            Whether to also return a :class:`Basis` containing the elements
            with which the error generator terms were constructed.

        logscale_nonham : bool, optional
            Whether or not the non-hamiltonian error generator coefficients
            should be scaled so that the returned dict contains:
            `(1 - exp(-d^2 * coeff)) / d^2` instead of `coeff`.  This
            essentially converts the coefficient into a rate that is
            the contribution this term would have within a depolarizing
            channel where all stochastic generators had this same coefficient.
            This is the value returned by :meth:`error_rates`.

        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients.
        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `lindblad_term_dict` to basis matrices.
        """
        return self.error_map.errorgen_coefficients(return_basis, logscale_nonham, label_type)

    def set_errorgen_coefficients(self, lindblad_term_dict, action="update", logscale_nonham=False, truncate=True):
        """
        Sets the coefficients of terms in the error generator of this state.

        The dictionary `lindblad_term_dict` has tuple-keys describing the type of term and the basis
        elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are the coefficients of these error generators,
            and should be real except for the 2-basis-label case.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :meth:`errorgen_coefficients`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :meth:`set_error_rates`.

        truncate : bool, optional
            Whether to allow adjustment of the errogen coefficients in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given coefficients
            cannot be set as specified.

        Returns
        -------
        None
        """
        self.error_map.set_errorgen_coefficients(lindblad_term_dict, action, logscale_nonham, truncate)
        self._update_rep()
        self.dirty = True

    def errorgen_coefficients_array_deriv_wrt_params(self):
        """
        The jacobian of :meth:`errogen_coefficients_array` with respect to this state's parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(num_coeffs, num_params)` where `num_coeffs` is the number of
            coefficients of this operation's error generator and `num_params` is this operation's
            number of parameters.
        """
        return self.error_map.errorgen_coefficients_array_deriv_wrt_params()

    #TODO - add more errorgen coefficient related methods as in ComposedOp
