"""
The IdentityPlusErrorgenOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

import numpy as _np
import scipy.linalg as _spl
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladParameterization as _LindbladParameterization
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.modelmembers.errorgencontainer import ErrorGeneratorContainer as _ErrorGeneratorContainer
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.tools import matrixtools as _mt
from pygsti import SpaceT
IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero
TODENSE_TRUNCATE = 1e-11


class IdentityPlusErrorgenOp(_LinearOperator, _ErrorGeneratorContainer):
    """
    An operation consisting of the identity added to what would ordinarily be an error generator.

    This is the first-order expansion, exp(L) = 1 + L, and yields a CPTP map whenever L is
    a valid Lindbladian.

    Parameters
    ----------
    errorgen : LinearOperator
        The error generator for this operator.
    """

    def __init__(self, errorgen):
        # Extract superop dimension from 'errorgen'
        state_space = errorgen.state_space
        self.errorgen = errorgen  # don't copy (allow object reuse)

        evotype = self.errorgen._evotype

        #Create representation object
        rep_type_order = ('dense', '1+L') if evotype.prefer_dense_reps else ('1+L', 'dense')
        rep = None
        for rep_type in rep_type_order:
            try:
                if rep_type == '1+L':
                    rep = evotype.create_identitypluserrorgen_rep(self.errorgen._rep)
                    self._ident = None  # not needed
                elif rep_type == 'dense':
                    rep = evotype.create_dense_superop_rep(None, self.errorgen.matrix_basis,
                                                           state_space)  # None => init to identity
                    self._ident = _np.identity(state_space.dim, 'd')
                else:
                    assert(False), "Logic error!"

                self._rep_type = rep_type
                break

            except AttributeError:
                pass  # just go to the next rep_type

        if rep is None:
            raise ValueError("Unable to construct representation with evotype: %s" % str(evotype))

        # Caches in case terms are used
        self.terms = {}
        self.local_term_poly_coeffs = {}

        _LinearOperator.__init__(self, rep, evotype)
        _ErrorGeneratorContainer.__init__(self, self.errorgen)
        self.init_gpindices()  # initialize our gpindices based on sub-members
        self._update_rep()  # updates self._rep
        #Done with __init__(...)

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        errorgen = serial_memo[mm_dict['submembers'][0]]
        return cls(errorgen)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.errorgen]

    def _update_rep(self, close=False):
        """
        Updates self._rep as needed after parameters have changed.
        """
        if self._rep_type == 'dense':
            # compute 1 + errorgen explicitly
            self._rep.base.flags.writeable = True
            self._rep.base[:, :] = self._ident + self.errorgen.to_dense("HilbertScmidt")
            self._rep.base.flags.writeable = False
        else:
            pass  # nothing to do -- no need to even notify OpRepIdentityPlusErrorgen that errorgen has changed

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
        _modelmember.ModelMember.set_gpindices(self, gpindices, parent, memo)
        self.terms = {}  # clear terms cache since param indices have changed now
        self.local_term_poly_coeffs = {}

    def to_dense(self, on_space: SpaceT='minimal'):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        if self._rep_type == 'dense':
            # Then self._rep contains a dense version already
            return self._rep.base  # copy() unnecessary since we set to readonly

        else:
            # Construct a dense version from scratch (more time consuming)
            return _np.identity(self.state_space.dim, 'd') + self.errorgen.to_dense(on_space)
            # Note: above fails with shape mismatch if try to get dense Hilbert-space op - is this desired?

    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def to_sparse(self, on_space: SpaceT='minimal'):
        """
        Return the operation as a sparse matrix.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        #if self._rep_type == 'dense':
        return _sps.csr_matrix(self.to_dense(on_space))
        #else:
        # return _sps.identity(...) + self.errorgen.to_sparse(on_space)  # TODO: ADD sparse gymnastics

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        return self.errorgen.deriv_wrt_params(wrt_filter)

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return self.errorgen.has_nonzero_hessian()

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this operation with respect to its parameters.

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
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        return self.errorgen.hessian_wrt_params(wrt_filter1, wrt_filter2)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.errorgen.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.errorgen.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying operation parameters from this operation.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.errorgen.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the operation using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of operation parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this operation's current
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
        self.errorgen.from_vector(v, close, dirty_value)
        self._update_rep(close)
        self.dirty = dirty_value

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the operation's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the operation's parent (usually a :class:`Model`), not the
        operation's local parameter array (i.e. that returned from `to_vector`).

        Parameters
        ----------
        order : int
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

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
            self._compute_taylor_order_terms(order, max_polynomial_vars)

        if return_coeff_polys:
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_polynomial_vars):  # separated for profiling

        mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[ii] = i

        def _compose_poly_indices(terms):
            for term in terms:
                #term.map_indices_inplace(lambda x: tuple(_modelmember._compose_gpindices(
                #    self.gpindices, _np.array(x, _np.int64))))
                term.mapvec_indices_inplace(mapvec)
            return terms

        assert(self.gpindices is not None), "LindbladOp must be added to a Model before use!"
        mpv = max_polynomial_vars

        #Note: for now, *all* of an error generator's terms are considered 0-th order,
        # so the below call to taylor_order_terms just gets all of them.  In the FUTURE
        # we might want to allow a distinction among the error generator terms, in which
        # case this term-exponentiation step will need to become more complicated...
        identTerm = _term.RankOnePolynomialOpTerm.create_from(_Polynomial({(): 1.0}, mpv),
                                                              None, None, self._evotype, self.state_space)  # identity
        errorgen_terms = self.errorgen.taylor_order_terms(order, max_polynomial_vars)  # *local* poly indices
        # Note: errorgen_terms contain polynomils with *local* indices b/c it is an *errorgen* object.

        loc_terms = [identTerm] if (order == 0) else []
        loc_terms.extend(errorgen_terms)

        #TODO: make this a utility routine - used also in experrorgenop.py
        poly_coeffs = [t.coeff for t in loc_terms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)

        self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

        # only cache terms with *global* indices to avoid confusion...
        self.terms[order] = _compose_poly_indices(loc_terms)

    def taylor_order_terms_above_mag(self, order, max_polynomial_vars, min_term_mag):
        """
        Get the `order`-th order Taylor-expansion terms of this operation that have magnitude above `min_term_mag`.

        This function constructs the terms at the given order which have a magnitude (given by
        the absolute value of their coefficient) that is greater than or equal to `min_term_mag`.
        It calls :meth:`taylor_order_terms` internally, so that all the terms at order `order`
        are typically cached for future calls.

        The coefficients of these terms are typically polynomials of the operation's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the operation's parent (usually a :class:`Model`), not the
        operation's local parameter array (i.e. that returned from `to_vector`).

        Parameters
        ----------
        order : int
            The order of terms to get (and filter).

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        min_term_mag : float
            the minimum term magnitude.

        Returns
        -------
        list
            A list of :class:`Rank1Term` objects.
        """
        mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[ii] = i

        assert(self.gpindices is not None), "LindbladOp must be added to a Model before use!"
        mpv = max_polynomial_vars

        identTerm = _term.RankOnePolynomialOpTerm.create_from(_Polynomial({(): 1.0}, mpv),
                                                              None, None, self._evotype, self.state_space)  # identity
        identTerm = identTerm.copy_with_magnitude(1.0)

        errorgen_terms = self.errorgen.taylor_order_terms(order, max_polynomial_vars)  # *local* poly indices
        # Note: errorgen_terms contain polynomils with *local* indices b/c it is an *errorgen* object.

        #evaluate errgen_terms' coefficients using their local vector of parameters
        # (which happends to be the same as our paramvec in this case)
        egvec = self.errorgen.to_vector()   # we need errorgen's vector (usually not in rep) to perform evaluation
        errorgen_terms = [egt.copy_with_magnitude(abs(egt.coeff.evaluate(egvec))) for egt in errorgen_terms]

        terms = []
        for term in filter(lambda t: t.magnitude >= min_term_mag, errorgen_terms):
            term.mapvec_indices_inplace(mapvec)  # local -> global indices
            terms.append(term)

        return terms

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this operator's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this operator in a Taylor series.

        Returns
        -------
        float
        """
        return self.errorgen.total_term_magnitude

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this operator's terms.

        Computes the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        return self.errorgen.total_term_magnitude_deriv

    def set_dense(self, m):
        """
        Set the dense-matrix value of this operation.

        Attempts to modify operation parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the operation action.

        Returns
        -------
        None
        """
        mx = _LinearOperator.convert_to_matrix(m)
        mx_minus_I = mx - _np.identity(mx.shape[0], 'd')
        errgen_cls = self.errorgen.__class__

        #Note: this only really works for LindbladErrorGen objects now... make more general in FUTURE?
        truncate = TODENSE_TRUNCATE  # can't just be 'True' since we need to throw errors when appropriate
        new_errgen = errgen_cls.from_error_generator_and_blocks(
            mx_minus_I, self.errorgen.coefficient_blocks, 'auto', self.errorgen.matrix_basis,
            truncate, self.errorgen.evotype, self.errorgen.state_space)
        self.errorgen.from_vector(new_errgen.to_vector())
        self._update_rep()  # needed to rebuild exponentiated error gen
        self.dirty = True

    def transform_inplace(self, s):
        """
        Update operation matrix `O` with `inv(s) * O * s`.

        Generally, the transform function updates the *parameters* of
        the operation such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the operation parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        raise NotImplementedError("1+L ops don't gauge transform nicely!")
        #self.errorgen.transform_inplace(s)
        #self._update_rep()  # needed to rebuild exponentiated error gen
        #self.dirty = True

    def spam_transform_inplace(self, s, typ):
        """
        Update operation matrix `O` with `inv(s) * O` OR `O * s`, depending on the value of `typ`.

        This functions as `transform_inplace(...)` but is used when this
        operation is used as a part of a SPAM vector.  When `typ == "prep"`,
        the spam vector is assumed to be `rho = dot(self, <spamvec>)`,
        which transforms as `rho -> inv(s) * rho`, so `self -> inv(s) * self`.
        When `typ == "effect"`, `e.dag = dot(e.dag, self)` (note that
        `self` is NOT `self.dag` here), and `e.dag -> e.dag * s`
        so that `self -> self * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).

        Returns
        -------
        None
        """
        raise NotImplementedError("1+L ops don't gauge transform nicely!")

    def __str__(self):
        s = "Identity + errorgen operation map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        return s

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "adds Identity to"
