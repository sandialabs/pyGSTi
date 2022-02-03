"""
The ExpErrorgenOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero
MAX_EXPONENT = _np.log(_np.finfo('d').max) - 10.0  # so that exp(.) doesn't overflow
SPAM_TRANSFORM_TRUNCATE = 1e-4


class ExpErrorgenOp(_LinearOperator, _ErrorGeneratorContainer):
    """
    An operation parameterized by the coefficients of an exponentiated sum of Lindblad-like terms.
    TODO: update docstring!

    The exponentiated terms give the operation's action.

    Parameters
    ----------
    errorgen : LinearOperator
        The error generator for this operator.  That is, the `L` if this
        operator is `exp(L)`.
    """

    def __init__(self, errorgen):
        # Extract superop dimension from 'errorgen'
        state_space = errorgen.state_space
        self.errorgen = errorgen  # don't copy (allow object reuse)

        evotype = self.errorgen._evotype

        #Create representation object
        rep_type_order = ('dense', 'experrgen') if evotype.prefer_dense_reps else ('experrgen', 'dense')
        rep = None
        for rep_type in rep_type_order:
            try:
                if rep_type == 'experrgen':
                    # "sparse mode" => don't ever compute matrix-exponential explicitly
                    rep = evotype.create_experrorgen_rep(self.errorgen._rep)
                elif rep_type == 'dense':
                    rep = evotype.create_dense_superop_rep(None, state_space)

                    # Cache values - for later work with dense rep
                    self.exp_err_gen = None   # used for dense_rep=True mode to cache qty needed in deriv_wrt_params
                    self.base_deriv = None
                    self.base_hessian = None
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
        self.exp_terms_cache = {}  # used for repeated calls to the exponentiate_terms function
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
            # compute matrix-exponential explicitly
            self.exp_err_gen = _spl.expm(self.errorgen.to_dense(on_space='HilbertSchmidt'))  # used in deriv_wrt_params

            dense = self.exp_err_gen
            self._rep.base.flags.writeable = True
            self._rep.base[:, :] = dense
            self._rep.base.flags.writeable = False
            self.base_deriv = None
            self.base_hessian = None
        else:  # if not close:
            self._rep.errgenrep_has_changed(self.errorgen.onenorm_upperbound())

            #CHECK that sparsemx action is correct (DEBUG CHECK)
            #from pygsti.modelmembers.states import StaticState
            #Mdense = _spl.expm(self.errorgen.to_dense())
            #if Mdense.shape == (4,4):
            #    for i in range(4):
            #        v = _np.zeros(4); v[i] = 1.0
            #
            #        staterep = StaticState(v)._rep
            #        check_acton = self._rep.acton(staterep).data
            #
            #        #check_sparse_scipy = _spsl.expm_multiply(self.errorgen.to_sparse(), v.copy())
            #        prep = _mt.expm_multiply_prep(self.errorgen.to_sparse())
            #        check_sparse = _mt.expm_multiply_fast(prep, v)
            #        check_dense = _np.dot(Mdense, v)
            #
            #        diff = _np.linalg.norm(check_dense - check_acton)
            #        #diff2 = _np.linalg.norm(check_sparse_scipy - check_sparse)
            #        if diff > 1e-6: # or diff2 > 1e-3:
            #            print("PROBLEM (%d)!!" % i, " Expop diff = ", diff)

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
        self.exp_terms_cache = {}
        self.local_term_poly_coeffs = {}

    def to_dense(self, on_space='minimal'):
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
            return _spl.expm(self.errorgen.to_dense(on_space))

    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def to_sparse(self, on_space='minimal'):
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
        if self._rep_type == 'dense':
            return _sps.csr_matrix(self.to_dense(on_space))
        else:
            return _spsl.expm(self.errorgen.to_sparse(on_space).tocsc()).tocsr()

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
        if not self._rep_type == 'dense':
            #raise NotImplementedError("deriv_wrt_params(...) can only be used when a dense representation is used!")
            #_warnings.warn("Using finite differencing to compute ExpErrogenOp derivative!")
            return super(ExpErrorgenOp, self).deriv_wrt_params(wrt_filter)

        if self.base_deriv is None:
            d2 = self.dim

            #Deriv wrt hamiltonian params
            derrgen = self.errorgen.deriv_wrt_params(None)  # apply filter below; cache *full* deriv
            derrgen.shape = (d2, d2, -1)  # separate 1st d2**2 dim to (d2,d2)
            dexpL = _d_exp_x(self.errorgen.to_dense(on_space='minimal'), derrgen, self.exp_err_gen)
            derivMx = dexpL.reshape(d2**2, self.num_params)  # [iFlattenedOp,iParam]

            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL), \
                ("Deriv matrix has imaginary part = %s.  This can result from "
                 "evaluating a Model derivative at a 'bad' point where the "
                 "error generator is large.  This often occurs when GST's "
                 "starting Model has *no* stochastic error and all such "
                 "parameters affect error rates at 2nd order.  Try "
                 "depolarizing the seed Model.") % str(_np.linalg.norm(_np.imag(derivMx)))
            # if this fails, uncomment around "DB COMMUTANT NORM" for further debugging.
            derivMx = _np.real(derivMx)
            self.base_deriv = derivMx

            #check_deriv_wrt_params(self, derivMx, eps=1e-7)
            #fd_deriv = finite_difference_deriv_wrt_params(self, wrt_filter, eps=1e-7)
            #derivMx = fd_deriv

        if wrt_filter is None:
            return self.base_deriv.view()
            #view because later setting of .shape by caller can mess with self.base_deriv!
        else:
            return _np.take(self.base_deriv, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return True

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
        if not self._rep_type == 'dense':
            #raise NotImplementedError("hessian_wrt_params is only implemented for *dense-rep* LindbladOps")
            #_warnings.warn("Using finite differencing to compute ExpErrogenOp Hessian!")
            return super(ExpErrorgenOp, self).hessian_wrt_params(wrt_filter1, wrt_filter2)

        if self.base_hessian is None:
            d2 = self.dim
            nP = self.num_params
            hessianMx = _np.zeros((d2**2, nP, nP), 'd')

            #Deriv wrt other params
            dEdp = self.errorgen.deriv_wrt_params(None)  # filter later, cache *full*
            d2Edp2 = self.errorgen.hessian_wrt_params(None, None)  # hessian
            dEdp.shape = (d2, d2, nP)  # separate 1st d2**2 dim to (d2,d2)
            d2Edp2.shape = (d2, d2, nP, nP)  # ditto

            series, series2 = _d2_exp_series(self.errorgen.to_dense(on_space='minimal'), dEdp, d2Edp2)
            term1 = series2
            term2 = _np.einsum("ija,jkq->ikaq", series, series)
            d2expL = _np.einsum("ikaq,kj->ijaq", term1 + term2,
                                self.exp_err_gen)
            hessianMx = d2expL.reshape((d2**2, nP, nP))

            #hessian has been made so index as [iFlattenedOp,iDeriv1,iDeriv2]
            assert(_np.linalg.norm(_np.imag(hessianMx)) < IMAG_TOL)
            hessianMx = _np.real(hessianMx)  # d2O block of hessian

            self.base_hessian = hessianMx

            #TODO: check hessian with finite difference here?

        if wrt_filter1 is None:
            if wrt_filter2 is None:
                return self.base_hessian.view()
                #view because later setting of .shape by caller can mess with self.base_hessian!
            else:
                return _np.take(self.base_hessian, wrt_filter2, axis=2)
        else:
            if wrt_filter2 is None:
                return _np.take(self.base_hessian, wrt_filter1, axis=1)
            else:
                return _np.take(_np.take(self.base_hessian, wrt_filter1, axis=1),
                                wrt_filter2, axis=2)

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
            output of :method:`Polynomial.compact`.
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
        postTerm = _term.RankOnePolynomialOpTerm.create_from(_Polynomial({(): 1.0}, mpv),
                                                             None, None, self._evotype, self.state_space)  # identity
        loc_terms = _term.exponentiate_terms(self.errorgen.taylor_order_terms(0, max_polynomial_vars),
                                             order, postTerm, self.exp_terms_cache)
        #OLD: loc_terms = [ t.collapse() for t in loc_terms ] # collapse terms for speed

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
        It calls :method:`taylor_order_terms` internally, so that all the terms at order `order`
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

        postTerm = _term.RankOnePolynomialOpTerm.create_from(_Polynomial({(): 1.0}, mpv), None, None,
                                                             self._evotype, self.state_space)  # identity term
        postTerm = postTerm.copy_with_magnitude(1.0)
        #Note: for now, *all* of an error generator's terms are considered 0-th order,
        # so the below call to taylor_order_terms just gets all of them.  In the FUTURE
        # we might want to allow a distinction among the error generator terms, in which
        # case this term-exponentiation step will need to become more complicated...
        errgen_terms = self.errorgen.taylor_order_terms(0, max_polynomial_vars)

        #DEBUG CHECK MAGS OF ERRGEN COEFFS
        #poly_coeffs = [t.coeff for t in errgen_terms]
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #v = self.to_vector()
        #errgen_coeffs = _bulk_eval_compact_polynomials_complex(
        #    vtape, ctape, v, (len(errgen_terms),))  # an array of coeffs
        #for coeff, t in zip(errgen_coeffs, errgen_terms):
        #    coeff2 = t.coeff.evaluate(v)
        #    if not _np.isclose(coeff,coeff2):
        #        assert(False), "STOP"
        #    t.set_magnitude(abs(coeff))

        #evaluate errgen_terms' coefficients using their local vector of parameters
        # (which happends to be the same as our paramvec in this case)
        egvec = self.errorgen.to_vector()   # we need errorgen's vector (usually not in rep) to perform evaluation
        errgen_terms = [egt.copy_with_magnitude(abs(egt.coeff.evaluate(egvec))) for egt in errgen_terms]

        terms = []
        for term in _term.exponentiate_terms_above_mag(errgen_terms, order,
                                                       postTerm, min_term_mag=min_term_mag):
            #poly_coeff = term.coeff
            #compact_poly_coeff = poly_coeff.compact(complex_coeff_tape=True)
            term.mapvec_indices_inplace(mapvec)  # local -> global indices

            # DEBUG CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs)
            # t = term
            # vt, ct = t._rep.coeff.compact_complex()
            # coeff_array = _bulk_eval_compact_polynomials_complex(vt, ct, self.parent.to_vector(), (1,))
            # if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
            #     print(coeff_array[0], "vs.", t._rep.magnitude)
            #     import bpdb; bpdb.set_trace()
            #     c1 = _Polynomial.from_rep(t._rep.coeff)

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
        # return exp( mag of errorgen ) = exp( sum of absvals of errgen term coeffs )
        # (unitary postfactor has weight == 1.0 so doesn't enter)
        return _np.exp(min(self.errorgen.total_term_magnitude, MAX_EXPONENT))
        #return _np.exp(self.errorgen.total_term_magnitude)  # overflows sometimes

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
        return _np.exp(self.errorgen.total_term_magnitude) * self.errorgen.total_term_magnitude_deriv

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
        #assert(_np.allclose(U, _np.linalg.inv(Uinv)))
        #just conjugate postfactor and Lindbladian exponent by U:
        self.errorgen.transform_inplace(s)
        self._update_rep()  # needed to rebuild exponentiated error gen
        self.dirty = True

    def spam_transform_inplace(self, s, typ):
        """
        Update operation matrix `O` with `inv(s) * O` OR `O * s`, depending on the value of `typ`.

        This functions as `transform_inplace(...)` but is used when this
        Lindblad-parameterized operation is used as a part of a SPAM
        vector.  When `typ == "prep"`, the spam vector is assumed
        to be `rho = dot(self, <spamvec>)`, which transforms as
        `rho -> inv(s) * rho`, so `self -> inv(s) * self`. When
        `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
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
        assert(typ in ('prep', 'effect')), "Invalid `typ` argument: %s" % typ
        from pygsti.models import gaugegroup as _gaugegroup

        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) \
           or isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse
            mx = self.to_dense(on_space='minimal') if self._rep_type == 'dense' else self.to_sparse(on_space='minimal')

            #just act on postfactor and Lindbladian exponent:
            if typ == "prep":
                mx = _mt.safe_dot(Uinv, mx)
            else:
                mx = _mt.safe_dot(mx, U)

            errgen_cls = self.errorgen.__class__
            #Note: this only really works for LindbladErrorGen objects now... make more general in FUTURE?
            truncate = SPAM_TRANSFORM_TRUNCATE  # can't just be 'True' since we need to throw errors when appropriate
            #REMOVE param = _LindbladParameterization(self.errorgen.parameterization.nonham_mode,
            #REMOVE                                   self.errorgen.parameterization.param_mode,
            #REMOVE                                   len(self.errorgen.ham_basis) > 0, len(self.errorgen.other_basis) > 0)
            #REMOVE transformed_errgen = errgen_cls.from_operation_matrix(mx, param, self.errorgen.lindblad_basis,
            #REMOVE                                                       self.errorgen.matrix_basis, truncate,
            #REMOVE                                                       self.errorgen.evotype)
            transformed_errgen = errgen_cls.from_operation_matrix_and_blocks(
                mx, self.errorgen.coefficient_blocks, 'auto', self.errorgen.matrix_basis,
                truncate, self.errorgen.evotype, self.errorgen.state_space)

            self.errorgen.from_vector(transformed_errgen.to_vector())

        self._update_rep()  # needed to rebuild exponentiated error gen
        self.dirty = True

    def __str__(self):
        s = "Exponentiated operation map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        return s

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "exponentiates"


def _d_exp_series(x, dx):
    TERM_TOL = 1e-12
    tr = len(dx.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    assert((tr - 2) in (1, 2)), "Currently, dx can only have 1 or 2 derivative dimensions"
    #assert( len( (_np.isnan(dx)).nonzero()[0] ) == 0 ) # NaN debugging
    #assert( len( (_np.isnan(x)).nonzero()[0] ) == 0 ) # NaN debugging
    series = dx.copy()  # accumulates results, so *need* a separate copy
    last_commutant = term = dx; i = 2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL:  # _np.linalg.norm(term)
        if tr == 3:
            #commutant = _np.einsum("ik,kja->ija",x,last_commutant) - \
            #            _np.einsum("ika,kj->ija",last_commutant,x)
            commutant = _np.tensordot(x, last_commutant, (1, 0)) - \
                _np.transpose(_np.tensordot(last_commutant, x, (1, 0)), (0, 2, 1))
        elif tr == 4:
            #commutant = _np.einsum("ik,kjab->ijab",x,last_commutant) - \
            #        _np.einsum("ikab,kj->ijab",last_commutant,x)
            commutant = _np.tensordot(x, last_commutant, (1, 0)) - \
                _np.transpose(_np.tensordot(last_commutant, x, (1, 0)), (0, 3, 1, 2))

        term = 1 / _np.math.factorial(i) * commutant

        #Uncomment some/all of this when you suspect an overflow due to x having large norm.
        #print("DB COMMUTANT NORM = ",_np.linalg.norm(commutant)) # sometimes this increases w/iter -> divergence => NaN
        #assert(not _np.isnan(_np.linalg.norm(term))), \
        #    ("Haddamard series = NaN! Probably due to trying to differentiate "
        #     "exp(x) where x has a large norm!")

        #DEBUG
        #if not _np.isfinite(_np.linalg.norm(term)): break # DEBUG high values -> overflow for nqubit operations
        #if len( (_np.isnan(term)).nonzero()[0] ) > 0: # NaN debugging
        #    #WARNING: stopping early b/c of NaNs!!! - usually caused by infs
        #    break

        series += term  # 1/_np.math.factorial(i) * commutant
        last_commutant = commutant; i += 1
    return series


def _d2_exp_series(x, dx, d2x):
    TERM_TOL = 1e-12
    tr = len(dx.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    tr2 = len(d2x.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    assert((tr - 2, tr2 - 2) in [(1, 2), (2, 4)]), "Current support for only 1 or 2 derivative dimensions"

    series = dx.copy()  # accumulates results, so *need* a separate copy
    series2 = d2x.copy()  # accumulates results, so *need* a separate copy
    term = last_commutant = dx
    last_commutant2 = term2 = d2x
    i = 2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL or _np.amax(_np.abs(term2)) > TERM_TOL:
        if tr == 3:
            commutant = _np.einsum("ik,kja->ija", x, last_commutant) - \
                _np.einsum("ika,kj->ija", last_commutant, x)
            commutant2A = _np.einsum("ikq,kja->ijaq", dx, last_commutant) - \
                _np.einsum("ika,kjq->ijaq", last_commutant, dx)
            commutant2B = _np.einsum("ik,kjaq->ijaq", x, last_commutant2) - \
                _np.einsum("ikaq,kj->ijaq", last_commutant2, x)

        elif tr == 4:
            commutant = _np.einsum("ik,kjab->ijab", x, last_commutant) - \
                _np.einsum("ikab,kj->ijab", last_commutant, x)
            commutant2A = _np.einsum("ikqr,kjab->ijabqr", dx, last_commutant) - \
                _np.einsum("ikab,kjqr->ijabqr", last_commutant, dx)
            commutant2B = _np.einsum("ik,kjabqr->ijabqr", x, last_commutant2) - \
                _np.einsum("ikabqr,kj->ijabqr", last_commutant2, x)

        term = 1 / _np.math.factorial(i) * commutant
        term2 = 1 / _np.math.factorial(i) * (commutant2A + commutant2B)
        series += term
        series2 += term2
        last_commutant = commutant
        last_commutant2 = (commutant2A + commutant2B)
        i += 1
    return series, series2


def _d_exp_x(x, dx, exp_x=None):
    """
    Computes the derivative of the exponential of x(t) using
    the Haddamard lemma series expansion.

    Parameters
    ----------
    x : ndarray
        The 2-tensor being exponentiated

    dx : ndarray
        The derivative of x; can be either a 3- or 4-tensor where the
        3rd+ dimensions are for (multi-)indexing the parameters which
        are differentiated w.r.t.  For example, in the simplest case
        dx is a 3-tensor s.t. dx[i,j,p] == d(x[i,j])/dp.

    exp_x : ndarray, optional
        The value of `exp(x)`, which can be specified in order to save
        a call to `scipy.linalg.expm`.  If None, then the value is
        computed internally.

    Returns
    -------
    ndarray
        The derivative of `exp(x)` given as a tensor with the
        same shape and axes as `dx`.
    """
    tr = len(dx.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    assert((tr - 2) in (1, 2)), "Currently, dx can only have 1 or 2 derivative dimensions"

    series = _d_exp_series(x, dx)
    if exp_x is None: exp_x = _spl.expm(x)

    if tr == 3:
        #dExpX = _np.einsum('ika,kj->ija', series, exp_x)
        dExpX = _np.transpose(_np.tensordot(series, exp_x, (1, 0)), (0, 2, 1))
    elif tr == 4:
        #dExpX = _np.einsum('ikab,kj->ijab', series, exp_x)
        dExpX = _np.transpose(_np.tensordot(series, exp_x, (1, 0)), (0, 3, 1, 2))

    return dExpX
