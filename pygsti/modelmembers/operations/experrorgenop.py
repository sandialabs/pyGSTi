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

import numpy as _np
import scipy.sparse as _sps
import scipy.linalg as _spl
import scipy.sparse.linalg as _spsl
import warnings as _warnings
from .linearop import LinearOperator as _LinearOperator
from .denseop import DenseOperatorInterface as _DenseOperatorInterface

from .. import modelmember as _modelmember
from ..errorgencontainer import ErrorGeneratorContainer as _ErrorGeneratorContainer
from ...tools import optools as _ot
from ...tools import jamiolkowski as _jt
from ...tools import matrixtools as _mt
from ...objects.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from ...objects import term as _term
from ...objects.polynomial import Polynomial as _Polynomial
from ...objects import gaugegroup as _gaugegroup
IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero
MAX_EXPONENT = _np.log(_np.finfo('d').max) - 10.0  # so that exp(.) doesn't overflow


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

    dense_rep : bool, optional
        Whether to internally implement this operation as a dense matrix.
        If `True` the error generator is rendered as a dense matrix and
        exponentiation is "exact".  If `False`, then this operation
        implements exponentiation in an approximate way that treats the
        error generator as a sparse matrix and only uses its action (and
        its adjoint's action) on a state.  Setting `dense_rep=False` is
        typically more efficient when `errorgen` has a large dimension,
        say greater than 100.
    """

    def __init__(self, errorgen, dense_rep=False):
        # Extract superop dimension from 'errorgen'
        state_space = errorgen.state_space
        self.errorgen = errorgen  # don't copy (allow object reuse)

        evotype = self.errorgen._evotype

        #TODO REMOVE
        #if evotype in ("svterm", "cterm"):
        #    dense_rep = True  # we need *dense* unitary postfactors for the term-based processing below
        #
        ## make unitary postfactor sparse when dense_rep == False and vice versa.
        ## (This doens't have to be the case, but we link these two "sparseness" notions:
        ##  when we perform matrix exponentiation in a "sparse" way we assume the matrices
        ##  are large and so the unitary postfactor (if present) should be sparse).
        ## FUTURE: warn if there is a sparsity mismatch btwn basis and postfactor?
        #if unitary_postfactor is not None:
        #    if self.dense_rep and _sps.issparse(unitary_postfactor):
        #        unitary_postfactor = unitary_postfactor.toarray()  # sparse -> dense
        #    elif not self.dense_rep and not _sps.issparse(unitary_postfactor):
        #        unitary_postfactor = _sps.csr_matrix(_np.asarray(unitary_postfactor))  # dense -> sparse

        #Finish initialization based on evolution type
        self.dense_rep = dense_rep
        if self.dense_rep:
            rep = evotype.create_dense_superop_rep(None, state_space)

            # Cache values - for later work with dense rep
            self.exp_err_gen = None   # used for dense_rep=True mode to cache qty needed in deriv_wrt_params
            self.base_deriv = None
            self.base_hessian = None
        else:
            # "sparse mode" => don't ever compute matrix-exponential explicitly
            rep = evotype.create_experrorgen_rep(self.errorgen._rep)

        # Caches in case terms are used
        self.terms = {}
        self.exp_terms_cache = {}  # used for repeated calls to the exponentiate_terms function
        self.local_term_poly_coeffs = {}
        # TODO REMOVE self.direct_terms = {}
        # TODO REMOVE self.direct_term_poly_coeffs = {}

        _LinearOperator.__init__(self, rep, evotype)
        _ErrorGeneratorContainer.__init__(self, self.errorgen)
        self._update_rep()  # updates self._rep
        #Done with __init__(...)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.errorgen]

    def copy(self, parent=None, memo=None):
        """
        Copy this object.

        Parameters
        ----------
        parent : Model, optional
            The parent model to set for the copy.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that error map has its
        # parent reset correctly.
        if memo is not None and id(self) in memo: return memo[id(self)]

        #TODO REMOVE
        #if self.unitary_postfactor is None:
        #    upost = None
        #elif self._evotype == "densitymx":
        #    upost = self.unitary_postfactor
        #else:
        #    #self.unitary_postfactor is actually the *unitary* not the postfactor
        #    termtype = "dense" if self._evotype == "svterm" else "clifford"
        #
        #    # automatically "up-convert" operation to CliffordOp if needed
        #    if termtype == "clifford":
        #        assert(isinstance(self.unitary_postfactor, CliffordOp))  # see __init__
        #        U = self.unitary_postfactor.unitary
        #    else: U = self.unitary_postfactor
        #    op_std = _gt.unitary_to_process_mx(U)
        #    upost = _bt.change_basis(op_std, 'std', self.errorgen.matrix_basis)

        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.errorgen.copy(parent, memo), self.dense_rep)
        return self._copy_gpindices(copyOfMe, parent, memo)

    def _update_rep(self, close=False):
        """
        Updates self._rep as needed after parameters have changed.
        """
        if self.dense_rep:
            # compute matrix-exponential explicitly
            self.exp_err_gen = _spl.expm(self.errorgen.to_dense())  # used in deriv_wrt_params

            #TODO REMOVE
            #if self.unitary_postfactor is not None:
            #    dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
            #else: dense = self.exp_err_gen
            dense = self.exp_err_gen
            self._rep.base.flags.writeable = True
            self._rep.base[:, :] = dense
            self._rep.base.flags.writeable = False
            self.base_deriv = None
            self.base_hessian = None
        elif not close:
            self._rep.errgenrep_has_changed(self.errorgen.onenorm_upperbound())


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
        #TODO REMOVE self.direct_terms = {}
        #TODO REMOVE self.direct_term_poly_coeffs = {}

    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        if self.dense_rep:
            # Then self._rep contains a dense version already
            return self._rep.base  # copy() unnecessary since we set to readonly

        else:
            # Construct a dense version from scratch (more time consuming)
            exp_errgen = _spl.expm(self.errorgen.to_dense())

            #TODO REMOVE
            #if self.unitary_postfactor is not None:
            #    if self._evotype in ("svterm", "cterm"):
            #        if self._evotype == "cterm":
            #            assert(isinstance(self.unitary_postfactor, CliffordOp))  # see __init__
            #            U = self.unitary_postfactor.unitary
            #        else: U = self.unitary_postfactor
            #        op_std = _gt.unitary_to_process_mx(U)
            #        upost = _bt.change_basis(op_std, 'std', self.errorgen.matrix_basis)
            #    else:
            #        upost = self.unitary_postfactor
            #
            #    dense = _mt.safe_dot(exp_errgen, upost)
            #else:
            dense = exp_errgen
            return dense

    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def to_sparse(self):
        """
        Return the operation as a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        _warnings.warn(("Constructing the sparse matrix of a LindbladDenseOp."
                        "  Usually this is *NOT* acutally sparse (the exponential of a"
                        " sparse matrix isn't generally sparse)!"))
        if self.dense_rep:
            return _sps.csr_matrix(self.to_dense())
        else:
            exp_err_gen = _spsl.expm(self.errorgen.to_sparse().tocsc()).tocsr()
            #TODO REMOVE
            #if self.unitary_postfactor is not None:
            #    return exp_err_gen.dot(self.unitary_postfactor)
            #else:
            return exp_err_gen

    #def torep(self):
    #    """
    #    Return a "representation" object for this operation.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        if self.sparse_expm:
    #            if self.unitary_postfactor is None:
    #                Udata = _np.empty(0, 'd')
    #                Uindices = Uindptr = _np.empty(0, _np.int64)
    #            else:
    #                assert(_sps.isspmatrix_csr(self.unitary_postfactor)), \
    #                    "Internal error! Unitary postfactor should be a *sparse* CSR matrix!"
    #                Udata = self.unitary_postfactor.data
    #                Uindptr = _np.ascontiguousarray(self.unitary_postfactor.indptr, _np.int64)
    #                Uindices = _np.ascontiguousarray(self.unitary_postfactor.indices, _np.int64)
    #
    #            mu, m_star, s, eta = self.err_gen_prep
    #            errorgen_rep = self.errorgen.torep()
    #            return replib.DMOpRepLindblad(errorgen_rep,
    #                                           mu, eta, m_star, s,
    #                                           Udata, Uindices, Uindptr) # HERE
    #        else:
    #            if self.unitary_postfactor is not None:
    #                dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
    #            else: dense = self.exp_err_gen
    #            return replib.DMOpRepDense(_np.ascontiguousarray(dense, 'd'))
    #    else:
    #        raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
    #                         (self._evotype, self.__class__.__name__))

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
        if not self.dense_rep:
            #raise NotImplementedError("deriv_wrt_params(...) can only be used when a dense representation is used!")
            _warnings.warn("Using finite differencing to compute ExpErrogenOp derivative!")
            return super(ExpErrorgenOp, self).deriv_wrt_params(wrt_filter)

        if self.base_deriv is None:
            d2 = self.dim

            #Deriv wrt hamiltonian params
            derrgen = self.errorgen.deriv_wrt_params(None)  # apply filter below; cache *full* deriv
            derrgen.shape = (d2, d2, -1)  # separate 1st d2**2 dim to (d2,d2)
            dexpL = _d_exp_x(self.errorgen.to_dense(), derrgen, self.exp_err_gen)
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
        if not self.dense_rep:
            #raise NotImplementedError("hessian_wrt_params is only implemented for *dense-rep* LindbladOps")
            _warnings.warn("Using finite differencing to compute ExpErrogenOp Hessian!")
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

            series, series2 = _d2_exp_series(self.errorgen.to_dense(), dEdp, d2Edp2)
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

    #REMOVE or revive this later - it doesn't seem like something that's really needed
    #def set_dense(self, m):
    #    """
    #    Set the dense-matrix value of this operation.
    #
    #    Attempts to modify operation parameters so that the specified raw
    #    operation matrix becomes mx.  Will raise ValueError if this operation
    #    is not possible.
    #
    #    Parameters
    #    ----------
    #    m : array_like or LinearOperator
    #        An array of shape (dim, dim) or LinearOperator representing the operation action.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #
    #    #TODO: move this function to errorgen?
    #    if not isinstance(self.errorgen, ExpErrorgenOp):
    #        raise NotImplementedError(("Can only set the value of a LindbladDenseOp that "
    #                                   "contains a single LindbladErrorgen error generator"))
    #
    #    tOp = ExpErrorgenOp.from_operation_matrix(
    #        m, self.unitary_postfactor,
    #        self.errorgen.ham_basis, self.errorgen.other_basis,
    #        self.errorgen.param_mode, self.errorgen.nonham_mode,
    #        True, self.errorgen.matrix_basis, self._evotype)
    #
    #    #Note: truncate=True to be safe
    #    self.errorgen.from_vector(tOp.errorgen.to_vector())
    #    self._update_rep()
    #    self.dirty = True

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

        #DEBUG: CHECK MAGS OF ERRGEN COEFFS  REMOVE
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
        egvec = self.errorgen.to_vector()   # HERE - pass vector in?  we need errorgen's vector (usually not in rep) to perform evaluation...
        errgen_terms = [egt.copy_with_magnitude(abs(egt.coeff.evaluate(egvec))) for egt in errgen_terms]

        #DEBUG!!!  REMOVE
        #import bpdb; bpdb.set_trace()
        #loc_terms = _term.exponentiate_terms_above_mag(errgen_terms, order, postTerm, min_term_mag=-1)
        #loc_terms_chk = _term.exponentiate_terms(errgen_terms, order, postTerm)
        #assert(len(loc_terms) == len(loc_terms2))
        #poly_coeffs = [t.coeff for t in loc_terms_chk]
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #v = self.to_vector()
        #coeffs = _bulk_eval_compact_polynomials_complex(
        #    vtape, ctape, v, (len(loc_terms_chk),))  # an array of coeffs
        #for coeff, t, t2 in zip(coeffs, loc_terms, loc_terms_chk):
        #    coeff2 = t.coeff.evaluate(v)
        #    if not _np.isclose(coeff,coeff2):
        #        assert(False), "STOP"
        #    t.set_magnitude(abs(coeff))

        #for ii,t in enumerate(loc_terms):
        #    coeff1 = t.coeff.evaluate(egvec)
        #    if not _np.isclose(abs(coeff1), t.magnitude):
        #        assert(False),"STOP"
        #    #t.set_magnitude(abs(t.coeff.evaluate(egvec)))

        #FUTURE:  maybe use bulk eval of compact polys? Something like this:
        #coeffs = _bulk_eval_compact_polynomials_complex(
        #    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        #for coeff, t in zip(coeffs, terms_at_order):
        #    t.set_magnitude(abs(coeff))

        terms = []
        for term in _term.exponentiate_terms_above_mag(errgen_terms, order,
                                                       postTerm, min_term_mag=min_term_mag):
            #poly_coeff = term.coeff
            #compact_poly_coeff = poly_coeff.compact(complex_coeff_tape=True)
            term.mapvec_indices_inplace(mapvec)  # local -> global indices

            # CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs)
            # REMOVE later
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
        #TODO REMOVE:
        #print("  DB: LindbladOp.get_totat_term_magnitude: (errgen type =",self.errorgen.__class__.__name__)
        #egttm = self.errorgen.total_term_magnitude
        #print("  DB: exp(", egttm, ") = ",_np.exp(egttm))
        #return _np.exp(egttm)
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
        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse
            #assert(_np.allclose(U, _np.linalg.inv(Uinv)))

            #just conjugate postfactor and Lindbladian exponent by U:
            #REMOVE
            #if self.unitary_postfactor is not None:
            #    self.unitary_postfactor = _mt.safe_dot(Uinv, _mt.safe_dot(self.unitary_postfactor, U))
            self.errorgen.transform_inplace(s)
            self._update_rep()  # needed to rebuild exponentiated error gen
            self.dirty = True

            #CHECK WITH OLD (passes) TODO move to unit tests?
            #tMx = _np.dot(Uinv,_np.dot(self.base, U)) #Move above for checking
            #tOp = LindbladDenseOp(tMx,self.unitary_postfactor,
            #                                self.ham_basis, self.other_basis,
            #                                self.cptp,self.nonham_diagonal_only,
            #                                True, self.matrix_basis)
            #assert(_np.linalg.norm(tOp.paramvals - self.paramvals) < 1e-6)
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(s)))

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

        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse

            #Note: this code may need to be tweaked to work with sparse matrices
            if typ == "prep":
                tMx = _mt.safe_dot(Uinv, self.to_dense())
            else:
                tMx = _mt.safe_dot(self.to_dense(), U)
            trunc = bool(isinstance(s, _gaugegroup.UnitaryGaugeGroupElement))
            #TODO: update this -- need a way to initialize a exp(errorgen) from a matrix... 
            tOp = ExpErrorgenOp.from_operation_matrix(tMx, self.unitary_postfactor,
                                                      self.errorgen.ham_basis, self.errorgen.other_basis,
                                                      self.errorgen.param_mode, self.errorgen.nonham_mode,
                                                      trunc, self.errorgen.matrix_basis)
            self.from_vector(tOp.to_vector())
            #Note: truncate=True above for unitary transformations because
            # while this trunctation should never be necessary (unitaries map CPTP -> CPTP)
            # sometimes a unitary transform can modify eigenvalues to be negative beyond
            # the tight tolerances checked when truncate == False. Maybe we should be able
            # to give a tolerance as `truncate` in the future?

            #NOTE: This *doesn't* work as it does in the 'operation' case b/c this isn't a
            # similarity transformation!
            ##just act on postfactor and Lindbladian exponent:
            #if typ == "prep":
            #    if self.unitary_postfactor is not None:
            #        self.unitary_postfactor = _mt.safe_dot(Uinv, self.unitary_postfactor)
            #else:
            #    if self.unitary_postfactor is not None:
            #        self.unitary_postfactor = _mt.safe_dot(self.unitary_postfactor, U)
            #
            #self.errorgen.spam_transform(s, typ)
            #self._update_rep()  # needed to rebuild exponentiated error gen
            #self.dirty = True
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(s)))

    def __str__(self):
        s = "Lindblad Parameterized operation map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        return s


class ExpErrorgenDenseOp(ExpErrorgenOp, _DenseOperatorInterface):
    """
    TODO: update docstring
    An operation matrix that is parameterized by a Lindblad-form expression.

    Specifically, each parameter multiplies a particular term in the Lindblad
    form that is exponentiated to give the operation matrix up to an optional
    unitary prefactor).  The basis used by the Lindblad form is referred to as
    the "projection basis".

    Parameters
    ----------
    errorgen : LinearOperator
        The error generator for this operator.  That is, the `L` if this
        operator is `exp(L)*unitary_postfactor`.

    dense_rep : bool, optional
        Whether the internal representation is dense.  This currently *must*
        be set to `True`.
    """

    def __init__(self, errorgen, dense_rep=True):
        assert(dense_rep), "LindbladDenseOp must be created with `dense_rep == True`"
        #Note: cannot remove the evotype argument b/c we need to maintain the same __init__
        # signature as LindbladOp so its @classmethods will work on us.

        #Start with base class construction
        ExpErrorgenOp.__init__(self, errorgen, dense_rep=True)
        _DenseOperatorInterface.__init__(self)


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
