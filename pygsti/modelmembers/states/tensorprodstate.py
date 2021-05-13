import numpy as _np
import itertools as _itertools
import functools as _functools

from .state import State as _State
from .. import modelmember as _modelmember
from ...objects import term as _term
from ...tools import matrixtools as _mt
from ...tools import listtools as _lt


class TensorProductState(_State):
    """
    A state vector that is a tensor-product of other state vectors.

    Parameters
    ----------
    factors : list of States
        a list of the component states to take the tensor product of.
    """

    def __init__(self, factors):
        assert(len(factors) > 0), "Must have at least one factor!"

        self.factors = factors  # do *not* copy - needs to reference common objects
        self.Np = sum([fct.num_params for fct in factors])

        evotype = self.factors[0]._evotype
        rep = evotype.create_tensorproduct_state_rep(factors)

        _State.__init__(self, rep, evotype)
        self._update_rep()  # initializes rep data
        #sets gpindices, so do before stuff below

        # don't init our own gpindices (prep case), since our parent
        # is likely to be a Model and it will init them correctly.
        #But do set the indices of self.factors, since they're now
        # considered "owned" by this product-prep-vec (different from
        # the "effect" case when the factors are shared).
        off = 0
        for fct in factors:
            assert(isinstance(fct, _State)), "Factors must be State objects!"
            N = fct.num_params
            fct.set_gpindices(slice(off, off + N), self); off += N
        assert(off == self.Np)

    #REMOVE
    #def _fill_fast_kron(self):
    #    """ Fills in self._fast_kron_array based on current self.factors """
    #    if self._prep_or_effect == "prep":
    #        for i, factor_dim in enumerate(self._fast_kron_factordims):
    #            self._fast_kron_array[i][0:factor_dim] = self.factors[i].to_dense()
    #    else:
    #        factorPOVMs = self.factors
    #        for i, (factor_dim, Elbl) in enumerate(zip(self._fast_kron_factordims, self.effectLbls)):
    #            self._fast_kron_array[i][0:factor_dim] = factorPOVMs[i][Elbl].to_dense()

    def _update_rep(self):
        self._rep.reps_have_changed()
    #    if self._evotype in ("statevec", "densitymx"):
    #        if self._prep_or_effect == "prep":
    #            self._rep.base[:] = self.to_dense()
    #        else:
    #            self._fill_fast_kron()  # updates effect reps
    #    elif self._evotype == "stabilizer":
    #        if self._prep_or_effect == "prep":
    #            #we need to update self._rep, which is a SBStateRep object.  For now, we
    #            # kinda punt and just create a new rep and copy its data over to the existing
    #            # rep instead of having some kind of update method within SBStateRep...
    #            # (TODO FUTURE - at least a .copy_from method?)
    #            sframe_factors = [f.to_dense() for f in self.factors]  # StabilizerFrame for each factor
    #            new_rep = _stabilizer.sframe_kronecker(sframe_factors).to_rep()
    #            self._rep.smatrix[:, :] = new_rep.smatrix[:, :]
    #            self._rep.pvectors[:, :] = new_rep.pvectors[:, :]
    #            self._rep.amps[:, :] = new_rep.amps[:, :]
    #        else:
    #            pass  # I think the below (e.g. 'outcomes') is not altered by any parameters
    #            #factor_povms = self.factors
    #            #factorVecs = [factor_povms[i][self.effectLbls[i]] for i in range(1, len(factor_povms))]
    #            #outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in factorVecs])), _np.int64)
    #            #rep = replib.SBEffectRep(outcomes)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        vl = _np.empty(self.Np, dtype=object); off = 0
        for fct in self.factors:
            N = fct.num_params
            vl[off:off + N] = fct.parameter_labels; off += N
        return vl

    def to_dense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        return self._rep.to_dense()

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
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
        from .operation import EmbeddedOp as _EmbeddedGateMap
        terms = []
        fnq = [int(round(_np.log2(f.dim))) // 2 for f in self.factors]  # num of qubits per factor
        # assumes density matrix evolution
        total_nQ = sum(fnq)  # total number of qubits

        for p in _lt.partition_into(order, len(self.factors)):
            factor_lists = [self.factors[i].taylor_order_terms(pi, max_polynomial_vars) for i, pi in enumerate(p)]

            # When possible, create COLLAPSED factor_lists so each factor has just a single
            # (SPAMVec) pre & post op, which can be formed into the new terms'
            # TensorProdSPAMVec ops.
            # - DON'T collapse stabilizer states & clifford ops - can't for POVMs
            collapsible = False  # bool(self._evotype =="svterm") # need to use reps for collapsing now... TODO?

            if collapsible:
                factor_lists = [[t.collapse_vec() for t in fterms] for fterms in factor_lists]

            for factors in _itertools.product(*factor_lists):
                # create a term with a TensorProdSPAMVec - Note we always create
                # "prep"-mode vectors, since even when self._prep_or_effect == "effect" these
                # vectors are created with factor (prep- or effect-type) SPAMVecs not factor POVMs
                # we workaround this by still allowing such "prep"-mode
                # TensorProdSPAMVecs to be represented as effects (i.e. in torep('effect'...) works)
                coeff = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
                pre_op = TensorProductState("prep", [f.pre_ops[0] for f in factors
                                                     if (f.pre_ops[0] is not None)])
                post_op = TensorProductState("prep", [f.post_ops[0] for f in factors
                                                      if (f.post_ops[0] is not None)])
                term = _term.RankOnePolynomialPrepTerm.create_from(coeff, pre_op, post_op, self._evotype)

                if not collapsible:  # then may need to add more ops.  Assume factor ops are clifford gates
                    # Embed each factors ops according to their target qubit(s) and just daisy chain them
                    stateSpaceLabels = tuple(range(total_nQ)); curQ = 0
                    for f, nq in zip(factors, fnq):
                        targetLabels = tuple(range(curQ, curQ + nq)); curQ += nq
                        term.pre_ops.extend([_EmbeddedGateMap(stateSpaceLabels, targetLabels, op)
                                             for op in f.pre_ops[1:]])  # embed and add ops
                        term.post_ops.extend([_EmbeddedGateMap(stateSpaceLabels, targetLabels, op)
                                              for op in f.post_ops[1:]])  # embed and add ops

                terms.append(term)

        if return_coeff_polys:
            def _decompose_indices(x):
                return tuple(_modelmember._decompose_gpindices(
                    self.gpindices, _np.array(x, _np.int64)))

            poly_coeffs = [t.coeff.map_indices(_decompose_indices) for t in terms]  # with *local* indices
            tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
            if len(tapes) > 0:
                vtape = _np.concatenate([t[0] for t in tapes])
                ctape = _np.concatenate([t[1] for t in tapes])
            else:
                vtape = _np.empty(0, _np.int64)
                ctape = _np.empty(0, complex)
            coeffs_as_compact_polys = (vtape, ctape)
            #self.local_term_poly_coeffs[order] = coeffs_as_compact_polys #FUTURE?
            return terms, coeffs_as_compact_polys
        else:
            return terms  # Cache terms in FUTURE?

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.Np

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.concatenate([fct.to_vector() for fct in self.factors], axis=0)

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of SPAM vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this SPAM vector's current
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
        for sv in self.factors:
            sv.from_vector(v[sv.gpindices], close, dirty_value)  # factors hold local indices

        #Update representation, which may be a dense matrix or
        # just fast-kron arrays or a stabilizer state.
        self._update_rep()  # TODO - how does this apply to state reps??

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this SPAM vector.

        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

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
        assert(self._evotype in ("statevec", "densitymx"))
        typ = complex if self._evotype == "statevec" else 'd'
        derivMx = _np.zeros((self.dim, self.num_params), typ)

        #Product rule to compute jacobian
        for i, fct in enumerate(self.factors):  # loop over the spamvec/povm we differentiate wrt
            vec = fct if (self._prep_or_effect == "prep") else fct[self.effectLbls[i]]

            if vec.num_params == 0: continue  # no contribution
            deriv = vec.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this gate...
            deriv.shape = (vec.dim, vec.num_params)

            if i > 0:  # factors before ith
                pre = self.factors[0].to_dense()
                for vecA in self.factors[1:i]:
                    pre = _np.kron(pre, vecA.to_dense())
                deriv = _np.kron(pre[:, None], deriv)  # add a dummy 1-dim to 'pre' and do kron properly...

            if i + 1 < len(self.factors):  # factors after ith
                post = self.factors[i + 1].to_dense()
                for vecA in self.factors[i + 2:]:
                    post = _np.kron(post, vecA.to_dense())
                deriv = _np.kron(deriv, post[:, None])  # add a dummy 1-dim to 'post' and do kron properly...

            local_inds = fct.gpindices  # factor vectors hold local indices

            assert(local_inds is not None), \
                "Error: gpindices has not been initialized for factor %d - cannot compute derivative!" % i
            derivMx[:, local_inds] += deriv

        derivMx.shape = (self.dim, self.num_params)  # necessary?
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this SPAM vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return False

    def __str__(self):
        s = "Tensor product %s vector with length %d\n" % (self._prep_or_effect, self.dim)
        #ar = self.to_dense()
        #s += _mt.mx_to_string(ar, width=4, prec=2)

        # factors are just other SPAMVecs
        s += " x ".join([_mt.mx_to_string(fct.to_dense(), width=4, prec=2) for fct in self.factors])
        return s
