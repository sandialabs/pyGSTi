import itertools as _itertools
import functools as _functools

import numpy as _np
from .effect import POVMEffect as _POVMEffect
from .. import modelmember as _modelmember
from ..states.tensorprodstate import TensorProductState as _TensorProductState
from ...tools import slicetools as _slct
from ...tools import listtools as _lt
from ...tools import matrixtools as _mt
from ...objects import term as _term


class TensorProductPOVMEffect(_POVMEffect):
    """
    A state vector that is a tensor-product of other state vectors.

    Parameters
    ----------
    factors : list of POVMs
        a list of "reference" POVMs into which `povm_effect_lbls` indexes.

    povm_effect_lbls : array-like
        The effect label of each factor POVM which is tensored together to form
        this effect vector.
    """

    def __init__(self, typ, factors, povm_effect_lbls=None):
        assert(len(factors) > 0), "Must have at least one factor!"

        self.factors = factors  # do *not* copy - needs to reference common objects
        self.Np = sum([fct.num_params for fct in factors])
        self.effectLbls = _np.array(povm_effect_lbls)

        evotype = self.factors[0]._evotype
        rep = evotype.create_tensor_product_effect(factors, self.effectLbls)

        _POVMEffect.__init__(self, rep, evotype)
        self._rep.factor_effects_have_changed()  # initializes rep data
        #sets gpindices, so do before stuff below

        #Set our parent and gpindices based on those of factor-POVMs, which
        # should all be owned by a TensorProdPOVM object.
        # (for now say we depend on *all* the POVMs parameters (even though
        #  we really only depend on one element of each POVM, which may allow
        #  using just a subset of each factor POVMs indices - but this is tricky).
        self.set_gpindices(_slct.list_to_slice(
            _np.concatenate([fct.gpindices_as_array()
                             for fct in factors], axis=0), True, False),
                           factors[0].parent)  # use parent of first factor
        # (they should all be the same)

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
            factorPOVMs = self.factors
            factor_lists = [factorPOVMs[i][Elbl].taylor_order_terms(pi, max_polynomial_vars)
                            for i, (pi, Elbl) in enumerate(zip(p, self.effectLbls))]

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
                pre_op = _TensorProductState([f.pre_ops[0] for f in factors
                                              if (f.pre_ops[0] is not None)])
                post_op = _TensorProductState([f.post_ops[0] for f in factors
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
        raise ValueError(("'`to_vector` should not be called on effect-like"
                          " TensorProdSPAMVecs (instead it should be called"
                          " on the POVM)"))

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
        if all([self.effectLbls[i] == list(povm.keys())[0]
                for i, povm in enumerate(self.factors)]):
            #then this is the *first* vector in the larger TensorProdPOVM
            # and we should initialize all of the factor_povms
            for povm in self.factors:
                local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, povm.gpindices)
                povm.from_vector(v[local_inds], close, dirty_value)

        #Update representation, which may be a dense matrix or
        # just fast-kron arrays or a stabilizer state.
        self._rep.factor_effects_have_changed()

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
                pre = self.factors[0][self.effectLbls[0]].to_dense()
                for j, fctA in enumerate(self.factors[1:i], start=1):
                    pre = _np.kron(pre, fctA[self.effectLbls[j]].to_dense())
                deriv = _np.kron(pre[:, None], deriv)  # add a dummy 1-dim to 'pre' and do kron properly...

            if i + 1 < len(self.factors):  # factors after ith
                post = self.factors[i + 1][self.effectLbls[i + 1]].to_dense()
                for j, fctA in enumerate(self.factors[i + 2:], start=i + 2):
                    post = _np.kron(post, fctA[self.effectLbls[j]].to_dense())
                deriv = _np.kron(deriv, post[:, None])  # add a dummy 1-dim to 'post' and do kron properly...

            # POVM-factors hold global indices (b/c they're meant to be shareable)
            local_inds = _modelmember._decompose_gpindices(
                self.gpindices, fct.gpindices)

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

        # factors are POVMs
        s += " x ".join([_mt.mx_to_string(fct[self.effectLbls[i]].to_dense(), width=4, prec=2)
                         for i, fct in enumerate(self.factors)])
        return s
