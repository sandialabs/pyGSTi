"""
The TensorProductPOVMEffect class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import itertools as _itertools

import numpy as _np

from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import listtools as _lt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import slicetools as _slct
from pygsti import SpaceT


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

    state_space : StateSpace, optional
        The state space for this operation.
    """

    def __init__(self, factors, povm_effect_lbls, state_space):
        assert(len(factors) > 0), "Must have at least one factor!"

        self.factors = factors  # do *not* copy - needs to reference common objects
        self.effectLbls = _np.array(povm_effect_lbls)

        evotype = self.factors[0]._evotype
        rep = evotype.create_tensorproduct_effect_rep(factors, self.effectLbls, state_space)

        _POVMEffect.__init__(self, rep, evotype)
        self._rep.factor_effects_have_changed()  # initializes rep data

        #Set our parent and gpindices based on those of factor-POVMs, which
        # should all be owned by a TensorProdPOVM object.
        # (for now say we depend on *all* the POVMs parameters (even though
        #  we really only depend on one element of each POVM, which may allow
        #  using just a subset of each factor POVMs indices - but this is tricky).
        self.init_gpindices()

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factors  # factor POVM object

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
        mm_dict['subpovm_effect_labels'] = self.effectLbls.tolist()
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        factors = [serial_memo[i] for i in mm_dict['submembers']]
        return cls(factors, mm_dict['subpovm_effect_labels'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return (self.state_space == other.state_space
                and _np.array_equal(self.effectLbls, other.effectLbls))

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        vl = _np.empty(self.num_params, dtype=object)
        for povm, povm_local_inds in zip(self.factors, self._submember_rpindices):
            vl[povm_local_inds] = povm.parameter_labels
        return vl

    def to_dense(self, on_space: SpaceT='minimal', scratch=None):
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
        return self._rep.to_dense(on_space)

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this POVM effect vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        POVMEffect's parameters, where the polynomial's variable indices index the
        *global* parameters of the POVMEffect's parent (usually a :class:`Model`)
        , not the POVMEffect's local parameter array (i.e. that returned from
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
        terms = []
        fnq = [int(round(_np.log2(f.dim))) // 2 for f in self.factors]  # num of qubits per factor
        # assumes density matrix evolution
        total_nQ = sum(fnq)  # total number of qubits

        for p in _lt.partition_into(order, len(self.factors)):
            factorPOVMs = self.factors
            factor_lists = [factorPOVMs[i][Elbl].taylor_order_terms(pi, max_polynomial_vars)
                            for i, (pi, Elbl) in enumerate(zip(p, self.effectLbls))]

            # When possible, create COLLAPSED factor_lists so each factor has just a single
            # (POVMEffect) pre & post op, which can be formed into the new terms'
            # TensorProdPOVMEffect ops.
            # - DON'T collapse stabilizer states & clifford ops - can't for POVMs
            collapsible = False  # bool(self._evotype =="svterm") # need to use reps for collapsing now... TODO?

            if collapsible:
                factor_lists = [[t.collapse_vec() for t in fterms] for fterms in factor_lists]

            for factors in _itertools.product(*factor_lists):
                coeff = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])

                #Some gymnastics here to create an EffectRep that is the tensor product of a fixed
                # set of other effect reps.  This isn't a TensorProductPOVMEffect, as that takes a
                # set of POVMs as factors.  So we convert the factor effects -> dense -> states
                # and then use a conjugated tensor-product-state as the final effect object.
                #Note: we set basis=None below because I don't think these args are actually needed,
                # as they'd only be used if the evotype was like densitymx and needed to convert to
                # a dense superoperator.
                pre_state_rep = self._evotype.create_tensorproduct_state_rep(
                    [self._evotype.create_pure_state_rep(f.pre_effect.to_dense("Hilbert"), None,
                                                         f.pre_effect.state_space)
                     for f in factors if (f.pre_effect is not None)], self.state_space)
                pre_rep = self._evotype.create_conjugatedstate_effect_rep(pre_state_rep)

                post_state_rep = self._evotype.create_tensorproduct_state_rep(
                    [self._evotype.create_pure_state_rep(f.post_effect.to_dense("Hilbert"), None,
                                                         f.post_effect.state_space)
                     for f in factors if (f.post_effect is not None)], self.state_space)
                post_rep = self._evotype.create_conjugatedstate_effect_rep(post_state_rep)

                term = _term.RankOnePolynomialEffectTerm.create_from(coeff, pre_rep, post_rep,
                                                                     self._evotype, self.state_space)

                if not collapsible:  # then may need to add more ops.  Assume factor ops are clifford gates
                    # Embed each factors ops according to their target qubit(s) and just daisy chain them
                    ss = _statespace.QubitSpace(total_nQ); curQ = 0
                    for f, nq in zip(factors, fnq):
                        targetLabels = tuple(range(curQ, curQ + nq)); curQ += nq
                        term._rep.pre_ops.extend([self._evotype.create_embedded_rep(ss, targetLabels, op)
                                                  for op in f.pre_ops])  # embed and add ops
                        term._rep.post_ops.extend([self._evotype.create_embedded_rep(ss, targetLabels, op)
                                                   for op in f.post_ops])  # embed and add ops

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
        raise ValueError(("'`to_vector` should not be called on effect-like"
                          " TensorProdPOVMEffects (instead it should be called"
                          " on the POVM)"))

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
        if all([self.effectLbls[i] == iter(povm.keys()).__next__()
                for i, povm in enumerate(self.factors)]):
            #then this is the *first* vector in the larger TensorProdPOVM
            # and we should initialize all of the factor_povms
            for povm, povm_local_inds in zip(self.factors, self._submember_rpindices):
                #local_inds = _modelmember._decompose_gpindices(
                #    self.gpindices, povm.gpindices)
                povm.from_vector(v[povm_local_inds], close, dirty_value)

        #Update representation, which may be a dense matrix or
        # just fast-kron arrays or a stabilizer state.
        self._rep.factor_effects_have_changed()

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
        typ = self.factors[0].to_dense("minimal").dtype if len(self.factors) > 0 else 'd'

        #HACK to deal with fact that output of to_dense is really what is differentiated
        # but this may not match self.dim == self.state_space.dim, e.g. for pure state vecs.
        dims = [len(fct.to_dense("minimal")) for fct in self.factors]
        dim = int(_np.prod(dims))

        derivMx = _np.zeros((dim, self.num_params), typ)

        #Product rule to compute jacobian
        # loop over the spamvec/povm we differentiate wrt:
        for i, (fct, fct_local_inds, fct_dim) in enumerate(zip(self.factors, self._submember_rpindices, dims)):
            vec = fct[self.effectLbls[i]]

            if vec.num_params == 0: continue  # no contribution
            deriv = vec.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this gate...
            deriv.shape = (fct_dim, vec.num_params)

            if i > 0:  # factors before ith
                pre = self.factors[0][self.effectLbls[0]].to_dense("minimal")
                for j, fctA in enumerate(self.factors[1:i], start=1):
                    pre = _np.kron(pre, fctA[self.effectLbls[j]].to_dense("minimal"))
                deriv = _np.kron(pre[:, None], deriv)  # add a dummy 1-dim to 'pre' and do kron properly...

            if i + 1 < len(self.factors):  # factors after ith
                post = self.factors[i + 1][self.effectLbls[i + 1]].to_dense("minimal")
                for j, fctA in enumerate(self.factors[i + 2:], start=i + 2):
                    post = _np.kron(post, fctA[self.effectLbls[j]].to_dense("minimal"))
                deriv = _np.kron(deriv, post[:, None])  # add a dummy 1-dim to 'post' and do kron properly...

            assert(fct_local_inds is not None), \
                "Error: gpindices has not been initialized for factor %d - cannot compute derivative!" % i
            derivMx[:, fct_local_inds] += deriv

        derivMx.shape = (dim, self.num_params)  # necessary?
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

    def __str__(self):
        s = "Tensor product %s vector with length %d\n" % (self._prep_or_effect, self.dim)
        #ar = self.to_dense()
        #s += _mt.mx_to_string(ar, width=4, prec=2)

        # factors are POVMs
        s += " x ".join([_mt.mx_to_string(fct[self.effectLbls[i]].to_dense("minimal"), width=4, prec=2)
                         for i, fct in enumerate(self.factors)])
        return s
