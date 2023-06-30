"""
The TensorProductState class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import functools as _functools
import itertools as _itertools

import numpy as _np

from pygsti.modelmembers.states.state import State as _State
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import listtools as _lt
from pygsti.tools import matrixtools as _mt


class TensorProductState(_State):
    """
    A state vector that is a tensor-product of other state vectors.

    Parameters
    ----------
    factors : list of States
        a list of the component states to take the tensor product of.

    state_space : StateSpace, optional
        The state space for this operation.
    """

    def __init__(self, factors, state_space):
        assert(len(factors) > 0), "Must have at least one factor!"

        self.factors = factors  # do *not* copy - needs to reference common objects

        evotype = self.factors[0]._evotype
        rep = evotype.create_tensorproduct_state_rep([f._rep for f in factors], state_space)

        _State.__init__(self, rep, evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members
        self._update_rep()  # initializes rep data

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        factors = [serial_memo[i] for i in mm_dict['submembers']]
        return cls(factors, state_space)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factors  # factor POVM object

    def _update_rep(self):
        self._rep.reps_have_changed()

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        vl = _np.empty(self.num_params, dtype=object)
        for factor_state, factor_local_inds in zip(self.factors, self._submember_rpindices):
            vl[factor_local_inds] = factor_state.parameter_labels
        return vl

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
        return self._rep.to_dense(on_space)

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
        terms = []
        fnq = [int(round(_np.log2(f.dim))) // 2 for f in self.factors]  # num of qubits per factor
        # assumes density matrix evolution
        total_nQ = sum(fnq)  # total number of qubits

        for p in _lt.partition_into(order, len(self.factors)):
            factor_lists = [self.factors[i].taylor_order_terms(pi, max_polynomial_vars) for i, pi in enumerate(p)]

            # When possible, create COLLAPSED factor_lists so each factor has just a single
            # (State) pre & post op, which can be formed into the new terms'
            # TensorProdState ops.
            # - DON'T collapse stabilizer states & clifford ops - can't for POVMs
            collapsible = False  # bool(self._evotype =="svterm") # need to use reps for collapsing now... TODO?

            if collapsible:
                factor_lists = [[t.collapse_vec() for t in fterms] for fterms in factor_lists]

            for factors in _itertools.product(*factor_lists):
                # create a term with a TensorProdState - Note we always create
                # "prep"-mode vectors, since even when self._prep_or_effect == "effect" these
                # vectors are created with factor (prep- or effect-type) States not factor POVMs
                # we workaround this by still allowing such "prep"-mode
                # TensorProdStates to be represented as effects (i.e. in torep('effect'...) works)
                coeff = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
                pre_rep = self._evotype.create_tensorproduct_state_rep(
                    [f.pre_state for f in factors if (f.pre_state is not None)], self.state_space)
                post_rep = self._evotype.create_tensorproduct_state_rep(
                    [f.post_state for f in factors if (f.post_state is not None)], self.state_space)
                term = _term.RankOnePolynomialPrepTerm.create_from(coeff, pre_rep, post_rep,
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
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        v = _np.empty(self.num_params, 'd')
        for factor_state, factor_local_inds in zip(self.factors, self._submember_rpindices):
            v[factor_local_inds] = factor_state.to_vector()
        return v

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
        for factor_state, factor_local_inds in zip(self.factors, self._submember_rpindices):
            factor_state.from_vector(v[factor_local_inds], close, dirty_value)

        #Update representation, which may be a dense matrix or
        # just fast-kron arrays or a stabilizer state.
        self._update_rep()  # TODO - how does this apply to state reps??

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
        typ = self.factors[0].to_dense(on_space='minimal').dtype if len(self.factors) > 0 else 'd'

        #HACK to deal with fact that output of to_dense is really what is differentiated
        # but this may not match self.dim == self.state_space.dim, e.g. for pure state vecs.
        dims = [len(fct.to_dense(on_space='minimal')) for fct in self.factors]
        dim = int(_np.product(dims))

        derivMx = _np.zeros((dim, self.num_params), typ)

        #Product rule to compute jacobian
        # loop over the spamvec/povm we differentiate wrt:
        for i, (fct, fct_local_inds, fct_dim) in enumerate(zip(self.factors, self._submember_rpindices, dims)):
            vec = fct

            if vec.num_params == 0: continue  # no contribution
            deriv = vec.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this gate...
            deriv.shape = (fct_dim, vec.num_params)

            if i > 0:  # factors before ith
                pre = self.factors[0].to_dense(on_space='minimal')
                for vecA in self.factors[1:i]:
                    pre = _np.kron(pre, vecA.to_dense(on_space='minimal'))
                deriv = _np.kron(pre[:, None], deriv)  # add a dummy 1-dim to 'pre' and do kron properly...

            if i + 1 < len(self.factors):  # factors after ith
                post = self.factors[i + 1].to_dense(on_space='minimal')
                for vecA in self.factors[i + 2:]:
                    post = _np.kron(post, vecA.to_dense(on_space='minimal'))
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
        Whether this state vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return False

    def __str__(self):
        s = "Tensor product %s vector with length %d\n" % (self._prep_or_effect, self.dim)
        #ar = self.to_dense()
        #s += _mt.mx_to_string(ar, width=4, prec=2)

        # factors are just other States
        s += " x ".join([_mt.mx_to_string(fct.to_dense(on_space='minimal'), width=4, prec=2) for fct in self.factors])
        return s
