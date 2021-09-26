"""
The ComposedOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools

import numpy as _np

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import ExplicitBasis as _ExplicitBasis
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GlobalElementaryErrorgenLabel
from pygsti.tools import listtools as _lt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import slicetools as _slct


class ComposedOp(_LinearOperator):
    """
    An operation that is the composition of a number of map-like factors (possibly other `LinearOperator`s).

    Parameters
    ----------
    ops_to_compose : list
        List of `LinearOperator`-derived objects
        that are composed to form this operation map.  Elements are composed
        with vectors  in  *left-to-right* ordering, maintaining the same
        convention as operation sequences in pyGSTi.  Note that this is
        *opposite* from standard matrix multiplication order.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
        The special value `"auto"` is equivalent to the evolution type
        of `ops_to_compose[0]` *if* there's at least one operation being composed.

    state_space : StateSpace or "auto"
        State space of this error generator.  Can be set to `"auto"` to take
        the state space from `errgens_to_compose[0]` *if* there's at least one
        error generator being composed.
    """

    def __init__(self, ops_to_compose, evotype="auto", state_space="auto"):
        assert(len(ops_to_compose) > 0 or state_space != "auto"), \
            "Must compose at least one operation when state_space='auto'!"
        self.factorops = list(ops_to_compose)

        if state_space == "auto":
            state_space = ops_to_compose[0].state_space
        else:
            state_space = _statespace.StateSpace.cast(state_space)
        assert(all([state_space.is_compatible_with(operation.state_space) for operation in ops_to_compose])), \
            "All operations must have compatible state spaces (%d expected)!" % str(state_space)

        if evotype == "auto":
            evotype = ops_to_compose[0]._evotype
        assert(all([evotype == operation._evotype for operation in ops_to_compose])), \
            "All operations must have the same evolution type (%s expected)!" % evotype
        evotype = _Evotype.cast(evotype)

        #Create representation object
        rep_type_order = ('dense', 'composed') if evotype.prefer_dense_reps else ('composed', 'dense')
        rep = None
        for rep_type in rep_type_order:
            try:
                if rep_type == 'composed':
                    factor_op_reps = [op._rep for op in self.factorops]
                    rep = evotype.create_composed_rep(factor_op_reps, state_space)
                elif rep_type == 'dense':
                    rep = evotype.create_dense_superop_rep(None, state_space)
                else:
                    assert(False), "Logic error!"

                self._rep_type = rep_type
                break

            except AttributeError:
                pass  # just go to the next rep_type

        if rep is None:
            raise ValueError("Unable to construct representation with evotype: %s" % str(evotype))

        # caches in case terms are used
        self.terms = {}
        self.local_term_poly_coeffs = {}

        _LinearOperator.__init__(self, rep, evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members
        if self._rep_type == 'dense': self._update_denserep()  # update dense rep if needed

    def _update_denserep(self):
        """Performs additional update for the case when we use a dense underlying representation."""
        if len(self.factorops) == 0:
            mx = _np.identity(self.state_space.dim, 'd')
        else:
            mx = self.factorops[0].to_dense(on_space='HilbertSchmidt')
            for op in self.factorops[1:]:
                mx = _np.dot(op.to_dense(on_space='HilbertSchmidt'), mx)

        self._rep.base.flags.writeable = True
        self._rep.base[:, :] = mx
        self._rep.base.flags.writeable = False

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        ops_to_compose = [serial_memo[i] for i in mm_dict['submembers']]
        return cls(ops_to_compose, mm_dict['evotype'], state_space)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factorops

    def set_time(self, t):
        """
        Sets the current time for a time-dependent operator.

        For time-independent operators (the default), this function does nothing.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        None
        """
        memo = set()
        for factor in self.factorops:
            if id(factor) in memo: continue
            factor.set_time(t); memo.add(id(factor))

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
        _modelmember.ModelMember.set_gpindices(self, gpindices, parent, memo)

    def append(self, *factorops_to_add):
        """
        Add one or more factors to this operator.

        Parameters
        ----------
        *factors_to_add : LinearOperator
            One or multiple factor operators to add on at the *end* (evaluated
            last) of this operator.

        Returns
        -------
        None
        """
        self.factorops.extend(factorops_to_add)
        if self._rep_type == 'dense':
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # if our params may have changed
            self._parent = None  # mark this object for re-allocation

    def insert(self, insert_at, *factorops_to_insert):
        """
        Insert one or more factors into this operator.

        Parameters
        ----------
        insert_at : int
            The index at which to insert `factorops_to_insert`.  The factor at this
            index and those after it are shifted back by `len(factorops_to_insert)`.

        *factors_to_insert : LinearOperator
            One or multiple factor operators to insert within this operator.

        Returns
        -------
        None
        """
        self.factorops[insert_at:insert_at] = list(factorops_to_insert)
        if self._rep_type == 'dense':
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # if our params may have changed
            self._parent = None  # mark this object for re-allocation

    def remove(self, *factorop_indices):
        """
        Remove one or more factors from this operator.

        Parameters
        ----------
        *factorop_indices : int
            One or multiple factor indices to remove from this operator.

        Returns
        -------
        None
        """
        for i in sorted(factorop_indices, reverse=True):
            del self.factorops[i]
        if self._rep_type == 'dense':
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed
            self._parent = None  # mark this object for re-allocation

    # REMOVE - unnecessary and doesn't work correctly when, e.g., multiple factors are the same object
    #def copy(self, parent=None, memo=None):
    #    """
    #    Copy this object.
    #
    #    Parameters
    #    ----------
    #    parent : Model, optional
    #        The parent model to set for the copy.
    #
    #    Returns
    #    -------
    #    LinearOperator
    #        A copy of this object.
    #    """
    #    # We need to override this method so that factor operations have their
    #    # parent reset correctly.
    #    if memo is not None and id(self) in memo: return memo[id(self)]
    #    cls = self.__class__  # so that this method works for derived classes too
    #    copyOfMe = cls([g.copy(parent, memo) for g in self.factorops], self._evotype, self.state_space)
    #    return self._copy_gpindices(copyOfMe, parent, memo)

    def to_sparse(self, on_space='minimal'):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        mx = self.factorops[0].to_sparse(on_space)
        for op in self.factorops[1:]:
            mx = op.to_sparse(on_space).dot(mx)
        return mx

    def to_dense(self, on_space='minimal'):
        """
        Return this operation as a dense matrix.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        numpy.ndarray
        """
        if self._rep_type == 'dense':
            #We already have a dense version stored
            return self._rep.to_dense(on_space)
        elif len(self.factorops) == 0:
            return _np.identity(self.state_space.dim, 'd')
        else:
            mx = self.factorops[0].to_dense(on_space)
            for op in self.factorops[1:]:
                mx = _np.dot(op.to_dense(on_space), mx)
            return mx

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        plabels_per_local_index = _collections.defaultdict(list)
        for operation, factorgate_local_inds in zip(self.factorops, self._submember_rpindices):
            for i, plbl in zip(_slct.to_array(factorgate_local_inds), operation.parameter_labels):
                plabels_per_local_index[i].append(plbl)

        vl = _np.empty(self.num_params, dtype=object)
        for i in range(self.num_params):
            vl[i] = ', '.join(plabels_per_local_index[i])
        return vl

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        assert(self.gpindices is not None), "Must set a ComposedOp's .gpindices before calling to_vector"
        v = _np.empty(self.num_params, 'd')
        for operation, factorgate_local_inds in zip(self.factorops, self._submember_rpindices):
            #factorgate_local_inds = _modelmember._decompose_gpindices(
            #    self.gpindices, operation.gpindices)
            v[factorgate_local_inds] = operation.to_vector()
        return v

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
        assert(self.gpindices is not None), "Must set a ComposedOp's .gpindices before calling from_vector"
        for operation, factorgate_local_inds in zip(self.factorops, self._submember_rpindices):
            #factorgate_local_inds = _modelmember._decompose_gpindices(
            #    self.gpindices, operation.gpindices)
            operation.from_vector(v[factorgate_local_inds], close, dirty_value)
        if self._rep_type == 'dense': self._update_denserep()
        self.dirty = dirty_value

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
            Array of derivatives with shape (dimension^2, num_params)
        """
        typ = complex if any([_np.iscomplexobj(op.to_dense(on_space='minimal'))
                              for op in self.factorops]) else 'd'
        derivMx = _np.zeros((self.dim, self.dim, self.num_params), typ)

        #Product rule to compute jacobian
        # loop over the operation we differentiate wrt
        for i, (op, factorgate_local_inds) in enumerate(zip(self.factorops, self._submember_rpindices)):
            if op.num_params == 0: continue  # no contribution
            deriv = op.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this operation...
            deriv.shape = (self.dim, self.dim, op.num_params)

            if i > 0:  # factors before ith
                pre = self.factorops[0].to_dense(on_space='minimal')
                for opA in self.factorops[1:i]:
                    pre = _np.dot(opA.to_dense(on_space='minimal'), pre)
                #deriv = _np.einsum("ija,jk->ika", deriv, pre )
                deriv = _np.transpose(_np.tensordot(deriv, pre, (1, 0)), (0, 2, 1))

            if i + 1 < len(self.factorops):  # factors after ith
                post = self.factorops[i + 1].to_dense(on_space='minimal')
                for opA in self.factorops[i + 2:]:
                    post = _np.dot(opA.to_dense(on_space='minimal'), post)
                #deriv = _np.einsum("ij,jka->ika", post, deriv )
                deriv = _np.tensordot(post, deriv, (1, 0))

            #factorgate_local_inds = _modelmember._decompose_gpindices(
            #    self.gpindices, op.gpindices)
            derivMx[:, :, factorgate_local_inds] += deriv

        derivMx.shape = (self.dim**2, self.num_params)
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

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
        if order not in self.terms:
            self._compute_taylor_order_terms(order, max_polynomial_vars, self.gpindices_as_array())

        if return_coeff_polys:
            #Return coefficient polys in terms of *local* parameters (get_taylor_terms
            #  and above composition gives polys in terms of *global*, model params)
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_polynomial_vars, gpindices_array):  # separated for profiling
        terms = []
        for p in _lt.partition_into(order, len(self.factorops)):
            factor_lists = [self.factorops[i].taylor_order_terms(pi, max_polynomial_vars) for i, pi in enumerate(p)]
            for factors in _itertools.product(*factor_lists):
                terms.append(_term.compose_terms(factors))
        self.terms[order] = terms

        #def _decompose_indices(x):
        #    return tuple(_modelmember._decompose_gpindices(
        #        self.gpindices, _np.array(x, _np.int64)))

        mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars, _np.int64))
        for ii, i in enumerate(gpindices_array):
            mapvec[i] = ii

        #poly_coeffs = [t.coeff.map_indices(_decompose_indices) for t in terms]  # with *local* indices
        poly_coeffs = [t.coeff.mapvec_indices(mapvec) for t in terms]  # with *local* indices
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)
        self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

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
        terms = []
        factor_lists_cache = [
            [rep.taylor_order_terms_above_mag(i, max_polynomial_vars, min_term_mag) for i in range(order + 1)]
            for rep in self.factorops
        ]
        for p in _lt.partition_into(order, len(self.factorops)):
            # factor_lists = [self.factorops[i].get_taylor_order_terms_above_mag(pi, max_polynomial_vars, min_term_mag)
            #                 for i, pi in enumerate(p)]
            factor_lists = [factor_lists_cache[i][pi] for i, pi in enumerate(p)]
            for factors in _itertools.product(*factor_lists):
                mag = _np.product([factor.magnitude for factor in factors])
                if mag >= min_term_mag:
                    terms.append(_term.compose_terms_with_mag(factors, mag))
        return terms
        #def _decompose_indices(x):
        #    return tuple(_modelmember._decompose_gpindices(
        #        self.gpindices, _np.array(x, _np.int64)))
        #
        #mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars,_np.int64))
        #for ii,i in enumerate(self.gpindices_as_array()):
        #    mapvec[i] = ii
        #
        ##poly_coeffs = [t.coeff.map_indices(_decompose_indices) for t in terms]  # with *local* indices
        #poly_coeffs = [t.coeff.mapvec_indices(mapvec) for t in terms]  # with *local* indices
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #coeffs_as_compact_polys = (vtape, ctape)
        #self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

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
        # In general total term mag == sum of the coefficients of all the terms (taylor expansion)
        #  of an errorgen or operator.
        # In this case, since the taylor expansions are composed (~multiplied),
        # the total term magnitude is just the product of those of the components.
        return _np.product([f.total_term_magnitude for f in self.factorops])

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
        opmags = [f.total_term_magnitude for f in self.factorops]
        product = _np.product(opmags)
        ret = _np.zeros(self.num_params, 'd')
        for opmag, f, f_local_inds in zip(opmags, self.factorops, self._submember_rpindices):
            #f_local_inds = _modelmember._decompose_gpindices(
            #    self.gpindices, f.gpindices)
            local_deriv = product / opmag * f.total_term_magnitude_deriv
            ret[f_local_inds] += local_deriv
        return ret

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return any([op.has_nonzero_hessian() for op in self.factorops])

    def transform_inplace(self, s):
        """
        Update operation matrix `O` with `inv(s) * O * s`.

        Generally, the transform function updates the *parameters* of
        the operation such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the operation parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `s` is an instance of `TPGaugeGroupElement` or
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        for operation in self.factorops:
            operation.transform_inplace(s)

    def errorgen_coefficients(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients of this operation.

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
            This is the value returned by :method:`error_rates`.

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
        #*** Note: this function is nearly identitcal to ComposedErrorgen.coefficients() ***
        Ltermdict = _collections.OrderedDict()
        basisdict = _collections.OrderedDict()
        first_nonempty_basis = None
        constant_basis = None  # the single same Basis used for every factor with a nonempty basis

        for op in self.factorops:
            try:
                factor_coeffs = op.errorgen_coefficients(return_basis, logscale_nonham)
            except AttributeError:
                continue  # just skip members that don't implemnt errorgen_coefficients (?)

            if return_basis:
                ltdict, factor_basis = factor_coeffs
                if len(factor_basis) > 0:
                    if first_nonempty_basis is None:
                        first_nonempty_basis = factor_basis
                        constant_basis = factor_basis  # seed constant_basis
                    elif factor_basis != constant_basis:
                        constant_basis = None  # factors have different bases - no constant_basis!

                # see if we need to update basisdict and ensure we do so in a consistent
                # way - if factors use the same basis labels these must refer to the same
                # basis elements.
                #FUTURE: maybe a way to do this without always accessing basis *elements*?
                #  (maybe do a pass to check for a constant_basis without any .elements refs?)
                for lbl, basisEl in zip(factor_basis.labels, factor_basis.elements):
                    if lbl in basisdict:
                        assert(_mt.safe_norm(basisEl - basisdict[lbl]) < 1e-6), "Ambiguous basis label %s" % lbl
                    else:
                        basisdict[lbl] = basisEl
            else:
                ltdict = factor_coeffs

            for key, coeff in ltdict.items():
                if key in Ltermdict:
                    Ltermdict[key] += coeff
                else:
                    Ltermdict[key] = coeff

        if return_basis:
            #Use constant_basis or turn basisdict into a Basis to return
            if constant_basis is not None:
                basis = constant_basis
            elif first_nonempty_basis is not None:
                #Create an ExplictBasis using the matrices in basisdict plus the identity
                lbls = ['I'] + list(basisdict.keys())
                mxs = [first_nonempty_basis[0]] + list(basisdict.values())
                basis = _ExplicitBasis(mxs, lbls, name=None,
                                       real=first_nonempty_basis.real,
                                       sparse=first_nonempty_basis.sparse)
            return Ltermdict, basis
        else:
            return Ltermdict

    def errorgen_coefficient_labels(self):
        """
        The elementary error-generator labels corresponding to the elements of :method:`errorgen_coefficients_array`.

        Returns
        -------
        tuple
            A tuple of (<type>, <basisEl1> [,<basisEl2]) elements identifying the elementary error
            generators of this gate.
        """
        return tuple(_itertools.chain(*[op.errorgen_coefficient_labels() for op in self.factorops]))

    def errorgen_coefficients_array(self):
        """
        The weighted coefficients of this operation's error generator in terms of "standard" error generators.

        Constructs a 1D array of all the coefficients returned by :method:`errorgen_coefficients`,
        weighted so that different error generators can be weighted differently when a
        `errorgen_penalty_factor` is used in an objective function.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to the number of coefficients in the linear combination
            of standard error generators that is this operation's error generator.
        """
        return _np.concatenate([op.errorgen_coefficients_array() for op in self.factorops])

    def errorgen_coefficients_array_deriv_wrt_params(self):
        """
        The jacobian of :method:`errogen_coefficients_array` with respect to this operation's parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(num_coeffs, num_params)` where `num_coeffs` is the number of
            coefficients of this operation's error generator and `num_params` is this operation's
            number of parameters.
        """
        deriv_mxs = [op.errorgen_coefficients_array_deriv_wrt_params() for op in self.factorops]
        return _np.concatenate([mx for mx in deriv_mxs if mx.size > 0], axis=0)  # allow (0,0)-shaped matrices to be ignored.

    def error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this operation.

        The "error rate" for an individual Hamiltonian error is the angle
        about the "axis" (generalized in the multi-qubit case)
        corresponding to a particular basis element, i.e. `theta` in
        the unitary channel `U = exp(i * theta/2 * BasisElement)`.

        The "error rate" for an individual Stochastic error is the
        contribution that basis element's term would have to the
        error rate of a depolarization channel.  For example, if
        the rate corresponding to the term ('S','X') is 0.01 this
        means that the coefficient of the rho -> X*rho*X-rho error
        generator is set such that if this coefficient were used
        for all 3 (X,Y, and Z) terms the resulting depolarizing
        channel would have error rate 3*0.01 = 0.03.

        Note that because error generator terms do not necessarily
        commute with one another, the sum of the returned error
        rates is not necessarily the error rate of the overall
        channel.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case.
        """
        return self.errorgen_coefficients(return_basis=False, logscale_nonham=True)

    def set_errorgen_coefficients(self, lindblad_term_dict, action="update", logscale_nonham=False, truncate=True):
        """
        Sets the coefficients of terms in the error generator of this operation.

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
            "equivalent" depolarizing channel, see :method:`errorgen_coefficients`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :method:`set_error_rates`.

        truncate : bool, optional
            Whether to allow adjustment of the errogen coefficients in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given coefficients
            cannot be set as specified.

        Returns
        -------
        None
        """
        sslbls = self.state_space.tensor_product_block_labels(0)
        values_to_set = {_GlobalElementaryErrorgenLabel.cast(k, sslbls): v for k, v in lindblad_term_dict.items()}

        for op in self.factorops:
            try:
                available_factor_coeffs = op.errorgen_coefficients(False, logscale_nonham)
            except AttributeError:
                continue  # just skip members that don't implemnt errorgen_coefficients (?)

            Ltermdict_local = _collections.OrderedDict([(k, v) for k, v in values_to_set.items()
                                                        if k in available_factor_coeffs])
            op.set_errorgen_coefficients(Ltermdict_local, action, logscale_nonham, truncate)
            for k in Ltermdict_local:
                del values_to_set[k]  # remove the values that we just set

        if len(values_to_set) > 0:  # then there were some values that could not be set!
            raise ValueError("These errorgen coefficients could not be set: %s" %
                             (",".join(map(str, values_to_set.keys()))))

        if self._rep_type == 'dense': self._update_denserep()
        self.dirty = True

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in the error generator of this operation.

        Values are set so that the contributions of the resulting channel's
        error rate are given by the values in `lindblad_term_dict`.  See
        :method:`error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case, when they may be complex.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_errorgen_coefficients(lindblad_term_dict, action, logscale_nonham=True)

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "composed of %d factors" % len(self.factorops)

    def __str__(self):
        """ Return string representation """
        s = "Composed operation of %d factors:\n" % len(self.factorops)
        for i, operation in enumerate(self.factorops):
            s += "Factor %d:\n" % i
            s += str(operation)
        return s
