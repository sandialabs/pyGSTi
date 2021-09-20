"""
The ComposedErrorgen class and supporting functionality.
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

import numpy as _np
import scipy.sparse as _sps

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers import modelmember as _modelmember
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import ExplicitBasis as _ExplicitBasis
from pygsti.tools import matrixtools as _mt


class ComposedErrorgen(_LinearOperator):
    """
    A composition (sum!) of several Lindbladian exponent operators.

    That is, a *sum* (not product) of other error generators.

    Parameters
    ----------
    errgens_to_compose : list
        List of `LinearOperator`-derived objects that are summed together (composed)
        to form this error generator.

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

    def __init__(self, errgens_to_compose, evotype="auto", state_space="auto"):
        assert(len(errgens_to_compose) > 0 or state_space != "auto"), \
            "Must compose at least one error generator when state_space='auto'!"
        self.factors = errgens_to_compose

        if state_space == "auto":
            state_space = errgens_to_compose[0].state_space
        else:
            state_space = _statespace.StateSpace.cast(state_space)

        assert(all([state_space.is_compatible_with(eg.state_space) for eg in errgens_to_compose])), \
            "All error generators must have compatible state spaces (%s expected)!" % str(state_space)

        if evotype == "auto":
            evotype = errgens_to_compose[0]._evotype
        evotype = _Evotype.cast(evotype)
        assert(all([evotype == eg._evotype for eg in errgens_to_compose])), \
            "All error generators must have the same evolution type (%s expected)!" % evotype

        # set "API" error-generator members (to interface properly w/other objects)
        # FUTURE: create a base class that defines this interface (maybe w/properties?)
        #self.sparse = errgens_to_compose[0].sparse \
        #    if len(errgens_to_compose) > 0 else False
        #assert(all([self.sparse == eg.sparse for eg in errgens_to_compose])), \
        #    "All error generators must have the same sparsity (%s expected)!" % self.sparse

        self.matrix_basis = errgens_to_compose[0].matrix_basis \
            if len(errgens_to_compose) > 0 else None
        assert(all([self.matrix_basis == eg.matrix_basis for eg in errgens_to_compose])), \
            "All error generators must have the same matrix basis (%s expected)!" % self.matrix_basis

        #Create representation object
        factor_reps = [op._rep for op in self.factors]
        rep = evotype.create_sum_rep(factor_reps, state_space)

        _LinearOperator.__init__(self, rep, evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        errgens_to_compose = [serial_memo[i] for i in mm_dict['submembers']]
        return cls(errgens_to_compose, mm_dict['evotype'], state_space)

    def coefficients(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients of this error generator.

        Note that these are not necessarily the parameter values, as these
        coefficients are generally functions of the parameters (so as to keep
        the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool
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
        Ltermdict : dict
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
            keys of `Ltermdict` to basis matrices.
        """
        Ltermdict = _collections.OrderedDict()
        basisdict = _collections.OrderedDict()
        first_nonempty_basis = None
        constant_basis = None  # the single same Basis used for every factor with a nonempty basis

        for eg in self.factors:
            factor_coeffs = eg.coefficients(return_basis, logscale_nonham)

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

    def coefficients_array(self):
        """
        The weighted coefficients of this error generator in terms of "standard" error generators.

        Constructs a 1D array of all the coefficients returned by :method:`coefficients`,
        weighted so that different error generators can be weighted differently when a
        `errorgen_penalty_factor` is used in an objective function.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to the number of coefficients in the linear
            combination of standard error generators that is this error generator.
        """
        return _np.concatenate([eg.coefficients_array() for eg in self.factors])

    def coefficients_array_deriv_wrt_params(self):
        """
        The jacobian of :method:`coefficients_array` with respect to this error generator's parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(num_coeffs, num_params)` where `num_coeffs` is the number of
            coefficients in the linear combination of standard error generators that is this error
            generator, and `num_params` is this error generator's number of parameters.
        """
        return _np.concatenate([eg.coefficients_array_deriv_wrt_params() for eg in self.factors], axis=0)

    def error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this error generator.

        These error rates pertaining to the *channel* formed by exponentiating this object.

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
        return self.coefficients(return_basis=False, logscale_nonham=True)

    def set_coefficients(self, lindblad_term_dict, action="update", logscale_nonham=False):
        """
        Sets the coefficients of terms in this error generator.

        The dictionary `lindblad_term_dict` has tuple-keys describing the type
        of term and the basis elements used to construct it, e.g. `('H','X')`.

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

        Returns
        -------
        None
        """
        factor_coeffs_list = [eg.coefficients(False, logscale_nonham) for eg in self.factors]
        perfactor_Ltermdicts = [_collections.OrderedDict() for eg in self.factors]
        unused_Lterm_keys = set(lindblad_term_dict.keys())

        #Divide lindblad_term_dict in per-factor Ltermdicts
        for k, val in lindblad_term_dict.items():
            for d, coeffs in zip(perfactor_Ltermdicts, factor_coeffs_list):
                if k in coeffs:
                    d[k] = val; unused_Lterm_keys.remove(k)
                    # only apply a given lindblad_term_dict entry once,
                    # even if it can be applied to multiple factors
                    break

        if len(unused_Lterm_keys) > 0:
            raise KeyError("Invalid L-term descriptor key(s): %s" % str(unused_Lterm_keys))

        #Set the L-term coefficients of each factor separately
        for d, eg in zip(perfactor_Ltermdicts, self.factors):
            eg.set_coefficients(d, action, logscale_nonham)

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in this error generator.

        Cofficients are set so that the contributions of the resulting channel's
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
        self.set_coefficients(lindblad_term_dict, action, logscale_nonham=True)

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per operation parameter.

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
        #TODO: in the furture could do this more cleverly so
        # each factor gets an appropriate wrt_filter instead of
        # doing all filtering at the end

        d2 = self.state_space.dim
        derivMx = _np.zeros((d2**2, self.num_params), 'd')
        for eg, rel_indices in zip(self.factors, self._submember_rpindices):
            factor_deriv = eg.deriv_wrt_params(None)  # do filtering at end
            derivMx[:, rel_indices] += factor_deriv[:, :]

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

        return derivMx

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this error generator with respect to its parameters.

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
        #TODO: in the furture could do this more cleverly so
        # each factor gets an appropriate wrt_filter instead of
        # doing all filtering at the end

        d2 = self.state_space.dim
        nP = self.num_params
        hessianMx = _np.zeros((d2**2, nP, nP), 'd')
        for eg in self.factors:
            factor_hessian = eg.hessian_wrt_params(None, None)  # do filtering at end
            rel_gpindices = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            hessianMx[:, rel_gpindices, rel_gpindices] += factor_hessian[:, :, :]

        if wrt_filter1 is None:
            if wrt_filter2 is None:
                return hessianMx
            else:
                return _np.take(hessianMx, wrt_filter2, axis=2)
        else:
            if wrt_filter2 is None:
                return _np.take(hessianMx, wrt_filter1, axis=1)
            else:
                return _np.take(_np.take(hessianMx, wrt_filter1, axis=1),
                                wrt_filter2, axis=2)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factors

    def append(self, *factors_to_add):
        """
        Add one or more factors to this operator.

        Parameters
        ----------
        *factors_to_add : LinearOperator
            One or multiple factor operators to add on at the *end* (summed
            last) of this operator.

        Returns
        -------
        None
        """
        self.factors.extend(factors_to_add)
        if self._rep is not None:
            self._rep.reinit_factor_reps([op._rep for op in self.factors])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed
            self._parent = None  # mark this object for re-allocation

    def insert(self, insert_at, *factors_to_insert):
        """
        Insert one or more factors into this operator.

        Parameters
        ----------
        insert_at : int
            The index at which to insert `factors_to_insert`.  The factor at this
            index and those after it are shifted back by `len(factors_to_insert)`.

        *factors_to_insert : LinearOperator
            One or multiple factor operators to insert within this operator.

        Returns
        -------
        None
        """
        self.factors[insert_at:insert_at] = list(factors_to_insert)
        if self._rep is not None:
            self._rep.reinit_factor_reps([op._rep for op in self.factors])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed
            self._parent = None  # mark this object for re-allocation

    def remove(self, *factor_indices):
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
        for i in sorted(factor_indices, reverse=True):
            del self.factors[i]
        if self._rep is not None:
            self._rep.reinit_factor_reps([op._rep for op in self.factors])
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
    #    # We need to override this method so that factors have their
    #    # parent reset correctly.
    #    if memo is not None and id(self) in memo: return memo[id(self)]
    #    cls = self.__class__  # so that this method works for derived classes too
    #    copyOfMe = cls([f.copy(parent, memo) for f in self.factors], self._evotype, self.state_space)
    #    return self._copy_gpindices(copyOfMe, parent, memo)

    def to_sparse(self, on_space='minimal'):
        """
        Return this error generator as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if len(self.factors) == 0:
            return _sps.csr_matrix((self.dim, self.dim), dtype='d')
        mx = self.factors[0].to_sparse(on_space)
        for eg in self.factors[1:]:
            mx += eg.to_sparse(on_space)
        return mx

    def to_dense(self, on_space='minimal'):
        """
        Return this error generator as a dense matrix

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
        if len(self.factors) == 0:
            return _np.zeros((self.dim, self.dim), 'd')
        mx = self.factors[0].to_dense(on_space)
        for eg in self.factors[1:]:
            mx += eg.to_dense(on_space)
        return mx

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        assert(self.gpindices is not None), "Must set a ComposedErrorgen's .gpindices before calling parameter_labels"
        vl = _np.empty(self.num_params, dtype=object)
        for eg, factor_local_inds in zip(self.factors, self._submember_rpindices):
            vl[factor_local_inds] = eg.parameter_labels
        return vl

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this error generator.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the error generator parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        assert(self.gpindices is not None), "Must set a ComposedErrorgen's .gpindices before calling to_vector"
        v = _np.empty(self.num_params, 'd')
        for eg, factor_local_inds in zip(self.factors, self._submember_rpindices):
            v[factor_local_inds] = eg.to_vector()
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
        assert(self.gpindices is not None), "Must set a ComposedErrorgen's .gpindices before calling from_vector"
        for eg, factor_local_inds in zip(self.factors, self._submember_rpindices):
            eg.from_vector(v[factor_local_inds], close, dirty_value)
        self.dirty = dirty_value

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this error generator..

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
        assert(order == 0), \
            "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        assert(return_coeff_polys is False)

        #Need to adjust indices b/c in error generators we (currently) expect terms to have local indices
        ret = []
        for eg in self.factors:
            eg_terms = [t.copy() for t in eg.taylor_order_terms(order, max_polynomial_vars, return_coeff_polys)]
            mapvec = _np.ascontiguousarray(_modelmember._decompose_gpindices(
                self.gpindices, _modelmember._compose_gpindices(eg.gpindices, _np.arange(eg.num_params))))
            for t in eg_terms:
                # t.map_indices_inplace(lambda x: tuple(_modelmember._decompose_gpindices(
                #     # map global to *local* indices
                #     self.gpindices, _modelmember._compose_gpindices(eg.gpindices, _np.array(x, _np.int64)))))
                t.mapvec_indices_inplace(mapvec)
            ret.extend(eg_terms)
        return ret
        # return list(_itertools.chain(
        #     *[eg.get_taylor_order_terms(order, max_polynomial_vars, return_coeff_polys) for eg in self.factors]
        # ))

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
        # In this case, since composed error generators are just summed, the total term
        # magnitude is just the sum of the components

        #DEBUG CHECK (can REMOVE)
        #factor_ttms = [eg.get_total_term_magnitude() for eg in self.factors]
        #print("DB: ComposedErrorgen.total_term_magnitude = sum(",factor_ttms,") -- ",
        #      [eg.__class__.__name__ for eg in self.factors])
        #for k,eg in enumerate(self.factors):
        #    sub_egterms = eg.get_taylor_order_terms(0)
        #    sub_mags = [ abs(t.evaluate_coeff(eg.to_vector()).coeff) for t in sub_egterms ]
        #    print(" -> ",k,": total terms mag = ",sum(sub_mags), "(%d)" % len(sub_mags),"\n", sub_mags)
        #    print("     gpindices = ",eg.gpindices)
        #
        #ret = sum(factor_ttms)
        #egterms = self.taylor_order_terms(0)
        #mags = [ abs(t.evaluate_coeff(self.to_vector()).coeff) for t in egterms ]
        #print("ComposedErrgen term mags (should concat above) ",len(egterms),":\n",mags)
        #print("gpindices = ",self.gpindices)
        #print("ComposedErrorgen CHECK = ",sum(mags), " vs ", ret)
        #assert(sum(mags) <= ret+1e-4)

        return sum([eg.total_term_magnitude for eg in self.factors])

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
        ret = _np.zeros(self.num_params, 'd')
        for eg, eg_local_inds in zip(self.factors, self._submember_rpindices):
            ret[eg_local_inds] += eg.total_term_magnitude_deriv
        return ret

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
        for eg in self.factors:
            eg.transform_inplace(s)

    def onenorm_upperbound(self):
        """
        Returns an upper bound on the 1-norm for this error generator (viewed as a matrix).

        Returns
        -------
        float
        """
        # b/c ||A + B|| <= ||A|| + ||B||
        return sum([eg.onenorm_upperbound() for eg in self.factors])

    def __str__(self):
        """ Return string representation """
        s = "Composed error generator of %d factors:\n" % len(self.factors)
        for i, eg in enumerate(self.factors):
            s += "Factor %d:\n" % i
            s += str(eg)
        return s
