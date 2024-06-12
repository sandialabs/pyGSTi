"""
The State class and supporting functionality.
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

from pygsti.baseobjs.opcalc import bulk_eval_compact_polynomials_complex as _bulk_eval_compact_polynomials_complex
from pygsti.modelmembers import modelmember as _modelmember
from pygsti.baseobjs import _compatibility as _compat
from pygsti.tools import optools as _ot


class State(_modelmember.ModelMember):
    """
    TODO: update docstring
    A parameterized state preparation OR POVM effect vector (operator).

    This class is the  common base class for all specific
    parameterizations of a state vector.

    Parameters
    ----------
    rep : object
        A representation object containing the core data for this spam vector.

    evotype : Evotype
        The evolution type.

    Attributes
    ----------
    size : int
        The number of independent elements in this state vector (when viewed as a dense array).
    """

    def __init__(self, rep, evotype):
        """ Initialize a new state Vector """
        super(State, self).__init__(rep.state_space, evotype)
        self._rep = rep

    @property
    def dim(self):
        """
        Return the dimension of this state (when viewed as a dense array)

        Returns
        -------
        int
        """
        return self.state_space.dim

    @property
    def hilbert_schmidt_size(self):
        """
        Return the number of independent elements in this state as a dense Hilbert-Schmidt super-ket.

        Returns
        -------
        int
        """
        return self._rep.state_space.dim

    def set_dense(self, vec):
        """
        Set the dense-vector value of this state vector.

        Attempts to modify this state vector's parameters so that the raw
        state vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or State
            A numpy array representing a state vector, or a State object.

        Returns
        -------
        None
        """
        raise ValueError("Cannot set the value of a %s directly!" % self.__class__.__name__)

    def set_time(self, t):
        """
        Sets the current time for a time-dependent operator.

        For time-independent operators (the default), this function does absolutely nothing.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        None
        """
        pass

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
        raise NotImplementedError("to_dense(...) not implemented for %s objects!" % self.__class__.__name__)

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
        raise NotImplementedError("taylor_order_terms(...) not implemented for %s objects!" %
                                  self.__class__.__name__)

    def highmagnitude_terms(self, min_term_mag, force_firstorder=True, max_taylor_order=3, max_polynomial_vars=100):
        """
        Get terms with magnitude above `min_term_mag`.

        Get the terms (from a Taylor expansion of this state vector) that have
        magnitude above `min_term_mag` (the magnitude of a term is taken to
        be the absolute value of its coefficient), considering only those
        terms up to some maximum Taylor expansion order, `max_taylor_order`.

        Note that this function also *sets* the magnitudes of the returned
        terms (by calling `term.set_magnitude(...)`) based on the current
        values of this state vector's parameters.  This is an essential step
        to using these terms in pruned-path-integral calculations later on.

        Parameters
        ----------
        min_term_mag : float
            the threshold for term magnitudes: only terms with magnitudes above
            this value are returned.

        force_firstorder : bool, optional
            if True, then always return all the first-order Taylor-series terms,
            even if they have magnitudes smaller than `min_term_mag`.  This
            behavior is needed for using GST with pruned-term calculations, as
            we may begin with a guess model that has no error (all terms have
            zero magnitude!) and still need to compute a meaningful jacobian at
            this point.

        max_taylor_order : int, optional
            the maximum Taylor-order to consider when checking whether term-
            magnitudes exceed `min_term_mag`.

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        Returns
        -------
        highmag_terms : list
            A list of the high-magnitude terms that were found.  These
            terms are *sorted* in descending order by term-magnitude.

        first_order_indices : list
            A list of the indices into `highmag_terms` that mark which
            of these terms are first-order Taylor terms (useful when
            we're forcing these terms to always be present).
        """
        #NOTE: SAME as for LinearOperator class -- TODO consolidate in FUTURE
        #print("DB: state get_high_magnitude_terms")
        v = self.to_vector()
        taylor_order = 0
        terms = []; last_len = -1; first_order_magmax = 1.0
        while len(terms) > last_len:  # while we keep adding something
            if taylor_order > 1 and first_order_magmax**taylor_order < min_term_mag:
                break  # there's no way any terms at this order reach min_term_mag - exit now!

            MAX_CACHED_TERM_ORDER = 1
            if taylor_order <= MAX_CACHED_TERM_ORDER:
                #print("order ",taylor_order," : ",len(terms), "terms")
                terms_at_order, cpolys = self.taylor_order_terms(taylor_order, max_polynomial_vars, True)
                coeffs = _bulk_eval_compact_polynomials_complex(
                    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
                mags = _np.abs(coeffs)
                last_len = len(terms)
                #OLD: terms_at_order = [ t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order) ]

                if taylor_order == 1:
                    #OLD: first_order_magmax = max([t.magnitude for t in terms_at_order])
                    first_order_magmax = max(mags)

                    if force_firstorder:
                        terms.extend([(taylor_order, t.copy_with_magnitude(mag))
                                      for coeff, mag, t in zip(coeffs, mags, terms_at_order)])
                    else:
                        for mag, t in zip(mags, terms_at_order):
                            if mag >= min_term_mag:
                                terms.append((taylor_order, t.copy_with_magnitude(mag)))
                else:
                    for mag, t in zip(mags, terms_at_order):
                        if mag >= min_term_mag:
                            terms.append((taylor_order, t.copy_with_magnitude(mag)))

            else:
                eff_min_term_mag = 0.0 if (taylor_order == 1 and force_firstorder) else min_term_mag
                terms.extend([(taylor_order, t) for t in
                              self.taylor_order_terms_above_mag(taylor_order,
                                                                max_polynomial_vars, eff_min_term_mag)])

            taylor_order += 1
            if taylor_order > max_taylor_order: break

        #Sort terms based on magnitude
        sorted_terms = sorted(terms, key=lambda t: t[1].magnitude, reverse=True)
        first_order_indices = [i for i, t in enumerate(sorted_terms) if t[0] == 1]
        return [t[1] for t in sorted_terms], first_order_indices

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
        v = self.to_vector()
        terms_at_order, cpolys = self.taylor_order_terms(order, max_polynomial_vars, True)
        coeffs = _bulk_eval_compact_polynomials_complex(
            cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        terms_at_order = [t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order)]
        return [t for t in terms_at_order if t.magnitude >= min_term_mag]

    def frobeniusdist_squared(self, other_spam_vec, transform=None,
                              inv_transform=None):
        """
        Return the squared frobenius difference between this operation and `other_spam_vec`.

        Optionally transforms this vector first using `transform` and
        `inv_transform`.

        Parameters
        ----------
        other_spam_vec : State
            The other spam vector

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `tranform`.

        Returns
        -------
        float
        """
        vec = self.to_dense(on_space='minimal')
        if inv_transform is None:
            return _ot.frobeniusdist_squared(vec, other_spam_vec.to_dense(on_space='minimal'))
        else:
            return _ot.frobeniusdist_squared(_np.dot(inv_transform, vec),
                                             other_spam_vec.to_dense(on_space='minimal'))

    def residuals(self, other_spam_vec, transform=None, inv_transform=None):
        """
        Return a vector of residuals between this spam vector and `other_spam_vec`.

        Optionally transforms this vector first using `transform` and
        `inv_transform`.

        Parameters
        ----------
        other_spam_vec : State
            The other spam vector

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `tranform`.

        Returns
        -------
        float
        """
        vec = self.to_dense(on_space='minimal')
        if inv_transform is not None:
            vec = inv_transform @ vec
        return (vec - other_spam_vec.to_dense(on_space='minimal')).ravel()


    def transform_inplace(self, s):
        """
        Update state preparation (column) vector V as inv(s) * V.

        Note that this is equivalent to state preparation vectors getting
        mapped: `rho -> inv(s) * rho`.

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
        Si = s.transform_matrix_inverse
        self.set_dense(_np.dot(Si, self.to_dense(on_space='minimal')))

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
            equal to one less than the dimension of the gate. All but the
            first element of the spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        if isinstance(amount, float) or _compat.isint(amount):
            D = _np.diag([1] + [1 - amount] * (self.dim - 1))
        else:
            assert(len(amount) == self.dim - 1)
            D = _np.diag([1] + list(1.0 - _np.array(amount, 'd')))
        self.set_dense(_np.dot(D, self.to_dense(on_space='minimal')))

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 0  # no parameters

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd')  # no parameters

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
        assert(len(v) == 0)  # should be no parameters, and nothing to do

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
        dtype = complex if self._evotype == 'statevec' else 'd'
        derivMx = _np.zeros((self.dim, 0), dtype)
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
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.num_params > 0

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
        if not self.has_nonzero_hessian():
            return _np.zeros(self.size, self.num_params, self.num_params)

        # FUTURE: create a finite differencing hessian method?
        raise NotImplementedError("hessian_wrt_params(...) is not implemented for %s objects" % self.__class__.__name__)

    #Note: no __str__ fn

    @staticmethod
    def _to_vector(v):
        """
        Static method that converts a vector-like object to a 2D numpy dim x 1 column array.

        Parameters
        ----------
        v : array_like
            Array-like object to convert to a numpy array.

        Returns
        -------
        numpy array
        """
        if isinstance(v, State):
            vector = v.to_dense(on_space='minimal').copy()
            vector.shape = (vector.size, 1)
        elif isinstance(v, _np.ndarray):
            vector = v.copy()
            if len(vector.shape) == 1:  # convert (N,) shape vecs to (N,1)
                vector.shape = (vector.size, 1)
        else:
            try:
                len(v)
                # XXX this is an abuse of exception handling
            except:
                raise ValueError("%s doesn't look like an array/list" % v)
            try:
                d2s = [len(row) for row in v]
            except TypeError:  # thrown if len(row) fails because no 2nd dim
                d2s = None

            if d2s is not None:
                if any([len(row) != 1 for row in v]):
                    raise ValueError("%s is 2-dimensional but 2nd dim != 1" % v)

                typ = 'd' if _np.all(_np.isreal(v)) else 'complex'
                try:
                    vector = _np.array(v, typ)  # vec is already a 2-D column vector
                except TypeError:
                    raise ValueError("%s doesn't look like an array/list" % v)
            else:
                typ = 'd' if _np.all(_np.isreal(v)) else 'complex'
                vector = _np.array(v, typ)[:, None]  # make into a 2-D column vec

        assert(len(vector.shape) == 2 and vector.shape[1] == 1)
        return vector.ravel()  # HACK for convention change -> (N,) instead of (N,1)
