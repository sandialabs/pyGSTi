"""
The EmbeddedOp class and supporting functionality.
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
import itertools as _itertools
import collections as _collections

from .linearop import LinearOperator as _LinearOperator
from .denseop import DenseOperatorInterface as _DenseOperatorInterface

from .. import modelmember as _modelmember
from ...objects.basis import EmbeddedBasis as _EmbeddedBasis


class EmbeddedOp(_LinearOperator):
    """
    An operation containing a single lower (or equal) dimensional operation within it.

    An EmbeddedOp acts as the identity on all of its domain except the
    subspace of its contained operation, where it acts as the contained operation does.

    Parameters
    ----------
    state_space_labels : a list of tuples
        This argument specifies the density matrix space upon which this
        operation acts.  Each tuple corresponds to a block of a density matrix
        in the standard basis (and therefore a component of the direct-sum
        density matrix space). Elements of a tuple are user-defined labels
        beginning with "L" (single Level) or "Q" (two-level; Qubit) which
        interpret the d-dimensional state space corresponding to a d x d
        block as a tensor product between qubit and single level systems.
        (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

    target_labels : list of strs
        The labels contained in `state_space_labels` which demarcate the
        portions of the state space acted on by `operation_to_embed` (the
        "contained" operation).

    operation_to_embed : LinearOperator
        The operation object that is to be contained within this operation, and
        that specifies the only non-trivial action of the EmbeddedOp.

    dense_rep : bool, optional
        Whether this operator should be internally represented using a dense
        matrix.  This is expert-level functionality, and you should leave their
        the default value unless you know what you're doing.
    """

    def __init__(self, state_space_labels, target_labels, operation_to_embed, dense_rep=False):
        """
        Initialize an EmbeddedOp object.

        Parameters
        ----------
        state_space_labels : a list of tuples
            This argument specifies the density matrix space upon which this
            operation acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        target_labels : list of strs
            The labels contained in `state_space_labels` which demarcate the
            portions of the state space acted on by `operation_to_embed` (the
            "contained" operation).

        operation_to_embed : LinearOperator
            The operation object that is to be contained within this operation, and
            that specifies the only non-trivial action of the EmbeddedOp.

        dense_rep : bool, optional
            Whether this operator should be internally represented using a dense
            matrix.  This is expert-level functionality, and you should leave their
            the default value unless you know what you're doing.
        """
        from ...models.labeldicts import StateSpaceLabels as _StateSpaceLabels
        self.state_space_labels = _StateSpaceLabels(state_space_labels,
                                                    evotype=operation_to_embed._evotype)
        self.target_labels = target_labels
        self.embedded_op = operation_to_embed
        self.dense_rep = dense_rep
        self._iter_elements_cache = None  # speeds up _iter_matrix_elements significantly

        evotype = operation_to_embed._evotype
        dim = self.state_space_labels.dim

        #Create representation
        if dense_rep:
            rep = evotype.create_dense_rep(dim)
        else:
            rep = evotype.create_embedded_rep(self.state_space_labels, self.target_labels, self.embedded_op._rep)

        _LinearOperator.__init__(self, rep, evotype)
        if self.dense_rep: self._update_denserep()

    def _update_denserep(self):
        """Performs additional update for the case when we use a dense underlying representation."""
        self._rep.base.flags.writeable = True
        self._rep.base[:, :] = self.to_dense()
        self._rep.base.flags.writeable = False

    def __getstate__(self):
        # Don't pickle 'instancemethod' or parent (see modelmember implementation)
        return _modelmember.ModelMember.__getstate__(self)

    def __setstate__(self, d):
        if "dirty" in d:  # backward compat: .dirty was replaced with ._dirty in ModelMember
            d['_dirty'] = d['dirty']; del d['dirty']
        self.__dict__.update(d)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.embedded_op]

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
        self.embedded_op.set_time(t)

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
        # We need to override this method so that embedded operation has its
        # parent reset correctly.
        if memo is not None and id(self) in memo: return memo[id(self)]
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.state_space_labels, self.target_labels,
                       self.embedded_op.copy(parent, memo))
        return self._copy_gpindices(copyOfMe, parent, memo)

    def _iter_matrix_elements_precalc(self):
        divisor = 1; divisors = []
        for l in self.target_labels:
            divisors.append(divisor)
            divisor *= self.state_space_labels.labeldims[l]  # e.g. 4 or 2 for qubits (depending on evotype)

        iTensorProdBlk = [self.state_space_labels.tpb_index[label] for label in self.target_labels][0]
        tensorProdBlkLabels = self.state_space_labels.labels[iTensorProdBlk]
        basisInds = [list(range(self.state_space_labels.labeldims[l])) for l in tensorProdBlkLabels]
        # e.g. [0,1,2,3] for densitymx qubits (I, X, Y, Z) OR [0,1] for statevec qubits (std *complex* basis)

        basisInds_noop = basisInds[:]
        basisInds_noop_blankaction = basisInds[:]
        labelIndices = [tensorProdBlkLabels.index(label) for label in self.target_labels]
        for labelIndex in sorted(labelIndices, reverse=True):
            del basisInds_noop[labelIndex]
            basisInds_noop_blankaction[labelIndex] = [0]

        sorted_bili = sorted(list(enumerate(labelIndices)), key=lambda x: x[1])
        # for inserting target-qubit basis indices into list of noop-qubit indices

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(map(len, basisInds[1:])))))), _np.int64)

        # number of basis elements preceding our block's elements
        blockDims = self.state_space_labels.tpb_dims
        offset = sum([blockDims[i] for i in range(0, iTensorProdBlk)])

        return divisors, multipliers, sorted_bili, basisInds_noop, offset

    def _iter_matrix_elements(self, rel_to_block=False):
        """ Iterates of (op_i,op_j,embedded_op_i,embedded_op_j) tuples giving mapping
            between nonzero elements of operation matrix and elements of the embedded operation matrx """
        if self._iter_elements_cache is not None:
            for item in self._iter_elements_cache:
                yield item
            return

        def _merge_op_and_noop_bases(op_b, noop_b, sorted_bili):
            """
            Merge the Pauli basis indices for the "operation"-parts of the total
            basis contained in op_b (i.e. of the components of the tensor
            product space that are operated on) and the "noop"-parts contained
            in noop_b.  Thus, len(op_b) + len(noop_b) == len(basisInds), and
            this function merges together basis indices for the operated-on and
            not-operated-on tensor product components.
            Note: return value always have length == len(basisInds) == number
            of components
            """
            ret = list(noop_b[:])  # start with noop part...
            for bi, li in sorted_bili:
                ret.insert(li, op_b[bi])  # ... and insert operation parts at proper points
            return ret

        def _decomp_op_index(indx, divisors):
            """ Decompose index of a Pauli-product matrix into indices of each
                Pauli in the product """
            ret = []
            for d in reversed(divisors):
                ret.append(indx // d)
                indx = indx % d
            return ret

        divisors, multipliers, sorted_bili, basisInds_noop, nonrel_offset = self._iter_matrix_elements_precalc()
        offset = 0 if rel_to_block else nonrel_offset

        #Begin iteration loop
        self._iter_elements_cache = []
        for op_i in range(self.embedded_op.dim):     # rows ~ "output" of the operation map
            for op_j in range(self.embedded_op.dim):  # cols ~ "input"  of the operation map
                op_b1 = _decomp_op_index(op_i, divisors)  # op_b? are lists of dm basis indices, one index per
                # tensor product component that the operation operates on (2 components for a 2-qubit operation)
                op_b2 = _decomp_op_index(op_j, divisors)

                # loop over all state configurations we don't operate on
                for b_noop in _itertools.product(*basisInds_noop):
                    # - so really a loop over diagonal dm elements
                    # using same b_noop for in and out says we're acting
                    b_out = _merge_op_and_noop_bases(op_b1, b_noop, sorted_bili)
                    # as the identity on the no-op state space
                    b_in = _merge_op_and_noop_bases(op_b2, b_noop, sorted_bili)
                    # index of output dm basis el within vec(tensor block basis)
                    out_vec_index = _np.dot(multipliers, tuple(b_out))
                    # index of input dm basis el within vec(tensor block basis)
                    in_vec_index = _np.dot(multipliers, tuple(b_in))

                    item = (out_vec_index + offset, in_vec_index + offset, op_i, op_j)
                    self._iter_elements_cache.append(item)
                    yield item

    def to_sparse(self):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        embedded_sparse = self.embedded_op.to_sparse().tolil()
        finalOp = _sps.identity(self.dim, embedded_sparse.dtype, format='lil')

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements():
            finalOp[i, j] = embedded_sparse[gi, gj]
        return finalOp.tocsr()

    def to_dense(self):
        """
        Return the operation as a dense matrix

        Returns
        -------
        numpy.ndarray
        """

        #FUTURE: maybe here or in a new "tosymplectic" method, could
        # create an embeded clifford symplectic rep as follows (when
        # evotype == "stabilizer"):
        #def tosymplectic(self):
        #    #Embed operation's symplectic rep in larger "full" symplectic rep
        #    #Note: (qubit) labels are in first (and only) tensor-product-block
        #    qubitLabels = self.state_space_labels.labels[0]
        #    smatrix, svector = _symp.embed_clifford(self.embedded_op.smatrix,
        #                                            self.embedded_op.svector,
        #                                            self.qubit_indices,len(qubitLabels))

        embedded_dense = self.embedded_op.to_dense()
        # operates on entire state space (direct sum of tensor prod. blocks)
        finalOp = _np.identity(self.dim, embedded_dense.dtype)

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements():
            finalOp[i, j] = embedded_dense[gi, gj]
        return finalOp

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.embedded_op.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.embedded_op.num_params

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        return self.embedded_op.to_vector()

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
        assert(len(v) == self.num_params)
        self.embedded_op.from_vector(v, close, dirty_value)
        if self.dense_rep: self._update_denserep()
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
        # Note: this function exploits knowledge of EmbeddedOp internals!!
        embedded_deriv = self.embedded_op.deriv_wrt_params(wrt_filter)
        derivMx = _np.zeros((self.dim**2, embedded_deriv.shape[1]), embedded_deriv.dtype)
        M = self.embedded_op.dim

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements():
            derivMx[i * self.dim + j, :] = embedded_deriv[gi * M + gj, :]  # fill row of jacobian
        return derivMx  # Note: wrt_filter has already been applied above

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
        #Reduce labeldims b/c now working on *state-space* instead of density mx:
        sslbls = self.state_space_labels.copy()
        sslbls.reduce_dims_densitymx_to_state_inplace()
        if return_coeff_polys:
            terms, coeffs = self.embedded_op.taylor_order_terms(order, max_polynomial_vars, True)
            embedded_terms = [t.embed(sslbls, self.target_labels) for t in terms]
            return embedded_terms, coeffs
        else:
            return [t.embed(sslbls, self.target_labels)
                    for t in self.embedded_op.taylor_order_terms(order, max_polynomial_vars, False)]

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
        sslbls = self.state_space_labels.copy()
        sslbls.reduce_dims_densitymx_to_state_inplace()
        return [t.embed(sslbls, self.target_labels)
                for t in self.embedded_op.taylor_order_terms_above_mag(order, max_polynomial_vars, min_term_mag)]

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
        # In this case, since the coeffs of the terms of an EmbeddedOp are the same as those
        # of the operator being embedded, the total term magnitude is the same:

        #DEBUG TODO REMOVE
        #print("DB: Embedded.total_term_magnitude = ",self.embedded_op.get_total_term_magnitude()," -- ",
        #   self.embedded_op.__class__.__name__)
        #ret = self.embedded_op.get_total_term_magnitude()
        #egterms = self.taylor_order_terms(0)
        #mags = [ abs(t.evaluate_coeff(self.to_vector()).coeff) for t in egterms ]
        #print("EmbeddedErrorgen CHECK = ",sum(mags), " vs ", ret)
        #assert(sum(mags) <= ret+1e-4)

        return self.embedded_op.total_term_magnitude

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
        return self.embedded_op.total_term_magnitude_deriv

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
        # I think we could do this but extracting the approprate parts of the
        # s and Sinv matrices... but haven't needed it yet.
        raise NotImplementedError("Cannot transform an EmbeddedDenseOp yet...")

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
        #*** Note: this function is nearly identitcal to EmbeddedErrorgen.coefficients() ***
        embedded_coeffs = self.embedded_op.errorgen_coefficients(return_basis, logscale_nonham)
        embedded_Ltermdict = _collections.OrderedDict()

        if return_basis:
            # embed basis
            Ltermdict, basis = embedded_coeffs
            embedded_basis = _EmbeddedBasis(basis, self.state_space_labels, self.target_labels)
            bel_map = {lbl: embedded_lbl for lbl, embedded_lbl in zip(basis.labels, embedded_basis.labels)}

            #go through and embed Ltermdict labels
            for k, val in Ltermdict.items():
                embedded_key = (k[0],) + tuple([bel_map[x] for x in k[1:]])
                embedded_Ltermdict[embedded_key] = val
            return embedded_Ltermdict, embedded_basis
        else:
            #go through and embed Ltermdict labels
            Ltermdict = embedded_coeffs
            for k, val in Ltermdict.items():
                embedded_key = (k[0],) + tuple([_EmbeddedBasis.embed_label(x, self.target_labels) for x in k[1:]])
                embedded_Ltermdict[embedded_key] = val
            return embedded_Ltermdict

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
        return self.embedded_op.errorgen_coefficients_array()

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
        return self.embedded_op.errorgen_coefficients_array_deriv_wrt_params()

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

    def depolarize(self, amount):
        """
        Depolarize this operation by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the operation such that the resulting operation matrix is depolarized.  If
        such an update cannot be done (because the operation parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the operation. In standard
            bases, depolarization corresponds to multiplying the operation matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        self.embedded_op.depolarize(amount)
        if self.dense_rep: self._update_denserep()

    def rotate(self, amount, mx_basis="gm"):
        """
        Rotate this operation by the given `amount`.

        Generally, the rotate function updates the *parameters* of
        the operation such that the resulting operation matrix is rotated.  If
        such an update cannot be done (because the operation parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The operation's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        self.embedded_op.rotate(amount, mx_basis)
        if self.dense_rep: self._update_denserep()

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return self.embedded_op.has_nonzero_hessian()

    def __str__(self):
        """ Return string representation """
        s = "Embedded operation with full dimension %d and state space %s\n" % (self.dim, self.state_space_labels)
        s += " that embeds the following %d-dimensional operation into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.target_labels))
        s += str(self.embedded_op)
        return s


class EmbeddedDenseOp(EmbeddedOp, _DenseOperatorInterface):
    """
    An operation containing a single lower (or equal) dimensional operation within it.

    An EmbeddedDenseOp acts as the identity on all of its domain except the
    subspace of its contained operation, where it acts as the contained operation does.

    Parameters
    ----------
    state_space_labels : a list of tuples
        This argument specifies the density matrix space upon which this
        operation acts.  Each tuple corresponds to a block of a density matrix
        in the standard basis (and therefore a component of the direct-sum
        density matrix space). Elements of a tuple are user-defined labels
        beginning with "L" (single Level) or "Q" (two-level; Qubit) which
        interpret the d-dimensional state space corresponding to a d x d
        block as a tensor product between qubit and single level systems.
        (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

    target_labels : list of strs
        The labels contained in `state_space_labels` which demarcate the
        portions of the state space acted on by `operation_to_embed` (the
        "contained" operation).

    operation_to_embed : DenseOperator
        The operation object that is to be contained within this operation, and
        that specifies the only non-trivial action of the EmbeddedDenseOp.
    """

    def __init__(self, state_space_labels, target_labels, operation_to_embed):
        """
        Initialize a EmbeddedDenseOp object.

        Parameters
        ----------
        state_space_labels : a list of tuples
            This argument specifies the density matrix space upon which this
            operation acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        target_labels : list of strs
            The labels contained in `state_space_labels` which demarcate the
            portions of the state space acted on by `operation_to_embed` (the
            "contained" operation).

        operation_to_embed : DenseOperator
            The operation object that is to be contained within this operation, and
            that specifies the only non-trivial action of the EmbeddedDenseOp.
        """
        EmbeddedOp.__init__(self, state_space_labels, target_labels,
                            operation_to_embed, dense_rep=True)
        _DenseOperatorInterface.__init__(self)

    @property
    def parameter_labels(self):  # Needed because method resolution finds __getattr__ before base class property
        return EmbeddedOp.parameter_labels.fget(self)
