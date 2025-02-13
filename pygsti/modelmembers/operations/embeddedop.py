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

import collections as _collections
import itertools as _itertools

import numpy as _np
import scipy.sparse as _sps

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers import modelmember as _modelmember
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GlobalElementaryErrorgenLabel, LocalElementaryErrorgenLabel as _LocalElementaryErrorgenLabel


class EmbeddedOp(_LinearOperator):
    """
    An operation containing a single lower (or equal) dimensional operation within it.

    An EmbeddedOp acts as the identity on all of its domain except the
    subspace of its contained operation, where it acts as the contained operation does.

    Parameters
    ----------
    state_space : StateSpace
        Specifies the density matrix space upon which this operation acts.

    target_labels : list of strs
        The labels contained in `state_space` which demarcate the
        portions of the state space acted on by `operation_to_embed` (the
        "contained" operation).

    operation_to_embed : LinearOperator
        The operation object that is to be contained within this operation, and
        that specifies the only non-trivial action of the EmbeddedOp.
    """

    def __init__(self, state_space, target_labels, operation_to_embed, allocated_to_parent=None):
        self.target_labels = tuple(target_labels) if (target_labels is not None) else None
        self.embedded_op = operation_to_embed
        self._iter_elements_cache = {"Hilbert": None, "HilbertSchmidt": None}  # speeds up _iter_matrix_elements

        assert(_StateSpace.cast(state_space).contains_labels(target_labels)), \
            "`target_labels` (%s) not found in `state_space` (%s)" % (str(target_labels), str(state_space))
        assert(self.embedded_op.state_space.num_tensor_product_blocks == 1), \
            "EmbeddedOp objects can only embed operations whose state spaces contain just a single tensor product block"
        assert(len(self.embedded_op.state_space.sole_tensor_product_block_labels) == len(target_labels)), \
            "Embedded operation's state space has a different number of components than the number of target labels!"

        evotype = operation_to_embed._evotype
        rep = self._create_rep_object(evotype, state_space)

        self._cached_embedded_errorgen_labels_global = None
        self._cached_embedded_errorgen_labels_local = None
        self._cached_embedded_label_identity_label = None

        _LinearOperator.__init__(self, rep, evotype)
        self.init_gpindices(allocated_to_parent)  # initialize our gpindices based on sub-members
        if self._rep_type == 'dense': self._update_denserep()

    def _create_rep_object(self, evotype, state_space):
        #Create representation object
        rep_type_order = ('dense', 'embedded') if evotype.prefer_dense_reps else ('embedded', 'dense')
        rep = None
        for rep_type in rep_type_order:
            try:
                if rep_type == 'embedded':
                    rep = evotype.create_embedded_rep(state_space, self.target_labels, self.embedded_op._rep)
                elif rep_type == 'dense':
                    # UNSPECIFIED BASIS -- we set basis=None below, which may not work with all evotypes,
                    #  and should be replaced with the basis of contained ops (if any) once we establish
                    #  a common .basis or ._basis attribute of representations (which could still be None)
                    rep = evotype.create_dense_superop_rep(None, None, state_space)
                else:
                    assert(False), "Logic error!"

                self._rep_type = rep_type
                break

            except AttributeError:
                pass  # just go to the next rep_type

        if rep is None:
            raise ValueError("Unable to construct representation with evotype: %s" % str(evotype))
        return rep

    def _update_denserep(self):
        """Performs additional update for the case when we use a dense underlying representation."""
        self._rep.base.flags.writeable = True
        self._rep.base[:, :] = self.to_dense(on_space='minimal')
        self._rep.base.flags.writeable = False

    def _update_submember_state_spaces(self, old_parent_state_space, new_parent_state_space):
        self._rep = self._create_rep_object(self.evotype, new_parent_state_space)  # update representation
        # No need to update submembers

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

    def _iter_matrix_elements_precalc(self, on_space):
        divisor = 1; divisors = []
        for l in self.target_labels:
            divisors.append(divisor)
            dim = self.state_space.label_udimension(l) if on_space == "Hilbert" \
                else self.state_space.label_dimension(l)   # e.g. 4 or 2 for qubits (depending on on_space)
            divisor *= dim

        iTensorProdBlk = [self.state_space.label_tensor_product_block_index(label) for label in self.target_labels][0]
        tensorProdBlkLabels = self.state_space.tensor_product_block_labels(iTensorProdBlk)
        if on_space == "Hilbert":
            basisInds = [list(range(self.state_space.label_udimension(l))) for l in tensorProdBlkLabels]
        else:
            basisInds = [list(range(self.state_space.label_dimension(l))) for l in tensorProdBlkLabels]
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
        if on_space == "Hilbert":
            blockDims = [_np.prod(tpb_dims) for tpb_dims in self.state_space.tensor_product_blocks_udimensions]
        else:
            blockDims = [_np.prod(tpb_dims) for tpb_dims in self.state_space.tensor_product_blocks_dimensions]
        offset = sum(blockDims[0:iTensorProdBlk])

        return divisors, multipliers, sorted_bili, basisInds_noop, offset

    def _iter_matrix_elements(self, on_space, rel_to_block=False):
        """ Iterates of (op_i,op_j,embedded_op_i,embedded_op_j) tuples giving mapping
            between nonzero elements of operation matrix and elements of the embedded operation matrix """
        if self._iter_elements_cache[on_space] is not None:
            for item in self._iter_elements_cache[on_space]:
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

        divisors, multipliers, sorted_bili, basisInds_noop, nonrel_offset = \
            self._iter_matrix_elements_precalc(on_space)
        offset = 0 if rel_to_block else nonrel_offset

        #Begin iteration loop
        self._iter_elements_cache[on_space] = []
        embedded_dim = self.embedded_op.state_space.udim if on_space == "Hilbert" else self.embedded_op.state_space.dim
        for op_i in range(embedded_dim):     # rows ~ "output" of the operation map
            for op_j in range(embedded_dim):  # cols ~ "input"  of the operation map
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
                    self._iter_elements_cache[on_space].append(item)
                    yield item

    def to_sparse(self, on_space='minimal'):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        embedded_sparse = self.embedded_op.to_sparse(on_space).tolil()
        if on_space == 'minimal':  # resolve 'minimal' based on embedded rep type
            on_space = 'Hilbert' if embedded_sparse.shape[0] == self.embedded_op.state_space.udim \
                else 'HilbertSchmidt'

        finalOp = _sps.identity(self.state_space.udim if (on_space == 'Hilbert') else self.state_space.dim,
                                embedded_sparse.dtype, format='lil')

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements(on_space):
            finalOp[i, j] = embedded_sparse[gi, gj]
        return finalOp.tocsr()

    def to_dense(self, on_space='minimal'):
        """
        Return the operation as a dense matrix

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

        #FUTURE: maybe here or in a new "tosymplectic" method, could
        # create an embeded clifford symplectic rep as follows (when
        # evotype == "stabilizer"):
        #def tosymplectic(self):
        #    #Embed operation's symplectic rep in larger "full" symplectic rep
        #    #Note: (qubit) labels are in first (and only) tensor-product-block
        #    qubitLabels = self.state_space.sole_tensor_product_block_labels
        #    smatrix, svector = _symp.embed_clifford(self.embedded_op.smatrix,
        #                                            self.embedded_op.svector,
        #                                            self.qubit_indices,len(qubitLabels))

        embedded_dense = self.embedded_op.to_dense(on_space)
        if on_space == 'minimal':  # resolve 'minimal' based on embedded rep type
            on_space = 'Hilbert' if embedded_dense.shape[0] == self.embedded_op.state_space.udim else 'HilbertSchmidt'

        # operates on entire state space (direct sum of tensor prod. blocks)
        finalOp = _np.identity(self.state_space.udim if (on_space == 'Hilbert') else self.state_space.dim,
                               embedded_dense.dtype)

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements(on_space):
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
        # Note: this function exploits knowledge of EmbeddedOp internals!!
        embedded_deriv = self.embedded_op.deriv_wrt_params(wrt_filter)

        # resolve on_space as if it were 'minimal', based on embedded rep type
        on_space = 'Hilbert' if embedded_deriv.shape[0] == self.embedded_op.state_space.udim else 'HilbertSchmidt'
        dim = self.state_space.udim if (on_space == 'Hilbert') else self.state_space.dim

        derivMx = _np.zeros((dim**2, embedded_deriv.shape[1]), embedded_deriv.dtype)
        M = self.embedded_op.state_space.udim if (on_space == 'Hilbert') else self.embedded_op.state_space.dim
        assert(M**2 == embedded_deriv.shape[0]), \
            "Mismatch between embedded gate's state space dim/udim and it's deriv_wrt_params value"

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements(on_space):
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
            output of :meth:`Polynomial.compact`.
        """
        #Reduce labeldims b/c now working on *state-space* instead of density mx:
        sslbls = self.state_space.copy()
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
        It calls :meth:`taylor_order_terms` internally, so that all the terms at order `order`
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
        sslbls = self.state_space.copy()
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

        #DEBUG CHECK
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
        raise NotImplementedError("Cannot transform an EmbeddedOp yet...")

    def errorgen_coefficients(self, return_basis=False, logscale_nonham=False, label_type='global', identity_label='I'):
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
            This is the value returned by :meth:`error_rates`.

        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.
        
        identity_label : str, optional (default 'I')
            An optional string specifying the basis element label for the
            identity. Used when label_type is 'local' to allow for embedding
            local basis element labels into the appropriate higher dimensional
            space. Only change when using a basis for which 'I' does not denote
            the identity.

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
        #*** Note: this function is nearly identical to EmbeddedErrorgen.coefficients() ***
        coeffs_to_embed = self.embedded_op.errorgen_coefficients(return_basis, logscale_nonham, label_type)
        #print(f'{embedded_coeffs=}')
        
        if coeffs_to_embed:
            embedded_labels = self.errorgen_coefficient_labels(label_type=label_type, identity_label=identity_label)
            embedded_coeffs = {lbl:val for lbl, val in zip(embedded_labels, coeffs_to_embed.values())}
            #first_coeff_lbl = next(iter(coeffs_to_embed))
            #if isinstance(first_coeff_lbl, _GlobalElementaryErrorgenLabel):
#        if self.target_labels != self.embedded_op.state_space.sole_tensor_product_block_labels:
                #mapdict = {loc: tgt for loc, tgt in zip(self.embedded_op.state_space.sole_tensor_product_block_labels,
                #                                        self.target_labels)}
                #embedded_coeffs = {k.map_state_space_labels(mapdict): v for k, v in coeffs_to_embed.items()}
            #elif isinstance(first_coeff_lbl, _LocalElementaryErrorgenLabel):
            #    embedded_labels = self.errorgen_coefficient_labels()
            #    #use different embedding scheme for local labels
            #    base_label = [identity_label for _ in range(self.state_space.num_qudits)]
            #    embedded_labels = []
            #    for lbl in coeff_lbls_to_embed:
            #        new_bels = []
            #        for bel in lbl.basis_element_labels:
            #            base_label = [identity_label for _ in range(self.state_space.num_qudits)]
            #            for target, pauli in zip(self.target_labels, bel):
            #                base_label[target] = pauli
            #            new_bels.append(''.join(base_label))
            #        embedded_labels.append(_LocalElementaryErrorgenLabel(lbl.errorgen_type, tuple(new_bels)))
            #    embedded_coeffs = {lbl:val for lbl, val in zip(embedded_labels, coeffs_to_embed.values())}
            #else:
            #    raise ValueError(f'Invalid error generator label type {first_coeff_lbl}')
        else:
            embedded_coeffs = dict()

        return embedded_coeffs

    def errorgen_coefficient_labels(self, label_type='global', identity_label='I'):
        """
        The elementary error-generator labels corresponding to the elements of :meth:`errorgen_coefficients_array`.

        Parameters
        ----------
        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.

        identity_label : str, optional (default 'I')
            An optional string specifying the basis element label for the
            identity. Used when label_type is 'local' to allow for embedding
            local basis element labels into the appropriate higher dimensional
            space. Only change when using a basis for which 'I' does not denote
            the identity.

        Returns
        -------
        tuple
            A tuple of (<type>, <basisEl1> [,<basisEl2]) elements identifying the elementary error
            generators of this gate.
        """
        if label_type=='global' and self._cached_embedded_errorgen_labels_global is not None:
            return self._cached_embedded_errorgen_labels_global
        elif label_type=='local' and self._cached_embedded_errorgen_labels_local is not None and self._cached_embedded_label_identity_label==identity_label:
            return self._cached_embedded_errorgen_labels_local

        labels_to_embed = self.embedded_op.errorgen_coefficient_labels(label_type)
        #print(f'{embedded_labels=}')
    
        #if self.target_labels != self.embedded_op.state_space.sole_tensor_product_block_labels:
        #print(f'{self.target_labels=}')
        #print(f'{self.embedded_op.state_space.sole_tensor_product_block_labels=}')
        if len(labels_to_embed)>0:
            if isinstance(labels_to_embed[0], _GlobalElementaryErrorgenLabel):
                mapdict = {loc: tgt for loc, tgt in zip(self.embedded_op.state_space.sole_tensor_product_block_labels,
                                                        self.target_labels)}
                embedded_labels = tuple([k.map_state_space_labels(mapdict) for k in labels_to_embed])
                self._cached_embedded_errorgen_labels_global = embedded_labels
            elif isinstance(labels_to_embed[0], _LocalElementaryErrorgenLabel):
                #use different embedding scheme for local labels
                embedded_labels = []
                base_label = [identity_label for _ in range(self.state_space.num_qudits)]
                for lbl in labels_to_embed:
                    new_bels = []
                    for bel in lbl.basis_element_labels:
                        base_label = [identity_label for _ in range(self.state_space.num_qudits)]
                        for target, pauli in zip(self.target_labels, bel):
                            base_label[target] = pauli
                        new_bels.append(''.join(base_label))
                    embedded_labels.append(_LocalElementaryErrorgenLabel(lbl.errorgen_type, tuple(new_bels)))
                embedded_labels = tuple(embedded_labels)
                self._cached_embedded_errorgen_labels_local = embedded_labels
                self._cached_embedded_label_identity_label = identity_label
            else:
                raise ValueError(f'Invalid error generator label type {labels_to_embed[0]}')
        else:
            embedded_labels = tuple()

        return embedded_labels

    def errorgen_coefficients_array(self):
        """
        The weighted coefficients of this operation's error generator in terms of "standard" error generators.

        Constructs a 1D array of all the coefficients returned by :meth:`errorgen_coefficients`,
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
        The jacobian of :meth:`errogen_coefficients_array` with respect to this operation's parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(num_coeffs, num_params)` where `num_coeffs` is the number of
            coefficients of this operation's error generator and `num_params` is this operation's
            number of parameters.
        """
        return self.embedded_op.errorgen_coefficients_array_deriv_wrt_params()

    def error_rates(self, label_type='global', identity_label='I'):
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

        Parameters
        ----------
        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.

        identity_label : str, optional (default 'I')
            An optional string specifying the basis element label for the
            identity. Used when label_type is 'local' to allow for embedding
            local basis element labels into the appropriate higher dimensional
            space. Only change when using a basis for which 'I' does not denote
            the identity.

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
        return self.errorgen_coefficients(return_basis=False, logscale_nonham=True, label_type=label_type, identity_label=identity_label)

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
            "equivalent" depolarizing channel, see :meth:`errorgen_coefficients`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :meth:`set_error_rates`.

        truncate : bool, optional
            Whether to allow adjustment of the errogen coefficients in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given coefficients
            cannot be set as specified.

        Returns
        -------
        None
        """
        #determine is we need to unembed the error generator labels in lindblad_term_dict.
        if lindblad_term_dict: 
            first_coeff_lbl = next(iter(lindblad_term_dict))
            if isinstance(first_coeff_lbl, _GlobalElementaryErrorgenLabel):
                if self.target_labels != self.embedded_op.state_space.sole_tensor_product_block_labels:
                    mapdict = {tgt: loc for loc, tgt in zip(self.embedded_op.state_space.sole_tensor_product_block_labels,
                                                            self.target_labels)}
                    unembedded_coeffs = {k.map_state_space_labels(mapdict): v for k, v in lindblad_term_dict.items()}
                else:
                    unembedded_coeffs = lindblad_term_dict
            elif isinstance(first_coeff_lbl, _LocalElementaryErrorgenLabel):
                #if the length of the basis element labels are the same as the length of this
                #embedded op's target labels then assume those are associated.
                if len(first_coeff_lbl.basis_element_labels[0]) == len(self.target_labels):
                    unembedded_coeffs = lindblad_term_dict
                #if the length is equal to the number of qudits then we need to unembed.
                elif len(first_coeff_lbl.basis_element_labels[0]) == self.state_space.num_qudits:
                    unembedded_labels = list(lindblad_term_dict.keys())
                    for lbl in unembedded_labels:
                        new_bels = []
                        for bel in lbl.basis_element_labels:
                            new_bels.append("".join(bel[target] for target in self.target_labels))
                        lbl.basis_element_labels = tuple(new_bels)
                    unembedded_coeffs = {lbl:val for lbl, val in zip(unembedded_labels, lindblad_term_dict.values())}
                else:
                    msg = "Could not parse error generator labels. Expected either length equal to this embedded op's"\
                         +" target_labels or equal to the number of qudits."
                    raise ValueError(msg)

            self.embedded_op.set_errorgen_coefficients(unembedded_coeffs, action, logscale_nonham, truncate)
            if self._rep_type == 'dense': self._update_denserep()
            self.dirty = True

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in the error generator of this operation.

        Values are set so that the contributions of the resulting channel's
        error rate are given by the values in `lindblad_term_dict`.  See
        :meth:`error_rates` for more details.

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
        if self._rep_type == 'dense': self._update_denserep()

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
        if self._rep_type == 'dense': self._update_denserep()

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return self.embedded_op.has_nonzero_hessian()

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
        mm_dict['target_labels'] = self.target_labels
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])
        return cls(state_space, mm_dict['target_labels'], serial_memo[mm_dict['submembers'][0]])

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return (self.target_labels == other.target_labels) and (self.state_space == other.state_space)

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "embeds %s into %s" % (str(self.target_labels), str(self.state_space))

    def __str__(self):
        """ Return string representation """
        s = "Embedded operation with full dimension %d and state space %s\n" % (self.dim, self.state_space)
        s += " that embeds the following %d-dimensional operation into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.target_labels))
        s += str(self.embedded_op)
        return s
