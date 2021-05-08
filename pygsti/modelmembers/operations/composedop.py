class ComposedOp(LinearOperator):
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

    dim : int or "auto"
        Dimension of this operation.  Can be set to `"auto"` to take dimension
        from `ops_to_compose[0]` *if* there's at least one operation being
        composed.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
        The evolution type of this operation.  Can be set to `"auto"` to take
        the evolution type of `ops_to_compose[0]` *if* there's at least
        one operation being composed.

    dense_rep : bool, optional
        Whether this operator should be internally represented using a dense
        matrix.  This is expert-level functionality, and you should leave their
        the default value unless you know what you're doing.
    """

    def __init__(self, ops_to_compose, dim="auto", evotype="auto", dense_rep=False):
        """
        Creates a new ComposedOp.

        Parameters
        ----------
        ops_to_compose : list
            List of `LinearOperator`-derived objects
            that are composed to form this operation map.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.

        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one operation being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one operation being composed.

        dense_rep : bool, optional
            Whether this operator should be internally represented using a dense
            matrix.  This is expert-level functionality, and you should leave their
            the default value unless you know what you're doing.
        """
        assert(len(ops_to_compose) > 0 or dim != "auto"), \
            "Must compose at least one operation when dim='auto'!"
        self.factorops = list(ops_to_compose)
        self.dense_rep = dense_rep

        if dim == "auto":
            dim = ops_to_compose[0].dim
        assert(all([dim == operation.dim for operation in ops_to_compose])), \
            "All operations must have the same dimension (%d expected)!" % dim

        if evotype == "auto":
            evotype = ops_to_compose[0]._evotype
        assert(all([evotype == operation._evotype for operation in ops_to_compose])), \
            "All operations must have the same evolution type (%s expected)!" % evotype

        #Term cache dicts (only used for "svterm" and "cterm" evotypes)
        self.terms = {}
        self.local_term_poly_coeffs = {}

        #Create representation object
        factor_op_reps = [op._rep for op in self.factorops]
        if evotype == "densitymx":
            if dense_rep:
                rep = replib.DMOpRepDense(_np.require(_np.identity(dim, 'd'),
                                                      requirements=['OWNDATA', 'C_CONTIGUOUS']))
            else:
                rep = replib.DMOpRepComposed(factor_op_reps, dim)
        elif evotype == "statevec":
            if dense_rep:
                rep = replib.SVOpRepDense(_np.require(_np.identity(dim, complex),
                                                      requirements=['OWNDATA', 'C_CONTIGUOUS']))
            else:
                rep = replib.SVOpRepComposed(factor_op_reps, dim)
        elif evotype == "stabilizer":
            assert(not dense_rep), "Cannot require a dense representation with stabilizer evotype!"
            nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
            rep = replib.SBOpRepComposed(factor_op_reps, nQubits)
        else:
            assert(not dense_rep), "Cannot require a dense representation with %s evotype!" % evotype
            rep = dim  # no proper representation (_rep will be set to None by LinearOperator)

        LinearOperator.__init__(self, rep, evotype)
        if self.dense_rep: self._update_denserep()  # update dense rep if needed

    def _update_denserep(self):
        if len(self.factorops) == 0:
            mx = _np.identity(self.dim, 'd')
        else:
            mx = self.factorops[0].to_dense()
            for op in self.factorops[1:]:
                mx = _np.dot(op.to_dense(), mx)

        self._rep.base.flags.writeable = True
        self._rep.base[:, :] = mx
        self._rep.base.flags.writeable = False

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
        if self.dense_rep:
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed

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
        if self.dense_rep:
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed

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
        # We need to override this method so that factor operations have their
        # parent reset correctly.
        if memo is not None and id(self) in memo: return memo[id(self)]
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls([g.copy(parent, memo) for g in self.factorops], self.dim, self._evotype)
        return self._copy_gpindices(copyOfMe, parent, memo)

    def to_sparse(self):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        mx = self.factorops[0].to_sparse()
        for op in self.factorops[1:]:
            mx = op.to_sparse().dot(mx)
        return mx

    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        if self.dense_rep:
            #We already have a dense version stored
            return self._rep.base
        elif len(self.factorops) == 0:
            return _np.identity(self.dim, 'd')
        else:
            mx = self.factorops[0].to_dense()
            for op in self.factorops[1:]:
                mx = _np.dot(op.to_dense(), mx)
            return mx

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
    #    factor_op_reps = [operation.torep() for operation in self.factorops]
    #    #FUTURE? factor_op_reps = [ repmemo.get(id(operation), operation.torep(debug_time_dict)) for operation in self.factorops ] #something like this? # noqa
    #
    #    if self._evotype == "densitymx":
    #        return replib.DMOpRepComposed(factor_op_reps, self.dim)
    #    elif self._evotype == "statevec":
    #        return replib.SVOpRepComposed(factor_op_reps, self.dim)
    #    elif self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))  # "stabilizer" is a unitary-evolution type mode
    #        return replib.SBOpRepComposed(factor_op_reps, nQubits)
    #
    #    assert(False), "Invalid internal _evotype: %s" % self._evotype

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
            self._compute_taylor_order_terms(order, max_polynomial_vars)

        if return_coeff_polys:
            #Return coefficient polys in terms of *local* parameters (get_taylor_terms
            #  and above composition gives polys in terms of *global*, model params)
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_polynomial_vars):  # separated for profiling
        terms = []

        #DEBUG TODO REMOVE
        #print("Composed op getting order",order,"terms:")
        #for i,fop in enumerate(self.factorops):
        #    print(" ",i,fop.__class__.__name__,"totalmag = ",fop.get_total_term_magnitude())
        #    hmdebug,_ = fop.highmagnitude_terms(0.00001, True, order)
        #    print("  hmterms w/max order=",order," have magnitude ",sum([t.magnitude for t in hmdebug]))

        for p in _lt.partition_into(order, len(self.factorops)):
            factor_lists = [self.factorops[i].taylor_order_terms(pi, max_polynomial_vars) for i, pi in enumerate(p)]
            for factors in _itertools.product(*factor_lists):
                terms.append(_term.compose_terms(factors))
        self.terms[order] = terms

        #def _decompose_indices(x):
        #    return tuple(_modelmember._decompose_gpindices(
        #        self.gpindices, _np.array(x, _np.int64)))

        mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
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
            [ops.taylor_order_terms_above_mag(i, max_polynomial_vars, min_term_mag) for i in range(order + 1)]
            for ops in self.factorops
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
        for opmag, f in zip(opmags, self.factorops):
            f_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, f.gpindices)
            local_deriv = product / opmag * f.total_term_magnitude_deriv
            ret[f_local_inds] += local_deriv
        return ret

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        vl = _np.empty(self.num_params, dtype=object)
        for operation in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, operation.gpindices)
            vl[factorgate_local_inds] = operation.parameter_labels
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
        for operation in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, operation.gpindices)
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
        for operation in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, operation.gpindices)
            operation.from_vector(v[factorgate_local_inds], close, dirty_value)
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
        typ = complex if any([_np.iscomplexobj(op.to_dense()) for op in self.factorops]) else 'd'
        derivMx = _np.zeros((self.dim, self.dim, self.num_params), typ)

        #Product rule to compute jacobian
        for i, op in enumerate(self.factorops):  # loop over the operation we differentiate wrt
            if op.num_params == 0: continue  # no contribution
            deriv = op.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this operation...
            deriv.shape = (self.dim, self.dim, op.num_params)

            if i > 0:  # factors before ith
                pre = self.factorops[0].to_dense()
                for opA in self.factorops[1:i]:
                    pre = _np.dot(opA.to_dense(), pre)
                #deriv = _np.einsum("ija,jk->ika", deriv, pre )
                deriv = _np.transpose(_np.tensordot(deriv, pre, (1, 0)), (0, 2, 1))

            if i + 1 < len(self.factorops):  # factors after ith
                post = self.factorops[i + 1].to_dense()
                for opA in self.factorops[i + 2:]:
                    post = _np.dot(opA.to_dense(), post)
                #deriv = _np.einsum("ij,jka->ika", post, deriv )
                deriv = _np.tensordot(post, deriv, (1, 0))

            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, op.gpindices)
            derivMx[:, :, factorgate_local_inds] += deriv

        derivMx.shape = (self.dim**2, self.num_params)
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

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
            factor_coeffs = op.errorgen_coefficients(return_basis, logscale_nonham)

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
        return _np.concatenate([op.errorgen_coefficients_array_deriv_wrt_params() for op in self.factorops], axis=0)

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

    def get_chp_str(self, targets=None):
        """Return a string suitable for printing to a CHP input file from all underlying operations.

        Parameters
        ----------
        targets: list of int
            Qubits to be applied to (if None, uses stored CHP strings directly)

        Returns
        -------
        s : str
            String of CHP code
        """
        s = ""
        for op in self.factorops:
            s += op.get_chp_str(targets)
        return s

    def __str__(self):
        """ Return string representation """
        s = "Composed operation of %d factors:\n" % len(self.factorops)
        for i, operation in enumerate(self.factorops):
            s += "Factor %d:\n" % i
            s += str(operation)
        return s


class ComposedDenseOp(ComposedOp, DenseOperatorInterface):
    """
    An operation that is the composition of a number of matrix factors (possibly other operations).

    Parameters
    ----------
    ops_to_compose : list
        A list of 2D numpy arrays (matrices) and/or `DenseOperator`-derived
        objects that are composed to form this operation.  Elements are composed
        with vectors  in  *left-to-right* ordering, maintaining the same
        convention as operation sequences in pyGSTi.  Note that this is
        *opposite* from standard matrix multiplication order.

    dim : int or "auto"
        Dimension of this operation.  Can be set to `"auto"` to take dimension
        from `ops_to_compose[0]` *if* there's at least one operation being
        composed.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
        The evolution type of this operation.  Can be set to `"auto"` to take
        the evolution type of `ops_to_compose[0]` *if* there's at least
        one operation being composed.
    """

    def __init__(self, ops_to_compose, dim="auto", evotype="auto"):
        """
        Creates a new ComposedDenseOp.

        Parameters
        ----------
        ops_to_compose : list
            A list of 2D numpy arrays (matrices) and/or `DenseOperator`-derived
            objects that are composed to form this operation.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.

        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one operation being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one operation being composed.
        """
        ComposedOp.__init__(self, ops_to_compose, dim, evotype, dense_rep=True)
        DenseOperatorInterface.__init__(self)

    @property
    def parameter_labels(self):  # Needed because method resolution finds __getattr__ before base class property
        return ComposedOp.parameter_labels.fget(self)


class ExponentiatedOp(LinearOperator):
    """
    An operation map that is the composition of a number of map-like factors (possibly other `LinearOperator`s)

    Parameters
    ----------
    op_to_exponentiate : list
        A `LinearOperator`-derived object that is exponentiated to
        some integer power to produce this operator.

    power : int
        the power to exponentiate `op_to_exponentiate` to.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
        the evolution type.  `"auto"` uses the evolution type of
        `op_to_exponentiate`.
    """

    def __init__(self, op_to_exponentiate, power, evotype="auto"):
        """
        Creates a new ExponentiatedOp.

        Parameters
        ----------
        op_to_exponentiate : list
            A `LinearOperator`-derived object that is exponentiated to
            some integer power to produce this operator.

        power : int
            the power to exponentiate `op_to_exponentiate` to.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            the evolution type.  `"auto"` uses the evolution type of
            `op_to_exponentiate`.
        """
        #We may not actually need to save these, since they can be inferred easily
        self.exponentiated_op = op_to_exponentiate
        self.power = power

        dim = op_to_exponentiate.dim

        if evotype == "auto":
            evotype = op_to_exponentiate._evotype

        if evotype == "densitymx":
            rep = replib.DMOpRepExponentiated(self.exponentiated_op._rep, self.power, dim)
        elif evotype == "statevec":
            rep = replib.SVOpRepExponentiated(self.exponentiated_op._rep, self.power, dim)
        elif evotype == "stabilizer":
            nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
            rep = replib.SVOpRepExponentiated(self.exponentiated_op._rep, self.power, nQubits)
        else:
            raise ValueError("Invalid evotype: %s for ExponentiatedOp object" % evotype)

        LinearOperator.__init__(self, rep, evotype)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.exponentiated_op]

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
        self.exponentiated_op.set_time(t)

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
        # We need to override this method so that factor operations have their
        # parent reset correctly.
        if memo is not None and id(self) in memo: return memo[id(self)]
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.exponentiated_op.copy(parent, memo), self.power, self._evotype)
        return self._copy_gpindices(copyOfMe, parent, memo)

    def to_sparse(self):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if self.power == 0:
            return _sps.identity(self.dim, dtype=_np.dtype('d'), format='csr')

        op = self.exponentiated_op.to_sparse()
        mx = op.copy()
        for i in range(self.power - 1):
            mx = mx.dot(op)
        return mx

    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        op = self.exponentiated_op.to_dense()
        return _np.linalg.matrix_power(op, self.power)

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
    #        return replib.DMOpRepExponentiated(self.exponentiated_op.torep(), self.power, self.dim)
    #    elif self._evotype == "statevec":
    #        return replib.SVOpRepExponentiated(self.exponentiated_op.torep(), self.power, self.dim)
    #    elif self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))  # "stabilizer" is a unitary-evolution type mode
    #        return replib.SVOpRepExponentiated(self.exponentiated_op.torep(), self.power, nQubits)
    #    assert(False), "Invalid internal _evotype: %s" % self._evotype

    #FUTURE: term-related functions (maybe base off of ComposedOp or use a composedop to generate them?)
    # e.g. ComposedOp([self.exponentiated_op] * power, dim, evotype)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.exponentiated_op.paramter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.exponentiated_op.num_params

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        return self.exponentiated_op.to_vector()

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
        self.exponentiated_op.from_vector(v, close, dirty_value)
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Constructs a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

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
        mx = self.exponentiated_op.to_dense()

        mx_powers = {0: _np.identity(self.dim, 'd'), 1: mx}
        for i in range(2, self.power):
            mx_powers[i] = _np.dot(mx_powers[i - 1], mx)

        dmx = _np.transpose(self.exponentiated_op.deriv_wrt_params(wrt_filter))  # (num_params, dim^2)
        dmx.shape = (dmx.shape[0], self.dim, self.dim)  # set shape for multiplication below

        deriv = _np.zeros((self.dim, dmx.shape[0], self.dim), 'd')
        for k in range(1, self.power + 1):
            #deriv += mx_powers[k-1] * dmx * mx_powers[self.power-k]
            deriv += _np.dot(mx_powers[k - 1], _np.dot(dmx, mx_powers[self.power - k]))
            #        (D,D) * ((P,D,D) * (D,D)) => (D,D) * (P,D,D) => (D,P,D)

        deriv = _np.moveaxis(deriv, 1, 2)
        deriv = deriv.reshape((self.dim**2, deriv.shape[2]))
        return deriv

    def __str__(self):
        """ Return string representation """
        s = "Exponentiated operation that raise the below op to the %d power\n" % self.power
        s += str(self.exponentiated_op)
        return s
