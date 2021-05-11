

class OpRepBase(XXX):
    # These were taken from LinearOperator
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
        raise NotImplementedError("taylor_order_terms(...) not implemented for %s objects!" %
                                  self.__class__.__name__)

    def highmagnitude_terms(self, min_term_mag, force_firstorder=True, max_taylor_order=3, max_polynomial_vars=100):
        """
        Get terms with magnitude above `min_term_mag`.

        Get the terms (from a Taylor expansion of this operator) that have
        magnitude above `min_term_mag` (the magnitude of a term is taken to
        be the absolute value of its coefficient), considering only those
        terms up to some maximum Taylor expansion order, `max_taylor_order`.

        Note that this function also *sets* the magnitudes of the returned
        terms (by calling `term.set_magnitude(...)`) based on the current
        values of this operator's parameters.  This is an essential step
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
        #print("DB: OP get_high_magnitude_terms")
        v = self.to_vector()
        taylor_order = 0
        terms = []; last_len = -1; first_order_magmax = 1.0

        while len(terms) > last_len:  # while we keep adding something
            if taylor_order > 1 and first_order_magmax**taylor_order < min_term_mag:
                break  # there's no way any terms at this order reach min_term_mag - exit now!

            MAX_CACHED_TERM_ORDER = 1
            if taylor_order <= MAX_CACHED_TERM_ORDER:
                terms_at_order, cpolys = self.taylor_order_terms(taylor_order, max_polynomial_vars, True)
                coeffs = _bulk_eval_compact_polynomials_complex(
                    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
                terms_at_order = [t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order)]

                # CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs)
                # REMOVE later
                # for t in terms_at_order:
                #     vt, ct = t._rep.coeff.compact_complex()
                #     coeff_array = _bulk_eval_compact_polynomials_complex(vt, ct, self.parent.to_vector(), (1,))
                #     if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
                #         print(coeff_array[0], "vs.", t._rep.magnitude)
                #         import bpdb; bpdb.set_trace()

                if taylor_order == 1:
                    first_order_magmax = max([t.magnitude for t in terms_at_order])

                last_len = len(terms)
                for t in terms_at_order:
                    if t.magnitude >= min_term_mag or (taylor_order == 1 and force_firstorder):
                        terms.append((taylor_order, t))
            else:
                eff_min_term_mag = 0.0 if (taylor_order == 1 and force_firstorder) else min_term_mag
                terms.extend(
                    [(taylor_order, t)
                     for t in self.taylor_order_terms_above_mag(taylor_order, max_polynomial_vars, eff_min_term_mag)]
                )

            #print("order ", taylor_order, " : ", len(terms_at_order), " maxmag=",
            #      max([t.magnitude for t in terms_at_order]), len(terms), " running terms ",
            #      len(terms)-last_len, "added at this order")

            taylor_order += 1
            if taylor_order > max_taylor_order: break

        #Sort terms based on magnitude
        sorted_terms = sorted(terms, key=lambda t: t[1].magnitude, reverse=True)
        first_order_indices = [i for i, t in enumerate(sorted_terms) if t[0] == 1]

        #DEBUG TODO REMOVE
        #chk1 = sum([t[1].magnitude for t in sorted_terms])
        #chk2 = self.total_term_magnitude
        #print("HIGHMAG ",self.__class__.__name__, len(sorted_terms), " maxorder=",max_taylor_order,
        #      " minmag=",min_term_mag)
        #print("  sum of magnitudes =",chk1, " <?= ", chk2)
        #if chk1 > chk2:
        #    print("Term magnitudes = ", [t[1].magnitude for t in sorted_terms])
        #    egterms = self.errorgen.get_taylor_order_terms(0)
        #    #vtape, ctape = self.errorgen.Lterm_coeffs
        #    #coeffs = [ abs(x) for x in _bulk_eval_compact_polynomials_complex(vtape, ctape, self.errorgen.to_vector(),
        #    #  (len(self.errorgen.Lterms),)) ]
        #    mags = [ abs(t.evaluate_coeff(self.errorgen.to_vector()).coeff) for t in egterms ]
        #    print("Errorgen ", self.errorgen.__class__.__name__, " term magnitudes (%d): " % len(egterms),
        #    "\n",list(sorted(mags, reverse=True)))
        #    print("Errorgen sum = ",sum(mags), " vs ", self.errorgen.get_total_term_magnitude())
        #assert(chk1 <= chk2)

        return [t[1] for t in sorted_terms], first_order_indices

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
        v = self.to_vector()
        terms_at_order, cpolys = self.taylor_order_terms(order, max_polynomial_vars, True)
        coeffs = _bulk_eval_compact_polynomials_complex(
            cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        terms_at_order = [t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order)]

        #CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs) REMOVE later
        #for t in terms_at_order:
        #    vt,ct = t._rep.coeff.compact_complex()
        #    coeff_array = _bulk_eval_compact_polynomials_complex(vt,ct,self.parent.to_vector(),(1,))
        #    if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
        #        print(coeff_array[0], "vs.", t._rep.magnitude)
        #        import bpdb; bpdb.set_trace()

        return [t for t in terms_at_order if t.magnitude >= min_term_mag]



class OpRepComposed(XXX):

    def __init__(self, factor_op_reps, dim):
        #Term cache dicts (only used for "svterm" and "cterm" evotypes)
        self.terms = {}
        self.local_term_poly_coeffs = {}
        self.factor_ops = XXX  # see replib versions

    def param_indices_have_changed(self): # create this in base class so other evotypes don't need to implement it
        self.terms = {}  # clear terms cache since param indices have changed now
        self.local_term_poly_coeffs = {}

    def reinit_factor_op_reps(factor_op_reps):
        pass  # TODO

    def taylor_order_terms(self, order, gpindices_array, max_polynomial_vars=100, return_coeff_polys=False):
        """
        TODO docstring: add gpindices_array
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
            self._compute_taylor_order_terms(order, max_polynomial_vars, gpindices_array)

        if return_coeff_polys:
            #Return coefficient polys in terms of *local* parameters (get_taylor_terms
            #  and above composition gives polys in terms of *global*, model params)
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_polynomial_vars, gpindices_array):  # separated for profiling
        terms = []

        #DEBUG TODO REMOVE
        #print("Composed op getting order",order,"terms:")
        #for i,fop in enumerate(self.factorops):
        #    print(" ",i,fop.__class__.__name__,"totalmag = ",fop.get_total_term_magnitude())
        #    hmdebug,_ = fop.highmagnitude_terms(0.00001, True, order)
        #    print("  hmterms w/max order=",order," have magnitude ",sum([t.magnitude for t in hmdebug]))

        for p in _lt.partition_into(order, len(self.factor_reps)):
            factor_lists = [self.factor_reps[i].taylor_order_terms(pi, max_polynomial_vars) for i, pi in enumerate(p)]
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
            for rep in self.factor_reps
        ]
        for p in _lt.partition_into(order, len(self.factor_reps)):
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
        return _np.product([f.total_term_magnitude for f in self.factor_reps])

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
        opmags = [f.total_term_magnitude for f in self.factor_reps]
        product = _np.product(opmags)
        ret = _np.zeros(self.num_params, 'd')
        for opmag, f in zip(opmags, self.factor_reps):
            f_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, f.gpindices)  # HERE - need factors *reps* to have gpindices??
            local_deriv = product / opmag * f.total_term_magnitude_deriv
            ret[f_local_inds] += local_deriv
        return ret


class OpRepEmbedded(XXX):

    def __init__(self, state_space_labels, target_labels, embedded_rep):
        self.state_space_labels = state_space_labels
        self.target_labels = target_labels
        self.embedded_rep = embedded_rep

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
            terms, coeffs = self.embedded_rep.taylor_order_terms(order, max_polynomial_vars, True)
            embedded_terms = [t.embed(sslbls, self.target_labels) for t in terms]
            return embedded_terms, coeffs
        else:
            return [t.embed(sslbls, self.target_labels)
                    for t in self.embedded_rep.taylor_order_terms(order, max_polynomial_vars, False)]

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
                for t in self.embedded_rep.taylor_order_terms_above_mag(order, max_polynomial_vars, min_term_mag)]

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

        return self.embedded_rep.total_term_magnitude

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
        return self.embedded_rep.total_term_magnitude_deriv




class OpRepExpErrorgen(XXX):

    def __init__(self, errorgen_rep):

        #TODO: make terms init-able from sparse elements, and below code work with a *sparse* unitary_postfactor
        termtype = "dense" if evotype == "svterm" else "clifford"  # TODO - split to cterm case
        
        #TODO REMOVE
        # Store *unitary* as self.unitary_postfactor - NOT a superop
        #if unitary_postfactor is not None:  # can be None
        #    op_std = _bt.change_basis(unitary_postfactor, self.errorgen.matrix_basis, 'std')
        #    self.unitary_postfactor = _gt.process_mx_to_unitary(op_std)
        #
        #    # automatically "up-convert" operation to CliffordOp if needed
        #    if termtype == "clifford":
        #        self.unitary_postfactor = CliffordOp(self.unitary_postfactor)
        #else:
        #    self.unitary_postfactor = None

        rep = d2  # no representation object in term-mode (will be set to None by LinearOperator)
        self.errorgen_rep = errorgen_rep

        #Cache values
        self.terms = {}
        self.exp_terms_cache = {}  # used for repeated calls to the exponentiate_terms function
        self.local_term_poly_coeffs = {}
        # TODO REMOVE self.direct_terms = {}
        # TODO REMOVE self.direct_term_poly_coeffs = {}

    def param_indices_have_changed(self):
        self.terms = {}  # clear terms cache since param indices have changed now
        self.exp_terms_cache = {}
        self.local_term_poly_coeffs = {}
        #TODO REMOVE self.direct_terms = {}
        #TODO REMOVE self.direct_term_poly_coeffs = {}

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
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

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
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_polynomial_vars):  # separated for profiling

        mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[ii] = i

        def _compose_poly_indices(terms):
            for term in terms:
                #term.map_indices_inplace(lambda x: tuple(_modelmember._compose_gpindices(
                #    self.gpindices, _np.array(x, _np.int64))))
                term.mapvec_indices_inplace(mapvec)
            return terms

        assert(self.gpindices is not None), "LindbladOp must be added to a Model before use!"
        assert(not _sps.issparse(self.unitary_postfactor)
               ), "Unitary post-factor needs to be dense for term-based evotypes"
        # for now - until StaticDenseOp and CliffordOp can init themselves from a *sparse* matrix
        mpv = max_polynomial_vars
        postTerm = _term.RankOnePolynomialOpTerm.create_from(_Polynomial({(): 1.0}, mpv), self.unitary_postfactor,
                                                             self.unitary_postfactor, self._evotype)
        #Note: for now, *all* of an error generator's terms are considered 0-th order,
        # so the below call to taylor_order_terms just gets all of them.  In the FUTURE
        # we might want to allow a distinction among the error generator terms, in which
        # case this term-exponentiation step will need to become more complicated...
        loc_terms = _term.exponentiate_terms(self.errorgen_rep.taylor_order_terms(0, max_polynomial_vars),
                                             order, postTerm, self.exp_terms_cache)
        #OLD: loc_terms = [ t.collapse() for t in loc_terms ] # collapse terms for speed

        poly_coeffs = [t.coeff for t in loc_terms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)
        self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

        # only cache terms with *global* indices to avoid confusion...
        self.terms[order] = _compose_poly_indices(loc_terms)

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
        mapvec = _np.ascontiguousarray(_np.zeros(max_polynomial_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[ii] = i

        assert(self.gpindices is not None), "LindbladOp must be added to a Model before use!"
        assert(not _sps.issparse(self.unitary_postfactor)
               ), "Unitary post-factor needs to be dense for term-based evotypes"
        # for now - until StaticDenseOp and CliffordOp can init themselves from a *sparse* matrix
        mpv = max_polynomial_vars
        postTerm = _term.RankOnePolynomialOpTerm.create_from(_Polynomial({(): 1.0}, mpv), self.unitary_postfactor,
                                                             self.unitary_postfactor, self._evotype)
        postTerm = postTerm.copy_with_magnitude(1.0)
        #Note: for now, *all* of an error generator's terms are considered 0-th order,
        # so the below call to taylor_order_terms just gets all of them.  In the FUTURE
        # we might want to allow a distinction among the error generator terms, in which
        # case this term-exponentiation step will need to become more complicated...
        errgen_terms = self.errorgen_rep.taylor_order_terms(0, max_polynomial_vars)

        #DEBUG: CHECK MAGS OF ERRGEN COEFFS
        #poly_coeffs = [t.coeff for t in errgen_terms]
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #v = self.to_vector()
        #errgen_coeffs = _bulk_eval_compact_polynomials_complex(
        #    vtape, ctape, v, (len(errgen_terms),))  # an array of coeffs
        #for coeff, t in zip(errgen_coeffs, errgen_terms):
        #    coeff2 = t.coeff.evaluate(v)
        #    if not _np.isclose(coeff,coeff2):
        #        assert(False), "STOP"
        #    t.set_magnitude(abs(coeff))

        #evaluate errgen_terms' coefficients using their local vector of parameters
        # (which happends to be the same as our paramvec in this case)
        egvec = self.errorgen.to_vector()   # HERE - pass vector in?  we need errorgen's vector (usually not in rep) to perform evaluation...
        errgen_terms = [egt.copy_with_magnitude(abs(egt.coeff.evaluate(egvec))) for egt in errgen_terms]

        #DEBUG!!!
        #import bpdb; bpdb.set_trace()
        #loc_terms = _term.exponentiate_terms_above_mag(errgen_terms, order, postTerm, min_term_mag=-1)
        #loc_terms_chk = _term.exponentiate_terms(errgen_terms, order, postTerm)
        #assert(len(loc_terms) == len(loc_terms2))
        #poly_coeffs = [t.coeff for t in loc_terms_chk]
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #v = self.to_vector()
        #coeffs = _bulk_eval_compact_polynomials_complex(
        #    vtape, ctape, v, (len(loc_terms_chk),))  # an array of coeffs
        #for coeff, t, t2 in zip(coeffs, loc_terms, loc_terms_chk):
        #    coeff2 = t.coeff.evaluate(v)
        #    if not _np.isclose(coeff,coeff2):
        #        assert(False), "STOP"
        #    t.set_magnitude(abs(coeff))

        #for ii,t in enumerate(loc_terms):
        #    coeff1 = t.coeff.evaluate(egvec)
        #    if not _np.isclose(abs(coeff1), t.magnitude):
        #        assert(False),"STOP"
        #    #t.set_magnitude(abs(t.coeff.evaluate(egvec)))

        #FUTURE:  maybe use bulk eval of compact polys? Something like this:
        #coeffs = _bulk_eval_compact_polynomials_complex(
        #    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        #for coeff, t in zip(coeffs, terms_at_order):
        #    t.set_magnitude(abs(coeff))

        terms = []
        for term in _term.exponentiate_terms_above_mag(errgen_terms, order,
                                                       postTerm, min_term_mag=min_term_mag):
            #poly_coeff = term.coeff
            #compact_poly_coeff = poly_coeff.compact(complex_coeff_tape=True)
            term.mapvec_indices_inplace(mapvec)  # local -> global indices

            # CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs)
            # REMOVE later
            # t = term
            # vt, ct = t._rep.coeff.compact_complex()
            # coeff_array = _bulk_eval_compact_polynomials_complex(vt, ct, self.parent.to_vector(), (1,))
            # if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
            #     print(coeff_array[0], "vs.", t._rep.magnitude)
            #     import bpdb; bpdb.set_trace()
            #     c1 = _Polynomial.from_rep(t._rep.coeff)

            terms.append(term)
        return terms

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
        # return exp( mag of errorgen ) = exp( sum of absvals of errgen term coeffs )
        # (unitary postfactor has weight == 1.0 so doesn't enter)
        #TODO REMOVE:
        #print("  DB: LindbladOp.get_totat_term_magnitude: (errgen type =",self.errorgen.__class__.__name__)
        #egttm = self.errorgen.total_term_magnitude
        #print("  DB: exp(", egttm, ") = ",_np.exp(egttm))
        #return _np.exp(egttm)
        return _np.exp(min(self.errorgen_rep.total_term_magnitude, MAX_EXPONENT))
        #return _np.exp(self.errorgen.total_term_magnitude)  # overflows sometimes

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
        return _np.exp(self.errorgen_rep.total_term_magnitude) * self.errorgen_rep.total_term_magnitude_deriv


class OpRepStochastic(XXX):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state):
        self.basis = basis
        self.stochastic_superops = []
        for b in self.basis.elements[1:]:
            std_superop = _lbt.nonham_lindbladian(b, b, sparse=False)
            self.stochastic_superops.append(_bt.change_basis(std_superop, 'std', self.basis))
        self.rate_poly_dicts = rate_poly_dicts
        self.rates = initial_rates

    def update_rates(self, rates):
        self.rates[:] = rates

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
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

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

        def _compose_poly_indices(terms):
            for term in terms:
                term.map_indices_inplace(lambda x: tuple(_modelmember._compose_gpindices(
                    self.gpindices, _np.array(x, _np.int64))))
            return terms

        IDENT = None  # sentinel for the do-nothing identity op
        mpv = max_polynomial_vars
        if order == 0:
            polydict = {(): 1.0}
            for pd in self.rate_poly_dicts:
                polydict.update({k: -v for k, v in pd.items()})  # subtracts the "rate" `pd` from `polydict`
            loc_terms = [_term.RankOnePolynomialOpTerm.create_from(_Polynomial(polydict, mpv),
                                                                   IDENT, IDENT, self._evotype)]

        elif order == 1:
            loc_terms = [_term.RankOnePolynomialOpTerm.create_from(_Polynomial(pd, mpv), bel, bel, self._evotype)
                         for i, (pd, bel) in enumerate(zip(self.rate_poly_dicts, self.basis.elements[1:]))]
        else:
            loc_terms = []  # only first order "taylor terms"

        poly_coeffs = [t.coeff for t in loc_terms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)

        local_term_poly_coeffs = coeffs_as_compact_polys
        global_param_terms = _compose_poly_indices(loc_terms)

        if return_coeff_polys:
            return global_param_terms, local_term_poly_coeffs
        else:
            return global_param_terms

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
        # return exp( mag of errorgen ) = exp( sum of absvals of errgen term coeffs )
        # (unitary postfactor has weight == 1.0 so doesn't enter)
        return _np.sum(_np.abs(self.rates))

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
        ret = _np.zeros(len(self.gpindices_array), 'd')
        for pdict in self.rate_poly_dicts:
            # rate_poly_dicts:  keys give poly terms, e.g. (i,i) => x_i^2
            for k, v in pdict.items():
                if len(k) == 1:
                    ret[k[0]] += v
                elif len(k) == 2 and k[0] == k[1]:
                    ret[k[0]] += 2 * v * _np.sqrt(self.rates[k[0]])
                else:
                    raise NotImplementedError("This routine doesn't handle arbitrary rate_poly_dicts yet...")
        return ret

        #Specific to the stochastic op parameterization:
        # abs(rates) = rates = params**2
        # so d( sum(abs(rates)) )/dparam_i = 2*param_i = 2 * sqrt(rates)
        #return 2 * _np.sqrt(self.rates)


class OpRepSum(XXX):
    def __init__(self, factor_reps, dim):
        self.factor_reps = factor_reps

    def reinit_factor_reps(self, factor_reps):
        pass

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
        for eg in self.factor_reps:
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

        #DEBUG TODO REMOVE
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

        return sum([eg.total_term_magnitude for eg in self.factor_reps])

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
        for eg in self.factor_reps:
            eg_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            ret[eg_local_inds] += eg.total_term_magnitude_deriv
        return ret


class OpRepStandard(XXX):
    def __init__(self, name):
        std_unitaries = _itgs.standard_gatename_unitaries()
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        # do we need this?
