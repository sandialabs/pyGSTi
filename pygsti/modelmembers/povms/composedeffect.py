"""
The ComposedPOVMEffect class and supporting functionality.
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

from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.modelmembers.states.staticstate import StaticState as _StaticState


class ComposedPOVMEffect(_POVMEffect):  # , _ErrorMapContainer
    """
    TODO: update docstring
    A Lindblad-parameterized POVMEffect (that is also expandable into terms).

    Parameters
    ----------
    pure_vec : numpy array or POVMEffect
        An array or POVMEffect in the *full* density-matrix space (this
        vector will have dimension 4 in the case of a single qubit) which
        represents a pure-state preparation or projection.  This is used as
        the "base" preparation or projection that is followed or preceded
        by, respectively, the parameterized Lindblad-form error generator.
        (This argument is *not* copied if it is a POVMEffect.  A numpy array
         is converted to a new static POVM effect.)

    errormap : MapOperator
        The error generator action and parameterization, encapsulated in
        a gate object.  Usually a :class:`LindbladOp`
        or :class:`ComposedOp` object.  (This argument is *not* copied,
        to allow ComposedPOVMEffects to share error generator
        parameters with other gates and spam vectors.)
    """

    #@classmethod
    #def _from_spamvec_obj(cls, spamvec, typ, param_type="GLND", purevec=None,
    #                      proj_basis="pp", mx_basis="pp", truncate=True,
    #                      lazy=False):
    #    """
    #    Creates a LindbladSPAMVec from an existing SPAMVec object and some additional information.
    #
    #    This function is different from `from_spam_vector` in that it assumes
    #    that `spamvec` is a :class:`SPAMVec`-derived object, and if `lazy=True`
    #    and if `spamvec` is already a matching LindbladSPAMVec, it
    #    is returned directly.  This routine is primarily used in spam vector
    #    conversion functions, where conversion is desired only when necessary.
    #
    #    Parameters
    #    ----------
    #    spamvec : SPAMVec
    #        The spam vector object to "convert" to a
    #        `LindbladSPAMVec`.
    #
    #    typ : {"prep","effect"}
    #        Whether this is a state preparation or POVM effect vector.
    #
    #    param_type : str, optional
    #        The high-level "parameter type" of the gate to create.  This
    #        specifies both which Lindblad parameters are included and what
    #        type of evolution is used.  Examples of valid values are
    #        `"CPTP"`, `"H+S"`, `"S terms"`, and `"GLND clifford terms"`.
    #
    #    purevec : numpy array or SPAMVec object, optional
    #        A SPAM vector which represents a pure-state, taken as the "ideal"
    #        reference state when constructing the error generator of the
    #        returned `LindbladSPAMVec`.  Note that this vector
    #        still acts on density matrices (if it's a SPAMVec it should have
    #        a "densitymx", "svterm", or "cterm" evolution type, and if it's
    #        a numpy array it should have the same dimension as `spamvec`).
    #        If None, then it is taken to be `spamvec`, and so `spamvec` must
    #        represent a pure state in this case.
    #
    #    proj_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
    #        The basis used to construct the Lindblad-term error generators onto
    #        which the SPAM vector's error generator is projected.  Allowed values
    #        are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
    #        and Qutrit (qt), list of numpy arrays, or a custom basis object.
    #
    #    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
    #        The source and destination basis, respectively.  Allowed
    #        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
    #        and Qutrit (qt) (or a custom basis object).
    #
    #    truncate : bool, optional
    #        Whether to truncate the projections onto the Lindblad terms in
    #        order to meet constraints (e.g. to preserve CPTP) when necessary.
    #        If False, then an error is thrown when the given `spamvec` cannot
    #        be realized by the specified set of Lindblad projections.
    #
    #    lazy : bool, optional
    #        If True, then if `spamvec` is already a LindbladSPAMVec
    #        with the requested details (given by the other arguments), then
    #        `spamvec` is returned directly and no conversion/copying is
    #        performed. If False, then a new object is always returned.
    #
    #    Returns
    #    -------
    #    LindbladSPAMVec
    #    """
    #
    #    if not isinstance(spamvec, SPAMVec):
    #        spamvec = StaticSPAMVec(spamvec, typ=typ)  # assume spamvec is just a vector
    #
    #    if purevec is None:
    #        purevec = spamvec  # right now, we don't try to extract a "closest pure vec"
    #        # to spamvec - below will fail if spamvec isn't pure.
    #    elif not isinstance(purevec, SPAMVec):
    #        purevec = StaticSPAMVec(purevec, typ=typ)  # assume spamvec is just a vector
    #
    #    #Break param_type in to a "base" type and an evotype
    #    from .operation import LindbladOp as _LPGMap
    #    bTyp, evotype, nonham_mode, param_mode = _LPGMap.decomp_paramtype(param_type)
    #
    #    ham_basis = proj_basis if (("H" == bTyp) or ("H+" in bTyp) or bTyp in ("CPTP", "GLND")) else None
    #    nonham_basis = None if bTyp == "H" else proj_basis
    #
    #    def beq(b1, b2):
    #        """ Check if bases have equal names """
    #        b1 = b1.name if isinstance(b1, _Basis) else b1
    #        b2 = b2.name if isinstance(b2, _Basis) else b2
    #        return b1 == b2
    #
    #    def normeq(a, b):
    #        if a is None and b is None: return True
    #        if a is None or b is None: return False
    #        return _mt.safe_norm(a - b) < 1e-6  # what about possibility of Clifford gates?
    #
    #    if isinstance(spamvec, LindbladSPAMVec) \
    #       and spamvec._evotype == evotype and spamvec.typ == typ \
    #       and beq(ham_basis, spamvec.error_map.ham_basis) and beq(nonham_basis, spamvec.error_map.other_basis) \
    #       and param_mode == spamvec.error_map.param_mode and nonham_mode == spamvec.error_map.nonham_mode \
    #       and beq(mx_basis, spamvec.error_map.matrix_basis) and lazy:
    #        #normeq(gate.pure_state_vec,purevec) \ # TODO: more checks for equality?!
    #        return spamvec  # no creation necessary!
    #    else:
    #        #Convert vectors (if possible) to SPAMVecs
    #        # of the appropriate evotype and 0 params.
    #        bDiff = spamvec is not purevec
    #        spamvec = _convert_to_lindblad_base(spamvec, typ, evotype, mx_basis)
    #        purevec = _convert_to_lindblad_base(purevec, typ, evotype, mx_basis) if bDiff else spamvec
    #        assert(spamvec._evotype == evotype)
    #        assert(purevec._evotype == evotype)
    #
    #        return cls.from_spam_vector(
    #            spamvec, purevec, typ, ham_basis, nonham_basis,
    #            param_mode, nonham_mode, truncate, mx_basis, evotype)
    #
    #@classmethod
    #def from_spam_vector(cls, spam_vec, pure_vec, typ,
    #                     ham_basis="pp", nonham_basis="pp", param_mode="cptp",
    #                     nonham_mode="all", truncate=True, mx_basis="pp",
    #                     evotype="densitymx"):
    #    """
    #    Creates a Lindblad-parameterized spamvec from a state vector and a basis.
    #
    #    The basis specifies how to decompose (project) the vector's error generator.
    #
    #    Parameters
    #    ----------
    #    spam_vec : SPAMVec
    #        the SPAM vector to initialize from.  The error generator that
    #        tranforms `pure_vec` into `spam_vec` forms the parameterization
    #        of the returned LindbladSPAMVec.
    #
    #    pure_vec : numpy array or SPAMVec
    #        An array or SPAMVec in the *full* density-matrix space (this
    #        vector will have the same dimension as `spam_vec` - 4 in the case
    #        of a single qubit) which represents a pure-state preparation or
    #        projection.  This is used as the "base" preparation/projection
    #        when computing the error generator that will be parameterized.
    #        Note that this argument must be specified, as there is no natural
    #        default value (like the identity in the case of gates).
    #
    #    typ : {"prep","effect"}
    #        Whether this is a state preparation or POVM effect vector.
    #
    #    ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
    #        The basis is used to construct the Hamiltonian-type lindblad error
    #        Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
    #        and Qutrit (qt), list of numpy arrays, or a custom basis object.
    #
    #    nonham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
    #        The basis is used to construct the Stochastic-type lindblad error
    #        Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
    #        and Qutrit (qt), list of numpy arrays, or a custom basis object.
    #
    #    param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
    #        Describes how the Lindblad coefficients/projections relate to the
    #        SPAM vector's parameter values.  Allowed values are:
    #        `"unconstrained"` (coeffs are independent unconstrained parameters),
    #        `"cptp"` (independent parameters but constrained so map is CPTP),
    #        `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
    #        `"depol"` (same as `"reldepol"` but coeffs must be *positive*)
    #
    #    nonham_mode : {"diagonal", "diag_affine", "all"}
    #        Which non-Hamiltonian Lindblad projections are potentially non-zero.
    #        Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
    #        `"diag_affine"` (diagonal coefficients + affine projections), and
    #        `"all"` (the entire matrix of coefficients is allowed).
    #
    #    truncate : bool, optional
    #        Whether to truncate the projections onto the Lindblad terms in
    #        order to meet constraints (e.g. to preserve CPTP) when necessary.
    #        If False, then an error is thrown when the given `gate` cannot
    #        be realized by the specified set of Lindblad projections.
    #
    #    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
    #        The source and destination basis, respectively.  Allowed
    #        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
    #        and Qutrit (qt) (or a custom basis object).
    #
    #    evotype : {"densitymx","svterm","cterm"}
    #        The evolution type of the spamvec being constructed.  `"densitymx"` is
    #        usual Lioville density-matrix-vector propagation via matrix-vector
    #        products.  `"svterm"` denotes state-vector term-based evolution
    #        (spamvec is obtained by evaluating the rank-1 terms up to
    #        some order).  `"cterm"` is similar but stabilizer states.
    #
    #    Returns
    #    -------
    #    LindbladSPAMVec
    #    """
    #    #Compute a (errgen, pure_vec) pair from the given
    #    # (spam_vec, pure_vec) pair.
    #
    #    assert(pure_vec is not None), "Must supply `pure_vec`!"  # since there's no good default?
    #
    #    if not isinstance(spam_vec, SPAMVec):
    #        spam_vec = StaticSPAMVec(spam_vec, evotype, typ)  # assume spamvec is just a vector
    #    if not isinstance(pure_vec, SPAMVec):
    #        pure_vec = StaticSPAMVec(pure_vec, evotype, typ)  # assume spamvec is just a vector
    #    d2 = pure_vec.dim
    #
    #    #Determine whether we're using sparse bases or not
    #    sparse = None
    #    if ham_basis is not None:
    #        if isinstance(ham_basis, _Basis): sparse = ham_basis.sparse
    #        elif not isinstance(ham_basis, str) and len(ham_basis) > 0:
    #            sparse = _sps.issparse(ham_basis[0])
    #    if sparse is None and nonham_basis is not None:
    #        if isinstance(nonham_basis, _Basis): sparse = nonham_basis.sparse
    #        elif not isinstance(nonham_basis, str) and len(nonham_basis) > 0:
    #            sparse = _sps.issparse(nonham_basis[0])
    #    if sparse is None: sparse = False  # the default
    #
    #    if spam_vec is None or spam_vec is pure_vec:
    #        if sparse: errgen = _sps.csr_matrix((d2, d2), dtype='d')
    #        else: errgen = _np.zeros((d2, d2), 'd')
    #    else:
    #        #Construct "spam error generator" by comparing *dense* vectors
    #        pvdense = pure_vec.to_dense()
    #        svdense = spam_vec.to_dense()
    #        errgen = _ot.spam_error_generator(svdense, pvdense, mx_basis)
    #        if sparse: errgen = _sps.csr_matrix(errgen)
    #
    #    assert(pure_vec._evotype == evotype), "`pure_vec` must have evotype == '%s'" % evotype
    #
    #    from .operation import LindbladErrorgen as _LErrorgen
    #    from .operation import LindbladOp as _LPGMap
    #    from .operation import LindbladDenseOp as _LPOp
    #
    #    errgen = _LErrorgen.from_error_generator(errgen, ham_basis,
    #                                             nonham_basis, param_mode, nonham_mode,
    #                                             mx_basis, truncate, evotype)
    #    errcls = _LPOp if (pure_vec.dim <= 64 and evotype == "densitymx") else _LPGMap
    #    errmap = errcls(None, errgen)
    #
    #    return cls(pure_vec, errmap, typ)

    #@classmethod
    #def from_lindblad_terms(cls, pure_vec, lindblad_term_dict, typ, basisdict=None,
    #                        param_mode="cptp", nonham_mode="all", truncate=True,
    #                        mx_basis="pp", evotype="densitymx"):
    #    """
    #    Create a Lindblad-parameterized spamvec with a given set of Lindblad terms.
    #
    #    Parameters
    #    ----------
    #    pure_vec : numpy array or SPAMVec
    #        An array or SPAMVec in the *full* density-matrix space (this
    #        vector will have dimension 4 in the case of a single qubit) which
    #        represents a pure-state preparation or projection.  This is used as
    #        the "base" preparation or projection that is followed or preceded
    #        by, respectively, the parameterized Lindblad-form error generator.
    #
    #    lindblad_term_dict : dict
    #        A dictionary specifying which Linblad terms are present in the gate
    #        parameteriztion.  Keys are `(termType, basisLabel1, <basisLabel2>)`
    #        tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
    #        (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
    #        have a single basis label (so key is a 2-tuple) whereas Stochastic
    #        tuples with 1 basis label indicate a *diagonal* term, and are the
    #        only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
    #        Stochastic term tuples can include 2 basis labels to specify
    #        "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
    #        strings or integers.  Values are complex coefficients (error rates).
    #
    #    typ : {"prep","effect"}
    #        Whether this is a state preparation or POVM effect vector.
    #
    #    basisdict : dict, optional
    #        A dictionary mapping the basis labels (strings or ints) used in the
    #        keys of `lindblad_term_dict` to basis matrices (numpy arrays or Scipy sparse
    #        matrices).
    #
    #    param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
    #        Describes how the Lindblad coefficients/projections relate to the
    #        SPAM vector's parameter values.  Allowed values are:
    #        `"unconstrained"` (coeffs are independent unconstrained parameters),
    #        `"cptp"` (independent parameters but constrained so map is CPTP),
    #        `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
    #        `"depol"` (same as `"reldepol"` but coeffs must be *positive*)
    #
    #    nonham_mode : {"diagonal", "diag_affine", "all"}
    #        Which non-Hamiltonian Lindblad projections are potentially non-zero.
    #        Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
    #        `"diag_affine"` (diagonal coefficients + affine projections), and
    #        `"all"` (the entire matrix of coefficients is allowed).
    #
    #    truncate : bool, optional
    #        Whether to truncate the projections onto the Lindblad terms in
    #        order to meet constraints (e.g. to preserve CPTP) when necessary.
    #        If False, then an error is thrown when the given dictionary of
    #        Lindblad terms doesn't conform to the constrains.
    #
    #    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
    #        The source and destination basis, respectively.  Allowed
    #        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
    #        and Qutrit (qt) (or a custom basis object).
    #
    #    evotype : {"densitymx","svterm","cterm"}
    #        The evolution type of the spamvec being constructed.  `"densitymx"` is
    #        usual Lioville density-matrix-vector propagation via matrix-vector
    #        products.  `"svterm"` denotes state-vector term-based evolution
    #        (spamvec is obtained by evaluating the rank-1 terms up to
    #        some order).  `"cterm"` is similar but stabilizer states.
    #
    #    Returns
    #    -------
    #    LindbladOp
    #    """
    #    #Need a dimension for error map construction (basisdict could be completely empty)
    #    if not isinstance(pure_vec, SPAMVec):
    #        pure_vec = StaticSPAMVec(pure_vec, evotype, typ)  # assume spamvec is just a vector
    #    d2 = pure_vec.dim
    #
    #    from .operation import LindbladOp as _LPGMap
    #    errmap = _LPGMap(d2, lindblad_term_dict, basisdict, param_mode, nonham_mode,
    #                     truncate, mx_basis, evotype)
    #    return cls(pure_vec, errmap, typ)

    def __init__(self, static_effect, errormap):
        evotype = errormap._evotype
        #from .operation import LindbladOp as _LPGMap
        #assert(evotype in ("densitymx", "svterm", "cterm")), \
        #    "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        if not isinstance(static_effect, _POVMEffect):
            # UNSPECIFIED BASIS -- should be able to use static_effect._rep.basis once we get std attribute setup
            static_effect = _StaticState(static_effect, None, evotype)  # assume spamvec is just a vector

        assert(static_effect._evotype == evotype), \
            "`static_effect` evotype must match `errormap` ('%s' != '%s')" % (static_effect._evotype, evotype)
        assert(static_effect.num_params == 0), "`static_effect` 'reference' must have *zero* parameters!"

        self.effect_vec = static_effect
        self.error_map = errormap
        self.terms = {}
        self.local_term_poly_coeffs = {}

        #Create representation
        effectRep = self.effect_vec._rep
        errmapRep = self.error_map._rep
        rep = evotype.create_composed_effect_rep(errmapRep, effectRep, id(self.error_map), static_effect.state_space)
        # an effect that applies a *named* errormap before computing with effectRep

        _POVMEffect.__init__(self, rep, evotype)  # sets self.dim
        self.init_gpindices()  # initialize gpindices and subm_rpindices from sub-members
        #_ErrorMapContainer.__init__(self, self.error_map)

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        error_map = serial_memo[mm_dict['submembers'][0]]
        static_effect = serial_memo[mm_dict['submembers'][1]]
        return cls(static_effect, error_map)

    @property
    def hilbert_schmidt_size(self):
        return self.effect_vec.hilbert_schmidt_size

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.error_map, self.effect_vec]

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
        super().set_gpindices(gpindices, parent, memo)

    def to_dense(self, on_space='minimal', scratch=None):
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
        #Note: self.error_map is the
        # map that acts on the *state* vector before dmVec acts
        # as an effect:  E.T -> dot(E.T,errmap) ==> E -> dot(errmap.T,E)
        return _np.dot(self.error_map.to_dense(on_space).conjugate().T, self.effect_vec.to_dense(on_space))

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
            output of :method:`Polynomial.compact`.
        """
        if order not in self.terms:
            assert(self.gpindices is not None), "ComposedPOVMEffect must be added to a Model before use!"

            state_terms = self.effect_vec.taylor_order_terms(0, max_polynomial_vars); assert(len(state_terms) == 1)
            stateTerm = state_terms[0]
            err_terms, cpolys = self.error_map.taylor_order_terms(order, max_polynomial_vars, True)

            # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
            # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
            # too - the difference comes when this POVMEffect is used as an effect rather than a prep.
            terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's

            #OLD: now this is done within calculator when possible b/c not all terms can be collapsed
            #terms = [ t.collapse() for t in terms ] # collapse terms for speed
            # - resulting in terms with just a single pre/post op, each == a pure state

            #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen
            self.local_term_poly_coeffs[order] = cpolys
            self.terms[order] = terms

        if return_coeff_polys:
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def taylor_order_terms_above_mag(self, order, max_polynomial_vars, min_term_mag):
        """
        Get the `order`-th order Taylor-expansion terms of this POVM effect that have magnitude above `min_term_mag`.

        This function constructs the terms at the given order which have a magnitude (given by
        the absolute value of their coefficient) that is greater than or equal to `min_term_mag`.
        It calls :method:`taylor_order_terms` internally, so that all the terms at order `order`
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
        state_terms = self.effect_vec.taylor_order_terms(0, max_polynomial_vars); assert(len(state_terms) == 1)
        stateTerm = state_terms[0]
        stateTerm = stateTerm.copy_with_magnitude(1.0)
        #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen

        err_terms = self.error_map.taylor_order_terms_above_mag(
            order, max_polynomial_vars, min_term_mag / stateTerm.magnitude)

        #This gives the appropriate logic, but *both* prep or effect results in *same* expression, so just run it:
        #if self._prep_or_effect == "prep":
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        #else:  # "effect"
        #    # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
        #    # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
        #    # too - the difference comes when this POVMEffect is used as an effect rather than a prep.
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        terms = [_term.compose_terms_with_mag((stateTerm, t), stateTerm.magnitude * t.magnitude)
                 for t in err_terms]  # t ops occur *after* stateTerm's
        return terms

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this POVM effect vector's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this POVM effect vector in a Taylor series.

        Returns
        -------
        float
        """
        # return (sum of absvals of *all* term coeffs)
        return self.error_map.total_term_magnitude  # error map is only part with terms

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this POVM effect vector's terms.

        Get the derivative of the total (sum) of the magnitudes of all this
        POVM effect vector's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        return self.error_map.total_term_magnitude_deriv

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
        dmVec = self.effect_vec.to_dense(on_space='minimal')

        derrgen = self.error_map.deriv_wrt_params(wrt_filter)  # shape (dim*dim, n_params)
        derrgen.shape = (self.dim, self.dim, derrgen.shape[1])  # => (dim,dim,n_params)

        # self.error_map acts on the *state* vector before dmVec acts
        # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
        #return _np.einsum("jik,j->ik", derrgen.conjugate(), dmVec) # return shape = (dim,n_params)
        return _np.tensordot(derrgen.conjugate(), dmVec, (0, 0))  # return shape = (dim,n_params)

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this POVM effect vector with respect to its parameters.

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
        dmVec = self.effect_vec.to_dense(on_space='minimal')

        herrgen = self.error_map.hessian_wrt_params(wrt_filter1, wrt_filter2)  # shape (dim*dim, nParams1, nParams2)
        herrgen.shape = (self.dim, self.dim, herrgen.shape[1], herrgen.shape[2])  # => (dim,dim,nParams1, nParams2)

        # self.error_map acts on the *state* vector before dmVec acts
        # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
        #return _np.einsum("jikl,j->ikl", herrgen.conjugate(), dmVec) # return shape = (dim,n_params)
        return _np.tensordot(herrgen.conjugate(), dmVec, (0, 0))  # return shape = (dim,n_params)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.error_map.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM effect vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.error_map.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.error_map.to_vector()

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
        self.error_map.from_vector(v, close, dirty_value)
        self.dirty = dirty_value

    def transform_inplace(self, s):
        """
        Update POVM effect (column) vector V as inv(s) * V or s^T * V

        Note that this is equivalent to state preparation vectors getting
        mapped: `rho -> inv(s) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * s`.

        Generally, the transform function updates the *parameters* of
        the POVM effect vector such that the resulting vector is altered as
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
        #Defer everything to LindbladOp's
        # `spam_tranform` function, which applies either
        # error_map -> inv(s) * error_map ("prep" case) OR
        # error_map -> error_map * s      ("effect" case)
        self.error_map.spam_transform_inplace(s, 'effect')
        self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this POVM effect vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the POVMEffect such that the resulting vector is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the spam vector. All but
            the first element of the spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        self.error_map.depolarize(amount)
