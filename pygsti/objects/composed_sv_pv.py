"""
TEMPORARY: Defines the Composed SPAMVec and POVM classes
Will be merged into spamvec.py and povm.py after evotype refactor merge
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
from pygsti.objects.operation import ComposedOp
import numpy as _np

#from . import labeldicts as _ld
from . import modelmember as _gm
from . import replib
from . import term as _term
from .povm import POVM, ComputationalBasisPOVM
from .spamvec import ComputationalSPAMVec, SPAMVec


class ComposedSPAMVec(SPAMVec):
    def __init__(self, pure_vec, noise_op, typ):
        """
        Initialize a ComposedSPAMVec object.

        Essentially a pure state preparation or projection that is followed
        or preceded by, respectively, the action of a general LinearOperator.
        The state prep/projection must be parameterized by a ComputationalSPAMVec,
        currently.

        Parameters
        ----------
        pure_vec : numpy array or ComputationalSPAMVec
            An array of zvals or a ComputationalSPAMVec which
            represents a pure-state preparation or projection. This is used as
            the "base" preparation or projection that is followed or preceded
            by, respectively, the noisy operation.
            (This argument is *not* copied if it is a ComputationalSPAMVec;
            otherwise, it is converted to a new ComputationalSPAMVec.)

        noise_op : LinearOperator
            The noisy operation to follow or precede the pure SPAMVec
            (This argument is *not* copied to allow the LinearOperator
            to share error parameters with other gates and spam vectors.)

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.
        """
        evotype = noise_op._evotype
        assert(evotype in ("densitymx", "svterm", "cterm", "chp")), \
            "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        if not isinstance(pure_vec, ComputationalSPAMVec):
            pure_vec = ComputationalSPAMVec(pure_vec, evotype, typ)  # assume spamvec is just a vector

        assert(pure_vec._evotype == evotype), \
            "`pure_vec` evotype must match `noise_op` ('%s' != '%s')" % (pure_vec._evotype, evotype)
        assert(pure_vec.num_params == 0), "`pure_vec` 'reference' must have *zero* parameters!"

        d2 = pure_vec.dim
        self.state_vec = pure_vec
        self.noise_op = noise_op
        self.terms = {} if evotype in ("svterm", "cterm") else None
        self.local_term_poly_coeffs = {} if evotype in ("svterm", "cterm") else None
        # TODO REMOVE self.direct_terms = {} if evotype in ("svterm","cterm") else None
        # TODO REMOVE self.direct_term_poly_coeffs = {} if evotype in ("svterm","cterm") else None

        #Create representation
        if evotype == "densitymx":
            assert(self.state_vec._prep_or_effect == typ), \
                "ComposedSPAMVec prep/effect mismatch with given statevec!"

            if typ == "prep":
                dmRep = self.state_vec._rep
                errmapRep = self.noise_op._rep
                rep = errmapRep.acton(dmRep)  # FUTURE: do this acton in place somehow? (like C-reps do)
                #maybe make a special _Errgen *state* rep?

            else:  # effect
                dmEffectRep = self.state_vec._rep
                errmapRep = self.noise_op._rep
                # TODO: This is vestigial from LindbladOp, may or may not always work...
                rep = replib.DMEffectRepErrgen(errmapRep, dmEffectRep, id(self.noise_op))
                # an effect that applies a *named* errormap before computing with dmEffectRep
        else:
            rep = d2  # no representations for term-based evotypes or chp

        SPAMVec.__init__(self, rep, evotype, typ)  # sets self.dim

    def _update_rep(self):
        if self._evotype == "densitymx":
            if self._prep_or_effect == "prep":
                # _rep is a DMStateRep
                dmRep = self.state_vec._rep
                errmapRep = self.noise_op._rep
                self._rep.base[:] = errmapRep.acton(dmRep).base[:]  # copy from "new_rep"

            else:  # effect
                # _rep is a DMEffectRepErrgen, which just holds references to the
                # effect and error map's representations (which we assume have been updated)
                # so there's no need to update anything here
                pass

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.noise_op]

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
        # We need to override this method so that embedded gate has its
        # parent reset correctly.
        if memo is not None and id(self) in memo: return memo[id(self)]
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.state_vec, self.noise_op.copy(parent), self._prep_or_effect)
        return self._copy_gpindices(copyOfMe, parent, memo)

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
        # All parameters should belong to noise op
        #self.noise_op.set_gpindices(gpindices, parent, memo)
        _gm.ModelMember.set_gpindices(self, gpindices, parent, memo)

    def to_dense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        if self._prep_or_effect == "prep":
            #error map acts on dmVec
            return _np.dot(self.noise_op.to_dense(), self.state_vec.to_dense())
        else:
            #Note: if this is an effect vector, self.error_map is the
            # map that acts on the *state* vector before dmVec acts
            # as an effect:  E.T -> dot(E.T,errmap) ==> E -> dot(errmap.T,E)
            return _np.dot(self.noise_op.to_dense().conjugate().T, self.state_vec.to_dense())
        
    def set_dense(self, vec):
        """
        Set the dense-vector value of this SPAM vector.

        Attempts to modify this SPAM vector's parameters so that the raw
        SPAM vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        This does not modify the noisy operation/error map, but only the
        SPAM vector itself.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        self.state_vec.set_dense(vec)
        self.dirty = True

    #def torep(self, typ, outvec=None):
    #    """
    #    Return a "representation" object for this SPAMVec.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    StateRep
    #    """
    #    if self._evotype == "densitymx":
    #
    #        if typ == "prep":
    #            dmRep = self.state_vec.torep(typ)
    #            errmapRep = self.error_map.torep()
    #            return errmapRep.acton(dmRep)  # FUTURE: do this acton in place somehow? (like C-reps do)
    #            #maybe make a special _Errgen *state* rep?
    #
    #        else:  # effect
    #            dmEffectRep = self.state_vec.torep(typ)
    #            errmapRep = self.error_map.torep()
    #            return replib.DMEffectRepErrgen(errmapRep, dmEffectRep, id(self.error_map))
    #            # an effect that applies a *named* errormap before computing with dmEffectRep
    #
    #    else:
    #        #framework should not be calling "torep" on states w/a term-based evotype...
    #        # they should call torep on the *terms* given by taylor_order_terms(...)
    #        raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
    #                         (self._evotype, self.__class__.__name__))

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
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
            if self._evotype not in ('svterm', 'cterm'):
                raise ValueError("Invalid evolution type %s for calling `taylor_order_terms`" % self._evotype)
            assert(self.gpindices is not None), "ComposedSPAMVec must be added to a Model before use!"

            state_terms = self.state_vec.taylor_order_terms(0, max_polynomial_vars); assert(len(state_terms) == 1)
            stateTerm = state_terms[0]
            err_terms, cpolys = self.noise_op.taylor_order_terms(order, max_polynomial_vars, True)
            if self._prep_or_effect == "prep":
                terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
            else:  # "effect"
                # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
                # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
                # too - the difference comes when this SPAMVec is used as an effect rather than a prep.
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
        Get the `order`-th order Taylor-expansion terms of this SPAM vector that have magnitude above `min_term_mag`.

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
        state_terms = self.state_vec.taylor_order_terms(0, max_polynomial_vars); assert(len(state_terms) == 1)
        stateTerm = state_terms[0]
        stateTerm = stateTerm.copy_with_magnitude(1.0)
        #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen

        err_terms = self.noise_op.taylor_order_terms_above_mag(
            order, max_polynomial_vars, min_term_mag / stateTerm.magnitude)

        #This gives the appropriate logic, but *both* prep or effect results in *same* expression, so just run it:
        #if self._prep_or_effect == "prep":
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        #else:  # "effect"
        #    # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
        #    # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
        #    # too - the difference comes when this SPAMVec is used as an effect rather than a prep.
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        terms = [_term.compose_terms_with_mag((stateTerm, t), stateTerm.magnitude * t.magnitude)
                 for t in err_terms]  # t ops occur *after* stateTerm's
        return terms

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this SPAM vector's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this SPAM vector in a Taylor series.

        Returns
        -------
        float
        """
        # return (sum of absvals of *all* term coeffs)
        return self.noise_op.total_term_magnitude  # error map is only part with terms

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this SPAM vector's terms.

        Get the derivative of the total (sum) of the magnitudes of all this
        SPAM vector's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        return self.noise_op.total_term_magnitude_deriv

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this SPAM vector.

        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per SPAM vector parameter.

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
        dmVec = self.state_vec.to_dense()

        derrgen = self.noise_op.deriv_wrt_params(wrt_filter)  # shape (dim*dim, n_params)
        derrgen.shape = (self.dim, self.dim, derrgen.shape[1])  # => (dim,dim,n_params)

        if self._prep_or_effect == "prep":
            #derror map acts on dmVec
            #return _np.einsum("ijk,j->ik", derrgen, dmVec) # return shape = (dim,n_params)
            return _np.tensordot(derrgen, dmVec, (1, 0))  # return shape = (dim,n_params)
        else:
            # self.error_map acts on the *state* vector before dmVec acts
            # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
            #return _np.einsum("jik,j->ik", derrgen.conjugate(), dmVec) # return shape = (dim,n_params)
            return _np.tensordot(derrgen.conjugate(), dmVec, (0, 0))  # return shape = (dim,n_params)

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

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
        dmVec = self.state_vec.to_dense()

        herrgen = self.noise_op.hessian_wrt_params(wrt_filter1, wrt_filter2)  # shape (dim*dim, nParams1, nParams2)
        herrgen.shape = (self.dim, self.dim, herrgen.shape[1], herrgen.shape[2])  # => (dim,dim,nParams1, nParams2)

        if self._prep_or_effect == "prep":
            #derror map acts on dmVec
            #return _np.einsum("ijkl,j->ikl", herrgen, dmVec) # return shape = (dim,n_params)
            return _np.tensordot(herrgen, dmVec, (1, 0))  # return shape = (dim,n_params)
        else:
            # self.error_map acts on the *state* vector before dmVec acts
            # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
            #return _np.einsum("jikl,j->ikl", herrgen.conjugate(), dmVec) # return shape = (dim,n_params)
            return _np.tensordot(herrgen.conjugate(), dmVec, (0, 0))  # return shape = (dim,n_params)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.noise_op.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.noise_op.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.noise_op.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of SPAM vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this SPAM vector's current
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
        self.noise_op.from_vector(v, close, dirty_value)
        self._update_rep()
        self.dirty = dirty_value

    def transform_inplace(self, s, typ):
        """
        Update SPAM (column) vector V as inv(s) * V or s^T * V for preparation or  effect SPAM vectors, respectively.

        Note that this is equivalent to state preparation vectors getting
        mapped: `rho -> inv(s) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * s`.

        Generally, the transform function updates the *parameters* of
        the SPAM vector such that the resulting vector is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).

        Returns
        -------
        None
        """
        #Defer everything to LinearOperator's
        # `spam_tranform` function, which applies either
        # error_map -> inv(s) * error_map ("prep" case) OR
        # error_map -> error_map * s      ("effect" case)
        self.noise_op.spam_transform_inplace(s, typ)
        self._update_rep()
        self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this SPAM vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the SPAMVec such that the resulting vector is depolarized.  If
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
        self.noise_op.depolarize(amount)
        self._update_rep()


class ComposedPOVM(POVM):
    """
    A POVM that is effectively a *single* noisy gate followed by a computational-basis POVM.

    Parameters
    ----------
    noise_op : LinearOperator
        The error generator action and parameterization, encapsulated in
        a gate object. (This argument is *not* copied,to allow
        ComposedPOVMs to share parameters with other gates and spam vectors.)

    povm : POVM, optional
        A sub-POVM which supplies the set of "reference" effect vectors
        that `noise_op` acts on to produce the final effect vectors of
        this LindbladPOVM.  This POVM must be *static*
        (have zero parameters) and its evolution type must match that of
        `noise_op`.  If None, then a :class:`ComputationalBasisPOVM` is
        used on the number of qubits appropriate to `noise_op`'s dimension.
    """

    def __init__(self, noise_op, povm=None):
        """
        Creates a new ComposedPOVM object.

        Parameters
        ----------
        noise_op : MapOperator
            The error generator action and parameterization, encapsulated in
            a gate object.  Usually a :class:`LindbladOp`
            or :class:`ComposedOp` object.  (This argument is *not* copied,
            to allow LindbladSPAMVecs to share error generator
            parameters with other gates and spam vectors.)

        povm : POVM, optional
            A sub-POVM which supplies the set of "reference" effect vectors
            that `noise_op` acts on to produce the final effect vectors of
            this LindbladPOVM.  This POVM must be *static*
            (have zero parameters) and its evolution type must match that of
            `noise_op`.  If None, then a :class:`ComputationalBasisPOVM` is
            used on the number of qubits appropriate to `noise_op`'s dimension.
        """
        self.noise_op = noise_op
        dim = self.noise_op.dim
        evotype = self.noise_op._evotype

        if povm is None:
            factor = 2 if evotype in ['densitymx', 'svterm', 'cterm'] else 1
            nqubits = int(round(_np.log2(dim) / factor))
            assert(_np.isclose(nqubits, _np.log2(dim) / factor)), \
                ("A default computational-basis POVM can only be used with an"
                 " integral number of qubits!")
            povm = ComputationalBasisPOVM(nqubits, evotype)
        else:
            assert(povm._evotype == evotype), \
                ("Evolution type of `povm` (%s) must match that of "
                 "`noise_op` (%s)!") % (povm._evotype, evotype)
            assert(povm.num_params == 0), \
                "Given `povm` must be static (have 0 parameters)!"
        self.base_povm = povm

        items = []  # init as empty (lazy creation of members)
        POVM.__init__(self, dim, evotype, items)

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        return bool(key in self.base_povm)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self.base_povm)

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in self.base_povm.keys():
            yield k

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            pureVec = self.base_povm[key]
            effect = ComposedSPAMVec(pureVec, self.noise_op, "effect")
            effect.set_gpindices(self.noise_op.gpindices, self.parent)
            # initialize gpindices of "child" effect (should be in simplify_effects?)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this ComposedPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (ComposedPOVM, (self.noise_op.copy(), self.base_povm.copy()),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    def allocate_gpindices(self, starting_index, parent, memo=None):
        """
        Sets gpindices array for this object or any objects it contains (i.e. depends upon).

        Indices may be obtained from contained objects which have already been
        initialized (e.g. if a contained object is shared with other top-level
        objects), or given new indices starting with `starting_index`.

        Parameters
        ----------
        starting_index : int
            The starting index for un-allocated parameters.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : set, optional
            Used to prevent duplicate calls and self-referencing loops.  If
            `memo` contains an object's id (`id(self)`) then this routine
            will exit immediately.

        Returns
        -------
        num_new : int
            The number of *new* allocated parameters (so
            the parent should mark as allocated parameter
            indices `starting_index` to `starting_index + new_new`).
        """
        if memo is None: memo = set()
        if id(self) in memo: return 0
        memo.add(id(self))

        assert(self.base_povm.num_params == 0)  # so no need to do anything w/base_povm
        num_new_params = self.noise_op.allocate_gpindices(starting_index, parent, memo)  # *same* parent as self
        _gm.ModelMember.set_gpindices(
            self, self.noise_op.gpindices, parent)
        return num_new_params

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.noise_op]

    def relink_parent(self, parent):  # Unnecessary?
        """
        Sets the parent of this object *without* altering its gpindices.

        In addition to setting the parent of this object, this method
        sets the parent of any objects this object contains (i.e.
        depends upon) - much like allocate_gpindices.  To ensure a valid
        parent is not overwritten, the existing parent *must be None*
        prior to this call.

        Parameters
        ----------
        parent : Model or ModelMember
            The parent of this POVM.

        Returns
        -------
        None
        """
        self.noise_op.relink_parent(parent)
        _gm.ModelMember.relink_parent(self, parent)

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
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        assert(self.base_povm.num_params == 0)  # so no need to do anything w/base_povm
        self.noise_op.set_gpindices(gpindices, parent, memo)
        self.terms = {}  # clear terms cache since param indices have changed now
        _gm.ModelMember._set_only_my_gpindices(self, gpindices, parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.noise_op.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        # Recall self.base_povm.num_params == 0
        return self.noise_op.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        # Recall self.base_povm.num_params == 0
        return self.noise_op.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
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
        # Recall self.base_povm.num_params == 0
        self.noise_op.from_vector(v, close, dirty_value)

    def transform_inplace(self, s):
        """
        Update each POVM effect E as s^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        self.noise_op.spam_transform_inplace(s, 'effect')  # only do this *once*
        for lbl, effect in self.items():
            effect._update_rep()  # these two lines mimic the bookeepging in
            effect.dirty = True   # a "effect.transform_inplace(s, 'effect')" call.
        self.dirty = True
    
    def depolarize(self, amount):
        """
        Depolarize this POVM by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        self.noise_op.depolarize(amount)
        for lbl, effect in self.items():
            effect._update_rep()  # these two lines mimic the bookeepging in
            effect.dirty = True   # a "effect.transform_inplace(s, 'effect')" call.
        self.dirty = True

    def __str__(self):
        s = "Composed POVM of length %d\n" \
            % (len(self))
        return s
