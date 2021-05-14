"""
Defines classes which represent terms in gate expansions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import math as _math
import numpy as _np
import itertools as _itertools
import numbers as _numbers
import functools as _functools
from operator import add as _add


from .polynomial import Polynomial as _Polynomial
from . import replib
from ..modelmembers import modelmember as _mm
from ..modelmembers import operations as _op
from ..modelmembers import states as _state
from ..modelmembers import povms as _povm

def compose_terms_with_mag(terms, magnitude):
    """
    Compose multiple terms *and* set the resulting term's magnitude.

    Usually, for performance reasons, the magnitude of a composed term
    is not set.  This function explicitly does so.

    Parameters
    ----------
    terms : sequence
        A sequence of :class:`RankOneTerm`s.

    magnitude : float
        The magnitude of the compsed term.

    Returns
    -------
    RankOneTerm
    """
    assert(len(terms) > 0)
    return terms[0].compose(terms, magnitude)


def compose_terms(terms):
    """
    Compose a sequence of terms.

    Composition is done with *time* ordered left-to-right. Thus composition
    order is NOT the same as usual matrix order.
    E.g. if there are three terms:
    `terms[0]` = T0: rho -> A*rho*A
    `terms[1]` = T1: rho -> B*rho*B
    `terms[2]` = T2: rho -> C*rho*C
    Then the resulting term T = T0*T1*T2 : rho -> CBA*rho*ABC, so
    that term[0] is applied *first* not last to a state.

    Parameters
    ----------
    terms : list
        A list of :class:`RankOneTerm`s to compose.

    Returns
    -------
    RankOneTerm
    """
    assert(len(terms) > 0)  # otherwise return something like RankOneTerm(1.0, None, None)?
    return terms[0].compose(terms)


def exponentiate_terms(terms, order, postterm, cache=None, order_base=None):
    """
    Exponentiate a list of terms.

    This function collects those terms of the orders given
    in `orders`. Optionally post-multiplies the single term `postterm` (so this
    term actually acts *before* the exponential-derived terms).

    Parameters
    ----------
    terms : list
        The list of terms to exponentiate.  All these terms are
        considered "first order" terms.

    order : int
        An integers specifying the order of the terms to collect.

    postterm : RankOneTerm
        A term that is composed *first* (so "post" in the sense of matrix
        multiplication, not composition).  May be the identity (no-op) term.

    cache : dict, optional
        A dictionary used to cache results for speeding up repeated calls
        to this function.  Usually an empty dictionary is supplied to the
        first call.

    order_base : float, optional
        What constitutes 1 order of magnitude.  If None, then
        polynomial coefficients are used.

    Returns
    -------
    list
        A list of :class:`RankOneTerm` objects giving the terms at `order`.
    """

    #FUTURE: add "term_order" argument to specify what order a term in `terms`
    # is considered to be (not all necessarily = 1)
    persistent_cache = cache is not None
    if not persistent_cache: cache = {}

    #create terms for each order from terms and base action

    # expand exp(L) = I + L + 1/2! L^2 + ... (n-th term 1/n! L^n)
    # expand 1/n! L^n into a list of rank-1 terms -> cache[n]
    cache[0] = [postterm]
    #one_over_factorial = 1 / _np.math.factorial(order)

    def build_terms(order_to_build):
        if order_to_build in cache:  # Note: 0th order is *always* in cache
            return cache[order_to_build]
        previous_order_terms = build_terms(order_to_build - 1)
        a = 1.0 / order_to_build  # builds up 1/factorial prefactor
        premultiplied_terms = [a * factor for factor in terms]
        cache[order_to_build] = [compose_terms((previous_order_term, a_factor))
                                 for previous_order_term in previous_order_terms
                                 for a_factor in premultiplied_terms]
        return cache[order_to_build]

    if persistent_cache:
        return [t.copy() for t in build_terms(order)]  # copy the terms if we need to store them in cache
    else:
        return build_terms(order)


def exponentiate_terms_above_mag(terms, order, postterm, cache=None, min_term_mag=None):
    """
    Exponentiate a list of terms with magnitude above `min_term_mag`.

    This function collects those terms of the orders given
    in `orders`. Optionally post-multiplies the single term `postterm` (so this
    term actually acts *before* the exponential-derived terms).

    Parameters
    ----------
    terms : list
        The list of terms to exponentiate.  All these terms are
        considered "first order" terms.

    order : int
        Integer specifying the order to compute.

    postterm : RankOneTerm
        A term that is composed *first* (so "post" in the sense of matrix
        multiplication, not composition).

    cache : dict, optional
        A cache to speedup repeated calls.  Currently unused because
        `min_term_mag` is likely to be different on every call.

    min_term_mag : float, optional
        The minimum term magnitude.

    Returns
    -------
    list
    """
    cache = {}  # DON'T allow the user to pass in a previous cache
    # since this may have used a different min_term_mag
    #Note: for performance, `cache` stores only term-*reps* not the full
    # term objects themselves.  Term objects are build (wrapped around
    # reps) at the end based on the type of postterm.

    #create terms for each order from terms and base action

    # expand exp(L) = I + L + 1/2! L^2 + ... (n-th term 1/n! L^n)
    # expand 1/n! L^n into a list of rank-1 terms -> cache[n]
    cache[0] = [postterm._rep]
    termType = postterm.__class__
    composeFn = postterm._rep.__class__.composed
    termreps = [t._rep for t in terms]

    def build_terms(order_to_build):
        if order_to_build in cache:  # Note: 0th order is *always* in cache
            return cache[order_to_build]
        previous_order_terms = build_terms(order_to_build - 1)
        a = 1.0 / order_to_build  # builds up 1/factorial prefactor

        #OLD - when we used full objects
        #premultiplied_terms = [ a * factor for factor in terms ]  # terms are expected to have their magnitudes set.
        # cache[order_to_build] = [t for t in (compose_terms_with_mag(
        #     (previous_order_term, a_factor),
        #     previous_order_term.magnitude * a_factor.magnitude)
        #                                      for previous_order_term in previous_order_terms
        #                                      for a_factor in premultiplied_terms) if t.magnitude >= min_term_mag ]

        # terms are expected to have their magnitudes set.
        premultiplied_terms = [factor.scalar_mult(a) for factor in termreps]
        tuples_to_iter = [(previous_order_term, a_factor)
                          for previous_order_term in previous_order_terms
                          for a_factor in premultiplied_terms
                          if previous_order_term.magnitude * a_factor.magnitude >= min_term_mag]
        cache[order_to_build] = [composeFn((previous_order_term, a_factor),
                                           previous_order_term.magnitude * a_factor.magnitude)
                                 for (previous_order_term, a_factor) in tuples_to_iter]

        # **Assume** individual term magnitudes are <= 1.0 so that we
        # don't include any order terms that have magnitude < min_term_mag.
        return cache[order_to_build]

    #return build_terms(order)  #OLD - when cache held full objects
    return [termType(rep) for rep in build_terms(order)]


def _embed_oprep(state_space_labels, target_labels, rep_to_embed, evotype):
    """Variant of EmbeddedOp.__init__ used to create embeddedop reps without a corresponding embedded op.
    For use w/terms where there are just reps
    """

    opDim = state_space_labels.dim

    #Create representation
    if evotype == "stabilizer":
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        assert(len(state_space_labels.labels) == 1
               and all([ld == 2 for ld in state_space_labels.labeldims.values()])), \
            "All state space labels must correspond to *qubits*"

        #Cache info to speedup representation's acton(...) methods:
        # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
        qubitLabels = state_space_labels.labels[0]
        qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                   for targetLbl in target_labels], _np.int64)

        nQubits = int(round(_np.log2(opDim)))
        rep = replib.SBOpRepEmbedded(rep_to_embed,
                                     nQubits, qubit_indices)

    elif evotype in ("statevec", "densitymx"):

        iTensorProdBlks = [state_space_labels.tpb_index[label] for label in target_labels]
        # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) != 1:
            raise ValueError("All qubit labels of a multi-qubit gate must correspond to the"
                             " same tensor-product-block of the state space -- checked previously")  # pragma: no cover # noqa

        iTensorProdBlk = iTensorProdBlks[0]  # because they're all the same (tested above) - this is "active" block
        tensorProdBlkLabels = state_space_labels.labels[iTensorProdBlk]
        # count possible *density-matrix-space* indices of each component of the tensor product block
        numBasisEls = _np.array([state_space_labels.labeldims[l] for l in tensorProdBlkLabels], _np.int64)

        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        actionInds = _np.array(labelIndices, _np.int64)
        assert(_np.product([numBasisEls[i] for i in actionInds]) == rep_to_embed.dim), \
            "Embedded gate has dimension (%d) inconsistent with the given target labels (%s)" % (
                rep_to_embed.dim, str(target_labels))

        nBlocks = state_space_labels.num_tensor_prod_blocks()
        iActiveBlock = iTensorProdBlk
        nComponents = len(state_space_labels.labels[iActiveBlock])
        embeddedDim = rep_to_embed.dim
        blocksizes = _np.array([_np.product(state_space_labels.tensor_product_block_dims(k))
                                for k in range(nBlocks)], _np.int64)
        if evotype == "statevec":
            rep = replib.SVOpRepEmbedded(rep_to_embed,
                                         numBasisEls, actionInds, blocksizes, embeddedDim,
                                         nComponents, iActiveBlock, nBlocks, opDim)
        else:  # "densitymx"
            rep = replib.DMOpRepEmbedded(rep_to_embed,
                                         numBasisEls, actionInds, blocksizes, embeddedDim,
                                         nComponents, iActiveBlock, nBlocks, opDim)
    else:
        raise ValueError("Invalid evotype `%s`" % evotype)
    return rep

#def embed_term(term, state_space_labels, target_labels):
#    """
#    Embed a term to it acts within a larger state space.
#
#    Internally, this simply converts a term's gate operators to embedded gate
#    operations.
#
#    Parameters
#    ----------
#    term : RankOneTerm
#        The term to embed
#
#    state_space_labels : a list of tuples
#        This argument specifies the density matrix space upon which the
#        constructed term will act.  Each tuple corresponds to a block of a
#        density matrix in the standard basis (and therefore a component of
#        the direct-sum density matrix space).
#
#    target_labels : list
#        The labels contained in `state_space_labels` which demarcate the
#        portions of the state space acted on by `term`.
#
#    Returns
#    -------
#    RankOneTerm
#    """
#    from ..modelmembers import operations as _op
#    ret = RankOneTerm(term.coeff, None, None, term.termtype, term._evotype)
#    ret.pre_ops = [_op.EmbeddedOp(state_space_labels, target_labels, op)
#                   for op in term.pre_ops]
#    ret.post_ops = [_op.EmbeddedOp(state_space_labels, target_labels, op)
#                    for op in term.post_ops]
#    return ret


class RankOneTerm(object):
    """
    An operation, like a gate, that maps a density matrix to another density matrix but in a restricted way.

    While a RankOneTerm doesn't have to map pure states to pure states, its action
    can be written:

    `rho -> A*rho*B`

    Where `A` and `B` are unitary state operations.  This means that if `rho`
    can be written `rho = |psi1><psi2|` then the action of a RankOneTerm
    preserves the separable nature or `rho` (which need not always be a valid
    density matrix since it can be just a portion of one).

    A RankOneTerm anticipates its application to "separable" (as defined above)
    states, and can even be used to represent such a separable state or an
    analagous POVM effect.  This occurs when the first element of `pre_ops` and
    `post_ops` is a preparation or POVM effect vector instead of a gate operation.

    Note that operations are stored in *composition (time) order* rather than
    matrix order, and that adjoint operations are stored in `post_ops` so that
    they can be applied directly to the adjoint of the "bra" part of the state
    (which is a "ket" - a usual state).

    Finally, a coefficient (usually a number or a :class:`Polynomial`) is held,
    representing the prefactor for this term as a part of a larger density
    matrix evolution.

    Parameters
    ----------
    rep : SVTermRep or SBTermRep
        The term representation object the acts as the core of this term.
    """

    def __init__(self, rep):
        self._rep = rep

    def torep(self):
        """
        Access to the underlying representation object.
        """
        return self._rep

    def copy(self):
        """
        Copies this term.

        Returns
        -------
        RankOneTerm
        """
        return self.__class__(self._rep.copy())

    def __mul__(self, x):
        """ Multiply by scalar """
        return self.__class__(self._rep.scalar_mult(x))

    def __rmul__(self, x):
        return self.__mul__(x)

    #Not needed - but we would use this if we changed
    # the "effect term" convention so that the pre/post ops
    # were associated with the pre/post effect vector and
    # not vice versa (right now the post effect is preceded
    # by the *pre* ops, and vice versa).  If the reverse
    # were true we'd need to conjugate the terms created
    # for effect-type LindbladSPAMVec objects, for example.
    #def conjugate(self):
    #    return self.__class__(self._rep.conjugate())


class _HasMagnitude(object):
    """
    A base class that adds a `magnitude` property to a term class.

    Magnitudes are used in pruned-path or direct term-based (path
    integral) forward simulation.

    Attributes
    ----------
    magnitude : float
        This term's magnitude.

    logmagnitude : float
        The logarithm of this term's magnitude (held separately for performance).
    """

    @property
    def magnitude(self):
        """
        This term's magnitude.
        """
        return self._rep.magnitude

    @property
    def logmagnitude(self):
        """
        The logarithm of this term's magnitude (held separately for performance).
        """
        return self._rep.logmagnitude

    #WHY NOT A CLASSMETHOD? -- needs to be inherited by derived classes, but I think that should work.
    def compose(self, all_terms, magnitude):
        """
        Compose `all_terms` and set the composed term's magnitude to `magnitude`.

        Parameters
        ----------
        all_terms : list
            List of the terms to compose.

        magnitude : float
            The magnitude of the composed term.

        Returns
        -------
        RankOneTerm
        """
        return self.__class__(self._rep.__class__.composed([t._rep for t in all_terms], magnitude))


class _NoMagnitude(object):
    """
    A base class that adds a `magnitude` property to a term class.
    """
    def compose(self, all_terms):
        """
        Compose `all_terms` and set the composed term's magnitude to `magnitude`.

        Parameters
        ----------
        all_terms : list
            List of the terms to compose.

        Returns
        -------
        RankOneTerm
        """
        return self.__class__(self._rep.__class__.composed([t._rep for t in all_terms], 1.0))


class RankOnePrepTerm(RankOneTerm, _NoMagnitude):
    """
    A state preparation term.
    """

    @classmethod
    def create_from(cls, coeff, pre_state, post_state, evotype):
        """
        Creates a :class:`RankOnePrepTerm`s using natural arguments.

        Parameters
        ----------
        coeff : Polynomial or complex
            The term's coefficient.

        pre_state : SPAMVec
            The 'ket' (left-side) state of the operator.

        post_state : SPAMVec
            The 'bra' (right-side) state of the operator

        evotype : str
            The evolution type

        Returns
        -------
        RankOnePrepTerm
        """
        if evotype not in ('svterm', 'cterm'):
            raise ValueError("Invalid evotype: %s" % evotype)

        if not isinstance(pre_state, _mm.ModelMember):
            if evotype == "svterm":
                pre_state = _state.StaticState(pre_state, "statevec")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)
        if not isinstance(post_state, _mm.ModelMember):
            if evotype == "svterm":
                pre_state = _state.StaticState(post_state, "statevec")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)

        reptype = replib.SVTermRep if (evotype == "svterm") else replib.SBTermRep
        rep = reptype(cls._coeff_rep(coeff), 1.0, 0.0,
                      pre_state._rep, post_state._rep, None, None, [], [])
        return cls(rep)

    def embed(self, state_space_labels, target_labels):
        """
        Embeds this term's action as a part of a larger state space.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels
            The state space labels for the full space.

        target_labels : tuple
            A tuple of the target labels.

        Returns
        -------
        RankOnePrepTerm
        """
        evotype = "statevec" if isinstance(self._rep, replib.SVTermRep) else "stabilizer"
        pre_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                   for oprep in self._rep.pre_ops]
        post_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                    for oprep in self._rep.post_ops]
        return self.__class__(self._rep.__class__(self._rep.coeff, 1.0, 0.0, self._rep.pre_state, self._rep.post_state,
                                                  None, None, pre_ops, post_ops))


class RankOneEffectTerm(RankOneTerm, _NoMagnitude):
    """
    A POVM effect term.
    """

    @classmethod
    def create_from(cls, coeff, pre_effect, post_effect, evotype):
        """
        Creates a :class:`RankOneEffectTerm`s using natural arguments.

        Parameters
        ----------
        coeff : Polynomial or complex
            The term's coefficient.

        pre_effect : SPAMVec
            The 'bra' (left-side) effect of the operator.

        post_effect : SPAMVec
            The 'ket' (right-side) effect of the operator

        evotype : str
            The evolution type

        Returns
        -------
        RankOneEffectTerm
        """
        if evotype not in ('svterm', 'cterm'):
            raise ValueError("Invalid evotype: %s" % evotype)

        if not isinstance(pre_effect, _mm.ModelMember):
            if evotype == "svterm":
                pre_effect = _povm.StaticPOVMEffect(pre_effect, "statevec")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)
        if not isinstance(post_effect, _mm.ModelMember):
            if evotype == "svterm":
                pre_effect = _povm.StaticPOVMEffect(post_effect, "statevec")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)

        reptype = replib.SVTermRep if (evotype == "svterm") \
            else replib.SBTermRep
        rep = reptype(cls._coeff_rep(coeff), 1.0, 0.0,
                      None, None, pre_effect._rep, post_effect._rep,
                      [], [])
        return cls(rep)

    def embed(self, state_space_labels, target_labels):
        """
        Embeds this term's action as a part of a larger state space.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels
            The state space labels for the full space.

        target_labels : tuple
            A tuple of the target labels.

        Returns
        -------
        RankOneEffectTerm
        """
        evotype = "statevec" if isinstance(self._rep, replib.SVTermRep) else "stabilizer"
        pre_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                   for oprep in self._rep.pre_ops]
        post_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                    for oprep in self._rep.post_ops]
        return self.__class__(self._rep.__class__(self._rep.coeff, 1.0, 0.0, None, None,
                                                  self._rep.pre_effect, self._rep.post_effect, pre_ops, post_ops))


class RankOneOpTerm(RankOneTerm, _NoMagnitude):
    """
    An operation term.
    """

    @classmethod
    def create_from(cls, coeff, pre_op, post_op, evotype):
        """
        Creates a :class:`RankOneOpTerm`s using natural arguments.

        Parameters
        ----------
        coeff : Polynomial or complex
            The term's coefficient.

        pre_op : SPAMVec
            The left-side operator, i.e. `A` in `rho => A rho B^dag`.

        post_op : SPAMVec
            The right-side operator, i.e. `B` in `rho => A rho B^dag`.

        evotype : str
            The evolution type

        Returns
        -------
        RankOneOpTerm
        """
        if evotype not in ('svterm', 'cterm'):
            raise ValueError("Invalid evotype: %s" % evotype)

        pre_ops = []
        post_ops = []

        if pre_op is not None:
            if not isinstance(pre_op, _mm.ModelMember):
                if evotype == "svterm":
                    pre_op = _op.StaticDenseOp(pre_op, "statevec")
                elif evotype == "cterm":
                    pre_op = _op.CliffordOp(pre_op)  # ------------------------------------------ evotype ??????????????????????
                else:
                    raise ValueError("Invalid `evotype` argument: %s" % evotype)
            pre_ops.append(pre_op)

        if post_op is not None:
            if not isinstance(post_op, _mm.ModelMember):
                if evotype == "svterm":
                    post_op = _op.StaticDenseOp(post_op, "statevec")
                elif evotype == "cterm":
                    post_op = _op.CliffordOp(post_op)  # ------------------------------------------ evotype ??????????????????????
                else:
                    raise ValueError("Invalid `evotype` argument: %s" % evotype)
            post_ops.append(post_op)

        reptype = replib.SVTermRep if (evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in pre_ops]
        post_op_reps = [op._rep for op in post_ops]
        rep = reptype(cls._coeff_rep(coeff), 1.0, 0.0,
                      None, None, None, None,
                      pre_op_reps, post_op_reps)
        return cls(rep)

    def embed(self, state_space_labels, target_labels):
        """
        Embeds this term's action as a part of a larger state space.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels
            The state space labels for the full space.

        target_labels : tuple
            A tuple of the target labels.

        Returns
        -------
        RankOneOpTerm
        """
        evotype = "statevec" if isinstance(self._rep, replib.SVTermRep) else "stabilizer"
        pre_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                   for oprep in self._rep.pre_ops]
        post_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                    for oprep in self._rep.post_ops]
        return self.__class__(self._rep.__class__(self._rep.coeff, 1.0, 0.0, None, None,
                                                  None, None, pre_ops, post_ops))


class RankOnePrepTermWithMagnitude(RankOneTerm, _HasMagnitude):
    """
    A state preparation term with magnitude tracking.
    """
    def embed(self, state_space_labels, target_labels):
        """
        Embeds this term's action as a part of a larger state space.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels
            The state space labels for the full space.

        target_labels : tuple
            A tuple of the target labels.

        Returns
        -------
        RankOnePrepTermWithMagnitude
        """
        evotype = "statevec" if isinstance(self._rep, replib.SVTermRep) else "stabilizer"
        pre_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                   for oprep in self._rep.pre_ops]
        post_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                    for oprep in self._rep.post_ops]
        return self.__class__(self._rep.__class__(
            self._rep.coeff, self._rep.magnitude, self._rep.logmagnitude,
            self._rep.pre_state, self._rep.post_state, None, None, pre_ops, post_ops
        ))


class RankOneEffectTermWithMagnitude(RankOneTerm, _HasMagnitude):
    """
    A POVM effect term with magnitude tracking.
    """
    def embed(self, state_space_labels, target_labels):
        """
        Embeds this term's action as a part of a larger state space.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels
            The state space labels for the full space.

        target_labels : tuple
            A tuple of the target labels.

        Returns
        -------
        RankOneEffectTermWithMagnitude
        """
        evotype = "statevec" if isinstance(self._rep, replib.SVTermRep) else "stabilizer"
        pre_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                   for oprep in self._rep.pre_ops]
        post_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                    for oprep in self._rep.post_ops]
        return self.__class__(self._rep.__class__(
            self._rep.coeff, self._rep.magnitude, self._rep.logmagnitude,
            None, None, self._rep.pre_effect, self._rep.post_effect, pre_ops, post_ops
        ))


class RankOneOpTermWithMagnitude(RankOneTerm, _HasMagnitude):
    """
    An operation term with magnitude tracking.
    """
    def embed(self, state_space_labels, target_labels):
        """
        Embeds this term's action as a part of a larger state space.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels
            The state space labels for the full space.

        target_labels : tuple
            A tuple of the target labels.

        Returns
        -------
        RankOneOpTermWithMagnitude
        """
        evotype = "statevec" if isinstance(self._rep, replib.SVTermRep) else "stabilizer"
        pre_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                   for oprep in self._rep.pre_ops]
        post_ops = [_embed_oprep(state_space_labels, target_labels, oprep, evotype)
                    for oprep in self._rep.post_ops]
        return self.__class__(self._rep.__class__(self._rep.coeff, self._rep.magnitude, self._rep.logmagnitude,
                                                  None, None, None, None, pre_ops, post_ops))


class _HasNumericalCoefficient(object):
    """
    A base class for terms that have numerical coefficients.

    Attributes
    ----------
    coeff : complex or float
        The term's coefficient.
    """
    @classmethod
    def _coeff_rep(cls, coeff):
        return coeff

    @property
    def coeff(self):
        """
        The term's coefficient (float or complex).
        """
        return self._rep.coeff


class _HasPolynomialCoefficient(object):
    """
    A base class for terms that have polynomial coefficients..

    Attributes
    ----------
    coeff : Polynomial
        The term's coefficient.
    """

    @classmethod
    def _coeff_rep(cls, coeff):
        return coeff.to_rep()

    @property
    def coeff(self):
        """
        The term's coefficient (a :class:`Polynomial`).
        """
        return _Polynomial.from_rep(self._rep.coeff)

    #def _coeff_copy(self):
    #    return self.coeff.copy()

    def map_indices_inplace(self, mapfn):
        """
        Performs a bulk find & replace on the coefficient polynomial's variable indices.

        This function should only be called when this term's
        coefficient is a :class:`Polynomial`.

        Parameters
        ----------
        mapfn : function
            A function that takes as input an "old" variable-index-tuple
            (a key of this Polynomial) and returns the updated "new"
            variable-index-tuple.

        Returns
        -------
        None
        """
        assert(hasattr(self._rep.coeff, 'map_indices_inplace')), \
            "Coefficient (type %s) must implements `map_indices_inplace`" % str(type(self.coeff))
        #self.coeff.map_indices_inplace(mapfn)
        self._rep.coeff.map_indices_inplace(mapfn)
        raise NotImplementedError("Need to add compact_complex() update as mapvec version does now")

    def mapvec_indices_inplace(self, mapvec):
        """
        Performs a bulk find & replace on this polynomial's variable indices.

        This function is similar to :method:`map_indices` but uses a *vector*
        to describe *individual* index updates instead of a function for
        increased performance.

        This function should only be called when this term's
        coefficient is a :class:`Polynomial`.

        Parameters
        ----------
        mapvec : numpy.ndarray
            An array whose i-th element gives the updated "new" index for
            the i-th variable.  Note that this vector maps *individual*
            variable indices old->new, whereas `mapfn` in :method:`map_indices`
            maps between *tuples* of indices.

        Returns
        -------
        None
        """
        #self.coeff.mapvec_indices_inplace(mapvec)
        self._rep.mapvec_indices_inplace(mapvec)


class RankOnePolynomialPrepTerm(RankOnePrepTerm, _HasPolynomialCoefficient):
    """
    A state preparation term with polynomial coefficient.
    """
    def copy_with_magnitude(self, mag):
        """
        Copy and set the magnitude of the copy to `mag`

        Parameters
        ----------
        mag : float
            Magnitude to set.

        Returns
        -------
        RankOnePolynomialPrepTermWithMagnitude
        """
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        rep = self._rep.copy()
        rep.set_magnitude(mag)
        return RankOnePolynomialPrepTermWithMagnitude(rep)


class RankOnePolynomialEffectTerm(RankOneEffectTerm, _HasPolynomialCoefficient):
    """
    A POVM effect term with polynomial coefficient.
    """
    def copy_with_magnitude(self, mag):
        """
        Copy and set the magnitude of the copy to `mag`

        Parameters
        ----------
        mag : float
            Magnitude to set.

        Returns
        -------
        RankOnePolynomialEffectTermWithMagnitude
        """
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        rep = self._rep.copy()
        rep.set_magnitude(mag)
        return RankOnePolynomialEffectTermWithMagnitude(rep)


class RankOnePolynomialOpTerm(RankOneOpTerm, _HasPolynomialCoefficient):
    """
    An operation term with polynomial coefficient.
    """
    def copy_with_magnitude(self, mag):
        """
        Copy and set the magnitude of the copy to `mag`

        Parameters
        ----------
        mag : float
            Magnitude to set.

        Returns
        -------
        RankOnePolynomialOpTermWithMagnitude
        """
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        rep = self._rep.copy()
        rep.set_magnitude(mag)
        return RankOnePolynomialOpTermWithMagnitude(rep)


class RankOnePolynomialPrepTermWithMagnitude(RankOnePrepTermWithMagnitude, _HasPolynomialCoefficient):
    """
    A state preparation term with polynomial coefficient and magnitude tracking.
    """
    pass


class RankOnePolynomialEffectTermWithMagnitude(RankOneEffectTermWithMagnitude, _HasPolynomialCoefficient):
    """
    A POVM effect term with polynomial coefficient and magnitude tracking.
    """
    pass


class RankOnePolynomialOpTermWithMagnitude(RankOneOpTermWithMagnitude, _HasPolynomialCoefficient):
    """
    An operation term with polynomial coefficient and magnitude tracking.
    """
    pass


class RankOneDirectPrepTerm(RankOnePrepTerm, _HasNumericalCoefficient):
    """
    A state preparation term with numerical coefficient (and *no* magnitude tracking).
    """
    pass


class RankOneDirectEffectTerm(RankOneEffectTerm, _HasNumericalCoefficient):
    """
    A POVM effect term with numerical coefficient (and *no* magnitude tracking).
    """
    pass


class RankOneDirectOpTerm(RankOneOpTerm, _HasNumericalCoefficient):
    """
    An operation term with numerical coefficient (and *no* magnitude tracking).
    """
    pass
