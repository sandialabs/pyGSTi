""" Defines classes which represent terms in gate expansions """
from __future__ import division, print_function, absolute_import, unicode_literals
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
from .polynomial import Polynomial as _Polynomial
from . import replib

LARGE = 1000000000  # a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.


def compose_terms(terms, magnitude=None):
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
        A list of terms to compose.

    magnitude : float, optional
        The magnitude of the composed term (fine to leave as None
        if you don't care about keeping track of magnitudes).

    Returns
    -------
    RankOneTerm
    """
    #TODO: this should be a classmethod of RankOneTerm, as it builds a new RankOneTerm from scratch
    if len(terms) == 0:
        return RankOneTerm(1.0, None, None)
    else:
        #Note: will need to check instance/type and do np.product if coeffs are floats!
        first = terms[0]
        ret = RankOneTerm.__new__(RankOneTerm)
        ret.termtype = first.termtype
        ret._evotype = first._evotype
        ret.coeff = _Polynomial.product([t.coeff for t in terms])
        #ret.pre_ops = list(_itertools.chain(*[t.pre_ops for t in terms]))
        #ret.post_ops = list(_itertools.chain(*[t.post_ops for t in terms]))

        ret.pre_ops = []
        ret.post_ops = []
        for t in terms:
            ret.pre_ops.extend(t.pre_ops)
            ret.post_ops.extend(t.post_ops)
            
        if magnitude is not None:
            ret.set_magnitude(magnitude)
        else:  # just use default
            ret.magnitude = 1.0
            ret.logmagnitude = 0.0
        return ret
                           
                           
    #OLD
    #ret = terms[0].copy()
    #for t in terms[1:]:
    #    ret.compose(t)
    #
    #            self.coeff *= term.coeff
    #    self.pre_ops.extend(term.pre_ops)
    #    self.post_ops.extend(term.post_ops)
    #
    #return ret


def exp_terms(terms, order, postterm, cache=None, order_base=None):
    """
    Exponentiate a list of terms, collecting those terms of the orders given
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
        previous_order_terms = build_terms(order_to_build-1)
        a = 1.0 / order_to_build  # builds up 1/factorial prefactor
        premultiplied_terms = [ a * factor for factor in terms ]
        cache[order_to_build] = [ compose_terms((previous_order_term, a_factor))
                                  for previous_order_term in previous_order_terms
                                  for a_factor in premultiplied_terms ]
        return cache[order_to_build]
    
    if persistent_cache:
        return [t.copy() for t in build_terms(order)]  # copy the terms if we need to store them in cache
    else:
        return build_terms(order)


#OLD VERSION TODO REMOVE
def OLDexp_terms(terms, order, postterm, cache=None, order_base=None):
    #create terms for each order from terms and base action
    
    # expand exp(L) = I + L + 1/2! L^2 + ... (n-th term 1/n! L^n)
    if order == 0:
        return [postterm]
    one_over_factorial = 1 / _np.math.factorial(order)
    
    # expand 1/n! L^n into a list of rank-1 terms
    #termLists = [terms]*order
    final_terms = []
    
    #Alternate method
    def add_terms(term_list_index, composed_factors_so_far):
        if term_list_index == order:
            final_terms.append(composed_factors_so_far)
            return
        for factor in terms:  # termLists[term_list_index]:
            add_terms(term_list_index + 1, compose_terms((composed_factors_so_far, factor)))
    
    add_terms(0, one_over_factorial * postterm)    
    return final_terms


def exp_terms_above_mag(terms, order, postterm, cache=None, min_term_mag=None):
    """
    Exponentiate a list of terms, collecting those terms of the orders given
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

    TODO: docstring min_term_mag & return val

    Returns
    -------
    list
    """
    cache = {}  # DON'T allow the user to pass in a previous cache
                # since this may have used a different min_term_mag

    #create terms for each order from terms and base action

    # expand exp(L) = I + L + 1/2! L^2 + ... (n-th term 1/n! L^n)
    # expand 1/n! L^n into a list of rank-1 terms -> cache[n]
    cache[0] = [postterm]
    
    def build_terms(order_to_build):
        if order_to_build in cache:  # Note: 0th order is *always* in cache
            return cache[order_to_build]
        previous_order_terms = build_terms(order_to_build-1)
        a = 1.0 / order_to_build  # builds up 1/factorial prefactor
        premultiplied_terms = [ a * factor for factor in terms ]  # terms are expected to have their magnitudes set.
        for t in premultiplied_terms: t.set_magnitude(t.magnitude * a)  # because a * factor doesn't update magnitude by efficiency
        cache[order_to_build] = [ t for t in (compose_terms((previous_order_term, a_factor),
                                                            magnitude=previous_order_term.magnitude * a_factor.magnitude)
                                              for previous_order_term in previous_order_terms
                                              for a_factor in premultiplied_terms) if t.magnitude >= min_term_mag ]
        # **Assume** individual term magnitudes are <= 1.0 so that we
        # don't include any order terms that have magnitude < min_term_mag.
        return cache[order_to_build]
    return build_terms(order)


def embed_term(term, stateSpaceLabels, targetLabels):
    """
    Embed a term to it acts within a larger state space.

    Internally, this simply converts a term's gate operators to embedded gate
    operations.

    Parameters
    ----------
    term : RankOneTerm
        The term to embed

    stateSpaceLabels : a list of tuples
        This argument specifies the density matrix space upon which the
        constructed term will act.  Each tuple corresponds to a block of a
        density matrix in the standard basis (and therefore a component of
        the direct-sum density matrix space).

    targetLabels : list
        The labels contained in `stateSpaceLabels` which demarcate the
        portions of the state space acted on by `term`.

    Returns
    -------
    RankOneTerm
    """
    from . import operation as _op
    ret = RankOneTerm(term.coeff, None, None, term.termtype, term._evotype)
    ret.pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in term.pre_ops]
    ret.post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in term.post_ops]
    return ret


class RankOneTerm(object):
    """
    An operation, like a gate, that maps a density matrix to another density
    matrix but in a more restricted way.  While a RankOneTerm doesn't have to
    map pure states to pure states, its action can be written:

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
    """
    import_cache = None  # to avoid slow re-importing withing RankOneTerm.__init__

    # For example, a term for the action:
    #
    # rho -> 5.0 * CBA * rho * AD
    #
    # will have members:
    # coeff = 5.0
    # pre_ops = [A, B, C]
    # post_ops = [ A^dag, D^dag ]

    # TODO: change typ to evotype and maybe allow "auto"?  should only need/allow "statevec" and "stabilizer" types?
    def __init__(self, coeff, pre_op, post_op, termtype, evotype):
        """
        Initialize a new RankOneTerm.

        Parameters
        ----------
        coeff : object
            The coefficient of this term.

        pre_op : object
            Typically a LinearOperator- or SPAMVec-derived object giving the
            left-hand ("pre-rho") unitary action, pure state, or projection of
            the term.  Can be None to indicate no operation/state.

        post_op : object
            Typically a LinearOperator- or SPAMVec-derived object giving the *adjoint* of
            the right-hand ("post-rho") unitary action, pure state, or
            projection of the term. Can be None to indicate no operation/state.

        termtype : {"gate","prep","effect"}
            The type of term this is, essentially indicating what type of
            operation the initial pre and post ops are (gates, preps,
            or effects).

        evotype : {"svterm", "cterm"}
            Whether this term propagates (or *is*, or measures, depending on `termtype`)
            dense state-vectors or stabilizer states.
        """
        if self.__class__.import_cache is None:
            # slows function down significantly if don't put these in an if-block (surprisingly)
            from . import modelmember as _mm
            from . import operation as _op
            from . import spamvec as _spamvec
            self.__class__.import_cache = (_mm, _op, _spamvec)
        else:
            _mm, _op, _spamvec = self.__class__.import_cache

        self.coeff = coeff  # potentially a Polynomial
        if isinstance(self.coeff, _numbers.Number):
            #self.evaluated_coeff = coeff #REMOVE
            self.magnitude = abs(coeff)
            self.logmagnitude = _np.log10(self.magnitude) if self.magnitude > 0 else -LARGE
        else:
            #self.evaluated_coeff = None #REMOVE
            self.magnitude = 1.0
            self.logmagnitude = 0.0

        self.pre_ops = []  # list of ops to perform - in order of operation to a ket
        self.post_ops = []  # list of ops to perform - in order of operation to a bra
        self.termtype = termtype
        self._evotype = evotype

        #NOTE: self.post_ops holds the *adjoints* of the actual post-rho-operators, so that
        #evolving a bra with the post_ops can be accomplished by flipping the bra -> ket and
        #applying the stored adjoints in the order stored in self.post_ops (similar to
        #acting with pre_ops in-order on a ket

        if pre_op is not None:
            if not isinstance(pre_op, _mm.ModelMember):
                if termtype == "gate":
                    if evotype == "svterm":
                        pre_op = _op.StaticDenseOp(pre_op, "statevec")
                    elif evotype == "cterm":
                        pre_op = _op.CliffordOp(pre_op)
                    else: assert(False), "Invalid `evotype` argument: %s" % evotype
                else:
                    if evotype == "svterm":
                        pre_op = _spamvec.StaticSPAMVec(pre_op, "statevec", termtype)
                    else: assert(False), "No default term vector for evotype=%s" % evotype
            self.pre_ops.append(pre_op)
        if post_op is not None:
            if not isinstance(post_op, _mm.ModelMember):
                if termtype == "gate":
                    if evotype == "svterm":
                        post_op = _op.StaticDenseOp(post_op, "statevec")
                    elif evotype == "cterm":
                        post_op = _op.CliffordOp(post_op)
                    else: assert(False), "Invalid `evotype` argument: %s" % evotype
                else:
                    if evotype == "svterm":
                        post_op = _spamvec.StaticSPAMVec(post_op, "statevec", termtype)
                    else: assert(False), "No default term vector for evotype=%s" % evotype
            self.post_ops.append(post_op)

    def __mul__(self, x):
        """ Multiply by scalar """
        ret = self.copy()
        ret.coeff *= x
        return ret

    def __rmul__(self, x):
        return self.__mul__(x)

    #UNUSED TODO REMOVE
    #def scalar_mult(self, x):
    #    """
    #    Multiplies this term by a scalar `x`.
    #
    #    This simply returns a new `RankOneTerm` that
    #    is the same as this term except its coefficient
    #    has been multiplied by `x`.
    #
    #    Parameters
    #    ----------
    #    x : float or complex
    #        The value to multiply by.
    #
    #    Returns
    #    -------
    #    RankOneTerm
    #    """
    #    ret = self.copy()
    #    if isinstance(self.coeff, _numbers.Number):
    #        ret.coeff *= x
    #    else:
    #        ret.coeff = ret.coeff.scalar_mult(x)
    #    return ret

    def set_magnitude(self, mag):
        """
        Sets the "magnitude" of this term used in path-pruning.  Sets
        both .magnitude and .logmagnitude attributes of this object.

        Parameters
        ----------
        mag : float
            The magnitude to set.

        Returns
        -------
        None
        """
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        self.magnitude = mag
        self.logmagnitude = _math.log10(mag) if mag > 0 else -LARGE

#TODO REMOVE
#     def set_evaluated_coeff(self, evaluated_coeff):
#         """
#         TODO: docstring
#         """
#         self.evaluated_coeff = evaluated_coeff

    def compose(self, term):
        """
        Compose with `term`, which since it occurs to the *right*
        of this term, is applied *after* this term.

        Parameters
        ----------
        term : RankOneTerm
            The term to compose with.

        Returns
        -------
        None
        """
        self.coeff *= term.coeff
        self.pre_ops.extend(term.pre_ops)
        self.post_ops.extend(term.post_ops)

    #def collapse(self):
    #    """
    #    Returns a copy of this term with all pre & post ops by reduced
    #    ("collapsed") by matrix composition, so that resulting
    #    term has only a single pre/post op. Ops must be compatible with numpy
    #    dot products.
    #
    #    Returns
    #    -------
    #    RankOneTerm
    #    """
    #    if self._evotype != "svterm":
    #        raise NotImplementedError("Term collapse for types other than 'svterm' are not implemented yet!")
    #
    #    if len(self.pre_ops) >= 1:
    #        pre = self.pre_ops[0]  # .to_matrix() FUTURE??
    #        for B in self.pre_ops[1:]:
    #            pre = _np.dot(B, pre)  # FUTURE - something more general (compose function?)
    #    else: pre = None
    #
    #    if len(self.post_ops) >= 1:
    #        post = self.post_ops[0]
    #        for B in self.post_ops[1:]:
    #            post = _np.dot(B, post)
    #    else: post = None
    #
    #    return RankOneTerm(self.coeff, pre, post)
    #
    ##FUTURE: maybe have separate GateRankOneTerm and SPAMRankOneTerm which
    ## derive from RankOneTerm, and only one collapse() function (also
    ## this would avoid try/except logic elsewhere).
    #def collapse_vec(self):
    #    """
    #    Returns a copy of this term with all pre & post ops by reduced
    #    ("collapsed") by action of LinearOperator ops on an initial SPAMVec.  This results
    #    in a term with only a single pre/post op which are SPAMVecs.
    #
    #    Returns
    #    -------
    #    RankOneTerm
    #    """
    #
    #    if self._evotype != "svterm":
    #        raise NotImplementedError("Term collapse_vec for types other than 'svterm' are not implemented yet!")
    #
    #    if len(self.pre_ops) >= 1:
    #        pre = self.pre_ops[0].todense()  # first op is a SPAMVec
    #        for B in self.pre_ops[1:]:  # and the rest are Gates
    #            pre = B.acton(pre)
    #    else: pre = None
    #
    #    if len(self.post_ops) >= 1:
    #        post = self.post_ops[0].todense()  # first op is a SPAMVec
    #        for B in self.post_ops[1:]:  # and the rest are Gates
    #            post = B.acton(post)
    #    else: post = None
    #
    #    return RankOneTerm(self.coeff, pre, post)

    def copy(self):
        """
        Copy this term.

        Returns
        -------
        RankOneTerm
        """
        coeff = self.coeff if isinstance(self.coeff, _numbers.Number) \
            else self.coeff.copy()
        copy_of_me = RankOneTerm(coeff, None, None, self.termtype, self._evotype)
        copy_of_me.pre_ops = self.pre_ops[:]
        copy_of_me.post_ops = self.post_ops[:]
        copy_of_me.magnitude = self.magnitude
        copy_of_me.logmagnitude = self.logmagnitude
        return copy_of_me

    def map_indices_inplace(self, mapfn):
        """
        Performs a bulk find & replace on the coefficient polynomial's variable
        indices.  This function should only be called when this term's
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
        assert(hasattr(self.coeff, 'map_indices_inplace')), \
            "Coefficient (type %s) must implements `map_indices_inplace`" % str(type(self.coeff))
        self.coeff.map_indices_inplace(mapfn)

    def mapvec_indices_inplace(self, mapvec):
        """
        TODO: docstring: similar to map_indices_inplace, but uses vector (see polynomial.py)
        Performs a bulk find & replace on the coefficient polynomial's variable
        indices.  This function should only be called when this term's
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
        self.coeff.mapvec_indices_inplace(mapvec)


    def torep(self): 
        """
        Construct a representation of this term.

        "Representations" are lightweight versions of objects used to improve
        the efficiency of intensely computational tasks, used primarily
        internally within pyGSTi.

        Parameters
        ----------
        max_num_vars : int
            The maximum number of variables for the coefficient polynomial's
            represenatation.

        typ : { "prep", "effect", "gate" }
            What type of representation is needed (these correspond to
            different types of representation objects).  Given the type of
            operations stored within a term, only one of "gate" and
            "prep"/"effect" is appropriate.

        Returns
        -------
        SVTermRep or SBTermRep
        """
        if isinstance(self.coeff, _numbers.Number):
            coeffrep = self.coeff
            RepTermType = replib.SVTermDirectRep if (self._evotype == "svterm") \
                else replib.SBTermDirectRep
        else:
            coeffrep = self.coeff.torep()
            RepTermType = replib.SVTermRep if (self._evotype == "svterm") \
                else replib.SBTermRep

        if self.termtype == "prep":  # first el of pre_ops & post_ops is a state vec
            return RepTermType(coeffrep, self.magnitude, self.logmagnitude,
                               self.pre_ops[0]._rep,
                               self.post_ops[0]._rep, None, None,
                               [op._rep for op in self.pre_ops[1:]],
                               [op._rep for op in self.post_ops[1:]])
        elif self.termtype == "effect":  # first el of pre_ops & post_ops is an effect vec
            return RepTermType(coeffrep, self.magnitude, self.logmagnitude,
                               None, None, self.pre_ops[0]._rep,
                               self.post_ops[0]._rep,
                               [op._rep for op in self.pre_ops[1:]],
                               [op._rep for op in self.post_ops[1:]])
        else:
            assert(self.termtype == "gate"), "Invalid termtype in RankOneTerm: %s" % self.termtype
            return RepTermType(coeffrep, self.magnitude, self.logmagnitude,
                               None, None, None, None,
                               [op._rep for op in self.pre_ops],
                               [op._rep for op in self.post_ops])

    def evaluate_coeff(self, variable_values):
        """
        Evaluate this term's polynomial coefficient for a given set of variable values.

        Parameters
        ----------
        variable_values : array-like
            An object that can be indexed so that `variable_values[i]` gives the
            numerical value for i-th variable (x_i) in this term's coefficient.

        Returns
        -------
        RankOneTerm
            A shallow copy of this object with floating-point coefficient
        """
        coeff = self.coeff.evaluate(variable_values)
        copy_of_me = RankOneTerm(coeff, None, None, self.termtype, self._evotype)
        copy_of_me.pre_ops = self.pre_ops[:]
        copy_of_me.post_ops = self.post_ops[:]
        return copy_of_me
