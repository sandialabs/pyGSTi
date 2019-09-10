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
import functools as _functools
from operator import add as _add


from .polynomial import Polynomial as _Polynomial
from . import replib
from . import modelmember as _mm
from . import operation as _op
from . import spamvec as _spamvec

LARGE = 1000000000  # a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

def compose_terms_with_mag(terms, magnitude):
    """ TODO: docstring """
    assert(len(terms) > 0)
    return terms[0].compose(terms[1:], magnitude)

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
        A list of terms to compose.

    magnitude : float, optional
        The magnitude of the composed term (fine to leave as None
        if you don't care about keeping track of magnitudes).

    Returns
    -------
    RankOneTerm
    """
    assert(len(terms) > 0)
    return terms[0].compose(terms)
    
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
        #REMOVE for t in premultiplied_terms: t.set_magnitude(t.magnitude * a)  # because a * factor doesn't update magnitude by efficiency
        cache[order_to_build] = [ t for t in (compose_terms_with_mag((previous_order_term, a_factor),
                                                                     previous_order_term.magnitude * a_factor.magnitude)
                                              for previous_order_term in previous_order_terms
                                              for a_factor in premultiplied_terms) if t.magnitude >= min_term_mag ]
        # **Assume** individual term magnitudes are <= 1.0 so that we
        # don't include any order terms that have magnitude < min_term_mag.
        return cache[order_to_build]
    return build_terms(order)


#def embed_term(term, stateSpaceLabels, targetLabels):
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
#    stateSpaceLabels : a list of tuples
#        This argument specifies the density matrix space upon which the
#        constructed term will act.  Each tuple corresponds to a block of a
#        density matrix in the standard basis (and therefore a component of
#        the direct-sum density matrix space).
#
#    targetLabels : list
#        The labels contained in `stateSpaceLabels` which demarcate the
#        portions of the state space acted on by `term`.
#
#    Returns
#    -------
#    RankOneTerm
#    """
#    from . import operation as _op
#    ret = RankOneTerm(term.coeff, None, None, term.termtype, term._evotype)
#    ret.pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
#                   for op in term.pre_ops]
#    ret.post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
#                    for op in term.post_ops]
#    return ret


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

    def __rmul__(self, x):
        return self.__mul__(x)

    def __mul__(self, x):
        """ Multiply by scalar """
        ret = self.copy()
        ret.coeff = ret.coeff * x
        return ret


class RankOnePrepTerm(RankOneTerm):
    
    @classmethod
    def simple_init(cls, coeff, pre_state, post_state, evotype):
        if evotype not in ('svterm', 'cterm'):
            raise ValueError("Invalid evotype: %s" % evotype)

        if not isinstance(pre_state, _mm.ModelMember):
            if evotype == "svterm":
                pre_state = _spamvec.StaticSPAMVec(pre_state, "statevec", "prep")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)
        if not isinstance(post_state, _mm.ModelMember):
            if evotype == "svterm":
                pre_state = _spamvec.StaticSPAMVec(post_state, "statevec", "prep")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)
        
        return cls(coeff, pre_state, post_state, [], [], evotype)
    
    def __init__(self, coeff, pre_state, post_state,
                 pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.coeff = coeff
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms):
        """
        TODO: docstring
        """
        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        coeff = self._coeff_product([t.coeff for t in all_terms])   
        return self.__class__(coeff,
                               self.pre_state, self.post_state, pre_ops, post_ops, self.evotype)

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, self.pre_state, self.post_state, pre_ops, post_ops, self.evotype)

    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]
        return reptype(self._coeff_rep(), 1.0, 0.0,
                                self.pre_state._rep, self.post_state._rep, None, None,
                                pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.pre_state, self.post_state, 
                               self.pre_ops[:], self.post_ops[:], self.evotype)



        
class RankOneEffectTerm(RankOneTerm):
    @classmethod
    def simple_init(cls, coeff, pre_effect, post_effect, evotype):    
        if evotype not in ('svterm', 'cterm'):
            raise ValueError("Invalid evotype: %s" % evotype)

        if not isinstance(pre_effect, _mm.ModelMember):
            if evotype == "svterm":
                pre_effect = _spamvec.StaticSPAMVec(pre_effect, "statevec", "effect")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)
        if not isinstance(post_effect, _mm.ModelMember):
            if evotype == "svterm":
                pre_effect = _spamvec.StaticSPAMVec(post_effect, "statevec", "effect")
            else:
                raise ValueError("No default term vector for evotype=%s" % evotype)
            
        return cls(coeff, pre_effect, post_effect, [], [], evotype)

            
    def __init__(self, coeff, pre_effect, post_effect,
                 pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.coeff = coeff
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms):
        """
        TODO: docstring
        """
        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        coeff = self._coeff_product([t.coeff for t in all_terms])   
        return self.__class__(coeff,
                                 self.pre_effect, self.post_effect, pre_ops, post_ops, self.evotype)

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, self.pre_effect, self.post_effect, pre_ops, post_ops, self.evotype)

    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]        
        return reptype(self._coeff_rep(), 1.0, 0.0,
                            None, None, self.pre_effect._rep, self.post_effect._rep,
                            pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.pre_effect, self.post_effect, 
                               self.pre_ops[:], self.post_ops[:], self.evotype)


    
class RankOneOpTerm(RankOneTerm):
    @classmethod
    def simple_init(cls, coeff, pre_op, post_op, evotype):
        if evotype not in ('svterm', 'cterm'):
            raise ValueError("Invalid evotype: %s" % evotype)

        pre_ops = []
        post_ops = []

        if pre_op is not None:
            if not isinstance(pre_op, _mm.ModelMember):
                if evotype == "svterm":
                    pre_op = _op.StaticDenseOp(pre_op, "statevec")
                elif evotype == "cterm":
                    pre_op = _op.CliffordOp(pre_op)
                else:
                    raise ValueError("Invalid `evotype` argument: %s" % evotype)
            pre_ops.append(pre_op)
                
        if post_op is not None:
            if not isinstance(post_op, _mm.ModelMember):
                if evotype == "svterm":
                    post_op = _op.StaticDenseOp(post_op, "statevec")
                elif evotype == "cterm":
                    post_op = _op.CliffordOp(post_op)
                else:
                    raise ValueError("Invalid `evotype` argument: %s" % evotype)
            post_ops.append(post_op)
            
        return cls(coeff, pre_ops, post_ops, evotype)

    def __init__(self, coeff, pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.coeff = coeff
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms):
        """
        TODO: docstring
        """
        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        coeff = self._coeff_product([t.coeff for t in all_terms])   
        return self.__class__(coeff,
                             pre_ops, post_ops, self.evotype)

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, pre_ops, post_ops, self.evotype)

    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]
        return reptype(self._coeff_rep(), 1.0, 0.0,
                            None, None, None, None,
                            pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.pre_ops[:], self.post_ops[:], self.evotype)



class RankOnePrepTermWithMagnitude(RankOneTerm):
    
    def __init__(self, coeff, magnitude, pre_state, post_state,
                 pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.magnitude = magnitude
        self.logmagnitude = _math.log10(magnitude) if magnitude > 0 else -LARGE
        self.coeff = coeff
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms, magnitude):
        """
        TODO: docstring
        """
        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops +=  t.pre_ops
            post_ops += t.post_ops
        coeff = self._coeff_product([t.coeff for t in all_terms])   
        return self.__class__(coeff, magnitude,
                              self.pre_state, self.post_state, pre_ops, post_ops, self.evotype)

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, self.magnitude, self.pre_state, self.post_state, pre_ops, post_ops, self.evotype)

    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]
        return reptype(self._coeff_rep(), self.magnitude, self.logmagnitude,
                                self.pre_state._rep, self.post_state._rep, None, None,
                                pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.magnitude, self.pre_state, self.post_state, 
                               self.pre_ops[:], self.post_ops[:], self.evotype)


        
class RankOneEffectTermWithMagnitude(RankOneTerm):
    def __init__(self, coeff, magnitude, pre_effect, post_effect,
                 pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.magnitude = magnitude
        self.logmagnitude = _math.log10(magnitude) if magnitude > 0 else -LARGE
        self.coeff = coeff
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms, magnitude):
        """
        TODO: docstring
        """
        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        coeff = self._coeff_product([t.coeff for t in all_terms])   
        return self.__class__(coeff, magnitude,
                                 self.pre_effect, self.post_effect, pre_ops, post_ops, self.evotype)

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, self.magnitude, self.pre_effect, self.post_effect, pre_ops, post_ops, self.evotype)


    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]
        return reptype(self._coeff_rep(), self.magnitude, self.logmagnitude,
                            None, None, self.pre_effect._rep, self.post_effect._rep,
                            pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.magnitude, self.pre_effect, self.post_effect, 
                               self.pre_ops[:], self.post_ops[:], self.evotype)


    
class RankOneOpTermWithMagnitude(RankOneTerm):
    def __init__(self, coeff, magnitude, pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.magnitude = magnitude
        self.logmagnitude = _math.log10(magnitude) if magnitude > 0 else -LARGE
        self.coeff = coeff
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms, magnitude):
        """
        TODO: docstring
        """
        #SLOWER
        #pre_ops = _functools.reduce(_add, [t.pre_ops for t in all_terms])
        #post_ops = _functools.reduce(_add, [t.post_ops for t in all_terms])

        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        coeff = self._coeff_product([t.coeff for t in all_terms])
        return self.__class__(coeff, magnitude,
                              pre_ops, post_ops, self.evotype)

        # --- INLINE of __init__ speeds things up a little; doesn't seem so worth it ---
        #ret = self.__class__.__new__(self.__class__)
        #ret.evotype = self.evotype
        #ret.magnitude = magnitude
        #ret.logmagnitude = _math.log10(magnitude) if magnitude > 0 else -LARGE
        #ret.coeff = self._coeff_product([t.coeff for t in all_terms])
        #ret.pre_ops = []
        #ret.post_ops = []
        #for t in all_terms:
        #    ret.pre_ops += t.pre_ops
        #    ret.post_ops += t.post_ops
        #return ret

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, self.magnitude, pre_ops, post_ops, self.evotype)


    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]
        return reptype(self._coeff_rep(), self.magnitude, self.logmagnitude,
                            None, None, None, None,
                            pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.magnitude,
                               self.pre_ops[:], self.post_ops[:], self.evotype)


class OLDRankOneOpTermWithMagnitude(RankOneTerm):
    def __init__(self, coeff, magnitude, pre_ops, post_ops, evotype):
        self.evotype = evotype
        self.magnitude = magnitude
        self.logmagnitude = _math.log10(magnitude) if magnitude > 0 else -LARGE
        self.coeff = coeff
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def compose(self, all_terms, magnitude):
        """
        TODO: docstring
        """
        #pre_ops = _functools.reduce(_add, [t.pre_ops for t in all_terms])
        #post_ops = _functools.reduce(_add, [t.post_ops for t in all_terms])

        pre_ops = []
        post_ops = []
        for t in all_terms:
            pre_ops += t.pre_ops
            post_ops += t.post_ops
            
        coeff = _Polynomial.product([t.coeff for t in all_terms])
        #coeff = self._coeff_product([t.coeff for t in all_terms])
            
        return self.__class__(coeff, magnitude,
                             pre_ops, post_ops, self.evotype)

    def embed(self, stateSpaceLabels, targetLabels):
        pre_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                   for op in self.pre_ops]
        post_ops = [_op.EmbeddedOp(stateSpaceLabels, targetLabels, op)
                    for op in self.post_ops]
        return self.__class__(self.coeff, self.magnitude, pre_ops, post_ops, self.evotype)


    def torep(self):
        reptype = replib.SVTermRep if (self.evotype == "svterm") \
            else replib.SBTermRep
        pre_op_reps = [op._rep for op in self.pre_ops]
        post_op_reps = [op._rep for op in self.post_ops]
        return reptype(self._coeff_rep(), self.magnitude, self.logmagnitude,
                            None, None, None, None,
                            pre_op_reps, post_op_reps)

    def copy(self):
        return self.__class__(self._coeff_copy(), self.magnitude,
                               self.pre_ops[:], self.post_ops[:], self.evotype)



class HasNumericalCoefficient(object):
    _coeff_product = _np.product
    #def _coeff_product(self, coeffs):
    #    return _np.product(coeffs)

    def _coeff_rep(self):
        return self.coeff

    def _coeff_copy(self):
        return self.coeff
        
class HasPolyCoefficient(object):
    _coeff_product = _Polynomial.product
    #def _coeff_product(self, coeffs):
    #    return _Polynomial.product(coeffs)
    
    def _coeff_rep(self):
        return self.coeff.torep()

    def _coeff_copy(self):
        return self.coeff.copy()

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
        
                 
class RankOnePolyPrepTerm(RankOnePrepTerm, HasPolyCoefficient):
    def copy_with_magnitude(self, mag):
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        return RankOnePolyPrepTermWithMagnitude(self.coeff, mag, self.pre_state, self.post_state, self.pre_ops, self.post_ops, self.evotype)

class RankOnePolyEffectTerm(RankOneEffectTerm, HasPolyCoefficient):
    def copy_with_magnitude(self, mag):
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        return RankOnePolyEffectTermWithMagnitude(self.coeff, mag, self.pre_effect, self.post_effect, self.pre_ops, self.post_ops, self.evotype)

class RankOnePolyOpTerm(RankOneOpTerm, HasPolyCoefficient):
    def copy_with_magnitude(self, mag):
        assert(mag <= 1.0), "Individual term magnitudes should be <= 1.0 so that '*_above_mag' routines work!"
        return RankOnePolyOpTermWithMagnitude(self.coeff, mag, self.pre_ops, self.post_ops, self.evotype)

class RankOnePolyPrepTermWithMagnitude(RankOnePrepTermWithMagnitude, HasPolyCoefficient): pass
class RankOnePolyEffectTermWithMagnitude(RankOneEffectTermWithMagnitude, HasPolyCoefficient): pass
class RankOnePolyOpTermWithMagnitude(RankOneOpTermWithMagnitude, HasPolyCoefficient): pass
    
class RankOneDirectPrepTerm(RankOnePrepTerm, HasNumericalCoefficient): pass
class RankOneDirectEffectTerm(RankOneEffectTerm, HasNumericalCoefficient): pass
class RankOneDirectOpTerm(RankOneOpTerm, HasNumericalCoefficient): pass
