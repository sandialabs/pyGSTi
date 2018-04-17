""" Defines classes which represent terms in gate expansions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import itertools as _itertools
import numbers as _numbers
from .polynomial import Polynomial as _Polynomial

def compose_terms(terms):
    if len(terms) == 0: return RankOneTerm(1.0,None,None)
    ret = terms[0].copy()
    for t in terms[1:]:
        ret.compose(t)
    return ret

def exp_terms(terms, orders, postterm=None):
    """ postterm is "post" in the matrix-order sense,
        which means it's applied *first* """ 
    #create terms for each order from terms and base action
    final_terms = {}
    if postterm is not None:
        Uterm_tup = (postterm,)
    else: Uterm_tup = ()

    for order in orders: # expand exp(L) = I + L + 1/2! L^2 + ... (n-th term 1/n! L^n)
        if order == 0:
            final_terms[order] = [ Uterm_tup[0] ]; continue
            
        # expand 1/n! L^n into a list of rank-1 terms
        termLists = [terms]*order
        final_terms[order] = []
        for factors in _itertools.product(*termLists):
            final_terms[order].append( 1/_np.math.factorial(order) * compose_terms(Uterm_tup + factors) ) # apply Uterm first
            
    return final_terms

def embed_term(term, stateSpaceLabels, targetLabels, basisdim=None):
    """ TODO docstring - converts a term's gate operators to embedded gate ops"""
    from . import gate as _gate
    mode="unitary" # not an argument currently, since there's no reason this should be "superop"
    ret = RankOneTerm(term.coeff, None, None)
    ret.pre_ops = [ _gate.EmbeddedGateMap(stateSpaceLabels, targetLabels, op, basisdim, mode)
                    for op in term.pre_ops ]
    ret.post_ops = [ _gate.EmbeddedGateMap(stateSpaceLabels, targetLabels, op, basisdim, mode)
                    for op in term.post_ops ]
    return ret


class RankOneTerm(object):
    def __init__(self, coeff, pre_op, post_op):
        """ TODO docstring
        NOTE: the post_ops hold the *adjoints* of the actual post-rho-operators, so that
        evolving a bra with the post_ops can be accomplished by flipping the bra -> ket and
        applying the stored adjoints in the order stored in self.post_ops (similar to 
        acting with pre_ops in-order on a ket
        """
        from . import gatesetmember as _gsm
        from . import gate as _gate
        from . import spamvec as _spamvec
        self.coeff = coeff # potentially a Polynomial
        self.pre_ops = [] # list of ops to perform - in order of operation to a ket
        self.post_ops = [] # list of ops to perform - in order of operation to a bra
        if pre_op is not None:
            if not isinstance(pre_op,_gsm.GateSetMember):
                try:
                    pre_op = _gate.StaticGate(pre_op) #default to static gates
                except ValueError: # raised when size/shape is wrong
                    pre_op = _spamvec.StaticSPAMVec(pre_op) # ... or spam vecs
            self.pre_ops.append(pre_op)
        if post_op is not None:
            if not isinstance(post_op,_gsm.GateSetMember):
                try:
                    post_op = _gate.StaticGate(post_op) #default to static gates
                except ValueError: # raised when size/shape is wrong
                    post_op = _spamvec.StaticSPAMVec(post_op) # ... or spam vecs
            self.post_ops.append(post_op)

    def __mul__(self,x):
        """ Multiply by scalar """
        ret = self.copy()
        self.coeff *= x
        return ret
    
    def __rmul__(self, x):
        return self.__mul__(x)
        
    def compose(self, term):
        """ 
        Compose with `term`, which since it occurs to the *right*
        of this term, is applied *after* this term.
        """
        self.coeff *= term.coeff
        self.pre_ops.extend(term.pre_ops)
        self.post_ops.extend(term.post_ops)

    def collapse(self):
        """
        Returns a copy of this term with all pre & post ops by reduced
        ("collapsed") by matrix composition, so that resulting
        term has only a single pre/post op. Ops must be compatible with numpy
        dot products.

        Returns
        -------
        RankOneTerm
        """
        
        if len(self.pre_ops) >= 1:
            pre = self.pre_ops[0] #.to_matrix() FUTURE??
            for B in self.pre_ops[1:]:
                pre = _np.dot(B,pre) # FUTURE - something more general (compose function?)
        else: pre = None
            
        if len(self.post_ops) >= 1:
            post = self.post_ops[0]
            for B in self.post_ops[1:]:
                post = _np.dot(B,post)
        else: post = None
            
        return RankOneTerm(self.coeff, pre, post)

    #FUTURE: maybe have separate GateRankOneTerm and SPAMRankOneTerm which
    # derive from RankOneTerm, and only one collapse() function (also
    # this would avoid try/except logic elsewhere).
    def collapse_vec(self):
        """
        Returns a copy of this term with all pre & post ops by reduced
        ("collapsed") by action of Gate ops on an initial SPAMVec.  This results
        in a term with only a single pre/post op which are SPAMVecs.

        Returns
        -------
        RankOneTerm
        """
        
        if len(self.pre_ops) >= 1:
            pre = self.pre_ops[0].toarray() # first op is a SPAMVec
            for B in self.pre_ops[1:]: # and the rest are Gates 
                pre = B.acton(pre)
        else: pre = None
            
        if len(self.post_ops) >= 1:
            post = self.post_ops[0].toarray() # first op is a SPAMVec
            for B in self.post_ops[1:]: # and the rest are Gates 
                post = B.acton(post)
        else: post = None
            
        return RankOneTerm(self.coeff, pre, post)

    
    def copy(self):
        coeff = self.coeff if isinstance(self.coeff, _numbers.Number) \
                else self.coeff.copy()
        copy_of_me = RankOneTerm(coeff, None, None)
        copy_of_me.pre_ops = self.pre_ops[:]
        copy_of_me.post_ops = self.post_ops[:]
        return copy_of_me

    def map_indices(self, mapfn):
        """ TODO: docstring - mapfn should map old->new variable-index-tuples """
        assert(hasattr(self.coeff, 'map_indices')), \
            "Coefficient (type %s) must implements `map_indices`" % str(type(self.coeff))
        self.coeff.map_indices(mapfn)
        
           


    
