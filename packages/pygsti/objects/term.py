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


class RankOneTerm(object):
    def __init__(self, coeff, pre_op, post_op):
        """ TODO docstring
        NOTE: the post_ops hold the *adjoints* of the actual post-rho-operators, so that
        evolving a bra with the post_ops can be accomplished by flipping the bra -> ket and
        applying the stored adjoints in the order stored in self.post_ops (similar to 
        acting with pre_ops in-order on a ket
        """
        from . import gate as _gate
        self.coeff = coeff # potentially a Polynomial
        self.pre_ops = [] # list of ops to perform - in order of operation to a ket
        self.post_ops = [] # list of ops to perform - in order of operation to a bra
        if pre_op is not None:
            if not isinstance(pre_op,_gate.Gate):
                pre_op = _gate.StaticGate(pre_op)                
            self.pre_ops.append(pre_op)
        if post_op is not None:
            if not isinstance(post_op,_gate.Gate):
                post_op = _gate.StaticGate(post_op)
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
        
        


    


    
