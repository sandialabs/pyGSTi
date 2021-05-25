# encoding: utf-8
# cython: profile=True
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import sys
import numpy as _np
import copy as _copy

import itertools as _itertools
from ...models.statespace import StateSpace as _StateSpace
from scipy.sparse.linalg import LinearOperator


cdef class TermRep(_basereps_cython.TermRep):

    @classmethod
    def composed(cls, terms_to_compose, double magnitude):
        cdef double logmag = log10(magnitude) if magnitude > 0 else -LARGE
        cdef TermRep first = terms_to_compose[0]
        cdef PolynomialRep coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return TermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                         first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __cinit__(self, PolynomialRep coeff, double mag, double logmag,
                  StateRep pre_state, StateRep post_state,
                  EffectRep pre_effect, EffectRep post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.compact_coeff = coeff.compact_complex()
        self.pre_ops = pre_ops
        self.post_ops = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[OpCRep*] c_pre_ops = vector[OpCRep_ptr](npre)
        cdef vector[OpCRep*] c_post_ops = vector[OpCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<OpRep?>pre_ops[i]).c_rep
        for i in range(npost):
            c_post_ops[i] = (<OpRep?>post_ops[i]).c_rep

        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.pre_state = pre_state
            self.post_state = post_state
            self.pre_effect = self.post_effect = None
            self.c_term = new TermCRep(coeff.c_polynomial, mag, logmag, pre_state.c_state, post_state.c_state,
                                         c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.pre_effect = pre_effect
            self.post_effect = post_effect
            self.pre_state = self.post_state = None
            self.c_term = new TermCRep(coeff.c_polynomial, mag, logmag, pre_effect.c_effect, post_effect.c_effect,
                                         c_pre_ops, c_post_ops);
        else:
            self.pre_state = self.post_state = None
            self.pre_effect = self.post_effect = None
            self.c_term = new TermCRep(coeff.c_polynomial, mag, logmag, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term

    def __reduce__(self):
        return (TermRep, (self.coeff, self.magnitude, self.logmagnitude,
                self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                self.pre_ops, self.post_ops))

    def set_magnitude(self, double mag):
        self.c_term._magnitude = mag
        self.c_term._logmagnitude = log10(mag) if mag > 0 else -LARGE

    def set_magnitude_only(self, double mag):  # TODO: better name?
        self.c_term._magnitude = mag

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)
        self.compact_coeff = self.coeff.compact_complex()  #b/c indices have been updated!

    @property
    def magnitude(self):
        return self.c_term._magnitude

    @property
    def logmagnitude(self):
        return self.c_term._logmagnitude

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return TermRep(coeff, self.magnitude * x, self.logmagnitude + log10(x),
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    def copy(self):
        return TermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    #Not needed - and this implementation is quite right as it will need to change
    # the ordering of the pre/post ops also.
    #def conjugate(self):
    #    return TermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
    #                     self.post_state, self.pre_state, self.post_effect, self.pre_effect,
    #                     self.post_ops, self.pre_ops)


#Note: to use direct term reps (numerical coeffs) we'll need to update
# what the members are called and add methods as was done for TermRep.
cdef class TermDirectRep(_basereps_cython.TermRep):

    def __cinit__(self, double complex coeff, double mag, double logmag,
                  StateRep pre_state, StateRep post_state,
                  EffectRep pre_effect, EffectRep post_effect, pre_ops, post_ops):
        self.list_of_preops_ref = pre_ops
        self.list_of_postops_ref = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[OpCRep*] c_pre_ops = vector[OpCRep_ptr](npre)
        cdef vector[OpCRep*] c_post_ops = vector[OpCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<OpRep?>pre_ops[i]).c_rep
        for i in range(npost):
            c_post_ops[i] = (<OpRep?>post_ops[i]).c_rep

        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.state_ref1 = pre_state
            self.state_ref2 = post_state
            self.c_term = new TermDirectCRep(coeff, mag, logmag, pre_state.c_state, post_state.c_state,
                                               c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.effect_ref1 = pre_effect
            self.effect_ref2 = post_effect
            self.c_term = new TermDirectCRep(coeff, mag, logmag, pre_effect.c_effect, post_effect.c_effect,
                                               c_pre_ops, c_post_ops);
        else:
            self.c_term = new TermDirectCRep(coeff, mag, logmag, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term

    def set_coeff(self, coeff):
        self.c_term._coeff = coeff

    def set_magnitude(self, double mag, double logmag):
        self.c_term._magnitude = mag
        self.c_term._logmagnitude = logmag

    @property
    def magnitude(self):
        return self.c_term._magnitude

    @property
    def logmagnitude(self):
        return self.c_term._logmagnitude
