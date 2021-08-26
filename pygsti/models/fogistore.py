"""
Defines the FirstOrderGaugeInvariantStore class and supporting functionality.
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
import scipy.sparse as _sps
import copy as _copy
import warnings as _warnings
import itertools as _itertools
import collections as _collections
from pygsti.baseobjs import Basis as _Basis
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.tools import fogitools as _fogit


#BEGIN NEW CODE - just support code; could put this somewhere else:
class ElementaryErrorgenBasis(object):
    """
    A basis for error-generator space defined by a set of elementary error generators.

    Elements are ordered (have definite indices) and labeled.
    Intersection and union can be performed as a set.
    """

    def label_indices(self, labels):
        """ TODO: docstring """
        return [self.label_index(lbl) for lbl in labels]

    def __len__(self):
        """ Number of elementary errorgen elements in this basis """
        return len(self.labels)


class ExplicitElementaryErrorgenBasis(ElementaryErrorgenBasis):

    def __init__(self, state_space, labels, basis1q=None):
        # TODO: docstring - labels must be of form (sslbls, elementary_errorgen_lbl)
        self._labels = tuple(labels) if not isinstance(labels, tuple) else labels
        self._label_indices = _collections.OrderedDict([(lbl, i) for i, lbl in enumerate(self._labels)])
        self.basis_1q = basis1q if (basis1q is not None) else _Basis.cast('pp', 4)

        self.state_space = state_space
        assert(self.state_space.is_entirely_qubits()), "FOGI only works for models containing just qubits (so far)"
        sslbls = self.state_space.tensor_product_block_labels(0)  # all the model's state space labels
        self.sslbls = sslbls  # the "support" of this space - the qubit labels
        self._cached_elements = None

    @property
    def labels(self):
        return self._labels

    @property
    def elemgen_supports_and_matrices(self):
        if self._cached_elements is None:
            self._cached_elements = tuple(
                ((support, _ot.lindblad_error_generator(elemgen_label, self.basis_1q, normalize=True,
                                                        sparse=False, tensorprod_basis=True))
                 for support, elemgen_label in self.labels))
        return self._cached_elements

    def label_index(self, label):
        """
        TODO: docstring
        """
        return self._label_indices[label]

    #@property
    #def sslbls(self):
    #    """ The support of this errorgen space, e.g., the qubits where its elements may be nontrivial """
    #    return self.sslbls

    def create_subbasis(self, must_overlap_with_these_sslbls):
        """
        Create a sub-basis of this basis by including only the elements
        that overlap the given support (state space labels)
        """
        sub_sslbls = set(must_overlap_with_these_sslbls)

        def overlaps(sslbls):
            ret = len(set(sslbls).intersection(must_overlap_with_these_sslbls)) > 0
            if ret: sub_sslbls.update(sslbls)  # keep track of all overlaps
            return ret

        sub_labels, sub_indices = zip(*[(lbl, i) for i, lbl in enumerate(self._labels)
                                        if overlaps(lbl[0])])

        sub_state_space = self.state_space.create_subspace(sub_sslbls)
        return ExplicitElementaryErrorgenBasis(sub_state_space, sub_labels, self.basis_1q)

    def union(self, other_basis):
        present_labels = self._label_indices.copy()  # an OrderedDict, indices don't matter here
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            present_labels.update(other_basis._label_indices)
        else:

            for other_lbl in other_basis.labels:
                if other_lbl not in present_labels:
                    present_labels[other_lbl] = True

        union_state_space = self.state_space.union(other_basis.state_space)
        return ExplicitElementaryErrorgenBasis(union_state_space, tuple(present_labels.keys()), self.basis_1q)

    def intersection(self, other_basis):
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            common_labels = tuple((lbl for lbl in self.labels if lbl in other_basis._label_indices))
        else:
            other_labels = set(other_basis.labels)
            common_labels = tuple((lbl for lbl in self.labels if lbl in other_labels))

        intersection_state_space = self.state_space.intersection(other_basis.state_space)
        return ExplicitElementaryErrorgenBasis(intersection_state_space, common_labels, self.basis_1q)

    def difference(self, other_basis):
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            remaining_labels = tuple((lbl for lbl in self.labels if lbl not in other_basis._label_indices))
        else:
            other_labels = set(other_basis.labels)
            remaining_labels = tuple((lbl for lbl in self.labels if lbl not in other_labels))

        remaining_state_space = self.state_space  # TODO: see if we can reduce this space based on remaining_labels?
        return ExplicitElementaryErrorgenBasis(remaining_state_space, remaining_labels, self.basis_1q)


class CompleteElementaryErrorgenBasis(ElementaryErrorgenBasis):
    """
    Spanned by the elementary error generators of given type(s) (e.g. "Hamiltonian" and/or "other")
    and with elements corresponding to a `Basis`, usually of Paulis.
    """

    @classmethod
    def _create_diag_labels_for_support(cls, support, type_str, nontrivial_bels):
        weight = len(support)

        def _basis_el_strs(possible_bels, wt):
            for els in _itertools.product(*([possible_bels] * wt)):
                yield ''.join(els)

        return [(support, (type_str, bel)) for bel in _basis_el_strs(nontrivial_bels, weight)]

    @classmethod
    def _create_all_labels_for_support(cls, support, left_support, type_str, trivial_bel, nontrivial_bels):
        n = len(support)  # == weight
        all_bels = trivial_bel + nontrivial_bels
        left_weight = len(left_support)
        if len(left_weight) < n:  # n1 < n
            factors = [nontrivial_bels if x in left_support else trivial_bel for x in support] \
                + [all_bels if x in left_support else nontrivial_bels for x in support]
            return [(support, (type_str, ''.join(beltup[0:n]), ''.join(beltup[n:])))
                    for beltup in _itertools.product(*factors)]
            # (factors == left_factors + right_factors above)
        else:  # n1 == n
            ret = []
            for left_beltup in _itertools.product(*([nontrivial_bels] * n)):  # better itertools call here TODO
                left_bel = ''.join(left_beltup)
                right_it = _itertools.product(*([all_bels] * n))  # better itertools call here TODO
                right_it.next()  # advance past first (all I) element - assume trivial el = first!!
                ret.extend([(support, (type_str, left_bel, ''.join(right_beltup)))
                            for right_beltup in right_it])
            return ret

    @classmethod
    def _create_ordered_labels(cls, type_str, basis_1q, state_space, mode='diagonal',
                               max_weight=None, must_overlap_with_these_sslbls=None,
                               include_offsets=False):
        offsets = {}
        labels = []
        #labels_by_support = _collections.OrderedDict()
        #all_bels = basis_1q.labels[0:]
        trivial_bel = [basis_1q.labels[0]]
        nontrivial_bels = basis_1q.labels[1:]  # assume first element is identity

        if max_weight is None:
            assert(state_space.is_entirely_qubits()), "FOGI only works for models containing just qubits (so far)"
            sslbls = state_space.tensor_product_block_labels(0)  # all the model's state space labels
            max_weight = len(sslbls)

        # Let k be len(nontrivial_bels)
        if mode == "diagonal":
            # --> for each set of n qubit labels, there are k^n Hamiltonian terms with weight n
            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):  # NOTE: combinations *MUST* be deterministic
                    if (must_overlap_with_these_sslbls is not None
                       and len(set(must_overlap_with_these_sslbls).intersection(support)) == 0):
                        continue
                    offsets[support] = len(labels)
                    labels.extend(cls._create_diag_labels_for_support(support, type_str, nontrivial_bels))

        elif mode == "all":
            # --> for each weight n, must compute all *pairs* that have this weight.
            #  This is done via algorithm:
            #    Given n qubit labels (the support, number of labels == weight),
            #    loop over all left-hand weights n1 ranging from 1 to n
            #      loop over all left-supports of size n1 (choose some number of left factors to be nontrivial)
            #        Note: right-side *must* be nontrivial on complement of left support, can can be anything
            #              on factors in the left support (since the left side is nontrivial here) *except*
            #              the right side can't be all the trivial element.
            #        loop over all left-side elements (nNontrivialBELs^n1 of them)
            #          loop over right factors - the number of elements will be:
            #          nNontrivialBELs^(n-n1) * n1QBasisEls^n1 if (n1 < n) else (n1QBasisEls^n - 1)

            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):
                    if (must_overlap_with_these_sslbls is not None
                       and len(must_overlap_with_these_sslbls.intersection(support)) == 0):
                        continue

                    for left_weight in range(1, weight + 1):
                        for left_support in _itertools.combinations(support, left_weight):
                            offsets[(support, left_support)] = len(labels)
                            labels.extend(cls._create_all_labels_for_support(support, left_support, type_str,
                                                                             trivial_bel, nontrivial_bels))
        else:
            raise ValueError("Invalid mode: %s" % str(mode))
        offsets['END'] = len(labels)

        return (labels, offsets) if include_offsets else labels

    @classmethod
    def _create_ordered_label_offsets(cls, type_str, basis_1q, state_space, mode='diagonal',
                                      max_weight=None, must_overlap_with_these_sslbls=None,
                                      return_total_support=False):
        """ same as _create_ordered_labels but doesn't actually create the labels - just counts them to get offsets. """
        offsets = {}
        off = 0  # current number of labels that we would have created
        n1Q_bels = len(basis_1q.labels)
        n1Q_nontrivial_bels = n1Q_bels - 1  # assume first element is identity
        total_support = set()

        if max_weight is None:
            assert(state_space.is_entirely_qubits()), "FOGI only works for models containing just qubits (so far)"
            sslbls = state_space.tensor_product_block_labels(0)  # all the model's state space labels
            max_weight = len(sslbls)

        # Let k be len(nontrivial_bels)
        if mode == "diagonal":
            # --> for each set of n qubit labels, there are k^n Hamiltonian terms with weight n
            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):  # NOTE: combinations *MUST* be deterministic
                    if (must_overlap_with_these_sslbls is not None
                       and len(set(must_overlap_with_these_sslbls).intersection(support)) == 0):
                        continue
                    offsets[support] = off
                    off += n1Q_nontrivial_bels**weight
                    total_support.update(support)

        elif mode == "all":
            # --> for each weight n, must compute all *pairs* that have this weight.
            #  This is done via algorithm:
            #    Given n qubit labels (the support, number of labels == weight),
            #    loop over all left-hand weights n1 ranging from 1 to n
            #      loop over all left-supports of size n1 (choose some number of left factors to be nontrivial)
            #        Note: right-side *must* be nontrivial on complement of left support, can can be anything
            #              on factors in the left support (since the left side is nontrivial here) *except*
            #              the right side can't be all the trivial element.
            #        loop over all left-side elements (nNontrivialBELs^n1 of them)
            #          loop over right factors - the number of elements will be:
            #          nNontrivialBELs^(n-n1) * n1QBasisEls^n1 if (n1 < n) else (n1QBasisEls^n - 1)

            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):
                    if (must_overlap_with_these_sslbls is not None
                       and len(must_overlap_with_these_sslbls.intersection(support)) == 0):
                        continue

                    total_support.update(support)
                    for left_weight in range(1, weight + 1):
                        n, n1 = weight, left_weight
                        for left_support in _itertools.combinations(support, left_weight):
                            offsets[(support, left_support)] = off
                            off += (n1Q_nontrivial_bels**(n - n1) * n1Q_bels**n1) if (n1 < n) else (n1Q_bels**n - 1)
        else:
            raise ValueError("Invalid mode: %s" % str(mode))
        offsets['END'] = off

        return (offsets, total_support) if return_total_support else offsets

    def __init__(self, basis_1q, state_space, other_mode, max_ham_weight=None, max_other_weight=None,
                 must_overlap_with_these_sslbls=None):
        self._basis_1q = basis_1q
        self._other_mode = other_mode
        self.state_space = state_space
        self._max_ham_weight = max_ham_weight
        self._max_other_weight = max_other_weight
        self._must_overlap_with_these_sslbls = must_overlap_with_these_sslbls

        assert(self.state_space.is_entirely_qubits()), "FOGI only works for models containing just qubits (so far)"

        self._h_offsets, hsup = self._create_ordered_label_offsets('H', self._basis_1q, self.state_space,
                                                                   'diagonal', self._max_ham_weight,
                                                                   self._must_overlap_with_these_sslbls,
                                                                   return_total_support=True)
        self._hs_border = self._h_offsets['END']
        self._s_offsets, ssup = self._create_ordered_label_offsets('S', self._basis_1q, self.state_space,
                                                                   other_mode, self._max_other_weight,
                                                                   self._must_overlap_with_these_sslbls,
                                                                   return_total_support=True)

        #Note: state space can have additional labels that aren't in support
        # (this is, I think, only true when must_overlap_with_these_sslbls != None)
        sslbls = self.state_space.tensor_product_block_labels(0)  # all the model's state space labels
        present_sslbls = hsup.union(ssup)  # set union
        if set(sslbls) == present_sslbls:
            self.sslbls = sslbls  # the "support" of this space - the qubit labels
        elif present_sslbls.issubset(sslbls):
            self.state_space = self.state_space.create_subspace(present_sslbls)
            self.sslbls = present_sslbls
        else:
            # this should never happen - somehow the statespace doesn't have all the labels!
            assert(False), "Logic error! State space doesn't contain all of the present labels!!"

        #FUTURE: cache these for speed?  - but could just create an explicit basis which would be more transparent
        #self._cached_labels = None
        #self._cached_elements = None

        # Notes on ordering of labels:
        # - let there be k nontrivial 1-qubit basis elements (usually k=3)
        # - loop over all sets of qubit labels, increasing in length
        # - for a set of n qubit labels, there are k^n Hamiltonian terms with weight n,
        #    and either k^n "other" terms of weight n or something much more complicated:
        #    all pairs such that 1 or 2 members is nontrivial for each qubit:

        #    e.g. for 1 qubit: [(ntriv,), (ntriv,)] or  (X,X)  Note: can't have an all-triv element (I..I excluded)
        #    e.g. for 2 qubit: [(triv,ntriv), (ntriv,triv)] or   (IX,XI)  -- k * k elements
        #                      [(triv,ntriv), (ntriv,ntriv)] or   (IX,XX) -- k^3 elements
        #                      [(ntriv,triv), (triv,ntriv)] or   (XI,IX)  -- etc...
        #                      [(ntriv,triv), (ntriv,ntriv)] or   (XI,XX)
        #                      [(ntriv,ntriv), (triv,ntriv)] or   (XX,IX)
        #                      [(ntriv,ntriv), (ntriv,triv)] or   (XX,XI)
        #                      [(ntriv,ntriv), (ntriv,ntriv)] or   (XX,XX) -- k^4 elements (up to k^(2n) in general)
        #    e.g. for 3 qubit: (IIX,XXI)  # start with weight-1's on left
        #                      (IIX,XXX)  #   loop over filling in the (at least 1) nontrivial left-index with trival & nontrivial on right
        #                      (IXI,XIX)
        #                      (IXI,XXX)
        #                      ...
        #                      (IXX,XII) # move to weight-2s on left
        #                      (IXX,XXI) #   on right, loop over all possible choices of at least one, an at most m,
        #                      (IXX,XXX) #    nontrivial indices to place within the m nontrivial left indices (1 & 2 in this case)

    def __len__(self):
        """ Number of elementary errorgen elements in this basis """
        return self._h_offsets['END'] + self._s_offsets['END']

    def to_explicit_basis(self):
        return ExplicitElementaryErrorgenBasis(self.state_space, self.labels, self._basis_1q)

    @property
    def labels(self):
        hlabels = self._create_ordered_labels('H', self._basis_1q, self.state_space,
                                              'diagonal', self._max_ham_weight,
                                              self._must_overlap_with_these_sslbls)
        slabels = self._create_ordered_labels('S', self._basis_1q, self.state_space,
                                              self._other_mode, self._max_other_weight,
                                              self._must_overlap_with_these_sslbls)
        return tuple(hlabels + slabels)

    @property
    def elemgen_supports_and_matrices(self):
        return tuple(((support,
                       _ot.lindblad_error_generator(elemgen_label, self._basis_1q, normalize=True,
                                                    sparse=False, tensorprod_basis=True))
                      for support, elemgen_label in self.labels))

    def label_index(self, label):
        """
        TODO: docstring
        """
        support, elemgen_lbl = label
        type_str = elemgen_lbl[0]
        trivial_bel = self._basis_1q.labels[0]  # assumes first element is identity
        nontrivial_bels = self._basis_1q.labels[1:]

        if type_str == 'H' or (type_str == 'S' and self._other_mode == 'diagonal'):
            base = self._h_offsets[support] if (type_str == 'H') else (self._hs_border + self._s_offsets[support])
            indices = {lbl: i for i, lbl in enumerate(self._create_diag_labels_for_support(support, type_str,
                                                                                           nontrivial_bels))}
        elif elemgen_lbl[0] == 'S':
            assert(self._other_mode == 'all'), "Invalid 'other' mode: %s" % str(self._other_mode)
            assert(len(trivial_bel) == 1)  # assumes this is a single character
            nontrivial_inds = [i for i, letter in enumerate(elemgen_lbl[1]) if letter != trivial_bel]
            left_support = tuple([self.sslbls[i] for i in nontrivial_inds])
            base = self._hs_border + self._s_offsets[(support, left_support)]

            indices = {lbl: i for i, lbl in enumerate(self._create_all_labels_for_support(
                support, left_support, 'S', [trivial_bel], nontrivial_bels))}
        else:
            raise ValueError("Invalid label type: %s" % str(elemgen_lbl[0]))

        return base + indices[label]

    #@property
    #def sslbls(self):
    #    """ The support of this errorgen space, e.g., the qubits where its elements may be nontrivial """
    #    return self.sslbls

    def create_subbasis(self, must_overlap_with_these_sslbls, retain_max_weights=True):
        """
        Create a sub-basis of this basis by including only the elements
        that overlap the given support (state space labels)
        """
        #Note: state_space is automatically reduced within __init__ when necessary, e.g., when
        # `must_overlap_with_these_sslbls` is non-None and considerably reduces the basis.
        return CompleteElementaryErrorgenBasis(self._basis_1q, self.state_space, self._other_mode,
                                               self._max_ham_weight if retain_max_weights else None,
                                               self._max_other_weight if retain_max_weights else None,
                                               must_overlap_with_these_sslbls)

    def union(self, other_basis):
        # don't convert this basis to an explicit one unless it's necessary -
        # if `other_basis` is already an explicit basis then let it do the work.
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            return other_basis.union(self)
        else:
            return self.to_explicit_basis().union(other_basis)

    def intersection(self, other_basis):
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            return other_basis.intersection(self)
        else:
            return self.to_explicit_basis().intersection(other_basis)

    def difference(self, other_basis):
        return self.to_explicit_basis().difference(other_basis)


#OLD - maybe not needed?
#class LowWeightElementaryErrorgenBasis(ElementaryErrorgenBasis):
#    """
#    Spanned by the elementary error generators of given type(s) (e.g. "Hamiltonian" and/or "other")
#    and with elements corresponding to a `Basis`, usually of Paulis.
#    """
#
#    def __init__(self, basis_1q, state_space, other_mode, max_ham_weight=None, max_other_weight=None,
#                 must_overlap_with_these_sslbls=None):
#        self._basis_1q = basis_1q
#        self._other_mode = other_mode
#        self.state_space = state_space
#        self._max_ham_weight = max_ham_weight
#        self._max_other_weight = max_other_weight
#        self._must_overlap_with_these_sslbls = must_overlap_with_these_sslbls
#
#        assert(self.state_space.is_entirely_qubits()), "FOGI only works for models containing just qubits (so far)"
#        sslbls = self.state_space.tensor_product_block_labels(0)  # all the model's state space labels
#        self.sslbls = sslbls  # the "support" of this space - the qubit labels
#
#        self._cached_label_indices = None
#        self._cached_labels_by_support = None
#        self._cached_elements = None
#
#        #Needed?
#        # self.dim = len(self.labels)  # TODO - update this so we don't always need to build labels
#        # # (this defeats lazy building via property below) - we can just compute this, especially if
#        # # not too fancy
#
#    @property
#    def labels(self):
#        if self._cached_label_indices is None:
#
#            def _basis_el_strs(possible_bels, wt):
#                for els in _itertools.product(*([possible_bels] * wt)):
#                    yield ''.join(els)
#
#            labels = {}
#            all_bels = self.basis_1q.labels[1:]  # assume first element is identity
#            nontrivial_bels = self.basis_1q.labels[1:]  # assume first element is identity
#
#            max_weight = self._max_ham_weight if (self._max_ham_weight is not None) else len(self.sslbls)
#            for weight in range(1, max_weight + 1):
#                for support in _itertools.combinations(self.sslbls, weight):
#                    if (self._must_overlap_with_these_sslbls is not None
#                       and len(self._must_overlap_with_these_sslbls.intersection(support)) == 0):
#                        continue
#                    if support not in labels: labels[support] = []  # always True?
#                    labels[support].extend([('H', bel) for bel in _basis_el_strs(nontrivial_bels, weight)])
#
#            max_weight = self._max_other_weight if (self._max_other_weight is not None) else len(self.sslbls)
#            if self._other_mode != "all":
#                for weight in range(1, max_weight + 1):
#                    for support in _itertools.combinations(self.sslbls, weight):
#                        if (self._must_overlap_with_these_sslbls is not None
#                           and len(self._must_overlap_with_these_sslbls.intersection(support)) == 0):
#                            continue
#                        if support not in labels: labels[support] = []
#                        labels[support].extend([('S', bel) for bel in _basis_el_strs(nontrivial_bels, weight)])
#            else:
#                #This is messy code that relies on basis labels being single characters -- TODO improve(?)
#                idle_char = self.basis_1q.labels[1:]  # assume first element is identity
#                assert(len(idle_char) == 1 and all([len(c) == 1 for c in nontrivial_bels])), \
#                    "All basis el labels must be single chars for other_mode=='all'!"
#                for support in _itertools.combinations(self.sslbls, max_weight):
#                    # Loop over all possible basis elements for this max-weight support, computing the actual support
#                    # of each one individually and appending it to the appropriate list
#                    for bel1 in _basis_el_strs(all_bels, max_weight):
#                        nonidle_indices1 = [i for i in range(max_weight) if bel1[i] != idle_char]
#                        for bel2 in _basis_el_strs(all_bels, max_weight):
#                            nonidle_indices2 = [i for i in range(max_weight) if bel2[i] != idle_char]
#                            nonidle_indices = list(sorted(set(nonidle_indices1) + set(nonidle_indices2)))
#                            bel1 = ''.join([bel1[i] for i in nonidle_indices])  # trim to actual support
#                            bel2 = ''.join([bel2[i] for i in nonidle_indices])  # trim to actual support
#                            actual_support = tuple([support[i] for i in nonidle_indices])
#
#                            if (self._must_overlap_with_these_sslbls is not None
#                               and len(self._must_overlap_with_these_sslbls.intersection(actual_support)) == 0):
#                                continue
#
#                            if actual_support not in labels: labels[actual_support] = []
#                            labels[actual_support].append(('S', bel1, bel2))
#
#            self._cached_labels_by_support = labels
#            self._cached_label_indices = _collections.OrderedDict(((support_lbl, i) for i, support_lbl in enumerate(
#                ((support, lbl) for support, lst in labels.items() for lbl in lst))))
#
#        return tuple(self._cached_label_indices.keys())
#
#    @property
#    def element_supports_and_matrices(self):
#        if self._cached_elements is None:
#            self._cached_elements = tuple(
#                ((support, _ot.lindblad_error_generator(elemgen_label, self.basis_1q, normalize=True, sparse=False))
#                 for support, elemgen_label in self.labels))
#        return self._cached_elements
#
#    def element_index(self, label):
#        """
#        TODO: docstring
#        """
#        if self._cached_label_indices is None:
#            self.labels  # triggers building of labels
#        return self._cached_label_indices[label]
#
#    @property
#    def sslbls(self):
#        """ The support of this errorgen space, e.g., the qubits where its elements may be nontrivial """
#        return self.sslbls
#
#    def create_subbasis(self, must_overlap_with_these_sslbls, retain_max_weights=True):
#        """
#        Create a sub-basis of this basis by including only the elements
#        that overlap the given support (state space labels)
#        """
#        #Note: can we reduce self.state_space?
#        return CompleteErrorgenBasis(self._basis_1q, self.state_space, self._other_mode,
#                                     self._max_ham_weight if retain_max_weights else None,
#                                     self._max_other_weight if retain_max_weights else None,
#                                     self._must_overlap_with_these_sslbls)


class ErrorgenSpace(object):
    """
    A vector space of error generators, spanned by some basis.

    This object collects the information needed to specify a space
    within the space of all error generators.
    """

    def __init__(self, vectors, basis):
        self.vectors = vectors
        self.elemgen_basis = basis
        #Question: have multiple bases or a single one?
        #self._vectors = [] if (items is None) else items  # list of (basis, vectors_mx) pairs
        # map sslbls => (vectors, basis) where basis.sslbls == sslbls
        # or basis => vectors if bases can hash well(?)

    def intersection(self, other_space, free_on_unspecified_space=False):
        """
        TODO: docstring
        """
        #Note: currently we assume self.vectors is a *dense* numpy array, but this may/should be expanded to
        # work with or solely utilize SPARSE matrices in the future.
        dtype = self.vectors.dtype

        if free_on_unspecified_space:
            common_basis = self.elemgen_basis.union(other_space.elemgen_basis)
            diff_self = common_basis.difference(self.elemgen_basis)
            diff_other = common_basis.difference(other_space.elemgen_basis)
            Vl, Vli, Wl, Wli = (self.vectors.shape[1], len(diff_self), other_space.vectors.shape[1], len(diff_other))

            #Fill in matrix to take nullspace of: [ V I | W I ] where V and W are self's and other_space's vectors
            # in the common basis and I's stand for filling in the identity on rows corresponding to missing elements
            # in each spaces basis, respectively.
            i = 0  # column offset
            VIWI = _np.zeros((len(common_basis), Vl + Vli + Wl + Wli), dtype)  # SPARSE in future?
            VIWI[common_basis.label_indices(self.elemgen_basis.labels), 0:Vl] = self.vectors[:, :]; i += Vl
            VIWI[common_basis.label_indices(diff_self.labels), i:i + Vli] = _np.identity(Vli, dtype); i += Vli
            VIWI[common_basis.label_indices(other_space.elemgen_basis.labels), i:i + Wl] = other_space.vectors[:, :]
            i += Wl
            VIWI[common_basis.label_indices(diff_other.labels), i:i + Wli] = _np.identity(Wli, dtype)

            ns = _mt.nullspace(VIWI)
            intersection_vecs = _np.dot(VIWI[:, 0:(Vl + Vli)], ns[0:(Vl + Vli), :])  # on common_basis

        else:
            common_basis = self.elemgen_basis.intersection(other_space.elemgen_basis)
            Vl, Wl = (self.vectors.shape[1], other_space.vectors.shape[1])

            #Fill in matrix to take nullspace of: [ V | W ] restricted to rows corresponding to shared elementary
            # error generators (if one space has a elemgen in its basis that the other doesn't, then any intersection
            # vector cannot contain this elemgen).
            VW = _np.zeros((len(common_basis), Vl + Wl), dtype)  # SPARSE in future?
            VW[:, 0:Vl] = self.vectors[self.elemgen_basis.label_indices(common_basis.labels), :]
            VW[:, Vl:] = other_space.vectors[other_space.elemgen_basis.label_indices(common_basis.labels), :]

            ns = _mt.nullspace(VW)
            intersection_vecs = _np.dot(VW[:, 0:Vl], ns[0:Vl, :])  # on common_basis

        return ErrorgenSpace(intersection_vecs, common_basis)

    def union(self, other_space):
        """
        TODO: docstring
        """
        raise NotImplementedError("TODO in FUTURE")


#class LowWeightErrorgenSpace(ErrorgenSpace):
#    """
#    Like a SimpleErrorgenSpace but spanned by only the elementary error generators corresponding to
#    low-weight (up to some maximum weight) basis elements
#    (so far, only Pauli-product bases work for this, since `Basis` objects don't keep track of each
#    element's weight (?)).
#    """
#    pass

#END NEW CODE


class FirstOrderGaugeInvariantStore(object):
    """
    An object that computes and stores the first-order-gauge-invariant quantities of a model.

    Currently, it is only compatible with :class:`ExplicitOpModel` objects.
    """

    def __init__(self, gauge_action_matrices_by_op, gauge_action_gauge_spaces_by_op, errorgen_coefficient_labels_by_op,
                 op_label_abbrevs=None, reduce_to_model_space=True,
                 dependent_fogi_action='drop', norm_order=None):
        """
        TODO: docstring
        """

        self.primitive_op_labels = tuple(gauge_action_matrices_by_op.keys())

        # Construct common gauge space by special way of intersecting the gauge spaces for all the ops
        # Note: the gauge_space of each op is constructed (see `setup_fogi`) so that the gauge action is
        #  zero on any elementary error generator not in the elementary-errorgen basis associated with
        #  gauge_space (gauge_space is the span of linear combos of a elemgen basis that was chosen to
        #  include all possible non-trivial (non-zero) gauge actions by the operator (gauge action is the
        #  *difference* K - UKU^dag in the Lindblad mapping under gauge transform exp(K),  L -> L + K - UKU^dag)
        common_gauge_space = None
        for op_label, gauge_space in gauge_action_gauge_spaces_by_op.items():
            print("DEBUG gauge space of ", op_label, "has dim", gauge_space.vectors.shape[1])
            if common_gauge_space is None:
                common_gauge_space = gauge_space
            else:
                common_gauge_space = common_gauge_space.intersection(gauge_space,
                                                                     free_on_unspecified_space=True)

        # column space of self.fogi_directions
        print("DEBUG common gauge space of has dim", common_gauge_space.vectors.shape[1])
        self.gauge_space = common_gauge_space

        # row space of self.fogi_directions - "errgen-set space" lookups
        # -- maybe make this into an "ErrgenSetSpace" object in FUTURE?
        self.elem_errorgen_labels_by_op = errorgen_coefficient_labels_by_op

        #HERE - need to build this out of row spaces?
        self.op_errorgen_indices = _fogit._create_op_errgen_indices_dict(self.primitive_op_labels,
                                                                         self.elem_errorgen_labels_by_op)
        self.errorgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                                    for elem_lbl in self.elem_errorgen_labels_by_op[op_label]])

        num_elem_errgens = sum([len(labels) for labels in self.elem_errorgen_labels_by_op.values()])
        allop_gauge_action = _sps.lil_matrix((num_elem_errgens, self.gauge_space.vectors.shape[1]), dtype=complex)

        # Now update (restrict) each op's gauge_action to use common gauge space
        # - need to write the vectors of the common (final) gauge space, w_i, as linear combos of
        #   the op's original gauge space vectors, v_i.
        # - ignore elemgens that are not in the op's orig_gauge_space's elemgen basis
        # W = V * alpha, and we want to find alpha.  W and V are the vectors in the op's elemgen basis
        #  (these could be seen as staring in the union of the common gauge's and op's elemgen
        #   bases - which would just be the common gauge's elemgen basis since it's strictly larger -
        #   restricted to the op's elemben basis)
        for op_label, orig_gauge_space in gauge_action_gauge_spaces_by_op.items():
            print("DEBUG: ", op_label, orig_gauge_space.vectors.shape, len(orig_gauge_space.elemgen_basis))
            gauge_action = gauge_action_matrices_by_op[op_label]  # a sparse matrix

            op_elemgen_lbls = orig_gauge_space.elemgen_basis.labels
            W = common_gauge_space.vectors[common_gauge_space.elemgen_basis.label_indices(op_elemgen_lbls), :]
            V = orig_gauge_space.vectors
            alpha = _np.dot(_np.linalg.pinv(V), W)  # make SPARSE compatible in future if space vectors are sparse
            alpha = _sps.csr_matrix(alpha)  # convert to dense -> CSR for now, as if we did sparse math above

            # update gauge action to use common gauge space
            sparse_gauge_action = gauge_action.dot(alpha)
            allop_gauge_action[self.op_errorgen_indices[op_label], :] = sparse_gauge_action[:, :]
            gauge_action_matrices_by_op[op_label] = sparse_gauge_action.toarray()  # make **DENSE** here
            # Hopefully matrices aren't too larget after this reduction and dense matrices are ok,
            # otherwise we need to change downstream nullspace and pseduoinverse operations to be sparse compatible.

            #FUTURE: if update above creates zero-rows in gauge action matrix, maybe remove
            # these rows from the row basis, i.e. self.elem_errorgen_labels_by_op[op_label]

        self.gauge_action_for_op = gauge_action_matrices_by_op

        (indep_fogi_directions, indep_fogi_metadata, dep_fogi_directions, dep_fogi_metadata) = \
            _fogit.construct_fogi_quantities(self.primitive_op_labels, self.gauge_action_for_op,
                                             self.elem_errorgen_labels_by_op, self.op_errorgen_indices,
                                             self.gauge_space, op_label_abbrevs, dependent_fogi_action, norm_order)
        self.fogi_directions = _sps.hstack((indep_fogi_directions, dep_fogi_directions))
        self.fogi_metadata = indep_fogi_metadata + dep_fogi_metadata  # list concatenation
        self.dependent_dir_indices = _np.arange(len(indep_fogi_metadata), len(self.fogi_metadata))
        for j, meta in enumerate(self.fogi_metadata):
            meta['raw'] = _fogit.op_elem_vec_name(self.fogi_directions[:, j], self.errorgen_space_op_elem_labels,
                                                  op_label_abbrevs if (op_label_abbrevs is not None) else {})

        assert(len(self.errorgen_space_op_elem_labels) == self.fogi_directions.shape[0])

        # Note: currently PUNT on doing below with sparse math, as a sparse nullspace routine is unavailable

        #First order gauge *variant* directions (the complement of FOGI directions in errgen set space)
        # (directions in errorgen space that correspond to gauge transformations -- to first order)
        #fogv_directions = _mt.nice_nullspace(self.fogi_directions.T)  # can be dependent!
        self.fogv_directions = _mt.nullspace(self.fogi_directions.toarray().T)  # complement of fogi directions
        self.fogv_directions = _sps.csc_matrix(self.fogv_directions)  # as though we used sparse math above
        self.fogv_labels = ["%d gauge action" % i for i in range(self.fogv_directions.shape[1])]
        #self.fogv_labels = ["%s gauge action" % nm
        #                    for nm in _fogit.elem_vec_names(gauge_space_directions, gauge_elemgen_labels)]

        #Get gauge-space directions corresponding to the fogv directions
        self.allop_gauge_action = allop_gauge_action
        pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action.toarray(), rcond=1e-7)  # errgen-set -> gauge-gen space
        gauge_space_directions = _np.dot(pinv_allop_gauge_action, self.fogv_directions)  # in gauge-generator space
        self.gauge_space_directions = gauge_space_directions

        #Notes on error-gen vs gauge-gen space:
        # self.fogi_directions and self.fogv_directions are dual vectors in error-generator space,
        # i.e. with elements corresponding to the elementary error generators given by self.errorgen_space_op_elem_labels
        # self.fogi_gaugespace_directions contains, when applicable, a gauge-space direction that
        # correspondings to the FOGI quantity in self.fogi_directions.  Such a gauge-space direction
        # exists for relational FOGI quantities, where the FOGI quantity is constructed by taking the
        # *difference* of a gauge-space action (given by the gauge-space direction) on two operations
        # (or sets of operations).

        #Store auxiliary info for later use
        self.norm_order = norm_order
        self._dependent_fogi_action = dependent_fogi_action

        #Assertions to check that everything looks good
        if True:
            fogi_dirs = self.fogi_directions.toarray()  # don't bother with sparse math yet
            fogv_dirs = self.fogv_directions.toarray()

            # We must reduce X_gauge_action to the "in-model gauge space" before testing whether the computed vecs are FOGI:
            assert(_np.linalg.norm(_np.dot(self.allop_gauge_action.toarray().T, fogi_dirs)) < 1e-8)

            #Check that pseudo-inverse was computed correctly (~ matrices are full rank)
            # fogi_coeffs = dot(fogi_directions.T, elem_errorgen_vec), where elem_errorgen_vec is filled from model params,
            #                 since fogi_directions columns are *dual* vectors in error-gen space.  Thus, to go in reverse:
            # elem_errogen_vec = dot(pinv_fogi_dirs_T, fogi_coeffs), where dot(fogi_directions.T, pinv_fogi_dirs_T) == I
            # (This will only be the case when fogi_vecs are linearly independent, so when dependent_indices == 'drop')

            if dependent_fogi_action == 'drop':
                #assert(_mt.columns_are_orthogonal(self.fogi_directions))  # not true unless we construct them so...
                assert(_np.linalg.norm(_np.dot(fogi_dirs.T, _np.linalg.pinv(fogi_dirs.T))
                                       - _np.identity(fogi_dirs.shape[1], 'd')) < 1e-6)

            # A similar relationship should always hold for the gauge directions, except for these we never
            #  keep linear dependencies
            assert(_mt.columns_are_orthogonal(fogv_dirs))
            assert(_np.linalg.norm(_np.dot(fogv_dirs.T, _np.linalg.pinv(fogv_dirs.T))
                                   - _np.identity(fogv_dirs.shape[1], 'd')) < 1e-6)

    def find_nice_fogiv_directions(self):
        # BELOW: an attempt to find nice FOGV directions - but we'd like all the vecs to be
        #  orthogonal and this seems to interfere with that, so we'll just leave the fogv dirs messy for now.
        #
        # # like to find LCs mix s.t.  dot(gauge_space_directions, mix) ~= identity, so use pinv
        # # then propagate this mixing to fogv_directions = dot(allop_gauge_action, mixed_gauge_space_directions)
        # mix = _np.linalg.pinv(gauge_space_directions)[:, 0:fogv_directions.shape[1]]  # use "full-rank" part of pinv
        # mixed_gauge_space_dirs = _np.dot(gauge_space_directions, mix)
        #
        # #TODO - better mix matrix?
        # #print("gauge_space_directions shape = ",gauge_space_directions.shape)
        # #print("mixed_gauge_space_dirs = ");  _mt.print_mx(gauge_space_directions, width=6, prec=2)
        # #U, s, Vh = _np.linalg.svd(gauge_space_directions, full_matrices=True)
        # #inv_s = _np.array([1/x if abs(x) > 1e-4 else 0 for x in s])
        # #print("shapes = ",U.shape, s.shape, Vh.shape)
        # #print("s = ",s)
        # #print(_np.linalg.norm(_np.dot(U,_np.conjugate(U.T)) - _np.identity(U.shape[0])))
        # #_mt.print_mx(U, width=6, prec=2)
        # #print("U * Udag = ")
        # #_mt.print_mx(_np.dot(U,_np.conjugate(U.T)), width=6, prec=2)
        # #print(_np.linalg.norm(_np.dot(Vh,Vh.T) - _np.identity(Vh.shape[0])))
        # #full_mix = _np.dot(Vh.T, _np.dot(_np.diag(inv_s), U.T))  # _np.linalg.pinv(gauge_space_directions)
        # #full_mixed_gauge_space_dirs = _np.dot(gauge_space_directions, full_mix)
        # #print("full_mixed_gauge_space_dirs = ");  _mt.print_mx(full_mixed_gauge_space_dirs, width=6, prec=2)
        #
        # self.fogv_labels = ["%s gauge action" % nm
        #                     for nm in _fogit.elem_vec_names(mixed_gauge_space_dirs, gauge_elemgen_labels)]
        # self.fogv_directions = _np.dot(self.allop_gauge_action, mixed_gauge_space_dirs)
        # # - directions in errorgen space that correspond to gauge transformations (to first order)

        #UNUSED - maybe useful just as a check? otherwise REMOVE
        #pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action, rcond=1e-7)  # maps error -> gauge-gen space
        #gauge_space_directions = _np.dot(pinv_allop_gauge_action, self.fogv_directions)  # in gauge-generator space
        #assert(_np.linalg.matrix_rank(gauge_space_directions) <= self._gauge_space_dim)  # should be nearly full rank
        pass

    #TODO: REMOVE (just kept for reference)
    def __OLDinit__(self, gauge_action_matrices_by_op, errorgen_coefficient_labels_by_op,
                    elem_errorgen_labels, op_label_abbrevs=None, reduce_to_model_space=True,
                    dependent_fogi_action='drop', norm_order=None):
        """
        TODO: docstring
        """

        self.primitive_op_labels = tuple(gauge_action_matrices_by_op.keys())
        #self.gauge_action_for_op = gauge_action_matrices_by_op
        #errorgen_coefficient_labels_by_op

        self.allop_gauge_action, self.gauge_action_for_op, self.elem_errorgen_labels_by_op, self.gauge_linear_combos = \
            _fogit.construct_gauge_space_for_model(self.primitive_op_labels, gauge_action_matrices_by_op,
                                                   errorgen_coefficient_labels_by_op, elem_errorgen_labels,
                                                   reduce_to_model_space)

        self.errorgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                                    for elem_lbl in self.elem_errorgen_labels_by_op[op_label]])

        if self.gauge_linear_combos is not None:
            self._gauge_space_dim = self.gauge_linear_combos.shape[1]
            self.gauge_elemgen_labels = [('G', str(i)) for i in range(self._gauge_space_dim)]
        else:
            self._gauge_space_dim = len(elem_errorgen_labels)
            self.gauge_elemgen_labels = elem_errorgen_labels

        (self.fogi_opsets, self.fogi_directions, self.fogi_r_factors, self.fogi_gaugespace_directions,
         self.dependent_dir_indices, self.op_errorgen_indices, self.fogi_labels, self.abbrev_fogi_labels) = \
            _fogit.construct_fogi_quantities(self.primitive_op_labels, self.gauge_action_for_op,
                                             self.elem_errorgen_labels_by_op, self.gauge_elemgen_labels,
                                             op_label_abbrevs, dependent_fogi_action, norm_order)
        self.norm_order = norm_order

        self.errorgen_space_labels = [(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                      for elem_lbl in self.elem_errorgen_labels_by_op[op_label]]  # same as flattened self.errgen_space_op_elem_labels
        assert(len(self.errorgen_space_labels) == self.fogi_directions.shape[0])

        #fogv_directions = _mt.nice_nullspace(self.fogi_directions.T)  # can be dependent!
        fogv_directions = _mt.nullspace(self.fogi_directions.T)  # complement of fogi directions

        pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action, rcond=1e-7)  # errgen-set -> gauge-gen space
        gauge_space_directions = _np.dot(pinv_allop_gauge_action, fogv_directions)  # in gauge-generator space
        self.gauge_space_directions = gauge_space_directions

        self.fogv_labels = ["%d gauge action" % i for i in range(gauge_space_directions.shape[1])]
        #self.fogv_labels = ["%s gauge action" % nm
        #                    for nm in _fogit.elem_vec_names(gauge_space_directions, gauge_elemgen_labels)]
        self.fogv_directions = fogv_directions
        # - directions in errorgen space that correspond to gauge transformations (to first order)

        # BELOW: an attempt to find nice FOGV directions - but we'd like all the vecs to be
        #  orthogonal and this seems to interfere with that, so we'll just leave the fogv dirs messy for now.
        #
        # # like to find LCs mix s.t.  dot(gauge_space_directions, mix) ~= identity, so use pinv
        # # then propagate this mixing to fogv_directions = dot(allop_gauge_action, mixed_gauge_space_directions)
        # mix = _np.linalg.pinv(gauge_space_directions)[:, 0:fogv_directions.shape[1]]  # use "full-rank" part of pinv
        # mixed_gauge_space_dirs = _np.dot(gauge_space_directions, mix)
        #
        # #TODO - better mix matrix?
        # #print("gauge_space_directions shape = ",gauge_space_directions.shape)
        # #print("mixed_gauge_space_dirs = ");  _mt.print_mx(gauge_space_directions, width=6, prec=2)
        # #U, s, Vh = _np.linalg.svd(gauge_space_directions, full_matrices=True)
        # #inv_s = _np.array([1/x if abs(x) > 1e-4 else 0 for x in s])
        # #print("shapes = ",U.shape, s.shape, Vh.shape)
        # #print("s = ",s)
        # #print(_np.linalg.norm(_np.dot(U,_np.conjugate(U.T)) - _np.identity(U.shape[0])))
        # #_mt.print_mx(U, width=6, prec=2)
        # #print("U * Udag = ")
        # #_mt.print_mx(_np.dot(U,_np.conjugate(U.T)), width=6, prec=2)
        # #print(_np.linalg.norm(_np.dot(Vh,Vh.T) - _np.identity(Vh.shape[0])))
        # #full_mix = _np.dot(Vh.T, _np.dot(_np.diag(inv_s), U.T))  # _np.linalg.pinv(gauge_space_directions)
        # #full_mixed_gauge_space_dirs = _np.dot(gauge_space_directions, full_mix)
        # #print("full_mixed_gauge_space_dirs = ");  _mt.print_mx(full_mixed_gauge_space_dirs, width=6, prec=2)
        #
        # self.fogv_labels = ["%s gauge action" % nm
        #                     for nm in _fogit.elem_vec_names(mixed_gauge_space_dirs, gauge_elemgen_labels)]
        # self.fogv_directions = _np.dot(self.allop_gauge_action, mixed_gauge_space_dirs)
        # # - directions in errorgen space that correspond to gauge transformations (to first order)

        #UNUSED - maybe useful just as a check? otherwise REMOVE
        #pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action, rcond=1e-7)  # maps error -> gauge-gen space
        #gauge_space_directions = _np.dot(pinv_allop_gauge_action, self.fogv_directions)  # in gauge-generator space
        #assert(_np.linalg.matrix_rank(gauge_space_directions) <= self._gauge_space_dim)  # should be nearly full rank

        #Notes on error-gen vs gauge-gen space:
        # self.fogi_directions and self.fogv_directions are dual vectors in error-generator space,
        # i.e. with elements corresponding to the elementary error generators given by self.errorgen_space_labels.
        # self.fogi_gaugespace_directions contains, when applicable, a gauge-space direction that
        # correspondings to the FOGI quantity in self.fogi_directions.  Such a gauge-space direction
        # exists for relational FOGI quantities, where the FOGI quantity is constructed by taking the
        # *difference* of a gauge-space action (given by the gauge-space direction) on two operations
        # (or sets of operations).

        self.raw_fogi_labels = _fogit.op_elem_vec_names(self.fogi_directions,
                                                        self.errorgen_space_labels, op_label_abbrevs)

        # We must reduce X_gauge_action to the "in-model gauge space" before testing whether the computed vecs are FOGI:
        assert(_np.linalg.norm(_np.dot(self.allop_gauge_action.T, self.fogi_directions)) < 1e-8)

        #Check that pseudo-inverse was computed correctly (~ matrices are full rank)
        # fogi_coeffs = dot(fogi_directions.T, elem_errorgen_vec), where elem_errorgen_vec is filled from model params,
        #                 since fogi_directions columns are *dual* vectors in error-gen space.  Thus, to go in reverse:
        # elem_errogen_vec = dot(pinv_fogi_dirs_T, fogi_coeffs), where dot(fogi_directions.T, pinv_fogi_dirs_T) == I
        # (This will only be the case when fogi_vecs are linearly independent, so when dependent_indices == 'drop')
        self._dependent_fogi_action = dependent_fogi_action
        if dependent_fogi_action == 'drop':
            #assert(_mt.columns_are_orthogonal(self.fogi_directions))  # not true unless we construct them so...
            assert(_np.linalg.norm(_np.dot(self.fogi_directions.T, _np.linalg.pinv(self.fogi_directions.T))
                                   - _np.identity(self.fogi_directions.shape[1], 'd')) < 1e-6)

        # A similar relationship should always hold for the gauge directions, except for these we never
        #  keep linear dependencies
        assert(_mt.columns_are_orthogonal(self.fogv_directions))
        assert(_np.linalg.norm(_np.dot(self.fogv_directions.T, _np.linalg.pinv(self.fogv_directions.T))
                               - _np.identity(self.fogv_directions.shape[1], 'd')) < 1e-6)

    @property
    def errorgen_space_dim(self):
        return self.fogi_directions.shape[0]

    @property
    def gauge_space_dim(self):
        return self.gauge_space.vectors.shape[1]

    @property
    def num_fogi_directions(self):
        return self.fogi_directions.shape[1]

    @property
    def num_fogv_directions(self):
        return self.fogv_directions.shape[1]

    def fogi_errorgen_direction_labels(self, typ='normal'):
        """ typ can be 'raw' or 'abbrev' too """
        if typ == 'normal': return tuple([meta['name'] for meta in self.fogi_metadata])
        elif typ == 'raw': return tuple([meta['raw'] for meta in self.fogi_metadata])
        elif typ == 'abrev': return tuple([meta['abbrev'] for meta in self.fogi_metadata])
        else: raise ValueError("Invalid `typ` argument: %s" % str(typ))

    def fogv_errorgen_direction_labels(self, typ='normal'):
        if typ == 'normal': labels = self.fogv_labels
        else: labels = [''] * len(self.fogv_labels)
        return tuple(labels)

    def errorgen_vec_to_fogi_components_array(self, errorgen_vec):
        fogi_coeffs = self.fogi_directions.transpose().dot(errorgen_vec)
        assert(_np.linalg.norm(fogi_coeffs.imag) < 1e-8)
        return fogi_coeffs.real

    def errorgen_vec_to_fogv_components_array(self, errorgen_vec):
        fogv_coeffs = self.fogv_directions.transpose().dot(errorgen_vec)
        assert(_np.linalg.norm(fogv_coeffs.imag) < 1e-8)
        return fogv_coeffs.real

    def opcoeffs_to_fogi_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec)

    def opcoeffs_to_fogv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def opcoeffs_to_fogiv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec), \
            self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def fogi_components_array_to_errorgen_vec(self, fogi_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogi components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(self.fogi_directions.toarray().T, rcond=1e-7), fogi_components)

    def fogv_components_array_to_errorgen_vec(self, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogv components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(self.fogv_directions.toarray().T, rcond=1e-7), fogv_components)

    def fogiv_components_array_to_errorgen_vec(self, fogi_components, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogiv components to an errorgen-set vec when fogi directions are "
             "linearly-dependent!  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(
            _np.concatenate((self.fogi_directions.asarray(), self.fogv_directions.asarray()), axis=1).T,
            rcond=1e-7), _np.concatenate((fogi_components, fogv_components)))

    def errorgen_vec_to_opcoeffs(self, errorgen_vec):
        op_coeffs = {op_label: {} for op_label in self.primitive_op_labels}
        for (op_label, elem_lbl), coeff_value in zip(self.errorgen_space_op_elem_labels, errorgen_vec):
            op_coeffs[op_label][elem_lbl] = coeff_value
        return op_coeffs

    def fogi_components_array_to_opcoeffs(self, fogi_components):
        return self.errorgen_vec_to_opcoeffs(self.fogi_components_array_to_errorgen_vec(fogi_components))

    def fogv_components_array_to_opcoeffs(self, fogv_components):
        return self.errorgen_vec_to_opcoeffs(self.fogv_components_array_to_errorgen_vec(fogv_components))

    def fogiv_components_array_to_opcoeffs(self, fogi_components, fogv_components):
        return self.errorgen_vec_to_opcoeffs(self.fogiv_components_array_to_errorgen_vec(
            fogi_components, fogv_components))

    def create_binned_fogi_infos(self, tol=1e-5):
        """
        Creates an 'info' dictionary for each FOGI quantity and places it within a
        nested dictionary structure by the operators involved, the types of error generators,
        and the qubits acted upon (a.k.a. the "target" qubits).
        TODO: docstring

        Returns
        -------
        dict
        """

        # Construct a dict of information for each elementary error-gen basis element (the basis for error-gen space)
        elemgen_info = {}
        for k, (op_label, eglabel) in enumerate(self.errorgen_space_op_elem_labels):
            elemgen_info[k] = {
                'type': eglabel[0],
                'qubits': set([i for bel_lbl in eglabel[1:] for i, char in enumerate(bel_lbl) if char != 'I']),
                'op_label': op_label,
                'elemgen_label': eglabel,
            }

        bins = {}
        dependent_indices = set(self.dependent_dir_indices)  # indices of one set of linearly dep. fogi dirs
        for i, meta in enumerate(self.fogi_metadata):
            fogi_dir = self.fogi_directions[:, i].asarray()
            label = meta['name']
            label_raw = meta['raw']
            label_abbrev = meta['abbrev']
            gauge_dir = meta['gaugespace_dir']
            r_factor = meta['r']

            present_elgen_indices = _np.where(_np.abs(fogi_dir) > tol)[0]

            #Aggregate elemgen_info data for all elemgens that contribute to this FOGI qty (as determined by `tol`)
            ops_involved = set(); qubits_acted_upon = set(); types = set(); # basismx = None
            for k in present_elgen_indices:
                k_info = elemgen_info[k]
                ops_involved.add(k_info['op_label'])
                qubits_acted_upon.update(k_info['qubits'])
                types.add(k_info['type'])

            #Create the "info" dictionary for this FOGI quantity
            info = {'op_set': ops_involved,
                    'types': types,
                    'qubits': qubits_acted_upon,
                    'fogi_index': i,
                    'label': label,
                    'label_raw': label_raw,
                    'label_abbrev': label_abbrev,
                    'dependent': bool(i in dependent_indices),
                    'gauge_dir': gauge_dir,
                    'fogi_dir': fogi_dir,
                    'r_factor': r_factor
                    }
            ops_involved = tuple(sorted(ops_involved))
            types = tuple(sorted(types))
            qubits_acted_upon = tuple(sorted(qubits_acted_upon))
            if ops_involved not in bins: bins[ops_involved] = {}
            if types not in bins[ops_involved]: bins[ops_involved][types] = {}
            if qubits_acted_upon not in bins[ops_involved][types]: bins[ops_involved][types][qubits_acted_upon] = []
            bins[ops_involved][types][qubits_acted_upon].append(info)

        return bins

    @classmethod
    def merge_binned_fogi_infos(cls, binned_fogi_infos, index_offsets):
        """
        Merge together multiple FOGI-info dictionaries created by :method:`create_binned_fogi_infos`.

        Parameters
        ----------
        binned_fogi_infos : list
            A list of FOGI-info dictionaries.

        index_offsets : list
            A list of length `len(binned_fogi_infos)` that gives the offset
            into an assumed-to-exist corresponding vector of components for
            all the FOGI infos.

        Returns
        -------
        dict
            The merged dictionary
        """
        def _merge_into(dest, src, offset, nlevels_to_merge, store_index):
            if nlevels_to_merge == 0:  # special last-level case where src and dest are *lists*
                for info in src:
                    new_info = _copy.deepcopy(info)
                    new_info['fogi_index'] += offset
                    new_info['store_index'] = store_index
                    dest.append(new_info)
            else:
                for k, d in src.items():
                    if k not in dest: dest[k] = {} if (nlevels_to_merge > 1) else []  # last level = list
                    _merge_into(dest[k], d, offset, nlevels_to_merge - 1, store_index)

        bins = {}
        nLevels = 3  # ops_involved, types, qubits_acted_upon
        for i, (sub_bins, offset) in enumerate(zip(binned_fogi_infos, index_offsets)):
            _merge_into(bins, sub_bins, offset, nLevels, i)
        return bins
