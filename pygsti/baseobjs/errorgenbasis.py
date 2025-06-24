"""
Defines the ElementaryErrorgenBasis class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import itertools as _itertools

from pygsti.baseobjs import Basis as _Basis
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GlobalElementaryErrorgenLabel,\
LocalElementaryErrorgenLabel as _LocalElementaryErrorgenLabel
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.tools import optools as _ot
from pygsti.baseobjs.statespace import StateSpace as _StateSpace


class ElementaryErrorgenBasis(_NicelySerializable):
    """
    A basis for error-generator space defined by a set of elementary error generators.

    Elements are ordered (have definite indices) and labeled.
    Intersection and union can be performed as a set.
    """

    def label_indices(self, labels, ok_if_missing=False):
        """ 
        Return a list of indices into this basis's label list
        for the specifed list of `ElementaryErrorgenLabels`.

        Parameters
        ----------
        labels : list of `ElementaryErrorgenLabel`
            A list of elementary error generator labels to extract the
            indices of.
        
        ok_if_missing : bool
           If True, then returns `None` instead of an integer when the given label is not present
        """
        return [self.label_index(lbl, ok_if_missing) for lbl in labels]

    def __len__(self):
        """ 
        Number of elementary errorgen elements in this basis.
        """
        return len(self.labels)

#helper function for checking label types.
def _all_elements_same_type(lst):
    if not lst:  # Check if the list is empty
        return True  # An empty list can be considered to have all elements of the same type
    
    first_type = type(lst[0])  # Get the type of the first element
    for element in lst:
        if type(element) != first_type:
            return False
    return True

class ExplicitElementaryErrorgenBasis(ElementaryErrorgenBasis):
    """
    This basis object contains the information  necessary for building, 
    storing and accessing a set of explicitly represented basis elements for a user
    specified set of of elementary error generators.
    """

    def __init__(self, state_space, labels, basis_1q=None):
        """
        Instantiate a new explicit elementary error generator basis. 

        Parameters
        ----------
        state_space : `StateSpace`
            An object describing the struture of the entire state space upon which the elements
            of this error generator basis act.

        labels : list or tuple of `ElementaryErrorgenLabel`
            A list of elementary error generator labels for which basis elements will be
            constructed.

        basis1q : `Basis` or str, optional (default None)
            A `Basis` object, or str which can be cast to one
            corresponding to the single-qubit basis elements which
            comprise the basis element labels for the values of the
            `ElementaryErrorgenLabels` in `labels`.
        """
        super().__init__()
        labels = tuple(labels)

        #add an assertion that the labels are ElementaryErrorgenLabels and that all of the labels are the same type.
        msg = '`labels` should be either LocalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel objects.' 
        if labels:
            assert isinstance(labels[0], (_GlobalElementaryErrorgenLabel, _LocalElementaryErrorgenLabel)), msg
            assert _all_elements_same_type(labels), 'All of the elementary error generator labels should be of the same type.'

        self._labels = labels
        self._label_indices = {lbl: i for i, lbl in enumerate(self._labels)}
        
        if isinstance(basis_1q, _Basis):
            self._basis_1q = basis_1q
        elif isinstance(basis_1q, str):
            self._basis_1q = _Basis.cast(basis_1q, 4)
        else:
            self._basis_1q = _Basis.cast('PP', 4)

        self.state_space = state_space
        assert(self.state_space.is_entirely_qubits), "FOGI only works for models containing just qubits (so far)"
        sslbls = self.state_space.sole_tensor_product_block_labels  # all the model's state space labels
        self.sslbls = sslbls  # the "support" of this space - the qubit labels
        
        #Caching
        self._cached_matrices = None
        self._cached_dual_matrices = None
        self._cached_supports = None

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'state_space' : self.state_space._to_nice_serialization(),
                      'labels' : [label.__str__() for label in self.labels],
                      '_basis_1q' : self._basis_1q if isinstance(self._basis_1q, str) else self._basis_1q._to_nice_serialization()

        }
        )
        return state
    @classmethod
    def from_nice_serialization(cls, state):
        return cls(_StateSpace.from_nice_serialization(state['state_space']), [_GlobalElementaryErrorgenLabel.cast(label) for label in state['labels']], state['_basis_1q'] if isinstance(state['_basis_1q'], str) else _Basis.from_nice_serialization(state['_basis_1q']))
    @property
    def labels(self):
        return self._labels
    
    @property
    def elemgen_supports(self):
        """
        Returns a tuple of tuples, each corresponding to the support
        of the elementary error generators in this basis, returned in
        the same order as they appear in `labels`.
        """
        if self._cached_supports is None:
            if isinstance(self._labels[0], _GlobalElementaryErrorgenLabel):
                self._cached_supports = tuple([elemgen_label.sslbls for elemgen_label in self._labels])
            #Otherwise these are LocalElementaryErrorgenLabels
            else:
                #LocalElementaryErrorgenLabel doesn't have a sslbls attribute indicating
                #support like GlobalElementaryErrorgenLabel does, do index into the `sslbls`
                #attribute for this object.
                self._cached_supports = tuple([tuple([self.sslbls[i] for i in elemgen_label.support_indices()]) 
                                               for elemgen_label in self._labels])
        return self._cached_supports
    
    #TODO: The implementations of some of the following properties are the same as in
    #CompleteElementaryErrorgen, refactor some of this into the parent class.
    @property
    def elemgen_dual_matrices(self):
        """
        Returns a tuple of matrices, each corresponding to the 
        of the matrix representation of the dual elementary error generators 
        in this basis, returned in the same order as they appear in `labels`.
        """
        if self._cached_dual_matrices is None:
            elemgen_types = [elemgen_label.errorgen_type for elemgen_label in self._labels]
            elemgen_labels = [elemgen_label.basis_element_labels for elemgen_label in self._labels]
            self._cached_dual_matrices = tuple(_ot.bulk_create_elementary_errorgen_nqudit_dual(
                                            elemgen_types, elemgen_labels,
                                            self._basis_1q, normalize=False, sparse=False,
                                            tensorprod_basis=True))
        return self._cached_dual_matrices
    
    @property
    def elemgen_matrices(self):
        """
        Returns a tuple of matrices, each corresponding to the 
        of the matrix representation of the elementary error generators 
        in this basis, returned in the same order as they appear in `labels`.
        """
        if self._cached_matrices is None:
            elemgen_types = [elemgen_label.errorgen_type for elemgen_label in self._labels]
            elemgen_labels = [elemgen_label.basis_element_labels for elemgen_label in self._labels]
            self._cached_matrices = tuple(_ot.bulk_create_elementary_errorgen_nqudit(
                                            elemgen_types, elemgen_labels,
                                            self._basis_1q, normalize=False, sparse=False,
                                            tensorprod_basis=True))
        return self._cached_matrices

    @property
    def elemgen_supports_and_dual_matrices(self):
        """
        Returns a tuple of tuples, each containing a tuple of support and a dual matrix representation
        each corresponding to an elementary error generator in this basis, returned in the same 
        order as they appear in `labels`.
        """
        return  tuple(zip(self.elemgen_supports, self.elemgen_dual_matrices))

    @property
    def elemgen_supports_and_matrices(self):
        """
        Returns a tuple of tuples, each containing a tuple of support and a matrix representation
        each corresponding to an elementary error generator in this basis, returned in the same 
        order as they appear in `labels`.
        """
        return  tuple(zip(self.elemgen_supports, self.elemgen_matrices))

    def label_index(self, label, ok_if_missing=False):
        """
        Return the index of the specified elementary error generator label
        in this basis' `labels` list.
        
        Parameters
        ----------
        label : `ElementaryErrorgenLabel`
            Elementary error generator label to return index for.

        ok_if_missing : bool
           If True, then returns `None` instead of an integer when the given label is not present.
        """
        if ok_if_missing and label not in self._label_indices:
            return None
        return self._label_indices[label]

    def create_subbasis(self, sslbl_overlap):
        """
        Create a sub-basis of this basis by including only the elements
        that overlap the given support (state space labels)

        Parameters
        ----------
        sslbl_overlap : list of sslbls
            A list of state space labels corresponding to qudits the support of
            an error generator must overlap with (i.e. the support must include at least
            one of these qudits) in order to be included in this subbasis.

        """
        #need different logic for LocalElementaryErrorgenLabels
        if isinstance(self.labels[0], _GlobalElementaryErrorgenLabel):
            sub_sslbls = set(sslbl_overlap)
            def overlaps(sslbls):
                ret = len(set(sslbls).intersection(sslbl_overlap)) > 0
                if ret: sub_sslbls.update(sslbls)  # keep track of all overlaps
                return ret

            sub_labels, sub_indices = zip(*[(lbl, i) for i, lbl in enumerate(self._labels)
                                            if overlaps(lbl[0])])
            sub_sslbls = sorted(sub_sslbls)
            sub_state_space = self.state_space.create_subspace(sub_sslbls)
        else:
            sub_labels = []
            for lbl in self.labels:
                non_trivial_bel_indices = lbl.support_indices()
                for sslbl in sslbl_overlap:
                    if sslbl in non_trivial_bel_indices:
                        sub_labels.append(lbl)
                        break
            #since using local labels keep the full original state space (the labels won't have gotten any shorter).
            sub_state_space = self.state_space.copy()    

        return ExplicitElementaryErrorgenBasis(sub_state_space, sub_labels, self._basis_1q)

    def union(self, other_basis):
        """
        Create a new `ExplicitElementaryErrorgenBasis` corresponding to the union of
        this basis with another.

        Parameters
        ----------
        other_basis : `ElementaryErrorgenBasis`
            `ElementaryErrorgenBasis` to construct the union with.
        """
        #assert that these two bases have compatible label types.
        msg = 'Incompatible `ElementaryErrrogenLabel` types, the two `ElementaryErrorgenBasis` should have the same label type.'
        assert type(self._labels[0]) == type(other_basis.labels[0]), msg
        #Get the union of the two bases labels.
        union_labels = set(self._labels) | set(other_basis.labels)
        union_state_space = self.state_space.union(other_basis.state_space)
        return ExplicitElementaryErrorgenBasis(union_state_space, sorted(union_labels, key=lambda label: label.__str__()), self._basis_1q)

    def intersection(self, other_basis):
        """
        Create a new `ExplicitElementaryErrorgenBasis` corresponding to the intersection of
        this basis with another.

        Parameters
        ----------
        other_basis : `ElementaryErrorgenBasis`
            `ElementaryErrorgenBasis` to construct the intersection with.
        """

        intersection_labels = set(self._labels) & set(other_basis.labels)
        intersection_state_space = self.state_space.intersection(other_basis.state_space)
        return ExplicitElementaryErrorgenBasis(intersection_state_space, sorted(intersection_labels, key=lambda label: label.__str__()), self._basis_1q)

    def difference(self, other_basis):
        """
        Create a new `ExplicitElementaryErrorgenBasis` corresponding to the difference of
        this basis with another. (i.e. A basis consisting of the labels contained in this basis
        but not the other)

        Parameters
        ----------
        other_basis : `ElementaryErrorgenBasis`
            `ElementaryErrorgenBasis` to construct the difference with.
        """
        difference_labels = set(self._labels) - set(other_basis.labels)
        #TODO: Making the state space equal to the true difference breaks some stuff in the FOGI code
        #that relied on the old (kind of incorrect behavior). Revert back to old version temporarily.
        #difference_state_space = self.state_space.difference(other_basis.state_space)
        difference_state_space = self.state_space
        return ExplicitElementaryErrorgenBasis(difference_state_space, sorted(difference_labels, key=lambda label: label.__str__()), self._basis_1q)

class CompleteElementaryErrorgenBasis(ElementaryErrorgenBasis):
    """
    This basis object contains the information  necessary for building, 
    storing and accessing a set of explicitly represented basis elements 
    for a basis of elementary error generators spanned by the elementary
    error generators of given type(s) (e.g. "Hamiltonian" and/or "other").
    """

    @classmethod
    def _create_diag_labels_for_support(cls, support, type_str, nontrivial_bels):
        assert(type_str in ('H', 'S'))  # the types of "diagonal" generators
        weight = len(support)

        def _basis_el_strs(possible_bels, wt):
            for els in _itertools.product(*([possible_bels] * wt)):
                yield ''.join(els)

        return [_GlobalElementaryErrorgenLabel(type_str, (bel,), support)
                for bel in _basis_el_strs(nontrivial_bels, weight)]

    @classmethod
    def _create_uptriangle_labels_for_support(cls, support, left_support, type_str, trivial_bel, nontrivial_bels):
        ret = []
        n = len(support)  # == weight
        all_bels = trivial_bel + nontrivial_bels
        left_weight = len(left_support)
        left_factors = [nontrivial_bels if x in left_support else trivial_bel for x in support]
        left_indices = [range(len(factors)) for factors in left_factors]
        right_factors = [all_bels if x in left_support else nontrivial_bels for x in support]
        right_lengths = [len(factors) for factors in right_factors]
        placevals = _np.cumprod(list(reversed(right_lengths[1:] + [1])))[::-1]
        ifirst_trivial = min([i for i in range(n) if support[i] not in left_support] + [n])  # [n] prevents empty list

        for left_inds in _itertools.product(*left_indices):  # better itertools call here TODO
            left_bel = ''.join([factors[i] for i, factors in zip(left_inds, left_factors)])
            right_offsets = [(i + 1 if ii < ifirst_trivial else 0) for ii, i in enumerate(left_inds)]
            if left_weight == n:  # n1 == n, so left_support == entire support (== no I's on left side)
                right_offsets[-1] += 1  # advance past diagonal element
            right_base_it = _itertools.product(*right_factors)
            start_at = _np.dot(right_offsets, placevals)
            right_it = _itertools.islice(right_base_it, int(start_at), None)
            # Above: int(.) needed for python 3.6, to convert from np.int64 -> int
            for right_beltup in right_it:
                ret.append(_GlobalElementaryErrorgenLabel(type_str, (left_bel, ''.join(right_beltup)), support))
        return ret

    @classmethod
    def _count_uptriangle_labels_for_support(cls, support, left_support, type_str, trivial_bel, nontrivial_bels):
        cnt = 0
        n = len(support)  # == weight
        n1 = len(left_support)  # == left_weight
        all_bels = trivial_bel + nontrivial_bels
        n1Q_nontrivial_bels = len(nontrivial_bels)
        n1Q_bels = len(all_bels)
        left_indices = [range(n1Q_nontrivial_bels if x in left_support else 1) for x in support]
        right_lengths = [(n1Q_bels if x in left_support else n1Q_nontrivial_bels) for x in support]
        placevals = _np.cumprod(list(reversed(right_lengths[1:] + [1])))[::-1]
        ifirst_trivial = min([i for i in range(n) if support[i] not in left_support] + [n])  # [n] prevents empty list

        for left_inds in _itertools.product(*left_indices):  # better itertools call here TODO
            # offsets for leading elements of right side corresponding to non-trivial (left_support) elements on right
            # side.  "+1" in "i+1" because right side indexes *all* bels whereas left index is into nontrivial bels.
            right_offsets = [(i + 1 if ii < ifirst_trivial else 0) for ii, i in enumerate(left_inds)]
            if n1 == n: right_offsets[-1] += 1  # advance past diagonal element
            start_at = _np.dot(right_offsets, placevals)
            cnt += _np.prod(right_lengths) - start_at

        return cnt


    @classmethod
    def _create_ordered_labels(cls, type_str, basis_1q, state_space,
                               max_weight=None, sslbl_overlap=None,
                               include_offsets=False, initial_offset=0):
        offsets = {'BEGIN': initial_offset}
        labels = []
        trivial_bel = [basis_1q.labels[0]]
        nontrivial_bels = basis_1q.labels[1:]  # assume first element is identity

        if sslbl_overlap is not None and not isinstance(sslbl_overlap, set):
            sslbl_overlap = set(sslbl_overlap)

        assert(state_space.is_entirely_qubits), "FOGI only works for models containing just qubits (so far)"
        sslbls = state_space.sole_tensor_product_block_labels  # all the model's state space labels
        if max_weight is None:
            max_weight = len(sslbls)

        # Let k be len(nontrivial_bels)
        if type_str in ('H', 'S'):
            # --> for each set of n qubit labels, there are k^n Hamiltonian terms with weight n
            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):  # NOTE: combinations *MUST* be deterministic
                    if (sslbl_overlap is not None
                       and len(sslbl_overlap.intersection(support)) == 0):
                        continue
                    offsets[support] = len(labels) + initial_offset
                    labels.extend(cls._create_diag_labels_for_support(support, type_str, nontrivial_bels))

        elif type_str in ('C', 'A'):
            # --> for each weight n, must compute all non-diagonal and upper-triangle *pairs* that have this weight.
            #  This is done via algorithm:
            #    Given n qubit labels (the support, len(support) == weight),
            #    loop over all left-hand weights n1 ranging from 1 to n
            #      loop over all left-supports of size n1 (choose some number of left factors to be nontrivial)
            #        Note: right-side *must* be nontrivial on complement of left support, and can be anything
            #              on factors in the left support (since the left side is nontrivial here) *except*
            #              the right side can't be all the trivial element.
            #        loop over all left-side elements (nNontrivialBELs^n1 of them)
            #          loop over right factors - the number of elements will be complicated...
            #                      (see _create_ordered_label_offsets)
            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):
                    if (sslbl_overlap is not None
                       and len(sslbl_overlap.intersection(support)) == 0):
                        continue

                    for left_weight in range(1, weight + 1):
                        for left_support in _itertools.combinations(support, left_weight):
                            offsets[(support, left_support)] = len(labels) + initial_offset
                            labels.extend(cls._create_uptriangle_labels_for_support(support, left_support, type_str,
                                                                                    trivial_bel, nontrivial_bels))
        else:
            raise ValueError("Invalid elementary type: %s" % str(type_str))
        offsets['END'] = len(labels) + initial_offset

        return (labels, offsets) if include_offsets else labels

    @classmethod
    def _create_ordered_label_offsets(cls, type_str, basis_1q, state_space,
                                      max_weight=None, sslbl_overlap=None,
                                      return_total_support=False, initial_offset=0):
        """ same as _create_ordered_labels but doesn't actually create the labels - just counts them to get offsets. """
        offsets = {'BEGIN': initial_offset}
        off = 0  # current number of labels that we would have created
        trivial_bel = [basis_1q.labels[0]]
        nontrivial_bels = basis_1q.labels[1:]  # assume first element is identity
        n1Q_bels = len(basis_1q.labels)
        n1Q_nontrivial_bels = n1Q_bels - 1  # assume first element is identity
        total_support = set()

        if sslbl_overlap is not None and not isinstance(sslbl_overlap, set):
            sslbl_overlap = set(sslbl_overlap)

        assert(state_space.is_entirely_qubits), "FOGI only works for models containing just qubits (so far)"
        sslbls = state_space.sole_tensor_product_block_labels  # all the model's state space labels
        if max_weight is None:
            max_weight = len(sslbls)

        # Let k be len(nontrivial_bels)
        if type_str in ('H', 'S'):
            # --> for each set of n qubit labels, there are k^n Hamiltonian terms with weight n
            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):  # NOTE: combinations *MUST* be deterministic
                    if (sslbl_overlap is not None
                       and len(sslbl_overlap.intersection(support)) == 0):
                        continue
                    offsets[support] = off + initial_offset
                    off += n1Q_nontrivial_bels**weight
                    total_support.update(support)

        elif type_str in ('C', 'A'):
            for weight in range(1, max_weight + 1):
                for support in _itertools.combinations(sslbls, weight):
                    if (sslbl_overlap is not None
                       and len(sslbl_overlap.intersection(support)) == 0):
                        continue

                    total_support.update(support)
                    for left_weight in range(1, weight + 1):
                        for left_support in _itertools.combinations(support, left_weight):
                            offsets[(support, left_support)] = off + initial_offset
                            off += cls._count_uptriangle_labels_for_support(support, left_support, type_str,
                                                                            trivial_bel, nontrivial_bels)
        else:
            raise ValueError("Invalid elementary type: %s" % str(type_str))
        offsets['END'] = off + initial_offset

        return (offsets, total_support) if return_total_support else offsets

    def __init__(self, basis_1q, state_space, elementary_errorgen_types=('H', 'S', 'C', 'A'),
                 max_weights=None, sslbl_overlap=None, default_label_type='global'):
        """
        Parameters
        ----------
        basis_1q : `Basis` or str
            A `Basis` object, or str which can be cast to one
            corresponding to the single-qubit basis elements which
            comprise the basis element labels for the values of the
            `ElementaryErrorgenLabels` in `labels`.

        state_space : `StateSpace`
            An object describing the struture of the entire state space upon which the elements
            of this error generator basis act.

        elementary_errorgen_types : tuple of str, optional (default ('H', 'S', 'C', 'A'))
            Tuple of strings designating elementary error generator types to include in this
            basis.

        max_weights : dict, optional (default None)
            A dictionary containing the maximum weight for each of the different error generator
            types to include in the constructed basis. If None then 
            there is no maximum weight. If specified, any error generator
            types without entries will have no maximum weight associated
            with them.

        sslbl_overlap : list of sslbls, optional (default None)
            A list of state space labels corresponding to qudits the support of
            an error generator must overlap with (i.e. the support must include at least
            one of these qudits) in order to be included in this basis.

        default_label_type : str, optional (default 'global')
            String specifying the type of error generator label to use by default.
            i.e. the type of label returned by `labels`. This also impacts the
            construction of the error generator matrices.
            Supported options are 'global' or 'local', which correspond to 
            `GlobalElementaryErrorgenLabel` and `LocalElementaryErrorgenLabel`,
            respectively.
        """

        if isinstance(basis_1q, _Basis):
            self._basis_1q = basis_1q
        elif isinstance(basis_1q, str):
            self._basis_1q = _Basis.cast(basis_1q, 4)
        else:
            self._basis_1q = _Basis.cast('pp', 4)

        self._elementary_errorgen_types = tuple(elementary_errorgen_types)  # so works for strings like "HSCA"
        self.state_space = state_space
        self.max_weights = max_weights if max_weights is not None else dict()
        self._sslbl_overlap = sslbl_overlap
        self._default_lbl_typ = default_label_type

        assert(self.state_space.is_entirely_qubits), "FOGI only works for models containing just qubits (so far)"
        assert(all([eetyp in ('H', 'S', 'C', 'A') for eetyp in elementary_errorgen_types])), \
            "Invalid elementary errorgen type in %s" % str(elementary_errorgen_types)

        self._offsets = dict()
        present_sslbls = set()
        istart = 0

        for eetyp in elementary_errorgen_types:
            self._offsets[eetyp], sup = self._create_ordered_label_offsets(
                eetyp, self._basis_1q, self.state_space,
                self.max_weights.get(eetyp, None),
                self._sslbl_overlap, return_total_support=True, initial_offset=istart)
            present_sslbls = present_sslbls.union(sup)  # set union
            istart = self._offsets[eetyp]['END']

        #Note: state space can have additional labels that aren't in support
        # (this is, I think, only true when sslbl_overlap != None)
        sslbls = self.state_space.sole_tensor_product_block_labels  # all the model's state space labels

        if set(sslbls) == present_sslbls:
            self.sslbls = sslbls  # the "support" of this space - the qubit labels
        elif present_sslbls.issubset(sslbls):
            self.state_space = self.state_space.create_subspace(present_sslbls)
            self.sslbls = present_sslbls
        else:
            # this should never happen - somehow the statespace doesn't have all the labels!
            assert(False), "Logic error! State space doesn't contain all of the present labels!!"

        self._cached_global_labels = None
        self._cached_local_labels = None
        self._cached_matrices = None
        self._cached_dual_matrices = None
        self._cached_supports = None

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
        #                      (IIX,XXX)  #   loop over filling in the (at least 1) nontrivial left-index with trival
        #                      (IXI,XIX)  #                                                     & nontrivial on right
        #                      (IXI,XXX)
        #                      ...
        #                      (IXX,XII) # move to weight-2s on left
        #                      (IXX,XXI) #   on right, loop over all possible choices of at least one, an at most m,
        #                      (IXX,XXX) #    nontrivial indices to place within the m nontriv left indices (1 & 2 here)

    def __len__(self):
        """ Number of elementary errorgen elements in this basis """
        return self._offsets[self._elementary_errorgen_types[-1]]['END']

    def to_explicit_basis(self):
        """
        Creates a new `ExplicitElementaryErrorgenBasis` based on this Basis' elements.
        """
        return ExplicitElementaryErrorgenBasis(self.state_space, self.labels, self._basis_1q)

    #TODO: Why can't this be done at initialization time?
    @property
    def labels(self):
        """
        Tuple of either `GlobalElementaryErrorgenLabel` or `LocalElementaryErrorgenLabel` objects
        for this basis, with which one determined by the `default_label_type` specified on basis
        construction.

        For specific label types see the `global_labels` and `local_labels` methods.
        """

        if self._default_lbl_typ == 'global':
            return self.global_labels()
        else:
            return self.local_labels()
    
    def global_labels(self):
        """
        Return a list of labels for this basis as `GlobalElementaryErrorgenLabel`
        objects.
        """
        if self._cached_global_labels is None:
            labels = []
            for eetyp in self._elementary_errorgen_types:
                labels.extend(self._create_ordered_labels(eetyp, self._basis_1q, self.state_space,
                                                          self.max_weights.get(eetyp, None),
                                                          self._sslbl_overlap))
            
            self._cached_global_labels = tuple(labels)
        return self._cached_global_labels
    
    def local_labels(self):
        """
        Return a list of labels for this basis as `LocalElementaryErrorgenLabel`
        objects.
        """
        if self._cached_local_labels is None:
            if self._cached_global_labels is None:
                self._cached_global_labels = self.global_labels()
            self._cached_local_labels = tuple([_LocalElementaryErrorgenLabel.cast(lbl, sslbls=self.sslbls) for lbl in self._cached_global_labels])
        return self._cached_local_labels
    
    def sublabels(self, errorgen_type):
        """
        Return a tuple of labels within this basis for the specified error generator
        type (may be empty).

        Parameters
        ----------
        errorgen_type : 'H', 'S', 'C' or 'A'
            String specifying the error generator type to return the labels for.
        
        Returns
        -------
        tuple of either `GlobalElementaryErrorgenLabels` or `LocalElementaryErrorgenLabels`
        """
        #TODO: It should be possible to do this much faster than regenerating these from scratch.
        #Perhaps by caching the error generators by type at construction time.
        labels = self._create_ordered_labels(errorgen_type, self._basis_1q, self.state_space,
                                           self.max_weights.get(errorgen_type, None),
                                           self._sslbl_overlap)
        if self._default_lbl_typ == 'local':
            labels = tuple([_LocalElementaryErrorgenLabel.cast(lbl, sslbls=self.sslbls) for lbl in labels])
        return labels
    
    @property
    def elemgen_supports(self):
        """
        Returns a tuple of tuples, each corresponding to the support
        of the elementary error generators in this basis, returned in
        the same order as they appear in `labels`.
        """
        if self._cached_supports is None:
            self._cached_supports = tuple([elemgen_label.sslbls for elemgen_label in self.global_labels()])
        return self._cached_supports
    
    @property
    def elemgen_dual_matrices(self):
        """
        Returns a tuple of matrices, each corresponding to the 
        of the matrix representation of the dual elementary error generators 
        in this basis, returned in the same order as they appear in `labels`.
        """
        if self._cached_dual_matrices is None:
            elemgen_types = [elemgen_label.errorgen_type for elemgen_label in self.labels]
            elemgen_labels = [elemgen_label.basis_element_labels for elemgen_label in self.labels]
            self._cached_dual_matrices = tuple(_ot.bulk_create_elementary_errorgen_nqudit_dual(
                                            elemgen_types, elemgen_labels,
                                            self._basis_1q, normalize=False, sparse=False,
                                            tensorprod_basis=True))
        return self._cached_dual_matrices
    
    @property
    def elemgen_matrices(self):
        """
        Returns a tuple of matrices, each corresponding to the 
        of the matrix representation of the elementary error generators 
        in this basis, returned in the same order as they appear in `labels`.
        """
        if self._cached_matrices is None:
            elemgen_types = [elemgen_label.errorgen_type for elemgen_label in self.labels]
            elemgen_labels = [elemgen_label.basis_element_labels for elemgen_label in self.labels]
            self._cached_matrices = tuple(_ot.bulk_create_elementary_errorgen_nqudit(
                                            elemgen_types, elemgen_labels,
                                            self._basis_1q, normalize=False, sparse=False,
                                            tensorprod_basis=True))
        return self._cached_matrices

    @property
    def elemgen_supports_and_dual_matrices(self):
        """
        Returns a tuple of tuples, each containing a tuple of support and a dual matrix representation
        each corresponding to an elementary error generator in this basis, returned in the same 
        order as they appear in `labels`.
        """
        return  tuple(zip(self.elemgen_supports, self.elemgen_dual_matrices))

    @property
    def elemgen_supports_and_matrices(self):
        """
        Returns a tuple of tuples, each containing a tuple of support and a matrix representation
        each corresponding to an elementary error generator in this basis, returned in the same 
        order as they appear in `labels`.
        """
        return  tuple(zip(self.elemgen_supports, self.elemgen_matrices))

    def label_index(self, label, ok_if_missing=False, identity_label='I'):
        """
        Return the index of the specified elementary error generator label
        in this basis' `labels` list.
        
        Parameters
        ----------
        label : `ElementaryErrorgenLabel`
            Elementary error generator label to return index for.
        
        ok_if_missing : bool
           If True, then returns `None` instead of an integer when the given label is not present.
        
        identity_label : str, optional (default 'I')
            An optional string specifying the label used to denote the identity in basis element labels.
        """

        if isinstance(label, _LocalElementaryErrorgenLabel):
            label = _GlobalElementaryErrorgenLabel.cast(label, self.sslbls, identity_label=identity_label)

        support = label.sslbls
        eetype = label.errorgen_type
        bels = label.basis_element_labels
        trivial_bel = self._basis_1q.labels[0]  # assumes first element is identity
        nontrivial_bels = self._basis_1q.labels[1:]

        if ok_if_missing and eetype not in self._offsets:
            return None

        if eetype in ('H', 'S'):
            if ok_if_missing and support not in self._offsets[eetype]:
                return None
            base = self._offsets[eetype][support]
            indices = {lbl: i for i, lbl in enumerate(self._create_diag_labels_for_support(support, eetype,
                                                                                           nontrivial_bels))}
        elif eetype in ('C', 'A'):
            assert(len(trivial_bel) == 1)  # assumes this is a single character
            nontrivial_inds = [i for i, letter in enumerate(bels[0]) if letter != trivial_bel]
            left_support = tuple([self.sslbls[i] for i in nontrivial_inds])

            if ok_if_missing and (support, left_support) not in self._offsets[eetype]:
                return None
            base = self._offsets[eetype][(support, left_support)]

            indices = {lbl: i for i, lbl in enumerate(self._create_uptriangle_labels_for_support(
                support, left_support, eetype, [trivial_bel], nontrivial_bels))}
        else:
            raise ValueError("Invalid elementary errorgen type: %s" % str(eetype))

        return base + indices[label]
        
    def create_subbasis(self, sslbl_overlap, retain_max_weights=True):
        """
        Create a sub-basis of this basis by including only the elements
        that overlap the given support (state space labels)
        """
        #Note: state_space is automatically reduced within __init__ when necessary, e.g., when
        # `sslbl_overlap` is non-None and considerably reduces the basis.
        return CompleteElementaryErrorgenBasis(self._basis_1q, self.state_space, self._elementary_errorgen_types,
                                               self.max_weights if retain_max_weights else None,
                                               sslbl_overlap)

    def union(self, other_basis):
        """
        Create a new `ExplicitElementaryErrorgenBasis` corresponding to the union of
        this basis with another.

        Parameters
        ----------
        other_basis : `ElementaryErrorgenBasis`
            `ElementaryErrorgenBasis` to construct the union with.
        """
        # don't convert this basis to an explicit one unless it's necessary -
        # if `other_basis` is already an explicit basis then let it do the work.
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            return other_basis.union(self)
        else:
            return self.to_explicit_basis().union(other_basis)

    def intersection(self, other_basis):
        """
        Create a new `ExplicitElementaryErrorgenBasis` corresponding to the intersection of
        this basis with another.

        Parameters
        ----------
        other_basis : `ElementaryErrorgenBasis`
            `ElementaryErrorgenBasis` to construct the intersection with.
        """
        if isinstance(other_basis, ExplicitElementaryErrorgenBasis):
            return other_basis.intersection(self)
        else:
            return self.to_explicit_basis().intersection(other_basis)

    def difference(self, other_basis):
        """
        Create a new `ExplicitElementaryErrorgenBasis` corresponding to the difference of
        this basis with another. (i.e. A basis consisting of the labels contained in this basis
        but not the other)

        Parameters
        ----------
        other_basis : `ElementaryErrorgenBasis`
            `ElementaryErrorgenBasis` to construct the difference with.
        """
        return self.to_explicit_basis().difference(other_basis)