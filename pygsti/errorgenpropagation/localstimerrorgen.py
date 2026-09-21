#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from typing import Any, Iterable, Optional, Sequence, Union
from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel as _ElementaryErrorgenLabel, GlobalElementaryErrorgenLabel as _GEEL,\
LocalElementaryErrorgenLabel as _LEEL
try:
    import stim
except ImportError:
    pass


def bel_str(pauli: stim.PauliString) -> str:
    """
    The plain-string form of a basis element label (BEL) given as a `stim.PauliString`, e.g.
    `stim.PauliString('+_XY')` -> `'IXY'`.

    This is the representation `LocalElementaryErrorgenLabel` uses for its basis element
    labels, and the one `LocalStimErrorgenLabel` hashes on and the canonical C/A ordering
    (`bel_less_than`) sorts on. stim renders identities as '_' (ASCII 95, which would sort
    *after* 'X', 'Y' and 'Z'), so identities are rewritten to 'I' ('I' < 'X' < 'Y' < 'Z').

    The leading sign character is dropped, so signs of +1 and -1 are ignored. Imaginary
    phases (+i/-i) are not supported: they never occur for basis element labels, which are
    Hermitian Paulis.
    """
    return str(pauli)[1:].replace('_', 'I')


def bel_less_than(pauli1: stim.PauliString, pauli2: stim.PauliString) -> bool:
    """
    Returns True if `pauli1` sorts strictly before `pauli2` in the canonical basis element
    label ordering, i.e. lexicographically on the strings `bel_str` produces. This is the
    order in which the two basis element labels of 'C' and 'A' type error generator labels
    are stored everywhere in pyGSTi.

    Both inputs are assumed to be Hermitian (sign +1 or -1; the sign is ignored) and of the
    same length. When the strings are already available, compare them directly with `<`
    instead: it is two orders of magnitude cheaper than rendering them again.
    """
    return bel_str(pauli1) < bel_str(pauli2)


# Fixed numbering of the error generator sectors. `LocalStimErrorgenLabel.type_idx` is the
# index of a label's `errorgen_type` in this table; `pygsti.tools.errgenproptools` uses
# `4*type_idx_1 + type_idx_2` to index its per-type-pair dispatch tables for the error
# generator commutator and composition, so the order here and there must agree.
_ERRORGEN_TYPE_INDICES: dict[str, int] = {'H': 0, 'S': 1, 'C': 2, 'A': 3}


def _slow_support_mask(bel_strings: tuple[str, ...]) -> int:
    """
    Pure-python construction of the support bitmask of an error generator label from its
    'I'-padded basis element label strings: bit q is set iff some string is not 'I' at
    position q. Scanning with `str.find` per Pauli letter is several times faster than a
    per-character python loop and, for the low-weight labels that dominate in practice,
    also faster than collecting `stim.PauliString.pauli_indices()`.
    """
    mask = 0
    for s in bel_strings:
        for letter in 'XYZ':
            i = s.find(letter)
            while i != -1:
                mask |= 1 << i
                i = s.find(letter, i + 1)
    return mask


# Use the cython implementation of the mask construction when the extensions are built
# (2-8x faster than the pure-python version); fall back to the latter otherwise.
try:
    from pygsti.tools.fasterrgencalc import fast_support_mask as support_mask_from_strings
except ImportError:
    support_mask_from_strings = _slow_support_mask


#TODO: Split this into a parent class and subclass for markovian and non-markovian
#propagation. There is some overhead in instantiating the NM version of these labels
#which we can avoid and make markovian applications much more efficient (label instantiation
#is like a third of runtime when using higher-order BCH, e.g.)
class LocalStimErrorgenLabel(_ElementaryErrorgenLabel):

    """
    `LocalStimErrorgenLabel` is a specialized `ElementaryErrorgenLabel`
    designed to manage the propagation of error generator using Stim primitives for fast Pauli and
    Clifford operations, storing propagation related metadata, and storing metadata relevant to the
    evaluation of non-Markovian error propagators using cumulant expansion based techniques.

    Besides the `errorgen_type` string, each label carries the integer `type_idx`
    (`_ERRORGEN_TYPE_INDICES[errorgen_type]`, i.e. H=0, S=1, C=2, A=3) that the error generator
    commutator and composition routines in `pygsti.tools.errgenproptools` use to index their
    per-type-pair dispatch tables.
    """

    @classmethod
    def cast(cls, obj: Union[_ElementaryErrorgenLabel, tuple, list],
             sslbls: Optional[Sequence[int, str]] = None) -> LocalStimErrorgenLabel:
        """
        Method for casting objects to instances of LocalStimErrorgenLabel.

        Parameters
        ----------
        obj : `LocalStimErrorgenLabel`, ``LocalElementaryErrorgenLabel`, `GlobalElementaryErrorgenLabel`, tuple or list

        sslbls : tuple or list, optional (default None)
            A complete set of state space labels. Used when casting from a GlobalElementaryErrorgenLabel
            or from a tuple of length 3 (wherein the final element is interpreted as the set of ssblbs the error
            generator acts upon).

        Returns
        -------
        `LocalStimErrorgenLabel`
        """
        if isinstance(obj, LocalStimErrorgenLabel):
            return obj
        
        if isinstance(obj, _GEEL):
            #convert to a tuple representation
            assert sslbls is not None, 'Must specify sslbls when casting from `GlobalElementaryErrorgenLabel`.'
            obj = (obj.errorgen_type, obj.basis_element_labels, obj.sslbls)
            initial_label=None
        
        if isinstance(obj, _LEEL):
            #convert to a tuple representation
            initial_label = obj
            obj = (obj.errorgen_type, obj.basis_element_labels)
        
        if isinstance(obj, (tuple, list)):
            #In this case assert that the first element of the tuple is a string corresponding to the
            #error generator type.
            errorgen_type = obj[0]
            initial_label = None

            #two elements for a local label and three for a global one
            #second element should have the basis element labels
            assert len(obj)==2 or len(obj)==3 and isinstance(obj[1], (tuple, list)) 
            
            #if a global label tuple the third element should be a tuple or list.
            if len(obj)==3:
                assert isinstance(obj[2], (tuple, list))
                assert sslbls is not None, 'Must specify sslbls when casting from a tuple or list of length 3. See docstring.'
                #convert to local-style bels.
                indices_to_replace = [sslbls.index(sslbl) for sslbl in obj[2]]
                local_bels = []
                for global_lbl in obj[1]:
                    #start by figure out which initialization to use, either stim
                    #or a string.
                    local_bel = stim.PauliString('I'*len(sslbls))
                    for kk, k in enumerate(indices_to_replace):
                        local_bel[k] = global_lbl[kk]
                    local_bels.append(local_bel)
            else:
                local_bels = obj[1]

        #now build the LocalStimErrorgenLabel
        stim_bels = []
        for bel in local_bels:
            if isinstance(bel, str):
                stim_bels.append(stim.PauliString(bel))
            elif isinstance(bel, stim.PauliString):
                stim_bels.append(bel)
            else:
                raise ValueError('Only str and `stim.PauliString` basis element labels are supported presently.')
            
        return cls(errorgen_type, stim_bels, initial_label=initial_label)


    def __init__(self, errorgen_type: str, basis_element_labels: Iterable[stim.PauliString],
                 circuit_time: Optional[float] = None, initial_label: Optional[_ElementaryErrorgenLabel] = None,
                 label: Optional[str] = None, pauli_str_reps: Optional[tuple[str, ...]] = None) -> None:
        """
        Create a new instance of  `LocalStimErrorgenLabel`

        Parameters
        ----------
        errorgen_type : str
            A string corresponding to the error generator sector this error generator label is
            an element of. Allowed values are 'H', 'S', 'C' and 'A'.

        basis_element_labels : tuple or list
            A list or tuple of stim.PauliString labeling basis elements used to label this error generator.
            This is either length-1 for 'H' and 'S' type error generators, or length-2 for 'C' and 'A'
            type.

        circuit_time : float, optional (default None)
            An optional value which associates this error generator with a particular circuit time at
            which it arose. This is primarily utilized in the context of non-Markovian simulations and
            estimation where an error generator may notionally be associated with a stochastic process.

        initial_label : `ElementaryErrorgenLabel`, optional (default None)
            If not None, then this `ElementaryErrorgenLabel` is stored within this label and is interpreted
            as being the 'initial' value of this error generator, prior to any propagation or transformation
            during the course of its use. If None, then a `LocalElementaryErrorgenLabel` matching the
            `errorgen_type` and `basis_element_labels` of this label is used; it is constructed lazily,
            on the first access of the `initial_label` property, since the vast majority of labels
            (e.g. the intermediate terms produced by the commutator and composition routines in
            `pygsti.tools.errgenproptools`) never have it read.

        label : str, optional (default None)
            An optional label string which is included when printing the string representation of this
            label.

        pauli_str_reps : tuple of str, optional (default None)
            Optional tuple of python strings corresponding to the stim.PauliStrings in basis_element_labels.
            When specified can speed up construction of hashable label representations.
        """
        self.errorgen_type = errorgen_type
        try:
            self.type_idx = _ERRORGEN_TYPE_INDICES[errorgen_type]
        except KeyError:
            raise ValueError(f"Unknown error generator type {errorgen_type}; expected one of 'H', 'S', 'C', 'A'.")
        self.basis_element_labels = tuple(basis_element_labels) 
        self.label = label
        self.circuit_time = circuit_time

        # Cached string forms: the tuple of 'I'-padded basis element label strings, and the single
        # string this label hashes and compares on (type letter followed by the concatenated
        # label strings, e.g. 'HXI', 'CXIYI'; the type letter is needed to keep H_P and S_P apart).
        if pauli_str_reps is not None:
            self._hashable_basis_element_labels = pauli_str_reps
        else:
            self._hashable_basis_element_labels = self.bel_to_strings()
        self._hashable_string_rep = errorgen_type + ''.join(self._hashable_basis_element_labels)

        #additionally store a copy of the value of the original error generator label which will remain unchanged
        #during the course of propagation for later bookkeeping purposes. (None means "this label itself",
        #materialized on first access; see the `initial_label` property.)
        self._initial_label = initial_label
    #TODO: Update various methods to account for additional metadata that has been added.

    # Cache slot for the `support_mask` property: the class-level default stands in for "not
    # built yet" so that constructing a label (which the commutator/composition routines do
    # in great numbers) does not pay for one more instance attribute; the instance attribute
    # is only created when the mask is first needed. Instances from older pickles (without
    # the attribute) are covered by the same default.
    _support_mask: Optional[int] = None

    @property
    def support_mask(self) -> int:
        """
        The support of this error generator as a bitmask: bit q is set iff at least one of
        the basis element labels acts non-trivially on qubit q. Two error generators whose
        masks have no common bit (`m1 & m2 == 0`) act on disjoint sets of qubits and hence
        commute exactly; the layerwise commutator accumulation in
        `pygsti.tools.errgenproptools` uses this to skip such pairs, which are the vast
        majority at large qubit counts. Built on first access and cached, since the same
        label is typically tested against many others.

        Note: Bit mask returned is in reverse order compared to basis element label strings.
        E.g. "XII" -> 001.
        """
        mask = self._support_mask
        if mask is None:
            mask = self._support_mask = support_mask_from_strings(self._hashable_basis_element_labels)
        return mask

    @property
    def initial_label(self) -> _ElementaryErrorgenLabel:
        """
        The `ElementaryErrorgenLabel` this label originated from, prior to any propagation or
        transformation. Defaults to a `LocalElementaryErrorgenLabel` equivalent to this label,
        constructed on first access.
        """
        if self._initial_label is None:
            self._initial_label = self.to_local_eel()
        return self._initial_label

    def __hash__(self) -> int:
        #return hash((self.errorgen_type, self._hashable_basis_element_labels))
        return hash(self._hashable_string_rep)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore from a pickled/copied state, migrating states written by older versions
        of this class:

        - `type_idx` did not exist: derive it from `errorgen_type`.
        - `initial_label` was a plain attribute (it is now the read-only property backed
          by `_initial_label`): move it, so the stored pre-propagation label is kept.
        - the cached `_hashable_basis_element_labels` did not exist: rebuild it; the hash/equality
          string `_hashable_string_rep` is always rebuilt since its format has changed.
        """
        if 'initial_label' in state:
            state['_initial_label'] = state.pop('initial_label')
        if 'type_idx' not in state:
            state['type_idx'] = _ERRORGEN_TYPE_INDICES[state['errorgen_type']]
        if '_hashable_basis_element_labels' not in state:
            state['_hashable_basis_element_labels'] = tuple([bel_str(ps) for ps in state['basis_element_labels']])
        # always rebuilt: the format of this string has changed between versions.
        state['_hashable_string_rep'] = state['errorgen_type'] + ''.join(state['_hashable_basis_element_labels'])
        self.__dict__.update(state)

    def bel_to_strings(self) -> tuple[str, ...]:
        """
        Convert the elements of `basis_element_labels` to python strings
        (from stim.PauliString(s)) and return as a tuple. 
        """       
        return tuple([bel_str(ps) for ps in self.basis_element_labels])


    def __eq__(self, other: object) -> bool:
        """
        Performs equality check by seeing if the two error gen labels have the same `errorgen_type`
        and `basis_element_labels` (compared through their cached string representations, which
        is an order of magnitude cheaper than comparing the `stim.PauliString`s).
        """
        return isinstance(other, LocalStimErrorgenLabel) and self._hashable_string_rep == other._hashable_string_rep
    
 
    def __str__(self):
        if self.label is None:
            return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ")"
        else:
            return self.errorgen_type + " " + str(self.label)+ " " + "(" \
                   + ",".join(map(str, self.basis_element_labels)) + ")"

    def __repr__(self):
        if self.label is None:
            if self.circuit_time is not None:
                return f'({self.errorgen_type}, {self.basis_element_labels}, time={self.circuit_time})'
            else:
                return f'({self.errorgen_type}, {self.basis_element_labels})'
        else:
            if self.circuit_time is not None:
                return f'({self.errorgen_type}, {self.label}, {self.basis_element_labels}, time={self.circuit_time})'
            else:
                return f'({self.errorgen_type}, {self.label}, {self.basis_element_labels})'
    
      
    #TODO: Rework this to not directly modify the weights, and only return the sign modifier.
    #      (Revisit after the error generator commutator/composition refactor, which touches all callers.)
    def propagate_error_gen_tableau(self, slayer: stim.Tableau, weight: float) -> tuple[LocalStimErrorgenLabel, float]:
        """
        Parameters
        ----------
        slayer : `stim.Tableau`
            `stim.Tableau` object corresponding to an ideal Clifford operations for 
            a circuit layer which we will be propagating this error generator through. 

        weight : float
            Current weight of this error generator.
        
        Returns
        -------
        tuple of consisting of an `LocalStimErrorgenLabel` and an updated error generator
        weight, which may have changed by a sign. For 'C' and 'A' type labels the returned
        label's basis element labels are in canonical (sorted) order.
        """
        new_basis_labels = []
        weightmod = 1.0
        # `slayer(pauli)` returns a new PauliString, so its sign can be cleared in place
        # (cheaper than multiplying by the sign, which allocates another full-width string).
        if self.errorgen_type == 'S':
            for pauli in self.basis_element_labels:
                temp = slayer(pauli)
                temp.sign = 1
                new_basis_labels.append(temp)
        else:
            for pauli in self.basis_element_labels:
                temp = slayer(pauli)
                weightmod = temp.sign.real*weightmod
                temp.sign = 1
                new_basis_labels.append(temp)

            # Conjugation by a Clifford need not preserve the relative order of the two
            # basis element labels of a C or A generator (e.g. SWAP maps (IX, XI) to (XI, IX)),
            # so re-canonicalize. C is symmetric, C_{P,Q} = C_{Q,P}, so a swap is free; A is
            # antisymmetric, A_{P,Q} = -A_{Q,P}, so a swap must also flip the sign of the weight.
            if self.errorgen_type in ('C', 'A') and not bel_less_than(new_basis_labels[0], new_basis_labels[1]):
                new_basis_labels.reverse()
                if self.errorgen_type == 'A':
                    weightmod = -weightmod

        # Note: `self.initial_label` (the property, not `_initial_label`) is deliberately used
        # here. It materializes the pre-propagation label if it has not been already, which is
        # required: a `None` passed on would make the new label lazily build its initial label
        # from the *post*-propagation basis element labels.
        return (LocalStimErrorgenLabel(self.errorgen_type, new_basis_labels, initial_label=self.initial_label, circuit_time=self.circuit_time), 
                weightmod*weight)
    
    def to_global_eel(self, sslbls = None):
        """
        Returns a `GlobalElementaryErrorgenLabel` equivalent to this `LocalStimErrorgenLabel`.

        sslbls : list (optional, default None)
            A list of state space labels corresponding to the qubits corresponding to each
            of the paulis in the local basis element label. If None this defaults a list of integers
            ranging from 0 to N where N is the number of paulis in the basis element labels.
        """

        #first get the pauli strings corresponding to the stim.PauliString object(s) that are the
        #basis_element_labels.
        pauli_strings = self.bel_to_strings()
        if sslbls is None:
            sslbls = list(range(len(pauli_strings[0]))) #The two pauli strings should be the same length, so take the first.
        #GlobalElementaryErrorgenLabel should have built-in support for casting from a tuple of the error gen type
        #and the paulis for the basis element labels, so long as it is given appropriate sslbls to use.
        return _GEEL.cast((self.errorgen_type,) + pauli_strings, sslbls= sslbls)


    def to_local_eel(self):
        """
        Returns a `LocalElementaryErrorgenLabel` equivalent to this `LocalStimErrorgenLabel`.

        Returns
        -------
        `LocalElementaryErrorgenLabel`
        """
        return _LEEL(self.errorgen_type, self._hashable_basis_element_labels)


