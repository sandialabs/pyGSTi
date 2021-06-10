"""
Defines the StabilizerState and StabilizerFrame classes
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
import itertools as _itertools

import numpy as _np

from pygsti.tools import matrixmod2 as _mtx
from pygsti.tools import symplectic as _symp


#Notes regarding literature:
# Hostens, Dehaene, De Moor "Stabilizer states and Clifford operations for systems of arbitrary dimensions, and modular
# arithmetic" (arXiv:quant-ph/0408190)
#  - shows how to encode/manipulate cliffords and stabilizer states for qudits
#  - this is *exactly* the representation that we use, and formulas in this paper are
#     identical to the ones used in pyGSTi. (uses "11 = -iY" and mod4 phase vectors)
# Jeroen Dehaene and Bart De Moor "The Clifford group, stabilizer states, and linear and quadratic operations over
# GF(2)" (arXiv:quant-ph/0304125v1)
#  - shows how to encode/manipulate cliffords and stabilizer states for qubits in ptic
#  - uses bit (mod2) not mod 4 phase vectors, unlike what we do now, but may be useful
# Aaronson & Gottesman "Improved Simulation of Stabilizer Circuits" (arXiv:quant-ph/0406196)
#  - explains and implements anti-stabilizer for faster ops, but no stabilizer frames.
#  - explains how to get *magnitude* of inner product between two stabilizer states, but this is
#     all you can do without frames.
#  - uses "11 = Y" convention for Pauli encoding that is *different* than what we use here, so
#     there are modifications because of that.
# Garcia & Markov & Cross "Efficient Inner-product Algorithm for Stabilizer States" (arXiv:1210.6646v3) 2013
#  - gives good algorithm descriptions for stabilizer frames
# Garcia & Markov "Simulation of Quantum Circuits via Stabilizer Frames" (arXiv:1712.03554) 2017
#  - also useful for understanding stabilizer frames.

class StabilizerFrame(object):
    """
    Encapsulates a stabilizer frame (linear combo of stabilizer states).

    Stores stabilizer elements in the first n, and
    antistabilizer elements in the latter n *columns* of
    the "state matrix" (to facilitate composition),
    and phase vectors & amplitudes in parallel arrays.

    Parameters
    ----------
    state_s : numpy.ndarray
        A 2n x 2n binary matrix, where n is the number of qubits. The
        first n columns specify stabilizer elements and the latter n
        colunns specify anti-stabilizer elements.  In each column, bits
        (i,i+n) encode a Pauli on the i-th qubit: 00 = I, 10 = X, 01 = Z,
        11 = -iY. (Note the -iY is different from some representations
        in the literature which encode 11 = Y.)

    state_ps : numpy.ndarray, optional
        A mod-4 array of shape (k,2n) where n is the number of qubits
        and k is the number of components in the stabilizer frame.  Each
        row of `state_ps` is the  phase vector for the corresponding to
        to an amplitude in `amps`.  A phase vector encodes the overall phase
        of each of the stabilizer and anti-stabilizer elements (the columns
        of `state_s`) by specyfing the number of 'i' factors (0 to 3).  If
        None, then no phase vectors are stored.

    amps : numpy.ndarray, optional
        The (complex) amplitudes for each stabilizer-state component of the
        frame.  The length of this 1D array must equal 'k', the first
        dimension of `state_ps`.  `amps` should be None when and only when
        `state_ps` is None, corresponding to the case of zero components.

    Attributes
    ----------
    nqubits : int
        The number of qubits in the state this frame represents
    """

    @classmethod
    def from_zvals(cls, nqubits=None, zvals=None):
        """
        Create a StabilizerFrame for a computational basis state.

        Parameters
        ----------
        nqubits : int, optional
            The number of qubits.  If None, inferred from the length of `zvals`.

        zvals : iterable, optional
            An iterable over anything that can be cast as True/False
            to indicate the 0/1 value of each qubit in the Z basis.
            If None, the all-zeros state is created.

        Returns
        -------
        StabilizerFrame
        """
        if nqubits is None and zvals is None:
            raise ValueError("Must specify one of `nqubits` or `zvals`")

        if nqubits is None:
            nqubits = len(zvals)

        s = _np.fliplr(_np.identity(2 * nqubits, int))  # flip b/c stab cols are *first*
        p = _np.zeros(2 * nqubits, int)
        if zvals is not None:
            for i, z in enumerate(zvals):
                p[i] = p[i + nqubits] = 2 if bool(z) else 0
                # TODO: check this is right -- (how/need to update the destabilizers?)
        return cls(s, [p], [1.0])

    def __init__(self, state_s, state_ps=None, amps=None):
        """
        Initialize a new StabilizerFrame object.

        Parameters
        ----------
        state_s : numpy.ndarray
            A 2n x 2n binary matrix, where n is the number of qubits. The
            first n columns specify stabilizer elements and the latter n
            colunns specify anti-stabilizer elements.  In each column, bits
            (i,i+n) encode a Pauli on the i-th qubit: 00 = I, 10 = X, 01 = Z,
            11 = -iY. (Note the -iY is different from some representations
            in the literature which encode 11 = Y.)

        state_ps : numpy.ndarray, optional
            A mod-4 array of shape (k,2n) where n is the number of qubits
            and k is the number of components in the stabilizer frame.  Each
            row of `state_ps` is the  phase vector for the corresponding to
            to an amplitude in `amps`.  A phase vector encodes the overall phase
            of each of the stabilizer and anti-stabilizer elements (the columns
            of `state_s`) by specyfing the number of 'i' factors (0 to 3).  If
            None, then no phase vectors are stored.

        amps : numpy.ndarray, optional
            The (complex) amplitudes for each stabilizer-state component of the
            frame.  The length of this 1D array must equal 'k', the first
            dimension of `state_ps`.  `amps` should be None when and only when
            `state_ps` is None, corresponding to the case of zero components.
        """
        self.n = state_s.shape[0] // 2
        assert(state_s.shape == (2 * self.n, 2 * self.n))

        self.s = state_s.copy()
        if state_ps is not None:
            self.ps = _np.empty((len(state_ps), 2 * self.n), _np.int64)
            for i, p in enumerate(state_ps):
                self.ps[i, :] = p[:]
            #OLD self.ps = [ p for p in state_ps ]
        else:
            self.ps = _np.empty((0, 2 * self.n), _np.int64)
            #OLD self.ps = []

        if amps is not None:
            assert(len(amps) == len(self.ps)), \
                "Number of amplitudes must match number of phase vectors!"
            self.a = _np.array([complex(a) for a in amps], complex)
        else:
            self.a = _np.ones(self.ps.shape[0], complex)  # all == 1.0 by default

        n = self.n
        self.u = _np.zeros((2 * n, 2 * n), int)  # for colsum(...)
        self.u[n:2 * n, 0:n] = _np.identity(n, int)

        self.zblock_start = None  # first column of Z-block, set by _rref()
        self.view_filters = []   # holds qubit filters limiting the action
        # of clifford_update for this state
        self._rref()

    def to_rep(self, state_space):
        """
        Return a "representation" object for this StabilizerFrame

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        StateRep
        """
        from .statereps import StateRep as _StateRep
        return _StateRep(_np.ascontiguousarray(self.s, _np.int64),
                         _np.ascontiguousarray(self.ps, _np.int64),
                         _np.ascontiguousarray(self.a, complex), state_space)

    def copy(self):
        """
        Copy this stabilizer frame.

        Note that this also copies "view filters" setup via `push_view` calls.

        Returns
        -------
        StabilizerFrame
        """
        cpy = StabilizerFrame(self.s.copy(), [p.copy() for p in self.ps],
                              self.a.copy())  # NOT a view
        cpy.view_filters = self.view_filters[:]
        return cpy

    def push_view(self, qubit_filter):
        """
        Applies a filter to the action of `clifford_update`.

        After calling `push_view`, the stabilizer frame looks to some
        extent as though it were only a frame on the subset of qubits
        given by `qubit_fitler`.  In particular, calls to `clifford_update`
        should specify clifford operations that act only on the filtered
        qubits.  Furthermore, views can be nested.  For example, if on a frame
        starting with 10 qubits (labeled 0 to 9) push_view([3,4,5,6]) and
        then push_view([0,2]) are called, the current "view" will be of the
        original qubits 3 and 5.

        This is useful for applying "embedded" gates (those acting on just
        a subset of the state space).

        Parameters
        ----------
        qubit_filter : list
            A list of qubit indices to view, relative to the current view.

        Returns
        -------
        None
        """
        self.view_filters.append(qubit_filter)

    def pop_view(self):
        """
        Removes the last-applied (via :method:`push_view`) view filter.

        Returns
        -------
        list
            A list of qubit indices to view (a "view filter").
        """
        return self.view_filters.pop()

    @property
    def nqubits(self):
        """
        The number of qubits in the state this frame represents

        Returns
        -------
        int
        """
        return self.n  # == (self.s.shape[0] // 2)

    def _colsum(self, i, j):
        """ Col_i = Col_j * Col_i where '*' is group action"""
        n, s = self.n, self.s
        for p in self.ps:
            p[i] = (p[i] + p[j] + 2 * float(_np.dot(s[:, i].T, _np.dot(self.u, s[:, j])))) % 4
        for k in range(n):
            s[k, i] = s[k, j] ^ s[k, i]
            s[k + n, i] = s[k + n, j] ^ s[k + n, i]
            #TODO: use _np.bitwise_xor or logical_xor here? -- keep it obvious (&slow) for now...
        return

    def _colswap(self, i, j):
        """ Swaps Col_i & Col_j  """
        temp = self.s[:, i].copy()
        self.s[:, i] = self.s[:, j]
        self.s[:, j] = temp

        for p in self.ps:
            p[i], p[j] = p[j], p[i]

    def _rref(self):
        """ Update self.s and self.ps to be in reduced/canonical form
            Based on arXiv: 1210.6646v3 "Efficient Inner-product Algorithm for Stabilizer States"
        """
        n = self.n

        #Pass1: form X-block (of *columns*)
        i = 0  # current *column* (to match ref, but our rep is transposed!)
        for j in range(n):  # current *row*
            for k in range(i, n):  # set k = column with X/Y in j-th position
                if self.s[j, k] == 1: break  # X or Y check
            else: continue  # no k found => next column
            self._colswap(i, k)
            self._colswap(i + n, k + n)  # mirror in antistabilizer
            for m in range(n):
                if m != i and self.s[j, m] == 1:  # j-th literal of column m(!=i) is X/Y
                    self._colsum(m, i)
                    self._colsum(i + n, m + n)  # reverse-mirror in antistabilizer (preserves relations)
            i += 1

        self.zblock_start = i  # first column of Z-block

        #Pass2: form Z-block (of *columns*)
        for j in range(n):  # current *row*
            for k in range(i, n):  # set k = column with Z in j-th position
                if (self.s[j, k], self.s[j + n, k]) == (0, 1): break  # Z check
            else: continue  # no k found => next column
            self._colswap(i, k)
            self._colswap(i + n, k + n)  # mirror in antistabilizer
            for m in range(n):
                if m != i and self.s[j + n, m] == 1:  # j-th literal of column m(!=i) is Z/Y
                    self._colsum(m, i)
                    self._colsum(i + n, m + n)  # reverse-mirror in antistabilizer (preserves relations)
            i += 1
        return

    def _canonical_amplitudes(self, ip, target=None, qs_to_sample=None):
        """
        Canonical amplitudes are the ones that we assign to the components of a
        stabilizer state even though they're not actually specified by the
        stabilizer group -- these serve as an arbitrary baseline so that we
        can keep track of how amplitudes change by keeping track of amplitudes
        *relative* to these fixed "canonical" amplitudes.

        Extracts one or more canonical amplitudes from the
        ip-th stabilizer state: one if `target` is specified,
        otherwise a full set of values for the qubit indices in
        `qs_to_sample` (which, if it's the empty tuple, means
        that only one -- most convenient -- amplitude needs to be returned.
        """
        self._rref()  # ensure we're in reduced row echelon form
        n = self.n
        amplitudes = _collections.OrderedDict()
        if qs_to_sample is not None:
            remaining = 2**len(qs_to_sample)  # number we still need to find
            amp_samples = _np.nan * _np.ones(2**len(qs_to_sample), complex)
            # what we'll eventually return - holds amplitudes of all
            #  variations of qs_to_sample starting from anchor.

        debug = False  # DEBUG

        # Stage1: go through Z-block columns and find an "anchor" - the first
        # basis state that is allowed given the Z-block parity constraints.
        # (In Z-block, cols can have only Z,I literals)
        if debug: print("CanonicalAmps STAGE1: zblock_start = ", self.zblock_start)
        anchor = _np.zeros(n, int)  # "anchor" basis state (zvals), which gets amplitude 1.0 by definition
        lead = n
        for i in reversed(range(self.zblock_start, n)):  # index of current generator
            gen_p = self.ps[ip][i]  # phase of generator
            # counts number of Y's => -i's
            gen_p = (gen_p + 3 * int(_np.dot(self.s[:, i].T, _np.dot(self.u, self.s[:, i])))) % 4
            assert(gen_p in (0, 2)), "Logic error: phase should be +/- only!"

            # get positions of Zs
            zpos = []
            for j in range(n):
                if self.s[j + n, i] == 1: zpos.append(j)
            #OR: zpos = [ j for j in range(n) if self.s[j+n,i] == 1 ]

            # set values of anchor between zpos[0] and lead
            # (between current leading-Z position and the last iteration's,
            #  which marks the point at which anchor has been initialized to)
            fixed1s = 0  # relevant number of 1s fixed by the already-initialized part of 'anchor'
            target1s = 0  # number of 1s in target state, which we want to check for Z-block compatibility
            zpos_to_fill = []
            for j in zpos:
                if j >= lead:
                    if anchor[j] == 1: fixed1s += 1
                else: zpos_to_fill.append(j)
                if target is not None and target[j] == 1:
                    target1s += 1
            assert(len(zpos_to_fill) > 0)  # structure of rref Z-block should ensure this
            parity = gen_p // 2
            eff_parity = (parity - (fixed1s % 2)) % 2  # effective parity for zpos_to_fill

            if debug:  # DEBUG
                print("  Current gen = ", i, " phase = ", gen_p, " zpos=", zpos, " fixed1s=", fixed1s,
                      " tofill=", zpos_to_fill, " eff_parity=", eff_parity, "lead=", lead)
                print("   -anchor: ", anchor, end='')

            if target is not None and (target1s % 2) != parity:
                return 0.0 + 0j  # target fails this parity check -> it's amplitude == 0 (OK)

            if eff_parity == 0:  # even parity - fill with all 0s
                pass  # BUT already initalized to 0s, so don't need to do anything for anchor
            else:  # odd parity (= 1 or -1) - fill with all 0s except final zpos_to_fill = 1
                anchor[zpos_to_fill[-1]] = 1  # BUT just need to fill in the final 1
            lead = zpos_to_fill[0]  # update the leading-Z index
            if debug: print(" ==> ", anchor)  # DEBUG

        #Set anchor amplitude to appropriate 1.0/sqrt(2)^s
        # (by definition - serves as a reference pt)
        # Note: 's' equals the minimum number of generators that are *different*
        # between this state and the basis state we're extracting and ampl for.
        # Since any/all comp. basis state generators can form all and only the
        # Z-literal only (Z-block) generators 's' is simplly the number of
        # X-block generators (= self.zblock_start).
        s = self.zblock_start
        anchor_amp = 1 / (_np.sqrt(2.0)**s)
        amplitudes[tuple(anchor)] = anchor_amp

        if qs_to_sample is not None:
            remaining -= 1
            nk = len(qs_to_sample)
            anchor_indx = sum([anchor[qs_to_sample[k]] * (2**(nk - 1 - k)) for k in range(nk)])
            amp_samples[anchor_indx] = anchor_amp

        #STAGE 2b - for sampling a set
        if qs_to_sample is not None:
            #If we're trying to sample a set, check if any of the amplitudes
            # we're looking for are zero by the Z-block checks.  That is,
            # consider whether anchor with qs_to_sample indices updated
            # passes or fails each check
            for i in reversed(range(self.zblock_start, n)):  # index of current generator
                gen_p = self.ps[ip][i]  # phase of generator
                # counts number of Y's => -i's
                gen_p = (gen_p + 3 * int(_np.dot(self.s[:, i].T, _np.dot(self.u, self.s[:, i])))) % 4

                zpos = []
                for j in range(n):
                    if self.s[j + n, i] == 1: zpos.append(j)

                inds = []
                fixed1s = 0  # number of 1s in target state, which we want to check for Z-block compatibility
                for j in zpos:
                    if j in qs_to_sample:
                        inds.append(qs_to_sample.index(j))  # "sample" indices in parity check
                    elif anchor[j] == 1:
                        fixed1s += 1
                if len(inds) > 0:
                    parity = (gen_p // 2 - (fixed1s % 2)) % 2  # effective parity
                    for k, tup in enumerate(_itertools.product(*([[0, 1]] * len(qs_to_sample)))):
                        tup_parity = sum([tup[kk] for kk in inds]) % 2
                        if tup_parity != parity:  # parity among inds is NOT allowed => set ampl to zero
                            if _np.isnan(amp_samples[k]): remaining -= 1
                            amp_samples[k] = 0.0

        if debug: print("CanonicalAmps STAGE2: amps = ", list(amplitudes.items()))  # DEBUG

        #Check exit conditions
        if target is not None and _np.array_equal(anchor, target):
            return anchor_amp
        elif qs_to_sample is not None and remaining == 0:
            return (anchor, amp_samples)

        # Stage2: move through X-block processing existing amplitudes
        # (or processing only to move toward a target state?)
        def apply_xgen(igen, pgen, zvals_to_acton, ampl):
            """ Apply a given X-block generator """
            result = _np.array(zvals_to_acton, int); new_amp = -ampl if (pgen // 2 == 1) else ampl
            for j in range(n):  # for each element (literal) in generator
                if self.s[j, igen] == 1:  # X or Y
                    result[j] = 1 - result[j]  # flip!
                    # X => a' == a constraint on new/old amplitudes, so nothing to do
                    # Y => a' == i*a constraint, so:
                    if self.s[j + n, igen] == 1:  # Y
                        if result[j] == 1: new_amp *= 1j  # |0> -> i|1> (but "== 1" b/c result is already flipped)
                        else: new_amp *= -1j             # |1> -> -i|0>
                elif self.s[j + n, igen] == 1:  # Z
                    # Z => a' == -a constraint if basis[j] == |1> (otherwise a == a)
                    if result[j] == 1: new_amp *= -1
            #DEBUG print("DB PYTHON XGEN returns ",result,new_amp)
            return result, new_amp

        def get_target_ampl(tgt):
            #requires just a single pass through X-block
            zvals = anchor.copy(); amp = anchor_amp  # start with anchor state
            lead = -1
            for i in range(self.zblock_start):  # index of current generator
                gen_p = self.ps[ip][i]  # phase of generator
                # counts number of Y's => -i's
                gen_p = (gen_p + 3 * int(_np.dot(self.s[:, i].T, _np.dot(self.u, self.s[:, i])))) % 4
                assert(gen_p in (0, 2)), "Logic error: phase should be +/- only!"

                #Get leading flipped qubit (lowest # qubit which will flip when we apply this)
                for j in range(n):
                    if self.s[j, i] == 1:  # check for X/Y literal in qubit pos j
                        assert(j > lead)  # lead should be strictly increasing as we iterate due to rref structure
                        lead = j; break
                else: assert(False), "Should always break loop!"

                if debug: print("get_target_ampl: iter ", i, " lead=", lead, " genp=", gen_p, " amp=", amp)

                #Check whether we should apply this generator to zvals
                if zvals[lead] != tgt[lead]:
                    # then applying this generator is productive - do it!
                    if debug: print("Applying XGEN amp=", amp)
                    zvals, amp = apply_xgen(i, gen_p, zvals, amp)
                    if debug: print("Resulting amp = ", amp, " zvals = ", zvals)

                    #Check if we've found target
                    if _np.array_equal(zvals, tgt): return amp
            raise ValueError("Falied to find amplitude of target: ", tgt)

        if target is not None:
            if debug: print("Getting Target Amplitude")
            return get_target_ampl(target)
        elif qs_to_sample is not None:
            target = anchor.copy()
            for k, tup in enumerate(_itertools.product(*([[0, 1]] * len(qs_to_sample)))):
                if _np.isnan(amp_samples[k]):
                    target[list(qs_to_sample)] = tup
                    amp_samples[k] = get_target_ampl(target)
            return (anchor, amp_samples)

        else:
            # both target and qs_to_sample are None - just get & return as
            # many amplitudes as we can (for full state readout)

            num_ampl_added = 1  # just to kick off the loop
            while(num_ampl_added > 0):
                num_ampl_added = 0
                if debug: print("Starting X-block processing loop")  # DEBUG

                for i in range(self.zblock_start):  # index of current generator
                    gen_p = self.ps[ip][i]  # phase of generator
                    # counts number of Y's => -i's
                    gen_p = (gen_p + 3 * int(_np.dot(self.s[:, i].T, _np.dot(self.u, self.s[:, i])))) % 4
                    assert(gen_p in (0, 2)), "Logic error: phase should be +/- only!"

                    ##Get positions of qubits which will flip when we apply this
                    ## constraint to existing amplitudes (usefult to determining if we want to apply it)
                    #flippos = []
                    #for j in range(n):
                    #    if self.s[j,i] == 1: flippos.append(j) # check for X/Y literal in qubit pos j

                    if debug: print("  Current gen = ", i, " phase = ", gen_p)  # DEBUG

                    #Apply this generator to existing amplitudes (always)
                    existing_amps = list(amplitudes.keys())  # do this b/c loop can mutate amplitudes
                    for zvals in existing_amps:  # for all existing amplitudes
                        amp = amplitudes[zvals]
                        result, new_amp = apply_xgen(i, gen_p, zvals, amp)
                        t = tuple(result)
                        if t not in amplitudes:
                            amplitudes[tuple(result)] = new_amp
                            num_ampl_added += 1

                        if debug:
                            print("  -->Apply to b=", zvals, " (amp=", amp, ") ==> b'=", result, " (amp=", new_amp,
                                  ") acnt=", len(amplitudes))  # DEBUG
                            if not _np.isclose(amplitudes[tuple(result)], new_amp):
                                print("INCONSISTENCY w/existing amp = ", amplitudes[tuple(result)])
                        assert(_np.isclose(amplitudes[tuple(result)], new_amp)), "Inconsistency in amplitude generation"

            return list(amplitudes.items())

    def _canonical_amplitude(self, ip, zvals):
        """ Return the "intrinsic" amplitude of the given comp. basis state
            as encoded within the s,p matrices of the ip-th stabilizer state
            (alone) """
        return self._canonical_amplitudes(ip, target=zvals)

    def _sample_amplitude(self, ip, qs_to_sample=1):
        """ extract `count` convenient canonical amplitudes from the
            ip-th stabilizer state """
        return self._canonical_amplitudes(ip, qs_to_sample=qs_to_sample)

    def _apply_clifford_to_frame(self, s, p, qubit_filter):
        """
        Applies a clifford in the symplectic representation to this
        stabilize frame -- similar to `apply_clifford_to_stabilizer_state`
        but operates on an entire *frame*

        Parameters
        ----------
        s : numpy array
            The symplectic matrix over the integers mod 2 representing the Clifford

        p : numpy array
            The 'phase vector' over the integers mod 4 representing the Clifford
        """
        n = self.n
        assert(_symp.check_valid_clifford(s, p)), "The `s`,`p` matrix-vector pair is not a valid Clifford!"

        if qubit_filter is not None:
            s, p = _symp.embed_clifford(s, p, qubit_filter, n)  # for now, just embed then act normally
            #FUTURE: act just on the qubits we need to --> SPEEDUP!

        # Below we calculate the s and p for the output state using the formulas from
        # Hostens and De Moor PRA 71, 042315 (2005).
        out_s = _mtx.dot_mod2(s, self.s)

        inner = _np.dot(_np.dot(_np.transpose(s), self.u), s)
        vec1 = _np.dot(_np.transpose(self.s), p - _mtx.diagonal_as_vec(inner))
        matrix = 2 * _mtx.strictly_upper_triangle(inner) + _mtx.diagonal_as_matrix(inner)
        vec2 = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(self.s), matrix), self.s))

        self.s = out_s  # don't set this until we're done using self.s
        for i in range(len(self.ps)):
            self.ps[i] = (self.ps[i] + vec1 + vec2) % 4

    def format_state(self):
        """
        Get a string representing the full ip-th stabilizer state (w/global phase)

        Returns
        -------
        str
        """
        return "\n".join(["%s |%s>" % (str(amp), "".join(map(str, zvals)))
                          for zvals, amp in self.extract_all_amplitudes().items()])

    def extract_all_amplitudes(self):
        """
        Get a dictionary of the *full* amplitudes of each present computational basis state.

        This may take a while for many-qubit states, as it requires
        getting 2^(num_qubits) amplitudes.

        Returns
        -------
        dict
            Keys are tuples of z-values specifying the different computational
            basis states.  Values are the complex amplitudes.
        """
        amps = _collections.OrderedDict()
        for ip in range(len(self.ps)):
            g_amp = self.a[ip]  # global amplitude
            camplitudes = self._canonical_amplitudes(ip, target=None, qs_to_sample=None)
            for zvals, camp in camplitudes:
                if tuple(zvals) in amps:
                    amps[tuple(zvals)] += g_amp * camp
                else:
                    amps[tuple(zvals)] = g_amp * camp
        return amps

    def to_statevec(self):
        """
        Convert this stabilizer frame to dense length-2^(num_qubits) complex state vector of amplitudes.

        Returns
        -------
        numpy.ndarray
        """
        ret = _np.empty(2**self.n, complex)
        amps = self.extract_all_amplitudes()
        for k, zvals in enumerate(_itertools.product(*([[0, 1]] * self.n))):
            ret[k] = amps.get(zvals, 0.0)
        return ret

    def extract_amplitude(self, zvals):
        """
        Get the *full* (not just "canonical") amplitude of a given computational basis state.

        Parameters
        ----------
        zvals : numpy.ndarray
            An array of 0/1 elements specifying the desired basis state.

        Returns
        -------
        complex
        """
        ampl = 0
        for ip, a in enumerate(self.a):
            ampl += a * self._canonical_amplitude(ip, zvals)
        return ampl

    def clifford_update(self, smatrix, svector, u_mx, qubit_filter=None):
        """
        Update this stabilizer frame by the action of a Clifford operation.

        The Clifford operation is given in the usual symplectic representation.
        If there are any active views (from calling :method:`push_view`) and/or
        if `qubit_filter` is not None, then `smatrix`, `svector`, and `u_mx`
        should be sized for just the number of qubits in the current view.

        Parameters
        ----------
        smatrix : numpy.ndarray
            The symplectic matrix of shape (2n,2n), where n is the number of
            qubits (in the current view if applicable), representing the Clifford operation.

        svector : numpy.ndarray
            The phase vector of shape (2n,) representing the Clifford operation.

        u_mx : numpy.ndarray
            The dense unitary representation of the Clifford action, which is
            needed in order to track the global phase of the frame (state).
            This is a complex matrix of shape (2^n,2^n), where n is the number of
            qubits (in the current view if applicable).

        qubit_filter : list, optional
            An additional view filter to apply just for this function call (i.e.
            it is not stored on a stack as it is for :method:`push_view`.

        Returns
        -------
        None
        """
        debug = False

        qubits = list(range(self.n))  # start with all qubits being acted on
        for qfilter in self.view_filters:
            qubits = [qubits[i] for i in qfilter]  # apply each filter
        if qubit_filter is not None:
            qubits = [qubits[i] for i in qubit_filter]  # finally apply qubit_filter

        nQ = len(qubits)  # number of qubits being acted on (<= n in general)
        sampled_amplitudes = []

        #Step1: Update global amplitudes - Part A
        if debug: print("UPDATE GLOBAL AMPS: zstart=", self.zblock_start)  # DEBUG
        for ip in range(len(self.ps)):
            if debug: print("SAMPLE AMPLITUDES")
            base_state, ampls = self._sample_amplitude(ip, qubits)
            if debug: print("GOT ", base_state, ampls)
            sampled_amplitudes.append((base_state, ampls))

        #Step2: Apply clifford to stabilizer reps in self.s, self.ps
        if debug: print("APPLY CLIFFORD TO FRAME")
        self._apply_clifford_to_frame(smatrix, svector, qubits)
        self._rref()
        #print("DB: s = \n",self.s) # DEBUG
        #print("DB: ps = ",self.ps) # DEBUG

        #Step3: Update global amplitudes - Part B
        for ip, (base_state, ampls) in enumerate(sampled_amplitudes):

            # print("DB: u_mx = ",u_mx) # DEBUG
            if debug: print("APPLYING U to instate =", ampls)
            instate = ampls
            outstate = _np.dot(u_mx, instate)  # state-vector propagation
            if debug: print("OUTSTATE = ", outstate)
            #TODO: sometimes need a second state & set instate to the sum?
            # choose second state based on im/re amplitude & just run through U separately?

            #Look for nonzero output component and figure out how
            # phase *actually* changed as per state-vector propagation, then
            # update self.a (global amplitudes) to account for this.
            for k, comp in enumerate(outstate):  # comp is complex component of output state
                if abs(comp) > 1e-6:
                    k_zvals = _np.array([int(bool(k & (2**(nQ - 1 - i))))
                                         for i in range(nQ)], int)  # hack to extract binary(k)
                    zvals = _np.array(base_state, int)
                    zvals[qubits] = k_zvals
                    if debug: print("GETTING CANONICAL AMPLITUDE for B' = ", zvals, " actual=", comp)
                    if debug: print(str(self))
                    camp = self._canonical_amplitude(ip, zvals)
                    assert(abs(camp) > 1e-6), "Canonical amplitude zero when actual isn't!!"
                    if debug: print("GOT CANONICAL AMPLITUDE =", camp, " updating global amp w/", comp / camp)
                    self.a[ip] *= comp / camp  # "what we want" / "what stab. frame gives"
                    # this essentially updates a "global phase adjustment factor"
                    break  # move on to next stabilizer state & global amplitude
            else:
                raise ValueError("Outstate was completely zero!")
                # (this shouldn't happen if u_mx is unitary!)

    def measurement_probability(self, zvals, qubit_filter=None, return_state=False, check=False):
        """
        Extract the probability of obtaining a given computation-basis-measurement outcome.

        Parameters
        ----------
        zvals : numpy.ndarray
            An array of 0/1 elements specifying the computational basis outcomes.

        qubit_filter : list, optional
            A list specifying a subset of the qubits to measure. `len(zvals)`
            should always equal `len(qubit_filter)`.  If None, then all qubits
            are measured.
            **Currently unsupported.**

        return_state : bool, optional
            Whether the post-measurement state (frame) should be returned.
            **Currently unsupported.**

        check : bool, optional
            Whether to perform internal self-consistency checks (for debugging,
            makes function run more slowly).

        Returns
        -------
        float
        """

        if qubit_filter is not None or return_state:
            raise NotImplementedError("`qubit_filter` and `return_state` args are not functional yet")

        # Could make this faster in the future by using anticommutator?
        # - maybe could use a _canonical_probability for each ip that is
        #   essentially the 'stabilizer_measurement_prob' fn? -- but need to
        #   preserve *amplitudes* upon measuring & getting output state, which
        #   isn't quite done in the 'pauli_z_meaurement' function.
        amp = self.extract_amplitude(zvals)
        p = abs(amp)**2

        #2nd method using anticommutator - but we don't know how to update
        # global phase of state here either.
        if check:
            p_chk = sum([abs(a)**2 * _symp.stabilizer_measurement_prob((self.s, pr), zvals)
                         for a, pr in zip(self.a, self.ps)])
            assert(_np.isclose(p, p_chk)), \
                "Stabilizer-frame meas. probability check failed: %g != %g" % (p, p_chk)

        return p

    def __str__(self):
        print_anti = True
        n = self.n; K = len(self.ps)
        nrows = 2 * n if print_anti else n  # number of rows
        s = ""

        # Output columns as rows of literals to conform with usual picture
        s += "Global amplitudes = " + ", ".join(map(str, self.a)) + "\n"
        s += "   " * K + "  " + "----" * n + "-\n"

        for i in range(nrows):  # column index - show only stabilizer or now, not antistabilizer

            # print divider before Zblock
            if i == self.zblock_start and 0 < i < n:
                s += "   " * K + " |" + "----" * n + "-|\n"

            if i == n:  # when we print the antistabilizer
                s += "   " * K + " |" + "====" * n + "=|\n"

            # print leading signs (one per stabilizer state)
            for p in self.ps:
                if p[i] == 0: s += "  +"
                elif p[i] == 1: s += "  i"
                elif p[i] == 2: s += "  -"
                elif p[i] == 3: s += " -i"
                else: s += " ??"
            s += " |"

            # print common generator corresponding to this column
            for j in range(n):
                lc = (self.s[j, i], self.s[j + n, i])  # code for literal
                if lc == (0, 0): s += "    "
                elif lc == (0, 1): s += "   Z"
                elif lc == (1, 0): s += "   X"
                elif lc == (1, 1): s += " -iY"
                else: s += " ???"

            s += " |\n"

        s += "   " * K + "  " + "----" * n + "-\n"
        return s


def sframe_kronecker(sframe_factors):
    """
    Computes a tensor-product StabilizerFrame from a set of factors.

    Parameters
    ----------
    sframe_factors : list of StabilizerFrame objects
        The factors to tensor together in the given left-to-right order.

    Returns
    -------
    StabilizerFrame
    """

    #Similar to symplectic_kronecker
    n = sum([sf.nqubits for sf in sframe_factors])  # total number of qubits

    # (common) state matrix
    sout = _np.zeros((2 * n, 2 * n), int)
    k = 0  # current qubit index
    for sf in sframe_factors:
        nq = sf.nqubits
        sout[k:k + nq, k:k + nq] = sf.s[0:nq, 0:nq]
        sout[k:k + nq, n + k:n + k + nq] = sf.s[0:nq, nq:2 * nq]
        sout[n + k:n + k + nq, k:k + nq] = sf.s[nq:2 * nq, 0:nq]
        sout[n + k:n + k + nq, n + k:n + k + nq] = sf.s[nq:2 * nq, nq:2 * nq]
        k += nq

    # phase vectors and amplitudes
    ps_out = []; amps_out = []
    inds = [range(len(sf.ps)) for sf in sframe_factors]
    for ii in _itertools.product(*inds):
        pout = _np.zeros(2 * n, int); amp = 1.0
        for i, sf in zip(ii, sframe_factors):
            nq = sf.nqubits
            pout[k:k + nq] = sf.ps[i][0:nq]
            pout[n + k:n + k + nq] = sf.ps[i][nq:2 * nq]
            amp *= sf.a[i]
        ps_out.append(pout)
        amps_out.append(amp)

    return StabilizerFrame(sout, ps_out, amps_out)
