#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Pauli state/operation/outcome objects for Idle Tomography """

import numpy as _np
from ... import objects as _objs
from ...objects.label import Label as _Lbl

#Helper function


def _commute_parity(P1, P2):
    """ 1 if P1 commutes w/P2, -1 if they anticommute """
    return 1 if (P1 == "I" or P2 == "I" or P1 == P2) else -1


class NQOutcome(object):
    """
    A string of 0's and 1's representing a definite outcome in the Z-basis.
    """

    @classmethod
    def Weight1String(cls, N, i):
        """ creates a `N`-bit string with a 1 in location `i`. """
        ident = list("0" * N)
        ident[i] = "1"
        return cls(''.join(ident))

    @classmethod
    def Weight2String(cls, N, i, j):
        """ creates a `N`-bit string with 1s in locations `i` and `j`. """
        ident = list("0" * N)
        ident[i] = "1"
        ident[j] = "1"
        return cls(''.join(ident))

    def __init__(self, string_rep):
        """
        Create a NQOutcome.

        Parameters
        ----------
        string_rep : str
            A string of 0s and 1s, one per qubit, e.g. "0010".
        """
        self.rep = string_rep

    def __str__(self):
        return self.rep

    def __repr__(self):
        return "NQOutcome[%s]" % self.rep

    def __eq__(self, other):
        return self.rep == other.rep

    def __hash__(self):
        return hash(self.rep)

    def flip(self, *indices):
        """
        Flip "0" <-> "1" at any number of indices.
        This function takes a variable number of integer arguments
        specifying the qubit indices whose value should be flipped.

        Returns
        -------
        NQOutcome
            A *new* outcome object with flipped bits.
        """
        outcomes = [self.rep[i] for i in range(len(self.rep))]
        for i in indices:
            if outcomes[i] == '0': outcomes[i] = '1'
            elif outcomes[i] == '1': outcomes[i] = '0'
        return NQOutcome(''.join(outcomes))


class NQPauliState(object):
    """
    A N-qubit state that is the tensor product of N
    1-qubit Pauli eigenstates.  These can be represented as
    a string of Xs, Ys and Zz (but not Is) *each* with a +/-
    sign indicating which of the two eigenstates is meant.

    A NQPauliState object can also be used to represent a POVM
    whose effects are the projections onto the 2^N tensor products
    of (the given) Pauli eigenstates.  The +/- sign in this case
    indicates which eigenstate is equated with the "0" (vs "1") outcome.
    """

    def __init__(self, string_rep, signs=None):
        """
        Create a NQPauliState

        Parameters
        ----------
        string_rep : str
            A string with letters in {X,Y,Z} (note: I is not allowed!),
            specifying the Pauli basis for each qubit.

        signs : tuple, optional
            A tuple of 0s and/or 1s.  A zero means the "+" eigenvector is
            either prepared or corresponds to the "0" outcome (if this
            NQPauliState is used to describe a measurment basis).  A one
            means the opposite: the "-" eigenvector is prepared and it
            corresponds to a "0" outcome.  The default is all zeros.
        """
        assert("I" not in string_rep), "'I' cannot be in a NQPauliState"
        self.rep = string_rep
        if signs is None:
            signs = (0,) * len(self.rep)
        self.signs = signs

    def __len__(self):
        return len(self.rep)

    def __str__(self):
        sgn = {1: '+', -1: '-'}
        return "".join(["%s%s" % (sgn[s], let)
                        for s, let in zip(self.signs, self.rep)])

    def __repr__(self):
        return "State[" + str(self) + "]"

    def __eq__(self, other):
        return (self.rep == other.rep) and (self.signs == other.signs)

    def __hash__(self):
        return hash(str(self))

    def to_circuit(self, pauliBasisDict):
        """
        Convert this Pauli basis state or measurement to a fiducial operation sequence.

        When the returned operation sequence follows a preparation in the |0...0>
        Z-basis state or is followed by a Z-basis measurement (with all "+"
        signs), then the Pauli state preparation or measurement described by
        this object will be performed.

        Parameters
        ----------
        pauliBasisDict : dict
            A dictionary w/keys like `"+X"` or `"-Y"` and values that
            are tuples of gate *names* (not labels, which include qubit or
            other state-space designations), e.g. `("Gx","Gx")`.  This
            dictionary describes how to prepare or measure in Pauli bases.

        Returns
        -------
        Circuit
        """
        opstr = []
        sgn = {1: '+', -1: '-'}
        nQubits = len(self.signs)
        for i, (s, let) in enumerate(zip(self.signs, self.rep)):
            key = sgn[s] + let  # e.g. "+X", "-Y", etc
            if key not in pauliBasisDict and s == +1:
                key = let  # try w/out "+"
            if key not in pauliBasisDict:
                raise ValueError("'%s' is not in `pauliBasisDict` (keys = %s)"
                                 % (key, str(list(pauliBasisDict.keys()))))
            opstr.extend([_Lbl(opname, i) for opname in pauliBasisDict[key]])
            # pauliBasisDict just has 1Q gate *names* -- need to make into labels
        return _objs.Circuit(opstr, num_lines=nQubits).parallelize()


class NQPauliOp(object):
    """
    A N-qubit pauli operator, consisting of
    a 1-qubit Pauli operation on each qubits.
    """

    @classmethod
    def Weight1Pauli(cls, N, i, P):
        """
        Creates a `N`-qubit Pauli operator with the Pauli indexed
        by `P` in location `i`.

        Parameters
        ----------
        N : int
            The number of qubits

        i : int
            The index of the single non-trivial Pauli operator.

        P : int
            An integer 0 <= `P` <= 2 indexing the non-trivial Pauli at location
            `i` as follows: 0='X', 1='Y', 2='Z'.

        Returns
        -------
        NQPauliOp
        """
        ident = list("I" * N)
        ident[i] = ["X", "Y", "Z"][P]
        return cls(''.join(ident))

    @classmethod
    def Weight2Pauli(cls, N, i, j, P1, P2):
        """
        Creates a `N`-qubit Pauli operator with the Paulis indexed
        by `P1` and `P2` in locations `i` and `j` respectively.

        Parameters
        ----------
        N : int
            The number of qubits

        i, j : int
            The indices of the non-trivial Pauli operators.

        P1,P2 : int
            Integers 0 <= `P` <= 2 indexing the non-trivial Paulis at locations
            `i` and `j`, respectively, as follows: 0='X', 1='Y', 2='Z'.

        Returns
        -------
        NQPauliOp
        """

        """
        Creates a `N`-qubit Pauli operator with Paulis `P1` and `P2` in locations
        `i` and `j` respectively.
        """
        ident = list("I" * N)
        ident[i] = ["X", "Y", "Z"][P1]
        ident[j] = ["X", "Y", "Z"][P2]
        return cls(''.join(ident))

    def __init__(self, string_rep, sign=1):
        """
        Create a NQPauliOp.

        Parameters
        ----------
        string_rep : str
            A string with letters in {I,X,Y,Z}, specifying the Pauli operator
            for each qubit.

        sign : {1, -1}
            An overall sign (prefactor) for this operator.
        """
        self.rep = string_rep
        self.sign = sign  # +/- 1

    def __len__(self):
        return len(self.rep)

    def __str__(self):
        return "%s%s" % ('-' if (self.sign == -1) else ' ', self.rep)

    def __repr__(self):
        return "NQPauliOp[%s%s]" % ('-' if (self.sign == -1) else ' ', self.rep)

    def __eq__(self, other):
        return (self.rep == other.rep) and (self.sign == other.sign)

    def __hash__(self):
        return hash(str(self))

    def subpauli(self, indices):
        """
        Returns a new `NQPauliOp` object which sets all (1-qubit) operators to
        "I" except those in `indices`, which remain as they are in this object.

        Parameters
        ----------
        indices : iterable
            A sequence of integer indices between 0 and N-1, where N is
            the number of qubits in this pauli operator.

        Returns
        -------
        NQPauliOp
        """
        ident = list("I" * len(self.rep))
        for i in indices:
            ident[i] = self.rep[i]
        return NQPauliOp(''.join(ident))

    def dot(self, other):
        """
        Computes the Hilbert-Schmidt dot product (normed to 1) between this
        Pauli operator and `other`.

        Parameters
        ----------
        other : NQPauliOp
            The other operator to take a dot product with.

        Returns
        -------
        integer
            Either 0, 1, or -1.
        """
        assert(len(self) == len(other)), "Length mismatch!"
        if other.rep == self.rep:
            return self.sign * other.sign
        else:
            return 0

    def statedot(self, state):
        """
        Computes a dot product between `state` and this operator.
        (note that an X-basis '+' state is represented by (I+X) not just X)

        Parameters
        ----------
        state : NQPauliState

        Returns
        -------
        int
        """
        # Instead of computing P1*P2 on each Pauli in self (other), it computes P1*(I+P2).
        # (this is only correct if all the Paulis in `other` are *not* I)

        assert(isinstance(state, NQPauliState))
        assert(len(self) == len(state)), "Length mismatch!"

        ret = self.sign  # keep track of -1s
        for P1, P2, state_sign in zip(self.rep, state.rep, state.signs):
            if _commute_parity(P1, P2) == -1: return 0
            # doesn't commute so => P1+P1*P2 = P1+Q = traceless
            elif P1 == 'I':  # I*(I+/-P) => (I+/-P) and "sign" of i-th el of state doesn't matter
                pass
            elif state_sign == -1:  # P*(I-P) => (P-I) and so sign (neg[i]) gets moved to I and affects the trace
                assert(P1 == P2)
                ret *= -1
        return ret

    def commuteswith(self, other):
        """
        Determine whether this operator commutes (or anticommutes) with `other`.

        Parameters
        ----------
        other : NQPauliOp

        Returns
        -------
        bool
        """
        assert(len(self) == len(other)), "Length mismatch!"
        return bool(_np.prod([_commute_parity(P1, P2) for P1, P2 in zip(self.rep, other.rep)]) == 1)

    def icommutatorOver2(self, other):
        """
        Compute `i[self, other]/2` where `[,]` is the commutator.

        Parameters
        ----------
        other : NQPauliOp or NQPauliState
            The operator to take a commutator with.  A `NQPauliState` is treated
            as an operator (i.e. 'X' basis state => 'X' Pauli operation) with
            sign given by the product of its 1-qubit basis signs.

        Returns
        -------
        NQPauliOp
        """

        #Pauli commutators:
        # i(  ... x Pi Qi x ...
        #   - ... x Qi Pi x ... )
        # Now, Pi & Qi either commute or anticommute, i.e.
        # PiQi = QiPi or PiQi = -QiPi.  Let Si be the sign (or *parity*) so
        # by definition PiQi = Si*QiPi.  Note that Si==1 iff Pi==Qi or either == I.
        # If prod(Si) == 1, then the commutator is zero.  If prod(Si) == -1 then
        # the commutator is
        # 2*i*( ... x Pi Qi x ... ) = 2*i*( ... x Ri x ... ) where
        #  Ri = I if Pi==Qi (exactly when Si==1), or
        #  Ri = Pi or Qi if Pi==I or Qi==I , otherwise
        #  Ri = i(+/-1)P' where P' is another Pauli. (this is same as case when Si == -1)

        def Ri_operator(P1, P2):
            """ the *operator* (no sign) part of R = P1*P2 """
            if P1 + P2 in ("XY", "YX", "IZ", "ZI"): return "Z"
            if P1 + P2 in ("XZ", "ZX", "IY", "YI"): return "Y"
            if P1 + P2 in ("YZ", "ZY", "IX", "XI"): return "X"
            if P1 + P2 in ("II", "XX", "YY", "ZZ"): return "I"
            assert(False)

        def Ri_sign(P1, P2, parity):
            """ the +/-1 *sign* part of R = P1*P2 (doesn't count the i-factor in 3rd case)"""
            if parity == 1: return 1  # pass commuteParity(P1,P2) to save computation
            return 1 if P1 + P2 in ("XY", "YZ", "ZX") else -1

        assert(len(self) == len(other)), "Length mismatch!"
        s1, s2 = self.rep, other.rep
        parities = [_commute_parity(P1, P2) for P1, P2 in zip(s1, s2)]
        if _np.prod(parities) == 1: return None  # an even number of minus signs => commutator = 0

        op = ''.join([Ri_operator(P1, P2) for P1, P2 in zip(s1, s2)])
        num_i = parities.count(-1)  # number of i factors from 3rd Ri case above
        sign = (-1)**((num_i + 1) / 2) * _np.prod([Ri_sign(P1, P2, p) for P1, P2, p in zip(s1, s2, parities)])
        if isinstance(other, NQPauliOp): other_sign = other.sign
        elif isinstance(other, NQPauliState): other_sign = _np.product(other.signs)
        else: raise ValueError("Can't take commutator with %s type" % str(type(other)))

        return NQPauliOp(op, sign * self.sign * other_sign)
