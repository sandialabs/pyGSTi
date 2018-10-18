""" Pauli state/operation/outcome objects for Idle Tomography """
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
from ... import objects as _objs
from ...baseobjs.label import Label as _Lbl

#Helper function
def _commute_parity(P1,P2):
    """ 1 if P1 commutes w/P2, -1 if they anticommute """
    return 1 if (P1 == "I" or P2 == "I" or P1==P2) else -1


class NQOutcome(object):
    """ A string of 0's and 1's representing a definite outcome in the Z-basis """

    @classmethod
    def Weight1String(cls, N, i):
        """ creates a `N`-bit string with a 1 in location `i`. """
        ident = list("0"*N)
        ident[i] =  "1"
        return cls(''.join(ident))
    
    
    @classmethod
    def Weight2String(cls, N, i, j):
        """ creates a `N`-bit string with 1s in locations `i` and `j`. """
        ident = list("0"*N)
        ident[i] =  "1"
        ident[j] =  "1"
        return cls(''.join(ident))

    def __init__(self, string_rep):
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
        TODO: docstring
        Flip "0" <-> "1" at any number of indices.
        Returns a *new* NQOutcome with flipped bits.
        """
        outcomes = [ self.rep[i] for i in range(len(self.rep)) ]
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
        TODO: docstring - by default signs are all +
        """
        assert("I" not in string_rep), "'I' cannot be in a NQPauliState"
        self.rep = string_rep
        if signs is None: 
            signs = (1,)*len(self.rep)
        self.signs = signs

    def __len__(self):
        return len(self.rep)

    def __str__(self):
        sgn = {1:'+', -1:'-'}
        return "".join(["%s%s" % (sgn[s],let)
                        for s,let in zip(self.signs,self.rep)])

    def __repr__(self):
        return "State[" + str(self) + "]"

    def __eq__(self, other):
        return (self.rep == other.rep) and (self.signs == other.signs)

    def __hash__(self):
        return hash(str(self))

    def to_gatestring(self, pauliDict):
        """ TODO: docstring """
        gstr = []
        sgn = {1:'+', -1:'-'}
        for i,(s,let) in enumerate(zip(self.signs,self.rep)):
            key = sgn[s] + let # e.g. "+X", "-Y", etc
            if key not in pauliDict and s == +1: 
                key = let # try w/out "+"
            if key not in pauliDict:
                raise ValueError("'%s' is not in `pauliDict` (keys = %s)"
                                 % (key,str(list(pauliDict.keys()))))
            gstr.extend( [ _Lbl(gatenm,i) for gatenm in pauliDict[key] ] )
              # pauliDict just has 1Q gate *names* -- need to make into labels
        return _objs.GateString(gstr).parallelize()



class NQPauliOp(object):
    """
    A N-qubit pauli operator, consisting of 
    a 1-qubit gate on each of a set of qubits.
    """

    @classmethod
    def Weight1Pauli(cls, N, i, P ):
        """
        Creates a `N`-qubit Pauli operator with 1-qubit Pauli `P` in location `i`.
        """
        ident = list("I"*N)
        ident[i] =  ["X","Y","Z"][P]
        return cls(''.join(ident))

    @classmethod
    def Weight2Pauli(cls, N, i, j, P1, P2 ):
        """
        Creates a `N`-qubit Pauli operator with Paulis `P1` and `P2` in locations
        `i` and `j` respectively.
        """
        ident = list("I"*N)
        ident[i] =  ["X","Y","Z"][P1]
        ident[j] =  ["X","Y","Z"][P2]
        return cls(''.join(ident))

    def __init__(self, string_rep, sign=1):
        self.rep = string_rep
        self.sign = sign # +/- 1

    def __len__(self):
        return len(self.rep)

    def __str__(self):
        return "%s%s" % ('-' if (self.sign == -1) else ' ',self.rep)

    def __repr__(self):
        return "NQPauliOp[%s%s]" % ('-' if (self.sign == -1) else ' ',self.rep)

    def __eq__(self, other):
        return (self.rep == other.rep) and (self.sign == other.sign)

    def __hash__(self):
        return hash(str(self))

    def subpauli(self, indices ):
        """
        Returns a new N-qubit Pauli which sets all 1-qubit operations to "I"
        except those in `indices`.
        """
        ident = list("I"*len(self.rep))
        for i in indices:
            ident[i]=self.rep[i]
        return NQPauliOp(''.join(ident))
        
    def dot( self, other ):
        """ 
        Computes the Hilbert-Schmidt dot product (normed to 1) between this Pauli
        and `other`.
        """
        assert(len(self) == len(other)), "Length mismatch!"
        if other.rep == self.rep:
            return self.sign * other.sign
        else:
            return 0

    def statedot(self, state):
        # Instead of computing P1*P2 on each Pauli in self (other), it computes P1*(I+P2).
        # (this is only correct if all the Paulis in `other` are *not* I)

        assert(isinstance(state, NQPauliState))
        assert(len(self) == len(state)), "Length mismatch!"

        ret = self.sign # keep track of -1s
        for P1,P2,state_sign in zip(self.rep,state.rep,state.signs):
            if _commute_parity(P1,P2) == -1: return 0 
              # doesn't commute so => P1+P1*P2 = P1+Q = traceless
            elif P1 == 'I': # I*(I+/-P) => (I+/-P) and "sign" of i-th el of state doesn't matter
                pass
            elif state_sign == -1: #  P*(I-P) => (P-I) and so sign (neg[i]) gets moved to I and affects the trace
                assert(P1 == P2)
                ret *= -1
        return ret
            
    def commuteswith(self, other):
        assert(len(self) == len(other)), "Length mismatch!"
        return bool(_np.prod([_commute_parity(P1,P2) for P1,P2 in zip(s1,s2)])==1)

    def icommutatorOver2(self, other):
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

        def Ri_operator(P1,P2):
            """ the *operator* (no sign) part of R = P1*P2 """
            if P1+P2 in ("XY","YX","IZ","ZI"): return "Z"
            if P1+P2 in ("XZ","ZX","IY","YI"): return "Y"
            if P1+P2 in ("YZ","ZY","IX","XI"): return "X"
            if P1+P2 in ("II","XX","YY","ZZ"): return "I"    
            assert(False)

        def Ri_sign(P1,P2,parity):
            """ the +/-1 *sign* part of R = P1*P2 (doesn't count the i-factor in 3rd case)"""
            if parity == 1: return 1 # pass commuteParity(P1,P2) to save computation
            return 1 if P1+P2 in ("XY","YZ","ZX") else -1
            
        assert(len(self) == len(other)), "Length mismatch!"
        s1,s2 = self.rep, other.rep
        parities = [ _commute_parity(P1,P2) for P1,P2 in zip(s1,s2) ]
        if _np.prod(parities) == 1: return None # an even number of minus signs => commutator = 0

        op = ''.join( [Ri_operator(P1,P2) for P1,P2 in zip(s1,s2)] )
        num_i = parities.count(-1) # number of i factors from 3rd Ri case above
        sign = (-1)**((num_i+1)/2) * _np.prod( [Ri_sign(P1,P2,p) for P1,P2,p in zip(s1,s2,parities)] )
        if isinstance(other, NQPauliOp): other_sign = other.sign
        elif isinstance(other, NQPauliState): other_sign = _np.product(other.signs)
        else: raise ValueError("Can't take commutator with %s type" % str(type(other)))

        return NQPauliOp(op, sign * self.sign * other_sign)
