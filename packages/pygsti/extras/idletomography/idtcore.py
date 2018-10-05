from __future__ import print_function, division

import math as _math
import numpy as _np
import itertools as _itertools
import time as _time

import pygsti
import pygsti.construction.nqnoiseconstruction as _nqn
import pygsti.tools.listtools as _lt
from pygsti.baseobjs.label import Label as _Lbl

# This module implements idle tomography, which deals only with
# many-qubit idle gates (on some number of qubits) and single-
# qubit gates (or tensor products of them) used to fiducials.
# As such, it is conventient to represent operations as native
# Python strings, where there is one I,X,Y, or Z letter per
# qubit.


#def _iPauliCommutatorSign( P1, P2 ):
#    """
#    Computes the SIGN of i[P1,P2] -- e.g. i[X,Y] = -Z, so the sign is -1.
#    """
#    if P1+P2 in ["XY","YZ","ZX"]:
#        return -1
#    if P1+P2 in ["YX","ZY","XZ"]:
#        return 1
#    if P1 == "I" or P2 == "I" or P1==P2:
#        return 0
#    assert(False), "bad Paulis "+P1+", "+P2

#def _iPauliCommuteParity( P1, P2 ):
#    """
#    Returns 1 if P1 and P2 commute, or -1 otherwise
#    """
#    if iPauliCommutatorSign(P1,P2) == 0:
#        return 1
#    else:
#        return -1


# iPauliCommutatorDummySign() is an inelegant function that returns +1 *unless* P1 & P2 anticommute 
# *and* have a negative sign.  Nothing elegant about this, but it turns out to be what I need at one point.
#def iPauliCommutatorDummySign( P1, P2 ):
#    if P1+P2 in ["XY","YZ","ZX"]:
#        return -1
#    if P1+P2 in ["YX","ZY","XZ"]:
#        return 1
#    if P1 == "I" or P2 == "I" or P1==P2:
#        return 1
#    print( "bad Paulis "+P1+", "+P2)
#    return 0


#Helper functions
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

    def __eq__(self, other):
        return self.rep == other.rep

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
        return "State[" + "".join(["%s%s" % (sgn[s],let)
                                   for s,let in zip(self.signs,self.rep)]) + "]"

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
        return pygsti.obj.GateString(gstr).parallelize()



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


class IdleTomographyResults(object):
    def __init__(self):
        self.data = {}

        

#REMOVE
## PauliStateDot() is a variation of PauliDot.  Instead of computing P1*P2 on each Pauli in s1 (s2), it computes P1*(I+P2).
## Note that this is only correct if all the Paulis in s1/s2 are *not* I, so it's a hack...
#def PauliStateDot( s1, s2 ):
#    N = len(s1)
#    if N != len(s2):
#        print( "Wrong length!")
#        return 0
#    #assert('I' not in s1),"s1 = %s" % s1
#    assert('I' not in s2),"s2 = %s" % s2
#    if (-1) in [iPauliCommuteParity(s1[i],s2[i]) for i in range(N)]:
#        return 0
#    else:
#        return 1






# HamiltonianMatrixElement() computes the matrix element for a Hamiltonian error Jacobian:
# how the expectation value of "Observable" in state "Prep" changes due to Hamiltonian "Error".
# dp/deps where p = eps * i * Tr(Obs (Err*rho - rho*Err)) = eps * i * ( Tr(Obs Err rho) - Tr(Obs rho Err))
#                 = eps * i * Tr([Obs,Err] * rho)  so dp/deps just drops eps factor
def hamiltonian_jac_element( Prep, Error, Observable):    
    """
    TODO:
    Prep: NQPauliState
    Error: NQPauliOp
    Observable: NQPauliState
    """
    com = Error.icommutatorOver2(Observable)
    return 0 if (com is None) else com.statedot(Prep)


    
# Outcome() defines the N-bit string that results when the N-qubit Pauli basis defined by "PrepMeasBasis"
# is prepared and measured, and "Err" occurs in between.
def stochastic_outcome( prep, error, meas ):
    """ 
    Note: PrepBasis can have different signs than MeasBasis but must be the same Paulis.
    Prep: NQPauliState
    Err: NQPauliOp
    Meas: NQPauliState
    """
    # We can consider each qubit separately, since Tr(A x B) = Tr(A)Tr(B).
    # If for the i-th qubit the prep basis is s1*P and the meas basis is s2*P
    #  (s1 and s2 are the signs -- either +1 or -1 -- and P is the common 
    #  Pauli whose eigenstates 1. form the measurement basis and 2. contain the
    #  state prep), then we're trying to sell which of:
    # Tr( (I+s2*P) Err (I+s1*P) ) ['+' or '0' outcome] OR
    # Tr( (I-s2*P) Err (I+s1*P) ) ['-' or '1' outcome] is nonzero.
    # Combining these two via use of '+-' and expanding gives:
    # Tr( Err + s1* Err P +- s2* P Err +- s1*s2* P Err P )
    #   assuming Err != I so Tr(Err) = 0 and Tr(P Err P) = 0 (b/c P either
    #   commutes or anticommutes w/Err and P^2 == I) =>
    # Tr( s1* Err P +- s2* P Err )
    # if [Err,P] = 0, then the '+'/'0' branch is nonzero when s1==s2
    #   and the '-'/'1' branch is nonzero when s1!=s2
    # if {Err,P} = 0, then the opposite is true: '+'/'0' branch is nonzero
    #   when s1!=s2, etc.
    # Takeaway: if basis (P) commutes with Err then outcome is '0' if s1==s2, "1" otherwise ...
    outcome_str = ""
    for s1,P1,s2,P2,Err in zip(prep.signs, prep.rep, meas.signs, meas.rep, error.rep):
        assert(P1 == P2), "Stochastic outcomes must prep & measure along same bases!"
        P = P1 # ( = P2)
        if _commute_parity(P,Err) == 1: # commutes: [P,Err] == 0
            outcome_str += "0" if (s1 == s2) else "1"
        else: # anticommutes: {P,Err} == 0
            outcome_str += "1" if (s1 == s2) else "0"

    return NQOutcome(outcome_str)


# Now we can define the functions that do the real work for stochastic tomography.

# StochasticMatrixElement() computes the derivative of the probability of "Outcome" with respect
# to the rate of "Error" if the N-qubit Pauli basis defined by "PrepMeas" is prepped and measured.
def stochastic_jac_element( prep, error, meas, outcome ):
    """
    Prep: NQPauliState
    Err: NQPauliOp
    Meas: NQPauliState
    outcome: NQOutcome
    """
    return 1 if (stochastic_outcome(prep,error,meas) == outcome) else 0



# AffineMatrixElement() computes the actual Jacobian element of "OutcomeString" XOR "Mask" due to "Error",
# in the definite-outcome experiment described by PrepMeas.
def affine_jac_element( prep, error, meas, outcome):
    """
    Prep: NQPauliState
    Err: NQPauliOp
    Meas: NQPauliState
    outcome: NQOutcome
    """
    # Note an error of 'ZI' does *not* mean the "ZI affine error":
    #   rho -> (Id[rho] + eps*AffZI[rho]) = rho + eps*ZI
    #   where ZI = diag(1,1,-1,-1), so this adds prob to 00 and 01 and removes from 10 and 11.
    # Instead it means the map AffZ x Id where AffZ : rho -> rho + eps Z and Id : rho -> rho.

    def _affhelper( prepSign, prepBasis, errP, measSign, measBasis, outcome_bit ):
        """
        Answers this question:  
        If a qubit is prepped in state (prepSign,prepBasis) & measured
        using POVM (measSign,measBasis), and experiences an affine error given
        (at this qubit) by Pauli "errP", then at what rate does that change probability of outcome "bit"?
        This is going to get multiplied over all qubits.  A zero indicates that the affine error is orthogonal
        to the measurement basis, which means the probability of *all* outcomes including this bit are unaffected.
        
        Returns 0, +1, or -1.
        """
        # Specifically, this computes Tr( (I+/-P) AffErr[ (I+/-P) ] ) where the two
        # P's represent the prep & measure bases (and can be different).  Here AffErr
        # outputs ErrP if ErrP != 'I', otherwise it's just the identity map (see above).
        #
        # Thus, when ErrP != 'I', we have Tr( (I+/-P) ErrP ) which equals 0 whenever
        # ErrP != P and +/-1 if ErrP == P.  The sign equals measSign when outcome_bit == "0",
        # and is reversed when it == "1".
        # When ErrP == 'I', we have Tr( (I+/-P) (I+/-P) ) = Tr( I + sign*I)
        #  = 1 where sign = prepSign*measSign when outcome == "0" and -1 times
        #      this when == "1".
        #  = 0 otherwise
    
        assert(prepBasis in ("X","Y","Z")) # 'I', for instance, is invalid
        assert(measBasis in ("X","Y","Z")) # 'I', for instance, is invalid
        assert(prepBasis == measBasis) # always true
        outsign = 1 if (outcome_bit == "0") else -1 # b/c we often just flip a sign when == "1"
          # i.e. the sign used in I+/-P for measuring is measSign * outsign
    
        if errP == 'I': # special case: no affine action on this space
            if prepBasis == measBasis:
                return 1 if (prepSign*measSign*outsign == 1) else 0
                #OLD: if outcome_bit == "0":
                #OLD:     return 1 if (prepSign == measSign) else 0
                #OLD: if outcome_bit == "1":
                #OLD:     return 0 if (prepSign == measSign) else 1
            else: return 1 # bases don't match
    
        if measBasis != errP: #then they don't commute (b/c neither can be 'I')
            return 0  # so there's no change along this axis (see docstring)
        else: # measBasis == errP != 'I'            
            if outcome_bit == "0": return measSign
            else: return measSign*-1
            #OLD: if measSign == 1:
            #OLD:     return 1 if (outcome_bit == "0") else -1
            #OLD: else: #flipped basis, so now "0" decreases in probability.
            #OLD:     return -1 if (outcome_bit == "0") else 1    

    return _np.prod( [_affhelper(s1,P1,Err,s2,P2,o) for s1,P1,s2,P2,Err,o
                      in zip(prep.signs, prep.rep, meas.signs, meas.rep,
                             error.rep, outcome.rep)] )

# Computes the Jacobian element of Tr(observable * error * prep) with basis
# convention given by `meas` (dictates sign of outcome).
# (observable should be equal to meas when it's not equal to 'I', up to sign)
def affine_jac_obs_element( prep, error, meas, observable ):
    """
    Prep: NQPauliState
    Err: NQPauliOp
    Meas: NQPauliState
    observable: NQPauliOp
    """
    # Note: as in affine_jac_element, 'I's in error mean that this affine error
    # doesn't act (acts as the identity) on that qubit.

    def _affhelper( prepSign, prepBasis, errP, measSign, measBasis, obsP ):
        assert(prepBasis in ("X","Y","Z")) # 'I', for instance, is invalid
        assert(measBasis in ("X","Y","Z")) # 'I', for instance, is invalid

        # want Tr(obsP * AffErr[ I+/-P ] ).  There are several cases:
        # 1) if obsP == 'I':
        #   - if errP == 'I' (so AffErr = Id), Tr(I +/- P) == 1 always
        #   - if errP != 'I', Tr(ErrP) == 0 since ErrP != 'I'
        # 2) if obsP != 'I' (so Tr(obsP) == 0)
        #   - if errP == 'I', Tr(obsP * (I +/- P)) = prepSign if (obsP == prepBasis) else 0
        #   - if errP != 'I', Tr(obsP * errP) = 1 if (obsP == errP) else 0
        #      (and actually this counts at 2 instead of 1 b/c obs isn't normalized (I think?))

        if obsP == 'I':
            return 1 if (errP == 'I') else 0
        elif errP == 'I':
            assert(prepBasis != measBasis) # I think this is always how this function is called at least for now...
            return prepSign if (prepBasis == obsP) else 0
        else:
            return 2 if (obsP == errP) else 0
    

    #OLD: Unnecessary: Get minus sign due to measurement bases
    #OLD: observed_indices = [ i for i,letter in enumerate(observable.rep) if letter != 'I' ]
    #OLD: minus_sign = _np.prod([meas.signs[i] for i in observed_indices]) # not accounted for below?

    return _np.prod( [_affhelper(s1,P1,Err,s2,P2,o) for s1,P1,s2,P2,Err,o
                      in zip(prep.signs, prep.rep, meas.signs, meas.rep,
                             error.rep, observable.rep)] )


    # REMOVE
    # prod = _np.prod( [_affhelper(s1,P1,Err,s2,P2,'0') for s1,P1,s2,P2,Err
    #                   in zip(prep.signs, prep.rep, meas.signs, meas.rep,
    #                          error.rep)] )
    # prod *= 2**len(observed_indices) #* minus_sign
    # return prod * observable.sign
    # 
    # ret = 1
    # for i in observed_indices:
    #     ret *= 2 * _affhelper(prep.signs[i],prep.rep[i],error.rep[i],
    #                           meas.signs[i],meas.rep[i],"0")
    # return ret * observable.sign * minus_sign # add observable's sign if it had one

    #REMOVE - more scratch
        #w/minus sign if meas has minus sign (using 'obs'
        # is equivalent to ('+' - '-')/2 result, as it is:
        # == ( Tr( (I+obs) * ErrorOp[ prep ] ) - Tr((I-obs) * ErrorOp[ prep ] ) )/2
        # == ( '+' - '-' ) /2  which == ('0'-'1')/2 if meas_sign==1 and == ('1'-'0')/2
        # if meas_sign==-1.
        
        # What if observable has just 1 index but error has 2 observed indices, e.g. if
        # observable = "ZIIZ" and error = "ZZII"?
        # qubit 4: I on error (no error): Tr( Z * (I+/-P) ) = +/- if P == Z (if measuring in same basis as prep)
        # qubit 3: I's on both: Tr( I * I * (I+/-P) ) = 1 regardless of prep/measure basis
        # qubit 2: I on observable: Tr( I * AffZ[ I+/-P ]) = Tr( I * Z ) = 0
        # qubit 1: No I's: Tr( Z * AffZ[ I+/-P ]) = Tr( I ) = 1
        
        # cmp with Tr( (I+/-P) (I+/-P) ) = Tr( I + s1*s2*I) 


    #SCRATCH REMOVE
    ##Get observable using all "+" bases:
    ## <Z> = 0 count - 1 count (if measFid sign is +1, otherwise reversed via minus_sign)
    #if len(observed_indices) == 1: 
    #    i = observed_indices[0] # the qubit we care about
    #    v0 = v1 = 0
    #    #for outcome,cnt in drow.counts.items():
    #    for outcome in alloutcomes(prep, meas, maxErrWeight):
    #        val = affine_jac_element( prep, error, meas, outcome)
    #        if outcome.rep[i] == '0': v0 += val
    #        else: v1 += val
    #    exptn = float(cnt0 - cnt1)/total
    #    
    #    # <ZZ> = 00 count - 01 count - 10 count + 11 count (* minus_sign)
    #    elif len(observed_indices) == 2: 
    #        i,j = observed_indices # the qubits we care about
    #        cnt_even = cnt_odd = 0
    #        for outcome,cnt in drow.counts.items():
    #            if outcome[0][i] == outcome[0][j]: cnt_even += cnt
    #            else: cnt_odd += cnt
    #        exptn = float(cnt_even - cnt_odd)/total
    #        fp = 0.5 + 0.5*float(cnt_even - cnt_odd + 1)/(total + 2)
    #    else:
    #        raise NotImplementedError("Expectation values of weight > 2 observables are not implemented!")



#Experiment generation: HERE
# we want a structure for the gate sequences that holds separately the prep & meas fiducials
# b/c when actually doing idle tomography we don't want to rely on a particular set of fiducial
# pairs (since this is non-unique) for Hamiltonian, Stochastic, etc.
# 
# Structure:
# do_idle_tomography(X-type-fidpairs, max-err-weight=2, nQubits, ...):
#   for all X-type experiments(X-type-fidpairs, max-err-weight, nQubits): # "tiles" fidpairs to n qubits?
#     for all outcomes/observables(max-err-weight, ?)
#       for all errors(max-err-weight, ?)
#         get_X_matrix_element(error, fidpair, outcome)
#       get_obs_X_err_rate(fidpair, outcome/observable, ...)


def alloutcomes(prep, meas, maxweight):
    """
    Lists every "error bit string" that could be caused by an error of weight
    up to `maxweight` when performing prep & meas (must be in same basis, but may 
    have different signs).

    prep, meas : NQPauliState
    """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweight <= 2 is currently supported")
    assert(prep.rep == meas.rep), "`prep` and `meas` must specify the same basis!"
    expected = ["0" if s1==s2 else "1" for s1,s2 in zip(prep.signs,meas.signs)]
      #whether '0' or '1' outcome is expected, i.e. what is an "error"

    N = len(prep) # == len(meas)
    eoutcome = NQOutcome(''.join(expected))
    if maxweight == 1:
        return [eoutcome.flip(i) for i in range(N)]
    else:
        return [eoutcome.flip(i) for i in range(N)] + \
               [eoutcome.flip(i,j) for i in range(N) for j in range(i+1,N)]


def allerrors(N, maxweight):
    """ lists every Pauli error for N qubits with weight <= `maxweight` """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweigth <= 2 is currently supported")
    if maxweight == 1:
        return [NQPauliOp.Weight1Pauli(N,loc,p) for loc in range(N) for p in range(3)]
    else:
        return [NQPauliOp.Weight1Pauli(N,loc,p) for loc in range(N) for p in range(3)] + \
               [NQPauliOp.Weight2Pauli(N,loc1,loc2,p1,p2) for loc1 in range(N) 
                for loc2 in range(loc1+1,N)
                for p1 in range(3) for p2 in range(3)]

def allobservables( meas, maxweight ):
    """
    Lists every weight <= `maxweight` observable whose expectation value can be
    extracted from the local Pauli measurement described by "Meas".

    Meas: NQPauliState (*not* precisely an N-qubit Pauli -- a list of 1-qubit Paulis, technically)
    """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweight <= 2 is currently supported")
    #Note: returned observables always have '+' sign (i.e. .sign == +1).  We're
    # not interested in meas.signs - this is take into account when we compute the
    # expectation value of our observable given a prep & measurement fiducial.
    if maxweight == 1:
        return [NQPauliOp(meas.rep).subpauli([i]) for i in range(len(meas))]
    else:
        return [NQPauliOp(meas.rep).subpauli([i]) for i in range(len(meas))] + \
               [NQPauliOp(meas.rep).subpauli([i,j]) for i in range(len(meas)) for j in range(i+1,len(meas))]

#HERE - more TODO

#Use NQPauliState.to_gatestring()
#def paulibasisletters_to_gatestring(letters,typ):
#    gstr = []
#    for i,letter in enumerate(letters):
#        gstr.extend( basis_to_fiducial(letter,i,typ))
#    return pygsti.obj.GateString(gstr)

def tile_pauli_fidpairs(base_fidpairs, nQubits, maxweight):
    """ 
    TODO: docstring - note els of "base_fidpairs" are 2-tuples of NQPauliState
    objects of maxweight qubits, so sign of basis is included too.
    """
    nqubit_fidpairs = []
    tmpl = _nqn.get_kcoverage_template(nQubits, maxweight)
    for base_prep,base_meas in base_fidpairs:
        for tmpl_row in tmpl:
            #Replace 0...weight-1 integers in tmpl_row with Pauli basis
            # designations (e.g. +X) to construct NQPauliState objects.
            prep = NQPauliState( [ base_prep.rep[i] for i in tmpl_row ],
                                 [ base_prep.signs[i] for i in tmpl_row] )
            meas = NQPauliState( [ base_meas.rep[i] for i in tmpl_row ],
                                 [ base_meas.signs[i] for i in tmpl_row] )
            nqubit_fidpairs.append((prep,meas))

    _lt.remove_duplicates_in_place(nqubit_fidpairs)
    return nqubit_fidpairs




#Scratch REMOVE
#  prepDict & measDict are dicts w/keys like "+X","X" or "-X", etc. and values
#  are gatename lists, e.g. ('Gx','Gx') without any qubit index.
#    prepDict,measDict = pauliDicts
#    fidpairs = [ (x.to_gatestring(prepDict),x.to_gatestring(measDict))
#                 for x,y in base_fidpairs ] # e.g. convert ("XY","ZX") to tuple of GateStrings
#    gatename_fidpair_lists = fidpairs_to_gatename_fidpair_list(fidpairs, nQubits)
#    return _nqn.tile_idle_fidpairs(nQubits, gatename_fidpair_lists, maxweight) # returns *GateStrings*!


def idle_tomography_fidpairs(nQubits, maxweight=2, include_hamiltonian=True,
                             include_stochastic=True, include_affine=True,
                             ham_tmpl=("ZY","ZX","XZ","YZ","YX","XY"),
                             preferred_prep_basis_signs=("+","+","+"),
                             preferred_meas_basis_signs=("+","+","+") ):
    """ TODO: docstring 
    Returns a list of 2-tuples of NQPauliState objects of length `nQubits`
    representing fiducial pairs.
    """
#                             sto_tmpl=("X","Y","Z") # could be "-X", etc. ("XX","YY","ZZ","ZY","ZX","XZ","YZ","YX","XY"),
#                             aff_tmpl=??, maxweight=2):


    fidpairs = [] # list of 2-tuples of NQPauliState objects to return

    #convert +'s and -'s to dictionaries of +/-1 used later:
    conv = lambda x: 1 if x=="+" else -1
    base_prep_signs = { l:conv(s) for l,s in zip(('X','Y','Z'), preferred_prep_basis_signs) }
    base_meas_signs = { l:conv(s) for l,s in zip(('X','Y','Z'), preferred_meas_basis_signs) }
      #these dicts give the preferred sign for prepping or measuring along each 1Q axis.

    if include_stochastic:
        if include_affine:
            # in general there are 2^maxweight different permutations of +/- signs
            # in maxweight==1 case, need 2 of 2 permutations
            # in maxweight==2 case, need 3 of 4 permutations
            # higher maxweight?

            if maxweight == 1:
                flips = [ (1,), (-1,) ] # consider both cases of not-flipping & flipping the preferred basis signs

            elif maxweight == 2:
                flips = [ (1,1), # don't flip anything
                          (1,-1), (-1,1) ] #flip 2nd or 1st pauli basis (weight = 2)
            else:
                raise NotImplementedError("No implementation for affine errors and maxweight > 2!")
                #need to do more work to figure out how to generalize this to maxweight > 2
        else:
            flips = [ (1,)*maxweight ] # don't flip anything

        #Build up "template" of 2-tuples of NQPauliState objects acting on 
        # maxweight qubits that should be tiled to full fiducial pairs.
        sto_tmpl_pairs = []
        for fliptup in flips: # elements of flips must have length=maxweight

            # Create a set of "template" fiducial pairs using the current flips
            for basisLets in _itertools.product(('X','Y','Z'), repeat=maxweight):

                # flip base (preferred) basis signs as instructed by fliptup
                prepSigns = [ f*base_prep_signs[l] for f,l in zip(fliptup,basisLets)]
                measSigns = [ f*base_meas_signs[l] for f,l in zip(fliptup,basisLets)]
                sto_tmpl_pairs.append( (NQPauliState(''.join(basisLets), prepSigns),
                                        NQPauliState(''.join(basisLets), measSigns)) )

        fidpairs.extend( tile_pauli_fidpairs(sto_tmpl_pairs, nQubits, maxweight) )

    elif include_affine:
        raise ValueError("Cannot include affine sequences without also including stochastic ones!")


    if include_hamiltonian:

        nextPauli = {"X":"Y","Y":"Z","Z":"X"}
        prevPauli = {"X":"Z","Y":"X","Z":"Y"}
        def prev( expt ): return ''.join([prevPauli[p] for p in expt])
        def next( expt ): return ''.join([nextPauli[p] for p in expt])

        ham_tmpl_pairs = []
        for tmplLets in ham_tmpl: # "Lets" = "letters", i.e. 'X', 'Y', or 'Z'
            prepLets, measLets = prev(tmplLets), next(tmplLets)

            # basis sign doesn't matter for hamiltonian terms, 
            #  so just use preferred signs
            prepSigns = [ base_prep_signs[l] for l in prepLets]
            measSigns = [ base_meas_signs[l] for l in measLets]
            ham_tmpl_pairs.append( (NQPauliState(prepLets, prepSigns),
                                    NQPauliState(measLets, measSigns)) )
            
        fidpairs.extend( tile_pauli_fidpairs(ham_tmpl_pairs, nQubits, maxweight) )

    return fidpairs


def preferred_signs_from_paulidict(pauliDict):
    """ 
    Infers what the preferred basis signs are based on what is available
    in `pauliDict`, a dictionary w/keys like "+X" or "-Y" and values that
    are tuples of gate names.
    TODO: docstring
    """
    preferred_signs = ()
    for let in ('X','Y','Z'):
        if "+"+let in pauliDict: plusKey = "+"+let
        elif let in pauliDict: plusKey = let
        else: plusKey = None

        if "-"+let in pauliDict: minusKey = '-'+let
        else: minusKey = None

        if minusKey and plusKey:
            if len(pauliDict[plusKey]) <= len(pauliDict[minusKey]):
                preferred_sign = '+'
            else:
                preferred_sign = '-'
        elif plusKey:
            preferred_sign = '+'
        elif minusKey:
            preferred_sign = '-'
        else:
            raise ValueError("No entry for %s-basis!" % let)

        preferred_signs += (preferred_sign,)

    return preferred_signs


def make_idle_tomography_list(nQubits, pauliDicts, maxLengths, maxErrWeight=2,
                              includeHamSeqs=True, includeStochasticSeqs=True, includeAffineSeqs=True,
                              ham_tmpl=("ZY","ZX","XZ","YZ","YX","XY"), 
                              preferred_prep_basis_signs="auto",
                              preferred_meas_basis_signs="auto",
                              GiStr = ('Gi',)):

    prepDict,measDict = pauliDicts
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)
        
    GiStr = pygsti.obj.GateString( GiStr )

    pauli_fidpairs = idle_tomography_fidpairs(
        nQubits, maxErrWeight, includeHamSeqs, includeStochasticSeqs,
        includeAffineSeqs, ham_tmpl, preferred_prep_basis_signs,
        preferred_meas_basis_signs)

    fidpairs = [ (x.to_gatestring(prepDict),y.to_gatestring(measDict))
                 for x,y in pauli_fidpairs ] # e.g. convert ("XY","ZX") to tuple of GateStrings

    listOfExperiments = []
    for prepFid,measFid in fidpairs: #list of fidpairs / configs (a prep/meas that gets I^L placed btwn it)
        for L in maxLengths:
            listOfExperiments.append( prepFid + GiStr*L + measFid )
                
    return listOfExperiments


# Finally, HamiltonianExptList() generates a list of experiments for detecting Hamiltonian errors.
#def HamiltonianExptList( N ):
#    if N==1:
#        return ["X","Y","Z"]
#    else:
#        return ExtendPauliPairListToN( ["ZY","ZX","XZ","YZ","YX","XY"], N)
#
#
## Stochastic errors affect definite-outcome circuits, so we work with effects and their probabilities
## instead of traceless observables and their expectation values.
## These are some functions that play the same role as the simple Weight-1&2 Pauli functions used earlier
#
## StochasticExptList() provides a list of all the experiments (prep and measurement are in the same Pauli basis)
## required to diagnose stochastic Pauli errors
#def StochasticExptList( N ):
#    if N==1:
#        return ["X","Y","Z"]
#    else:
#        return ExtendPauliPairListToN( ["XX","YY","ZZ","ZY","ZX","XZ","YZ","YX","XY"], N)
#
## AffineExptList() generates a list of experiments that is sufficient for diagnosing affine errors.  There
## are 3x as many as stochastic -- unfortunately, it's not sufficient to just flip every experiment, nor
## to just flip one bit in each pair.  We need to do 3 out of 4 of { ZZ, ZW, WZ, WW} in order to separate all affines.
## I have chosen to do {ZZ, ZW, WZ}.
#def AffineExptList( N ):
#    Plist = ["X","Y","Z"]
#    XPlist = ["U","V","W"]
#    if N==1:
#        return Plist+XPlist
#    else:
#        Pairlist = [a+b for a in Plist for b in Plist] + \
#                   [a+b for a in Plist for b in XPlist] + \
#                   [a+b for a in XPlist for b in Plist]
#        return ExtendPauliPairListToN( Pairlist, N)



# Running idle tomography

# NEEDED?
def nontrivial_paulis(wt): 
    "List all nontrivial paulis of given weight `wt` as tuples of letters"
    ret = []
    for tup in _itertools.product( *([['X','Y','Z']]*wt) ):
        ret.append( tup )
    return ret

def set_Gi_errors(nQubits, gateset, errdict, rand_default=None, hamiltonian=True, stochastic=True, affine=True):
    """ 
    For setting specific or random error terms (for a data-generating gateset)
    within a `gateset` created by `build_nqnoise_gateset`.
    """
    rand_rates = []; i_rand_default = 0
    v = gateset.to_vector()
    for i,factor in enumerate(gateset.gates['Gi'].factorgates): # each factor applies to some set of the qubits (of size 1 to the max-error-weight)
        #print("Factor %d: target = %s, gpindices=%s" % (i,str(factor.targetLabels),str(factor.gpindices)))
        assert(isinstance(factor, pygsti.objects.EmbeddedGateMap)), "Expected Gi to be a composition of embedded gates!"
        sub_v = v[factor.gpindices]
        bsH = factor.embedded_gate.ham_basis_size
        bsO = factor.embedded_gate.other_basis_size
        if hamiltonian: hamiltonian_sub_v = sub_v[0:bsH-1] # -1s b/c bsH, bsO include identity in basis
        if stochastic:  stochastic_sub_v = sub_v[bsH-1:bsH-1+bsO-1]
        if affine:      affine_sub_v = sub_v[bsH-1+bsO-1:bsH-1+2*(bsO-1)]

        for k,tup in enumerate(nontrivial_paulis( len(factor.targetLabels) )):
            lst = ['I']*nQubits
            for ii,i in enumerate(factor.targetLabels):
                lst[int(i[1:])] = tup[ii] # i is something like "Q0" so int(i[1:]) extracts the 0
            label = "".join(lst)

            if "S(%s)" % label in errdict:
                Srate = errdict["S(%s)" % label]
            elif rand_default is None:
                Srate = 0.0
            elif isinstance(rand_default,float):
                Srate = rand_default *_np.random.random()
                rand_rates.append(Srate)
            else: #assume rand_default is array-like, and gives default rates
                Srate = rand_default[i_rand_default]
                i_rand_default += 1
                
            if "H(%s)" % label in errdict:
                Hrate = errdict["H(%s)" % label]
            elif rand_default is None:
                Hrate = 0.0
            elif isinstance(rand_default,float):
                Hrate = rand_default *_np.random.random()
                rand_rates.append(Hrate)
            else: #assume rand_default is array-like, and gives default rates
                Hrate = rand_default[i_rand_default]
                i_rand_default += 1

            if "A(%s)" % label in errdict:
                Arate = errdict["A(%s)" % label]
            elif rand_default is None:
                Arate = 0.0
            elif isinstance(rand_default,float):
                Arate = rand_default *_np.random.random()
                rand_rates.append(Arate)
            else: #assume rand_default is array-like, and gives default rates
                Arate = rand_default[i_rand_default]
                i_rand_default += 1

            if hamiltonian: hamiltonian_sub_v[k] = Hrate
            if stochastic: stochastic_sub_v[k] = _np.sqrt(Srate) # b/c param gets squared
            if affine: affine_sub_v[k] = Arate
            
    gateset.from_vector(v)
    return _np.array(rand_rates,'d') # the random rates that were chosen (to keep track of them for later)


def predicted_intrinsic_rates(nQubits, maxErrWeight, gateset, hamiltonian=True, stochastic=True, affine=True):
    #Get rates Compare datagen to idle tomography results
    error_labels = [str(pauliOp.rep) for pauliOp in allerrors(nQubits, maxErrWeight)]
    v = gateset.to_vector()
    
    if hamiltonian:
        ham_intrinsic_rates = _np.zeros(len(error_labels),'d')
    else: ham_intrinsic_rates = None
    
    if stochastic:
        sto_intrinsic_rates = _np.zeros(len(error_labels),'d')
    else: sto_intrinsic_rates = None

    if affine:
        aff_intrinsic_rates = _np.zeros(len(error_labels),'d')
    else: aff_intrinsic_rates = None

    for i,factor in enumerate(gateset.gates['Gi'].factorgates):
        #print("Factor %d: target = %s, gpindices=%s" % (i,str(factor.targetLabels),str(factor.gpindices)))
        assert(isinstance(factor, pygsti.objects.EmbeddedGateMap)), "Expected Gi to be a composition of embedded gates!"
        sub_v = v[factor.gpindices]
        bsH = factor.embedded_gate.ham_basis_size
        bsO = factor.embedded_gate.other_basis_size
        if hamiltonian: hamiltonian_sub_v = sub_v[0:bsH-1] # -1s b/c bsH, bsO include identity in basis
        if stochastic:  stochastic_sub_v = sub_v[bsH-1:bsH-1+bsO-1]
        if affine:      affine_sub_v = sub_v[bsH-1+bsO-1:bsH-1+2*(bsO-1)]
        
        for k,tup in enumerate(nontrivial_paulis( len(factor.targetLabels) )):
            lst = ['I']*nQubits
            for ii,i in enumerate(factor.targetLabels):
                lst[int(i[1:])] = tup[ii] # i is something like "Q0" so int(i[1:]) extracts the 0
            label = "".join(lst)
            if stochastic:  sval = stochastic_sub_v[k]
            if hamiltonian: hval = hamiltonian_sub_v[k]
            if affine:      aval = affine_sub_v[k]
            
            nTargetQubits = len(factor.targetLabels)
            
            if stochastic:
                # each Stochastic term has two Paulis in it (on either side of rho), each of which is
                # scaled by 1/sqrt(d), so 1/d in total, where d = 2**nQubits
                sscaled_val = sval**2 / (2**nTargetQubits) # val**2 b/c it's a *stochastic* term parameter
            
            if hamiltonian:
                # each Hamiltonian term, to fix missing scaling factors in Hamiltonian jacobian
                # elements, needs a sqrt(d) for each trivial ('I') Pauli... ??
                hscaled_val = hval * _np.sqrt(2**(2-nTargetQubits)) # TODO: figure this out...
                # 1Q: sqrt(2) 
                # 2Q: nqubits-targetqubits (sqrt(2) on 1Q)
                # 4Q: sqrt(2)**-2

            if affine:
                ascaled_val = aval * 1/(_np.sqrt(2)**nTargetQubits ) # not exactly sure how this is derived
                # 1Q: sqrt(2)/6
                # 2Q: 1/3 * 10-2

            result_index = error_labels.index(label)
            if hamiltonian: ham_intrinsic_rates[result_index] = hscaled_val
            if stochastic:  sto_intrinsic_rates[result_index] = sscaled_val
            if affine:      aff_intrinsic_rates[result_index] = ascaled_val

    return ham_intrinsic_rates, sto_intrinsic_rates, aff_intrinsic_rates



#This should be an *argument* (or automatic? - but can't unless we're given the gate matrices/reps, which
# we don't otherwise need) given to sequence gen functions - how to convert pauli-letters to fiducials,
# or, more precisely, how to associate fiducials with NQPauliState objects.
#def basis_to_fiducial(pauliLetter,i,typ):
#    rep = 1 if (typ == "prep") else 1 # FUTURE (maybe w/affine we'll need to make some prep/meaures 3 reps)
#    if pauliLetter == 'X': return (('Gy',i),)*rep # Gy preps in X state & measures in x-basis (when followed by Z POVM)
#    if pauliLetter == 'Y': return (('Gx',i),)*rep
#    if pauliLetter == 'Z': return () # native prep/meas is in Z-basis
#    assert(False)

    


def get_obs_stochastic_err_rate(dataset, pauli_fidpair, pauliDicts, GiStr, outcome, maxLengths, fitOrder=1):
    """
    TODO: docstring
    Returns a dict of info including observed error rate, data points that were fit, etc.
    """
    # fit number of given outcome counts to a line
    pauli_prep, pauli_meas = pauli_fidpair

    prepDict,measDict = pauliDicts
    prepFid = pauli_prep.to_gatestring(prepDict)
    measFid = pauli_meas.to_gatestring(measDict)
    
    #Note on weights: 
    # data point with frequency f and N samples should be weighted w/ sqrt(N)/sqrt(f*(1-f))
    # but in case f is 0 or 1 we use proxy f' by adding a dummy 0 and 1 count.
    def freq_and_weight(gatestring,outcome):
        cnts = dataset[gatestring].counts # a normal dict
        total = sum(cnts.values())
        f = cnts.get((outcome.rep,),0) / total # (py3 division) NOTE: outcomes are actually 1-tuples 
        fp = (cnts.get((outcome.rep,),0)+1)/ (total+2) # Note: can't == 1
        wt = _np.sqrt(total) / _np.sqrt(abs(fp*(1.0-fp))) # abs to deal with non-CP data (simulated using termorder:1)
        return f,wt

    #Get data to fit and weights to use in fitting
    data_to_fit = []; wts = []
    for L in maxLengths:
        gstr = prepFid + GiStr*L + measFid
        f,wt = freq_and_weight(gstr, outcome)
        data_to_fit.append( f )
        wts.append( wt )
        
    #curvefit -> slope
    coeffs = _np.polyfit(maxLengths,data_to_fit,fitOrder,w=wts) # when fitOrder = 1 = line
    if fitOrder == 1:
        slope = coeffs[0]
    elif fitOrder == 2:
        slope =  coeffs[1] # c2*x2 + c1*x + c0 ->deriv@x=0-> c1
    else: raise NotImplementedError("Only fitOrder <= 2 are supported!")
    
    #REMOVE - maybe use the description elsewhere?
    #if saved_fits is not None:
    #    desc = "Sto: fidpair=%s outcome=%s" % (str(pauli_fidpair), str(unfixed_outcome)) # so outcome is mostly 0s...
    #    saved_fits.append( (desc, maxLengths, data_to_fit, coeffs, debug_obs_rate) ) # description, Xs, Ys, fit_coeffs
    #print("FIT (%s,%s) %s: " % (str(prepFid),str(measFid), outcome.rep),maxLengths,data_to_fit," -> slope = ",slope)
    
    return { 'rate': slope, 'fitOrder': fitOrder, 'fitCoeffs': coeffs, 'data': data_to_fit, 'weights': wts }


def get_obs_hamiltonian_err_rate(dataset, pauli_fidpair, pauliDicts, GiStr, observable, maxLengths, fitOrder=1):
    """TODO: docstring """
    # fit expectation value of `observable` (trace over all I elements of it) to a line
    pauli_prep, pauli_meas = pauli_fidpair

    prepDict,measDict = pauliDicts
    prepFid = pauli_prep.to_gatestring(prepDict)
    measFid = pauli_meas.to_gatestring(measDict)
    
    #observable is always equal to pauli_meas (up to signs) with all but 1 or 2
    # (maxErrWt in general) of it's elements replaced with 'I', essentially just
    # telling us which 1 or 2 qubits to take the <Z> or <ZZ> expectation value of
    # (since the meas fiducial gets us in the right basis) -- i.e. the qubits to *not* trace over.
    obs_indices = [ i for i,letter in enumerate(observable.rep) if letter != 'I' ]
    N = len(observable) # number of qubits
    minus_sign = _np.prod([pauli_meas.signs[i] for i in obs_indices])
    
    #REMOVE
    ## any preps in "Y" or measures in "X" basis generate minus signs
    ## since we just use single "Gy" and "Gx" gates for this (instead of 3)
    #minus_sign_cnt = 0 
    #for i in obs_indices:
    #    #if pauli_prep[i] == 'Y': minus_sign_cnt += 1 # taken care of separately when computing Jacobian element
    #    if pauli_meas[i] == 'X': minus_sign_cnt += 1
    ##print("DB: ",obs_indices, pauli_prep, pauli_meas, minus_sign_cnt)

    def unsigned_exptn_and_weight(gatestring, observed_indices):
        #compute expectation value of observable
        drow = dataset[gatestring] # dataset row
        total = drow.total

        # <Z> = 0 count - 1 count (if measFid sign is +1, otherwise reversed via minus_sign)
        if len(observed_indices) == 1: 
            i = observed_indices[0] # the qubit we care about
            cnt0 = cnt1 = 0
            for outcome,cnt in drow.counts.items():
                if outcome[0][i] == '0': cnt0 += cnt # [0] b/c outcomes are actually 1-tuples 
                else: cnt1 += cnt
            exptn = float(cnt0 - cnt1)/total
            fp = 0.5 + 0.5*float(cnt0 - cnt1 + 1)/(total + 2)
        
        # <ZZ> = 00 count - 01 count - 10 count + 11 count (* minus_sign)
        elif len(observed_indices) == 2: 
            i,j = observed_indices # the qubits we care about
            cnt_even = cnt_odd = 0
            for outcome,cnt in drow.counts.items():
                if outcome[0][i] == outcome[0][j]: cnt_even += cnt
                else: cnt_odd += cnt
            exptn = float(cnt_even - cnt_odd)/total
            fp = 0.5 + 0.5*float(cnt_even - cnt_odd + 1)/(total + 2)
        else:
            raise NotImplementedError("Expectation values of weight > 2 observables are not implemented!")
            
        wt =_np.sqrt(total) / _np.sqrt(fp*(1.0-fp))
        return exptn, wt
    
    
    #Get data to fit and weights to use in fitting
    data_to_fit = []; wts = []
    for L in maxLengths:
        gstr = prepFid + GiStr*L + measFid
        exptn,wt = unsigned_exptn_and_weight(gstr, obs_indices)
        data_to_fit.append( minus_sign * exptn )
        wts.append( wt )
        
    #curvefit -> slope 
    coeffs = _np.polyfit(maxLengths,data_to_fit,fitOrder,w=wts) # when fitOrder = 1 = line
    if fitOrder == 1:
        slope = coeffs[0]
    elif fitOrder == 2:
        slope =  coeffs[1] # c2*x2 + c1*x + c0 ->deriv@x=0-> c1
    else: raise NotImplementedError("Only fitOrder <= 2 are supported!")
    
    #REMOVE
    #if saved_fits is not None:
    #    desc = "Ham: fidpair=%s obs=%s" % (str(pauli_fidpair), str(observable))
    #    saved_fits.append( (desc, maxLengths, data_to_fit, coeffs, (-1)**minus_sign_cnt * debug_obs_rate) ) # description Xs, Ys, fit_coeffs
    #print("FIT (%s,%s): " % (str(prepFid),str(measFid)),maxLengths,data_to_fit," -> slope = ",slope, " sign=", minus_sign)
    
    return { 'rate': slope, 'fitOrder': fitOrder, 'fitCoeffs': coeffs, 'data': data_to_fit, 'weights': wts }


def do_idle_tomography(nQubits, dataset, maxLengths, pauliDicts, maxErrWeight=2, GiStr=('Gi',),
                       extract_hamiltonian=True, extract_stochastic=True, extract_affine=True,
                       advancedOptions=None, comm=None, verbosity=0):
#HERE
# pauliDict, hamFidPairs, stoFidPairs
#fitOrder=1, extract_stochastic=True,
# extract_hamiltonian=True, saved_fits=None, debug_true_obs_rates=None,
    if advancedOptions is None: 
        advancedOptions = {}

    result = IdleTomographyResults()
    prepDict,measDict = pauliDicts
    GiStr = pygsti.obj.GateString( GiStr )
                       
    #idebug = 0
    rankStr = "" if (comm is None) else "Rank%d: " % comm.Get_rank()

    preferred_prep_basis_signs = advancedOptions.get('preferred_prep_basis_signs', 'auto')
    preferred_meas_basis_signs = advancedOptions.get('preferred_meas_basis_signs', 'auto')
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    errors = allerrors(nQubits,maxErrWeight)
    fitOrder = advancedOptions.get('fit order',1)

    if extract_stochastic:
        tStart = _time.time()
        pauli_fidpairs = idle_tomography_fidpairs(
            nQubits, maxErrWeight, False, extract_stochastic, extract_affine,
            advancedOptions.get('ham_tmpl', ("ZY","ZX","XZ","YZ","YX","XY")),
            preferred_prep_basis_signs, preferred_meas_basis_signs)

        #divide up strings among ranks
        indxFidpairList = list(enumerate(pauli_fidpairs))
        my_FidpairList, _,_ = pygsti.tools.mpitools.distribute_indices(indxFidpairList,comm,False)

        my_J = []; my_obs_infos = []
        #REM my_obs_err_rates = []
        #REM my_saved_fits = [] if (saved_fits is not None) else None
        for i,(ifp,pauli_fidpair) in enumerate(my_FidpairList):
            #REM pauli_fidpair = (expt,expt) # Stochastic: prep & measure in same basis
            #NOTE: pauli_fidpair is a 2-tuple of NQPauliState objects
            
            all_outcomes = alloutcomes(pauli_fidpair[0], pauli_fidpair[1], maxErrWeight)
            #REM debug_offset = iexpt * len(all_outcomes)
            t0 = _time.time()            
            for j,out in enumerate(all_outcomes):

                if verbosity > 1: print("  - outcome %d of %d" % (j,len(all_outcomes)))

                #REM def rev(x): return '0' if (x == '1') else '1'  # flips '0' <-> '1'
                #REM fixed_out = "".join([ (rev(out[ii]) if expt[ii] in ('X','Y') else out[ii]) for ii in range(len(out)) ])
                
                #form jacobian rows as we get extrinsic error rates
                Jrow = [ stochastic_jac_element( pauli_fidpair[0], err, pauli_fidpair[1], out )
                         for err in errors ]
                if extract_affine:
                    Jrow.extend( [ affine_jac_element( pauli_fidpair[0], err, pauli_fidpair[1], out )
                                   for err in errors ] )
                my_J.append(Jrow)

                #idebug = debug_offset + j
                #debug = debug_true_obs_rates[idebug] if (debug_true_obs_rates is not None) else None #; idebug += 1
                info = get_obs_stochastic_err_rate(dataset,pauli_fidpair,pauliDicts,GiStr,
                                                           out,maxLengths,fitOrder)

                #T=len(Jrow); print("(%s,%s) JAC ROW = " % (str(pauli_fidpair[0]),str(pauli_fidpair[1])),Jrow[:T//2],Jrow[T//2:])
                my_obs_infos.append(info)
                #print("Expt %s(%s): err-rate=%g" % (expt,out,obs_err_rate))
                #print("Jrow = ",Jrow)

            if verbosity > 0: print("%sStochastic fidpair %d of %d: %d outcomes analyzed [%.1fs]" % 
                                    (rankStr, i,len(my_ExptList),len(all_outcomes),_time.time()-t0))

        #Gather results
        info_list = [ my_obs_infos ] if (comm is None) else comm.gather(my_obs_infos, root=0)
        J_list = [ my_J ] if (comm is None) else comm.gather(my_J, root=0)

        #REMOVE
        #if saved_fits is not None:
        #    saved_fit_list = [ my_saved_fits ] if (comm is None) else comm.gather(my_saved_fits, root=0)
        #    if comm is None or comm.Get_rank() == 0:
        #        for lst in saved_fit_list: saved_fits.extend(lst)

        if comm is None or comm.Get_rank() == 0:
            # pseudo-invert J to get "intrinsic" error rates (labeled by AllErrors(nQubits))
            # J*intr = obs
            J = _np.concatenate(J_list, axis=0)
            info_list = list(_itertools.chain(*info_list)) # flatten ~ concatenate

            obs_err_rates = _np.array([info['rate'] for info in info_list])
            invJ = _np.linalg.pinv(J)
            intrinsic_stochastic_rates = _np.dot(invJ,obs_err_rates)
            print("INVJ = ",_np.linalg.norm(invJ))
            print("OBS = ",_np.linalg.norm(obs_err_rates))
            result.data['Stochastic error names'] = errors # "key" to intrinsic rates

            if extract_affine:
                Nrates = len(intrinsic_stochastic_rates)
                result.data['Intrinsic stochastic rates'] = intrinsic_stochastic_rates[0:Nrates//2]
                result.data['Intrinsic affine rates'] = intrinsic_stochastic_rates[Nrates//2:]
                print("STO RATES = %g" % _np.linalg.norm(intrinsic_stochastic_rates[:Nrates//2]))
                print("AFFINE RATES = %g" % _np.linalg.norm(intrinsic_stochastic_rates[Nrates//2:]))
            else:
                result.data['Intrinsic stochastic rates'] = intrinsic_stochastic_rates
                print("STO RATES = %g" % _np.linalg.norm(intrinsic_stochastic_rates))
            result.data['Stochastic/Affine fidpairs'] = pauli_fidpairs # give "key" to observed rates
            result.data['Observed stochastic/affine infos'] = info_list
            print("Completed Stochastic/Affine in %.2fs" % (_time.time()-tStart))

    elif extract_affine:
        raise ValueError("Cannot extract affine error rates without also extracting stochastic ones!")

    if extract_hamiltonian:
        tStart = _time.time()
        pauli_fidpairs = idle_tomography_fidpairs(
            nQubits, maxErrWeight, extract_hamiltonian, False, False,
            advancedOptions.get('ham_tmpl', ("ZY","ZX","XZ","YZ","YX","XY")),
            preferred_prep_basis_signs, preferred_meas_basis_signs)

        #divide up fiducial pairs among ranks
        indxFidpairList = list(enumerate(pauli_fidpairs))
        my_FidpairList, _,_ = pygsti.tools.mpitools.distribute_indices(indxFidpairList,comm,False)

        my_J = []; my_obs_infos = []; my_Jaff = []
        for i,(ifp,pauli_fidpair) in enumerate(my_FidpairList):
            #REM pauli_fidpair = (Prep(expt),Meas(expt))
            all_observables = allobservables( pauli_fidpair[1], maxErrWeight )

            #REM debug_offset = iexpt * len(all_observables) # all_observables is the same on every loop (TODO FIX)
            t0 = _time.time()            
            for j,obs in enumerate(all_observables):
                if verbosity > 1: print("  - observable %d of %d" % (j,len(all_observables)))

                #REM negs = [ bool(pauli_fidpair[0][ii] == 'Y') for ii in range(nQubits) ] 
                #REM   b/c use of 'Gx' for Y-basis prep really prepares -Y state
        
                #form jacobian rows as we get extrinsic error rates
                Jrow = [ hamiltonian_jac_element(pauli_fidpair[0], err, obs) for err in errors ]
                my_J.append(Jrow)

                # J_ham * Hintrinsic + J_aff * Aintrinsic = observed_rates, and Aintrinsic is known
                #  -> need to find J_aff, the jacobian of *observable expectation vales* w/affine params.
                if extract_affine:
                    Jaff_row = [ affine_jac_obs_element( pauli_fidpair[0], err, pauli_fidpair[1], obs )
                                   for err in errors ]
                    my_Jaff.append(Jaff_row)

                #REM idebug = debug_offset + j
                #REM debug = debug_true_obs_rates[idebug] if (debug_true_obs_rates is not None) else None #; idebug += 1
                info = get_obs_hamiltonian_err_rate(dataset, pauli_fidpair, pauliDicts, GiStr, obs,
                                                            maxLengths, fitOrder)
                my_obs_infos.append(info)
                #print("Jrow = ",Jrow)

            if verbosity > 0: print("%sHamiltonian fidpair %d of %d: %d observables analyzed [%.1fs]" % 
                                    (rankStr, i,len(my_ExptList),len(all_observables),_time.time()-t0))
                
        
        #Gather results
        info_list = [ my_obs_infos ] if (comm is None) else comm.gather(my_obs_infos, root=0)
        J_list = [ my_J ] if (comm is None) else comm.gather(my_J, root=0)
        if extract_affine:
            Jaff_list = [ my_Jaff ] if (comm is None) else comm.gather(my_Jaff, root=0)

        #REMOVE
        #if saved_fits is not None:
        #    saved_fit_list = [ my_saved_fits ] if (comm is None) else comm.gather(my_saved_fits, root=0)
        #    if comm is None or comm.Get_rank() == 0:
        #        for lst in saved_fit_list: saved_fits.extend(lst)

        if comm is None or comm.Get_rank() == 0:
            # pseudo-invert J to get "intrinsic" error rates (labeled by AllErrors(nQubits))
            # J*intr = obs
            J = _np.concatenate(J_list, axis=0)
            info_list = list(_itertools.chain(*info_list)) # flatten ~ concatenate

            obs_err_rates = _np.array([info['rate'] for info in info_list])

            if extract_affine:
                #'correct' observed rates due to known affine errors, i.e.:
                # J_ham * Hintrinsic = observed_rates - J_aff * Aintrinsic
                Jaff = _np.concatenate(Jaff_list, axis=0)
                Aintrinsic = result.data['Intrinsic affine rates']
                corr = _np.dot(Jaff, Aintrinsic)
                #print("RAW = ",Jaff)
                #print("RAW = ",Aintrinsic)
                #print("RAW = ",corr)
                print("AFFINE CORR = ",_np.linalg.norm(corr), _np.linalg.norm(Jaff), _np.linalg.norm(Aintrinsic))
                print("BEFORE = ",obs_err_rates)
                obs_err_rates -= corr
                print("AFTER = ",obs_err_rates)
            
            invJ = _np.linalg.pinv(J)
            intrinsic_hamiltonian_rates = _np.dot(invJ,obs_err_rates)
                        
            #print("HAM J = ",J)
            print("HAM RATES = ",intrinsic_hamiltonian_rates)
            print("HAM INVJ = ",_np.linalg.norm(invJ))
            print("HAM OBS = ",_np.linalg.norm(obs_err_rates))
            print("HAM RATES = ",_np.linalg.norm(intrinsic_hamiltonian_rates))
            result.data['Hamiltonian error names'] = errors # "key" to intrinsic rates
            result.data['Intrinsic hamiltonian rates'] = intrinsic_hamiltonian_rates
            result.data['Hamiltonian fidpairs'] = pauli_fidpairs # give "key" to observed rates
            result.data['Observed hamiltonian infos'] = info_list
            print("Completed Hamiltonian in %.2fs" % (_time.time()-tStart))

    if comm is None or comm.Get_rank() == 0:
        return result
    else: # no results on other ranks...
        return None




#SCRATCH BELOW --------------------------------------------------------------------------------------------------------
#SCRATCH BELOW --------------------------------------------------------------------------------------------------------
#SCRATCH BELOW --------------------------------------------------------------------------------------------------------

# In[5]:

# These functions are designed to build particular experiments that I have found to work for Hamiltonian tomography.

# The circuits that detect Hamiltonian errors treat each qubit independently.  And on each
# qubit, they prep in one basis and measure in another, in order to detect the 3rd.  So they
# can described by one Pauli for each qubit -- the one that the prep/measure pair on that qubit would detect.
# For example, a "Z" circuit has prep/measure {Y / X}.  A "YX" circuit has prep/measure {XZ / ZY}.
# The functions defined here create and define circuits like that.

## First, some obvious things that are useful:  NextPauli just maps X -> Y -> Z -> X.
#def NextPauli( P ):
#    return {"X":"Y","Y":"Z","Z":"X"}[P]
## PrevPauli does the inverse of NextPauli:  X -> Z -> Y -> X
#def PrevPauli( P ):
#    return {"X":"Z","Y":"X","Z":"Y"}[P]
## Prep takes an N-qubit Pauli string and maps each Pauli to the previous Pauli in [X,Y,Z]
#def Prep( expt ):
#    return ''.join([PrevPauli(p) for p in expt])
## Meas takes an N-qubit Pauli string and maps each Pauli to the next Pauli in [X,Y,Z]
#def Meas( expt ):
#    return ''.join([NextPauli(p) for p in expt])
#
## BStr defines strings like "01010101" and "00001111", and replaces 0 and 1 with symbols[0] and symbols[1]
#def BStr( N, bit, symbols ):
#    return ''.join([symbols[((i>>bit) & 1)] for i in range(N)])
#
## ExtendPauliPair() takes a 2-Pauli string "PauliPair" (e.g. "XY") and generates a set of N-Pauli strings 
## such that every pair of qubits takes those two values in at least one string.
#def ExtendPauliPairToN( PauliPair, N ):
#    if PauliPair[0] == PauliPair[1]:
#        return [PauliPair[0]*N]
#    else:
#        return [BStr(N,i,[PauliPair[0],PauliPair[1]]) for i in range(int(math.ceil(math.log(N,2))))]
#
## ExtendPauliPairListToN() just applies ExtendPauliPair to a list of Pauli pairs and flattens the result.
#def ExtendPauliPairListToN( PauliPairList, N ):
#    return [str for PP in PauliPairList for str in ExtendPauliPairToN(PP,N) ]





#REMOVE
## XOR() is just an XOR... but on string characters.  Duh.
#def XOR( b1, b2 ):
#    if b1==b2:
#        return "0"
#    else:
#        return "1"
#
## To detect affine errors, we need a shorthand for experiments that initialize in the -1 eigenstate of
## a given Pauli.  So I use U/V/W for this.  U ==> initialize in -X.  Same for V (-Y) and W (-Z).
## SignedSinglePauliToPauli() converts a "signed" Pauli (e.g. U,V,W) to a real Pauli (X,Y,Z).
#def SignedSinglePauliToPauli( SP ):
#    return {"X":"X","Y":"Y","Z":"Z","U":"X","V":"Y","W":"Z"}[SP]
#
## SignedSinglePauliToMaskBit() extracts the sign of a "signed" Pauli, returning 0 if it's a real Pauli and 1 if flipped.
#def SignedSinglePauliToMaskBit( SP ):
#    if SignedPauliToPauli(SP)==SP:
#        return "0"
#    else:
#        return "1"
## SignedPauliToPauli() just applies SignedSinglePauliToPauli() to a string.
#def SignedPauliToPauli( SPstring ):
#    return ''.join( [SignedSinglePauliToPauli(sp) for sp in SPstring] )
#
## SignedPauliToMask() just applies SignedSinglePauliToMaskBit() to a string.
#def SignedPauliToMask( SPstring ):
#    return ''.join( [SignedSinglePauliToMaskBit(sp) for sp in SPstring] )
#
#
## In[21]:
## ExtendedStochasticMatrixElement() just computes a stochastic Jacobian element, but converts signed Paulis
## to regular Paulis before doing so (sign has no effect on the behavior of a stochastic error)
#def ExtendedStochasticMatrixElement( PrepMeas, Error, OutcomeString ):
#    return StochasticMatrixElement( SignedPauliToPauli( PrepMeas ), Error, OutcomeString )
#
## ExtendedAffineMatrixElement() computes an affine Jacobian element, but converts signed Paulis
## to regular Paulis *and* extracts the sign and converts it to an explicit mask 
## (sign flips the effect of of affine errors)
#def ExtendedAffineMatrixElement( PrepMeas, Error, OutcomeString ):
#    return AffineMatrixElement( SignedPauliToPauli(PrepMeas), Error, OutcomeString, SignedPauliToMask(PrepMeas) )





#HAMILTONIAN

# In[4]:

# Now, some convenience functions



# In[6]:

# Okay:  let's put things together and show that we can do idle tomography on Hamiltonian errors.

def ShowThatHamiltonianWorks( N, verbose=False ):
    # Make a list of the experiments that my secret sauce says we need
    Expts = HamiltonianExptList( N )
    # Create the Jacobian by constructing observables for each experiment and computing matrix elements w/errors.
    J = _np.array( [[HamiltonianMatrixElement( Prep(expt), err, obs ) for err in AllErrors(N)]                       for expt in Expts for obs in AllObservables( Meas(expt) )] )

    # Compute and print out a bunch of statistics and descriptive stuff
    AllObsv = [obs for expt in Expts for obs in AllObservables(Meas(expt))]
    print( len(Expts), "Experiments on",N,"qubits")
    if verbose:
        print( len(AllObsv),"Observables =",AllObsv )
        print( len(AllErrors(N)),"Errors =",AllErrors(N) )
        print( J )
    else:
        print( len(AllObsv),"Observables" )
        print( len(AllErrors(N)),"Errors" )
    # Now compute the singular values of the Jacobian and see if there's a >0 one for each error
    sv = _np.linalg.svd(J)[1]
    print( len(_np.where( sv > 1e-3)[0]),"singular values > 1e-3" )
    print( "Smallest singular value is",min(sv) )
    if verbose:
        print("Singular values are",sv)



# In[14]:

# Next, let's turn to detecting stochastic errors.


# In[7]:


    
# StochasticJacobian() computes the entire Jacobian matrix describing how the outcome probabilities of
# all pairwise Pauli experiments vary with stochastic error rates.  This is a matrix that can multiply
# a vector of stochastic Pauli error rates, and the result is a vector of observable bit error rates
# (i.e., probabilities of weight-2 error strings.
def StochasticJacobian( ExptList ):
    N = len(ExptList[0])
    return _np.array( [[StochasticMatrixElement( expt, err, out ) for err in AllErrors(N)]
                      for expt in ExptList for out in AllOutcomes( N )] )

# Okay, let's put it all together
def ShowThatStochasticWorks( N ):
    M = StochasticJacobian( StochasticExptList(N) )
    print( len(StochasticExptList(N)),"experiments on",N,"qubits." )
    print( "N =",N )
    print( "Number of distinct weight-1&2 errors:", len(AllErrors(N)) )
    print( "Number of distinct experiments:",len(StochasticExptList(N)) )
    print( "Shape of Jacobian:",M.shape )
    print( "Rank of Jacobian:",_np.linalg.matrix_rank(M) )
    print( "Smallest singular value:",min(_np.linalg.svd(M)[1]) )



# In[20]:



# In[24]:

    
# AffineJacobian() just computes a Jacobian for the derivative of all the outcome probabilities of
# experiments in "ExptList" with respect to *both* stochastic and affine errors.
def AffineJacobian( ExptList ):
    N = len(ExptList[0])
    return _np.array( [[ExtendedStochasticMatrixElement( expt, err, out ) for err in AllErrors(N)] + \
                       [ExtendedAffineMatrixElement( expt, err, out ) for err in AllErrors(N)]
                       for expt in ExptList for out in AllOutcomes( N )] )

# Now let's put it all together and show that this actually works.
def ShowThatAffineWorks( N ):
    M = AffineJacobian( AffineExptList(N) )
    print( len(AffineExptList(N)),"experiments on",N,"qubits.")
    print( "Number of distinct stochastic+affine weight-1&2 errors:", len(AllErrors(N))*2)
    print( "Shape of Jacobian:",M.shape)
    print( "Rank of Jacobian:",_np.linalg.matrix_rank(M))
    print( "Smallest singular value:",min(_np.linalg.svd(M)[1]))


# -------------------------------------------------
# Routines for running Idle tomography (rough)
# -------------------------------------------------




def predicted_observable_rates(nQubits, gateset, hamiltonian=True, stochastic=True):
    stoJ = []
    hamJ = []

    ham_intrinsic_rates, sto_intrinsic_rates = \
        predicted_intrinsic_rates(nQubits, gateset, hamiltonian, stochastic)
    
    #Get jacobians
    if stochastic:
        ExptList = StochasticExptList(nQubits) #list of fidpairs / configs (a prep/meas that gets I^L placed btwn it)
        for expt in ExptList:
            for out in AllOutcomes( nQubits ): # probably don't need *all* these outcomes -- just those around 000..000 (TODO)
                stoJ.append( [ StochasticMatrixElement( expt, err, out ) for err in AllErrors(nQubits) ] )
        sto_observable_rates = _np.dot(stoJ,sto_intrinsic_rates)
        
    if hamiltonian:
        ExptList = HamiltonianExptList(nQubits)
        for expt in ExptList:
            pauli_fidpair = (Prep(expt),Meas(expt)) # Stochastic: prep & measure in same basis
            for obs in AllObservables( Meas(expt) ):
                negs = [ bool(pauli_fidpair[0][i] == 'Y') for i in range(nQubits) ] # b/c use of 'Gx' for Y-basis prep really prepares -Y state
                hamJ.append( [ HamiltonianMatrixElement( Prep(expt), err, obs, negs ) for err in AllErrors(nQubits) ] )
    
        ham_observable_rates = _np.dot(hamJ,ham_intrinsic_rates)
    else:
        ham_observable_rates = None

    return ham_observable_rates, sto_observable_rates



###########################################
#DEBUGGING helper functions               
###########################################
sqrt2 = _np.sqrt(2)
id2x2 = _np.array([[1,0],[0,1]])
sigma = { 'X': _np.array([[0,1],[1,0]]),
          'Y': _np.array([[0,-1.0j],[1.0j,0]]),
          'Z': _np.array([[1,0],[0,-1]]),
          'I': id2x2 }

def _pmx(s):
    """ 
    Construct the pauli matrix from a string `s` of I,X,Y,Z letters.
    (This is for debugging, as these matrices will get big fast.)
    """
    ret = sigma[s[0]].copy()
    for letter in s[1:]:
        ret = _np.kron(ret,sigma[letter])
    return ret

def _pcomm(s1,s2):
    """
    Construct the commutator (matrix) from a two pauli-strings `s1` and `s2`
    of I,X,Y,Z letters.
    (This is for debugging, as these matrices will get big fast.)
    """
    A = pmx(s1)
    B = pmx(s2)
    return _np.dot(A,B) - _np.dot(B,A)
