""" Core Idle Tomography routines """
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import itertools as _itertools
import time as _time

from ... import objects as _objs
from ... import tools as _tools

from . import pauliobjs as _pobjs
from . import tools as _idttools
from .idtresults import IdleTomographyResults as _IdleTomographyResults

# This module implements idle tomography, which deals only with
# many-qubit idle gates (on some number of qubits) and single-
# qubit gates (or tensor products of them) used to fiducials.
# As such, it is conventient to represent operations as native
# Python strings, where there is one I,X,Y, or Z letter per
# qubit.


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
        if _pobjs._commute_parity(P,Err) == 1: # commutes: [P,Err] == 0
            outcome_str += "0" if (s1 == s2) else "1"
        else: # anticommutes: {P,Err} == 0
            outcome_str += "1" if (s1 == s2) else "0"

    return _pobjs.NQOutcome(outcome_str)


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
            else: return 1 # bases don't match
    
        if measBasis != errP: #then they don't commute (b/c neither can be 'I')
            return 0  # so there's no change along this axis (see docstring)
        else: # measBasis == errP != 'I'            
            if outcome_bit == "0": return measSign
            else: return measSign*-1

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
    
    return _np.prod( [_affhelper(s1,P1,Err,s2,P2,o) for s1,P1,s2,P2,Err,o
                      in zip(prep.signs, prep.rep, meas.signs, meas.rep,
                             error.rep, observable.rep)] )


# -----------------------------------------------------------------------------
# Experiment generation: 
# -----------------------------------------------------------------------------

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




def idle_tomography_fidpairs(nQubits, maxweight=2, include_hamiltonian=True,
                             include_stochastic=True, include_affine=True,
                             ham_tmpl=("ZY","ZX","XZ","YZ","YX","XY"),
                             preferred_prep_basis_signs=("+","+","+"),
                             preferred_meas_basis_signs=("+","+","+") ):
    """ TODO: docstring 
    Returns a list of 2-tuples of NQPauliState objects of length `nQubits`
    representing fiducial pairs.
    """
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
                sto_tmpl_pairs.append( (_pobjs.NQPauliState(''.join(basisLets), prepSigns),
                                        _pobjs.NQPauliState(''.join(basisLets), measSigns)) )

        fidpairs.extend( _idttools.tile_pauli_fidpairs(sto_tmpl_pairs, nQubits, maxweight) )

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
            ham_tmpl_pairs.append( (_pobjs.NQPauliState(prepLets, prepSigns),
                                    _pobjs.NQPauliState(measLets, measSigns)) )
            
        fidpairs.extend( _idttools.tile_pauli_fidpairs(ham_tmpl_pairs, nQubits, maxweight) )

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
        
    GiStr = _objs.GateString( GiStr )

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


# -----------------------------------------------------------------------------
# Running idle tomography
# -----------------------------------------------------------------------------

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

    if advancedOptions is None: 
        advancedOptions = {}

    result = _IdleTomographyResults()
    prepDict,measDict = pauliDicts
    GiStr = _objs.GateString( GiStr )
                       
    #idebug = 0
    rankStr = "" if (comm is None) else "Rank%d: " % comm.Get_rank()

    preferred_prep_basis_signs = advancedOptions.get('preferred_prep_basis_signs', 'auto')
    preferred_meas_basis_signs = advancedOptions.get('preferred_meas_basis_signs', 'auto')
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    errors = _idttools.allerrors(nQubits,maxErrWeight)
    fitOrder = advancedOptions.get('fit order',1)

    if extract_stochastic:
        tStart = _time.time()
        pauli_fidpairs = idle_tomography_fidpairs(
            nQubits, maxErrWeight, False, extract_stochastic, extract_affine,
            advancedOptions.get('ham_tmpl', ("ZY","ZX","XZ","YZ","YX","XY")),
            preferred_prep_basis_signs, preferred_meas_basis_signs)

        #divide up strings among ranks
        indxFidpairList = list(enumerate(pauli_fidpairs))
        my_FidpairList, _,_ = _tools.mpitools.distribute_indices(indxFidpairList,comm,False)

        my_J = []; my_obs_infos = []
        #REM my_obs_err_rates = []
        #REM my_saved_fits = [] if (saved_fits is not None) else None
        for i,(ifp,pauli_fidpair) in enumerate(my_FidpairList):
            #REM pauli_fidpair = (expt,expt) # Stochastic: prep & measure in same basis
            #NOTE: pauli_fidpair is a 2-tuple of NQPauliState objects
            
            all_outcomes = _idttools.alloutcomes(pauli_fidpair[0], pauli_fidpair[1], maxErrWeight)
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
        my_FidpairList, _,_ = _tools.mpitools.distribute_indices(indxFidpairList,comm,False)

        my_J = []; my_obs_infos = []; my_Jaff = []
        for i,(ifp,pauli_fidpair) in enumerate(my_FidpairList):
            #REM pauli_fidpair = (Prep(expt),Meas(expt))
            all_observables = _idttools.allobservables( pauli_fidpair[1], maxErrWeight )

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
