""" Idle Tomography utility routines """
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import itertools as _itertools

from ... import objects as _objs
from ... import tools as _tools
from ...construction import nqnoiseconstruction as _nqn

from . import pauliobjs as _pobjs
  # maybe need to restructure in future - "tools" usually doesn't import "objects"

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
    eoutcome = _pobjs.NQOutcome(''.join(expected))
    if maxweight == 1:
        return [eoutcome.flip(i) for i in range(N)]
    else:
        return [eoutcome.flip(i) for i in range(N)] + \
               [eoutcome.flip(i,j) for i in range(N) for j in range(i+1,N)]


def allerrors(N, maxweight):
    """ lists every Pauli error for N qubits with weight <= `maxweight` """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweigth <= 2 is currently supported")
    if maxweight == 1:
        return [_pobjs.NQPauliOp.Weight1Pauli(N,loc,p) for loc in range(N) for p in range(3)]
    else:
        return [_pobjs.NQPauliOp.Weight1Pauli(N,loc,p) for loc in range(N) for p in range(3)] + \
               [_pobjs.NQPauliOp.Weight2Pauli(N,loc1,loc2,p1,p2) for loc1 in range(N) 
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
        return [_pobjs.NQPauliOp(meas.rep).subpauli([i]) for i in range(len(meas))]
    else:
        return [_pobjs.NQPauliOp(meas.rep).subpauli([i]) for i in range(len(meas))] + \
               [_pobjs.NQPauliOp(meas.rep).subpauli([i,j]) for i in range(len(meas)) for j in range(i+1,len(meas))]


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
            prep = _pobjs.NQPauliState( [ base_prep.rep[i] for i in tmpl_row ],
                                        [ base_prep.signs[i] for i in tmpl_row] )
            meas = _pobjs.NQPauliState( [ base_meas.rep[i] for i in tmpl_row ],
                                        [ base_meas.signs[i] for i in tmpl_row] )
            nqubit_fidpairs.append((prep,meas))

    _tools.remove_duplicates_in_place(nqubit_fidpairs)
    return nqubit_fidpairs


# ----------------------------------------------------------------------------
# Testing tools (only used in testing, not for running idle tomography)
# ----------------------------------------------------------------------------

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
        assert(isinstance(factor, _objs.EmbeddedGateMap)), "Expected Gi to be a composition of embedded gates!"
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
        assert(isinstance(factor, _objs.EmbeddedGateMap)), "Expected Gi to be a composition of embedded gates!"
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


def predicted_observable_rates(idtresults, typ, nQubits, maxErrWeight, gateset):
    """
    TODO: docstring - returns a dict of form: rate = ret[pauli_fidpair][obsORoutcome]
    """
    #Get intrinsic rates
    hamiltonian = stochastic = affine = False
    if typ == "hamiltonian": hamiltonian = True
    if typ == "stochastic" or "stochastic/affine": stochastic = True
    if typ == "affine" or "stochastic/affine": affine = True
    ham_intrinsic_rates, sto_intrinsic_rates, aff_intrinsic_rates = \
        predicted_intrinsic_rates(nQubits, maxErrWeight, gateset, hamiltonian, stochastic, affine)

    ret = {}
    if typ in ("stochastic","stochastic/affine"):

        intrinsic = _np.concatenate([sto_intrinsic_rates,aff_intrinsic_rates]) \
            if typ == "stochastic/affine" else sto_intrinsic_rates
        
        for fidpair, dict_of_infos in zip(idtresults.pauli_fidpairs[typ],
                                       idtresults.observed_rate_infos[typ]):
            ret[fidpair] = {}
            for obsORoutcome,info_dict in dict_of_infos.items():
                #Get jacobian row and compute predicted observed rate
                Jrow = info_dict['jacobian row']
                predicted_rate = _np.dot(Jrow,intrinsic)
                ret[fidpair][obsORoutcome] = predicted_rate
                
    elif typ == "hamiltonian":

        # J_ham * Hintrinsic = observed_rates - J_aff * Aintrinsic
        # so: observed_rates = J_ham * Hintrinsic + J_aff * Aintrinsic
        for fidpair, dict_of_infos in zip(idtresults.pauli_fidpairs[typ],
                                       idtresults.observed_rate_infos[typ]):
            ret[fidpair] = {}
            for obsORoutcome,info_dict in dict_of_infos.items():
                #Get jacobian row and compute predicted observed rate
                Jrow = info_dict['jacobian row']
                predicted_rate = _np.dot(Jrow,ham_intrinsic_rates)
                if 'affine jacobian row' in info_dict:
                    affJrow = info_dict['affine jacobian row']
                    predicted_rate += _np.dot(affJrow,aff_intrinsic_rates)
                ret[fidpair][obsORoutcome] = predicted_rate

    else:
        raise ValueError("Unknown `typ` argument: %s" % typ)

    return ret
