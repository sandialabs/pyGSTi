""" Utility functions which operate on GateSet objects."""
import numpy as _np
import scipy as _scipy
import numpy.random as _rndm

from ..tools import basistools as _bt
from .. import tools as _tools

import gatestring as _gs
import gate as _gate
import dataset as _ds

#################################################
# GateSet Tools
#################################################

def depolarizeSPAM(gateset,noise=None,max_noise=None,seed=None):
    """
    Apply depolarization uniformly or randomly to a gateset's SPAM.
    elements. You must specify either 'noise' or 'max_noise'.
    
    Parameters
    ----------
    gateset : GateSet
      the gate set to depolarize

    noise : float, optional
      apply depolarizing noise of strength 1-noise to all
      SPAM vectors in the gateset. (Multiplies the non-identity
      part of each assumed-Pauli-basis state preparation vector
      and measurement vector by (1.0-noise).

    max_noise : float, optional
      specified instead of 'noise'; apply a random depolarization
      with maximum strength 1-max_noise to SPAM vector in the gateset.

    seed : int, optional
      if not None, seed numpy's random number generator with this value
      before generating random depolarizations.
    
    Returns
    -------
    GateSet
        the depolarized GateSet
    """
    newGateset = gateset.copy() # start by just copying gateset
    gateDim = gateset.get_dimension()
    # nothing is applied to rhoVec or EVec

    if seed is not None:
        rdm.seed(seed)

    if max_noise is not None:
        if noise is not None: 
            raise ValueError("Must specify exactly one of 'noise' and 'max_noise' NOT both")

        #Apply random depolarization to each rho and E vector
        r = max_noise * _rndm.random( len(gateset.rhoVecs) )
        for (i,rhoVec) in enumerate(gateset.rhoVecs):
            D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
            newGateset.set_rhoVec( _np.dot(D,rhoVec), i)

        r = max_noise * _rndm.random( len(gateset.EVecs) )
        for (i,EVec) in enumerate(gateset.EVecs):
            D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
            newGateset.set_EVec( _np.dot(D,EVec), i)

            
    elif noise is not None:
        #Apply the same depolarization to each gate
        D = _np.diag( [1]+[1-noise]*(gateDim-1) )
        for (i,rhoVec) in enumerate(gateset.rhoVecs):
            newGateset.set_rhoVec( _np.dot(D,rhoVec), i)
        for (i,EVec) in enumerate(gateset.EVecs):
            newGateset.set_EVec( _np.dot(D,EVec), i)
    
    else: raise ValueError("Must specify either 'noise' or 'max_noise' -- neither was non-None")
    return newGateset


def depolarizeGateset(gateset,noise=None,max_noise=None,seed=None):
    """
    Apply depolarization uniformly or randomly to the gates
    of a GateSet.  You must specify either 'noise' or 'max_noise'.

    Parameters
    ----------
    gateset : GateSet
      the gate set to depolarize

    noise : float, optional
      apply depolarizing noise of strength 1-noise to all
      gates in the gateset. (Multiplies each assumed-Pauli-basis gate 
      matrix by the diagonal matrix with (1.0-noise) along all
      the diagonal elements except the first (the identity).

    max_noise : float, optional
      specified instead of 'noise'; apply a random depolarization
      with maximum strength 1-max_noise to each gate in the gateset.

    seed : int, optional
      if not None, seed numpy's random number generator with this value
      before generating random depolarizations.
    
    Returns
    -------
    GateSet
        the depolarized GateSet
    """
    newGateset = gateset.copy() # start by just copying gateset
    gateDim = gateset.get_dimension()
    # nothing is applied to rhoVec or EVec

    if seed is not None:
        rdm.seed(seed)

    if max_noise is not None:
        if noise is not None: 
            raise ValueError("Must specify exactly one of 'noise' and 'max_noise' NOT both")

        #Apply random depolarization to each gate
        r = max_noise * _rndm.random( len(gateset) )
        for (i,label) in enumerate(gateset):
            D = _np.diag( [1]+[1-r[i]]*(gateDim-1) )
            newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot(D,gateset[label]) ))
            
    elif noise is not None:
        #Apply the same depolarization to each gate
        D = _np.diag( [1]+[1-noise]*(gateDim-1) )
        for (i,label) in enumerate(gateset):
            newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot(D,gateset[label]) ))
    
    else: raise ValueError("Must specify either 'noise' or 'max_noise' -- neither was non-None")
    return newGateset


def rotateGateset(gateset, rotate=None, max_rotate=None, seed=None):
    """
    Apply rotation uniformly or randomly to a gateset.
    You must specify either 'rotate' or 'max_rotate'. 
    This method currently only works on single-qubit
    gatesets.

    Parameters
    ----------
    gateset : GateSet
      the gate set to rotate

    rotate : float or 3-tuple of floats, optional
      if a single float, apply rotation of rotate radians along
      each of the x, y, and z axes of all gates in the gateset.
      if a 3-tuple of floats, apply the values as x, y, and z rotations
      (in radians) to all of the gates in the gateset.

    max_rotate : float, optional
      specified instead of 'rotate'; apply a random rotation with
      maximum max_rotate radians along each of the x, y, and z axes
      of each each gate in the gateset.  That is, rotations of a 
      particular gate around different axes are different random amounts.

    seed : int, optional
      if not None, seed numpy's random number generator with this value
      before generating random depolarizations.
    
    Returns
    -------
    GateSet
        the rotated GateSet
    """
    newGateset = gateset.copy() # start by just copying gateset
    # nothing is applied to rhoVec or EVec

    for (i,rhoVec) in enumerate(gateset.rhoVecs):
        newGateset.set_rhoVec( rhoVec, i )  
    for (i,EVec) in enumerate(gateset.EVecs):
        newGateset.set_EVec( EVec, i )

    if gateset.get_dimension() != 4:
        raise ValueError("Gateset rotation can only be performed on a *single-qubit* gateset")

    if seed is not None:
        _rndm.seed(seed)

    if max_rotate is not None:
        if rotate is not None: 
            raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")

        #Apply random rotation to each gate
        r = max_rotate * _rndm.random( len(gateset) * 3 )
        for (i,label) in enumerate(gateset):
            rot = r[3*i:3*(i+1)]
            newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                        _bt.singleQubitGate(rot[0]/2.0,rot[1]/2.0,rot[2]/2.0), gateset[label]) ))
            
    elif rotate is not None:
        #Apply the same rotation to each gate
        #Specify rotation by a single value (to mean this rotation along each axis) or a 3-tuple
        if type(rotate) in (float,int): rx,ry,rz = rotate,rotate,rotate
        elif type(rotate) in (tuple,list):
            if len(rotate) != 3:
                raise ValueError("Rotation, when specified as a tuple must be of length 3, not: %s" % rotate)
            (rx,ry,rz) = rotate
        else: raise ValueError("Rotation must be specifed as a single number or as a lenght-3 list, not: %s" % rotate)
            
        for (i,label) in enumerate(gateset):
            newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                        _bt.singleQubitGate(rx/2.0,ry/2.0,rz/2.0), gateset[label]) ))

    else: raise ValueError("Must specify either 'rotate' or 'max_rotate' -- neither was non-None")
    return newGateset

def rotate2QGateset(gateset, rotate=None, max_rotate=None, seed=None):
    """
    Apply rotation uniformly or randomly to a two-qubut gateset.
    You must specify either 'rotate' or 'max_rotate'. 

    Parameters
    ----------
    gateset : GateSet
      the gate set to rotate

    rotate : float or 15-tuple of floats, optional
      if a single float, apply rotation of rotate radians along
      each of the 15 axes of all gates in the gateset.
      if a 15-tuple of floats, apply the values as ix,...,zz rotations
      (in radians) to all of the gates in the gateset.

    max_rotate : float, optional
      specified instead of 'rotate'; apply a random rotation with
      maximum max_rotate radians along each of the ix,...,zz axes
      of each each gate in the gateset.  That is, rotations of a 
      particular gate around different axes are different random amounts.

    seed : int, optional
      if not None, seed numpy's random number generator with this value
      before generating random depolarizations.
    
    Returns
    -------
    GateSet
        the rotated GateSet
    """
    newGateset = gateset.copy() # start by just copying gateset
    # nothing is applied to rhoVec or EVec

    for (i,rhoVec) in enumerate(gateset.rhoVecs):
        newGateset.set_rhoVec( rhoVec, i )  
    for (i,EVec) in enumerate(gateset.EVecs):
        newGateset.set_EVec( EVec, i )

    if gateset.get_dimension() != 16:
        raise ValueError("Gateset rotation can only be performed on a *two-qubit* gateset")

    if seed is not None:
        _rndm.seed(seed)

    if max_rotate is not None:
        if rotate is not None: 
            raise ValueError("Must specify exactly one of 'rotate' and 'max_rotate' NOT both")

        #Apply random rotation to each gate
        r = max_rotate * _rndm.random( len(gateset) * 15 )
        for (i,label) in enumerate(gateset):
            rot = r[15*i:15*(i+1)]
            newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                        _bt.twoQubitGate(rot[0]/2.0,rot[1]/2.0,rot[2]/2.0,
                                         rot[3]/2.0,rot[4]/2.0,rot[5]/2.0,
                                         rot[6]/2.0,rot[7]/2.0,rot[8]/2.0,
                                         rot[9]/2.0,rot[10]/2.0,rot[11]/2.0,
                                         rot[12]/2.0,rot[13]/2.0,rot[14]/2.0,
                                         ), gateset[label]) ))
            
    elif rotate is not None:
        #Apply the same rotation to each gate
        #Specify rotation by a single value (to mean this rotation along each axis) or a 3-tuple
        if type(rotate) in (float,int): 
            rix,riy,riz = rotate,rotate,rotate
            rxi,rxx,rxy,rxz = rotate,rotate,rotate,rotate
            ryi,ryx,ryy,ryz = rotate,rotate,rotate,rotate
            rzi,rzx,rzy,rzz = rotate,rotate,rotate,rotate
        elif type(rotate) in (tuple,list):
            if len(rotate) != 15:
                raise ValueError("Rotation, when specified as a tuple must be of length 15, not: %s" % rotate)
            (rix,riy,riz,rxi,rxx,rxy,rxz,ryi,ryx,ryy,ryz,rzi,rzx,rzy,rzz) = rotate
        else: raise ValueError("Rotation must be specifed as a single number or as a lenght-15 list, not: %s" % rotate)
            
        for (i,label) in enumerate(gateset):
            newGateset.set_gate(label, _gate.FullyParameterizedGate( _np.dot( 
                        _bt.twoQubitGate(rix/2.0,riy/2.0,riz/2.0,
                                         rxi/2.0,rxx/2.0,rxy/2.0,rxz/2.0,
                                         ryi/2.0,ryx/2.0,ryy/2.0,ryz/2.0,
                                         rzi/2.0,rzx/2.0,rzy/2.0,rzz/2.0,), gateset[label]) ))

    else: raise ValueError("Must specify either 'rotate' or 'max_rotate' -- neither was non-None")
    return newGateset


def randomizeGatesetWithUnitary(gatesetInPauliProdBasis,scale,seed=None):
    """
    Apply a random unitary to each element of a gateset.
    This method currently only works on single- and two-qubit
    gatesets.

    Parameters
    ----------
    gatesetInPauliProdBasis : GateSet
      the gate set, with matrices in the Pauli-product basis, to randomize.

    scale : float
      maximum element magnitude in the generator of each random unitary transform.

    seed : int, optional
      if not None, seed numpy's random number generator with this value
      before generating random depolarizations.
    
    Returns
    -------
    GateSet
        the randomized GateSet
    """
    gs_pauli = gatesetInPauliProdBasis.copy()

    if seed is not None:
        _np.random.seed(seed)

    gate_dim = gs_pauli.get_dimension()
    if gate_dim == 4: unitary_dim = 2
    elif gate_dim == 16: unitary_dim = 4
    else: raise ValueError("Gateset dimension must be either 4 (single-qubit) or 16 (two-qubit)")

    for gateLabel in gs_pauli.keys():
        randMat = scale * (_np.random.randn(unitary_dim,unitary_dim) + 1j * _np.random.randn(unitary_dim,unitary_dim))
        randMat = _np.dot(_np.transpose(_np.conjugate(randMat)),randMat) # make randMat Hermetian: (A_dag*A)^dag = (A_dag*A)
        randU   = _scipy.linalg.expm(-1j*randMat)

        if unitary_dim == 2:
            randUPP = _bt.stateUnitaryToPauliDensityMxOp(randU)
        elif unitary_dim == 4:
            randUPP = _bt.stateUnitaryToPauliDensityMxOp_2Q(randU)
        else: raise ValueError("Gateset dimension must be either 4 (single-qubit) or 16 (two-qubit)")

        gs_pauli.set_gate(gateLabel, _gate.FullyParameterizedGate(_np.dot(randUPP,gs_pauli[gateLabel])))

    return gs_pauli


def increaseGatesetDimension(gateset, newDimension):
    """
    Enlarge the spam vectors and gate matrices of gateset to a specified
    dimension.  Spam vectors are zero-padded and gate matrices are padded
    with 1's on the diagonal and zeros on the off-diagonal (effectively
    padded by identity operation).

    Parameters
    ----------
    gateset : GateSet
      the gate set to act on

    newDimension : int
      the dimension of the returned gateset.  That is,
      the returned gateset will have rho and E vectors that
      have shape (newDimension,1) and gate matrices with shape
      (newDimension,newDimension)
    
    Returns
    -------
    GateSet
        the increased-dimension GateSet
    """

    curDim = gateset.get_dimension()
    assert(newDimension > curDim)
    
    new_gateset = gateset.copy()
    new_gateset.gate_dim = newDimension;

    addedDim = newDimension-curDim
    vec_zeroPad = _np.zeros( (addedDim,1), 'd')

    #Increase dimension of rhoVecs and EVecs by zero-padding
    for i,rhoVec in enumerate(gateset.rhoVecs):
        assert( len(gateset.rhoVecs[i]) == curDim )
        new_gateset.rhoVecs[i] = _np.concatenate( (gateset.rhoVecs[i], vec_zeroPad) )

    for i,EVec in enumerate(gateset.EVecs):
        assert( len(gateset.EVecs[i]) == curDim )
        new_gateset.EVecs[i] = _np.concatenate( (gateset.EVecs[i], vec_zeroPad) )

    #Increase dimension of identityVec by zero-padding
    if gateset.identityVec is not None:
        new_gateset.identityVec = _np.concatenate( (gateset.identityVec, vec_zeroPad) )

    #Increase dimension of gates by assuming they act as identity on additional (unknown) space
    for gateLabel,gate in gateset.iteritems():
        assert( gate.shape == (curDim,curDim) )
        newGate = _np.zeros( (newDimension,newDimension) )
        newGate[ 0:curDim, 0:curDim ] = gate[:,:]
        for i in xrange(curDim,newDimension): newGate[i,i] = 1.0
        new_gateset.set_gate(gateLabel, _gate.FullyParameterizedGate(newGate))

    new_gateset.makeSPAMs()
    return new_gateset


def decreaseGatesetDimension(gateset, newDimension):
    """
    Shrink the spam vectors and gate matrices of gateset to a specified
    dimension.

    Parameters
    ----------
    gateset : GateSet
      the gate set to act on

    newDimension : int
      the dimension of the returned gateset.  That is,
      the returned gateset will have rho and E vectors that
      have shape (newDimension,1) and gate matrices with shape
      (newDimension,newDimension)
    
    Returns
    -------
    GateSet
        the decreased-dimension GateSet
    """

    curDim = gateset.get_dimension()
    assert(newDimension < curDim)
    
    new_gateset = gateset.copy()
    new_gateset.gate_dim = newDimension

    #Decrease dimension of rhoVecs and EVecs by truncation
    for i,rhoVec in enumerate(gateset.rhoVecs):
        assert( len(gateset.rhoVecs[i]) == curDim )
        new_gateset.rhoVecs[i] = gateset.rhoVecs[i][0:newDimension,:]

    for i,EVec in enumerate(gateset.EVecs):
        assert( len(gateset.EVecs[i]) == curDim )
        new_gateset.EVecs[i] = gateset.EVecs[i][0:newDimension,:]

    #Decrease dimension of identityVec by trunction
    if gateset.identityVec is not None:
        new_gateset.identityVec = gateset.identityVec[0:newDimension,:]

    #Decrease dimension of gates by truncation
    for gateLabel,gate in gateset.iteritems():
        assert( gate.shape == (curDim,curDim) )
        newGate = _np.zeros( (newDimension,newDimension) )
        newGate[ :, : ] = gate[0:newDimension,0:newDimension]
        new_gateset.set_gate(gateLabel, _gate.FullyParameterizedGate(newGate))

    new_gateset.makeSPAMs()
    return new_gateset

def randomKickGateset(gateset, absmag=1.0, bias=0):
    """
    Kick gateset by adding to each gate a random matrix with values
    uniformly distributed in the interval [bias-absmag,bias+absmag].

    Parameters
    ----------
    gateset : GateSet
        the gate set to kick.

    absmag : float, optional
        The maximum magnitude of the entries in the "kick" matrix
        relative to bias.
        
    bias : float, optional
        The bias of the entries in the "kick" matrix.

    Returns
    -------
    GateSet
        the kicked gate set.
    """
    kicked_gs = gateset.copy()
    for gateLabel,gate in gateset.iteritems():
        delta = absmag * 2.0*(_rndm.random(gate.shape)-0.5) + bias
        kicked_gs.set_gate(gateLabel, _gate.FullyParameterizedGate(kicked_gs[gateLabel] + delta))
    #kicked_gs.makeSPAMs() #if we modify rhoVecs or EVecs
    return kicked_gs


def generateFakeData(gatesetOrDataset, gateStringList, nSamples, sampleError="none", seed=None):
    """
    Creates a DataSet using the probabilities obtained from a gateset.

    Parameters
    ----------
    gatesetOrDataset : GateSet or DataSet object
        If a GateSet, the gate set whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    gateStringList : list of (tuples or GateStrings) or None
        Each tuple or GateString contains gate labels and 
        specifies a gate sequence whose counts are included 
        in the returned DataSet.
        e.g. [ (), ('Gx',), ('Gx','Gy') ] 

    nSamples : int or list of ints or None
        The simulated number of samples for each gate string.  This only
        has effect when  sampleError == "binomial" or "multinomial".  If
        an integer, all gate strings have this number of total samples. If
        a list, integer elements specify the number of samples for the 
        corresponding gate string.  If None, then gatesetOrDataset must be
        a DataSet, and total counts are taken from it (on a per-gatestring
        basis).

    sampleError : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sampl error: 
                  counts are floating point numbers such that the exact probabilty
                  can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest integer.
        - "binomial" - the number of counts is taken from a binomial distribution.
                     Distribution has parameters p = probability of the gate string
                     and n = number of samples.  This can only be used when there
                     are exactly two SPAM labels in gatesetOrDataset.
        - "multinomial" - counts are taken from a multinomial distribution.
                        Distribution has parameters p_k = probability of the 
                        gate string using the k-th SPAM label and n = number
                        of samples.

    seed : int, optional
        If not None, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.

    """
    if isinstance(gatesetOrDataset, _ds.DataSet):
        dsGen = gatesetOrDataset #dataset
        gsGen = None
        dataset = _ds.DataSet( spamLabels=dsGen.getSpamLabels() )
    else:
        gsGen = gatesetOrDataset #dataset
        dsGen = None
        dataset = _ds.DataSet( spamLabels=gsGen.get_SPAM_labels() )

    if seed is not None: _rndm.seed(seed)

    for k,s in enumerate(gateStringList):
      if gsGen:
          ps = gsGen.Probs(s) # a dictionary of probabilities; keys = spam labels
      else:
          ps = { sl: dsGen[s].fraction(sl) for sl in dsGen.getSpamLabels() }

      if nSamples is None and dsGen is not None:
          N = dsGen[s].total() #use the number of samples from the generating dataset
      else:
          try:
              N = nSamples[k] #try to treat nSamples as a list
          except:
              N = nSamples #if not indexable, nSamples should be a single number
      
      #Weight the number of samples according to a WeightedGateString
      if isinstance(s, _gs.WeightedGateString):
          nWeightedSamples = int(round(s.weight * N))
      else:
          nWeightedSamples = N
          
      counts = { }
      if sampleError == "binomial":
          assert(len(ps.keys()) == 2)
          spamLabel1, spamLabel2 = ps.keys(); p1 = ps[spamLabel1]
          if p1 < 0 and abs(p1) < 1e-6: p1 = 0
          if p1 > 1 and abs(p1-1.0) < 1e-6: p1 = 1
          if p1 < 0 or p1 > 1: print "Warning: probability == %g" % p1
          p1 = _np.clip(p1,0,1)
          counts[spamLabel1] = _rndm.binomial(nWeightedSamples, p1) #numpy.clip(p1,0,1) )
          counts[spamLabel2] = nWeightedSamples - counts[spamLabel1]
      elif sampleError == "multinomial":
          nOutcomes = len(ps.keys())
          countsArray = _rndm.multinomial(nWeightedSamples, ps.values(), size=1)
          for i,spamLabel in enumerate(ps.keys()):
              counts[spamLabel] = countsArray[0,i]
      else:
          for (spamLabel,p) in ps.iteritems():
              pc = _np.clip(p,0,1)
              if sampleError == "none":
                  counts[spamLabel] = float(nWeightedSamples * pc)
              elif sampleError == "round":
                  counts[spamLabel] = int(round(nWeightedSamples*pc))
              else: raise ValueError("Invalid sample error parameter: '%s'  Valid options are 'none', 'round', 'binomial', or 'multinomial'" % sampleError)

      dataset.addCountDict(s, counts)
    dataset.doneAddingData()
    return dataset



def printGatesetInfo(gateset):
    """
    Print to stdout relevant information about a gateset, 
      including the Choi matrices and their eigenvalues.
    
    Parameters
    ----------
    gateset : GateSet
        The gate set to print information about.

    Returns
    -------
    None
    """
    print gateset
    print "\n"
    print "Choi Matrices:"
    for (label,gate) in gateset.iteritems():
        print "Choi(%s) in pauli basis = \n" % label, 
        _tools.complexMxToString(_tools.opWithJamiolkowskiIsomorphism(gate))
        print "  --eigenvals = ", sorted( 
            [ev.real for ev in _np.linalg.eigvals(
                    _tools.opWithJamiolkowskiIsomorphism(gate))] ),"\n"
    print "Sum of negative Choi eigenvalues = ", _tools.sumOfNegativeJEvals(gateset)

    #OLD, and requires likelihoodfunctions...
    #rhoVecPenalty = sum( [ _lf.rhoVecPenalty(rhoVec) for rhoVec in gateset.rhoVecs ] )
    #EVecPenalty   = sum( [ _lf.EVecPenalty(EVec)     for EVec   in gateset.EVecs ] )
    #print "rhoVec Penalty (>0 if invalid rhoVecs) = ", rhoVecPenalty
    #print "EVec Penalty (>0 if invalid EVecs) = ", EVecPenalty


#def interpretGatesetInfo(gateset):
#    print "\n"
#    print "Interpeted Info:"
#    for (label,gate) in gateset.iteritems():
#        gate_evals,gate_evecs = _np.linalg.eig(gate)
#        choi = _tools.opWithJamiolkowskiIsomorphism(gate)
#        choi_evals = [ ev.real for ev in _np.linalg.eigvals(choi) ]
#        print "Gate: ", label
#        print " -- eigenvals = ", gate_evals
#        print " -- eigenvecs = "; _MOps.printMx(gate_evecs)
#        print " -- choi eigenvals = ", sorted(choi_evals)
#        print " -- choi trace = ", _MOps.trace(choi)
