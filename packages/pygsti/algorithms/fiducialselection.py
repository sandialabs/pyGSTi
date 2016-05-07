#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Functions for selecting a complete set of fiducials for a GST analysis."""

import numpy as _np
import itertools as _itertools
import math as _math
import sys as _sys

#def bool_list_to_ind_list(boolList):
#    output = _np.array([])
#    for i, boolVal in boolList:
#        if boolVal == 1:
#            output = _np.append(i)
#    return output

def make_prep_mxs(gs,prepFidList):
    dimRho = gs.get_dimension()
    numRho = len(gs.preps)
    numFid = len(prepFidList)
    outputMatList = []
    for rho in gs.preps.values():
        outputMat = _np.zeros([dimRho,numFid],float)
        counter = 0
        for prepFid in prepFidList:
            outputMat[:,counter] = _np.dot(gs.product(prepFid),rho).T[0]
            counter += 1
        outputMatList.append(outputMat)
    return outputMatList

def make_meas_mxs(gs,prepMeasList):
    dimE = gs.get_dimension()
    numE = len(gs.effects)
    numFid = len(prepMeasList)
    outputMatList = []
    for E in gs.effects.values():
        outputMat = _np.zeros([dimE,numFid],float)
        counter = 0
        for measFid in prepMeasList:
            outputMat[:,counter] = _np.dot(E.T,gs.product(measFid))[0]
            counter += 1
        outputMatList.append(outputMat)
    return outputMatList


            
#    outputMat = _np.zeros([dimRho,numRho*numFid],float)
#    counter = 0
#    for prepFid in prepFidList:
#        for rho in gs.preps.values():
#            outputMat[:,counter] = _np.dot(gs.product(prepFid),rho).T[0]
#            counter += 1
#    return outputMat
#    SqOutputMat = _np.dot(outputMat,outputMat.T)
#    return SqOutputMat

# def make_meas_mxs(gs,measFidList):
#     dimE = gs.get_dimension()
#     numE = len(gs.effects)
#     numFid = len(measFidList)
#     outputMat = _np.zeros([dimE,numE*numFid],float)
#     counter = 0
#     for measFid in measFidList:
#         for E in gs.effects.values():
#             outputMat[:,counter] = _np.dot(E.T,gs.product(measFid))[0]
#             counter += 1
#     SqOutputMat = _np.dot(outputMat,outputMat.T)
#     return SqOutputMat

#def score_fid_list(gs,fidList,kind=None):
#    if kind not in ('prep', 'meas'):
#        raise ValueError("Need to specify 'prep' or 'meas' for kind!")
#    if kind == 'prep':
#        matToScore = make_prep_mxs(gs,fidList)
#    else:
#        matToScore = make_meas_mxs(gs,fidList)
#    score = len(fidList) * _np.sum(1./_np.linalg.eigvalsh(matToScore))
#    return score


def test_fiducial_list(gateset,fidList,prepOrMeas=None,returnSpectrum=False,threshold=1e3):
    nFids = len(fidList)

    dimRho = gateset.get_dimension()

    fidLengths = _np.array( map(len,fidList), 'i')
    if prepOrMeas == 'prep':
        fidArrayList = make_prep_mxs(gateset,fidList)
    elif prepOrMeas == 'meas':
        fidArrayList = make_meas_mxs(gateset,fidList)
    else:
        raise Exception('prepOrMeas must be specified!')
    numMxs = len(fidArrayList)

    numFids = len(fidList)
    scoreMx = _np.zeros([dimRho,numFids *  numMxs],float)
    colInd = 0
#    wts = _np.array(wts)
#    wtsLoc = _np.where(wts)[0]
    for fidArray in fidArrayList:
        scoreMx[:,colInd:colInd+numFids] = fidArray
        colInd += numFids
    scoreSqMx = _np.dot(scoreMx,scoreMx.T)
    score = numFids * _np.sum(1./_np.linalg.eigvalsh(scoreSqMx))
    if (score <= 0 or _np.isinf(score)) or score > threshold:
        if returnSpectrum:
            return False, numFids * 1./_np.linalg.eigvalsh(scoreSqMx)
        else:
            return False
    else:
        if returnSpectrum:
            return True, numFids * 1./_np.linalg.eigvalsh(scoreSqMx)
        else:
            return True
    


                        

def optimize_integer_fiducials_slack(gateset, fidList, 
                              prepOrMeas = None,
                              initialWeights=None, 
                              maxIter=100, 
                              fixedSlack=False, slackFrac=False, 
                              returnAll=False, tol=1e-6, 
                              forceEmpty = True, forceEmptyScore = 1e100,
                              verbosity=1):
    """
    Find a locally optimal subset of the fiducials in fidList.

    Locally optimal here means that no single fid can be excluded
    without making the smallest non-gauge eigenvalue of the 
    Jacobian.H*Jacobian matrix smaller, i.e. less amplified,
    by more than a fixed or variable amount of "slack", as
    specified by fixedSlack or slackFrac.

    Parameters
    ----------
    gateset : GateSet
        The gate set (associates gate matrices with gate labels).

    fidList : list of GateStrings
        List of all fiducials gate sequences to consider.

    initialWeights : list-like
        List or array of either booleans or (0 or 1) integers
        specifying which fiducials in fidList comprise the initial
        fiduial set.  If None, then starting point includes all
        fiducials.

    gates : bool or list, optional
        Whether/which gates' parameters should be included as gateset
        parameters *and* considered as part of the gateset space.

      - True = all gates
      - False = no gates
      - list of gate labels = those particular gates.

    maxIter : int, optional
        The maximum number of iterations before giving up.

    fixedSlack : float, optional
        If not None, a floating point number which specifies that excluding a 
        fiducial is allowed to increase 1.0/smallest-non-gauge-eigenvalue by
        fixedSlack.  You must specify *either* fixedSlack or slackFrac.

    slackFrac : float, optional
        If not None, a floating point number which specifies that excluding a 
        fiducial is allowed to increase 1.0/smallest-non-gauge-eigenvalue by
        fixedFrac*100 percent.  You must specify *either* fixedSlack or slackFrac.

    returnAll : bool, optional
        If True, return the final "weights" vector and score dictionary
        in addition to the optimal fiducial list (see below).

    tol : float, optional
        Tolerance used for eigenvector degeneracy testing in twirling operation.

    verbosity : int, optional
        Integer >= 0 indicating the amount of detail to print.


    Returns
    -------
    finalFidList : list
        Sublist of fidList specifying the final, optimal, set of fiducials.

    weights : array
        Integer array, of length len(fidList), containing 0s and 1s to
        indicate which elements of fidList were chosen as finalFidList.
        Only returned when returnAll == True.

    scoreDictionary : dict
        Dictionary with keys == tuples of 0s and 1s of length len(fidList),
        specifying a subset of fiducials, and values == 1.0/smallest-non-gauge-
        eigenvalue "scores".  
    """
    if (fixedSlack and slackFrac) or (not fixedSlack and not slackFrac):
        raise ValueError("Either fixedSlack *or* slackFrac should be specified")
    lessWeightOnly = False  #Initially allow adding to weight. -- maybe make this an argument??

    nFids = len(fidList)

    dimRho = gateset.get_dimension()

    if verbosity > 0:
        print "Starting fiducial set optimization. Lower score is better."

    scoreD = {} 

    fidLengths = _np.array( map(len,fidList), 'i')
    if prepOrMeas == 'prep':
        fidArrayList = make_prep_mxs(gateset,fidList)
    elif prepOrMeas == 'meas':
        fidArrayList = make_meas_mxs(gateset,fidList)
    else:
        raise Exception('prepOrMeas must be specified!')
    numMxs = len(fidArrayList)
        
    def compute_score(wts):
        if forceEmpty and _np.count_nonzero(wts[:1]) != 1:
            score = forceEmptyScore
        else:
            numFids = _np.sum(wts)
            scoreMx = _np.zeros([dimRho,numFids *  numMxs],float)
            colInd = 0
            wts = _np.array(wts)
            wtsLoc = _np.where(wts)[0]
            for fidArray in fidArrayList:
                scoreMx[:,colInd:colInd+numFids] = fidArray[:,wtsLoc]
                colInd += numFids
            scoreSqMx = _np.dot(scoreMx,scoreMx.T)
            score = numFids * _np.sum(1./_np.linalg.eigvalsh(scoreSqMx))
            if score <= 0 or _np.isinf(score):
                score = 1e10
        scoreD[tuple(wts)] = score
        return score

    def get_neighbors(boolVec):
        for i in xrange(nFids):
            v = boolVec.copy()
            v[i] = (v[i] + 1) % 2 #toggle v[i] btwn 0 and 1
            yield v

    if initialWeights is not None:
        weights = _np.array( [1 if x else 0 for x in initialWeights ] )
    else:
        weights = _np.ones( nFids, 'i' ) #default: start with all germs
        lessWeightOnly = True #we're starting at the max-weight vector

    score = compute_score(weights)
    L1 = sum(weights) # ~ L1 norm of weights

    for iIter in xrange(maxIter):
        scoreD_keys = scoreD.keys() #list of weight tuples already computed

        if verbosity > 0:
            print "Iteration %d: score=%g, nFids=%d" % (iIter, score, L1)
        
        bFoundBetterNeighbor = False
        for neighborNum, neighbor in enumerate(get_neighbors(weights)):
            if tuple(neighbor) not in scoreD_keys:
                neighborL1 = sum(neighbor)
                neighborScore = compute_score(neighbor)
            else:
                neighborL1 = sum(neighbor)
                neighborScore = scoreD[tuple(neighbor)]

            #Move if we've found better position; if we've relaxed, we only move when L1 is improved.
            if neighborScore <= score and (neighborL1 < L1 or lessWeightOnly == False):
                weights, score, L1 = neighbor, neighborScore, neighborL1
                bFoundBetterNeighbor = True
                if verbosity > 1: print "Found better neighbor: nFids = %d score = %g" % (L1,score)


        if not bFoundBetterNeighbor: # Time to relax our search.
            lessWeightOnly=True #from now on, don't allow increasing weight L1

            if fixedSlack==False:
                slack = score*slackFrac #Note score is positive (for sum of 1/lambda)
            else:
                slack = fixedSlack
            assert(slack > 0)

            if verbosity > 1:
                print "No better neighbor. Relaxing score w/slack: %g => %g" % (score, score+slack)
            score += slack #artificially increase score and see if any neighbor is better now...

            for neighborNum, neighbor in enumerate(get_neighbors(weights)):
                if sum(neighbor) < L1 and scoreD[tuple(neighbor)] < score:
                    weights, score, L1 = neighbor, scoreD[tuple(neighbor)], sum(neighbor)
                    bFoundBetterNeighbor = True
                    if verbosity > 1: print "Found better neighbor: nFids = %d score = %g" % (L1,score)

            if not bFoundBetterNeighbor: #Relaxing didn't help!
                print "Stationary point found!";
                break #end main for loop
        
        print "Moving to better neighbor"
    else:
        print "Hit max. iterations"
    
    print "score = ", score
    print "weights = ",weights
    print "L1(weights) = ",sum(weights)

    goodFidList = []
    for index,val in enumerate(weights):
        if val==1:
            goodFidList.append(fidList[index])

    if returnAll:
        return goodFidList, weights, scoreD
    else:
        return goodFidList
