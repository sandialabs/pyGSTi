from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for reducing the number of required fiducial pairs for analysis."""

import numpy     as _np
import itertools as _itertools
import math      as _math
from ..construction import gatestringconstruction as _gsc
from ..tools        import remove_duplicates      as _remove_duplicates

from ..             import objects as _objs

def _nCr(n,r):
    """Number of combinations of r items out of a set of n.  Equals n!/(r!(n-r)!)"""
    f = _math.factorial
    return f(n) / f(r) / f(n-r)

def find_sufficient_fiducial_pairs(targetGateset, prepStrs, effectStrs, germList,
                                   testLs=(256,2048), spamLabels="all", tol=0.75,
                                   searchMode="sequential", nRandom=100, seed=None,
                                   verbosity=0, testPairList=None, memLimit=None):

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    """ Still in experimental stages.  TODO docstring. """
    #trim LSGST list of all f1+germ^exp+f2 strings to just those needed to get full rank jacobian. (compressed sensing like)

    #tol = 0.5 #fraction of expected amplification that must be observed to call a parameter "amplified"
    if spamLabels == "all":
        spamLabels = targetGateset.get_spam_labels()

    nGatesetParams = targetGateset.num_params()

    #Compute all derivative info: get derivative of each <E_i|germ^exp|rho_j> where i = composite EVec & fiducial index and j similar
    def get_derivs(L):
        dP = _np.empty( (len(germList),len(spamLabels),len(effectStrs)*len(prepStrs), nGatesetParams) )
           #indexed by [iGerm,iSpamLabel,iFiducialPair,iGatesetParam] : gives d(<SP|f0+exp_iGerm+f1|AM>)/d(iGatesetParam)

        for iGerm,germ in enumerate(germList):
            expGerm = _gsc.repeat_with_max_length(germ,L) # could pass exponent and set to germ**exp here
            lst = _gsc.create_gatestring_list("f0+expGerm+f1", f0=prepStrs, f1=effectStrs,
                                                         expGerm=expGerm, order=('f0','f1'))
            evTree = targetGateset.bulk_evaltree(lst)
            blkSz = None

            if memLimit is not None:
                #nSpam = targetGateset.num_preps() * targetGateset.num_effects()
                dim = targetGateset.get_dimension()
                nParams = targetGateset.num_params()
                memEstimate = 8.0*len(evTree)*dim**2*nParams #*nSPAM
                if memEstimate > memLimit:
                    printer.log("Germ %d/%d: Memory estimate = %.1fGB > %.1fGB" % \
                        (iGerm+1,len(germList),memEstimate/(1024.0**3),memLimit/(1024.0**3)))
                    if memEstimate/nParams < memLimit:
                        blkSz = int(nParams * memLimit / memEstimate)
                        printer.log(" --> Setting max block size = %s" % blkSz)
                    else:
                        # Tree and deriv splitting method
                        blkSz = 1
                        maxTreeLength = len(evTree) * memLimit / (memEstimate / nParams)
                        evTree.split(maxTreeLength) # Often fails because maxTreeLength is too small
                        printer.log(" --> Setting max subtree size = %s" % maxTreeLength)
                        evTree.print_analysis()

            dprobs = targetGateset.bulk_dprobs(evTree, wrtBlockSize=blkSz)
            for iSpamLabel,spamLabel in enumerate(spamLabels):
                dP[iGerm, iSpamLabel, :,:] = dprobs[spamLabel]
        return dP

    def get_number_amplified(M0,M1,L0,L1,verb):
        printer = _objs.VerbosityPrinter.build_printer(verb)
        L_ratio = float(L1)/float(L0)
        try:
            s0 = _np.linalg.svd(M0, compute_uv=False)
            s1 = _np.linalg.svd(M1, compute_uv=False)
        except:
            printer.warning("SVD error!!"); return 0
            #SVD did not converge -> just say no amplified params...

        numAmplified = 0
        printer.log("Amplified parameter test: matrices are %s and %s." % (M0.shape, M1.shape), 4)
        printer.log("Index : SV(L=%d)  SV(L=%d)  AmpTest ( > %g ?)" % (L0,L1,tol), 4)
        for i,(v0,v1) in enumerate(zip(sorted(s0,reverse=True),sorted(s1,reverse=True))):
            if abs(v0) > 0.1 and (v1/v0)/L_ratio > tol:
                numAmplified += 1
                printer.log("%d: %g  %g  %g YES" % (i,v0,v1, (v1/v0)/L_ratio ), 4)
            printer.log("%d: %g  %g  %g NO" % (i,v0,v1, (v1/v0)/L_ratio ), 4)
        return numAmplified

    #rank = len( [v for v in s if v > 0.001] )


    printer.log("------  Fiducial Pair Reduction --------")

    L0 = testLs[0]; dP0 = get_derivs(L0)
    L1 = testLs[1]; dP1 = get_derivs(L1)
    fullTestMx0 = dP0.view(); fullTestMx0.shape = ( (len(germList)*len(spamLabels)*len(prepStrs)*len(effectStrs), nGatesetParams) )
    fullTestMx1 = dP1.view(); fullTestMx1.shape = ( (len(germList)*len(spamLabels)*len(prepStrs)*len(effectStrs), nGatesetParams) )

    #Get number of amplified parameters in the "full" test matrix: the one we get when we use all possible fiducial pairs
    if testPairList is None:
        maxAmplified = get_number_amplified(fullTestMx0, fullTestMx1, L0, L1, verbosity+1)
        printer.log("maximum number of amplified parameters = %s" % maxAmplified)

    #Loop through fiducial pairs and add all derivative rows (1 x nGatesetParams) to test matrix
    # then check if testMatrix has full rank ( == nGatesetParams)

    nPossiblePairs = len(prepStrs)*len(effectStrs)
    allPairIndices = list(range(nPossiblePairs))

    nRhoStrs, nEStrs = len(prepStrs), len(effectStrs)
    germFctr = len(spamLabels)*len(prepStrs)*len(effectStrs); nGerms = len(germList)
    spamLabelFctr = len(prepStrs)*len(effectStrs); nSpamLabels = len(spamLabels)
    gateStringIndicesForPair = []
    for i in allPairIndices:
        indices = [ iGerm*germFctr + iSpamLabel*spamLabelFctr + i  for iGerm in range(nGerms) for iSpamLabel in range(nSpamLabels) ]
        gateStringIndicesForPair.append(indices)

    if testPairList is not None: #special mode for testing/debugging single pairlist
        gateStringIndicesForPairs = []
        for iRhoStr,iEStr in testPairList:
            gateStringIndicesForPairs.extend( gateStringIndicesForPair[iRhoStr*nEStrs + iEStr] )
        testMx0 = _np.take( fullTestMx0, gateStringIndicesForPairs, axis=0 )
        testMx1 = _np.take( fullTestMx1, gateStringIndicesForPairs, axis=0 )
        nAmplified = get_number_amplified(testMx0, testMx1, L0, L1, verbosity)
        printer.log("Number of amplified parameters = %s" % nAmplified)
        return None

    bestAmplified = 0
    for nNeededPairs in range(1,nPossiblePairs):
        printer.log("Beginning search for a good set of %d pairs (%d pair lists to test)" % \
                (nNeededPairs,_nCr(nPossiblePairs,nNeededPairs)))

        bestAmplified = 0
        if searchMode == "sequential":
            pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs)

        elif searchMode == "random":
            rand = _np.random.RandomState(seed)  # ok if seed is None
            nTotalPairCombos = _nCr(len(allPairIndices), nNeededPairs)
            if nRandom < nTotalPairCombos:
                randIndices = _remove_duplicates(sorted(rand.randint(0,nTotalPairCombos,size=nRandom)))
            else:
                randIndices = list(range(int(nTotalPairCombos)))

            def filterAll(it): #generator which filters iterator "it" using randIndices
                nxt = 0
                for i,val in enumerate(it):
                    if i == randIndices[nxt]:
                        yield val
                        nxt += 1
                        if nxt == len(randIndices): break

            pairIndicesToIterateOver = filterAll(_itertools.combinations(allPairIndices, nNeededPairs))


        for pairIndicesToTest in pairIndicesToIterateOver:
            gateStringIndicesForPairs = []
            for i in pairIndicesToTest:
                gateStringIndicesForPairs.extend( gateStringIndicesForPair[i] )
            testMx0 = _np.take( fullTestMx0, gateStringIndicesForPairs, axis=0 )
            testMx1 = _np.take( fullTestMx1, gateStringIndicesForPairs, axis=0 )
            nAmplified = get_number_amplified(testMx0, testMx1, L0, L1, verbosity)
            bestAmplified = max(bestAmplified, nAmplified)
            if printer.verbosity > 1:
                ret = []
                for i in pairIndicesToTest:
                    iRhoStr = i // nEStrs
                    iEStr   = i - iRhoStr*nEStrs
                    ret.append( (iRhoStr,iEStr) )
                printer.log("Pair list %s ==> %d amplified parameters" % (" ".join(map(str,ret)), nAmplified))

            if nAmplified == maxAmplified:
                ret = []
                for i in pairIndicesToTest:
                    iRhoStr = i // nEStrs
                    iEStr   = i - iRhoStr*nEStrs
                    ret.append( (iRhoStr,iEStr) )
                return ret

    printer.log(" --> Highest number of amplified parameters was %d" % bestAmplified)

    #if we tried all the way to nPossiblePairs-1 and no success, just return all the pairs, which by definition will hit the "max-amplified" target
    listOfAllPairs = [ (iRhoStr,iEStr)
                       for iRhoStr in range(nRhoStrs)
                       for iEStr in range(nEStrs) ]
    return listOfAllPairs





#def _old_TestPair(targetGateset, fiducialList, germList, L, testPairList, spamLabels="all"):
#
#    if spamLabels == "all":
#        spamLabels = targetGateset.get_spam_labels()
#
#    dprobs = []
#    for iGerm,germ in enumerate(germList):
#        expGerm = _gsc.repeat_with_max_length(germ,L)
#        lst = _gsc.create_gatestring_list("f0+expGerm+f1", f0=fiducialList, f1=fiducialList,
#                                        expGerm=expGerm, order=('f0','f1'))
#        evTree = targetGateset.bulk_evaltree(lst)
#        dprobs.append( targetGateset.bulk_dprobs(evTree) )
#
#    nGatesetParams = targetGateset.num_params()
#    testMatrix = _np.empty( (0,nGatesetParams) )
#    for (i0,i1) in testPairList: #[(0,0),(1,0),(2,3),(4,5)]:
#        iCmp = i0*len(fiducialList) + i1 #composite index of (f0,f1) in dprobs[iGerm][spamLabel]
#
#        for iGerm,germ in enumerate(germList):
#            for spamLabel in spamLabels:
#                testMatrix = _np.concatenate( (testMatrix, dprobs[iGerm][spamLabel][iCmp:iCmp+1,:] ), axis=0 )
#
#        U,s,V = _np.linalg.svd(testMatrix)
#        rank = len( [v for v in s if v > 0.001] )
#        sorteds = sorted(s,reverse=True)
#        #print "DEBUG: added (%d,%d): testMx=%s, rank=%d, iCmp=%d, s =\n%s\n" % (i0,i1,testMatrix.shape,rank,iCmp,'\n'.join(map(str,enumerate(sorted(s,reverse=True)))))
#
#    print "Singular values:\n", '\n'.join(map(str,enumerate(sorteds)))
#    #print "normalized 34th singular val = ",sorteds[33]/len(pairList)
#    #print "normalized 35th singular val = ",sorteds[34]/len(pairList)
