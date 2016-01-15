""" Functions for reducing the number of required fiducial pairs for analysis."""
import numpy as _np
import itertools as _itertools
import math as _math
import sys as _sys
from ..construction import gatestringconstruction as _gsc


def _nCr(n,r):
    """Number of combinations of r items out of a set of n.  Equals n!/(r!(n-r)!)"""
    f = _math.factorial
    return f(n) / f(r) / f(n-r)

def getSufficientFiducialPairs(targetGateset, rhoStrs, EStrs, germList, 
                               testLs=(256,2048), spamLabels="all", tol=0.75,
                               verbosity=0, testPairList=None):

    """ Still in experimental stages.  TODO docstring. """
    #trim LSGST list of all f1+germ^exp+f2 strings to just those needed to get full rank jacobian. (compressed sensing like)
    
    #tol = 0.5 #fraction of expected amplification that must be observed to call a parameter "amplified"
    if spamLabels == "all":
        spamLabels = targetGateset.get_SPAM_labels()

    nGatesetParams = targetGateset.getNumParams(SPAM=True)

    #Compute all derivative info: get derivative of each <E_i|germ^exp|rho_j> where i = composite EVec & fiducial index and j similar
    def getDerivs(L):
        dP = _np.empty( (len(germList),len(spamLabels),len(EStrs)*len(rhoStrs), nGatesetParams) )
           #indexed by [iGerm,iSpamLabel,iFiducialPair,iGatesetParam] : gives d(<SP|f0+exp_iGerm+f1|AM>)/d(iGatesetParam)

        for iGerm,germ in enumerate(germList):
            expGerm = _gsc.repeatWithMaxLength(germ,L) # could pass exponent and set to germ**exp here
            lst = _gsc.createGateStringList("f0+expGerm+f1", f0=rhoStrs, f1=EStrs,
                                                         expGerm=expGerm, order=('f0','f1'))
            evTree = targetGateset.Bulk_evalTree(lst)
            dProbs = targetGateset.Bulk_dProbs(evTree,SPAM=True)
            for iSpamLabel,spamLabel in enumerate(spamLabels):
                dP[iGerm, iSpamLabel, :,:] = dProbs[spamLabel]
        return dP
    
    def getNumberAmplified(M0,M1,L0,L1,verb):
        L_ratio = float(L1)/float(L0)
        try:
            s0 = _np.linalg.svd(M0, compute_uv=False)
            s1 = _np.linalg.svd(M1, compute_uv=False)
        except:
            print "Warning: SVD error!!"
            return 0 #SVD did not converge -> just say no amplified params...

        numAmplified = 0
        if verb > 3: 
            print "Amplified parameter test: matrices are %s and %s." % (M0.shape, M1.shape)
            print "Index : SV(L=%d)  SV(L=%d)  AmpTest ( > %g ?)" % (L0,L1,tol)
        for i,(v0,v1) in enumerate(zip(sorted(s0,reverse=True),sorted(s1,reverse=True))):
            if abs(v0) > 0.1 and (v1/v0)/L_ratio > tol: 
                numAmplified += 1
                if verb > 3: print "%d: %g  %g  %g YES" % (i,v0,v1, (v1/v0)/L_ratio )
            elif verb > 3: print "%d: %g  %g  %g NO" % (i,v0,v1, (v1/v0)/L_ratio )
        return numAmplified
            
    #rank = len( [v for v in s if v > 0.001] )
 
    if verbosity > 0: 
        print "------  Fiducial Pair Reduction --------"
            
    L0 = testLs[0]; dP0 = getDerivs(L0)
    L1 = testLs[1]; dP1 = getDerivs(L1)
    fullTestMx0 = dP0.view(); fullTestMx0.shape = ( (len(germList)*len(spamLabels)*len(rhoStrs)*len(EStrs), nGatesetParams) )
    fullTestMx1 = dP1.view(); fullTestMx1.shape = ( (len(germList)*len(spamLabels)*len(rhoStrs)*len(EStrs), nGatesetParams) )        

    #Get number of amplified parameters in the "full" test matrix: the one we get when we use all possible fiducial pairs
    if testPairList is None: 
        maxAmplified = getNumberAmplified(fullTestMx0, fullTestMx1, L0, L1, verbosity+1)
        if verbosity > 0: print "maximum number of amplified parameters = ",maxAmplified

    #Loop through fiducial pairs and add all derivative rows (1 x nGatesetParams) to test matrix
    # then check if testMatrix has full rank ( == nGatesetParams)
    fiducialPairs = []

    nPossiblePairs = len(rhoStrs)*len(EStrs)
    allPairIndices = range(nPossiblePairs)

    nRhoStrs, nEStrs = len(rhoStrs), len(EStrs)
    germFctr = len(spamLabels)*len(rhoStrs)*len(EStrs); nGerms = len(germList)
    spamLabelFctr = len(rhoStrs)*len(EStrs); nSpamLabels = len(spamLabels)
    gateStringIndicesForPair = []
    for i in allPairIndices:
        indices = [ iGerm*germFctr + iSpamLabel*spamLabelFctr + i  for iGerm in xrange(nGerms) for iSpamLabel in xrange(nSpamLabels) ]
        gateStringIndicesForPair.append(indices)

    if testPairList is not None: #special mode for testing/debugging single pairlist
        gateStringIndicesForPairs = []
        for iRhoStr,iEStr in testPairList:
            gateStringIndicesForPairs.extend( gateStringIndicesForPair[iRhoStr*nEStrs + iEStr] )
        testMx0 = _np.take( fullTestMx0, gateStringIndicesForPairs, axis=0 )
        testMx1 = _np.take( fullTestMx1, gateStringIndicesForPairs, axis=0 )
        nAmplified = getNumberAmplified(testMx0, testMx1, L0, L1, verbosity)
        print "Number of amplified parameters = ",nAmplified
        return None

    for nNeededPairs in range(1,nPossiblePairs):
        if verbosity > 0: 
            print "Beginning search for a good set of %d pairs (%d pair lists to test)" % \
                (nNeededPairs,_nCr(nPossiblePairs,nNeededPairs))
            _sys.stdout.flush()

        bestAmplified = 0
        for pairIndicesToTest in _itertools.combinations(allPairIndices, nNeededPairs):
            gateStringIndicesForPairs = []
            for i in pairIndicesToTest:
                gateStringIndicesForPairs.extend( gateStringIndicesForPair[i] )
            testMx0 = _np.take( fullTestMx0, gateStringIndicesForPairs, axis=0 )
            testMx1 = _np.take( fullTestMx1, gateStringIndicesForPairs, axis=0 )
            nAmplified = getNumberAmplified(testMx0, testMx1, L0, L1, verbosity)
            bestAmplified = max(bestAmplified, nAmplified)
            if verbosity > 1:
                ret = []
                for i in pairIndicesToTest:
                    iRhoStr = i // nEStrs
                    iEStr   = i - iRhoStr*nEStrs
                    ret.append( (iRhoStr,iEStr) )
                print "Pair list %s ==> %d amplified parameters" % (" ".join(map(str,ret)), nAmplified)

            if nAmplified == maxAmplified:
                ret = []
                for i in pairIndicesToTest:
                    iRhoStr = i // nEStrs
                    iEStr   = i - iRhoStr*nEStrs
                    ret.append( (iRhoStr,iEStr) )
                return ret
        if verbosity > 0: print " --> Higheset number of amplified parameters was %d" % bestAmplified

    #if we tried all the way to nPossiblePairs-1 and no success, just return all the pairs
    listOfAllPairs = [ (iRhoStr,iEStr) for iRhoStr in xrange(nRhoStrs) for iEStr in xrange(nEStrs) ]
    return listOfAllPairs





#def _old_TestPair(targetGateset, fiducialList, germList, L, testPairList, spamLabels="all"):
#
#    if spamLabels == "all":
#        spamLabels = targetGateset.get_SPAM_labels()
#
#    dProbs = []
#    for iGerm,germ in enumerate(germList):
#        expGerm = _gsc.repeatWithMaxLength(germ,L)
#        lst = _gsc.createGateStringList("f0+expGerm+f1", f0=fiducialList, f1=fiducialList,
#                                        expGerm=expGerm, order=('f0','f1'))
#        evTree = targetGateset.Bulk_evalTree(lst)
#        dProbs.append( targetGateset.Bulk_dProbs(evTree,SPAM=True) )
#
#    nGatesetParams = targetGateset.getNumParams(SPAM=True)
#    testMatrix = _np.empty( (0,nGatesetParams) )
#    for (i0,i1) in testPairList: #[(0,0),(1,0),(2,3),(4,5)]:
#        iCmp = i0*len(fiducialList) + i1 #composite index of (f0,f1) in dProbs[iGerm][spamLabel]
#
#        for iGerm,germ in enumerate(germList):
#            for spamLabel in spamLabels:
#                testMatrix = _np.concatenate( (testMatrix, dProbs[iGerm][spamLabel][iCmp:iCmp+1,:] ), axis=0 )
#                
#        U,s,V = _np.linalg.svd(testMatrix)
#        rank = len( [v for v in s if v > 0.001] )
#        sorteds = sorted(s,reverse=True)
#        #print "DEBUG: added (%d,%d): testMx=%s, rank=%d, iCmp=%d, s =\n%s\n" % (i0,i1,testMatrix.shape,rank,iCmp,'\n'.join(map(str,enumerate(sorted(s,reverse=True)))))
#    
#    print "Singular values:\n", '\n'.join(map(str,enumerate(sorteds)))
#    #print "normalized 34th singular val = ",sorteds[33]/len(pairList)
#    #print "normalized 35th singular val = ",sorteds[34]/len(pairList)
