""" Gate string list creation functions using truncated repeated-germ strings."""
import GST as _GST
import numpy as _np
import itertools as _itertools

def make_lsgst_lists(gateLabels, fiducialList, germList, maxLengthList, rhoEPairs=None):
    """
    Create a set of gate string lists for LSGST based on germs and max-lengths.

    Constructs a series of successively larger lists by iteratively adding to
    to a running list.  If maxLengthList[0] == 0 then the starting list is the 
    list of LGST strings, otherwise the starting list is empty.  For each 
    nonzero element of maxLengthList, call it L, add strings of the form:
    
    fiducial1 + GST.GateStringTools.repeatAndTruncate(germ,L) + fiducial2

    to the running list and add the resulting list to the list (with 
    duplicates removed) of gate string lists that is ultimately returned.

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels to determine needed LGST strings.  Only relevant
        when maxLengthList[0] == 0.

    fiducialList : list of GateStrings
        List of the fiducial gate strings.
        
    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths.  If maxLengthList[0] == 0 this results in 
        special behavior where LGST strings are included as the first 
        returned list.

    rhoEPairs : list of 2-tuples, optional
        Specifies a subset of all fiducial string pairs (fiducial1, fiducial2)
        to be used in the gate string lists.  Each element of rhoEPairs is a 
        (iFiducial1, iFidicial2) 2-tuple of integers, each indexing a string
        within fiducialList so that fiducial1 = fiducialList[iFiducial1], etc.

    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing
        repeated germs truncated to maxLengthList[0] through (and including)
        maxLengthList[i].  Note that a "0" exponent corresponds to the LGST
        strings.
    """
    return make_lsgst_lists_asymmetric_fids(gateLabels, fiducialList, fiducialList, germList, maxLengthList, rhoEPairs)

def make_lsgst_lists_asymmetric_fids(gateLabels, rhoStrs, EStrs, germList, maxLengthList, rhoEPairs=None):
    '''
    Same as make_lsgst_lists, except for asymmetric fiducial sets, specified by rhoStrs and EStrs.
    '''
    lgstStrings = _GST.listLGSTGateStrings(_GST.getRhoAndESpecs(rhoStrs = rhoStrs, EStrs = EStrs), gateLabels)
    lsgst_list = _GST.gateStringList([ () ]) #running list of all strings so far

    if rhoEPairs is not None:
        fiducialPairs = [ (rhoStrs[i],EStrs[j]) for (i,j) in rhoEPairs ]
    else:
        fiducialPairs = list(_itertools.product(rhoStrs, EStrs))
    
    if maxLengthList[0] == 0:
        lsgst_listOfLists = [ lgstStrings ]
        maxLengthList = maxLengthList[1:]
    else: lsgst_listOfLists = [ ]
        
    for maxLen in maxLengthList:
        lsgst_list += _GST.createGateStringList("f[0]+R(germ,N)+f[1]", f=fiducialPairs,
                                           germ=germList, N=maxLen,
                                           R=_GST.GateStringTools.repeatAndTruncate,
                                           order=('germ','f'))
        lsgst_listOfLists.append( _GST.ListTools.remove_duplicates(lgstStrings + lsgst_list) )

    #print "%d LSGST sets w/lengths" % len(lsgst_listOfLists),map(len,lsgst_listOfLists)
    return lsgst_listOfLists


def make_elgst_lists(gateLabels, germList, maxLengthList):
    """
    Create a set of gate string lists for eLGST based on germs and max-lengths.

    Constructs a series of successively larger lists by iteratively adding to
    to a running list.  If maxLengthList[0] == 0 then the starting list is the 
    list of length-1 gate label strings, otherwise the starting list is empty.
    For each nonzero element of maxLengthList, call it L, add strings of the form:
    
    GST.GateStringTools.repeatAndTruncate(germ,L)

    to the running list, and add the resulting list (with duplicates removed) 
    to the list of gate string lists that is ultimately returned.

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels .Only relevant when expList[0] == 0.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of the maximum lengths.  If maxLengthList[0] == 0 this results in
        special behavior where the length-1 gate label strings are included as
        the first returned list.

    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing
        repeated germs truncated to maxLengthList[0] through (and including)
        maxLengthList[i].  Note that a "0" exponent corresponds to the gate
        label strings.
    """
    singleGates = _GST.gateStringList([(g,) for g in gateLabels])
    elgst_list = _GST.gateStringList([ () ])  #running list of all strings so far
    
    if maxLengthList[0] == 0:
        elgst_listOfLists = [ singleGates ]
        maxLengthList = maxLengthList[1:]
    else: elgst_listOfLists = [ ]
        
    for maxLen in maxLengthList:
        elgst_list += _GST.createGateStringList("R(germ,N)", germ=germList, N=maxLen,
                                           R=_GST.GateStringTools.repeatAndTruncate)
        elgst_listOfLists.append( _GST.ListTools.remove_duplicates(singleGates + elgst_list) )

    #print "%d eLGST sets w/lengths" % len(elgst_listOfLists),map(len,elgst_listOfLists)
    return elgst_listOfLists


def getSufficientFiducialPairs(targetGateset, fiducialList, germList, L, spamLabels="all", pairList=None):
    """ Still in experimental stages -- TODO docstring """
    #trim LSGST or eLGST list of all f1+germ^exp+f2 strings to just those needed to get full rank jacobian. (compressed sensing like)
    
    if spamLabels == "all":
        spamLabels = targetGateset.get_SPAM_labels()

    #Compute all derivative info: get derivative of each <E_i|germ^exp|rho_j> where i = composite EVec & fiducial index and j similar
    dProbs = [] #indexed by [iGerm][spamLabel] to get mx of dPr's for (f0,f1) loop
    for iGerm,germ in enumerate(germList):
        expGerm = _GST.GateStringTools.repeatWithMaxLength(germ,L) # could pass exponent and set to germ**exp here
        lst = _GST.createGateStringList("f0+expGerm+f1", f0=fiducialList, f1=fiducialList,
                                       expGerm=expGerm, order=('f0','f1'))
        evTree = targetGateset.Bulk_evalTree(lst)
        dProbs.append( targetGateset.Bulk_dProbs(evTree,SPAM=True) )

    #Loop through fiducial pairs and add all derivative rows (1 x nGatesetParams) to test matrix
    # then check if testMatrix has full rank ( == nGatesetParams)
    nGatesetParams = targetGateset.getNumParams(SPAM=True)
    testMatrix = _np.empty( (0,nGatesetParams) )
    fiducialPairs = []
    print "DEBUG: nParams = ",nGatesetParams

    #for i0,f0 in enumerate(fiducialList):
    #    for i1,f1 in enumerate(fiducialList):
    for (i0,i1) in pairList: #[(0,0),(1,0),(2,3),(4,5)]:
            #add (f0,f1) pair
            fiducialPairs.append( (i0,i1) )
            iCmp = i0*len(fiducialList) + i1 #composite index of (f0,f1) in dProbs[iGerm][spamLabel]

            for iGerm,germ in enumerate(germList):
                for spamLabel in spamLabels:
                    testMatrix = _np.concatenate( (testMatrix, dProbs[iGerm][spamLabel][iCmp:iCmp+1,:] ), axis=0 )

            U,s,V = _np.linalg.svd(testMatrix)
            rank = len( [v for v in s if v > 0.001] )
            sorteds = sorted(s,reverse=True)
            #print "DEBUG: added (%d,%d): testMx=%s, rank=%d, iCmp=%d, s =\n%s\n" % (i0,i1,testMatrix.shape,rank,iCmp,'\n'.join(map(str,enumerate(sorted(s,reverse=True)))))

            if rank >= nGatesetParams:
                return fiducialPairs

    print "normalized 34th singular val = ",sorteds[33]/len(pairList)
    print "normalized 35th singular val = ",sorteds[34]/len(pairList)
    #print "Warning: rank >= number of gateset params was never obtained!"
    #return fiducialPairs

    
