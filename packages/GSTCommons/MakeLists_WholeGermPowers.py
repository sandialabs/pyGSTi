""" Gate string list creation functions using repeated-germs limited by a max-length."""
import GST as _GST
import itertools as _itertools

def make_lsgst_lists(gateLabels, fiducialList, germList, maxLengthList, rhoEPairs=None):
    """
    Create a set of gate string lists for LSGST based on germs and max-lengths.

    Constructs a series of successively larger lists by iteratively adding to
    to a running list.  If maxLengthList[0] == 0 then the starting list is the 
    list of LGST strings, otherwise the starting list is empty.  For each 
    nonzero element of maxLengthList, call it L, add strings of the form:
    
    fiducial1 + GST.GateStringTools.repeatWithMaxLength(germ,L) + fiducial2

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
        repeated germs limited to lengths maxLengthList[0] through (and including)
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
                                                R=_GST.GateStringTools.repeatWithMaxLength,
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
    
    GST.GateStringTools.repeatWithMaxLength(germ,L)

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
        repeated germs limited to lengths maxLengthList[0] through (and including)
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
                                           R=_GST.GateStringTools.repeatWithMaxLength)
        elgst_listOfLists.append( _GST.ListTools.remove_duplicates(singleGates + elgst_list) )

    #print "%d eLGST sets w/lengths" % len(elgst_listOfLists),map(len,elgst_listOfLists)
    return elgst_listOfLists


    
