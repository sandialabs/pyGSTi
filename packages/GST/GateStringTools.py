""" Utility functions for creating and acting on lists of gate strings."""
import itertools as _itertools
import numpy as _np
import numpy.random as _rndm
import ListTools as _LT
import gatestring as _gatestring

def _getNumPeriods(gateString, periodLen):
    n = 0
    if len(gateString) < periodLen: return 0
    while gateString[0:periodLen] == gateString[n*periodLen:(n+1)*periodLen]: 
        n += 1
    return n

def compressGateLabelTuple(gateString, minLenToCompress=20, maxPeriodToLookFor=20):
    """
    Compress a gate string.  The result is tuple with a special compressed-
    gate-string form form that is not useable by other GST methods but is
    typically shorter (especially for long gate strings with a repetative
    structure) than the original gate string tuple.

    Parameters
    ----------
    gateString : tuple of gate labels or GateString
        The gate string to compress.

    minLenToCompress : int, optional
        The minimum length string to compress.  If len(gateString)
        is less than this amount its tuple is returned.
        
    maxPeriodToLookFor : int, optional
        The maximum period length to use when searching for periodic
        structure within gateString.  Larger values mean the method
        takes more time but could result in better compressing.

    Returns
    -------
    tuple
        The compressed (or raw) gate string tuple.
    """
    gateString = tuple(gateString) # converts from GateString or list to tuple if needed
    L = len(gateString)
    if L < minLenToCompress: return tuple(gateString)
    compressed = ["CCC"] #list for appending, then make into tuple at the end
    start = 0
    while start < L:
        #print "Start = ",start
        score = _np.zeros( maxPeriodToLookFor+1, 'd' )
        numperiods = _np.zeros( maxPeriodToLookFor+1, 'i' )
        for periodLen in range(1,maxPeriodToLookFor+1):
            n = _getNumPeriods( gateString[start:], periodLen )
            if n == 0: score[periodLen] = 0
            elif n == 1: score[periodLen] = 4.1/periodLen
            else: score[periodLen] = _np.sqrt(periodLen)*n
            numperiods[periodLen] = n
        bestPeriodLen = _np.argmax(score)
        n = numperiods[bestPeriodLen]
        bestPeriod = gateString[start:start+bestPeriodLen]
        #print "Scores = ",score
        #print "num per = ",numperiods
        #print "best = %s ^ %d" % (str(bestPeriod), n)
        assert(n > 0 and bestPeriodLen > 0)
        if start > 0 and n == 1 and compressed[-1][1] == 1:
            compressed[-1] = (compressed[-1][0]+bestPeriod, 1)
        else:
            compressed.append( (bestPeriod, n) )
        start = start+bestPeriodLen*n
            
    return tuple(compressed)

def expandGateLabelTuple(compressedGateString):
    """
    Expand a compressed tuple created with compressGateLabelTuple(...)
    into a tuple of gate labels.

    Parameters
    ----------
    compressedGateString : tuple
        a tuple in the compressed form created by
        compressGateLabelTuple(...).

    Returns
    -------
    tuple
        A tuple of gate labels specifying the uncompressed gate string.
    """
    
    if len(compressedGateString) == 0: return ()
    if compressedGateString[0] != "CCC": return compressedGateString
    expandedString = []
    for (period,n) in compressedGateString[1:]:
        expandedString += period*n
    return tuple(expandedString)    


def _runExpression(str_expression, myLocals):
    exec( "result = " + str_expression, {"__builtins__": None}, myLocals )
    return myLocals.get("result",None)

def createGateStringList(*args,**kwargs):
    """
    Create a list of gate strings using a nested loop.  Positional arguments
    specify evaluation strings, which are evaluated within the inner-loop
    for a nested loop over all list or tuple type keyword arguments.
    
    Parameters
    ----------
    args : list of strings
        Positional arguments are strings that python can evaluate into either
        a tuple of gate labels or a GateString instance.  If evaluation raises
        an AssertionError (an assert statement fails) then that inner loop
        evaluation is skipped and list construction proceeds.

    kwargs : dict
        keys specify variable names that can be used in positional argument strings.

    Returns
    -------
    list of GateString objects
    """
    lst = []
    
    loopOrder = kwargs.pop('order',[])
    loopLists = {}; loopLocals = { 'True': True, 'False': False, 'str':str, 'int': int, 'float': float}
    for key,val in kwargs.iteritems():
        if type(val) in (list,tuple): #key describes a variable to loop over
            loopLists[key] = val
            if key not in loopOrder: 
                loopOrder.append(key)
        else: # callable(val): #key describes a function or variable to pass through to exec
            loopLocals[key] = val
    
    #print "DEBUG: looplists = ",loopLists
    for str_expression in args:
        if len(str_expression) == 0:
            lst.append( () ); continue #special case
            
        keysToLoop = [ key for key in loopOrder if key in str_expression ]
        loopListsToLoop = [ loopLists[key] for key in keysToLoop ] #list of lists
        for allVals in _itertools.product(*loopListsToLoop):
            myLocals = { key:allVals[i] for i,key in enumerate(keysToLoop) }
            myLocals.update( loopLocals )
            try:
                result = _runExpression(str_expression, myLocals)
            except AssertionError: continue #just don't append

            if isinstance(result,_gatestring.GateString):
                gateStr = result
            elif isinstance(result,list) or isinstance(result,tuple):
                gateStr = _gatestring.GateString(result)
            elif isinstance(result,str):
                gateStr = _gatestring.GateString(None, result)
            lst.append(gateStr)
            
    return lst


def repeat(x,nTimes,assertAtLeastOneRep=False):
    """
    Repeat x nTimes times.

    Parameters
    ----------
    x : tuple or GateString
       the gate string to repeat

    nTimes : int
       the number of times to repeat x

    assertAtLeastOneRep : bool, optional
       if True, assert that nTimes > 0.  This can be useful when used
       within a createGateStringList inner loop to build a gate string
       lists where a string must be repeated at least once to be added
       to the list.

    Returns
    -------
    tuple or GateString (whichever x was)
    """
    if assertAtLeastOneRep:  assert(nTimes > 0)
    return x*nTimes

def repeatCountWithMaxLength(x,maxLength,assertAtLeastOneRep=False):
    """
    Compute the number of times a gate string x must be repeated such that
    the repeated string has length <= maxLength.

    Parameters
    ----------
    x : tuple or GateString
       the gate string to repeat

    maxLength : int
       the maximum length

    assertAtLeastOneRep : bool, optional
       if True, assert that number of repetitions is > 0.
       This can be useful when used within a createGateStringList inner loop
       to build a gate string lists where a string must be repeated at
       least once to be added to the list.

    Returns
    -------
    int
      the number of repetitions.
    """
    l = len(x)
    if assertAtLeastOneRep:  assert(l <= maxLength)
    reps = maxLength//l if l > 0 else 0
    return reps

def repeatWithMaxLength(x,maxLength,assertAtLeastOneRep=False):
    """
    Repeat the gate string x an integer number of times such that
    the repeated string has length <= maxLength.

    Parameters
    ----------
    x : tuple or GateString
       the gate string to repeat.

    maxLength : int
       the maximum length.

    assertAtLeastOneRep : bool, optional
       if True, assert that number of repetitions is > 0.
       This can be useful when used within a createGateStringList inner loop
       to build a gate string lists where a string must be repeated at
       least once to be added to the list.

    Returns
    -------
    tuple or GateString (whichever x was)    
        the repeated gate string
    """
    return repeat(x,repeatCountWithMaxLength(x,maxLength,assertAtLeastOneRep),assertAtLeastOneRep)

#Useful for anything?
#def repeatEmpty(x,maxLength,assertAtLeastOneRep=False):
#    return ()

def repeatAndTruncate(x,N,assertAtLeastOneRep=False):
    """
    Repeat the gate string x so the repeated string has length greater than N,
    then truncate the string to be exactly length N.

    Parameters
    ----------
    x : tuple or GateString
       the gate string to repeat & truncate.

    N : int
       the truncation length.

    assertAtLeastOneRep : bool, optional
       if True, assert that number of repetitions is > 0.
       This is always the case when x has length > 0.

    Returns
    -------
    tuple or GateString (whichever x was)    
        the repeated-then-truncated gate string
    """
    reps = repeatCountWithMaxLength(x,N,assertAtLeastOneRep) + 1
    return (x*reps)[0:N]

def repeatRemainderForTruncation(x,N,assertAtLeastOneRep=False):
    """
    Repeat the gate string x the fewest number of times such that the repeated
    string has length greater than or equal to N.  Return the portion of this
    repeated string from the N-th position to the end. Note that this corresponds
    to what is truncated in a call to repeateAndTruncate(x,N,assertAtLeastOneRep).

    Parameters
    ----------
    x : tuple or GateString
       the gate string to operate on.

    N : int
       the truncation length.

    assertAtLeastOneRep : bool, optional
       if True, assert that number of repetitions is > 0.
       This is always the case when x has length > 0.

    Returns
    -------
    tuple or GateString (whichever x was)    
        the remainder gate string

    """
    reps = repeatCountWithMaxLength(x,N,assertAtLeastOneRep)
    return x[0:(N - reps*len(x))]


def simplifyStr(gateStringStr):
    """
    Simplify a string representation of a gate string.  The simplified
      string should evaluate to the same gate label tuple as the original.

    Parameters
    ----------
    gateStringStr : string
        the string representation of a gate string to be simplified.
        (e.g. "Gx{}", "Gy^1Gx")
        
    Returns
    -------
    string
        the simplified string representation.
    """
    s = gateStringStr.replace("{}","")
    s = s.replace("^1G","G")
    s = s.replace("^1(","(")
    s = s.replace("^1{","{")
    if s.endswith("^1"): s = s[:-2]
    return s if len(s) > 0 else "{}"



## gate-label-tuple function.  TODO: check if these are still needed.

def listAllGateStrings(gateLabels, minlength, maxlength):
    """
    List all the gate strings in a given length range.
    
    Parameters
    ----------
    gateLabels : tuple
        tuple of gate labels to include in gate strings.

    minlength : int
        the minimum gate string length to return

    maxlength : int
        the maximum gate string length to return

    Returns
    -------
    list
        A list of GateString objects.
    """
    ret = [ ]
    for l in range(minlength, maxlength+1):
        ret += listAllGateStringsOfLength(gateLabels, l)
    return ret

def genAllGateStrings(gateLabels, minlength, maxlength):
    """ Generator version of listAllGateStrings """
    ret = [ ]
    for l in range(minlength, maxlength+1):
        for s in genAllGateStringsOfLength(gateLabels, l):
            yield s

def listAllGateStringsOfLength(gateLabels, length):
    """
    List all the gate strings of a given length.
    
    Parameters
    ----------
    gateLabels : tuple
        tuple of gate labels to include in gate strings.

    length : int
        the gate string length

    Returns
    -------
    list
        A list of GateString objects.
    """
    if length == 0: return [ _gatestring.GateString( () ) ]
    if length == 1: return [ _gatestring.GateString( (g,) ) for g in gateLabels ]
    m1StrList = listAllGateStringsOfLength(gateLabels, length-1)
    return [ _gatestring.GateString( (g,) ) + s for g in gateLabels for s in m1StrList ]


def genAllGateStringsOfLength(gateLabels, length):
    """Generator version of listAllGateStringsOfLength"""
    if length == 0: yield _gatestring.GateString( () )
    elif length == 1: 
        for g in gateLabels:
            yield _gatestring.GateString( (g,) )
    else:
        for g in gateLabels:
            for s in genAllGateStringsOfLength(gateLabels, length-1):
                yield _gatestring.GateString( (g,) ) + s


def listAllGateStringsWithoutPowersAndCycles(gateLabels, maxLength):

    #Are we trying to add a germ that is a permutation of a germ we already have?  False if no, True if yes.
    def permCheck(testStr,strList): # works with python strings, so can use "in" to test for substring inclusion
        return any( [ testStr in s*2 for s in strList ] )
    
    #Are we trying to add a germ that is a power of a germ we already have?  False if no, True if yes.
    def powCheck(testStr,strListDict):
        L = len(testStr)
        for k in strListDict.keys():
            if L % k == 0:
                rep = L // k
                if any([testStr == s*rep for s in strListDict[k] ]):
                        return True
        return False

    outputDict = {}
    for length in _np.arange(1,maxLength+1):

        permCheckedStrs = []
        for s in genAllGateStringsOfLength(gateLabels, length):
            pys = gateStringToPythonString(s,gateLabels)
            if not permCheck(pys,permCheckedStrs):#Sequence is not a cycle of anything in permCheckedStrs
                permCheckedStrs.append(pys)

        outputDict[length] = []
        for pys in permCheckedStrs:#Now check to see if any elements of tempList2 are powers of elements already in output
            if not powCheck(pys,outputDict):#Seqeunce is not a power of anything in output
                outputDict[length].append(pys)

    output = []
    for length in _np.arange(1,maxLength+1):
        output.extend( [ pythonStringToGateString(pys, gateLabels) for pys in outputDict[length] ] )
    return output


def listRandomGateStringsOfLength(gateLabels, length, count):
    """
    Create a list of random gate strings of a given length.
    
    Parameters
    ----------
    gateLabels : tuple
        tuple of gate labels to include in gate strings.

    length : int
        the gate string length.

    count : int
        the number of random strings to create.

    Returns
    -------
    list of GateStrings
        A list of random gate strings as GateString objects.
    """    
    ret = [ ]
    for i in range(count):
        r = _rndm.random(length) * len(gateLabels)
        ret.append( _gatestring.GateString( [gateLabels[int(k)] for k in r]) )
    return ret

def listPartialStrings(gateString):
    """
    List the parial strings of gateString, that is,
      the strings that are the slices gateString[0:n]
      for 0 <= l <= len(gateString).

    Parameters
    ----------
    gateString : tuple of gate labels or GateString
        The gate string to act upon.

    Returns
    -------
    list of GateString objects.
       The parial gate strings.
    """
    ret = [ ]
    for l in range(len(gateString)+1):
        ret.append( tuple(gateString[0:l]) )
    return ret

def listLGSTGateStrings(specs, gateLabels):
    """
    List the gate strings required for runnsing LGST.

    Parameters
    ----------
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...).

    gateLabels : tuple
        tuple of gate labels to estimate using LGST.

    Returns
    -------
    list of GateString objects
        The list of required gate strings, without duplicates.
    """
    from Core import getRhoAndEStrs as _getRhoAndEStrs #move this to the top when this fn is split off of Core.py
    rStrings, eStrings = _getRhoAndEStrs(specs)
    singleGates = [ _gatestring.GateString( (gl,), "(%s)" % gl ) for gl in gateLabels ]
    ret = createGateStringList('eStr','rhoStr','rhoStr+eStr','rhoStr+g+eStr',
                               eStr=eStrings, rhoStr=rStrings, g=singleGates,
                               order=['g','rhoStr','eStr'] ) # LEXICOGRAPHICAL VS MATRIX ORDER
    return _LT.remove_duplicates(ret)



#For evalTree usage
def gateStringToPythonString(gateString,gateLabels):
    """
    Convert a gate string into a python string, where each gate label is
    represented as a **single** character, starting with 'A' and contining
    down the alphabet.  This can be useful for processing gate strings 
    using python's string tools (regex in particular).

    Parameters
    ----------
    gateString : tuple or GateString
        the gate label sequence that is converted to a string.
    
    gateLabels : tuple
       tuple containing all the gate labels that will be mapped to alphabet
       characters, beginning with 'A'.

    Returns
    -------
    string
        The converted gate string.
    
    Examples
    --------
        ('Gx','Gx','Gy','Gx') => "AABA"
    """
    assert(len(gateLabels) < 26) #Complain if we go beyond 'Z'
    translateDict = {}; c = 'A'
    for gateLabel in gateLabels:
        translateDict[gateLabel] = c
        c = chr(ord(c) + 1)
    return "".join([ translateDict[gateLabel] for gateLabel in gateString ])

def pythonStringToGateString(pythonString,gateLabels):
    """
    Convert a python string into a gate string, where each gate label is
    represented as a **single** character, starting with 'A' and contining
    down the alphabet.  This performs the inverse of gateStringToPythonString(...).

    Parameters
    ----------
    pythonString : string
        string whose individual characters correspond to the gate labels of a
        gate string.

    gateLabels : tuple
       tuple containing all the gate labels that will be mapped to alphabet
       characters, beginning with 'A'.

    Returns
    -------
    GateString
        The decoded python string as a GateString object (essentially 
        a tuple of gate labels).

    Examples
    --------
        "AABA" => ('Gx','Gx','Gy','Gx')
    """
    assert(len(gateLabels) < 26) #Complain if we go beyond 'Z'
    translateDict = {}; c = 'A'
    for gateLabel in gateLabels:
        translateDict[c] = gateLabel
        c = chr(ord(c) + 1)
    return _gatestring.GateString( tuple([ translateDict[c] for c in pythonString ]) )


def gateStringList( listOfGateLabelTuplesOrStrings ):
    """ 
    Converts a list of gate label tuples or strings to 
     a list of GateString objects.
     
    Parameters
    ----------
    listOfGateLabelTuplesOrStrings : list
        List which may contain a mix of GateString objects, tuples of gate
        labels, and strings in standard-text-format.

    Returns
    -------
    list of GateString objects
        Each item of listOfGateLabelTuplesOrStrings converted to a GateString.
    """
    ret = []
    for x in listOfGateLabelTuplesOrStrings:
        if isinstance(x,_gatestring.GateString):
            ret.append(x)
        if isinstance(x,tuple) or isinstance(x,list):
            ret.append( _gatestring.GateString(x) )
        elif isinstance(x,str):
            ret.append( _gatestring.GateString(None, x) )
        else:
            raise ValueError("Cannot convert type %s into a GateString" % str(type(x)))
    return ret


#Unneeded
#def listPeriodicGateStrings(gateLabels, max_period, minlength, maxlength, left_bookends=[()], right_bookends=[()]):
#    ret = [ ]
#    for lb in left_bookends:
#        for rb in right_bookends:
#            for l in range(minlength, maxlength+1):
#                pdic = listPeriodicGateStringsOfLength(gateLabels, max_period, l)
#                ret += [ tuple(lb) + tuple(p) + tuple(rb) for p in pdic ]
#    return _LT.remove_duplicates(ret)
#
#def listPeriodicGateStringsOfLength(gateLabels, max_period, length):
#    ret = [ ]
#    if max_period >= 0: ret.append( [] )
#    for period_length in range(1,min(max_period,length)+1):
#        nPeriods = _np.ceil(length / float(period_length))
#        for period in listAllGateStringsOfLength(gateLabels, period_length):
#            for k in range(1,period_length):
#                if period_length % k > 0: continue
#                if period in listPeriodicGateStringsOfLength(gateLabels, k, period_length):
#                    break # period is itself periodic with period k < len(period), so don't use it as the string it generates has already been found
#            else:
#                s = period * nPeriods
#                ret.append( tuple(s[0:length]) )
#    return _LT.remove_duplicates(ret)
#
#
#
#def listExponentiatedGermGateStrings(germs, exponents, ends=None, left_ends=None, right_ends=None):
#    if ends is not None:
#        if not left_ends and not right_ends:
#            left_ends = right_ends = ends
#        else: raise ValueError("Conflicting arguments to listExponentiatedGermGateStrings - specify either" + \
#                                   "'ends' or 'left_ends' and/or 'right_ends', not both")
#    if left_ends is None: left_ends = [()]
#    elif () not in left_ends: left_ends = [()] + left_ends
#    if right_ends is None: right_ends = [()]
#    elif () not in right_ends: right_ends = [()] + right_ends
#
#    ret = createGateStringList("lb+germ*exp+rb", germ=germs, exp=exponents,lb=left_ends, rb=right_ends,
#                               order=['germ','exp','lb','rb'] )
#    return ret
#
#
#def readGateStringList(filename,**kwargs):
#    emptyCode = kwargs.get("empty_code","") # code for the empty string
#    mode = kwargs.get("mode", "comma-delim") #or const-length
#    L = kwargs.get("length", 0)
#    if mode == "const-length": assert(L > 0)
#
#    ret = [ ]
#    for line in open(filename):
#        if line[0][0]=='#': continue        
#        splitline = line.split()
#        if len(splitline) == 0: continue
#        charStr = splitline[0]  
#        if charStr == emptyCode:
#            gateString = () 
#        elif mode == "comma-delim":
#            gateString = tuple( charStr.split(",") )
#        elif mode == "const-length":            
#            gateString = tuple( [ charStr[i:i+L] for i in range(0,len(charStr),L) ] )
#        else: 
#            raise ValueError("Invalid mode passed to readGateStringList: %s" % mode)
#
#        ret.append(gateString)
#        
#    return ret



