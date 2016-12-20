from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Utility functions for creating and acting on lists of gate strings."""

import itertools as _itertools
import numpy as _np
import numpy.random as _rndm

from ..tools import listtools as _lt
from ..objects import gatestring as _gs
from .spamspecconstruction import get_spam_strs as _get_spam_strs


def _runExpression(str_expression, myLocals):
    exec( "result = " + str_expression, {"__builtins__": None}, myLocals )
    return myLocals.get("result",None)

def create_gatestring_list(*args,**kwargs):
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
        keys specify variable names that can be used in positional argument
        strings.

    Returns
    -------
    list of GateString

    Examples
    --------
    >>> from pygsti.construction import create_gatestring_list
    >>> As = [('a1',), ('a2',)]
    >>> Bs = [('b1',), ('b2',)]
    >>> list1 = create_gatestring_list('a', 'a+b', a=As, b=Bs)
    >>> print(list(map(str, list1)))
    ['a1', 'a2', 'a1b1', 'a1b2', 'a2b1', 'a2b2']

    You can change the order in which the different iterables are advanced.

    >>> list2 = create_gatestring_list('a+b', a=As, b=Bs, order=['a', 'b'])
    >>> print(list(map(str, list2)))
    ['a1b1', 'a1b2', 'a2b1', 'a2b2']
    >>> list3 = create_gatestring_list('a+b', a=As, b=Bs, order=['b', 'a'])
    >>> print(list(map(str, list3)))
    ['a1b1', 'a2b1', 'a1b2', 'a2b2']


    """
    lst = []

    loopOrder = kwargs.pop('order',[])
    loopLists = {}; loopLocals = { 'True': True, 'False': False, 'str':str, 'int': int, 'float': float}
    for key,val in kwargs.items():
        if type(val) in (list,tuple): #key describes a variable to loop over
            loopLists[key] = val
            if key not in loopOrder:
                loopOrder.append(key)
        else: # callable(val): #key describes a function or variable to pass through to exec
            loopLocals[key] = val

    #print "DEBUG: looplists = ",loopLists
    for str_expression in args:
        if len(str_expression) == 0:
            lst.append( _gs.GateString( () ) ); continue #special case

        keysToLoop = [ key for key in loopOrder if key in str_expression ]
        loopListsToLoop = [ loopLists[key] for key in keysToLoop ] #list of lists
        for allVals in _itertools.product(*loopListsToLoop):
            myLocals = { key:allVals[i] for i,key in enumerate(keysToLoop) }
            myLocals.update( loopLocals )
            try:
                result = _runExpression(str_expression, myLocals)
            except AssertionError: continue #just don't append

            if isinstance(result,_gs.GateString):
                gateStr = result
            elif isinstance(result,list) or isinstance(result,tuple):
                gateStr = _gs.GateString(result)
            elif isinstance(result,str):
                gateStr = _gs.GateString(None, result)
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
       within a create_gatestring_list inner loop to build a gate string
       lists where a string must be repeated at least once to be added
       to the list.

    Returns
    -------
    tuple or GateString (whichever x was)
    """
    if assertAtLeastOneRep:  assert(nTimes > 0)
    return x*nTimes

def repeat_count_with_max_length(x,maxLength,assertAtLeastOneRep=False):
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
       This can be useful when used within a create_gatestring_list inner loop
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

def repeat_with_max_length(x,maxLength,assertAtLeastOneRep=False):
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
       This can be useful when used within a create_gatestring_list inner loop
       to build a gate string lists where a string must be repeated at
       least once to be added to the list.

    Returns
    -------
    tuple or GateString (whichever x was)
        the repeated gate string
    """
    return repeat(x,repeat_count_with_max_length(x,maxLength,assertAtLeastOneRep),assertAtLeastOneRep)

#Useful for anything?
#def repeat_empty(x,maxLength,assertAtLeastOneRep=False):
#    return ()

def repeat_and_truncate(x,N,assertAtLeastOneRep=False):
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
    reps = repeat_count_with_max_length(x,N,assertAtLeastOneRep) + 1
    return (x*reps)[0:N]

def repeat_remainder_for_truncation(x,N,assertAtLeastOneRep=False):
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
    reps = repeat_count_with_max_length(x,N,assertAtLeastOneRep)
    return x[0:(N - reps*len(x))]


def simplify_str(gateStringStr):
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

def list_all_gatestrings(gateLabels, minlength, maxlength):
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
    gateTuples = _itertools.chain(*[_itertools.product(gateLabels, repeat=N)
                                    for N in range(minlength, maxlength + 1)])
    return list(map(_gs.GateString, gateTuples))

def gen_all_gatestrings(gateLabels, minlength, maxlength):
    """ Generator version of list_all_gatestrings """
    gateTuples = _itertools.chain(*[_itertools.product(gateLabels, repeat=N)
                                    for N in range(minlength, maxlength + 1)])
    for gateTuple in gateTuples:
        yield _gs.GateString(gateTuple)

def list_all_gatestrings_onelen(gateLabels, length):
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
    gateTuples = _itertools.product(gateLabels, repeat=length)
    return list(map(_gs.GateString, gateTuples))


def gen_all_gatestrings_onelen(gateLabels, length):
    """Generator version of list_all_gatestrings_onelen"""
#    OLD
#    if length == 0: yield _gs.GateString( () )
#    elif length == 1:
#        for g in gateLabels:
#            yield _gs.GateString( (g,) )
#    else:
#        for g in gateLabels:
#            for s in gen_all_gatestrings_onelen(gateLabels, length-1):
#                yield _gs.GateString( (g,) ) + s
    for gateTuple in _itertools.product(gateLabels, repeat=length):
        yield _gs.GateString(gateTuple)


def list_all_gatestrings_without_powers_and_cycles(gateLabels, maxLength):

    #Are we trying to add a germ that is a permutation of a germ we already have?  False if no, True if yes.
    def perm_check(testStr,strList): # works with python strings, so can use "in" to test for substring inclusion
        return any( [ testStr in s*2 for s in strList ] )

    #Are we trying to add a germ that is a power of a germ we already have?  False if no, True if yes.
    def pow_check(testStr,strListDict):
        L = len(testStr)
        for k in list(strListDict.keys()):
            if L % k == 0:
                rep = L // k
                if any([testStr == s*rep for s in strListDict[k] ]):
                    return True
        return False

    outputDict = {}
    for length in _np.arange(1,maxLength+1):

        permCheckedStrs = []
        for s in gen_all_gatestrings_onelen(gateLabels, length):
            pys = s.to_pythonstr(gateLabels)
            if not perm_check(pys,permCheckedStrs):#Sequence is not a cycle of anything in permCheckedStrs
                permCheckedStrs.append(pys)

        outputDict[length] = []
        for pys in permCheckedStrs:#Now check to see if any elements of tempList2 are powers of elements already in output
            if not pow_check(pys,outputDict):#Seqeunce is not a power of anything in output
                outputDict[length].append(pys)

    output = []
    for length in _np.arange(1,maxLength+1):
        output.extend( [ _gs.GateString.from_pythonstr(pys, gateLabels) for pys in outputDict[length] ] )
    return output


def list_random_gatestrings_onelen(gateLabels, length, count, seed=None):
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

    seed : int, optional
        If not None, a seed for numpy's random number generator.


    Returns
    -------
    list of GateStrings
        A list of random gate strings as GateString objects.
    """
    ret = [ ]
    rndm = _rndm.RandomState(seed) # ok if seed is None
    for i in range(count): #pylint: disable=unused-variable
        r = rndm.random_sample(length) * len(gateLabels)
        ret.append( _gs.GateString( [gateLabels[int(k)] for k in r]) )
    return ret

def list_partial_strings(gateString):
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

def list_lgst_gatestrings(specs, gateLabels):
    """
    List the gate strings required for runnsing LGST.

    Parameters
    ----------
    specs : 2-tuple
        A (prepSpecs,effectSpecs) tuple usually generated by calling
        build_spam_specs(...).

    gateLabels : tuple
        tuple of gate labels to estimate using LGST.

    Returns
    -------
    list of GateString objects
        The list of required gate strings, without duplicates.
    """
    rStrings, eStrings = _get_spam_strs(specs)
    singleGates = [ _gs.GateString( (gl,), "(%s)" % gl ) for gl in gateLabels ]
    ret = create_gatestring_list('eStr','prepStr','prepStr+eStr','prepStr+g+eStr',
                               eStr=eStrings, prepStr=rStrings, g=singleGates,
                               order=['g','prepStr','eStr'] ) # LEXICOGRAPHICAL VS MATRIX ORDER
    return _lt.remove_duplicates(ret)


def list_strings_lgst_can_estimate(dataset, specs):
    """
      Compute the gate strings that LGST is able to estimate
      given a set of fiducial strings or prepSpecs and effectSpecs.

      Parameters
      ----------
      dataset : DataSet
          The data used to generate the LGST estimates

      specs : 2-tuple
          A (prepSpecs,effectSpecs) tuple usually generated by calling
          build_spam_specs(...)

      Returns
      -------
      list of lists of tuples
         each list of tuples specifyies a gate string that LGST can estimate.

    """

    #Process input parameters
    prepSpecs, effectSpecs = specs

    estimatable = []
    gateStrings = list(dataset.keys())
    pre = tuple(effectSpecs[0].str);     l0 = len(pre)   #the first effectSpec string prefix
    post = tuple(prepSpecs[0].str); l1 = len(post)  #the first prepSpec string postfix

    def root_is_ok(rootStr):
        for espec in effectSpecs:
            for rhospec in prepSpecs:
                if tuple(rhospec.str) + tuple(rootStr) + tuple(espec.str) not in gateStrings: # LEXICOGRAPHICAL VS MATRIX ORDER
                    return False
        return True

    #check if string has first fiducial at beginning & end, and if so
    # strip that first fiducial off, leaving a 'root' string that we can test
    for s in gateStrings:
        if s[0:l0] == pre and s[len(s)-l1:] == post:
            root = s[l0:len(s)-l1]
            if root_is_ok( root ):
                estimatable.append( root )

    return gatestring_list(estimatable)



def gatestring_list( listOfGateLabelTuplesOrStrings ):
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
        if isinstance(x,_gs.GateString):
            ret.append(x)
        elif isinstance(x,tuple) or isinstance(x,list):
            ret.append( _gs.GateString(x) )
        elif isinstance(x,str):
            ret.append( _gs.GateString(None, x) )
        else:
            raise ValueError("Cannot convert type %s into a GateString" % str(type(x)))
    return ret


def translate_gatestring_list(gatestringList, aliasDict):
    """
    Creates a new list of GateString objects from an existing one by replacing
    gate labels in `gatestringList` by (possibly multiple) new labels according
    to `aliasDict`.

    Parameters
    ----------
    gatestringList : list of GateStrings
        The list of gate strings to use as the base for find & replace
        operations.

    aliasDict : dict
        A dictionary whose keys are single gate labels and whose values are 
        lists or tuples of the new gate labels that should replace that key.

    Returns
    -------
    list of GateStrings
    """
    new_gatestrings = [ _gs.GateString(tuple(_itertools.chain(
                *[aliasDict.get(lbl,lbl) for lbl in gs])))
                        for gs in gatestringList ]
    return new_gatestrings


def compose_alias_dicts(aliasDict1, aliasDict2):
    """
    Composes two alias dicts.
    
    Assumes `aliasDict1` maps "A" labels to "B" labels and `aliasDict2` maps
    "B" labels to "C" labels.  The returned dictionary then maps "A" labels
    directly to "C" labels, and satisfies:

    `returned[A_label] = aliasDict2[ aliasDict1[ A_label ] ]`
    
    Parameters
    ----------
    aliasDict1, aliasDict2 : dict
        The two dictionaries to compose.

    Returns
    -------
    dict
    """
    ret = {}
    for A,Bs in aliasDict1.items():
        ret[A] = list(_itertools.chain(*[aliasDict2[B] for B in Bs]))
    return ret



#Unneeded
#def list_periodic_gatestrings(gateLabels, max_period, minlength, maxlength, left_bookends=[()], right_bookends=[()]):
#    ret = [ ]
#    for lb in left_bookends:
#        for rb in right_bookends:
#            for l in range(minlength, maxlength+1):
#                pdic = list_periodic_gatestrings_onelen(gateLabels, max_period, l)
#                ret += [ tuple(lb) + tuple(p) + tuple(rb) for p in pdic ]
#    return _lt.remove_duplicates(ret)
#
#def list_periodic_gatestrings_onelen(gateLabels, max_period, length):
#    ret = [ ]
#    if max_period >= 0: ret.append( [] )
#    for period_length in range(1,min(max_period,length)+1):
#        nPeriods = _np.ceil(length / float(period_length))
#        for period in list_all_gatestrings_onelen(gateLabels, period_length):
#            for k in range(1,period_length):
#                if period_length % k > 0: continue
#                if period in list_periodic_gatestrings_onelen(gateLabels, k, period_length):
#                    break # period is itself periodic with period k < len(period), so don't use it as the string it generates has already been found
#            else:
#                s = period * nPeriods
#                ret.append( tuple(s[0:length]) )
#    return _lt.remove_duplicates(ret)
#
#
#
#def list_exponentiated_germ_gatestrings(germs, exponents, ends=None, left_ends=None, right_ends=None):
#    if ends is not None:
#        if not left_ends and not right_ends:
#            left_ends = right_ends = ends
#        else: raise ValueError("Conflicting arguments to list_exponentiated_germ_gatestrings - specify either" + \
#                                   "'ends' or 'left_ends' and/or 'right_ends', not both")
#    if left_ends is None: left_ends = [()]
#    elif () not in left_ends: left_ends = [()] + left_ends
#    if right_ends is None: right_ends = [()]
#    elif () not in right_ends: right_ends = [()] + right_ends
#
#    ret = create_gatestring_list("lb+germ*exp+rb", germ=germs, exp=exponents,lb=left_ends, rb=right_ends,
#                               order=['germ','exp','lb','rb'] )
#    return ret
#
#
#def read_gatestring_list(filename,**kwargs):
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
#            raise ValueError("Invalid mode passed to read_gatestring_list: %s" % mode)
#
#        ret.append(gateString)
#
#    return ret
