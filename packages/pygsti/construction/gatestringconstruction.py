""" Utility functions for creating and acting on lists of gate strings."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import itertools as _itertools
import numpy as _np
import numpy.random as _rndm

from ..tools import listtools as _lt
from ..tools import compattools as _compat
from ..objects import gatestring as _gs
from ..objects import GateSet as _GateSet

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

    loopOrder = list(kwargs.pop('order',[]))
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
            elif _compat.isstr(result):
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
    for gateTuple in _itertools.product(gateLabels, repeat=length):
        yield _gs.GateString(gateTuple)


def list_all_gatestrings_without_powers_and_cycles(gateLabels, maxLength):
    """
    Generate all distinct gate strings up to a maximum length that are 
    aperiodic, i.e., that are not a shorter gate sequence raised to a power,
    and are also distinct up to cycling (e.g. `('Gx','Gy','Gy')` and 
    `('Gy','Gy','Gx')` are considered equivalent and only one would be
    included in the returned list).

    Parameters
    ----------
    gateLabels : list
        A list of the gate labels to for gate strings from.

    maxLength : int
        The maximum length strings to return.  Gatestrings from length 1
        to `maxLength` will be returned.

    Returns
    -------
    list
       Of :class:`GateString` objects.
    """

    #Are we trying to add a germ that is a permutation of a germ we already have?  False if no, True if yes.
    def _perm_check(testStr,strList): # works with python strings, so can use "in" to test for substring inclusion
        return any( [ testStr in s*2 for s in strList ] )

    #Are we trying to add a germ that is a power of a germ we already have?  False if no, True if yes.
    def _pow_check(testStr,strListDict):
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
            if not _perm_check(pys,permCheckedStrs):#Sequence is not a cycle of anything in permCheckedStrs
                permCheckedStrs.append(pys)

        outputDict[length] = []
        for pys in permCheckedStrs:#Now check to see if any elements of tempList2 are powers of elements already in output
            if not _pow_check(pys,outputDict):#Seqeunce is not a power of anything in output
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
    gateLabels = list(gateLabels) # b/c we need to index it below
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

def list_lgst_gatestrings(prepStrs, effectStrs, gateLabelSrc):
    """
    List the gate strings required for running LGST.

    Parameters
    ----------
    prepStrs,effectStrs : list of GateStrings
        Fiducial GateString lists used to construct a informationally complete
        preparation and measurement.

    gateLabelSrc : tuple or GateSet
        List/tuple of gate labels OR a GateSet whose gate and instrument
        labels should be used.

    Returns
    -------
    list of GateString objects
        The list of required gate strings, without duplicates.
    """
    if isinstance(gateLabelSrc, _GateSet):
        gateLabels = list(gateLabelSrc.gates.keys()) + \
                     list(gateLabelSrc.instruments.keys())
    else: gateLabels = gateLabelSrc

    singleGates = [ _gs.GateString( (gl,), "(%s)" % str(gl) ) for gl in gateLabels ]
    ret = create_gatestring_list('eStr','prepStr','prepStr+eStr','prepStr+g+eStr',
                               eStr=effectStrs, prepStr=prepStrs, g=singleGates,
                               order=['g','prepStr','eStr'] ) # LEXICOGRAPHICAL VS MATRIX ORDER
    return _lt.remove_duplicates(ret)


def list_strings_lgst_can_estimate(dataset, prepStrs, effectStrs):
    """
      Compute the gate strings that LGST is able to estimate
      given a set of fiducial strings.

      Parameters
      ----------
      dataset : DataSet
          The data used to generate the LGST estimates

      prepStrs,effectStrs : list of GateStrings
          Fiducial GateString lists used to construct a informationally complete
          preparation and measurement.

      Returns
      -------
      list of lists of tuples
         each list of tuples specifyies a gate string that LGST can estimate.

    """

    estimatable = []
    gateStrings = list(dataset.keys())
    pre = tuple(effectStrs[0]); l0 = len(pre)   #the first effect string
    post = tuple(prepStrs[0]); l1 = len(post)   #the first prep string

    def _root_is_ok(rootStr):
        for estr in effectStrs:
            for rhostr in prepStrs:
                if tuple(rhostr) + tuple(rootStr) + tuple(estr) not in gateStrings: # LEXICOGRAPHICAL VS MATRIX ORDER
                    return False
        return True

    #check if string has first fiducial at beginning & end, and if so
    # strip that first fiducial off, leaving a 'root' string that we can test
    for s in gateStrings:
        if s[0:l0] == pre and s[len(s)-l1:] == post:
            root = s[l0:len(s)-l1]
            if _root_is_ok( root ):
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
        elif _compat.isstr(x):
            ret.append( _gs.GateString(None, x) )
        else:
            raise ValueError("Cannot convert type %s into a GateString" % str(type(x)))
    return ret


def translate_gatestring(gatestring, aliasDict):
    """
    Creates a new GateString object from an existing one by replacing
    gate labels in `gatestring` by (possibly multiple) new labels according
    to `aliasDict`.

    Parameters
    ----------
    gatestring : GateString
        The gate string to use as the base for find & replace
        operations.

    aliasDict : dict
        A dictionary whose keys are single gate labels and whose values are 
        lists or tuples of the new gate labels that should replace that key.
        If `aliasDict is None` then `gatestring` is returned.

    Returns
    -------
    GateString
    """
    if aliasDict is None:
        return gatestring
    else:
        return _gs.GateString(tuple(_itertools.chain(
            *[aliasDict.get(lbl, (lbl,) ) for lbl in gatestring])))



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
        If `aliasDict is None` then `gatestringList` is returned.

    Returns
    -------
    list of GateStrings
    """
    if aliasDict is None:
        return gatestringList
    else:
        new_gatestrings = [ _gs.GateString(tuple(_itertools.chain(
            *[aliasDict.get(lbl,(lbl,)) for lbl in gs])))
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
        ret[A] = tuple(_itertools.chain(*[aliasDict2[B] for B in Bs]))
    return ret


def manipulate_gatestring(gatestring, sequenceRules):
    """
    Manipulates a GateString object according to `sequenceRules`.

    Each element of `sequenceRules` is of the form `(find,replace)`,
    and specifies a replacement rule.  For example,
    `('A',), ('B','C')` simply replaces each `A` with `B,C`.
    `('A', 'B'), ('A', 'B2'))` replaces `B` with `B2` when it follows `A`.
    `('B', 'A'), ('B2', 'A'))` replaces `B` with `B2` when it precedes `A`.

    Parameters
    ----------
    gatestring : GateString or tuple
        The gate string to manipulate.

    sequenceRules : list
        A list of `(find,replace)` 2-tuples which specify the replacement
        rules.  Both `find` and `replace` are tuples of gate labels 
        (or `GateString` objects).  If `sequenceRules is None` then
        `gatestring` is returned.

    Returns
    -------
    list of GateStrings
    """
    if sequenceRules is None:
        return gatestring #avoids doing anything to gatestring

    # flag labels as modified so signal they cannot be processed
    # by any further rules
    gatestring = tuple(gatestring) #make sure this is a tuple
    modified = _np.array([False]*len(gatestring))
    actions = [ [] for i in range(len(gatestring)) ]

    #Step 0: compute prefixes and postfixes of rules
    ruleInfo = []
    for rule, replacement in sequenceRules:
        n_pre = 0 #length of shared prefix btwn rule & replacement
        for a,b in zip(rule,replacement):
            if a==b: n_pre += 1
            else: break
        n_post = 0 #length of shared prefix btwn rule & replacement (if no prefix)
        if n_pre == 0:
            for a,b in zip(reversed(rule),reversed(replacement)):
                if a==b: n_post += 1
                else: break
        n = len(rule)
        ruleInfo.append( (n_pre,n_post,n) )
        #print("Rule%d " % k, rule, "n_pre = ",n_pre," n_post = ",n_post) #DEBUG

    #print("Gatestring = ",gatestring) #DEBUG
    
    #Step 1: figure out which actions (replacements) need to be performed at
    # which indices.  During this step, gatestring is unchanged, but regions
    # of it are marked as having been modified to avoid double-modifications.
    for i in range(len(gatestring)):    
        #print(" **** i = ",i) #DEBUG
        for k,(rule,replacement) in enumerate(sequenceRules):
            n_pre, n_post, n = ruleInfo[k]
            
            #if there's a match that doesn't double-modify
            if rule == gatestring[i:i+n] and not any(modified[i+n_pre:i+n-n_post]):
                # queue this replacement action
                actions[i].append(k)
                #print("MATCH! ==> acting rule %d at index %d" % (k,i)) #DEBUG

                # and mark the modified region of the original string
                modified[i+n_pre:i+n-n_post] = True
        i += 1


    #Step 2: perform the actions (in reverse order so indices don't get messed up!)
    N = len(gatestring)
    for i in range(N-1,-1,-1):
        for k in actions[i]:
            #apply rule k at index i of gatestring
            rule, replacement = sequenceRules[k]
            n_pre,n_post,n = ruleInfo[k]

            begin = gatestring[:i+n_pre]
            repl = replacement[n_pre:len(replacement)-n_post]
            end   = gatestring[i+n-n_post:]

            gatestring = begin + repl + end
            #print("Applied rule %d at index %d: " % (k,i), begin, repl, end, " ==> ", gatestring) #DEBUG

    return _gs.GateString(gatestring)


def manipulate_gatestring_list(gatestringList, sequenceRules):
    """
    Creates a new list of GateString objects from an existing one by performing
    replacements according to `sequenceRules` (see :func:`manipulate_gatestring`).

    Parameters
    ----------
    gatestringList : list of GateStrings
        The list of gate strings to use as the base for find & replace
        operations.

    sequenceRules : list
        A list of `(find,replace)` 2-tuples which specify the replacement
        rules.  Both `find` and `replace` are tuples of gate labels 
        (or `GateString` objects).  If `sequenceRules is None` then
        `gatestringList` is returned.

    Returns
    -------
    list of GateStrings
    """
    if sequenceRules is None:
        return gatestringList
    else:
        return [ manipulate_gatestring(gs, sequenceRules) for gs in gatestringList ]
