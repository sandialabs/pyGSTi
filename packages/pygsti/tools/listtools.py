from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Utility functions for working with lists """

def remove_duplicates_in_place(l,indexToTest=None):
    """
    Remove duplicates from the list passed as an argument.

    In the special case when l contains WeightedGateString instances, the
    duplicates are removed in such a way that the largest weight instance
    of any set of duplicates is kept.

    Parameters
    ----------
    l : list
        The list to remove duplicates from.

    indexToTest : int, optional
        If not None, the index within the elements of l to test.  For
        example, if all the elements of l contain 2 tuples (x,y) then
        set indexToTest == 1 to remove tuples with duplicate y-values.

    Returns
    -------
    None
    """
    s = set(); n = 0
    for x in l:
        t = x if indexToTest is None else x[indexToTest]

        #TODO: create a special duplicate removal function for use with
        #  WeighedGateStrings and include the below commented block:
        #Special case of weighted gate strings: if collisions
        # keep the hightest weight string
        #if isinstance(t, _WeightedGateString) and t in s:
        #    for gs in l[0:n]:
        #        if gs == t:
        #            if isinstance(gs, _WeightedGateString):
        #                gs.weight = max(gs.weight, t.weight)
        #            break

        if t not in s:
            s.add(t)
            l[n] = x; n += 1

    del l[n:]

def remove_duplicates(l,indexToTest=None):
    """
    Remove duplicates from the a list and return the result.

    In the special case when l contains WeightedGateString instances, the
    duplicates are removed in such a way that the largest weight instance
    of any set of duplicates is kept.

    Parameters
    ----------
    l : list
        The list to remove duplicates from.

    indexToTest : int, optional
        If not None, the index within the elements of l to test.  For
        example, if all the elements of l contain 2 tuples (x,y) then
        set indexToTest == 1 to remove tuples with duplicate y-values.

    Returns
    -------
    list
        the list after duplicates have been removed.
    """
    lcopy = l[:]; remove_duplicates_in_place(lcopy,indexToTest)
    return lcopy


def compute_occurance_indices(lst):
    """
    Returns a 0-based list of integers specifying which occurance,
    i.e. enumerated duplicate, each list item is.

    For example, if `lst` = [ 'A','B','C','C','A'] then the
    returned list will be   [  0 , 0 , 0 , 1 , 1 ].  This is useful
    when working with `DataSet` objects that have `collisionAction` 
    set to "keepseparate".

    Parameters
    ----------
    lst : list
        The list to process.

    Returns
    -------
    list
    """
    lookup = {}; ret = []
    for x in lst:
        if x not in lookup:
            lookup[x] = 0
        else:
            lookup[x] += 1
        ret.append( lookup[x] )
    return ret



# ------------------------------------------------------------------------------
# Machinery initially designed for an in-place take operation, which computes
# how to do in-place permutations of arrays/lists efficiently.  Kept here
# commented out in case this is needed some time in the future.
# ------------------------------------------------------------------------------
#
#def build_permute_copy_order(indices):
#    #Construct a list of the operations needed to "take" indices
#    # out of an array.
#
#    nIndices = len(indices)
#    flgs = _np.zeros(nIndices,'bool') #flags indicating an index has been processed
#    shelved = {}
#    copyList = []
#
#    while True: #loop until we've processed everything
#
#        #The cycle has ended.  Now find an unprocessed
#        # destination to begin a new cycle
#        for i in range(nIndices):
#            if flgs[i] == False:
#                if indices[i] == i: # index i is already where it need to be!
#                    flgs[i] = True
#                else:
#                    cycleFirstIndex = iDest = i
#                    if cycleFirstIndex in indices:
#                        copyList.append( (-1,i) ) # iDest == -1 means copy to offline storage
#                    break;
#        else:
#            break #everything has been processed -- we're done!
#
#        while True: # loop over cycle
#            
#            # at this point, data for index iDest has been stored or copied
#            iSrc = indices[iDest] # get source index for current destination
#    
#            # record appropriate copy command
#            if iSrc == cycleFirstIndex:
#                copyList.append( (iDest, -1) ) # copy from offline storage
#                flgs[iDest] = True
#    
#                #end of this cycle since we've hit our starting point, 
#                # but no need to shelve first cycle element in this case.
#                break #(end of cycle)
#            else:
#                if iSrc in shelved: #original iSrc is now at index shelved[iSrc]
#                    iSrc = shelved[iSrc]
#    
#                copyList.append( (iDest,iSrc) ) # => copy src -> dest
#                flgs[iDest] = True
#    
#                if iSrc < nIndices:
#                    #Continue cycle (swapping within "active" (index < nIndices) region)
#                    iDest = iSrc # make src the new dest
#                else:
#                    #end of this cycle, and first cycle index hasn't been
#                    # used, so shelve it (store it for later use) if it 
#                    # will be needed in the future.
#                    if cycleFirstIndex in indices:
#                        copyList.append( (iSrc,-1) )
#                        shelved[cycleFirstIndex] = iSrc
#
#                    break #(end of cycle)
#            
#    return copyList
#
## X  X     X
## 0  1  2  3 (nIndices == 4)
## 3, 0, 7, 4
## store 0
## 3 -> 0
## 4 -> 3
## stored[0] -> 4, shelved[0] = 4
## store 1
## shelved[0]==4 -> 1, NO((stored[1] -> 4, shelved[1] = 4)) B/C don't need index 1
## store 2
## 7 -> 2
## NO((Stored[2] -> 7, istore[2] = 7))
#
#
#def inplace_take(a, indices, axis=None, copyList=None):
#    check = a.take(indices, axis=axis) #DEBUGGING
#    return check #FIX FOR NOW = COPY
#
#    if axis is None:
#        def mkindex(i):
#            return i
#    else:
#        def mkindex(i):
#            sl = [slice(None)] * a.ndim
#            sl[axis] = i
#            return sl
#
#    if copyList is None:
#        copyList = build_permute_copy_order(indices)
#
#    store = None
#    for iDest,iSrc in copyList:
#        if iDest == -1: store = a[mkindex(iSrc)].copy() #otherwise just get a view!
#        elif iSrc == -1: a[mkindex(iDest)] = store
#        else: a[mkindex(iDest)] = a[mkindex(iSrc)]
#        
#    ret = a[mkindex(slice(0,len(indices)))]
#    if _np.linalg.norm(ret-check) > 1e-8 :
#        print("ERROR CHECK FAILED")
#        print("ret = ",ret)
#        print("check = ",check)
#        print("diff = ",_np.linalg.norm(ret-check))
#        assert(False)
#    #check = None #free mem?
#    #return ret
#    return check
