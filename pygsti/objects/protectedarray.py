"""Defines the ProtectedArray class"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy
from ..tools import compattools as _compat


class ProtectedArray(object):
    """
    A numpy ndarray-like class that allows certain elements to
    be treated as read-only.
    """

    def __init__(self, input_array, indicesToProtect=None):
        self.base = input_array

        #Get protected indices, a specified as:
        self.indicesToProtect = []
        if indicesToProtect is not None:
            if not (isinstance(indicesToProtect, tuple)
                    or isinstance(indicesToProtect, list)):
                indicesToProtect = (indicesToProtect,)

            assert(len(indicesToProtect) <= len(self.base.shape))
            for ky, L in zip(indicesToProtect, self.base.shape):
                if isinstance(ky, slice):
                    pindices = range(*ky.indices(L))
                elif _compat.isint(ky):
                    i = ky + L if ky < 0 else ky
                    if i < 0 or i > L:
                        raise IndexError("index (%d) is out of range." % ky)
                    pindices = (i,)
                elif isinstance(ky, list):
                    pindices = ky
                else: raise TypeError("Invalid index type: %s" % type(ky))
                self.indicesToProtect.append(pindices)

        if len(self.indicesToProtect) == 0:
            self.indicesToProtect = None

        #Note: no need to set self.base.flags.writeable = True anymore,
        # since this flag can only apply to a data owner as of numpy 1.16 or so.
        # Instead, we just copy the data whenever we return a readonly array.
        #Here, we just leave the writeable flag of self.base alone (either value is ok)

    #Mimic array behavior
    def __pos__(self): return self.base
    def __neg__(self): return -self.base
    def __abs__(self): return abs(self.base)
    def __add__(self, x): return self.base + x
    def __radd__(self, x): return x + self.base
    def __sub__(self, x): return self.base - x
    def __rsub__(self, x): return x - self.base
    def __mul__(self, x): return self.base * x
    def __rmul__(self, x): return x * self.base
    def __truediv__(self, x): return self.base / x
    def __rtruediv__(self, x): return x / self.base
    def __floordiv__(self, x): return self.base // x
    def __rfloordiv__(self, x): return x // self.base
    def __pow__(self, x): return self.base ** x
    def __eq__(self, x): return self.base == x
    def __len__(self): return len(self.base)
    def __int__(self): return int(self.base)
    def __long__(self): return int(self.base)
    def __float__(self): return float(self.base)
    def __complex__(self): return complex(self.base)

    #Pickle plumbing

    def __reduce__(self):
        return (ProtectedArray, (_np.zeros(self.base.shape),), self.__dict__)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, _copy.deepcopy(v, memo))
        return result

    def __setstate__(self, state):
        self.__dict__.update(state)

    #Access to underlying ndarray

    def __getattr__(self, attr):
        # set references to our memory as (entirely) read-only
        ret = getattr(self.__dict__['base'], attr)
        if isinstance(ret, _np.ndarray) and ret.base is self.base:
            ret = _np.require(ret.copy(), requirements=['OWNDATA'])  # copy to a new read-only array
            ret.flags.writeable = False  # as of numpy 1.16, this can only be set by OWNER
        return ret

    def __getslice__(self, i, j):
        #For special cases when getslice is still called, e.g. A[:] in Python 2.7
        return self.__getitem__(slice(i, j))

    def __getitem__(self, key):

        writeable = True

        #check if key matches/overlaps protected region
        if self.indicesToProtect is not None:
            new_indicesToProtect = []; nUnprotectedIndices = 0
            tup_key = key if isinstance(key, tuple) else (key,)

            while len(tup_key) < len(self.base.shape):
                tup_key = tup_key + (slice(None, None, None),)

            for ky, pindices, L in zip(tup_key, self.indicesToProtect, self.base.shape):

                #Get requested indices
                if isinstance(ky, slice):
                    indices = range(*ky.indices(L))

                    new_pindices = []
                    for ii, i in enumerate(indices):
                        if i in pindices:
                            new_pindices.append(ii)  # index of i within indices
                    new_pindices = sorted(list(set(new_pindices)))
                    new_indicesToProtect.append(new_pindices)

                    #tally how many indices in this dimension are unprotected
                    nTotalInDim = len(indices)
                    nUnprotectedInCurDim = (len(indices) - len(new_pindices))

                elif _compat.isint(ky):
                    i = ky + L if ky < 0 else ky
                    if i > L:
                        raise IndexError("The index (%d) is out of range." % ky)

                    nTotalInDim = 1
                    if i not in pindices:  # single index that is unprotected => all unprotected
                        nUnprotectedInCurDim = 1  # a single unprotected index
                    else:
                        nUnprotectedInCurDim = 0

                else: raise TypeError("Invalid index type: %s" % type(ky))

                nUnprotectedIndices += nUnprotectedInCurDim

                #if there exists a single dimension with no protected indices, then
                # the whole array is writeable.
                if nTotalInDim == nUnprotectedInCurDim:
                    writeable = True
                    new_indicesToProtect = None
                    break

            else:
                # if we didn't break b/c of above block, which means each dim has
                # at least one protected index

                #if there are no unprotected indices, then just set writeable == False
                if nUnprotectedIndices == 0:
                    writeable = False
                    new_indicesToProtect = None
                else:
                    #There is at least one writeable (unprotected) index in some dimension
                    # and at least one protected index in *every* dimension. We need to
                    # set indicesToProtect to describe what to protect
                    assert(len(new_indicesToProtect) > 0)  # b/c otherwise another case would hold
                    writeable = True
                    new_indicesToProtect = tuple(new_indicesToProtect)

        else:  # (if nothing is protected)
            writeable = True
            new_indicesToProtect = None

        ret = _np.ndarray.__getitem__(self.base, key)

        if not _np.isscalar(ret):
            if writeable:  # then some of the indices are writeable
                ret = ProtectedArray(ret)
                ret.indicesToProtect = new_indicesToProtect
            else:
                ret = _np.require(ret.copy(), requirements=['OWNDATA'])  # copy to a new read-only array
                ret.flags.writeable = False  # a read-only array
                ret = ProtectedArray(ret)  # return a ProtectedArray that is read-only

            #print "   writeable = ",ret.flags.writeable
            #print "   new_toProtect = ",ret.indicesToProtect
            #print "<< END getitem"
        return ret

    def __setitem__(self, key, val):
        #print "In setitem with key = ", key, "val = ",val

        protectionViolation = []  # per dimension
        if self.indicesToProtect is not None:
            tup_key = key if isinstance(key, tuple) else (key,)
            for ky, pindices, L in zip(tup_key, self.indicesToProtect, self.base.shape):

                #Get requested indices
                if isinstance(ky, slice):
                    indices = range(*ky.indices(L))
                    if any(i in pindices for i in indices):
                        protectionViolation.append(True)
                    else: protectionViolation.append(False)

                elif _compat.isint(ky):
                    i = ky + L if ky < 0 else ky
                    if i > L:
                        raise IndexError("The index (%d) is out of range." % ky)
                    protectionViolation.append(i in pindices)

                else: raise TypeError("Invalid index type: %s" % type(ky))

            if all(protectionViolation):  # assigns to a protected index in each dim
                raise ValueError("**assignment destination is read-only")
        return self.base.__setitem__(key, val)
