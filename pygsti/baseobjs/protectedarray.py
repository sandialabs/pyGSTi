"""
Defines the ProtectedArray class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from typing import Sequence, Union, Optional, Iterable
import copy as _copy
import numpy as _np
from warnings import warn

from pygsti.baseobjs import _compatibility as _compat

PROTECTEDARRAYERRORSTRING = "**some or all of assignment destination is read-only"

# A single index specification along each axis of the array, e.g. `(0, slice(None, None, None))`.
_IndexSpec = Sequence[Union[int, slice]]

class ProtectedArray(object):
    """
    A numpy ndarray-like class that allows certain elements to be treated as read-only.

    Note that all in-place arithmetic/logical operators (e.g. `+=`, `-=`,
    `*=`, `/=`, `//=`, `**=`, `%=`, `@=`, `<<=`, `>>=`, `|=`, `^=`, `&=`)
    unconditionally raise a `ValueError` on `ProtectedArray` instances, even
    when no indices are currently protected (i.e. when `protected_index_mask`
    is all-zero). Use `__setitem__` (e.g. `pa[:] = pa[:] + x`) instead, which
    only raises when a protected index would actually be modified.

    Parameters
    ----------
    input_array : numpy.ndarray
        The base array.

    indices_to_protect : int or list of tuples, optional
        A list of length `input_array.shape`, specifying
        the indices to protect along each axis.  Values may be
        integers, slices, or lists of integers,
        e.g. `(0, slice(None, None, None))`.
        Also supported are iterables over tuples/lists, each
        of length `input_array.shape`, specifying
        the indices to protect along each axis.

    protected_index_mask : numpy.ndarray, optional
        An optional array with the same shape as `input_array` which if
        specified is used to initialize the mask for protected indices
        used by this array. Note that is specified the value overrides
        any specification given in indices_to_protect, meaning that argument
        is ignored.
    """

    def __init__(self, input_array: _np.ndarray,
                 indices_to_protect: Optional[Union[int, _IndexSpec, Iterable[_IndexSpec]]]=None,
                 protected_index_mask: Optional[_np.ndarray]=None):
        self.base = input_array

        if protected_index_mask is not None:
            #check this has the correct shape
            assert protected_index_mask.shape == input_array.shape
            
            #Cast this to a binary dtype (to save space since we only
            #need boolean values).
            self.protected_index_mask = protected_index_mask.astype(_np.bool_)

        #otherwise use the value passed into indices to protect to construct
        #a mask.
        #add in support for multiple sets of indices to protect
        #by allowing a nested iterable format. Do this by forcing
        #everything into this format and then looping over the nested
        #submembers.
        elif indices_to_protect is not None:
            if isinstance(indices_to_protect, int):
                indices_to_protect= [(indices_to_protect,)]
            #if this is a list go through and wrap any integers
            #at the top level in a tuple.
            elif isinstance(indices_to_protect, (list, tuple)):
                #check whether this is a single-level tuple/list corresponding
                #containing only ints and/or slices. If so wrap this in a list.
                if all([isinstance(idx, (int, slice)) for idx in indices_to_protect]):
                    indices_to_protect = [indices_to_protect]
                
                #add some logic for mixing of unwrapped top-level ints and tuples/lists.
                indices_to_protect = [tuple(indices) if isinstance(indices, (list, tuple)) else (indices,) for indices in indices_to_protect]
            #initialize an empty mask
            self.protected_index_mask = _np.zeros(input_array.shape , dtype= _np.bool_)

            #now loop over the nested subelements and add them to the mask:
            for indices in indices_to_protect:
                assert(len(indices) <= len(self.base.shape))
                if len(indices) < len(self.base.shape):
                    warn("You must fully specify the indices you wish to protect." +
                         "e.g. for a (3,3) array you would need to specify the row and the column for each individual element" +
                         " in the array you wish to protect. Note that a slice can allow you to specify multiple at once." +
                         "So to specify the zeroth row, (0, slice(0, None, None).)", RuntimeWarning)
                else:
                    self.protected_index_mask[indices]=1
        #otherwise set the mask to all zeros.
        else:
            self.protected_index_mask = _np.zeros(input_array.shape , dtype= _np.bool_)
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

    # Arithmetic Assignment Operations
    def __imul__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __iadd__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __itruediv__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __ifloordiv__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __ipow__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)    
    
    def __isub__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __imod__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __imatmul__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)

    # Logical Assignment Operations
    def __ilshift__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)

    def __irshift__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __ior__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __ixor__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)
    
    def __iand__(self, other):
        raise ValueError(PROTECTEDARRAYERRORSTRING)

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
        #Use key to extract subarray of self.base and self.protected_index_mask
        ret = self.base[key]
        new_protected_mask = self.protected_index_mask[key]

        #If ret is not a scalar return a new ProtectedArray corresponding to the
        #selected subarray with the set of protected indices inherited over from the
        #original.
        if not _np.isscalar(ret):
            if not _np.all(new_protected_mask):  # then some of the indices are writeable
                ret = ProtectedArray(ret, protected_index_mask= new_protected_mask)
            else: #otherwise all of the values are masked off.
                ret = _np.require(ret.copy(), requirements=['OWNDATA'])  # copy to a new read-only array
                ret.flags.writeable = False  # a read-only array
                ret = ProtectedArray(ret, protected_index_mask=new_protected_mask)  # return a ProtectedArray that is read-only
        return ret

    def __setitem__(self, key, val):
                #check if any of the indices in key have been masked off.
        if _np.any(self.protected_index_mask[key]):  # assigns to a protected index in each dim
            raise ValueError(PROTECTEDARRAYERRORSTRING)
        #not sure what the original logic was for this return statement, but I don't see any
        #harm in keeping it.
        return self.base.__setitem__(key, val)

    #add a repr method that prints the base array, which is typically what
    #we want.
    def __repr__(self):
        return _np.array2string(self.base)
        