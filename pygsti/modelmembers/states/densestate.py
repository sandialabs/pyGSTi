"""
The DenseState and DensePureState classes and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import numpy as _np
import copy as _copy

from pygsti.modelmembers.states.state import State as _State
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs import _compatibility as _compat
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti import SpaceT


class DenseStateInterface(object):
    """
    REMOVED: This class formerly added a numpy-array-mimicking interface onto
    state objects. It has been deleted as part of expiring the deprecated dense
    interface (pyGSTi issue #447). Use .to_dense() to read the array
    representation and .set_dense() / .from_vector() to write it.

    This stub exists temporarily to produce descriptive errors at sites that
    still rely on the old interface; it will be fully deleted once all call
    sites are corrected.
    """

    def __init__(self):
        pass

    @staticmethod
    def _removed(what):
        raise TypeError(
            "The dense array interface has been removed. "
            "Attempted operation: '%s'. "
            "Use .to_dense() to obtain a numpy array for reading, "
            "and .set_dense() or .from_vector() to update the state." % what
        )

    def __copy__(self):
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def __deepcopy__(self, memo):
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, _copy.deepcopy(v, memo))
        return cpy

    def __getitem__(self, key): self._removed('__getitem__[%s]' % str(key))
    def __setitem__(self, key, val): self._removed('__setitem__[%s]' % str(key))
    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)
        self._removed('attribute access .%s' % attr)
    def __pos__(self): self._removed('unary +')
    def __neg__(self): self._removed('unary -')
    def __abs__(self): self._removed('abs()')
    def __add__(self, x): self._removed('__add__')
    def __radd__(self, x): self._removed('__radd__')
    def __sub__(self, x): self._removed('__sub__')
    def __rsub__(self, x): self._removed('__rsub__')
    def __mul__(self, x): self._removed('__mul__')
    def __rmul__(self, x): self._removed('__rmul__')
    def __truediv__(self, x): self._removed('__truediv__')
    def __rtruediv__(self, x): self._removed('__rtruediv__')
    def __floordiv__(self, x): self._removed('__floordiv__')
    def __rfloordiv__(self, x): self._removed('__rfloordiv__')
    def __pow__(self, x): self._removed('__pow__')
    def __eq__(self, x): self._removed('__eq__')
    def __len__(self): self._removed('__len__')
    def __int__(self): self._removed('__int__')
    def __float__(self): self._removed('__float__')
    def __complex__(self): self._removed('__complex__')


class DenseState(DenseStateInterface, _State):
    """
    TODO: update docstring
    A state preparation vector that is interfaced/behaves as a dense super-ket (a numpy array).

    This class is the common base class for parameterizations of a state vector
    that have a dense representation and can be accessed like a numpy array.

    Parameters
    ----------
    vec : numpy.ndarray
        The SPAM vector as a dense numpy array.

    evotype : {"statevec", "densitymx"}
        The evolution type.

    Attributes
    ----------
    _base_1d : numpy.ndarray
        Direct access to the underlying 1D array.

    base : numpy.ndarray
        Direct access the the underlying data as column vector,
        i.e, a (dim,1)-shaped array.
    """

    def __init__(self, vec, basis, evotype, state_space):
        vec = _State._to_vector(vec)
        if state_space is None:
            state_space = _statespace.default_space_for_dim(vec.shape[0])
        else:
            state_space = _statespace.StateSpace.cast(state_space)
        evotype = _Evotype.cast(evotype, state_space=state_space)
        self._basis = _Basis.cast(basis, state_space.dim)
        rep = evotype.create_dense_state_rep(vec, self._basis, state_space)

        _State.__init__(self, rep, evotype)
        DenseStateInterface.__init__(self)
        #assert(self._ptr.flags['C_CONTIGUOUS'] and self._ptr.flags['OWNDATA'])  # not true for TPState

    @property
    def _ptr(self):
        return self._rep.base

    def _ptr_has_changed(self):
        """ 
        Derived classes should override this function to handle rep updates
        when the `_ptr` property is changed. 
        """
        self._rep.base_has_changed()

    def to_dense(self, on_space: SpaceT='minimal', scratch=None):
        """
        Return the dense array used to represent this state within its evolution type.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        #don't use scratch since we already have memory allocated
        return self._rep.to_dense(on_space)  # both types of possible state reps implement 'to_dense'

    def to_memoized_dict(self, mmg_memo):
        """
        Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['dense_superket_vector'] = self._encodemx(self.to_dense())
        mm_dict['basis'] = self._basis.to_nice_serialization() if (self._basis is not None) else None

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        vec = cls._decodemx(mm_dict['dense_superket_vector'])
        try:
            state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
            basis = _Basis.from_nice_serialization(mm_dict['basis']) if (mm_dict['basis'] is not None) else None
            return cls(vec, basis, mm_dict['evotype'], state_space)
        except AssertionError as e:
            """
            This codepath can get hit when deserializing TPPOVM or UnconstrainedPOVM objects.
            
            More specifically, it can get hit when objects other than POVMEffect vectors were
            passed to the constructors of these classes. When that happens, their base class
            constructor (for BasePOVM) constructs the effects as FullPOVMEffect objects (which in
            turn rely on FullState and then DenseState) with None passed for the basis argument.
            Somewhere downstream that None gets cast to a Basis object where `.dim == 0`, which
            is inconsistent with the array representation of the effect having length > 0.
            
            In an ideal world we'd have POVMs enforce not only non-None bases but also *consistent*
            bases for all constituent effects. For now, our fix is to just try and recover the basis
            from serial_memo. (Perhaps not-coincidentally, this function wasn't using the serial_memo
            argument before this code path was added.)
            """
            se = str(e)
            if 'Basis object has unexpected dimension' in se and len(serial_memo) > 0:
                member = list(serial_memo.values())[0]
                basis = member.parent.basis
                state_space = basis.state_space
                return cls(vec, basis, mm_dict['evotype'], state_space)
            raise e

    def _is_similar(self, other, rtol, atol):
        """ 
        Returns True if `other` model member (which it guaranteed to be the same type as self) has
        the same local structure, i.e., not considering parameter values or submembers. 
        """
        
        return self._ptr.shape == other._ptr.shape  # similar (up to params) if have same data shape


class DensePureState(DenseStateInterface, _State):
    """
    TODO: docstring - a state that is interfaced as a dense ket
    """

    def __init__(self, purevec, basis, evotype, state_space):
        purevec = _State._to_vector(purevec)
        purevec = purevec.astype(complex)
        state_space = _statespace.default_space_for_udim(purevec.shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)
        evotype = _Evotype.cast(evotype, state_space=state_space)
        basis = _Basis.cast(basis, state_space.dim)  # basis for Hilbert-Schmidt (superop) space
        
        #Try to create a dense pure rep.  If this fails, see if a dense superket rep
        # can be created, as this type of rep can also hold arbitrary pure states.
        try:
            rep = evotype.create_pure_state_rep(purevec, basis, state_space)
            self._reptype = 'pure'
            self._purevec = None
            self._basis = basis #this was previously being set as None, not sure why.
        except Exception:
            if len(purevec) == basis.dim and _np.linalg.norm(purevec.imag) < 1e-10:
                # Special case when a *superket* was provided instead of a purevec
                superket_vec = purevec.real  # used as a convenience case that really shouldn't be used
            else:
                superket_vec = _bt.change_basis(_ot.state_to_dmvec(purevec), 'std', basis)
            rep = evotype.create_dense_state_rep(superket_vec, basis, state_space)
            self._reptype = 'superket'
            self._purevec = purevec; self._basis = basis

        _State.__init__(self, rep, evotype)
        DenseStateInterface.__init__(self)

    @property
    def _ptr(self):
        """
        Gives a handle/pointer to the base numpy array that this object can be accessed as.
        """
        return self._rep.base if self._reptype == 'pure' else self._purevec

    def _ptr_has_changed(self):
        """
        Derived classes should override this function to handle rep updates
        when the `_ptr` property is changed. 
        """
        if self._reptype == 'superket':
            self._rep.base[:] = _bt.change_basis(_ot.state_to_dmvec(self._purevec), 'std', self._basis)
        self._rep.base_has_changed()

    def to_dense(self, on_space: SpaceT='minimal', scratch=None):
        """
        Return the dense array used to represent this state within its evolution type.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        #don't use scratch since we already have memory allocated
        if self._reptype == 'superket' and on_space == 'Hilbert':
            return self._purevec
        return self._rep.to_dense(on_space)  # both types of possible state reps implement 'to_dense'

    def to_memoized_dict(self, mmg_memo):
        """
        Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['dense_state_vector'] = self._encodemx(self.to_dense("Hilbert"))
        mm_dict['basis'] = self._basis.to_nice_serialization() if (self._basis is not None) else None
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        vec = cls._decodemx(mm_dict['dense_state_vector'])
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        basis = _Basis.from_nice_serialization(mm_dict['basis'])
        return cls(vec, basis, mm_dict['evotype'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ 
        Returns True if `other` model member (which it guaranteed to be the same type as self) has
        the same local structure, i.e., not considering parameter values or submembers 
        """
        return self._ptr.shape == other._ptr.shape  # similar (up to params) if have same data shape
