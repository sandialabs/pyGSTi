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
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot


class DenseState(_State):
    """
    A state preparation vector that is parameterized as a dense super-ket.

    This class is the common base class for parameterizations of a state vector
    that have a dense representation.

    Attributes
    ----------
    _base_1d : numpy.ndarray
        Direct access to the underlying 1D array.

    base : numpy.ndarray
        Direct access the the underlying data as column vector,
        i.e, a (dim,1)-shaped array.
    """

    def __init__(self, vec, basis, evotype, state_space):
        """
        Parameters
        ----------
        vec : numpy.ndarray
            The SPAM vector as a dense numpy array.

        evotype : {"statevec", "densitymx"}
            The evolution type.
        
        state_space : `StateSpace`, optional (default None)
            The state space for this state.  If `None` a default state space
            with the appropriate number of qubits is used.
        """
        
        
        vec = _State._to_vector(vec)
        if state_space is None:
            state_space = _statespace.default_space_for_dim(vec.shape[0])
        else:
            state_space = _statespace.StateSpace.cast(state_space)
        evotype = _Evotype.cast(evotype, state_space=state_space)
        self._basis = _Basis.cast(basis, state_space.dim)
        rep = evotype.create_dense_state_rep(vec, self._basis, state_space)

        _State.__init__(self, rep, evotype)

    @property
    def _ptr(self):
        return self._rep.base

    def _ptr_has_changed(self):
        """ 
        Derived classes should override this function to handle rep updates
        when the `_ptr` property is changed. 
        """
        self._rep.base_has_changed()

    def to_dense(self, on_space='minimal', scratch=None):
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
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        basis = _Basis.from_nice_serialization(mm_dict['basis']) if (mm_dict['basis'] is not None) else None
        return cls(vec, basis, mm_dict['evotype'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ 
        Returns True if `other` model member (which it guaranteed to be the same type as self) has
        the same local structure, i.e., not considering parameter values or submembers. 
        """
        
        return self._ptr.shape == other._ptr.shape  # similar (up to params) if have same data shape

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__, self.dim)
        s += _mt.mx_to_string(self.to_dense(on_space='minimal'), width=4, prec=2)
        return s

class DensePureState(_State):
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

    def to_dense(self, on_space='minimal', scratch=None):
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

        mm_dict['dense_state_vector'] = self._encodemx(self.to_dense('Hilbert'))
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

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__, self.dim)
        s += _mt.mx_to_string(self.to_dense(on_space='minimal'), width=4, prec=2)
        return s