"""
State representations for "qibo" evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import functools as _functools

from .. import basereps as _basereps
from . import _get_densitymx_mode, _get_minimal_space
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.tools import internalgates as _itgs
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot

try:
    from ...tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None

try:
    import qibo as _qibo
except ImportError:
    _qibo = None


class StateRep(_basereps.StateRep):
    def __init__(self, qibo_circuit, qibo_state, state_space):
        self.qibo_circuit = qibo_circuit
        self.qibo_state = qibo_state
        self.state_space = _StateSpace.cast(state_space)
        assert(self.qibo_circuit is None or self.state_space.num_qubits == self.qibo_circuit.nqubits), \
            'Number-of-qubits mismatch between state space and circuit for "qubit" evotype'

    @property
    def num_qubits(self):
        return self.state_space.num_qubits

    def copy(self):
        return StateRep(self.qibo_circuit, self.qibo_state, self.state_space)

    def actionable_staterep(self):
        # return a state rep that can be acted on by op reps or mapped to
        # a probability/amplitude by POVM effect reps.
        return self  # for most classes, the rep itself is actionable


class StateRepDensePure(StateRep):
    def __init__(self, purevec, state_space, basis):
        state_space = _StateSpace.cast(state_space)
        qibo_circuit = _qibo.models.Circuit(state_space.num_qubits, density_matrix=_get_densitymx_mode())
        self.basis = basis
        super(StateRepDensePure, self).__init__(qibo_circuit, purevec, state_space)

    @property
    def base(self):
        return self.qibo_state

    def base_has_changed(self):
        pass

    def to_dense(self, on_space):
        if on_space == 'Hilbert' or (on_space == 'minimal' and _get_minimal_space() == 'Hilbert'):
            return self.base
        elif on_space == 'HilbertSchmidt' or (on_space == 'minimal' and _get_minimal_space() == 'HilbertSchmidt'):
            return _bt.change_basis(_ot.state_to_dmvec(self.base), 'std', self.basis)
        else:
            raise ValueError("Invalid `on_space` argument: %s" % str(on_space))


class StateRepDense(StateRep):
    def __init__(self, data, state_space, basis):
        assert(_get_densitymx_mode() is True), "Must set pygsti.evotypes.qibo.densitymx_mode=True to use dense states!"
        state_space = _StateSpace.cast(state_space)
        qibo_circuit = _qibo.models.Circuit(state_space.num_qubits, density_matrix=True)
        self.basis = basis
        self.std_basis = _Basis.cast('std', state_space.dim)  # the basis the qibo expects
        self.data = data
        self.udim = state_space.udim
        qibo_state = _bt.change_basis(data, basis, self.std_basis).reshape((self.udim, self.udim))
        super(StateRepDense, self).__init__(qibo_circuit, qibo_state, state_space)

    @property
    def base(self):
        return self.data  # state in self.basis (not self.std_basis)

    def base_has_changed(self):
        self.qibo_state = _bt.change_basis(self.data, self.basis, self.std_basis).reshape((self.udim, self.udim))

    def to_dense(self, on_space):
        if not (on_space == 'HilbertSchmidt' or (on_space == 'minimal' and _get_minimal_space() == 'HilbertSchmidt')):
            raise ValueError("'densitymx' evotype cannot produce Hilbert-space ops!")
        return self.data


class StateRepComputational(StateRep):
    def __init__(self, zvals, basis, state_space):
        assert all([nm in ('pp', 'PP') for nm in basis.name.split('*')]), \
            "Only Pauli basis is allowed for 'chp' evotype"

        #Convert zvals to dense vec:
        factor_dim = 2
        v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
        v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        v = (v0, v1)

        if _fastcalc is None:  # do it the slow way using numpy
            vec = _functools.reduce(_np.kron, [v[i] for i in zvals])
        else:
            typ = complex
            fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(zvals), factor_dim), typ))
            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(zvals), _np.int64))
            for i, zi in enumerate(zvals):
                fast_kron_array[i, :] = v[zi]
            vec = _np.ascontiguousarray(_np.empty(factor_dim**len(zvals), typ))
            _fastcalc.fast_kron_complex(vec, fast_kron_array, fast_kron_factordims)

        self.zvals = zvals
        self.basis = basis

        if _qibo is None: raise ValueError("qibo is not installed! Must `pip install qibo` to use the 'qibo' evotype")
        state_space = _StateSpace.cast(state_space)
        qibo_circuit = _qibo.models.Circuit(state_space.num_qubits, density_matrix=_get_densitymx_mode())
        super(StateRepComputational, self).__init__(qibo_circuit, vec, state_space)


class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__(None, None, state_space)  # this state rep is *not* actionable

    def reps_have_changed(self):
        pass  # not needed -- don't actually hold ops

    def actionable_staterep(self):
        state_rep = self.state_rep.actionable_staterep()
        return self.op_rep.acton(state_rep)

    @property
    def basis(self):
        # (all qibo state reps need to have a .basis property)
        return self.state_rep.basis

#REMOVE
#    def chp_ops(self, seed_or_state=None):
#        return self.state_rep.chp_ops(seed_or_state=seed_or_state) \
#            + self.op_rep.chp_ops(seed_or_state=seed_or_state)

# TODO: Untested, only support computational and composed for now
#class StateRepTensorProduct(StateRep):
#    def __init__(self, factor_state_reps, state_space):
#        self.factor_reps = factor_state_reps
#        super(StateRepTensorProduct, self).__init__([], state_space)
#        self.reps_have_changed()
#
#    def reps_have_changed(self):
#        chp_ops = []
#        current_iqubit = 0
#        for factor in self.factor_reps:
#            local_to_tp_index = {str(iloc): str(itp) for iloc, itp in
#                                 enumerate(range(current_iqubit, current_iqubit + factor.num_qubits))}
#            chp_ops.extend([_update_chp_op(op, local_to_tp_index) for op in self.chp_ops])
#            current_iqubit += factor.num_qubits
#        self.chp_ops = chp_ops
