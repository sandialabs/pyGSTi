"""
Operation representation classes for the `qibo` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import copy as _copy
from functools import partial as _partial

import numpy as _np
from scipy.sparse.linalg import LinearOperator
from numpy.random import RandomState as _RandomState

from . import _get_minimal_space
from .statereps import StateRep as _StateRep
from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.baseobjs.basis import Basis as _Basis
from ...tools import jamiolkowski as _jt
from ...tools import basistools as _bt
from ...tools import internalgates as _itgs
from ...tools import optools as _ot


try:
    import qibo as _qibo

    std_qibo_creation_fns = {  # functions that create the desired op given qubit indices & gate args
        'Gi': _qibo.gates.I,
        'Gxpi2': _partial(_qibo.gates.RX, theta=_np.pi / 2, trainable=False),
        'Gypi2': _partial(_qibo.gates.RY, theta=_np.pi / 2, trainable=False),
        'Gzpi2': _partial(_qibo.gates.RZ, theta=_np.pi / 2, trainable=False),
        'Gxpi': _qibo.gates.X,
        'Gypi': _qibo.gates.Y,
        'Gzpi': _qibo.gates.Z,
        'Gxmpi2': _partial(_qibo.gates.RX, theta=-_np.pi / 2, trainable=False),
        'Gympi2': _partial(_qibo.gates.RY, theta=-_np.pi / 2, trainable=False),
        'Gzmpi2': _partial(_qibo.gates.RZ, theta=-_np.pi / 2, trainable=False),
        'Gh': _qibo.gates.H,
        'Gp': _qibo.gates.S,
        'Gpdag': _partial(_qibo.gates.U1, theta=-_np.pi / 2, trainable=False),
        'Gt': _qibo.gates.T,
        'Gtdag': _partial(_qibo.gates.U1, theta=-_np.pi / 4, trainable=False),
        'Gcphase': _qibo.gates.CZ,
        'Gcnot': _qibo.gates.CNOT,
        'Gswap': _qibo.gates.SWAP,
        #'Gzr': _qibo.gates.RZ,  # takes (q, theta)
        #'Gczr': _qibo.gates.CRZ,  # takes (q0, q1, theta)
        'Gx': _partial(_qibo.gates.RX, theta=_np.pi / 2, trainable=False),
        'Gy': _partial(_qibo.gates.RY, theta=_np.pi / 2, trainable=False),
        'Gz': _partial(_qibo.gates.RZ, theta=_np.pi / 2, trainable=False)
    }
except ImportError:
    _qibo = None


class OpRep(_basereps.OpRep):
    def __init__(self, state_space):
        self.state_space = state_space

    @property
    def dim(self):
        return self.state_space.udim

    def create_qibo_ops_on(self, qubit_indices):
        raise NotImplementedError("Derived classes must implement this!")

    def acton(self, state):
        c = state.qibo_circuit.copy()
        # TODO: change below to: sole_tensor_product_block_labels
        for qibo_op in self.create_qibo_ops_on(self.state_space.tensor_product_block_labels(0)):
            c.add(qibo_op)
        return _StateRep(c, state.qibo_state.copy(), state.state_space)

    def adjoint_acton(self, state):
        raise NotImplementedError()

    def acton_random(self, state, rand_state):
        return self.acton(state)  # default is to ignore rand_state

    def adjoint_acton_random(self, state, rand_state):
        return self.adjoint_acton(state)  # default is to ignore rand_state

#    def aslinearoperator(self):
#        def mv(v):
#            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
#            in_state = _StateRepDensePure(_np.ascontiguousarray(v, complex), self.state_space, basis=None)
#            return self.acton(in_state).to_dense('Hilbert')
#
#        def rmv(v):
#            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
#            in_state = _StateRepDensePure(_np.ascontiguousarray(v, complex), self.state_space, basis=None)
#            return self.adjoint_acton(in_state).to_dense('Hilbert')
#        return LinearOperator((self.dim, self.dim), matvec=mv, rmatvec=rmv)  # transpose, adjoint, dot, matmat?

    def copy(self):
        return _copy.deepcopy(self)


class OpRepDenseUnitary(OpRep):
    def __init__(self, mx, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.udim, complex)
        assert(mx.ndim == 2 and mx.shape[0] == state_space.udim)
        self.basis = basis
        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        super(OpRepDenseUnitary, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        return [_qibo.gates.UnitaryChannel([1.0], [(qubit_indices, self.base)], seed=None)]

    def base_has_changed(self):
        pass  # nothing needed

    def to_dense(self, on_space):
        if on_space == 'Hilbert' or (on_space == 'minimal' and _get_minimal_space() == 'Hilbert'):
            return self.base
        elif on_space == 'HilbertSchmidt' or (on_space == 'minimal' and _get_minimal_space() == 'HilbertSchmidt'):
            return _ot.unitary_to_superop(self.base, self.basis)
        else:
            raise ValueError("Invalid `on_space` argument: %s" % str(on_space))

    def __str__(self):
        return "OpRepDenseUnitary:\n" + str(self.base)


class OpRepDenseSuperop(OpRep):
    def __init__(self, mx, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.dim, 'd')
        assert(mx.ndim == 2 and mx.shape[0] == state_space.dim)
        self.basis = basis
        assert(self.basis is not None), "Qibo evotype requires OpRepDenseSuperop be given a basis (to get Kraus ops!)"

        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        super(OpRepDenseSuperop, self).__init__(state_space)
        self.base_has_changed()  # sets self.kraus_ops

    def base_has_changed(self):
        #recompute Kraus ops for creating qibo op
        superop_mx = self.base; d = int(_np.round(_np.sqrt(superop_mx.shape[0])))
        std_basis = _Basis.cast('std', superop_mx.shape[0])
        choi_mx = _jt.jamiolkowski_iso(superop_mx, self.basis, std_basis) * d  # see NOTE below
        evals, evecs = _np.linalg.eig(choi_mx)
        assert(all([ev > -1e-7 for ev in evals])), \
            "Cannot compute Kraus decomposition of non-positive-definite superoperator (within OpRepDenseSuperop!)"
        self.kraus_ops = [evecs[:, i].reshape(d, d) * _np.sqrt(ev) for i, ev in enumerate(evals) if abs(ev) > 1e-7]

    def to_dense(self, on_space):
        if not (on_space == 'HilbertSchmidt' or (on_space == 'minimal' and _get_minimal_space() == 'HilbertSchmidt')):
            raise ValueError("'densitymx_slow' evotype cannot produce Hilbert-space ops!")
        return self.base

    def create_qibo_ops_on(self, qubit_indices):
        return [_qibo.gates.KrausChannel([(qubit_indices, Ki) for Ki in self.kraus_ops])]

    def __str__(self):
        return "OpRepDenseSuperop:\n" + str(self.base)

    def copy(self):
        return OpRepDenseSuperop(self.base.copy(), self.basis, self.state_space)


class OpRepStandard(OpRep):
    def __init__(self, name, basis, state_space):
        self.name = name
        if self.name not in std_qibo_creation_fns:
            raise ValueError("Standard name '%s' is not available in 'qibo' evotype" % self.name)

        self.basis = basis  # used anywhere?
        self.creation_fn = std_qibo_creation_fns[name]
        # create the desired op given qubit indices & gate args

        super(OpRepStandard, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        return [self.creation_fn(*qubit_indices)]


#class OpRepStochastic(OpRepDense):
# - maybe we could add this, but it wouldn't be a "dense" op here,
#   perhaps we need to change API?


class OpRepComposed(OpRep):
    # exactly the same as densitymx case
    def __init__(self, factor_op_reps, state_space):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_op_reps
        super(OpRepComposed, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        return list(_itertools.chain(*[f.create_qibo_ops_on(qubit_indices) for f in self.factor_reps]))

    def reinit_factor_op_reps(self, new_factor_op_reps):
        self.factors_reps = new_factor_op_reps


# This might work, but we won't need it unless we get OpRepExpErrorgen, etc, working.
#class OpRepSum(OpRep):
#    # exactly the same as densitymx case
#    def __init__(self, factor_reps, state_space):
#        #assert(len(factor_reps) > 0), "Composed gates must contain at least one factor gate!"
#        self.factor_reps = factor_reps
#        super(OpRepSum, self).__init__(state_space)
#
#    def acton(self, state):
#        """ Act this gate map on an input state """
#        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
#        for f in self.factor_reps:
#            output_state.data += f.acton(state).data
#        return output_state
#
#    def adjoint_acton(self, state):
#        """ Act the adjoint of this operation matrix on an input state """
#        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
#        for f in self.factor_reps:
#            output_state.data += f.adjoint_acton(state).data
#        return output_state
#
#    def acton_random(self, state, rand_state):
#        """ Act this gate map on an input state """
#        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
#        for f in self.factor_reps:
#            output_state.data += f.acton_random(state, rand_state).data
#        return output_state
#
#    def adjoint_acton_random(self, state, rand_state):
#        """ Act the adjoint of this operation matrix on an input state """
#        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
#        for f in self.factor_reps:
#            output_state.data += f.adjoint_acton_random(state, rand_state).data
#        return output_state


class OpRepEmbedded(OpRep):

    def __init__(self, state_space, target_labels, embedded_rep):
        self.target_labels = target_labels
        self.embedded_rep = embedded_rep
        super(OpRepEmbedded, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        # TODO: change below to: sole_tensor_product_block_labels
        assert(qubit_indices == self.state_space.tensor_product_block_labels(0))
        return self.embedded_rep.create_qibo_ops_on(self.target_labels)


#REMOVE
#class OpRepExpErrorgen(OpRep):
#
#    def __init__(self, errorgen_rep):
#        state_space = errorgen_rep.state_space
#        self.errorgen_rep = errorgen_rep
#        super(OpRepExpErrorgen, self).__init__(state_space)
#
#    def errgenrep_has_changed(self, onenorm_upperbound):
#        pass
#
#    def acton(self, state):
#        raise AttributeError("Cannot currently act with statevec.OpRepExpErrorgen - for terms only!")
#
#    def adjoint_acton(self, state):
#        raise AttributeError("Cannot currently act with statevec.OpRepExpErrorgen - for terms only!")


class OpRepRepeated(OpRep):
    def __init__(self, rep_to_repeat, num_repetitions, state_space):
        state_space = _StateSpace.cast(state_space)
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        super(OpRepRepeated, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        return [self.repeated_rep.create_qibo_ops_on(qubit_indices)] * self.num_repetitions


#REMOVE
#class OpRepLindbladErrorgen(OpRep):
#    def __init__(self, lindblad_coefficient_blocks, state_space):
#        super(OpRepLindbladErrorgen, self).__init__(state_space)
#        self.Lterms = None
#        self.Lterm_coeffs = None
#        self.lindblad_coefficient_blocks = lindblad_coefficient_blocks


class OpRepKraus(OpRep):
    def __init__(self, basis, kraus_reps, state_space):
        self.basis = basis
        self.kraus_reps = kraus_reps  # superop reps in this evotype (must be reps of *this* evotype)
        assert(all([isinstance(rep, OpRepDenseUnitary) for rep in kraus_reps]))
        state_space = _StateSpace.cast(state_space)
        assert(self.basis.dim == state_space.dim)
        super(OpRepKraus, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        kraus_ops = [Krep.base for Krep in self.kraus_reps]
        kraus_norms = list(map(_np.linalg.norm, kraus_ops))
        return [_qibo.gates.KrausChannel([(qubit_indices, Ki)
                                          for Ki, nrm in zip(kraus_ops, kraus_norms) if nrm > 1e-7])]

    def __str__(self):
        return "OpRepKraus with ops\n" + str(self.kraus_reps)

    def copy(self):
        return OpRepKraus(self.basis, list(self.kraus_reps), None, self.state_space)

    def to_dense(self, on_space):
        assert(on_space == 'HilbertSchmidt' or (on_space == 'minimal' and _get_minimal_space() == 'HilbertSchmidt')), \
            'Can only compute OpRepKraus.to_dense on HilbertSchmidt space!'
        return sum([rep.to_dense(on_space) for rep in self.kraus_reps])


class OpRepRandomUnitary(OpRep):
    def __init__(self, basis, unitary_rates, unitary_reps, seed_or_state, state_space):
        self.basis = basis
        self.unitary_reps = unitary_reps
        self.unitary_rates = unitary_rates.copy()

        if isinstance(seed_or_state, _RandomState):
            self.rand_state = seed_or_state
        else:
            self.rand_state = _RandomState(seed_or_state)

        self.state_space = _StateSpace.cast(state_space)
        assert(self.basis.dim == self.state_space.dim)
        super(OpRepRandomUnitary, self).__init__(state_space)

    def create_qibo_ops_on(self, qubit_indices):
        return [_qibo.gates.UnitaryChannel(self.unitary_rates, [(qubit_indices, Uk.to_dense('Hilbert'))
                                                                for Uk in self.unitary_reps],
                                           seed=self.rand_state.randint(0, 2**30))]  # HARDCODED 2**30!! (max seed)

    def __str__(self):
        return "OpRepRandomUnitary:\n" + " rates: " + str(self.unitary_rates)  # maybe show ops too?

    def copy(self):
        return OpRepRandomUnitary(self.basis, self.unitary_rates, list(self.unitary_reps),
                                  self.rand_state, self.state_space)

    def update_unitary_rates(self, rates):
        self.unitary_rates[:] = rates

    def to_dense(self, on_space):
        assert(on_space == 'HilbertSchmidt')  # below code only works in this case
        return sum([rate * rep.to_dense(on_space) for rate, rep in zip(self.unitary_rates, self.unitary_reps)])


class OpRepStochastic(OpRepRandomUnitary):

    def __init__(self, stochastic_basis, basis, initial_rates, seed_or_state, state_space):
        self.rates = initial_rates
        self.stochastic_basis = stochastic_basis
        rates = [1 - sum(initial_rates)] + list(initial_rates)
        reps = [OpRepDenseUnitary(bel, basis, state_space) for bel in stochastic_basis.elements]
        assert(len(reps) == len(rates))

        state_space = _StateSpace.cast(state_space)
        assert(basis.dim == state_space.dim)
        self.basis = basis

        super(OpRepStochastic, self).__init__(basis, _np.array(rates, 'd'), reps, seed_or_state, state_space)

    def update_rates(self, rates):
        unitary_rates = [1 - sum(rates)] + list(rates)
        self.rates[:] = rates
        self.update_unitary_rates(unitary_rates)
