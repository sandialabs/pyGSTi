#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import warnings
from typing import Union, TYPE_CHECKING

import numpy as np

from pygsti.baseobjs import Label
from pygsti.baseobjs.basis import TensorProdBasis, Basis, BuiltinBasis
from pygsti.tools import basistools as pgbt
from pygsti.tools import optools as pgot
from pygsti.tools.basistools import stdmx_to_vec

from pygsti.leakage.gaugeopt import _direct_sum_unitary_group

if TYPE_CHECKING:
    from pygsti.models import ExplicitOpModel
    from pygsti.processors import QubitProcessorSpec


# TODO: write a version of this that's flexible in what it promotes from and to.
def leaky_qubit_model_from_pspec(
        ps_2level: QubitProcessorSpec, mx_basis: Union[str, Basis]='l2p1',
        levels_readout_zero=(0,), default_idle_gatename: Label = Label(())
    ) -> ExplicitOpModel:
    """
    Return an ExplicitOpModel `m` whose (ideal) gates act on three-dimensional Hilbert space and whose members
    are represented in `mx_basis`, constructed as follows:

        The Hermitian matrix representation of m['rho0'] is the 3-by-3 matrix with a 1 in the upper-left
        corner and all other entries equal to zero.
    
        Operations in `m` are defined by taking each 2-by-2 unitary `u2` from ps_2level, and promoting it
        to a 3-by-3 unitary according to

            u3 = [u2[0, 0], u2[0, 1], 0]
                 [u2[1, 0], u2[1, 1], 0]
                 [       0,       0,  1]

        m['Mdefault'] has two effects, labeled "0" and "1". If E0 is the Hermitian matrix representation of
        effect "0", then E0[i,i]=1 for all i in levels_readout_zero, and E0 is zero in all other components.

    This function might be called in a workflow like the following:

        from pygsti.models     import create_explicit_model
        from pygsti.algorithms import find_fiducials, find_germs
        from pygsti.protocols  import StandardGST, StandardGSTDesign, ProtocolData

        # Step 1: Make the experiment design for the 1-qubit system.
        tm_2level = create_explicit_model( ps_2level, ideal_spam_type='CPTPLND', ideal_gate_type='CPTPLND' )
        fids    = find_fiducials( tm_2level )
        germs   = find_germs( tm_2level )
        lengths = [1, 2, 4, 8, 16, 32]
        design  = StandardGSTDesign( tm_2level, fids[0], fids[1], germs, lengths )
        
        # Step 2: ... run the experiment specified by "design"; store results in a directory "dir" ...

        # Step 3: read in the experimental data and run GST.
        pd  = ProtocolData.from_dir(dir)
        tm_3level = leaky_qubit_model_from_pspec( ps_2level, basis='l2p1' )
        gst = StandardGST( modes=('CPTPLND',), target_model=tm_3level, verbosity=4 )
        res = gst.run(pd)
    """
    from pygsti.models.explicitmodel import ExplicitOpModel
    from pygsti.baseobjs.statespace import ExplicitStateSpace
    from pygsti.modelmembers.povms import UnconstrainedPOVM
    from pygsti.modelmembers.states import FullState
    assert ps_2level.num_qubits == 1
    if '{idle}' in ps_2level.gate_names:
        ps_2level.rename_gate_inplace('{idle}', default_idle_gatename)

    if isinstance(mx_basis, str):
        mx_basis = BuiltinBasis(mx_basis, 9)
    assert isinstance(mx_basis, Basis)

    ql = ps_2level.qubit_labels[0]
    
    Us = ps_2level.gate_unitaries
    rho0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], complex)
    E0   = np.zeros((3, 3))
    E0[levels_readout_zero, levels_readout_zero] = 1
    E1   = np.eye(3, dtype=complex) - E0

    ss = ExplicitStateSpace([ql],[3])
    tm_3level = ExplicitOpModel(ss, mx_basis) # type: ignore
    tm_3level.preps['rho0']     =  FullState(stdmx_to_vec(rho0, mx_basis))
    tm_3level.povms['Mdefault'] =  UnconstrainedPOVM(
        [("0", stdmx_to_vec(E0, mx_basis)), ("1", stdmx_to_vec(E1, mx_basis))], evotype="default",
    )

    def u2x2_to_9x9_superoperator(u2x2):
        u3x3 = np.eye(3, dtype=np.complex128)
        u3x3[:2,:2] = u2x2
        superop_std = pgot.unitary_to_std_process_mx(u3x3)
        superop = pgbt.change_basis(superop_std, 'std', mx_basis)
        return superop

    for gatename, unitary in Us.items():
        gatekey = gatename if isinstance(gatename, Label) else Label((gatename, ql))
        tm_3level.operations[gatekey] = u2x2_to_9x9_superoperator(unitary)

    subgroup_bases = [Basis.cast('pp', 4), Basis.cast('pp', 1)]
    g_full = _direct_sum_unitary_group(subgroup_bases, mx_basis)
    tm_3level.default_gauge_group = g_full
    tm_3level.sim = 'map'  # can use 'matrix', if that's preferred for whatever reason.
    return tm_3level



def promote_bb_to_bt(
        qubit_model: ExplicitOpModel,
        sys0_basis: Union[str, Basis]='pp',
        sys1_basis: Union[str, Basis]='l2p1',
        levels_readout_zero=(0,), default_idle_gatename: Label = Label(())
    ) -> ExplicitOpModel:

    from pygsti.models import ExplicitOpModel
    from pygsti.baseobjs.statespace import ExplicitStateSpace
    from pygsti.modelmembers.povms import UnconstrainedPOVM
    from pygsti.modelmembers.states import FullState

    assert qubit_model.state_space.num_qubits == 2
    ps_4level = qubit_model.create_processor_spec()
    if '{idle}' in ps_4level.gate_names:
        ps_4level.rename_gate_inplace('{idle}', default_idle_gatename)
    sys0_name, sys1_name = ps_4level.qudit_labels

    sys0_basis = Basis.cast(sys0_basis, dim=4)
    sys1_basis = Basis.cast(sys1_basis, dim=9)
    mx_basis   = TensorProdBasis((sys0_basis, sys1_basis))
    ss_6level  = ExplicitStateSpace([sys0_name, sys1_name], [2, 3])
    tm_6level  = ExplicitOpModel(ss_6level, mx_basis) # type: ignore
    tm_6level.operations[default_idle_gatename] = np.eye(36)

    I_b    = np.eye(2, dtype='cdouble')
    I_t    = np.eye(3, dtype='cdouble')
    E0_b   = np.array([[1,0],[0,0]], dtype='cdouble')
    E1_b   = I_b - E0_b
    E0_t   = np.zeros((3, 3))
    E0_t[levels_readout_zero, levels_readout_zero] = 1
    E1_t   = I_t - E0_t
    effects = {
        '00': np.kron(E0_b, E0_t),
        '01': np.kron(E0_b, E1_t),
        '10': np.kron(E1_b, E0_t),
        '11': np.kron(E1_b, E1_t)
    }
    effect_superkets = [(k, stdmx_to_vec(v, mx_basis)) for k, v in effects.items()]
    tm_6level.povms['Mdefault'] = UnconstrainedPOVM(effect_superkets, evotype='default')

    rho0 = np.zeros((6, 6))
    rho0[0,0] = 1.0
    tm_6level.preps['rho0'] = FullState(stdmx_to_vec(rho0, mx_basis))
    
    PP_1q_basis : BuiltinBasis = Basis.cast( 'PP', 4  )  # type: ignore
    PP_2q_basis : BuiltinBasis = Basis.cast( 'PP', 16 )  # type: ignore
    I_6x6 = np.eye(6, dtype='complex')

    def lift_u_2q(u: np.ndarray) -> np.ndarray:
        u_6x6 = np.zeros((6,6), 'complex')
        tj_mx = np.zeros((3,3), 'complex')
        tj_mx[2,2] = 1.0
        for ij_lbl, pij_mx in zip(PP_2q_basis.labels, PP_2q_basis.elements):  # type: ignore
            c_ij = np.vdot(pij_mx, u) / 4
            i_lbl, j_lbl = ij_lbl
            pi_mx  = PP_1q_basis.ellookup[i_lbl]
            pj_mx  = PP_1q_basis.ellookup[j_lbl]
            tj_mx[:2,:2] = pj_mx
            u_6x6 += c_ij * np.kron(pi_mx, tj_mx)
        expect_I = u_6x6 @ u_6x6.T.conj()
        if (nrm := np.linalg.norm(expect_I - I_6x6)) > 1e-14:
            warnings.warn(f'Nominally-unitary operator {op_lbl} fails self-inverse check with norm {nrm}.')
        superop = pgot.unitary_to_superop(u_6x6, mx_basis)  # type: ignore
        return superop

    from pygsti.tools.internalgates import standard_gatename_unitaries
    u_swap = standard_gatename_unitaries()['Gswap']

    non_idle_ops = [k for k in qubit_model.operations.keys() if k != Label(())]
    for op_lbl in non_idle_ops:
        u = ps_4level.gate_unitaries[op_lbl[0]]
        op_registers  = op_lbl[1:]
        num_registers = len(op_registers)
        assert u.shape == (2**num_registers, 2**num_registers)
        if op_registers[0] == sys0_name:
            u_op = np.kron(u, I_b) if num_registers == 1 else u
        else:
            u_op = np.kron(I_b, u) if num_registers == 1 else u_swap @ u @ u_swap
        u_lifted_superop = lift_u_2q(u_op)
        tm_6level.operations[op_lbl] = u_lifted_superop

    from pygsti.models.gaugegroup import UnitaryGaugeGroup
    tm_6level.default_gauge_group = UnitaryGaugeGroup(tm_6level.state_space, mx_basis)
    tm_6level.sim = 'map'
    return tm_6level
