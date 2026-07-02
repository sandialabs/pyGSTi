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
from typing import Any, Union, TYPE_CHECKING

import numpy as np
import scipy.linalg as la

from pygsti.baseobjs import Label, ExplicitStateSpace
from pygsti.baseobjs.basis import TensorProdBasis, Basis, BuiltinBasis, BasisLike
from pygsti.tools import basistools as pgbt
from pygsti.tools import optools as pgot
from pygsti.tools.basistools import stdmx_to_vec

from pygsti.leakage.gaugeopt import _direct_sum_unitary_group

if TYPE_CHECKING:
    from pygsti.models import ExplicitOpModel
    from pygsti.processors import QubitProcessorSpec
    from pygsti.baseobjs.statespace import QuditSpace
    from pygsti.modelmembers.operations import EmbeddedOp


def _assert_hermitian_basis(mx_basis: Basis) -> None:
    """
    Reject Hilbert-Schmidt bases whose element matrices are not Hermitian.

    The models built here store their members in a *real* parameter vector. That is only
    consistent with a Hermitian (real) basis, in which the superkets/superoperators of
    physical (Hermitian-preserving) objects are real-valued. In a non-Hermitian basis such
    as the matrix-unit basis ``'std'``, gate superoperators are complex and their imaginary
    parts would be silently discarded when the parameter vector is rebuilt, quietly
    corrupting the model. Fail loudly instead.
    """
    if not mx_basis.real:
        raise ValueError(
            f"mx_basis {mx_basis.name!r} is not Hermitian: superoperators of physical "
            f"operations would be complex, but these models use a real parameter vector "
            f"(the imaginary parts would be silently discarded). Use a Hermitian basis "
            f"such as 'l2p1', 'gm', or 'qt'."
        )


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
    _assert_hermitian_basis(mx_basis)

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


def _lift_unitary_bb_to_bt(u: np.ndarray) -> np.ndarray:
    """
    Lift a two-qubit unitary to a "qubit ⨂ qutrit" unitary by adding a leakage
    level to the *second* register.

            NOTE: In what follows, C denotes the complex plane,
                  NOT the computational subspace.

    The input `u` is a 4-by-4 unitary on a tensor product of two two-dimensional
    Hilbert spaces (C^2 ⨂ C^2, "bit ⨂ bit"). The output is a 6-by-6 unitary on
    C^2 ⨂ C^3 ("bit ⨂ trit"), where the second register has gained a third level
    that models leakage.

    The computational subspace C^2 ⨂ C^2 embeds into C^2 ⨂ C^3 as the levels in
    which the second register is 0 or 1; its orthogonal complement is the leakage
    subspace (second register at level 2, for either value of the first register).
    The lift acts as `u` on the computational subspace and as the identity on the
    leakage subspace. Equivalently, in the computational-then-leakage ordering, it
    is the block-diagonal unitary u_6x6 = u ⨁ I_2.
    """
    assert u.shape == (4, 4)
    # Ordering of C^2 ⨂ C^3: flat index k = 3*a + b, where a in {0,1} is the first
    # register's level and b in {0,1,2} is the second register's level. The
    # computational levels (b in {0,1}) sit at k in {0,1,3,4}; the leakage levels
    # (b == 2) sit at k in {2,5}. Copy `u` (whose rows/cols are ordered 2*a + b) into
    # the computational block and leave the leakage block as the identity.
    comp = [3 * a + b for a in range(2) for b in range(2)]
    u_6x6 = np.eye(6, dtype='complex')
    u_6x6[np.ix_(comp, comp)] = u

    expect_zero = np.eye(6, dtype='complex') - u_6x6 @ u_6x6.T.conj()
    if (nrm := np.linalg.norm(expect_zero)) > 1e-12:
        warnings.warn(f'Nominally-unitary operator {u_6x6} fails adjoint-inverse check with norm {nrm}.')

    return u_6x6


def promote_bb_to_bt(
        qubit_model: ExplicitOpModel,
        sys0_basis: Union[str, Basis]='pp',
        sys1_basis: Union[str, Basis]='l2p1',
        levels_readout_zero=(0,), default_idle_gatename: Label = Label(())
    ) -> ExplicitOpModel:
    """
    Promote a two-qubit model to a six-dimensional "qubit ⨂ qutrit" model in which the
    second register carries an accessible leakage level.

    The input describes two qubits ("bit ⨂ bit", hence *bb*). The output describes a
    qubit tensored with a qutrit ("bit ⨂ trit", hence *bt*): the first register (`sys0`)
    stays a 2-level system, while the second register (`sys1`) is extended to a 3-level
    system whose third level models leakage. The returned model is represented in the
    tensor-product basis `sys0_basis ⨂ sys1_basis` (by default `pp ⨂ l2p1`).

    The model is built as follows:

        The Hermitian matrix representation of `rho0` is the 6-by-6 matrix with a 1 in the
        upper-left corner and zeros elsewhere.

        `Mdefault` has four effects, labeled "00", "01", "10", and "11". Effect "ab" is
        ``kron(Ea_b, Eb_t)``, where `Ea_b` is the qubit projector onto computational level
        `a` and `Eb_t` is the qutrit readout projector for outcome `b`. The qutrit's "0"
        outcome projects onto the levels in `levels_readout_zero`; its "1" outcome projects
        onto the complementary levels (including the leakage level if not in
        `levels_readout_zero`).

        Each non-idle gate of `qubit_model` is lifted to act on the 6-dimensional Hilbert
        space. Single-register gates are tensored with the identity on the other register;
        two-register gates are reordered (via SWAP conjugation) so the qubit register comes
        first. The reordered 4-by-4 unitary is lifted to a 6-by-6 unitary by
        `_lift_unitary_bb_to_bt` and converted to a superoperator in `mx_basis`.

    Parameters
    ----------
    qubit_model : ExplicitOpModel
        A two-qubit model. Its state space must have exactly two qubits.

    sys0_basis : str or Basis, optional
        Hilbert-Schmidt basis (dimension 4) for the qubit register.

    sys1_basis : str or Basis, optional
        Hilbert-Schmidt basis (dimension 9) for the qutrit register. Should imply leakage
        modeling (e.g. `'l2p1'`).

    levels_readout_zero : tuple of int, optional
        Qutrit levels that map to the "0" readout outcome. Levels not listed (including
        the leakage level if not in `levels_readout_zero`) map to the "1" outcome.

    default_idle_gatename : Label, optional
        Label used for the idle operation. If `qubit_model`'s processor spec exposes an
        `'{idle}'` gate, it is renamed to this label.

    Returns
    -------
    ExplicitOpModel
        A 6-dimensional model with a `UnitaryGaugeGroup` default gauge group and the
        `'map'` forward simulator.
    """

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
    _assert_hermitian_basis(mx_basis)
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
        u_6x6 = _lift_unitary_bb_to_bt(u_op)
        u_lifted_superop = pgot.unitary_to_superop(u_6x6, mx_basis)
        tm_6level.operations[op_lbl] = u_lifted_superop
    
    # TODO: add the idle gate to tm_6level, if applicable.

    from pygsti.models.gaugegroup import UnitaryGaugeGroup
    tm_6level.default_gauge_group = UnitaryGaugeGroup(tm_6level.state_space, mx_basis)
    tm_6level.sim = 'map'
    return tm_6level


def random_unitary_excitation(
    state_space: Union[ExplicitStateSpace, QuditSpace],
    target_subsys: Any,
    subsys_basis: BasisLike,
    ground_level: int,
    strength: float,
    rng_seed: Any=0
) -> tuple[EmbeddedOp, np.ndarray]:
    """
    Build a random unitary that couples two adjacent levels of one subsystem, embedded
    as the identity on the rest of `state_space`.

    The coupling acts on the 2-dimensional subspace of the target subsystem spanned by
    levels `ground_level` and `ground_level + 1`. We draw a random complex unit vector
    `p` supported on those two levels and form the rank-one Hermitian generator
    ``H = strength * |p><p|``. The returned operation embeds ``U = expm(1j * H)`` into
    the full state space, acting trivially on all other subsystems and levels.

    Parameters
    ----------
    state_space : ExplicitStateSpace or QuditSpace
        The full state space that the returned operation acts on. It must contain
        `target_subsys` as one of its subsystem labels.

    target_subsys : hashable
        The label (within `state_space`) of the subsystem whose levels are coupled.

    subsys_basis : str or Basis
        Hilbert-Schmidt basis for the target subsystem, used to represent the excitation
        as a superoperator. Its dimension must be the square of the subsystem's Hilbert
        space dimension (e.g. `'gm'` or `'l2p1'` of dimension 9 for a qutrit).

    ground_level : int
        Index of the lower of the two coupled levels. Both `ground_level` and
        `ground_level + 1` must be valid levels of the target subsystem, i.e. the
        subsystem's Hilbert space dimension must exceed `ground_level + 1`.

    strength : float
        Scale of the Hermitian generator; larger values produce stronger excitations.
        A value of 0 yields the identity.

    rng_seed : int or numpy.random.Generator, optional
        Seed (or generator) controlling the random unit vector `p`, forwarded to
        `numpy.random.default_rng`.

    Returns
    -------
    G_full : EmbeddedOp
        The excitation unitary embedded into `state_space`.

    p : numpy.ndarray
        The complex unit vector (of length equal to the subsystem's Hilbert space
        dimension) that defines the generator; only entries `ground_level` and
        `ground_level + 1` are nonzero.
    """
    from pygsti.modelmembers.operations import StaticUnitaryOp, EmbeddedOp

    subsys_udim = state_space.label_udimension(target_subsys)
    assert subsys_udim > ground_level + 1

    rng = np.random.default_rng(rng_seed)
    temp = rng.standard_normal((2,)) + 1j*rng.standard_normal((2,))
    p = np.zeros(subsys_udim, dtype=complex)
    p[ground_level:ground_level+2] = temp
    p /= la.norm(p)
    H = np.outer(p, p.conj())
    H *= strength
    U = la.expm(1j*H)

    ss_sub   = ExplicitStateSpace([target_subsys], [subsys_udim])
    G_excite = StaticUnitaryOp(U, basis=subsys_basis, state_space=ss_sub)
    G_full   = EmbeddedOp(state_space, (target_subsys,), G_excite)
    return G_full, p

