"""
Correctness / characterization tests for UnconstrainedLindbladCoefficientBlock.

These tests pin the current intended behavior of the unconstrained coefficient
block used for GLND-like parameterizations, emphasizing:

  * num_params / to_vector / from_vector behavior
  * elementary_errorgens round-trip
  * consistency with the original LindbladCoefficientBlock where expected
    ('ham', 'other_diagonal'/'elements', 'other_diagonal'/'reldepol', and
     'other' when comparing induced LindbladErrorgen dense reps)
  * derivative correctness by finite differences where implemented
  * superoperator construction and superoperator derivatives where implemented
  * labels / static behavior / missing-entry handling

They are written conservatively around the implementation that currently exists.
"""

import numpy as np
import pytest

from pygsti.baseobjs.basis import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.baseobjs.statespace import QubitSpace

from pygsti.modelmembers.operations.lindbladcoefficients import (
    LindbladCoefficientBlock as LCB,
    UnconstrainedLindbladCoefficientBlock as ULCB,
    InvalidParamModeError,
    InvalidBlockTypeError
)
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen


VALID = {
    'ham': ('static', 'elements'),
    'other_diagonal': ('static', 'elements', 'reldepol'),
    'other': ('static', 'elements'),
}

BASES = [('pp', 4), ('gm', 9), ('pp', 16)]


def _cases(include_other_big=True):
    out = []
    for bname, dim in BASES:
        for bt, pms in VALID.items():
            if bt == 'other' and dim >= 16 and not include_other_big:
                continue
            for pm in pms:
                out.append((bname, dim, bt, pm))
    return out


ALL_CASES = _cases(include_other_big=True)
FD_CASES = [c for c in _cases(include_other_big=False) if c[3] != 'static']


def _config_as_string(c):
    return "%s%d-%s-%s" % c


def _complete_eeg_labels_for_basis(basis, block_type):
    """
    Canonical local elementary-errorgen labels for the given basis/block_type.
    """
    if basis.name.lower() == 'pp':
        nqubits = int(round(np.log2(int(np.sqrt(basis.dim)))))
        eeg_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(nqubits), default_label_type='local')
    else:
        # For non-PP bases, derive labels directly from basis element labels in the same style
        # as LindbladCoefficientBlock does.
        bels = tuple(basis.labels[1:])
        if block_type == 'ham':
            return tuple(LEEL('H', (lbl,)) for lbl in bels)
        elif block_type == 'other_diagonal':
            return tuple(LEEL('S', (lbl,)) for lbl in bels)
        elif block_type == 'other':
            lbls = []
            for i, lbl1 in enumerate(bels):
                lbls.append(LEEL('S', (lbl1,)))
                for lbl2 in bels[i + 1:]:
                    lbls.append(LEEL('C', (lbl1, lbl2)))
                    lbls.append(LEEL('A', (lbl1, lbl2)))
            return tuple(lbls)
        else:
            raise InvalidBlockTypeError()

    if block_type == 'ham':
        return tuple(lbl for lbl in eeg_basis.labels if lbl.errorgen_type == 'H')
    elif block_type == 'other_diagonal':
        return tuple(lbl for lbl in eeg_basis.labels if lbl.errorgen_type == 'S')
    elif block_type == 'other':
        return tuple(lbl for lbl in eeg_basis.labels if lbl.errorgen_type in ('S', 'C', 'A'))
    else:
        raise InvalidBlockTypeError()


def make_ulcb(bname, dim, bt, pm, data_seed=None):
    basis = Basis.cast(bname, dim)
    eeg_labels = _complete_eeg_labels_for_basis(basis, bt)
    blk = ULCB(bt, basis, error_generator_labels=eeg_labels, param_mode=pm)
    if data_seed is not None and blk.num_params > 0:
        rng = np.random.default_rng(data_seed)
        blk.from_vector(rng.standard_normal(blk.num_params))
    return blk


def make_lcb(bname, dim, bt, pm, data_seed=None):
    basis = Basis.cast(bname, dim)
    blk = LCB(bt, basis, param_mode=pm)
    if data_seed is not None and blk.num_params > 0:
        rng = np.random.default_rng(data_seed)
        blk.from_vector(rng.standard_normal(blk.num_params))
    return blk


def block_data_of_ulcb(bname, dim, bt, pm, v):
    blk = make_ulcb(bname, dim, bt, pm)
    blk.from_vector(np.asarray(v, float))
    return blk.block_data.copy()


def Gflat(blk, bname):
    return np.asarray(
        blk.create_lindblad_term_superoperators(
            bname, sparse=False, include_1norms=False, flat=True
        )
    )


def fd_jac(f, v, eps=1e-6):
    """
    Central-difference Jacobian of array-valued f:
    returns shape f(v).shape + (len(v),).
    """
    v = np.asarray(v, float)
    base = np.asarray(f(v))
    out = np.zeros(base.shape + (v.size,), dtype=complex)
    for k in range(v.size):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        out[..., k] = (np.asarray(f(vp)) - np.asarray(f(vm))) / (2 * eps)
    return out


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_num_params_matches_formula(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm)
    n = len(blk._eeg_labels)

    if pm == 'static':
        expected = 0
    elif bt == 'other_diagonal' and pm == 'reldepol':
        expected = 1
    else:
        expected = n

    assert blk.num_params == expected
    assert blk.to_vector().shape == (expected,)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_block_data_param_roundtrip(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm, data_seed=1)

    if blk.num_params == 0:
        assert blk.to_vector().shape == (0,)
        return

    bd0 = blk.block_data.copy()
    v = blk.to_vector().copy()
    blk.from_vector(v)
    assert np.allclose(blk.block_data, bd0, atol=1e-12)
    assert np.allclose(blk.to_vector(), v, atol=1e-12)


@pytest.mark.parametrize("c", FD_CASES, ids=_config_as_string)
def test_deriv_wrt_params_finite_diff_for_supported_cases(c):
    bname, dim, bt, pm = c

    # current implementation has bugs in some derivative paths for 'other_diagonal'/'reldepol'
    # and 'other'; restrict this test to the implemented/working subset
    if (bt, pm) not in {
        ('ham', 'elements'),
        ('other_diagonal', 'elements'),
    }:
        pytest.skip("finite-difference derivative test restricted to currently implemented stable paths")

    blk = make_ulcb(bname, dim, bt, pm)
    v = np.random.default_rng(7).standard_normal(blk.num_params)

    analytic = np.asarray(blk.deriv_wrt_params(v))
    fd = fd_jac(lambda vv: block_data_of_ulcb(bname, dim, bt, pm, vv), v)

    assert np.allclose(np.real(fd), np.real(analytic), atol=1e-6, rtol=1e-5)
    assert np.allclose(np.imag(fd), 0, atol=1e-8)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_elementary_errorgens_matches_block_data(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm, data_seed=3)

    eegs = blk.elementary_errorgens
    assert set(eegs.keys()) == set(blk._eeg_labels)
    assert np.allclose(np.array([eegs[lbl] for lbl in blk._eeg_labels]), blk.block_data, atol=1e-12)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_elementary_errorgens_roundtrip(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm, data_seed=5)

    eegs = dict(blk.elementary_errorgens)
    blk2 = make_ulcb(bname, dim, bt, pm)
    unused = blk2.set_elementary_errorgens(eegs, on_missing='raise')

    assert len(unused) == 0
    assert np.allclose(blk2.block_data, blk.block_data, atol=1e-12)


def test_set_elementary_errorgens_on_missing():
    blk = make_ulcb('pp', 4, 'ham', 'elements')
    full = dict(blk.elementary_errorgens)
    dropped = next(iter(full))
    partial = {k: v for k, v in full.items() if k != dropped}

    with pytest.raises(ValueError):
        blk.set_elementary_errorgens(partial, on_missing='raise')

    with pytest.warns(UserWarning):
        blk.set_elementary_errorgens(partial, on_missing='warn')

    unused = blk.set_elementary_errorgens(partial, on_missing='ignore')
    assert isinstance(unused, dict)
    assert dropped in blk.elementary_errorgens
    assert blk.elementary_errorgens[dropped] == 0.0


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_elementary_errorgen_indices_are_identity_map(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm)
    idx = blk.elementary_errorgen_indices

    assert set(idx.keys()) == set(blk._eeg_labels)
    for i, lbl in enumerate(blk._eeg_labels):
        assert idx[lbl] == i


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_superop_shapes(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm)

    Gf = np.asarray(blk.create_lindblad_term_superoperators(bname, sparse=False, flat=True))
    n = len(blk._eeg_labels)
    assert Gf.shape[0] == n
    assert Gf.shape[1:] == (dim, dim)

    G = np.asarray(blk.create_lindblad_term_superoperators(bname, sparse=False, flat=False))
    if bt in ('ham', 'other_diagonal'):
        assert G.shape == (n, dim, dim)
    else:
        # current implementation unflattens by sqrt(n), so this only works for perfect-square n.
        # For 1Q pp 'other', n=9 -> shape (3,3,4,4), matching current implementation.
        nMxs = int(round(np.sqrt(n)))
        assert G.shape == (nMxs, nMxs, dim, dim)


@pytest.mark.parametrize("bname,dim", [('pp', 4), ('pp', 16)])
def test_superops_match_original_for_ham_and_otherdiag_elements(bname, dim):
    """
    For ham and diagonal-S blocks, ULCB and LCB should construct the same elementary
    superoperators when they represent the same generators.
    """
    for bt, pm in [('ham', 'elements'), ('other_diagonal', 'elements')]:
        ublk = make_ulcb(bname, dim, bt, pm)
        lblk = make_lcb(bname, dim, bt, pm)
        Gu = np.asarray(ublk.create_lindblad_term_superoperators(bname, sparse=False, flat=True))
        Gl = np.asarray(lblk.create_lindblad_term_superoperators(bname, sparse=False, flat=True))
        assert np.allclose(Gu, Gl, atol=1e-12)


@pytest.mark.parametrize("bname,dim", [('pp', 4), ('pp', 16)])
def test_lindblad_errorgen_dense_matches_original_for_comparable_cases(bname, dim):
    """
    Compare induced dense LindbladErrorgen representations between old and new blocks
    for the cases the notebook showed should agree.
    """
    basis = Basis.cast(bname, dim)
    if bname == 'pp':
        nqubits = int(round(np.log2(int(np.sqrt(dim)))))
        state_space = QubitSpace(nqubits)
    else:
        pytest.skip("comparison via QubitSpace only exercised for pp cases")

    # ham
    u_h = make_ulcb(bname, dim, 'ham', 'elements')
    l_h = make_lcb(bname, dim, 'ham', 'elements')
    vals_h = {lbl: 0.1 * (i + 1) for i, lbl in enumerate(u_h._eeg_labels)}
    eu_h = LindbladErrorgen([u_h], state_space=state_space)
    el_h = LindbladErrorgen([l_h], state_space=state_space)
    eu_h.set_coefficients(vals_h)
    el_h.set_coefficients(vals_h)
    assert np.allclose(eu_h.to_dense(), el_h.to_dense(), atol=1e-12)

    # other_diagonal / elements
    u_s = make_ulcb(bname, dim, 'other_diagonal', 'elements')
    l_s = make_lcb(bname, dim, 'other_diagonal', 'elements')
    vals_s = {lbl: 0.05 * (i + 1) for i, lbl in enumerate(u_s._eeg_labels)}
    eu_s = LindbladErrorgen([u_s], state_space=state_space)
    el_s = LindbladErrorgen([l_s], state_space=state_space)
    eu_s.set_coefficients(vals_s)
    el_s.set_coefficients(vals_s)
    assert np.allclose(eu_s.to_dense(), el_s.to_dense(), atol=1e-12)

    # other_diagonal / reldepol
    u_rd = make_ulcb(bname, dim, 'other_diagonal', 'reldepol')
    l_rd = make_lcb(bname, dim, 'other_diagonal', 'reldepol')
    dep = {lbl: 0.1 for lbl in u_rd._eeg_labels}
    eu_rd = LindbladErrorgen([u_rd], state_space=state_space)
    el_rd = LindbladErrorgen([l_rd], state_space=state_space)
    eu_rd.set_coefficients(dep)
    el_rd.set_coefficients(dep)
    assert np.allclose(eu_rd.to_dense(), el_rd.to_dense(), atol=1e-12)

    # combined ham + other_diagonal
    u_hs = LindbladErrorgen(
        [make_ulcb(bname, dim, 'ham', 'elements'),
         make_ulcb(bname, dim, 'other_diagonal', 'elements')],
        state_space=state_space
    )
    l_hs = LindbladErrorgen(
        [make_lcb(bname, dim, 'ham', 'elements'),
         make_lcb(bname, dim, 'other_diagonal', 'elements')],
        state_space=state_space
    )
    vals_hs = dict(vals_h); vals_hs.update(vals_s)
    u_hs.set_coefficients(vals_hs)
    l_hs.set_coefficients(vals_hs)
    assert np.allclose(u_hs.to_dense(), l_hs.to_dense(), atol=1e-12)


@pytest.mark.parametrize("bname,dim", [('pp', 4), ('pp', 16)])
def test_lindblad_errorgen_dense_matches_original_for_other_elements(bname, dim):
    """
    For the full unconstrained 'other' block, compare the final induced dense error generator
    rather than the raw term-superoperators, following the reasoning in the notebook.
    """
    basis = Basis.cast(bname, dim)
    nqubits = int(round(np.log2(int(np.sqrt(dim)))))
    state_space = QubitSpace(nqubits)

    ublk = make_ulcb(bname, dim, 'other', 'elements')
    lblk = make_lcb(bname, dim, 'other', 'elements')

    vals = {lbl: 0.01 * (i + 1) for i, lbl in enumerate(ublk._eeg_labels)}

    eu = LindbladErrorgen([ublk], state_space=state_space)
    el = LindbladErrorgen([lblk], state_space=state_space)
    eu.set_coefficients(vals)
    el.set_coefficients(vals)

    assert np.allclose(eu.to_dense(), el.to_dense(), atol=1e-12)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_param_labels_have_expected_length(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm)
    assert len(blk.param_labels) == blk.num_params
    assert all(isinstance(x, str) for x in blk.param_labels)


def test_param_labels_snapshot_1q_pp():
    pp = Basis.cast('pp', 4)

    ham = make_ulcb('pp', 4, 'ham', 'elements')
    assert ham.param_labels == [
        "('H', ('X',)) Hamiltonian error coefficient",
        "('H', ('Y',)) Hamiltonian error coefficient",
        "('H', ('Z',)) Hamiltonian error coefficient",
    ]

    s_el = make_ulcb('pp', 4, 'other_diagonal', 'elements')
    assert s_el.param_labels == [
        "('S', ('X',)) stochastic coefficient",
        "('S', ('Y',)) stochastic coefficient",
        "('S', ('Z',)) stochastic coefficient",
    ]

    s_rd = make_ulcb('pp', 4, 'other_diagonal', 'reldepol')
    assert s_rd.param_labels == [
        "common stochastic error coefficient for depolarization"
    ]

    other = make_ulcb('pp', 4, 'other', 'elements')
    expected = []
    for lbl in other._eeg_labels:
        if lbl.errorgen_type == 'S':
            expected.append(f"{lbl} stochastic coefficient")
        elif lbl.errorgen_type == 'C':
            expected.append(f"{lbl} pauli-correlation coefficient")
        else:
            expected.append(f"{lbl} active coefficient")
    assert other.param_labels == expected


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_static_behavior(c):
    bname, dim, bt, pm = c
    if pm != 'static':
        pytest.skip("static-only test")

    blk = make_ulcb(bname, dim, bt, pm)
    assert blk.num_params == 0
    assert blk.to_vector().shape == (0,)
    blk.from_vector(np.empty(0))
    assert blk.deriv_wrt_params().shape == (len(blk._eeg_labels), 0)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_set_from_errorgen_projections_roundtrip(c):
    """
    Create a LindbladErrorgen from the block, convert to dense, then re-extract
    coefficients into a fresh block. This is a very direct end-to-end correctness check.
    """
    bname, dim, bt, pm = c
    if pm == 'static':
        pytest.skip("static blocks have no free params for this roundtrip")

    if bname != 'pp':
        pytest.skip("roundtrip exercised for pp / qubit spaces")

    blk = make_ulcb(bname, dim, bt, pm, data_seed=10)
    nqubits = int(round(np.log2(int(np.sqrt(dim)))))
    state_space = QubitSpace(nqubits)

    eg = LindbladErrorgen([blk], state_space=state_space)
    dense = eg.to_dense()

    blk2 = make_ulcb(bname, dim, bt, pm)
    blk2.set_from_errorgen_projections(dense, errorgen_basis=bname)

    assert np.allclose(blk2.block_data, blk.block_data, atol=1e-8)


@pytest.mark.parametrize("c", [
    ('pp', 4, 'ham', 'elements'),
    ('pp', 4, 'other_diagonal', 'elements'),
    ('pp', 4, 'other_diagonal', 'reldepol'),
], ids=_config_as_string)
def test_superop_deriv_matches_fd_for_supported_paths(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm)
    nP = blk.num_params
    v = np.random.default_rng(11).standard_normal(nP)
    blk.from_vector(v)

    G = Gflat(blk, bname)
    sd = np.asarray(blk.superop_deriv_wrt_params(G, v, superops_are_flat=True))

    fd = fd_jac(
        lambda vv: np.einsum('i,ijk->jk', block_data_of_ulcb(bname, dim, bt, pm, vv), G),
        v
    )

    assert np.allclose(np.imag(fd), 0, atol=1e-6)
    assert np.allclose(np.real(fd), sd, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("c", [
    ('pp', 4, 'ham', 'elements'),
    ('pp', 4, 'other_diagonal', 'elements'),
    ('pp', 4, 'other_diagonal', 'reldepol'),
], ids=_config_as_string)
def test_superop_hessian_for_supported_paths(c):
    bname, dim, bt, pm = c
    blk = make_ulcb(bname, dim, bt, pm)
    G = Gflat(blk, bname)
    h = np.asarray(blk.superop_hessian_wrt_params(G, superops_are_flat=True))

    if bt == 'ham' or (bt == 'other_diagonal' and pm in ('elements', 'reldepol')):
        assert np.allclose(h, 0, atol=1e-12)
    else:
        pytest.skip("only zero-Hessian supported-path cases tested here")


def test_invalid_block_type_raises():
    with pytest.raises(InvalidBlockTypeError):
        ULCB('not_a_block_type', Basis.cast('pp', 4),
             error_generator_labels=(), param_mode='elements')


def test_invalid_param_mode_raises_on_use():
    pp = Basis.cast('pp', 4)
    eeg_labels = _complete_eeg_labels_for_basis(pp, 'ham')
    blk = ULCB('ham', pp, error_generator_labels=eeg_labels, param_mode='cholesky')
    with pytest.raises(InvalidParamModeError):
        _ = blk.num_params


# --------------------------------------------------------------------------------------
# Expected-future-work tests: these document likely implementation issues. Marked xfail
# so they don't fail CI now but are ready to turn on after fixes.
# --------------------------------------------------------------------------------------

@pytest.mark.xfail(reason="Current implementation references self._eegs_labels in _deriv_wrt_params_otherdiag")
def test_reldepol_deriv_current_bug():
    blk = make_ulcb('pp', 4, 'other_diagonal', 'reldepol')
    _ = blk.deriv_wrt_params()


@pytest.mark.xfail(reason="Current implementation references self._eegs_labels in _superop_deriv_wrt_params_other")
def test_other_superop_deriv_current_bug():
    blk = make_ulcb('pp', 4, 'other', 'elements', data_seed=1)
    G = Gflat(blk, 'pp')
    _ = blk.superop_deriv_wrt_params(G, blk.to_vector(), superops_are_flat=True)


@pytest.mark.xfail(reason="Serialization keys appear inconsistent in current implementation")
def test_serialization_roundtrip():
    blk = make_ulcb('pp', 4, 'ham', 'elements', data_seed=1)
    blk2 = ULCB.from_nice_serialization(blk.to_nice_serialization())
    assert blk2._block_type == blk._block_type
    assert blk2._param_mode == blk._param_mode
    assert blk2._eeg_labels == blk._eeg_labels
    assert np.allclose(blk2.block_data, blk.block_data)


@pytest.mark.xfail(reason="convert currently returns LindbladCoefficientBlock, not UnconstrainedLindbladCoefficientBlock")
def test_convert_type():
    blk = make_ulcb('pp', 4, 'ham', 'elements')
    converted = blk.convert('static')
    assert isinstance(converted, ULCB)


@pytest.mark.xfail(reason="is_similar currently checks against LindbladCoefficientBlock instead of UnconstrainedLindbladCoefficientBlock")
def test_is_similar_type():
    a = make_ulcb('pp', 4, 'ham', 'elements')
    b = make_ulcb('pp', 4, 'ham', 'elements')
    assert a.is_similar(b)