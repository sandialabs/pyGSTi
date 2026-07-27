"""
Tests for the flat / unconstrained non-Hamiltonian Lindblad coefficient block
(``block_type='other_unconstrained'``), folded into ``LindbladCoefficientBlock``.

The shared (block_type, param_mode) characterization matrix in
``test_lindbladcoefficients.py`` already exercises this block for num_params,
to/from_vector round-trips, ``deriv_wrt_params`` (finite diff), the superop
derivative / Hessian chain rules, elementary-errorgen round-trips, NicelySerializable
round-trips, and ``convert``.  In particular the superop-derivative finite-diff test
there covers what was an ``xfail``'d bug ('other' superop derivative) on the
feature branch, and the ``other_diagonal``/``reldepol`` derivative (another former
``xfail``) is covered there too.

This file adds the behavior unique to the block: the flat 1-D per-eeg representation,
reduced / explicit ``error_generator_labels``, equivalence with the legacy
Hermitian-matrix 'other' block (which also validates the ``normalized_paulis`` fix on
the 'pp' basis), the term-object reconstruction, and the fixed ``is_similar`` /
``convert`` / serialization behaviors for reduced blocks.
"""
import numpy as np
import pytest

from pygsti.baseobjs.basis import Basis
from pygsti.baseobjs.statespace import QubitSpace, default_space_for_dim
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.tools import basistools as _bt
from pygsti.modelmembers.operations.lindbladcoefficients import (
    LindbladCoefficientBlock as LCB, _OtherUnconstrainedCoeffBlock, InvalidParamModeError)
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen


BASES = [('pp', 4), ('gm', 9), ('pp', 16)]
_BASIS_IDS = ["%s%d" % b for b in BASES]


def _errorgen_dense(blk):
    """Dense error generator (Hilbert-Schmidt space, in the block's own basis) induced by `blk`."""
    ss = default_space_for_dim(blk._basis.dim)
    eg = LindbladErrorgen([blk], elementary_errorgen_basis=blk._basis,
                          mx_basis=blk._basis, state_space=ss)
    return eg.to_dense(on_space='HilbertSchmidt')


# ---------------------------------------------------------------- construction / labels

def test_block_type_dispatch_and_shape():
    pp = Basis.cast('pp', 4)
    blk = LCB('other_unconstrained', pp, param_mode='elements')
    assert isinstance(blk, _OtherUnconstrainedCoeffBlock)
    # full 1-qubit set: S(X),S(Y),S(Z) + C/A for (X,Y),(X,Z),(Y,Z) => 3 + 2*3 = 9
    assert blk.num_params == 9
    assert blk.block_data.shape == (9,)
    assert blk.block_data.dtype == np.dtype('d')
    assert {lbl.errorgen_type for lbl in blk._eeg_labels} == {'S', 'C', 'A'}


def test_invalid_param_mode_rejected():
    pp = Basis.cast('pp', 4)
    for pm in ('cholesky', 'depol', 'reldepol', 'not_a_mode'):
        with pytest.raises(InvalidParamModeError):
            LCB('other_unconstrained', pp, param_mode=pm)


def test_elementary_errorgen_indices_are_identity_map():
    pp = Basis.cast('pp', 4)
    blk = LCB('other_unconstrained', pp, param_mode='elements')
    idx = blk.elementary_errorgen_indices
    assert list(idx.keys()) == list(blk._eeg_labels)
    for i, combo in enumerate(idx.values()):
        assert combo == [(1.0, i)]
    assert blk._block_data_indices == {i: [(1.0, lbl)] for i, lbl in enumerate(blk._eeg_labels)}


def test_coefficient_and_param_labels():
    pp = Basis.cast('pp', 4)
    blk = LCB('other_unconstrained', pp, param_mode='elements')
    labels = blk.param_labels
    assert len(labels) == blk.num_params
    assert any('stochastic' in s for s in labels)
    assert any('pauli-correlation' in s for s in labels)
    assert any('active' in s for s in labels)
    assert blk.param_labels == labels   # 'elements' => one parameter per coefficient
    assert LCB('other_unconstrained', pp, param_mode='static').param_labels == []


def test_static_behavior():
    pp = Basis.cast('pp', 4)
    blk = LCB('other_unconstrained', pp, param_mode='static')
    assert blk.num_params == 0
    assert blk.to_vector().shape == (0,)
    assert blk.deriv_wrt_params().shape == (blk.block_data.shape[0], 0)
    blk.from_vector(np.empty(0))   # static carries no parameters -> no-op


# ---------------------------------------------------------------- equivalence with legacy 'other'

@pytest.mark.parametrize("bname,dim", BASES, ids=_BASIS_IDS)
def test_dense_errorgen_matches_legacy_other(bname, dim):
    """The flat block reproduces the legacy Hermitian 'other' block's induced dense error
    generator for matching elementary errorgens.  On the 'pp' basis this also validates the
    `normalized_paulis` scaling in create_elementary_errorgen_pauli."""
    basis = Basis.cast(bname, dim)
    flat = LCB('other_unconstrained', basis, param_mode='elements')
    leg = LCB('other', basis, param_mode='elements')
    rng = np.random.default_rng(0)
    coeffs = {lbl: rng.standard_normal() for lbl in flat.elementary_errorgens.keys()}
    flat.set_elementary_errorgens(dict(coeffs))
    leg.set_elementary_errorgens(dict(coeffs))
    assert np.allclose(_errorgen_dense(flat), _errorgen_dense(leg), atol=1e-9)


# ---------------------------------------------------------------- reduced parameterization

def test_reduced_parameterization():
    pp = Basis.cast('pp', 4)
    eegs = [LEEL('S', ('X',)), LEEL('C', ('X', 'Y')), LEEL('A', ('X', 'Y'))]
    blk = LCB('other_unconstrained', pp, error_generator_labels=eegs, param_mode='elements')
    assert blk.num_params == 3
    assert list(blk.elementary_errorgens.keys()) == eegs

    rng = np.random.default_rng(1)
    coeffs = {lbl: rng.standard_normal() for lbl in eegs}
    blk.set_elementary_errorgens(dict(coeffs))
    # a full block carrying only those three coefficients (others zero) induces the same errorgen
    full = LCB('other_unconstrained', pp, param_mode='elements')
    full.set_elementary_errorgens(dict(coeffs))   # absent labels default to 0
    assert np.allclose(_errorgen_dense(blk), _errorgen_dense(full), atol=1e-9)


def test_reduced_basis_element_labels_first_seen_order():
    pp = Basis.cast('pp', 4)
    # full block built from basis elements should round-trip its basis_element_labels
    full = LCB('other_unconstrained', pp, param_mode='elements')
    assert tuple(full.basis_element_labels) == ('X', 'Y', 'Z')
    # reduced block reports the distinct bels appearing in its eegs, first-seen
    red = LCB('other_unconstrained', pp,
              error_generator_labels=[LEEL('S', ('Z',)), LEEL('C', ('Z', 'X'))], param_mode='elements')
    assert tuple(red.basis_element_labels) == ('Z', 'X')


def test_invalid_label_inputs():
    pp = Basis.cast('pp', 4)
    with pytest.raises(ValueError):   # 'H' not managed by 'other_unconstrained'
        LCB('other_unconstrained', pp, error_generator_labels=[LEEL('H', ('X',))], param_mode='elements')
    with pytest.raises(ValueError):   # can't specify both label kinds
        LCB('other_unconstrained', pp, basis_element_labels=('X', 'Y'),
            error_generator_labels=[LEEL('S', ('X',))], param_mode='elements')
    with pytest.raises(ValueError):   # legacy block types reject error_generator_labels
        LCB('ham', pp, error_generator_labels=[LEEL('H', ('X',))], param_mode='elements')


# ---------------------------------------------------------------- set_from_errorgen_projections

def test_set_from_errorgen_projections_roundtrip():
    pp = Basis.cast('pp', 4)
    blk = LCB('other_unconstrained', pp, param_mode='elements')
    rng = np.random.default_rng(8)
    coeffs = {lbl: rng.standard_normal() for lbl in blk.elementary_errorgens.keys()}
    blk.set_elementary_errorgens(dict(coeffs))
    errgen = _errorgen_dense(blk)
    blk2 = LCB('other_unconstrained', pp, param_mode='elements')
    blk2.set_from_errorgen_projections(errgen, errorgen_basis='pp')
    assert np.allclose(blk2.block_data, blk.block_data, atol=1e-9)


# ---------------------------------------------------------------- fixed feature-branch xfails

@pytest.mark.parametrize("reduced", [False, True], ids=['full', 'reduced'])
def test_serialization_roundtrip_preserves_eeg_labels(reduced):
    """Former xfail: NicelySerializable round-trip.  The eeg labels (not just the bels) must
    survive so reduced/subset blocks reconstruct exactly."""
    pp = Basis.cast('pp', 4)
    if reduced:
        eegs = [LEEL('S', ('X',)), LEEL('C', ('X', 'Z')), LEEL('A', ('X', 'Z'))]
        blk = LCB('other_unconstrained', pp, error_generator_labels=eegs, param_mode='elements')
    else:
        blk = LCB('other_unconstrained', pp, param_mode='elements')
    blk.from_vector(np.random.default_rng(4).standard_normal(blk.num_params))
    blk2 = LCB.from_nice_serialization(blk.to_nice_serialization())
    assert isinstance(blk2, _OtherUnconstrainedCoeffBlock)
    assert blk2._eeg_labels == blk._eeg_labels
    assert blk2._param_mode == blk._param_mode
    assert np.allclose(blk2.block_data, blk.block_data, atol=1e-12)


def test_convert_returns_correct_type_and_preserves_eegs():
    """Former xfail: convert() must return an _OtherUnconstrainedCoeffBlock (not the legacy block)."""
    pp = Basis.cast('pp', 4)
    eegs = [LEEL('S', ('Y',)), LEEL('C', ('Y', 'Z'))]
    blk = LCB('other_unconstrained', pp, error_generator_labels=eegs, param_mode='elements')
    blk.from_vector(np.array([0.3, -0.2]))
    st = blk.convert('static')
    assert isinstance(st, _OtherUnconstrainedCoeffBlock)
    assert st._param_mode == 'static' and st._eeg_labels == blk._eeg_labels
    assert np.allclose(st.block_data, blk.block_data)
    back = st.convert('elements')
    assert isinstance(back, _OtherUnconstrainedCoeffBlock) and back._param_mode == 'elements'
    assert np.allclose(back.block_data, blk.block_data)


def test_is_similar_distinguishes_eeg_subsets():
    """Former xfail: is_similar must check the eeg label set (reduced != full) and the type."""
    pp = Basis.cast('pp', 4)
    full1 = LCB('other_unconstrained', pp, param_mode='elements')
    full2 = LCB('other_unconstrained', pp, param_mode='elements')
    assert full1.is_similar(full2)
    assert not full1.is_similar(LCB('other_unconstrained', pp, param_mode='static'))   # param mode differs
    red = LCB('other_unconstrained', pp, error_generator_labels=[LEEL('S', ('X',))], param_mode='elements')
    assert not full1.is_similar(red)        # different eeg subset
    assert not red.is_similar(full1)
    assert not full1.is_similar(LCB('other', pp, param_mode='elements'))   # legacy 'other' is a different type
    assert not full1.is_similar("not a block")


# ---------------------------------------------------------------- term-object construction

def _std_superop_from_terms(blk, params):
    """Reconstruct the std-basis error-generator superop from create_lindblad_term_objects.
    Each term acts rho -> coeff * A rho B^dag; with row-major vec the superop is kron(A, conj(B))."""
    terms = blk.create_lindblad_term_objects(0, 100, 'statevec', QubitSpace(1))
    d = blk._basis.elements[0].shape[0]
    S = np.zeros((d * d, d * d), complex)
    pt = {i: params[i] for i in range(len(params))}
    for t in terms:
        c = complex(t.coeff.evaluate(pt))
        A = np.eye(d, dtype=complex)
        for op in t._rep.pre_ops:
            A = A @ op.to_dense('Hilbert')
        B = np.eye(d, dtype=complex)
        for op in t._rep.post_ops:
            B = B @ op.to_dense('Hilbert')
        S += c * np.kron(A, np.conj(B))
    return S


# ---------------------------------------------------------------- LindbladErrorgen exposure (GLNDU)

def test_glndu_parameterization_cast():
    from pygsti.modelmembers.operations.lindbladerrorgen import LindbladParameterization
    p = LindbladParameterization.cast('GLNDU')
    assert p.block_types == ('ham', 'other_unconstrained')
    assert p.param_modes == ('elements', 'elements')


def test_from_elementary_errorgens_glndu_matches_glnd():
    """GLNDU (flat 'other_unconstrained') and GLND (Hermitian 'other') describe the same errorgen,
    and GLNDU uses the flat block carrying exactly the requested non-H elementary errorgens."""
    pp = Basis.cast('pp', 4)
    ss = default_space_for_dim(4)
    rng = np.random.default_rng(0)
    eegs = {LEEL('H', ('X',)): rng.standard_normal(),
            LEEL('S', ('X',)): rng.standard_normal(),
            LEEL('C', ('X', 'Y')): rng.standard_normal(),
            LEEL('A', ('X', 'Y')): rng.standard_normal(),
            LEEL('S', ('Z',)): rng.standard_normal()}
    egU = LindbladErrorgen.from_elementary_errorgens(dict(eegs), parameterization='GLNDU',
                                                     mx_basis=pp, state_space=ss)
    egG = LindbladErrorgen.from_elementary_errorgens(dict(eegs), parameterization='GLND',
                                                     mx_basis=pp, state_space=ss)
    assert np.allclose(egU.to_dense(), egG.to_dense(), atol=1e-9)
    ublocks = [b for b in egU.coefficient_blocks if isinstance(b, _OtherUnconstrainedCoeffBlock)]
    assert len(ublocks) == 1
    # reduced: exactly the requested non-Hamiltonian eegs, no full S/C/A fill-in
    assert set(ublocks[0]._eeg_labels) == {LEEL('S', ('X',)), LEEL('C', ('X', 'Y')),
                                           LEEL('A', ('X', 'Y')), LEEL('S', ('Z',))}


def test_from_error_generator_glndu_roundtrip():
    pp = Basis.cast('pp', 4)
    ss = default_space_for_dim(4)
    rng = np.random.default_rng(2)
    eegs = {LEEL('H', ('Y',)): rng.standard_normal(),
            LEEL('S', ('X',)): rng.standard_normal(),
            LEEL('C', ('X', 'Z')): rng.standard_normal()}
    errgen = LindbladErrorgen.from_elementary_errorgens(dict(eegs), parameterization='GLND',
                                                        mx_basis=pp, state_space=ss).to_dense()
    eg2 = LindbladErrorgen.from_error_generator(errgen, parameterization='GLNDU',
                                                mx_basis=pp, state_space=ss)
    assert any(isinstance(b, _OtherUnconstrainedCoeffBlock) for b in eg2.coefficient_blocks)
    assert np.allclose(eg2.to_dense(), errgen, atol=1e-8)


def test_term_objects_reconstruct_dense_errorgen_1q():
    """The term-simulator objects (statevec evotype) sum to the same error generator as the dense
    (densitymx) path.  Guards _create_lindblad_term_objects_impl."""
    pp = Basis.cast('pp', 4)
    blk = LCB('other_unconstrained', pp, param_mode='elements')
    rng = np.random.default_rng(5)
    coeffs = {lbl: rng.standard_normal() for lbl in blk.elementary_errorgens.keys()}
    blk.set_elementary_errorgens(dict(coeffs))
    v = blk.to_vector().copy()
    S = _std_superop_from_terms(blk, v)
    dense_std = _bt.change_basis(_errorgen_dense(blk), 'pp', 'std')
    assert np.allclose(S, dense_std, atol=1e-9)
    # sanity: every monomial references only this block's own parameters
    terms = blk.create_lindblad_term_objects(0, 100, 'statevec', QubitSpace(1))
    var_indices = {int(i) for t in terms for k in t.coeff.coeffs for i in k}
    assert all(0 <= i < blk.num_params for i in var_indices)
