"""
Characterization ("golden") tests for LindbladCoefficientBlock.

These pin the *current* input/output behavior of LindbladCoefficientBlock across the full
(block_type, param_mode) validity matrix, so the planned base+subclasses refactor (GitHub
issue 607) can be verified behavior-preserving.  They are written to pass on the un-refactored
class and must continue to pass *identically* after the refactor.

Relationships pinned here (anchored on block_data, which is unique; params are NOT unique
under cholesky/depol, so param round-trips are checked only via block_data):

  * num_params formula; to_vector/from_vector round-trip (on block_data).
  * deriv_wrt_params(v) == d(block_data)/dv                               (finite differences)
  * superop_deriv_wrt_params == d(superop)/dv                            (finite differences),
        where superop = einsum('i,ijk->jk', block_data.ravel(), G),
        G = create_lindblad_term_superoperators(..., flat=True)
  * the linearity / chain-rule identities (what makes the parameterization the *only*
    param_mode-specific piece -- see issue 607 plan):
        superop_deriv    == einsum('Dp,Djk->jkp',  deriv_wrt_params,        G)
        superop_hessian  == einsum('Dpq,Djk->jkpq', d(deriv_wrt_params)/dv, G)
  * elementary_errorgens values + round-trip; coefficient/param labels; NicelySerializable
    round-trip; convert; is_similar; near-singular 'other'/cholesky truncation.
"""
import numpy as np
from numpy.typing import ArrayLike
import pytest

from pygsti.baseobjs.basis import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB


# ---- validity matrix (block_type -> valid param_modes) ----
VALID = {
    'ham':            ('static', 'elements'),
    'other_diagonal': ('static', 'elements', 'cholesky', 'depol', 'reldepol'),
    'other':          ('static', 'elements', 'cholesky'),
}
# (basis_name, superop_dim).  'pp' requires a power-of-two Hilbert dim; 'gm' dim 9 == qutrit.
BASES = [('pp', 4), ('gm', 9), ('pp', 16)]


def _cases(include_other_big) -> list[tuple[str, int, str, str]]:
    out = []
    for bname, dim in BASES:
        for bt, pms in VALID.items():
            if bt == 'other' and dim >= 16 and not include_other_big:
                continue  # keep large 'other' (n*n params) out of the heavy finite-diff tests
            for pm in pms:
                out.append((bname, dim, bt, pm))
    return out


ALL_CASES  = _cases(include_other_big=True)
FD_CASES   = [ c for c in _cases(include_other_big=False) if c[3] != 'static']
HESS_CASES = [ c for c in FD_CASES if not (c[2] == 'other' and c[1] > 4)     ]


def _config_as_string(c : tuple[str, int, str, str]) -> str:
    return "%s%d-%s-%s" % c


def make_block(bname: str, dim: int, bt: str, pm: str, data_seed=None):
    basis = Basis.cast(bname, dim)
    blk   = LCB(bt, basis, param_mode=pm)
    if data_seed is not None and blk.num_params > 0:
        rng = np.random.default_rng(data_seed)
        if bt == 'other' and pm == 'cholesky':
            # Use well-conditioned PD block_data so to_vector's cholesky is numerically
            # stable. The near-singular path is exercised separately by
            # test_other_cholesky_truncation_near_singular.
            n = len(blk.basis_element_labels)
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            blk.block_data[:, :] = A @ A.conj().T + n * np.eye(n)
            blk._coefficients_need_update = True
        else:
            blk.from_vector(rng.standard_normal(blk.num_params))
    return blk


def block_data_flat_of(bname: str, dim: int, bt: str, pm: str, v: ArrayLike):
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    blk.from_vector(np.asarray(v, float))
    return blk.block_data.ravel().copy()


def Gflat(blk: LCB, bname: str):
    return np.asarray(blk.create_lindblad_term_superoperators(bname, sparse=False,
                                                              include_1norms=False, flat=True))


def fd_jac(f, v, eps=1e-6):
    """Central-difference Jacobian of array-valued f: returns shape f(v).shape + (len(v),)."""
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
    blk = make_block(bname, dim, bt, pm)
    n = len(blk.basis_element_labels)
    if pm == 'static':
        expected = 0
    elif bt == 'ham':
        expected = n
    elif bt == 'other_diagonal':
        expected = 1 if pm in ('depol', 'reldepol') else n
    else:  # 'other'
        expected = n * n
    assert blk.num_params == expected
    assert blk.to_vector().shape == (expected,)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_block_data_param_roundtrip(c):
    bname, dim, bt, pm = c
    blk = make_block(bname, dim, bt, pm, data_seed=1)
    if blk.num_params == 0:
        assert blk.to_vector().shape == (0,)
        return
    bd0 = blk.block_data.copy()
    v = blk.to_vector()
    blk.from_vector(v)
    assert np.allclose(blk.block_data, bd0, atol=1e-9)   # block_data is the unique invariant
    assert np.allclose(blk.to_vector(), v, atol=1e-9)    # to_vector idempotent


@pytest.mark.parametrize("c", FD_CASES, ids=_config_as_string)
def test_deriv_wrt_params_finite_diff(c):
    bname, dim, bt, pm = c
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    v = np.random.default_rng(7).standard_normal(blk.num_params)
    analytic = np.asarray(blk.deriv_wrt_params(v)).reshape(-1, blk.num_params)
    fd = fd_jac(lambda vv: block_data_flat_of(bname, dim, bt, pm, vv), v)
    assert np.allclose(np.real(fd), np.real(analytic), atol=1e-5, rtol=1e-4)
    assert np.allclose(np.imag(fd), np.imag(analytic), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("c", FD_CASES, ids=_config_as_string)
def test_superop_deriv_finite_diff_and_chain_rule(c):
    bname, dim, bt, pm = c
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    nP = blk.num_params
    v = np.random.default_rng(11).standard_normal(nP)
    blk.from_vector(v)   # 'other' block reads its Cholesky factor from _cache_mx (set by from_vector)
    G = Gflat(blk, bname)                                   # (D, d2, d2)
    d2a, d2b = G.shape[1], G.shape[2]

    sd = np.asarray(blk.superop_deriv_wrt_params(G, v, superops_are_flat=True)).reshape(d2a, d2b, nP)

    fd = fd_jac(lambda vv: np.einsum('i,ijk->jk', block_data_flat_of(bname, dim, bt, pm, vv), G), v)
    assert np.allclose(np.imag(fd), 0, atol=1e-6)
    assert np.allclose(np.real(fd), sd, atol=1e-5, rtol=1e-4)

    # linearity / chain rule: superop_deriv == einsum('Dp,Djk->jkp', d(block_data)/dv, G)
    deriv = np.asarray(blk.deriv_wrt_params(v)).reshape(-1, nP)
    chain = np.einsum('Dp,Djk->jkp', deriv, G)
    assert np.allclose(np.imag(chain), 0, atol=1e-7)
    assert np.allclose(np.real(chain), sd, atol=1e-7)


@pytest.mark.parametrize("c", HESS_CASES, ids=_config_as_string)
def test_superop_hessian_chain_rule(c):
    bname, dim, bt, pm = c
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    nP = blk.num_params
    v = np.random.default_rng(13).standard_normal(nP)
    G = Gflat(blk, bname)
    d2a, d2b = G.shape[1], G.shape[2]

    h = np.asarray(blk.superop_hessian_wrt_params(G, v, superops_are_flat=True)).reshape(d2a, d2b, nP, nP)

    def deriv_of(vv):
        return np.asarray(LCB(bt, Basis.cast(bname, dim), param_mode=pm).deriv_wrt_params(vv)).reshape(-1, nP)
    bd_hess = fd_jac(deriv_of, v)                          # (D, nP, nP)
    chain = np.einsum('Dpq,Djk->jkpq', bd_hess, G)
    assert np.allclose(np.imag(chain), 0, atol=1e-5)
    assert np.allclose(np.real(chain), h, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_elementary_errorgens_roundtrip(c):
    bname, dim, bt, pm = c
    blk = make_block(bname, dim, bt, pm, data_seed=3)
    eegs = blk.elementary_errorgens
    blk2 = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    unused = blk2.set_elementary_errorgens(dict(eegs), on_missing='raise')
    assert len(unused) == 0
    assert np.allclose(blk2.block_data, blk.block_data, atol=1e-9)


def test_other_elementary_errorgens_values_1q():
    """Pin the tricky 'other'-block C/A combinations of NH coefficients (1-qubit pp)."""
    blk = LCB('other', Basis.cast('pp', 4), param_mode='elements')
    bels = blk.basis_element_labels
    assert tuple(bels) == ('X', 'Y', 'Z')
    bd = np.zeros((3, 3), complex)
    bd[0, 0] = 0.5                       # S_X
    bd[0, 1] = 0.1 + 0.2j                # NH_XY
    bd[1, 0] = 0.1 - 0.2j                # NH_YX
    blk.block_data[:, :] = bd
    blk._coefficients_need_update = True
    eegs = blk.elementary_errorgens
    # C_XY = 0.5*NH_XY + 0.5*NH_YX = 0.1 ;  A_XY = 0.5j*NH_XY - 0.5j*NH_YX = -0.2
    assert np.isclose(eegs[LEEL('S', ('X',))], 0.5)
    assert np.isclose(eegs[LEEL('C', ('X', 'Y'))], 0.1)
    assert np.isclose(eegs[LEEL('A', ('X', 'Y'))], -0.2)


def test_labels_snapshot_1q():
    pp = Basis.cast('pp', 4)
    assert tuple(LCB('ham', pp).basis_element_labels) == ('X', 'Y', 'Z')

    assert LCB('ham', pp, param_mode='elements').param_labels == \
        ['X Hamiltonian error coefficient', 'Y Hamiltonian error coefficient', 'Z Hamiltonian error coefficient']
    assert LCB('ham', pp).coefficient_labels == \
        ['X Hamiltonian error coefficient', 'Y Hamiltonian error coefficient', 'Z Hamiltonian error coefficient']

    assert LCB('other_diagonal', pp, param_mode='cholesky').param_labels == \
        ['sqrt(X stochastic coefficient)', 'sqrt(Y stochastic coefficient)', 'sqrt(Z stochastic coefficient)']
    assert LCB('other_diagonal', pp, param_mode='elements').param_labels == \
        ['X stochastic coefficient', 'Y stochastic coefficient', 'Z stochastic coefficient']
    assert LCB('other_diagonal', pp, param_mode='depol').param_labels == \
        ['sqrt(common stochastic error coefficient for depolarization)']
    assert LCB('other_diagonal', pp, param_mode='reldepol').param_labels == \
        ['common stochastic error coefficient for depolarization']
    assert LCB('other_diagonal', pp).coefficient_labels == \
        ['(X,X) other error coefficient', '(Y,Y) other error coefficient', '(Z,Z) other error coefficient']

    # 'static' blocks have no parameters
    assert LCB('ham', pp, param_mode='static').param_labels == []


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_serialization_roundtrip(c):
    bname, dim, bt, pm = c
    blk = make_block(bname, dim, bt, pm, data_seed=5)
    blk2 = LCB.from_nice_serialization(blk.to_nice_serialization())
    assert blk2._block_type == blk._block_type
    assert blk2._param_mode == blk._param_mode
    assert tuple(blk2.basis_element_labels) == tuple(blk.basis_element_labels)
    assert blk2.num_params == blk.num_params
    assert np.allclose(blk2.block_data, blk.block_data, atol=1e-12)
    if blk.num_params:
        assert np.allclose(blk2.to_vector(), blk.to_vector(), atol=1e-12)


@pytest.mark.parametrize("c", ALL_CASES, ids=_config_as_string)
def test_convert_preserves_block_data(c):
    bname, dim, bt, pm = c
    blk = make_block(bname, dim, bt, pm, data_seed=9)
    for tm in VALID[bt]:
        blk_c = blk.convert(tm)              # convert does NOT re-validate feasibility (current behavior)
        assert blk_c._block_type == bt
        assert blk_c._param_mode == tm
        assert np.allclose(blk_c.block_data, blk.block_data, atol=1e-12)


def test_is_similar():
    pp = Basis.cast('pp', 4)
    a = LCB('other_diagonal', pp, param_mode='cholesky')
    assert a.is_similar(LCB('other_diagonal', pp, param_mode='cholesky'))
    assert not a.is_similar(LCB('other_diagonal', pp, param_mode='elements'))   # param_mode differs
    assert not a.is_similar(LCB('ham', pp, param_mode='cholesky'))              # block_type differs
    assert not a.is_similar(LCB('other', pp, param_mode='cholesky'))
    assert not a.is_similar("not a block")


def test_other_cholesky_truncation_near_singular():
    """Exercise the eigh/cholesky truncation path on a rank-deficient PSD 'other' matrix."""
    pp = Basis.cast('pp', 4)
    n = 3
    rng = np.random.default_rng(2)
    C = np.tril(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    C[:, 0] = 0.0                          # drop a column -> rank-deficient (a zero eigenvalue)
    bd = (C @ C.conj().T).astype(complex)
    blk = LCB('other', pp, initial_block_data=bd, param_mode='cholesky', truncate=True)
    assert np.allclose(blk.block_data, bd, atol=1e-6)


def test_static_deriv_shapes():
    pp = Basis.cast('pp', 4)
    for bt, coeff_shape in [('ham', (3,)), ('other_diagonal', (3,)), ('other', (3, 3))]:
        blk = LCB(bt, pp, param_mode='static')
        assert blk.num_params == 0
        assert blk.deriv_wrt_params().shape == coeff_shape + (0,)
        G = Gflat(blk, 'pp')
        assert blk.superop_deriv_wrt_params(G, np.empty(0), superops_are_flat=True).shape[-1] == 0


# Snapshot of the Term-simulator polynomial coefficients (1-qubit pp, fixed block_data below),
# guarding the block-type split's move of _create_lindblad_term_objects_*.  We pin each term's
# coefficient *evaluated at a fixed generic parameter point* (see _evaluated_term_coeffs) rather than
# a hash of its serialized form: polynomial values are platform-independent (deterministic IEEE
# arithmetic, identical on Linux/Windows to ~1e-15), whereas the serialized monomial keys, float
# repr, and +0.0/-0.0 signs are not -- which is what made the earlier hashed snapshot fail on Windows.
# Each entry is (term count, sorted multiset of evaluated coefficients); compared with a tolerance.
_TERM_SNAPSHOT_1Q = {
    ('ham', 'static'): (6, (-0.212132034j, -0.141421356j, -0.070710678j, 0.070710678j, 0.141421356j, 0.212132034j)),
    ('ham', 'elements'): (6, (-0.494974747j, -0.403050865j, -0.311126984j, 0.311126984j, 0.403050865j, 0.494974747j)),
    ('other_diagonal', 'static'): (9, (-0.075, -0.075, -0.05, -0.05, -0.025, -0.025, 0.05, 0.1, 0.15)),
    ('other_diagonal', 'elements'): (9, (-0.175, -0.175, -0.1425, -0.1425, -0.11, -0.11, 0.22, 0.285, 0.35)),
    ('other_diagonal', 'cholesky'): (9, (-0.1225, -0.1225, -0.081225, -0.081225, -0.0484, -0.0484, 0.0968, 0.16245, 0.245)),
    ('other_diagonal', 'depol'): (9, (-0.1225, -0.1225, -0.1225, -0.1225, -0.1225, -0.1225, 0.245, 0.245, 0.245)),
    ('other_diagonal', 'reldepol'): (9, (-0.175, -0.175, -0.175, -0.175, -0.175, -0.175, 0.35, 0.35, 0.35)),
    ('other', 'static'): (27, (-48.5, -48.5, -30.5, -30.5, -30.5, -30.5, -19.25, -19.25, -12.5, -12.5, -12.5, -12.5, -8.0, -8.0, -8.0, -8.0, -3.5, -3.5, 7.0, 16.0, 16.0, 25.0, 25.0, 38.5, 61.0, 61.0, 97.0)),
    ('other', 'elements'): (27, ((-0.1425-0.0775j), (-0.1425-0.0775j), (-0.11+0.02j), (-0.11+0.02j), (-0.105+0.025j), (-0.0775-0.1425j), (-0.0775-0.1425j), (-0.04+0.22j), (-0.0125+0.0525j), (-0.0125+0.0525j), -0.175j, -0.175j, -0.17j, -0.045j, -0.045j, 0.085j, 0.085j, 0.09j, 0.35j, (0.02-0.11j), (0.02-0.11j), (0.025-0.105j), (0.0525-0.0125j), (0.0525-0.0125j), (0.155+0.285j), (0.22-0.04j), (0.285+0.155j))),
    ('other', 'cholesky'): (27, (-0.1225, -0.1225, (-0.11335+0.044175j), (-0.11335+0.044175j), (-0.09055-0.006175j), (-0.09055-0.006175j), (-0.05425-0.09975j), (-0.05425-0.09975j), (-0.05425+0.09975j), (-0.05425+0.09975j), (-0.04705-0.04775j), (-0.04705-0.04775j), (-0.04705+0.04775j), (-0.04705+0.04775j), (-0.028-0.154j), (-0.028+0.154j), (0.014-0.077j), (0.014-0.077j), (0.014+0.077j), (0.014+0.077j), (0.0941-0.0955j), (0.0941+0.0955j), (0.1085-0.1995j), (0.1085+0.1995j), (0.1811+0.01235j), (0.2267-0.08835j), 0.245)),
}
_TERM_SNAPSHOT_CONFIGS = list(_TERM_SNAPSHOT_1Q.keys())


def _term_eval_point(num_params):
    """A fixed, generic point for this block's polynomial variables (indices 0 .. num_params-1)."""
    return {i: 0.7 - 0.13 * i for i in range(num_params)}


def _evaluated_term_coeffs(terms, num_params):
    """Sorted multiset of each term's Polynomial coefficient evaluated at ``_term_eval_point``.

    Comparing *evaluated* polynomials sidesteps the platform-dependent serialization of the
    coefficients (packed monomial keys, float ``repr``, +0.0 vs -0.0): the values are deterministic
    IEEE arithmetic, identical across Linux/Windows to ~1e-15, so a tolerance comparison guards the
    construction while being immune to those representation differences.  Sorting makes the
    comparison a multiset check (insensitive to any benign reordering of the term list).
    """
    pt = _term_eval_point(num_params)
    vals = [complex(t.coeff.evaluate(pt)) for t in terms]
    return sorted(vals, key=lambda z: (round(z.real, 9), round(z.imag, 9)))


@pytest.mark.parametrize("bt,pm", _TERM_SNAPSHOT_CONFIGS, ids=["%s-%s" % k for k in _TERM_SNAPSHOT_1Q])
def test_create_lindblad_term_objects_snapshot_1q(bt, pm):
    """Pin the Term-simulator polynomial coefficients (count + values at a fixed evaluation point)
    to guard the block-type split's move of _create_lindblad_term_objects_*."""
    from pygsti.baseobjs.statespace import QubitSpace
    pp = Basis.cast('pp', 4)
    blk = LCB(bt, pp, param_mode=pm)
    if bt == 'other':
        A = np.arange(1, 10, dtype=float).reshape(3, 3)
        blk.block_data[:, :] = (A @ A.T).astype(complex)        # fixed PD real matrix
    else:
        blk.block_data[:] = np.array([0.1, 0.2, 0.3])[:len(blk.basis_element_labels)]
    blk._coefficients_need_update = True
    terms = blk.create_lindblad_term_objects(0, 100, 'statevec', QubitSpace(1))
    expected_count, expected_coeffs = _TERM_SNAPSHOT_1Q[(bt, pm)]
    assert len(terms) == expected_count
    actual = _evaluated_term_coeffs(terms, blk.num_params)
    assert np.allclose(actual, np.asarray(expected_coeffs), atol=1e-7, rtol=1e-7)
    # sanity: every monomial references only this block's own parameters (0 .. num_params-1)
    var_indices = {int(i) for t in terms for k in t.coeff.coeffs for i in k}
    assert all(0 <= i < blk.num_params for i in var_indices)
