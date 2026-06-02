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
import hashlib

import numpy as np
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


def _cases(include_other_big):
    out = []
    for bname, dim in BASES:
        for bt, pms in VALID.items():
            if bt == 'other' and dim >= 16 and not include_other_big:
                continue  # keep large 'other' (n*n params) out of the heavy finite-diff tests
            for pm in pms:
                out.append((bname, dim, bt, pm))
    return out


ALL_CASES = _cases(include_other_big=True)
FD_CASES = [c for c in _cases(include_other_big=False) if c[3] != 'static']
HESS_CASES = [c for c in FD_CASES if not (c[2] == 'other' and c[1] > 4)]


def _ident(c):
    return "%s%d-%s-%s" % c


def make_block(bname, dim, bt, pm, data_seed=None):
    basis = Basis.cast(bname, dim)
    blk = LCB(bt, basis, param_mode=pm)
    if data_seed is not None and blk.num_params > 0:
        rng = np.random.default_rng(data_seed)
        if bt == 'other' and pm == 'cholesky':
            # Use well-conditioned PD block_data so to_vector's eigh+cholesky is numerically
            # stable (random Cholesky params can produce a near-singular C C^dag, whose params
            # are ill-conditioned).  The near-singular path is exercised separately by
            # test_other_cholesky_truncation_near_singular.
            n = len(blk.basis_element_labels)
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            blk.block_data[:, :] = A @ A.conj().T + n * np.eye(n)
            blk._coefficients_need_update = True
        else:
            blk.from_vector(rng.standard_normal(blk.num_params))
    return blk


def block_data_flat_of(bname, dim, bt, pm, v):
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    blk.from_vector(np.asarray(v, float))
    return blk.block_data.ravel().copy()


def Gflat(blk, bname):
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


@pytest.mark.parametrize("c", ALL_CASES, ids=_ident)
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


@pytest.mark.parametrize("c", ALL_CASES, ids=_ident)
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


@pytest.mark.parametrize("c", FD_CASES, ids=_ident)
def test_deriv_wrt_params_finite_diff(c):
    bname, dim, bt, pm = c
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    v = np.random.default_rng(7).standard_normal(blk.num_params)
    analytic = np.asarray(blk.deriv_wrt_params(v)).reshape(-1, blk.num_params)
    fd = fd_jac(lambda vv: block_data_flat_of(bname, dim, bt, pm, vv), v)
    assert np.allclose(np.real(fd), np.real(analytic), atol=1e-5, rtol=1e-4)
    assert np.allclose(np.imag(fd), np.imag(analytic), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("c", FD_CASES, ids=_ident)
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


@pytest.mark.parametrize("c", HESS_CASES, ids=_ident)
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


@pytest.mark.parametrize("c", ALL_CASES, ids=_ident)
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


@pytest.mark.parametrize("c", ALL_CASES, ids=_ident)
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


@pytest.mark.parametrize("c", ALL_CASES, ids=_ident)
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
    assert not a.is_similar(LCB('ham', pp, param_mode='elements'))              # block_type differs
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


# Snapshot of the Term-simulator polynomial coefficients (1-qubit pp, fixed block_data below),
# generated from the pre-refactor code.  Guards the block-type split's move of
# _create_lindblad_term_objects_* (count + a hash of all terms' Polynomial.coeffs).
_TERM_SNAPSHOT_1Q = {
    ('ham', 'static'):            (6,  'bb697653e167e1ea'),
    ('ham', 'elements'):          (6,  '167e8530daa8c3c2'),
    ('other_diagonal', 'static'): (9,  '8f4ecb17fd64a16c'),
    ('other_diagonal', 'elements'):  (9, 'e1e341741e994e87'),
    ('other_diagonal', 'cholesky'):  (9, '93b4c49fe6469e64'),
    ('other_diagonal', 'depol'):     (9, '26594d67358c52d7'),
    ('other_diagonal', 'reldepol'):  (9, '28850a5764660304'),
    ('other', 'static'):          (27, '0f434ddab05c2818'),
    ('other', 'elements'):        (27, '709cf132c89e1fb0'),
    ('other', 'cholesky'):        (27, '742ea93b1f30fccb'),
}


def _term_fingerprint(terms):
    sigs = []
    for t in terms:
        sig = tuple(sorted((tuple(int(i) for i in k), round(complex(v).real, 12), round(complex(v).imag, 12))
                           for k, v in t.coeff.coeffs.items()))
        sigs.append(sig)
    sigs.sort()
    return hashlib.sha1(repr(sigs).encode()).hexdigest()[:16]


@pytest.mark.parametrize("bt,pm", list(_TERM_SNAPSHOT_1Q.keys()),
                         ids=["%s-%s" % k for k in _TERM_SNAPSHOT_1Q])
def test_create_lindblad_term_objects_snapshot_1q(bt, pm):
    """Pin the Term-simulator polynomial coefficients (counts + hash) to guard the block-type split."""
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
    expected_count, expected_hash = _TERM_SNAPSHOT_1Q[(bt, pm)]
    assert len(terms) == expected_count
    assert _term_fingerprint(terms) == expected_hash


def test_static_deriv_shapes():
    pp = Basis.cast('pp', 4)
    for bt, coeff_shape in [('ham', (3,)), ('other_diagonal', (3,)), ('other', (3, 3))]:
        blk = LCB(bt, pp, param_mode='static')
        assert blk.num_params == 0
        assert blk.deriv_wrt_params().shape == coeff_shape + (0,)
        G = Gflat(blk, 'pp')
        assert blk.superop_deriv_wrt_params(G, np.empty(0), superops_are_flat=True).shape[-1] == 0
