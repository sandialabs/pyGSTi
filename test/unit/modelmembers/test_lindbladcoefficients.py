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


def test_static_deriv_shapes():
    pp = Basis.cast('pp', 4)
    for bt, coeff_shape in [('ham', (3,)), ('other_diagonal', (3,)), ('other', (3, 3))]:
        blk = LCB(bt, pp, param_mode='static')
        assert blk.num_params == 0
        assert blk.deriv_wrt_params().shape == coeff_shape + (0,)
        G = Gflat(blk, 'pp')
        assert blk.superop_deriv_wrt_params(G, np.empty(0), superops_are_flat=True).shape[-1] == 0


# Snapshot of the Term-simulator polynomial coefficients (1-qubit pp, fixed block_data below).
# Guards the block-type split's move of _create_lindblad_term_objects_* (term count + a
# platform-independent fingerprint of every term's Polynomial.coeffs).  Values are regenerated
# from the current, golden-validated code; the fingerprint is integer-quantized (see
# _term_fingerprint), so they are identical on Linux and Windows.
_TERM_SNAPSHOT_1Q = {
    ('ham', 'static'):            (6,  '51b66691aec58f41'),
    ('ham', 'elements'):          (6,  '4678afcfae6f2f46'),
    ('other_diagonal', 'static'): (9,  'a93ab7917787cbf5'),
    ('other_diagonal', 'elements'):  (9, '740224cf5de53b06'),
    ('other_diagonal', 'cholesky'):  (9, '420d40c4ef90f283'),
    ('other_diagonal', 'depol'):     (9, 'de25c201ccac84b6'),
    ('other_diagonal', 'reldepol'):  (9, '519adfe0c63611cd'),
    ('other', 'static'):          (27, 'ed030cf097f2f2bf'),
    ('other', 'elements'):        (27, 'f1b5685854dc434c'),
    ('other', 'cholesky'):        (27, '358c013b0ac41bb0'),
}
_TERM_SNAPSHOT_CONFIGS = list(_TERM_SNAPSHOT_1Q.keys())


# Quantize coefficients to this resolution before hashing.  Collapses cross-platform float
# artifacts -- +0.0 vs -0.0 and last-ULP wobble in math-library output -- to identical integers,
# making the fingerprint reproducible across Linux/Windows.  (Monomial keys are already small
# non-negative ints that pack identically on 32- and 64-bit platforms.)
_TERM_FP_SCALE = 10 ** 9


def _term_fingerprint(terms):
    """Platform-independent fingerprint of a list of polynomial terms.

    Each term's ``Polynomial.coeffs`` becomes a sorted tuple of ``(monomial, re, im)`` triples:
    ``monomial`` is the tuple of variable indices and ``re``/``im`` are the coefficient's real and
    imaginary parts quantized to ``_TERM_FP_SCALE``.  Numerically-zero monomials are dropped and the
    all-integer structure is hashed, so the result is independent of float ``repr`` -- the source of
    the earlier Linux-vs-Windows discrepancy.
    """
    sigs = []
    for t in terms:
        sig = []
        for k, v in t.coeff.coeffs.items():
            c = complex(v)
            re = int(round(c.real * _TERM_FP_SCALE))
            im = int(round(c.imag * _TERM_FP_SCALE))
            if re == 0 and im == 0:
                continue
            sig.append((tuple(int(i) for i in k), re, im))
        sig.sort()
        sigs.append(tuple(sig))
    sigs.sort()
    return hashlib.sha1(repr(sigs).encode()).hexdigest()[:16]


@pytest.mark.parametrize("bt,pm", _TERM_SNAPSHOT_CONFIGS, ids=["%s-%s" % k for k in _TERM_SNAPSHOT_1Q])
def test_create_lindblad_term_objects_snapshot_1q(bt, pm):
    """Pin the Term-simulator polynomial coefficients (count + platform-independent fingerprint)
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
    expected_count, expected_hash = _TERM_SNAPSHOT_1Q[(bt, pm)]
    assert len(terms) == expected_count
    assert _term_fingerprint(terms) == expected_hash
    # sanity: every monomial references only this block's own parameters (0 .. num_params-1)
    var_indices = {int(i) for t in terms for k in t.coeff.coeffs for i in k}
    assert all(0 <= i < blk.num_params for i in var_indices)


# =====================================================================================================
# Coverage-directed tests for the refactored module (Approach A).  These exercise behavior the golden
# matrix above did not reach: the elementary-errorgen Jacobian, set_elementary_errorgens' on_missing
# handling, 'other'-block labels / properties, the matrix-structured (non-flat) superop derivative, and
# the fail-fast block_type / param_mode validation introduced by the composition refactor.
# =====================================================================================================

@pytest.mark.parametrize("case", FD_CASES, ids=_config_as_string)
def test_elementary_errorgen_deriv_wrt_params_fd(case):
    """elementary_errorgen_deriv_wrt_params(v) == d(elementary_errorgens)/dv  (central finite diff)."""
    blk = make_block(*case, data_seed=2024)
    v0 = blk.to_vector().copy()  # copy: for 'elements' to_vector() returns a *view* of block_data,
    blk.from_vector(v0)          # which from_vector would then mutate underneath us during the FD loop
    lbls = list(blk.elementary_errorgens.keys())
    analytic = blk.elementary_errorgen_deriv_wrt_params(v0)
    assert analytic.shape == (len(lbls), blk.num_params)

    eps = 1e-6
    fd = np.zeros((len(lbls), blk.num_params))
    for p in range(blk.num_params):
        vp = v0.copy(); vp[p] += eps
        blk.from_vector(vp); plus = dict(blk.elementary_errorgens)
        vm = v0.copy(); vm[p] -= eps
        blk.from_vector(vm); minus = dict(blk.elementary_errorgens)
        fd[:, p] = [(plus[lbl] - minus[lbl]) / (2 * eps) for lbl in lbls]
    blk.from_vector(v0)  # restore
    assert np.allclose(analytic, fd, atol=1e-6, rtol=1e-5)


def test_set_elementary_errorgens_on_missing():
    """on_missing in {'ignore','warn','raise'} controls handling of absent elementary errorgens."""
    blk = LCB('ham', Basis.cast('pp', 4), param_mode='elements')
    full = dict(blk.elementary_errorgens)
    dropped = next(iter(full))
    partial = {k: v for k, v in full.items() if k != dropped}

    with pytest.raises(ValueError):
        blk.set_elementary_errorgens(partial, on_missing='raise')
    with pytest.warns(UserWarning):
        blk.set_elementary_errorgens(partial, on_missing='warn')
    # default 'ignore' assumes 0 for the missing entry -- no error or warning
    unused = blk.set_elementary_errorgens(partial)
    assert isinstance(unused, dict)


def test_coefficient_labels_block_type_and_caches():
    """coefficient_labels (incl. the n*n 'other' case), the block_type property, the static
    from_vector no-op, the cached elementary_errorgen_indices return, and convert()."""
    pp = Basis.cast('pp', 4)
    for bt, n_labels in [('ham', 3), ('other_diagonal', 3), ('other', 9)]:
        blk = LCB(bt, pp, param_mode='static')
        assert blk.block_type == bt
        labels = blk.coefficient_labels
        assert len(labels) == n_labels and all(isinstance(s, str) for s in labels)

    LCB('ham', pp, param_mode='static').from_vector(np.empty(0))  # static carries no params -> no-op

    blk = LCB('other_diagonal', pp, param_mode='elements')
    _ = blk.elementary_errorgens                                  # populate index cache + clear flag
    assert blk.elementary_errorgen_indices is blk.elementary_errorgen_indices  # cached return
    converted = blk.convert('cholesky')
    assert converted.block_type == 'other_diagonal' and converted._param_mode == 'cholesky'


def test_invalid_block_type_raises():
    from pygsti.modelmembers.operations.lindbladcoefficients import InvalidBlockTypeError
    with pytest.raises(InvalidBlockTypeError):
        LCB('not_a_block_type', Basis.cast('pp', 4))


def test_invalid_param_mode_raises_eagerly():
    """Approach A validates (block_type, param_mode) at construction time (fail-fast)."""
    from pygsti.modelmembers.operations.lindbladcoefficients import InvalidParamModeError
    pp = Basis.cast('pp', 4)
    with pytest.raises(InvalidParamModeError):
        LCB('ham', pp, param_mode='cholesky')           # 'cholesky' invalid for 'ham'
    with pytest.raises(InvalidParamModeError):
        LCB('other_diagonal', pp, param_mode='not_a_mode')


@pytest.mark.parametrize("case", [c for c in FD_CASES if c[2] == 'other'], ids=_config_as_string)
def test_superop_deriv_other_matrix_structured(case):
    """The matrix-structured (superops_are_flat=False) path of superop_deriv_wrt_params for the
    'other' block agrees with the flat path (Approach A's generic chain-rule implementation)."""
    bname, dim, bt, pm = case
    blk = make_block(*case, data_seed=7)
    v = blk.to_vector()
    G_struct = np.asarray(blk.create_lindblad_term_superoperators(bname, sparse=False, flat=False))
    G_flat = np.asarray(blk.create_lindblad_term_superoperators(bname, sparse=False, flat=True))
    d_struct = blk.superop_deriv_wrt_params(G_struct, v, superops_are_flat=False)  # (d2, d2, n, n)
    d_flat = blk.superop_deriv_wrt_params(G_flat, v, superops_are_flat=True)        # (d2, d2, n*n)
    assert np.allclose(d_struct.reshape(d_flat.shape), d_flat, atol=1e-9)
