"""
Unit tests for Torchable API implementations of FullArbitraryOp, EmbeddedOp (+EmbeddedErrorgen),
ComputationalBasisPOVM, ComposedPOVMEffect, RootConjOperator, Instrument, TPInstrument (and its
TPInstrumentOp members).

For each member we check that

  (1) ``type(obj).torch_base(obj.stateless_data(float64, 'cpu'), from_numpy(obj.to_vector()))``
      reproduces the numpy ``obj.to_dense('HilbertSchmidt')`` (or, for POVMs/instruments, the stacked
      per-effect / per-member denses); and
  (2) ``torch.func.jacrev(torch_base)`` reproduces the numpy Jacobian -- ``deriv_wrt_params`` where
      an analytic one exists, or a finite-difference reference for RootConjOperator.

These are skipped if torch is unavailable.
"""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygsti.baseobjs.statespace import QubitSpace
from pygsti.modelpacks import smq1Q_XYI
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.modelmembers.operations.fullarbitraryop import FullArbitraryOp
from pygsti.modelmembers.operations.fulltpop import FullTPOp
from pygsti.modelmembers.operations.embeddedop import EmbeddedOp
from pygsti.modelmembers.operations.embeddederrorgen import EmbeddedErrorgen
from pygsti.modelmembers.operations.experrorgenop import ExpErrorgenOp
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen
from pygsti.modelmembers.operations import RootConjOperator
from pygsti.modelmembers.povms.computationalpovm import ComputationalBasisPOVM
from pygsti.modelmembers.povms.composedeffect import ComposedPOVMEffect
from pygsti.modelmembers.instruments import Instrument, TPInstrument


@pytest.fixture(autouse=True)
def _float64_default():
    """Run these correctness checks in double precision, then restore the global default."""
    stashed = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(stashed)


def _value(obj):
    sd = obj.stateless_data(torch.float64, 'cpu')
    t = torch.from_numpy(np.ascontiguousarray(obj.to_vector()))
    return type(obj).torch_base(sd, t).detach().numpy()


def _jac(obj, out_size):
    sd = obj.stateless_data(torch.float64, 'cpu')
    t = torch.from_numpy(np.ascontiguousarray(obj.to_vector()))
    J = torch.func.jacrev(lambda tp: type(obj).torch_base(sd, tp))(t)
    return J.reshape(out_size, -1).detach().numpy()


def _glnd_errorgen(seed, scale=0.03):
    eg = LindbladErrorgen.from_error_generator(np.zeros((4, 4)), parameterization='GLND', mx_basis='pp')
    eg.from_vector(scale * np.random.default_rng(seed).standard_normal(eg.num_params))
    return eg


# --------------------------------------------------------------------------------------------------
# FullArbitraryOp
# --------------------------------------------------------------------------------------------------

def test_fullarbitraryop_torch_base():
    rng = np.random.default_rng(0)
    op = FullArbitraryOp(np.eye(4) + 0.1 * rng.standard_normal((4, 4)), basis='pp')
    ref = op.to_dense('HilbertSchmidt')
    assert np.allclose(_value(op), ref, atol=1e-12)
    assert np.allclose(_jac(op, ref.size), op.deriv_wrt_params(), atol=1e-10)  # jac == identity


# --------------------------------------------------------------------------------------------------
# EmbeddedOp (+ EmbeddedErrorgen)
# --------------------------------------------------------------------------------------------------

def test_embeddedop_torch_base():
    rng = np.random.default_rng(1)
    ss = QubitSpace(['Q0', 'Q1'])

    # a noisy TP op embedded on Q1
    tp1 = np.eye(4)
    tp1[1:, :] += 0.05 * rng.standard_normal((3, 4))
    emb = EmbeddedOp(ss, ['Q1'], FullTPOp(tp1, basis='pp'))
    ref = emb.to_dense('HilbertSchmidt')
    assert np.allclose(_value(emb), ref, atol=1e-12)
    assert np.allclose(_jac(emb, ref.size), emb.deriv_wrt_params(), atol=1e-10)

    # an ExpErrorgenOp (CPTP gate) embedded on Q0
    emb2 = EmbeddedOp(ss, ['Q0'], ExpErrorgenOp(_glnd_errorgen(2)))
    ref2 = emb2.to_dense('HilbertSchmidt')
    assert np.allclose(_value(emb2), ref2, atol=1e-9)
    assert np.allclose(_jac(emb2, ref2.size), emb2.deriv_wrt_params(), atol=1e-7)


def test_embeddederrorgen_torch_base():
    ss = QubitSpace(['Q0', 'Q1'])
    emb = EmbeddedErrorgen(ss, ['Q0'], _glnd_errorgen(3))
    ref = emb.to_dense('HilbertSchmidt')
    assert np.allclose(_value(emb), ref, atol=1e-9)
    with warnings.catch_warnings():  # EmbeddedErrorgen.deriv_wrt_params emits a (benign) UserWarning
        warnings.simplefilter('ignore')
        dref = emb.deriv_wrt_params()
    assert np.allclose(_jac(emb, ref.size), dref, atol=1e-7)


# --------------------------------------------------------------------------------------------------
# ComputationalBasisPOVM (zero parameters)
# --------------------------------------------------------------------------------------------------

def test_computational_povm_torch_base():
    povm = ComputationalBasisPOVM(2, 'default')
    assert povm.num_params == 0
    rows = np.stack([povm[k].to_dense('HilbertSchmidt') for k in povm.keys()])  # effect-order duals
    assert np.allclose(_value(povm), rows, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# ComposedPOVMEffect
# --------------------------------------------------------------------------------------------------

def _noisy_cptplnd_povm(seed=5):
    m = smq1Q_XYI.target_model()
    m.set_all_parameterizations('CPTPLND')
    rng = np.random.default_rng(seed)
    m.from_vector(m.to_vector() + 0.03 * rng.standard_normal(m.num_params))
    return m.povms['Mdefault']


def test_composed_povm_effect_torch_base():
    povm = _noisy_cptplnd_povm()
    for k in povm.keys():
        eff = povm[k]
        assert isinstance(eff, ComposedPOVMEffect)
        ref = eff.to_dense('HilbertSchmidt')
        assert np.allclose(_value(eff), ref, atol=1e-9)
        assert np.allclose(_jac(eff, ref.size), eff.deriv_wrt_params(), atol=1e-8)


# --------------------------------------------------------------------------------------------------
# RootConjOperator (via Instrument.from_effects with spectrally-interior effects)
# --------------------------------------------------------------------------------------------------

def _interior_effect_rootconj(seed=11):
    """A RootConjOperator whose effect spectrum is interior to (0, 1) and non-degenerate (so the
    eigh-backward Jacobian is well-conditioned and no NumericalDomainWarning fires)."""
    model = std.target_model()
    E0 = np.diag([0.7, 0.3]).astype(complex)
    E1 = np.diag([0.3, 0.7]).astype(complex)
    instr = Instrument.from_effects({'p0': E0, 'p1': E1}, model.basis)
    rng = np.random.default_rng(seed)
    instr.from_vector(instr.to_vector() + 0.02 * rng.standard_normal(instr.num_params))
    rcop = instr['p0'].factorops[0]
    assert isinstance(rcop, RootConjOperator)
    return rcop


def test_rootconj_torch_base_value():
    rcop = _interior_effect_rootconj()
    assert np.allclose(_value(rcop), rcop.to_dense('HilbertSchmidt'), atol=1e-9)


def test_rootconj_torch_jacobian():
    rcop = _interior_effect_rootconj()
    ref = rcop.to_dense('HilbertSchmidt')
    J = _jac(rcop, ref.size)

    # finite-difference reference (RootConjOperator has no analytic deriv_wrt_params)
    v0 = rcop.to_vector()
    eps = 1e-6
    Jfd = np.zeros((ref.size, len(v0)))
    for k in range(len(v0)):
        vp = v0.copy(); vp[k] += eps; rcop.from_vector(vp); fp = rcop.to_dense('HilbertSchmidt').ravel()
        vm = v0.copy(); vm[k] -= eps; rcop.from_vector(vm); fm = rcop.to_dense('HilbertSchmidt').ravel()
        Jfd[:, k] = (fp - fm) / (2 * eps)
    rcop.from_vector(v0)
    assert np.allclose(J, Jfd, atol=1e-5)


def test_rootconj_torch_base_value_boundary_spectrum():
    """
    Consider a RootConjOperator built from *typical* projective effects (spectrum at the 0/1 boundary).
    Perturbing pushes eigenvalues slightly outside [0, 1], so the numpy reference (rootconj_superop)
    emits NumericalDomainWarning that must be suppressed. The torch path clamps silently.
    Since both clip to [0, 1], the value returned by this module's private `_value` helper
    should match the result of `to_dense()`.
    """
    model = std.target_model()
    E0 = np.diag([1.0, 0.0]).astype(complex)
    E1 = np.diag([0.0, 1.0]).astype(complex)
    instr = Instrument.from_effects({'p0': E0, 'p1': E1}, model.basis)
    rcop = instr['p0'].factorops[0]
    assert isinstance(rcop, RootConjOperator)

    # Let eigenvalues stray outside [0, 1] far enough to *warn* without *raising*, so a
    # realistically-sized perturbation stays in the boundary/clamp regime we mean to exercise.
    rcop.EIGTOL_ERROR = 1.0

    rng = np.random.default_rng(19)
    v = rcop.to_vector() + 0.02 * rng.standard_normal(rcop.num_params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # numpy path warns (NumericalDomainWarning) at the boundary
        rcop.from_vector(v)
        expected = rcop.to_dense('HilbertSchmidt')
    actual = _value(rcop)

    assert np.allclose(actual, expected, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# Instrument
# --------------------------------------------------------------------------------------------------

def _projector_superops():
    model = std.target_model()
    E = model.povms['Mdefault']['0'].to_dense().ravel()
    Erem = model.povms['Mdefault']['1'].to_dense().ravel()
    return {'plus': np.outer(E, E), 'minus': np.outer(Erem, Erem)}


def _stacked_denses(instrument):
    return np.stack([instrument[k].to_dense('HilbertSchmidt') for k in instrument.keys()])


def test_instrument_torch_base_fullarb():
    instr = Instrument(_projector_superops())  # FullArbitraryOp members
    rng = np.random.default_rng(12)
    instr.from_vector(instr.to_vector() + 0.02 * rng.standard_normal(instr.num_params))
    assert np.allclose(_value(instr), _stacked_denses(instr), atol=1e-12)


def test_instrument_torch_base_from_effects():
    model = std.target_model()
    E0 = np.diag([0.7, 0.3]).astype(complex)
    E1 = np.diag([0.3, 0.7]).astype(complex)
    instr = Instrument.from_effects({'p0': E0, 'p1': E1}, model.basis)  # ComposedOp members, shared POVM
    rng = np.random.default_rng(13)
    instr.from_vector(instr.to_vector() + 0.015 * rng.standard_normal(instr.num_params))
    assert np.allclose(_value(instr), _stacked_denses(instr), atol=1e-9)


# --------------------------------------------------------------------------------------------------
# TPInstrument + TPInstrumentOp
# --------------------------------------------------------------------------------------------------

def _noisy_tpinstrument(seed=31):
    # 3 outcomes so we exercise both TPInstrumentOp branches: index < n-1 and the last (all-param_ops).
    I4 = np.eye(4)
    tpi = TPInstrument({'a': 0.5 * I4, 'b': 0.3 * I4, 'c': 0.2 * I4})
    rng = np.random.default_rng(seed)
    tpi.from_vector(tpi.to_vector() + 0.02 * rng.standard_normal(tpi.num_params))
    return tpi


def test_tpinstrument_op_torch_base():
    tpi = _noisy_tpinstrument()
    keys = list(tpi.keys())
    n = len(keys)
    # a "middle" member (index < n-1) and the last member (the all-param_ops branch)
    for key in (keys[0], keys[-1]):
        member = tpi[key]
        ref = member.to_dense('HilbertSchmidt')
        assert np.allclose(_value(member), ref, atol=1e-10)
        assert np.allclose(_jac(member, ref.size), member.deriv_wrt_params(), atol=1e-8)
    assert tpi[keys[0]].index < n - 1
    assert tpi[keys[-1]].index == n - 1


def test_tpinstrument_torch_base():
    tpi = _noisy_tpinstrument()
    assert np.allclose(_value(tpi), _stacked_denses(tpi), atol=1e-10)
