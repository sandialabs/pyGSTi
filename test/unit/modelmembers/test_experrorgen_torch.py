"""
Torchable wiring for the Lindblad op chain (GitHub issue 607).

The block-level Torchable plumbing is checked in ``test_lindbladcoefficients_torch.py``.  This file
covers the operation classes that compose it so that ``TorchForwardSimulator`` can run on a CPTPLND/GLND
model: ``LindbladErrorgen`` -> ``ExpErrorgenOp`` -> ``ComposedOp``.  For each we check that

  (1) ``type(obj).torch_base(obj.stateless_data(torch.float32, 'cpu'), torch.from_numpy(obj.to_vector()))`` reproduces the
      numpy ``obj.to_dense('HilbertSchmidt')`` (error generator for the errorgen, ``exp(L)`` process
      matrix for the ops); and
  (2) ``torch.func.jacrev(torch_base)`` reproduces the numpy ``deriv_wrt_params`` -- i.e. PyTorch
      autodiff reproduces the manual Jacobian, which is what ``TorchForwardSimulator`` relies on.

Skipped if torch is unavailable.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygsti.modelpacks import smq1Q_XYI
from pygsti.modelmembers.operations.composedop import ComposedOp
from pygsti.modelmembers.operations.experrorgenop import ExpErrorgenOp
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen

# (parameterization, expected non-Hamiltonian block param_mode) -- GLND is unconstrained ('elements'),
# CPTPLND is the CP-constrained Cholesky parameterization ('cholesky').  Both have >1 coefficient block,
# which exercises the per-block parameter slicing in LindbladErrorgen.torch_base.
MODES = ['GLND', 'CPTPLND']


def _noisy_cptp_model(mode, seed=4):
    """1-qubit model with TP spam (Torchable) and ``mode`` operations, perturbed off the ideal."""
    m = smq1Q_XYI.target_model()
    m.convert_members_inplace('full TP')
    m.convert_members_inplace(mode, categories_to_convert='ops')
    rng = np.random.default_rng(seed)
    m.from_vector(m.to_vector() + 0.03 * rng.standard_normal(m.num_params))
    return m


def _noisy_full_cptp_model(seed=5):
    """1-qubit model with Lindblad (CPTPLND) SPAM *and* operations -- prep is a ComposedState, POVM is a
    ComposedPOVM, gates are ComposedOps -- perturbed off the ideal."""
    m = smq1Q_XYI.target_model()
    m.set_all_parameterizations('CPTPLND')
    rng = np.random.default_rng(seed)
    m.from_vector(m.to_vector() + 0.03 * rng.standard_normal(m.num_params))
    return m


def _a_composed_op(model):
    """A gate ``ComposedOp([static_ideal, ExpErrorgenOp])`` from the model."""
    for op in model.operations.values():
        if isinstance(op, ComposedOp) and len(op.factorops) > 1:
            return op
    raise AssertionError("no ComposedOp gate found")


def _torch_base_np(obj, t_param=None):
    sd = obj.stateless_data(torch.float32, 'cpu')
    if t_param is None:
        t_param = torch.from_numpy(np.ascontiguousarray(obj.to_vector()))
    return type(obj).torch_base(sd, t_param).detach().numpy()


def _jacrev_np(obj):
    sd = obj.stateless_data(torch.float32, 'cpu')
    t_param = torch.from_numpy(np.ascontiguousarray(obj.to_vector()))
    ref = obj.to_dense('HilbertSchmidt')
    J = torch.func.jacrev(lambda tp: type(obj).torch_base(sd, tp))(t_param)
    return J.reshape(ref.size, -1).detach().numpy()


@pytest.mark.parametrize("mode", MODES)
def test_lindbladerrorgen_torch_base(mode):
    """LindbladErrorgen.torch_base reproduces the dense error generator and its analytic Jacobian."""
    op = _a_composed_op(_noisy_cptp_model(mode))
    eg = op.factorops[1].errorgen
    assert isinstance(eg, LindbladErrorgen)

    assert np.allclose(_torch_base_np(eg), eg.to_dense('HilbertSchmidt'), atol=1e-10)
    assert np.allclose(_jacrev_np(eg), eg.deriv_wrt_params(), atol=1e-7)


@pytest.mark.parametrize("mode", MODES)
def test_experrorgenop_torch_base(mode):
    """ExpErrorgenOp.torch_base reproduces exp(L) (vs scipy expm) and its analytic Jacobian."""
    op = _a_composed_op(_noisy_cptp_model(mode))
    exp = op.factorops[1]
    assert isinstance(exp, ExpErrorgenOp)

    assert np.allclose(_torch_base_np(exp), exp.to_dense('HilbertSchmidt'), atol=1e-9)
    assert np.allclose(_jacrev_np(exp), exp.deriv_wrt_params(), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_composedop_torch_base(mode):
    """ComposedOp.torch_base composes its (static prefactor + ExpErrorgenOp) factors correctly."""
    op = _a_composed_op(_noisy_cptp_model(mode))
    # the ideal/target prefactor is a static (0-param) factor handled by StaticTorchable
    assert op.factorops[0].num_params == 0

    assert np.allclose(_torch_base_np(op), op.to_dense('HilbertSchmidt'), atol=1e-9)
    assert np.allclose(_jacrev_np(op), op.deriv_wrt_params(), atol=1e-6)


def test_composedstate_torch_base():
    """ComposedState (Lindblad SPAM prep) torch_base = error_map_superop @ static_ket; reproduces
    to_dense and its analytic Jacobian.  ComposedState.torch_base returns a super-ket (1-D)."""
    rho = _noisy_full_cptp_model().preps['rho0']
    sd = rho.stateless_data(torch.float32, 'cpu')
    t = torch.from_numpy(np.ascontiguousarray(rho.to_vector()))

    assert np.allclose(_torch_base_np(rho), rho.to_dense('HilbertSchmidt'), atol=1e-9)
    J = torch.func.jacrev(lambda tp: type(rho).torch_base(sd, tp))(t).detach().numpy()
    assert np.allclose(J, rho.deriv_wrt_params(), atol=1e-6)


def test_composedpovm_torch_base():
    """ComposedPOVM (Lindblad SPAM measurement) torch_base = base_effect_rows @ error_map_superop;
    reproduces the per-effect dual vectors and their analytic Jacobians.  Rows are in POVM-effect order
    (the row order TorchForwardSimulator multiplies against the super-ket)."""
    povm = _noisy_full_cptp_model().povms['Mdefault']
    sd = povm.stateless_data(torch.float32, 'cpu')
    t = torch.from_numpy(np.ascontiguousarray(povm.to_vector()))

    # value: row i is effect i's dual vector
    eff_rows = np.stack([povm[k].to_dense('HilbertSchmidt') for k in povm.keys()])
    assert np.allclose(type(povm).torch_base(sd, t).detach().numpy(), eff_rows, atol=1e-9)

    # autodiff Jacobian (n_effects, dim, n_params) == stacked per-effect deriv_wrt_params
    J = torch.func.jacrev(lambda tp: type(povm).torch_base(sd, tp))(t).detach().numpy()
    Jref = np.stack([povm[k].deriv_wrt_params() for k in povm.keys()])
    assert J.shape == Jref.shape
    assert np.allclose(J, Jref, atol=1e-6)
