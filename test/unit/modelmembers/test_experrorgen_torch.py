"""
Test that higher-level classes used in exponentiated-Lindblad parameterizations
correctly implement the Torchable API. Tests for lower-level classes can be found
in test_lindbladcoefficients_torch.py.

For the composition ``LindbladErrorgen`` -> ``ExpErrorgenOp`` -> ``ComposedOp``, we check that

  (1) ``type(obj).torch_base(obj.stateless_data(torch.float64, 'cpu'), torch.from_numpy(obj.to_vector()))`` reproduces the
      numpy ``obj.to_dense('HilbertSchmidt')`` (error generator for the errorgen, ``exp(L)`` process
      matrix for the ops); and
  (2) ``torch.func.jacrev(torch_base)`` reproduces the numpy ``deriv_wrt_params`` -- i.e. PyTorch
      autodiff reproduces the manual Jacobian, which is what ``TorchForwardSimulator`` relies on.

Skipped if torch is unavailable.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygsti.baseobjs import QubitSpace
from pygsti.modelpacks import smq1Q_XYI
from pygsti.modelmembers.operations.composedop import ComposedOp
from pygsti.modelmembers.operations.embeddederrorgen import EmbeddedErrorgen
from pygsti.modelmembers.operations.experrorgenop import ExpErrorgenOp
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen
from .torchable_test_utils import torch_base_value, torch_base_jacobian, noisy_full_cptplnd_model

# (parameterization, expected non-Hamiltonian block param_mode) -- GLND is unconstrained ('elements'),
# CPTPLND is the CP-constrained Cholesky parameterization ('cholesky'), and GLNDU is the flat,
# per-elementary-errorgen 'other_unconstrained' block ('elements').  All have >1 coefficient block,
# which exercises the per-block parameter slicing in LindbladErrorgen.torch_base.
MODES = ['GLND', 'CPTPLND', 'GLNDU']


def _noisy_cptp_model(mode, seed=4):
    """1-qubit model with TP spam (Torchable) and ``mode`` operations, perturbed off the ideal."""
    m = smq1Q_XYI.target_model()
    m.convert_members_inplace('full TP')
    m.convert_members_inplace(mode, categories_to_convert='ops')
    rng = np.random.default_rng(seed)
    m.from_vector(m.to_vector() + 0.03 * rng.standard_normal(m.num_params))
    return m


def _a_composed_op(model):
    """A gate ``ComposedOp([static_ideal, ExpErrorgenOp])`` from the model."""
    for op in model.operations.values():
        if isinstance(op, ComposedOp) and len(op.factorops) > 1:
            return op
    raise AssertionError("no ComposedOp gate found")


@pytest.mark.parametrize("mode", MODES)
def test_lindbladerrorgen_torch_base(mode):
    """LindbladErrorgen.torch_base reproduces the dense error generator and its analytic Jacobian."""
    op = _a_composed_op(_noisy_cptp_model(mode))
    eg = op.factorops[1].errorgen
    assert isinstance(eg, LindbladErrorgen)

    assert np.allclose(torch_base_value(eg), eg.to_dense('HilbertSchmidt'), atol=1e-10)
    assert np.allclose(torch_base_jacobian(eg), eg.deriv_wrt_params(), atol=1e-7)


@pytest.mark.parametrize("mode", MODES)
def test_experrorgenop_torch_base(mode):
    """ExpErrorgenOp.torch_base reproduces exp(L) (vs scipy expm) and its analytic Jacobian."""
    op = _a_composed_op(_noisy_cptp_model(mode))
    exp = op.factorops[1]
    assert isinstance(exp, ExpErrorgenOp)

    assert np.allclose(torch_base_value(exp), exp.to_dense('HilbertSchmidt'), atol=1e-9)
    assert np.allclose(torch_base_jacobian(exp), exp.deriv_wrt_params(), atol=1e-6)


def test_experrorgenop_embedded_errorgen_torch_base():
    """ExpErrorgenOp.torch_base must dispatch on the wrapped errorgen's type rather than assume
    LindbladErrorgen: here the errorgen is an EmbeddedErrorgen (the construction CloudNoiseModel
    uses with errcomp_type='errorgens'), so torch_base must compute exp(embed(L))."""
    op = _a_composed_op(_noisy_cptp_model('CPTPLND'))
    eg = op.factorops[1].errorgen
    assert isinstance(eg, LindbladErrorgen)
    exp = ExpErrorgenOp(EmbeddedErrorgen(QubitSpace(2), (0,), eg))

    assert np.allclose(torch_base_value(exp), exp.to_dense('HilbertSchmidt'), atol=1e-9)
    assert np.allclose(torch_base_jacobian(exp), exp.deriv_wrt_params(), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_composedop_torch_base(mode):
    """ComposedOp.torch_base composes its (static prefactor + ExpErrorgenOp) factors correctly."""
    op = _a_composed_op(_noisy_cptp_model(mode))
    # the ideal/target prefactor is a static (0-param) factor handled by StaticTorchable
    assert op.factorops[0].num_params == 0

    assert np.allclose(torch_base_value(op), op.to_dense('HilbertSchmidt'), atol=1e-9)
    assert np.allclose(torch_base_jacobian(op), op.deriv_wrt_params(), atol=1e-6)


def test_composedstate_torch_base():
    """ComposedState (Lindblad SPAM prep) torch_base = error_map_superop @ static_ket; reproduces
    to_dense and its analytic Jacobian.  ComposedState.torch_base returns a super-ket (1-D)."""
    rho = noisy_full_cptplnd_model().preps['rho0']

    assert np.allclose(torch_base_value(rho), rho.to_dense('HilbertSchmidt'), atol=1e-9)
    assert np.allclose(torch_base_jacobian(rho), rho.deriv_wrt_params(), atol=1e-6)


def test_composedpovm_torch_base():
    """ComposedPOVM (Lindblad SPAM measurement) torch_base = base_effect_rows @ error_map_superop;
    reproduces the per-effect dual vectors and their analytic Jacobians.  Rows are in POVM-effect order
    (the row order TorchForwardSimulator multiplies against the super-ket)."""
    povm = noisy_full_cptplnd_model().povms['Mdefault']
    sd = povm.stateless_data(torch.float64, 'cpu')
    t = torch.from_numpy(np.ascontiguousarray(povm.to_vector()))

    # value: row i is effect i's dual vector
    eff_rows = np.stack([povm[k].to_dense('HilbertSchmidt') for k in povm.keys()])
    assert np.allclose(torch_base_value(povm), eff_rows, atol=1e-9)

    # autodiff Jacobian (n_effects, dim, n_params) == stacked per-effect deriv_wrt_params -- a 3-D
    # output the generic torch_base_jacobian helper isn't shaped for, so this stays manual.
    J = torch.func.jacrev(lambda tp: type(povm).torch_base(sd, tp))(t).detach().numpy()
    Jref = np.stack([povm[k].deriv_wrt_params() for k in povm.keys()])
    assert J.shape == Jref.shape
    assert np.allclose(J, Jref, atol=1e-6)
