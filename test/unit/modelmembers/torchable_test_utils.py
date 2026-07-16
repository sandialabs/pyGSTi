"""
Shared helpers for exercising the Torchable API (stateless_data/torch_base) directly against a
ModelMember's numpy to_dense/deriv_wrt_params, used by test_torchable_members.py and
test_experrorgen_torch.py.

Import this only after ``torch = pytest.importorskip("torch")`` in the caller, since it imports
torch unconditionally.
"""
import numpy as np
import torch

from pygsti.modelpacks import smq1Q_XYI


def torch_base_value(obj, dtype=torch.float64, device='cpu'):
    """``type(obj).torch_base(obj.stateless_data(...), from_numpy(obj.to_vector()))``, as a numpy array."""
    sd = obj.stateless_data(dtype, device)
    t_param = torch.from_numpy(np.ascontiguousarray(obj.to_vector()))
    return type(obj).torch_base(sd, t_param).detach().numpy()


def torch_base_jacobian(obj, dtype=torch.float64, device='cpu'):
    """``torch.func.jacrev(torch_base)`` at obj's current parameters, as a numpy array shaped
    ``(obj.to_dense('HilbertSchmidt').size, obj.num_params)`` -- i.e. reproduces ``deriv_wrt_params``."""
    sd = obj.stateless_data(dtype, device)
    t_param = torch.from_numpy(np.ascontiguousarray(obj.to_vector()))
    out_size = obj.to_dense('HilbertSchmidt').size
    J = torch.func.jacrev(lambda tp: type(obj).torch_base(sd, tp))(t_param)
    return J.reshape(out_size, -1).detach().numpy()


def noisy_full_cptplnd_model(seed=5, scale=0.03):
    """1-qubit model with Lindblad (CPTPLND) SPAM *and* operations -- prep is a ComposedState, POVM is a
    ComposedPOVM, gates are ComposedOps -- perturbed off the ideal."""
    m = smq1Q_XYI.target_model()
    m.set_all_parameterizations('CPTPLND')
    rng = np.random.default_rng(seed)
    m.from_vector(m.to_vector() + scale * rng.standard_normal(m.num_params))
    return m
