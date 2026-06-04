"""
Torchable de-risking for LindbladCoefficientBlock (GitHub issue 607, Approach A).

Proves the design "sets up Torchable": each parameterization supplies a differentiable
``block_data_torch`` (params -> block_data), exposed on the block as ``stateless_data`` /
``torch_base``.  We check that

  (1) ``torch_base(stateless_data(torch.float32, 'cpu'), torch.from_numpy(to_vector()))`` reproduces ``block_data``;
  (2) ``torch.func.jacrev(torch_base)`` equals the numpy ``deriv_wrt_params`` -- i.e. PyTorch
      autodiff reproduces the manual block_data Jacobian (so the Torch path needs no hand-written
      derivatives); and
  (3) the LindbladErrorgen-level recipe ``einsum('i,ijk->jk', torch_base.ravel(), G)`` reproduces
      the numpy error-generator contribution, and its autodiff equals the numpy ``superop_deriv``.

These are skipped if torch is unavailable.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygsti.baseobjs.basis import Basis
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB

# Non-static (block_type, param_mode) cases, 1-qubit pp + a qutrit gm 'other'/cholesky for coverage.
CASES = [('pp', 4, 'ham', 'elements'),
         ('pp', 4, 'other_diagonal', 'elements'),
         ('pp', 4, 'other_diagonal', 'cholesky'),
         ('pp', 4, 'other_diagonal', 'depol'),
         ('pp', 4, 'other_diagonal', 'reldepol'),
         ('pp', 4, 'other', 'elements'),
         ('pp', 4, 'other', 'cholesky'),
         ('gm', 9, 'other', 'cholesky')]
IDS = ["%s%d-%s-%s" % c for c in CASES]


def _block_at_params(bname, dim, bt, pm, seed=5):
    blk = LCB(bt, Basis.cast(bname, dim), param_mode=pm)
    blk.from_vector(np.random.default_rng(seed).standard_normal(blk.num_params))
    return blk


def _jacrev_np(f, t_param):
    """numpy Jacobian of a (possibly complex-valued) torch function f w.r.t. the real tensor t_param.
    torch.func.jacrev requires real outputs, so for complex outputs we differentiate Re and Im
    separately (in practice TorchForwardSimulator differentiates the *real* circuit probabilities, so
    this split is only needed for these intermediate-tensor checks)."""
    out = f(t_param)
    if out.is_complex():
        Jre = torch.func.jacrev(lambda tp: f(tp).real)(t_param).detach().numpy()
        Jim = torch.func.jacrev(lambda tp: f(tp).imag)(t_param).detach().numpy()
        return Jre + 1j * Jim
    return torch.func.jacrev(f)(t_param).detach().numpy()


@pytest.mark.parametrize("bname,dim,bt,pm", CASES, ids=IDS)
def test_torch_base_matches_block_data_and_jacobian(bname, dim, bt, pm):
    blk = _block_at_params(bname, dim, bt, pm)
    vv = blk.to_vector()
    sd = blk.stateless_data(torch.float32, 'cpu')
    t_param = torch.from_numpy(np.ascontiguousarray(vv))

    # (1) torch_base reproduces block_data
    t_bd = type(blk).torch_base(sd, t_param).detach().numpy()
    assert np.allclose(t_bd, blk.block_data, atol=1e-10)

    # (2) autodiff Jacobian == numpy deriv_wrt_params
    J = _jacrev_np(lambda tp: type(blk).torch_base(sd, tp), t_param)
    Jnp = np.asarray(blk.deriv_wrt_params(vv))
    assert J.shape == Jnp.shape
    assert np.allclose(J, Jnp, atol=1e-8)


@pytest.mark.parametrize("bname,dim,bt,pm", CASES, ids=IDS)
def test_torch_errorgen_recipe_matches_superop_deriv(bname, dim, bt, pm):
    """einsum(G, torch_base) reproduces the numpy error-gen contribution, and its autodiff equals
    the numpy superop_deriv -- the LindbladErrorgen.torch_base recipe (the follow-on task)."""
    blk = _block_at_params(bname, dim, bt, pm)
    vv = blk.to_vector()
    sd = blk.stateless_data(torch.float32, 'cpu')
    G = np.asarray(blk.create_lindblad_term_superoperators(bname, sparse=False, flat=True))
    t_param = torch.from_numpy(np.ascontiguousarray(vv))
    t_G = torch.from_numpy(np.ascontiguousarray(G))

    def errorgen(tp):
        bd = type(blk).torch_base(sd, tp).reshape(-1).to(t_G.dtype)
        return torch.einsum('i,ijk->jk', bd, t_G)

    # value matches numpy errorgen contribution
    eg = errorgen(t_param).detach().numpy()
    eg_np = np.einsum('i,ijk->jk', blk.block_data.ravel(), G)
    assert np.allclose(eg, eg_np, atol=1e-10)

    # autodiff w.r.t. params == numpy superop_deriv (the error gen is real-valued; differentiate Re)
    nP = blk.num_params
    Jeg = torch.func.jacrev(lambda tp: errorgen(tp).real)(t_param).detach().numpy()   # (d2, d2, nP)
    sd_np = np.asarray(blk.superop_deriv_wrt_params(G, vv, superops_are_flat=True))
    sd_np = sd_np.reshape(sd_np.shape[0], sd_np.shape[1], nP)
    assert np.allclose(Jeg, sd_np, atol=1e-7)
