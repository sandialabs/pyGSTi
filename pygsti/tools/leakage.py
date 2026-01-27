#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

from functools import lru_cache

import copy
from pygsti.tools import optools as pgot
from pygsti.tools import basistools as pgbt
from pygsti.tools import matrixtools as pgmt
from pygsti.tools.basistools import stdmx_to_vec
from pygsti.baseobjs import Label
from pygsti.baseobjs.basis import TensorProdBasis, Basis, BuiltinBasis
import numpy as np
import scipy.linalg as la
import warnings

import warnings

from typing import Union, Dict, Optional, List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from pygsti.protocols.gst import ModelEstimateResults, GSTGaugeOptSuite
    from pygsti.models import ExplicitOpModel
    from pygsti.processors import QubitProcessorSpec



# Question: include the parenthetical in the heading below?
NOTATION = \
"""
Default notation (deferential to text above)
--------------------------------------------
 * H is a complex Hilbert space equipped with the standard basis.

 * C, the computational subspace, is the complex-linear span of the first dim(C)
   standard basis vectors of H.

 * Given a complex Hilbert space, U, we write M[U] to denote the space of linear
   operators from U to U. Elements of M[U] have natural matrix representations.

 * Given a space of linear operators, L, we write S[L] for the set of linear 
   transformations ("superoperators") from L to L.
   
        Matrix representations for elements of S[L] are only meaningful in the
        presence of a designated basis for L.
        
        If elements of L are naturally expressed as matrices, then a basis for L
        lets us identify elements of L with vectors of length dim(L).

 * If U denotes a complex Hilbert space (e.g., U=H or U=C), then we abbreviate 
   S[M[U]] by S[U].
\n
"""

def set_docstring(docstr):
    def assign(fn):
        fn.__doc__ = docstr
        return fn
    return assign


# MARK: metrics

@lru_cache
@set_docstring(
"""
Here, H has dimension dim, and C ⊂ H has co-dimension n_leak. 

This function returns a rank-1 density matrix, rho_t = |psi><psi|, whose state vector
representation is real and "maximally spread out" in C⨂C as a subspace of H⨂H.

*A remark.* The Choi matrix for a superoperator G in S[H] is G(rho_t), where rho_t is
the output of this function when n_leak=0.
""" + NOTATION)
def tensorized_teststate_density(dim: int, n_leak: int) -> np.ndarray:
    temp = np.eye(dim, dtype=np.complex128)
    if n_leak > 0:
        temp[-n_leak:,-n_leak:] = 0.0
    temp /= np.sqrt(dim - n_leak)
    psi = pgbt.stdmx_to_stdvec(temp).ravel()
    #
    #   |psi> = (|00> + |11> + ... + |dim - n_leak - 1>) / sqrt(dim - n_leak).
    #
    rho_test = np.outer(psi, psi)
    return rho_test


@set_docstring(
"""
The pair (op_x, op_y) represent some superoperators (X, Y) in S[H], using op_basis.

Let rho_t = tensorized_teststate_density(dim(H), n_leak), and set I to the identity
operator in S[H].

This function returns a triplet, consisting of
 * tb : a tensor product basis for S[H]⨂S[H],
 * a vector representation of X⨂I(rho_t) in the basis tb, and
 * a vector representation of Y⨂I(rho_t) in the basis tb.

*Warning!* At present, this function can only be used for gates over a single system
(e.g., a single qubit), not for tensor products of such systems.
""" + NOTATION)
def apply_tensorized_to_teststate(op_x: np.ndarray, op_y, op_basis: np.ndarray, n_leak: int=0) -> tuple[TensorProdBasis, np.ndarray, np.ndarray]:
    dim = int(np.sqrt(op_x.shape[0]))
    assert op_x.shape == (dim**2, dim**2)
    assert op_y.shape == (dim**2, dim**2)
    # op_x and op_y act on M[H].
    #
    # We care about op_x and op_y only up to their action on the subspace
    #    U = {rho in M[H] : <i|rho|i> = 0 for all i >= dim - n_leak }.
    #
    # It's easier to talk about this subspace (and related subspaces) if op_x and op_y are in
    # the standard basis. So the first thing we do is convert to that basis.
    std_basis = BuiltinBasis('std', dim**2)
    op_x_std = pgbt.change_basis(op_x, op_basis, std_basis)
    op_y_std = pgbt.change_basis(op_y, op_basis, std_basis)
   
    # Our next step is to construct lifted operators "lift_op_x" and "lift_op_y" that act on the
    # tensor product space M[H]⨂M[H] according to the identities
    #
    #   lift_op_x( sigma \otimes rho ) = op_x(sigma) \otimes rho
    #   lift_op_y( sigma \otimes rho ) = op_y(sigma) \otimes rho
    #
    # for all sigma, rho in M[H]. The way we do this implicitly fixes a basis for S[H]⨂S[H] as
    # the tensor product basis. We'll make that explicit later on.
    idle_gate = np.eye(dim**2, dtype=np.complex128)
    lift_op_x_std = np.kron(op_x_std, idle_gate)
    lift_op_y_std = np.kron(op_y_std, idle_gate)

    # Now we'll compare these lifted operators by how they act on specific state in M[H]⨂M[H].
    rho_test = tensorized_teststate_density(dim, n_leak)

    # lift_op_x and lift_op_y only act on states in their superket representations, so we convert
    # rho_test to a superket representation in the induced tensor product basis for S[H]⨂S[H].
    ten_std_basis = TensorProdBasis((std_basis, std_basis))
    rho_test_superket = pgbt.stdmx_to_vec(rho_test, ten_std_basis).ravel()

    temp1 = lift_op_x_std @ rho_test_superket
    temp2 = lift_op_y_std @ rho_test_superket

    return ten_std_basis, temp1, temp2


@lru_cache
def leading_dxd_submatrix_basis_vectors(d: int, n: int, current_basis: Basis):
    """
    Let "H" denote n^2 dimensional Hilbert-Schdmit space, and let "U" denote the d^2
    dimensional subspace of H spanned by vectors whose Hermitian matrix representations
    are zero outside the leading d-by-d submatrix.

    This function returns a column-unitary matrix "B" where P = B B^{\\dagger} is the
    orthogonal projector from H to U with respect to current_basis. We return B rather
    than P only because it's simpler to get P from B than it is to get B from P.
    
    See below for this function's original use-case.
    
    Raison d'etre
    -------------
    Suppose we canonically measure the distance between two process matrices (M1, M2) by

        D(M1, M2; H) = max || (M1 - M2) v ||
                            v is in H,                   (Eq. 1)
                            tr(v) = 1,
                            v is positive

    for some norm || * ||.  Suppose also that we want an analog of this distance when
    (M1, M2) are restricted to the linear subspace U consisting of all vectors in H
    whose matrix representations are zero outside of their leading d-by-d submatrix.

    One natural way to do this is via the function D(M1, M2; U) -- i.e., just replace
    H in (Eq. 1) with the subspace U. Using P to denote the orthogonal projector onto U,
    we claim that we can evaluate this function via the identity

        D(M1, M2; U) = D(M1 P, M2 P; H).                (Eq. 2)

    To see why this is the case, consider a positive vector v and its projection u = P v.
    Since a vector is called positive whenever its Hermitian matrix representation is positive
    semidefinite (PSD), we need to show that u is positive. This can be seen by considering
    block 2-by-2 partitions of the matrix representations of (u,v), where the leading block
    is d-by-d:

        mat(v) = [x11,  x12]         and      mat(u) = [x11,  0]
                 [x21,  x22]                           [  0,  0].
    
    In particular, u is positive if and only if x11 is PSD, and x11 must be PSD for v
    to be positive. Furthermore, positivity of v requires that x22 is PSD, which implies

        0 <= tr(u) = tr(x11) <= tr(v).
    
    Given this, it is easy to establish (Eq 2.) by considering how the following pair 
    of problems have the same optimal objective function value

        max || (M1 - M2) P v ||         and        max || (M1 - M2) P v || 
            mat(v) = [x11, x12]                         mat(v) = [x11, x12]
                     [x21, x22]                                  [x21, x22]
            mat(v) is PSD                               x11 is PSD
            tr(x11) + tr(x22) = 1                       tr(x11) <= 1.

    In fact, this can be taken a little further! The whole argument goes through unchanged
    if, instead of starting with the objective function || (M1 - M2) v ||, we started with
    f((M1 - M2) v) and f satisfied the property that f(c v) >= f(v) whenever c is a scalar
    greater than or equal to one.
    """
    assert d <= n
    current_basis = Basis.cast(current_basis, dim=n**2)
    X = current_basis.create_transform_matrix('std')
    X = X.T.conj()
    if d == n:
        return X
    # we have to select a proper subset of columns in current_basis
    std_basis = BuiltinBasis(name='std', dim_or_statespace=n**2)
    label2ind = std_basis.elindlookup
    basis_ind = []
    for i in range(d):
        for j in range(d):
            ell = f"({i},{j})"
            basis_ind.append(label2ind[ell])
    basis_ind = np.array(basis_ind)
    submatrix_basis_vectors = X[:, basis_ind]
    return submatrix_basis_vectors


CHOI_INDUCED_METRIC = \
"""
The pair (op_x, op_y) represent some superoperators (X, Y) in S[H], using op_basis.

Let rho_t = tensorized_teststate_density(dim(H), n_leak), and set I to the identity
operator in S[H].

This function returns the %s between X⨂I(rho_t) and Y⨂I(rho_t).

*Warning!* At present, this function can only be used for gates over a single system
(e.g., a single qubit), not for tensor products of such systems.
""" + NOTATION


@set_docstring(CHOI_INDUCED_METRIC % 'entanglement fidelity')
def subspace_entanglement_fidelity(op_x, op_y, op_basis, n_leak=0):
    ten_std_basis, temp1, temp2 = apply_tensorized_to_teststate(op_x, op_y, op_basis, n_leak)
    temp1_mx = pgbt.vec_to_stdmx(temp1, ten_std_basis, keep_complex=True)
    temp2_mx = pgbt.vec_to_stdmx(temp2, ten_std_basis, keep_complex=True)
    ent_fid = pgot.fidelity(temp1_mx, temp2_mx)
    return ent_fid


@set_docstring(CHOI_INDUCED_METRIC % 'jamiolkowski trace distance')
def subspace_jtracedist(op_x, op_y, op_basis, n_leak=0):
    ten_std_basis, temp1, temp2 = apply_tensorized_to_teststate(op_x, op_y, op_basis, n_leak)
    temp1_mx = pgbt.vec_to_stdmx(temp1, ten_std_basis, keep_complex=True)
    temp2_mx = pgbt.vec_to_stdmx(temp2, ten_std_basis, keep_complex=True)
    j_dist = pgot.tracedist(temp1_mx, temp2_mx)
    return j_dist


@set_docstring(
"""
The pair (op_x, op_y) represent some superoperators (X, Y) in S[H], using op_basis.

We return the Frobenius distance between op_x @ P and op_y @ P, where P is the
projector onto the computational subspace (i.e., C), of co-dimension n_leak.

*Warning!* At present, this function can only be used for gates over a single system
(e.g., a single qubit), not for tensor products of such systems.
""" + NOTATION)
def subspace_superop_fro_dist(op_x, op_y, op_basis, n_leak=0):
    diff = op_x -  op_y
    if n_leak == 0:
        return np.linalg.norm(diff, 'fro')
    
    d = int(np.sqrt(op_x.shape[0]))
    assert op_x.shape == op_y.shape == (d**2, d**2)
    B = leading_dxd_submatrix_basis_vectors(d-n_leak, d, op_basis)
    P = B @ B.T.conj()
    return np.linalg.norm(diff @ P)


def _direct_sum_unitary_group(subspace_bases, full_basis):
    from pygsti.models.gaugegroup import UnitaryGaugeGroup, DirectSumUnitaryGroup, TrivialGaugeGroup
    subgroups : list[Union[TrivialGaugeGroup, UnitaryGaugeGroup]] = []
    for sb in subspace_bases:
        udim = int(np.round(sb.dim**0.5))
        assert udim**2 == sb.dim
        if udim == 1:
            g = TrivialGaugeGroup(sb.state_space)
        else:
            g = UnitaryGaugeGroup(sb.state_space, sb)
        subgroups.append(g)
    g_full = DirectSumUnitaryGroup(tuple(subgroups), full_basis)
    return g_full

# MARK: model construction

def leaky_qubit_model_from_pspec(
        ps_2level: QubitProcessorSpec, mx_basis: Union[str, Basis]='l2p1',
        levels_readout_zero=(0,), default_idle_gatename: Label = Label(())
    ) -> ExplicitOpModel:
    """
    Return an ExplicitOpModel `m` whose (ideal) gates act on three-dimensional Hilbert space and whose members
    are represented in `mx_basis`, constructed as follows:

        The Hermitian matrix representation of m['rho0'] is the 3-by-3 matrix with a 1 in the upper-left
        corner and all other entries equal to zero.
    
        Operations in `m` are defined by taking each 2-by-2 unitary `u2` from ps_2level, and promoting it
        to a 3-by-3 unitary according to 

            u3 = [u2[0, 0], u2[0, 1], 0]
                 [u2[1, 0], u2[1, 1], 0]
                 [       0,       0,  1]

        m['Mdefault'] has two effects, labeled "0" and "1". If E0 is the Hermitian matrix representation of
        effect "0", then E0[i,i]=1 for all i in levels_readout_zero, and E0 is zero in all other components.

    This function might be called in a workflow like the following:

        from pygsti.models     import create_explicit_model
        from pygsti.algorithms import find_fiducials, find_germs
        from pygsti.protocols  import StandardGST, StandardGSTDesign, ProtocolData

        # Step 1: Make the experiment design for the 1-qubit system.
        tm_2level = create_explicit_model( ps_2level, ideal_spam_type='CPTPLND', ideal_gate_type='CPTPLND' )
        fids    = find_fiducials( tm_2level )
        germs   = find_germs( tm_2level )
        lengths = [1, 2, 4, 8, 16, 32]
        design  = StandardGSTDesign( tm_2level, fids[0], fids[1], germs, lengths )   
        
        # Step 2: ... run the experiment specified by "design"; store results in a directory "dir" ...

        # Step 3: read in the experimental data and run GST.
        pd  = ProtocolData.from_dir(dir)
        tm_3level = leaky_qubit_model_from_pspec( ps_2level, basis='l2p1' )
        gst = StandardGST( modes=('CPTPLND',), target_model=tm_3level, verbosity=4 )
        res = gst.run(pd)
    """
    from pygsti.models.explicitmodel import ExplicitOpModel
    from pygsti.baseobjs.statespace import ExplicitStateSpace
    from pygsti.modelmembers.povms import UnconstrainedPOVM
    from pygsti.modelmembers.states import FullState
    assert ps_2level.num_qubits == 1
    if ps_2level.idle_gate_names == ['{idle}']:
        ps_2level.rename_gate_inplace('{idle}', default_idle_gatename)

    if isinstance(mx_basis, str):
        mx_basis = BuiltinBasis(mx_basis, 9)
    assert isinstance(mx_basis, Basis)

    ql = ps_2level.qubit_labels[0]
    
    Us = ps_2level.gate_unitaries
    rho0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], complex)
    E0   = np.zeros((3, 3))
    E0[levels_readout_zero, levels_readout_zero] = 1
    E1   = np.eye(3, dtype=complex) - E0

    ss = ExplicitStateSpace([ql],[3])
    tm_3level = ExplicitOpModel(ss, mx_basis) # type: ignore
    tm_3level.preps['rho0']     =  FullState(stdmx_to_vec(rho0, mx_basis))
    tm_3level.povms['Mdefault'] =  UnconstrainedPOVM(
        [("0", stdmx_to_vec(E0, mx_basis)), ("1", stdmx_to_vec(E1, mx_basis))], evotype="default",
    )

    def u2x2_to_9x9_superoperator(u2x2):
        u3x3 = np.eye(3, dtype=np.complex128)
        u3x3[:2,:2] = u2x2
        superop_std = pgot.unitary_to_std_process_mx(u3x3)
        superop = pgbt.change_basis(superop_std, 'std', mx_basis)
        return superop

    for gatename, unitary in Us.items():
        gatekey = gatename if isinstance(gatename, Label) else Label((gatename, ql))
        tm_3level.operations[gatekey] = u2x2_to_9x9_superoperator(unitary)

    subgroup_bases = [Basis.cast('pp', 4), Basis.cast('pp', 1)]
    g_full = _direct_sum_unitary_group(subgroup_bases, mx_basis)
    tm_3level.default_gauge_group = g_full
    tm_3level.sim = 'map'  # can use 'matrix', if that's preferred for whatever reason.
    return tm_3level


# MARK: gaugeopt

def lagoified_gopparams_dicts(gopparams_dicts: List[Dict]) -> List[Dict]:
    """
    goppparams_dicts is a list-of-dicts (LoDs) representation of a gauge optimization suite
    suitable for models without leakage (e.g., a model of a 2-level system).

    This function returns a new gauge optimization suite (also in the LoDs representation)
    by applying leakage-specific modifications to a deep-copy of gopparams_dicts.

    Example
    -------
    Suppose we have a ModelEstimateResults object called `results` that includes a
    CPTPLND estimate, and we want to update the models of that estimate to include
    two types of leakage-aware gauge optimization.

        #
        # Step 1: get the input to this function
        #
        estimate = results.estimates['CPTPLND']
        model    = estimate.models['target']
        stdsuite = GSTGaugeOptSuite(gaugeopt_suite_names=('stdgaugeopt',))
        gopparams_dicts = stdsuite.to_dictionary(model)['stdgaugeopt']
        
        #
        # Step 2: use this function to build our GSTGaugeOptSuite.
        #
        s = lagoified_gopparams_dicts(gopparams_dicts)
        c = lagoified_gopparams_dicts(gopparams_dicts)
        c[-1]['gates_metric'] = 'fidelity'
        c[-1]['spam_metric']  = 'fidelity'
        # ^ this example's "custom" leakage gauge optimization uses an infidelity loss,
        #   rather than the default of squared Frobenius loss.
        specification = {'LAGO-std': s,'LAGO-custom': c}
        gos = GSTGaugeOptSuite(gaugeopt_argument_dicts=specification)
        
        #
        # Step 3: updating `estimate` requires that we modify `results`.
        #
        add_lago_models(results, 'CPTPLND', gos)

    After those lines execute, the `estimate.models` dict will have two new
    key-value pairs, where the keys are 'LAGO-std' and 'LAGO-custom'.
    """
    from pygsti.models.gaugegroup import UnitaryGaugeGroup
    tm = gopparams_dicts[0]['target_model']
    gopparams_dicts = [gp for gp in gopparams_dicts if 'TPSpam' not in str(type(gp['_gaugeGroupEl']))]
    # ^ That list will __usually__ be length-1.

    gopparams_dicts = copy.deepcopy(gopparams_dicts)
    for inner_dict in gopparams_dicts:
        inner_dict['method'] = 'L-BFGS-B'
        # ^ We need this optimizer because it doesn't require a gradient oracle.
        #
        inner_dict['n_leak'] = 1
        # ^ This function could accept n_leak as an argument instead. However,
        #   downstream functions for gauge optimization only support n_leak=0 or 1.
        # 
        #   When n_leak=1 we use subspace-restricted loss functions that only care
        #   about mismatches between an estimate and a target when restricted to the
        #   computational subspace. We have code for evaluating the loss functions
        #   themselves, but not their gradients.
        #
        gg = UnitaryGaugeGroup(tm.basis.state_space, tm.basis)
        inner_dict['gauge_group'] = gg
        inner_dict['_gaugeGroupEl'] = gg.compute_element(gg.initial_params)
        # ^ We insist on the unitary gauge group because other common gauge groups
        #   (e.g., FullTPGaugeGroup) impose requirements on tm.basis.
        #
        inner_dict['gates_metric'] = 'frobenius'
        inner_dict['spam_metric']  = 'frobenius'
        # ^ We use Frobenius rather than Frobenius squared to avoid biasing toward
        #   gauges where the leakage is "spread out" across multiple gates.
        #
        inner_dict['item_weights'] = {'gates': 1.0, 'spam': 1.0}
        # ^ The precise weights here are somewhat arbitrary. We place weight on
        #   SPAM because SPAM plays a distinguished role in defining the 
        #   computational subspace.
    
    # Below, we add a final gauge optimization step that preserves separation between 
    # what we've identified as the computational subspace and leakage space. This
    # step uses squared Frobenius norm as an objective, since that's better-behaved
    # than plain Frobenius norm.
    inner_dict = inner_dict.copy()
    gg = _direct_sum_unitary_group([Basis.cast('pp', 4), Basis.cast('pp', 1)], tm.basis)
    inner_dict['gauge_group'] = gg
    inner_dict['_gaugeGroupEl'] = gg.compute_element(gg.initial_params)
    inner_dict['gates_metric'] = 'frobenius squared'
    inner_dict['spam_metric']  = 'frobenius squared'
    inner_dict['item_weights'] = {'gates': 1.0, 'spam': 1.0}
    gopparams_dicts.append(inner_dict)
    return gopparams_dicts


def std_lago_gopsuite(model: ExplicitOpModel) -> dict[str, list[dict]]:
    """
    Return a dictionary of the form {'LAGO': v}, where v is a
    list-of-dicts representation of a gauge optimization suite.

    We construct v by getting the list-of-dicts representation of the
    "stdgaugeopt" suite for `model`, and then changing some of its 
    options to be suitable for leakage-aware gauge optimization. These
    changes are made in the `lagoified_gopparams_dicts` function.
    """
    from pygsti.protocols.gst import GSTGaugeOptSuite
    std_gop_suite = GSTGaugeOptSuite(gaugeopt_suite_names=('stdgaugeopt',))
    std_gos_lods  = std_gop_suite.to_dictionary(model)['stdgaugeopt']  # list of dictionaries
    lago_gos_lods = lagoified_gopparams_dicts(std_gos_lods)
    gop_params = {'LAGO': lago_gos_lods}
    return gop_params


def add_lago_models(results: ModelEstimateResults, est_key: Optional[str] = None, gos: Optional[GSTGaugeOptSuite] = None, verbosity: int = 0):
    """
    Update each estimate in results.estimates (or just results.estimates[est_key],
    if est_key is not None) with a model obtained by parameterization-preserving
    leakage-aware gauge optimization.
    
    If no gauge optimization suite is provided, then we construct one by making
    appropriate modifications to either the estimate's existing 'stdgaugeopt' suite
    (if that exists) or to the 'stdgaugeopt' suite induced by the target model.
    """
    from pygsti.protocols.gst import GSTGaugeOptSuite, _add_param_preserving_gauge_opt, _add_gauge_opt
    if isinstance(est_key, str):
        if gos is None:
            existing_est  = results.estimates[est_key]
            if 'stdgaugeopt' in existing_est.goparameters:
                std_gos_lods  = existing_est.goparameters['stdgaugeopt']
                lago_gos_lods = lagoified_gopparams_dicts(std_gos_lods)
                gop_params = {'LAGO': lago_gos_lods}
            else:
                gop_params = std_lago_gopsuite(results.estimates[est_key].models['target'])
            gos = GSTGaugeOptSuite(gaugeopt_argument_dicts=gop_params)
        try:
            _add_param_preserving_gauge_opt(results, est_key, gos, verbosity)
        except:
            # parameterization-preserving gauge optimization fails if the seed model
            # uses members other than ComposedState, ComposedOp, ComposedPOVM, etc..
            # So we add a non-preserving gauge optimization here.
            tm = results.estimates[est_key].models['target']
            _add_gauge_opt(results, est_key, gos, tm, tuple())
    elif est_key is None:
        for est_key in results.estimates.keys():
            add_lago_models(results, est_key, gos, verbosity)
    else:
        raise ValueError()
    return


# MARK: reports


def _add_all_hessians(mer: ModelEstimateResults, kwargs_for_projhess=None):
    # NOTE: this function is not leakage-specific.
    if kwargs_for_projhess is None:
        kwargs_for_projhess = {'projection_type': 'intrinsic error'}

    from pygsti.forwardsims import MatrixForwardSimulator, MapForwardSimulator
    for estname, est in mer.estimates.items():
        if estname == 'Target':
            continue
        for gop_name in est.goparameters:
            mdl = est.models[gop_name]
            if isinstance(mdl.sim, MatrixForwardSimulator):
                # Replace with MapForwardSimulator, since that's the only
                # ForwardSimulator that can compute objective function 
                # Hessians with a reasonable amount of memory.
                mdl.sim = MapForwardSimulator
            crf = est.add_confidence_region_factory(gop_name, 'final')
            crf.compute_hessian()
            crf.project_hessian(**kwargs_for_projhess)
    return


def _add_lago_estimates(mer: ModelEstimateResults, gaugeopt_verbosity: int = 0):
    # This is broken out into its own function in case someone wants to use
    # it in a report-generation function other than construct_leakage_report.
    for ek in mer.estimates:
        assert isinstance(ek, str)
        if ek == 'Target':
            # There's no need to gauge-optimize in this case.
            continue
        add_lago_models(mer, ek, verbosity=gaugeopt_verbosity)
    return


def construct_leakage_report(
        results : ModelEstimateResults,
        title : str = 'auto',
        *, # no positional args past "title."
        confidence_level    : Optional[Union[int,float]] = None,
        kwargs_projhess     : Optional[dict[str,Any]]    = None,
        kwargs_stdreport    : Optional[dict[str,Any]]    = None,
        gaugeopt_verbosity  : int = 0,
    ):
    """
    This is a wrapper around construct_standard_report. It generates a Report object
    with leakage analysis, and returns that object along with a copy of ``results`` which
    contains gauge-optimized models created during leakage analysis.
    
    If provided arguments indicate a desire for confidence intervals in the report,
    then this function also computes the necessary likelihood function Hessians.

    Notes
    -----
    The special gauge optimization performed by this function uses the unitary gauge group,
    and uses a modified version of the Frobenius distance loss function. The modification
    reflects how the target gates in a leakage model are only _really_ defined on the
    computational subspace.
    """
    if kwargs_stdreport is None:
        kwargs_stdreport = dict()

    # Prep work. Pack title and confidence_level into kwargs_stdreport.
    clobbering_kwargs = {'title': title, 'confidence_level': confidence_level}
    for k, a in clobbering_kwargs.items():
        kwargs_stdreport[k] = kwargs_stdreport.get(k, a)
        if a != (clobberable_a := kwargs_stdreport[k]):
            msg  = f"Clobbering {k} in kwargs_stdreport ({clobberable_a}) "
            msg += f"with this function's {k} argument ({a})."
            warnings.warn(msg)
            kwargs_stdreport[k] = a

    # Actual work. Deep-copy results and mutate that object in-place.
    results_out = copy.deepcopy(results)
    _add_lago_estimates( results_out, gaugeopt_verbosity )
    if kwargs_stdreport['confidence_level'] is not None:
        _add_all_hessians( results_out, kwargs_projhess )

    # Wrap it up in a bow.
    from pygsti.report import construct_standard_report
    report = construct_standard_report(
        results_out, advanced_options={'n_leak': 1}, **kwargs_stdreport
    )
    return report, results_out
