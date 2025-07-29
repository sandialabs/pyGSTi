#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy
from pygsti import modelmembers as pgmm
from pygsti.tools import optools as pgot
from pygsti.tools import basistools as pgbt
from pygsti.processors import QubitProcessorSpec
from pygsti.modelmembers.povms import TPPOVM
from pygsti.baseobjs.basis import TensorProdBasis, Basis, BuiltinBasis
import numpy as np

from typing import Union, Dict, TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    from pygsti.protocols.gst import ModelEstimateResults
else:
    ModelEstimateResults = TypeVar('ModelEstimateResults')


# MARK: metrics

def tensorized_teststate_density(dim, n_leak):
    # Return a test state density matrix rho_t = |psi><psi|, where
    #
    #   |psi> = (|00> + |11> + ... + |dim - n_leak - 1>) / sqrt(dim - n_leak).
    #
    temp = np.eye(dim, dtype=np.complex128)
    if n_leak > 0:
        temp[-n_leak:,-n_leak:] = 0.0
    temp /= np.sqrt(dim - n_leak)
    psi = pgbt.stdmx_to_stdvec(temp).ravel()
    rho_test = np.outer(psi, psi)
    return rho_test


def apply_tensorized_to_teststate(op_a, op_b, mx_basis, n_leak=0):
    # Note: this function is only really useful for gates on a single system (qubit, qutrit, qudit);
    # not tensor products of such systems.
    dim = int(np.sqrt(op_a.shape[0]))
    assert op_a.shape == (dim**2, dim**2)
    assert op_b.shape == (dim**2, dim**2)

    # op_a and op_b act on the smallest real-linear space "S" that contains density matrices 
    # for a dim-level system.
    #
    # We care about op_a and op_b only up to their action on the subspace
    #    U = {rho in S : <i|rho|i> = 0 for all i >= dim - n_leak }.
    #
    # It's easier to talk about this subspace (and related subspaces) if op_a and op_b are in
    # the standard basis. So the first thing we do is convert to that basis.
    std_basis = BuiltinBasis('std', dim**2)
    op_a = pgbt.change_basis(op_a, mx_basis, std_basis)
    op_b = pgbt.change_basis(op_b, mx_basis, std_basis)
   
    # Our next step is to construct lifted operators "lift_op_a" and "lift_op_b" that act on the
    # tensor product space S2 = (S \otimes S) according to the identities
    #
    #   lift_op_a( sigma \otimes rho ) = op_a(sigma) \otimes rho
    #   lift_op_b( sigma \otimes rho ) = op_b(sigma) \otimes rho
    #
    # for all sigma, rho in S. The way we do this implicitly fixes a basis for S2 as the
    # tensor product basis (std_basis \otimes std_basis). We'll make that explicit later on.
    idle_gate = np.eye(dim**2, dtype=np.complex128)
    lift_op_a = np.kron(op_a, idle_gate)
    lift_op_b = np.kron(op_b, idle_gate)

    # Now we'll compare these lifted operators by how they act on specific state in S2.
    # That state is rho_test = |psi><psi|, where
    #
    #   |psi> = (|00> + |11> + ... + |dim - n_leak - 1>) / sqrt(dim - n_leak).
    #
    rho_test = tensorized_teststate_density(dim, n_leak)

    # Of course, lift_op_a and lift_op_b only act on states in their superket representations.
    # We need the superket representation of rho_test in terms of the tensor product basis for S2.
    #
    # Luckily, pyGSTi has a class for generating bases for a tensor-product space given
    # bases for the constituent spaces appearing in the tensor product.
    ten_basis = TensorProdBasis((std_basis, std_basis))
    rho_test_superket = pgbt.stdmx_to_vec(rho_test, ten_basis).ravel()

    temp1 = lift_op_a @ rho_test_superket
    temp2 = lift_op_b @ rho_test_superket

    return temp1, temp2, ten_basis


def leading_dxd_submatrix_basis_vectors(d: int, n: int, current_basis):
    """
    Let "H" denote n^2 dimensional Hilbert-Schdmit space, and let "U" denote the d^2
    dimensional subspace of H spanned by vectors whose Hermitian matrix representations
    are zero outside the leading d-by-d submatrix.

    This function returns a column-unitary matrix "B" where P = B B^{\dagger} is the
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
    label2ind = {ell: idx for idx, ell in enumerate(std_basis.labels)}
    basis_ind = []
    for i in range(d):
        for j in range(d):
            ell = f"({i},{j})"
            basis_ind.append(label2ind[ell])
    basis_ind = np.array(basis_ind)
    submatrix_basis_vectors = X[:, basis_ind]
    return submatrix_basis_vectors


def leaky_entanglement_fidelity(op_a, op_b, mx_basis, n_leak=0):
    temp1, temp2, _ = apply_tensorized_to_teststate(op_a, op_b, mx_basis, n_leak)
    ent_fid = np.real(temp1.conj() @ temp2)
    return ent_fid


def leaky_jtracedist(op_a, op_b, mx_basis, n_leak=0):
    temp1, temp2, ten_basis = apply_tensorized_to_teststate(op_a, op_b, mx_basis, n_leak)
    temp1_std = pgbt.vec_to_stdmx(temp1, ten_basis, keep_complex=True)
    temp2_std = pgbt.vec_to_stdmx(temp2, ten_basis, keep_complex=True)
    j_dist = pgot.tracedist(temp1_std, temp2_std)
    return j_dist


# MARK: model construction

def to_3level_unitary(U_2level):
    U_3level = np.zeros((3, 3), complex)
    U_3level[0:2, 0:2] = U_2level
    U_3level[2, 2] = 1.0
    return U_3level


def leaky_qubit_model_from_pspec(ps_2level: QubitProcessorSpec, levels_readout_zero=(0,)):
    from pygsti.models.explicitmodel import ExplicitOpModel
    from pygsti.baseobjs.statespace import ExplicitStateSpace
    assert ps_2level.num_qubits == 1
    
    Us = ps_2level.gate_unitaries
    rho0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], complex)
    E0   = np.zeros((3, 3))
    E0[levels_readout_zero, levels_readout_zero] = 1
    E1   = np.eye(3, dtype=complex) - E0

    ss = ExplicitStateSpace([0],[3])
    tm_3level = ExplicitOpModel(ss, 'gm')
    tm_3level['rho0'] =  pgbt.stdmx_to_gmvec(rho0)
    tm_3level['Mdefault'] = TPPOVM(
        [("0", pygsti.tools.stdmx_to_gmvec(E0)), ("1", pygsti.tools.stdmx_to_gmvec(E1))], evotype="default",
    )

    def u2x2_to_9x9_gm_superoperator(u2x2):
        u3x3 = np.eye(3, dtype=np.complex128)
        u3x3[:2,:2] = u2x2
        superop_std = pgot.unitary_to_std_process_mx(u3x3)
        superop_gm = pgbt.change_basis(superop_std, 'std', 'gm')
        return superop_gm

    for gatename, unitary in Us.items():
        gatekey = (gatename, 0) if gatename != '{idle}' else ('Gi',0)
        tm_3level[gatekey] = u2x2_to_9x9_gm_superoperator(unitary)

    tm_3level.sim = 'map'  # can use 'matrix', if that's preferred for whatever reason.
    return tm_3level


# MARK: gauge optimization

def std_lago_gopsuite(model):
    from pygsti.protocols.gst import GSTGaugeOptSuite
    temp_gos = GSTGaugeOptSuite(gaugeopt_suite_names=('stdgaugeopt',))
    gop_params = temp_gos.to_dictionary(model)
    gop_params   = {'LAGO':
        [gp for gp in gop_params['stdgaugeopt'] if 'TPSpam' not in str(type(gp['gauge_group']))]
    }
    for inner_dict in gop_params['LAGO']:
        inner_dict['method'] = 'L-BFGS-B'
        inner_dict['n_leak'] = 1
        inner_dict['gates_metric'] = 'frobenius squared'
        inner_dict['spam_metric']  = 'frobenius squared'
        inner_dict['convert_model_to']['to_type'] = 'full'
    return gop_params


def lago_gaugeopt_params(existing_est):
    gop_params   = existing_est.goparameters
    gop_params   = copy.deepcopy(gop_params)
    gop_params   = {'LAGO':
        [gp for gp in gop_params['stdgaugeopt'] if 'TPSpam' not in str(type(gp['_gaugeGroupEl']))]
    }
    for inner_dict in gop_params['LAGO']:
        inner_dict['method'] = 'L-BFGS-B'
        inner_dict['n_leak'] = 1
        inner_dict['gates_metric'] = 'frobenius squared'
        inner_dict['spam_metric']  = 'frobenius squared'
        inner_dict['convert_model_to']['to_type'] = 'full'
    return gop_params


def transform_composed_model(mdl, s):
    """
    Gauge transform this model.

    Update each of the operation matrices G in this model with inv(s) * G * s,
    each rhoVec with inv(s) * rhoVec, and each EVec with EVec * s

    Parameters
    ----------
    s : GaugeGroupElement
        A gauge group element which specifies the "s" matrix
        (and it's inverse) used in the above similarity transform.

    Returns
    -------
    ExplicitOpModel
    """
    oldmdl = mdl
    mdl = oldmdl.copy()

    from pygsti.modelmembers.operations import ComposedOp, StaticArbitraryOp
    from pygsti.modelmembers.povms import ComposedPOVM
    from pygsti.modelmembers.states import ComposedState

    U    = StaticArbitraryOp(s.transform_matrix,         basis=oldmdl.basis)
    invU = StaticArbitraryOp(s.transform_matrix_inverse, basis=oldmdl.basis) 

    for key, rho in oldmdl.preps.items():
        assert isinstance(rho, ComposedState)
        static_rho = rho.state_vec
        errmap  = ComposedOp([rho.error_map, invU])
        mdl.preps[key] = ComposedState(static_rho, errmap)

    for key, povm in oldmdl.povms.items():
        assert isinstance(povm, ComposedPOVM)
        static_povm = povm.base_povm
        errmap = ComposedOp([U, povm.error_map])
        mdl.povms[key] = ComposedPOVM(errmap, static_povm, mx_basis=oldmdl.basis)

    for key, op in oldmdl.operations.items():
        op_s = ComposedOp([U, op, invU])
        mdl.operations[key] = op_s

    assert len(oldmdl.factories) == 0
    assert len(oldmdl.instruments) == 0

    mdl._clean_paramvec()  # transform may leave dirty members
    return mdl


def param_preserving_gauge_opt(gop_params_dict, results: ModelEstimateResults, est_key: str, verbosity: int = 0):
    from pygsti.protocols.gst import _add_gauge_opt
    est = results.estimates[est_key]
    seed_mdl = est.models['final iteration estimate']
    _add_gauge_opt(results, est_key, gop_params_dict, seed_mdl, verbosity=verbosity)
    for gop_name in gop_params_dict.keys():
        ggel = est._gaugeopt_suite.gaugeopt_argument_dicts[gop_name]['_gaugeGroupEl']
        model_implicit_gauge = transform_composed_model(est.models['final iteration estimate'], ggel)
        est.models[gop_name] = model_implicit_gauge
    return


def add_lago_model(results: ModelEstimateResults, est_key: str, verbosity: int = 0):
    existing_est = results.estimates[est_key]
    gop_params = lago_gaugeopt_params(existing_est)
    param_preserving_gauge_opt(gop_params, results, est_key, verbosity)
    return


# MARK: reports

def changebasis_3level_model(mdl, leakage_basis=None):
    """ 
    Create a copy of "mdl" where attached modelmembers are unconstrained
    and expressed in the leakage-friendly basis.

    This is needed because some modelmember classes (like TPPOVM) require
    that the identity matrix is an element of our basis for Hilbert-Schmidt
    space, and the leakage-friendly basis doesn't have that property.
    """
    if leakage_basis is None:
        leakage_basis = pgbt.leakage_friendly_basis_2plus1()

    new_mdl = mdl.copy()
    if mdl.basis.name == 'LeakageBasis':
        return new_mdl
    gm_basis = mdl.basis
    
    rho = mdl.preps["rho0"].to_dense()
    rho_new = pgbt.change_basis(rho, gm_basis, leakage_basis)
    new_mdl.preps["rho0"] = pgmm.states.FullState(rho_new)

    M0 = mdl.povms["Mdefault"]["0"].to_dense()
    M1 = mdl.povms["Mdefault"]["1"].to_dense()
    new_mdl.povms["Mdefault"] = pgmm.povms.UnconstrainedPOVM(
        [
            ("0", pgbt.change_basis(M0, gm_basis, leakage_basis)),
            ("1", pgbt.change_basis(M1, gm_basis, leakage_basis)),
        ],
        evotype="default",
    )

    for lbl, op in mdl.operations.items():
        op = op.to_dense()
        op_new = pgbt.change_basis(op, gm_basis, leakage_basis)
        new_mdl.operations[lbl] = pgmm.operations.FullArbitraryOp(op_new)
    new_mdl.basis = leakage_basis
    return new_mdl


def changebasis_3level_results(results : ModelEstimateResults):
    """ 
    Return a copy of "results" that changes the basis for every Model within "results"
    into the leakage-friendly basis.
    
    This is needed for report generation.
    """
    results = results.copy()
    leakage_basis = pgbt.leakage_friendly_basis_2plus1()
    for estlbl, est in results.estimates.items():
        for mlbl, mdl in est.models.items():
            if isinstance(mdl, (list, tuple)):  # assume a list/tuple of models
                new_mdl = [changebasis_3level_model(m, leakage_basis) for m in mdl]
            else:
                new_mdl = changebasis_3level_model(mdl, leakage_basis)
            est.models[mlbl] = new_mdl
        for gopsuitedict_or_list_thereof in est.goparameters.values():
            if not isinstance(gopsuitedict_or_list_thereof, list):
                gopsuite_listofdicts = [gopsuitedict_or_list_thereof]
            else:
                gopsuite_listofdicts = gopsuitedict_or_list_thereof
            for gopsuite_dict in gopsuite_listofdicts:
                m = gopsuite_dict['target_model']
                m = changebasis_3level_model(m, leakage_basis)
                gopsuite_dict['target_model'] = m
        results.estimates[estlbl] = est
    return results


def write_leakage_friendly_html_report(
        report_title: str,
        report_dir  : str,
        results : Union[ModelEstimateResults, Dict[str,ModelEstimateResults]],
        est_key: str ='KiteGST',
        gop_verbosity : int = 0
    ):
    if isinstance(results, ModelEstimateResults):
        add_lago_model(results, est_key, gop_verbosity)
        results_lfb = results
    else:
        assert isinstance(results, dict)
        assert len(results) > 0
        results_lfb = dict()
        for k, v in results.items():
            crf = v.estimates[est_key].add_confidence_region_factory('LAGO', 'final')
            crf.compute_hessian()
            crf_mdl = crf.parent.models[crf.model_lbl]
            nongauge_space, gauge_space = crf_mdl.compute_nongauge_and_gauge_spaces(
                {'spam': 0.0, 'gates': 1.0}
            )
            crf.hessian = crf._project_hessian(crf.hessian, nongauge_space, gauge_space, crf.jacobian)
            crf.project_hessian('none')
            results_lfb[k] = v

    from pygsti.report import construct_standard_report
    advanced_opts = {
        'n_leak': 1,
        'skip_sections' : ['variantdecomp' , 'varianterrorgen' ]
    }
    report = construct_standard_report(results_lfb, title=report_title, confidence_level=95, advanced_options=advanced_opts)
    report.write_html(report_dir)
    return


# MARK: kites

def project_qutrit_to_kite_model(model, negeigtol=1e-7):
    model = changebasis_3level_model(model)
    model.convert_members_inplace('full')

    lfb = model.basis
    expect_labels  = ['I', 'X', 'Y', 'Z', 'L']
    expect_indices = [0, 1, 2, 3, -1]
    for lbl, ind in zip(expect_labels, expect_indices):
        assert lfb.labels[ind] == lbl

    P = np.zeros((9,9))
    P[expect_indices, expect_indices] = 1.0

    for mm in model.preps.values():
        mm.set_dense( P @ mm.to_dense() )

    for mm in model.operations.values():
        Op = P @ mm.to_dense() @ P
        mm.set_dense( Op )
        J = pygsti.tools.jamiolkowski_iso(Op, lfb, lfb)
        e = np.linalg.eigvalsh(J)
        assert np.all(e > -negeigtol), f"min(e) = {np.min(e)}"

    for mm in model.povms.values():
        for elbl, e in mm.items():
            if elbl == mm.complement_label:
                continue
            e.set_dense(P @ e.to_dense())

    return model


def project_kite_results(results : ModelEstimateResults):
    """ 
    Return a copy of "results" that which projects all of its constituent models to a kite form.

    This is needed for report generation.
    """
    results = results.copy()
    for estlbl, est in results.estimates.items():
        for mlbl, mdl in est.models.items():
            if isinstance(mdl, (list, tuple)):  # assume a list/tuple of models
                new_mdl = [project_qutrit_to_kite_model(m) for m in mdl]
            else:
                new_mdl = project_qutrit_to_kite_model(mdl)
            est.models[mlbl] = new_mdl
        for gopsuitedict_or_list_thereof in est.goparameters.values():
            if not isinstance(gopsuitedict_or_list_thereof, list):
                gopsuite_listofdicts = [gopsuitedict_or_list_thereof]
            else:
                gopsuite_listofdicts = gopsuitedict_or_list_thereof
            for gopsuite_dict in gopsuite_listofdicts:
                m = gopsuite_dict['target_model']
                m = project_qutrit_to_kite_model(m)
                gopsuite_dict['target_model'] = m
        results.estimates[estlbl] = est
    return results


def kitified(m):
    from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp, ComposedOp, StaticArbitraryOp
    from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel
    from pygsti.modelmembers.states import ComposedState, StaticState
    from pygsti.modelmembers.povms import ComposedPOVM, StaticPOVMEffect, UnconstrainedPOVM
    lfb, cssb = pygsti.tools.basistools.leakage_friendly_basis_2plus1(return_subspace_basis=True)
    eegs = dict()
    eegs.update({LocalElementaryErrorgenLabel('H', be_lbl): 0.0 for be_lbl in cssb.labels })
    eegs.update({LocalElementaryErrorgenLabel('S', be_lbl): 0.0 for be_lbl in cssb.labels })
    eegs.update({LocalElementaryErrorgenLabel('C', [be_lbl1, be_lbl2]): 0.0 
                for be_lbl1 in cssb.labels for be_lbl2 in cssb.labels}
    )
    eegs.update({LocalElementaryErrorgenLabel('A', [be_lbl1, be_lbl2]): 0.0 
                for be_lbl1 in cssb.labels for be_lbl2 in cssb.labels}
    )
    ss = m.state_space
    LEG = LindbladErrorgen.from_elementary_errorgens(eegs, parameterization='CPTPLND', elementary_errorgen_basis=lfb, mx_basis=lfb, state_space=ss)
    
    def new_errorgen():
        return ExpErrorgenOp(LEG.copy())

    m_kite = m.copy()
    for k, op in m.operations.items():
        mat  = pgbt.change_basis(op.to_dense(), m.basis, lfb)
        kite_op = ComposedOp([StaticArbitraryOp(mat), new_errorgen()])
        m_kite.operations[k] = kite_op

    for k, rho in m.preps.items():
        vec = pgbt.change_basis(rho.to_dense(), m.basis, lfb)
        kite_rho = ComposedState(StaticState(vec, lfb), new_errorgen())
        m_kite.preps[k] = kite_rho
    
    for k, povm in m.povms.items():
        static_effects = dict()
        for lbl, eff in povm.items():
            vec = pgbt.change_basis(eff.to_dense(), m.basis, lfb)
            static_eff = StaticPOVMEffect(vec, lfb)
            static_effects[lbl] = static_eff
        static_povm = UnconstrainedPOVM(static_effects)
        kite_povm = ComposedPOVM(new_errorgen(), static_povm)
        m_kite.povms[k] = kite_povm

    m_kite.basis = lfb
    return m_kite
