import pygsti
from pygsti.extras import rb
import numpy as np

def test_rb_theory():
    
    # Tests can create the Cliffords gateset, using std1Q_Cliffords.
    from pygsti.construction import std1Q_Cliffords
    gs_target = std1Q_Cliffords.gs_target

    # Tests rb.group. This tests we can successfully construct a MatrixGroup
    clifford_group = rb.group.construct_1Q_Clifford_group()

    depol_strength = 1e-3
    gs = gs_target.depolarize(gate_noise=depol_strength)

    # Tests AGI function and p_to_r with AGI.
    AGI = pygsti.tools.average_gate_infidelity(gs['Gc0'],gs_target['Gc0'])
    r_AGI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='AGI')
    assert(np.abs(AGI-r_AGI)<10**(-10))

    # Tests EI function and p_to_r with EI.
    EI = pygsti.tools.entanglement_infidelity(gs['Gc0'],gs_target['Gc0'])
    r_EI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='EI')
    assert(np.abs(EI-r_EI)<10**(-10))

    # Tests uniform-average AGI function and the r-prediction function with uniform-weighting 
    AGsI = rb.theory.gateset_infidelity(gs,gs_target,itype='AGI')
    r_AGI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='AGI')
    r_pred_AGI = rb.theory.predicted_RB_number(gs,gs_target,rtype='AGI')
    assert(np.abs(AGsI-r_AGI)<10**(-10))
    assert(np.abs(r_pred_AGI-r_AGI)<10**(-10))

    # Tests uniform-average EI function and the r-prediction function with uniform-weighting 
    AEI = rb.theory.gateset_infidelity(gs,gs_target,itype='EI')
    r_EI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='EI')
    r_pred_EI = rb.theory.predicted_RB_number(gs,gs_target,rtype='EI')
    assert(np.abs(AEI-r_EI)<10**(-10))
    assert(np.abs(AEI-r_pred_EI)<10**(-10))

    # Tests the transform to RB gauge, and RB gauge transformation generating functions.
    gs_in_RB_gauge = rb.theory.transform_to_rb_gauge(gs, gs_target, eigenvector_weighting=0.5)
    AEI = rb.theory.gateset_infidelity(gs,gs_target,itype='EI')
    assert(np.abs(AEI-r_EI)<10**(-10))


    from pygsti.construction import std1Q_XY
    gs_target = std1Q_XY.gs_target.copy()
    gs = gs_target.copy()

    Zrot_unitary = np.array([[1.,0.],[0.,np.exp(-1j*0.01)]])
    Zrot_channel = pygsti.unitary_to_pauligate(Zrot_unitary)

    for key in gs_target.gates.keys():
        gs.gates[key] = np.dot(Zrot_channel,gs_target.gates[key])

    gs_in_RB_gauge = rb.theory.transform_to_rb_gauge(gs, gs_target, eigenvector_weighting=0.5)

    # A test that the RB gauge transformation behaves as expected -- a gateset that does not
    # have r = infidelity in its initial gauge does have this in the RB gauge. This also
    # tests that the r predictions are working for not-all-the-Cliffords gatesets.

    AEI = rb.theory.gateset_infidelity(gs_in_RB_gauge,gs_target,itype='EI')
    r_pred_EI = rb.theory.predicted_RB_number(gs,gs_target,rtype='EI')
    assert(np.abs(AEI-r_pred_EI)<10**(-10))

    # Test that weighted-infidelities + RB error rates functions working.
    gs_target = std1Q_XY.gs_target.copy()
    gs = gs_target.copy()

    depol_strength_X = 1e-3
    depol_strength_Y = 3e-3

    lx =1.-depol_strength_X
    depmap_X = np.array([[1.,0.,0.,0.],[0.,lx,0.,0.],[0.,0.,lx,0.],[0,0.,0.,lx]])
    ly =1.-depol_strength_Y
    depmap_Y = np.array([[1.,0.,0.,0.],[0.,ly,0.,0.],[0.,0.,ly,0.],[0,0.,0.,ly]])
    gs.gates['Gx'] = np.dot(depmap_X,gs_target['Gx'])
    gs.gates['Gy'] = np.dot(depmap_Y,gs_target['Gy'])

    Gx_weight = 1
    Gy_weight = 2
    weights = {'Gx':Gx_weight,'Gy':Gy_weight}
    WAEI = rb.theory.gateset_infidelity(gs,gs_target,weights=weights,itype='EI')
    GxAEI = pygsti.tools.entanglement_infidelity(gs['Gx'],gs_target['Gx'])
    GyAEI = pygsti.tools.entanglement_infidelity(gs['Gy'],gs_target['Gy'])
    manual_WAEI = (Gx_weight*GxAEI + Gy_weight*GyAEI)/(Gx_weight + Gy_weight)
    # Checks that a manual weighted-average agrees with the function
    assert(abs(manual_WAEI-WAEI)<10**(-10))

    gs_in_RB_gauge = rb.theory.transform_to_rb_gauge(gs, gs_target, weights=weights,
                                                     eigenvector_weighting=0.5)
    WAEI = rb.theory.gateset_infidelity(gs_in_RB_gauge,gs_target,weights=weights,itype='EI')
    # Checks the predicted RB number function works with specified weights
    r_pred_EI = rb.theory.predicted_RB_number(gs,gs_target,weights=weights,rtype='EI')
    # Checks predictions agree with weighted-infidelity
    assert(abs(r_pred_EI-WAEI)<10**(-10))


    # -------------------------------------- #
    #   Tests for R-matrix related functions
    # -------------------------------------- #

    # Test constructing the R matrix in the simplest case
    gs_target = std1Q_Cliffords.gs_target
    clifford_group = rb.group.construct_1Q_Clifford_group()
    R = rb.theory.R_matrix(gs_target, clifford_group, group_to_gateset=None, weights=None)

    # Test constructing the R matrix for a group-subset gateset with weights
    from pygsti.construction import std1Q_XYI
    clifford_compilation = std1Q_XYI.clifford_compilation
    gs_target = std1Q_XYI.gs_target.copy()
    group_to_gateset = {'Gc0':'Gi','Gc16':'Gx','Gc21':'Gy'}
    weights = {'Gi':1.,'Gx':1,'Gy':1}
    clifford_group = rb.group.construct_1Q_Clifford_group()
    R = rb.theory.R_matrix(gs_target, clifford_group, group_to_gateset=group_to_gateset, weights=weights)

    # Tests the p-prediction function works, and that we get the correct predictions from the R-matrix.
    p = rb.theory.R_matrix_predicted_RB_decay_parameter(gs_target, clifford_group, 
                                                        group_to_gateset=group_to_gateset, 
                                                        weights=weights)
    assert(abs(p - 1.) < 10**(-10))
    depol_strength = 1e-3
    gs = gs_target.depolarize(gate_noise=depol_strength)
    p = rb.theory.R_matrix_predicted_RB_decay_parameter(gs, clifford_group, 
                                                        group_to_gateset=group_to_gateset, 
                                                        weights=weights)
    assert(abs(p - (1.0-depol_strength)) < 10**(-10))

    # Tests the exact RB ASPs function on a Clifford gateset. 
    gs_target = std1Q_Cliffords.gs_target
    gs = std1Q_Cliffords.gs_target.depolarize(depol_strength)
    m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=1000, m_min=0, m_step=100, success_outcomelabel=('0',), 
                                      group_to_gateset=None, weights=None, compilation=None, 
                                      group_twirled=False)
    assert(abs(ASPs[1]- (0.5 + 0.5*(1.0-depol_strength)**101)) < 10**(-10))
    m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=1000, m_min=0, m_step=100, success_outcomelabel=('0',), 
                                      group_to_gateset=None, weights=None, compilation=None, 
                                      group_twirled=True)
    assert(abs(ASPs[1]- (0.5 + 0.5*(1.0-depol_strength)**102)) < 10**(-10))

    # Tests the exact RB ASPs function on a subset-of-Cliffords gateset. 
    clifford_compilation = std1Q_XY.clifford_compilation
    gs_target = std1Q_XY.gs_target.copy()
    group_to_gateset = {'Gc16':'Gx','Gc21':'Gy'}
    weights = {'Gx':5,'Gy':10}
    m, ASPs = rb.theory.exact_RB_ASPs(gs_target, clifford_group, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',), 
                                      group_to_gateset=group_to_gateset, weights=None, compilation=clifford_compilation, 
                                      group_twirled=False)
    assert(abs(np.sum(ASPs) - len(ASPs)) < 10**(-10))

    # Tests the function behaves reasonably with a depolarized gateset + works with group_twirled + weights.
    depol_strength = 1e-3
    gs = gs_target.depolarize(gate_noise=depol_strength)
    m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',), 
                                      group_to_gateset=group_to_gateset, weights=None, compilation=clifford_compilation, 
                                      group_twirled=False)
    assert(abs(ASPs[0] - 1) < 10**(-10))

    m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=10, m_min=0, m_step=3, success_outcomelabel=('0',), 
                                      group_to_gateset=group_to_gateset, weights=weights, compilation=clifford_compilation, 
                                      group_twirled=True)
    assert((ASPs > 0.99).all())


    # Check the L-matrix theory predictions work and are consistent with the exact predictions
    m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',), 
                                      group_to_gateset=group_to_gateset, weights=weights, compilation=clifford_compilation, 
                                      group_twirled=True)

    # Todo : change '1to1' to 'diamond' in 2 of 3 of the following, when diamonddist is working.
    L_m, L_ASPs, L_LASPs, L_UASPs = rb.theory.L_matrix_ASPs(gs, gs_target, m_max=10, m_min=0, m_step=1, 
                                                            success_outcomelabel=('0',), compilation=clifford_compilation,
                                                            group_twirled=True, weights=weights, gauge_optimize=True,
                                                            return_error_bounds=True, norm='1to1')

    assert((abs(ASPs-L_ASPs) < 0.001).all())

    # Check it works without the twirl, and gives plausable output
    L_m, L_ASPs = rb.theory.L_matrix_ASPs(gs, gs_target, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',),
                      compilation=clifford_compilation, group_twirled=False, weights=None, gauge_optimize=False, 
                      return_error_bounds=False, norm='1to1')
    assert((ASPs > 0.98).all())

    # Check it works with a Clifford gateset, and gives plausable output
    gs_target = std1Q_Cliffords.gs_target
    gs = std1Q_Cliffords.gs_target.depolarize(depol_strength)
    m, ASPs = rb.theory.L_matrix_ASPs(gs, gs_target, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',),
                      compilation=None, group_twirled=False, weights=None, gauge_optimize=False, 
                      return_error_bounds=False, norm='1to1')
    assert((ASPs > 0.98).all())