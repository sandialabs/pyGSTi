import unittest
import warnings
import collections
import pickle
import pygsti
import os
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..testutils import BaseTestCase, compare_files, temp_files

import numpy as np

from pygsti.report import reportables as rptbl

class TestReportables(BaseTestCase):

    def setUp(self):
        super(TestReportables, self).setUp()

    def test_helpers(self):
        self.assertTrue(rptbl._null_fn("Any arguments") is None)

        self.assertAlmostEqual(rptbl._project_to_valid_prob(-0.1), 0.0)
        self.assertAlmostEqual(rptbl._project_to_valid_prob(1.1), 1.0)
        self.assertAlmostEqual(rptbl._project_to_valid_prob(0.5), 0.5)

        nan_qty = rptbl.evaluate(None) # none function -> nan qty
        self.assertTrue( np.isnan(nan_qty.value) )

        #deprecated:
        rptbl.decomposition( std.target_model().operations['Gx'] )
        rptbl.decomposition( np.zeros( (4,4), 'd') )

    def test_functions(self):

        gs1 = std.target_model().depolarize(op_noise=0.1, spam_noise=0.05)
        gs2 = std.target_model()
        gl = "Gx" # operation label
        opstr = pygsti.obj.Circuit( ('Gx','Gx') )
        syntheticIdles = pygsti.construction.circuit_list( [
             ('Gx',)*4, ('Gy',)*4 ] )

        gatesetfn_factories = (  # model, oplabel
            rptbl.Choi_matrix,
            rptbl.Choi_evals,
            rptbl.Choi_trace,
            rptbl.GateEigenvalues, #GAP
            rptbl.Upper_bound_fidelity ,
            rptbl.Closest_ujmx,
            rptbl.Maximum_fidelity,
            rptbl.Maximum_trace_dist,

        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gl)
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model, circuit
            rptbl.CircuitEigenvalues,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,opstr)
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model_a, model_b, circuit
            rptbl.Rel_circuit_eigenvalues,
            rptbl.Circuit_fro_diff ,
            rptbl.Circuit_entanglement_infidelity,
            rptbl.Circuit_avg_gate_infidelity,
            rptbl.Circuit_jt_diff,
            rptbl.CircuitHalfDiamondNorm,
            rptbl.Circuit_nonunitary_entanglement_infidelity,
            rptbl.Circuit_nonunitary_avg_gate_infidelity,
            rptbl.Circuit_eigenvalue_entanglement_infidelity,
            rptbl.Circuit_eigenvalue_avg_gate_infidelity,
            rptbl.Circuit_eigenvalue_nonunitary_entanglement_infidelity,
            rptbl.Circuit_eigenvalue_nonunitary_avg_gate_infidelity,
            rptbl.Circuit_eigenvalue_diamondnorm,
            rptbl.Circuit_eigenvalue_nonunitary_diamondnorm,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gs2,opstr)
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model_a, model_b, povmlbl
            rptbl.POVM_entanglement_infidelity,
            rptbl.POVM_jt_diff,
            rptbl.POVM_half_diamond_norm,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gs2,"Mdefault")
            rptbl.evaluate(gsf)


        gatesetfn_factories = (  # model
            rptbl.Spam_dotprods,
            rptbl.Angles_btwn_rotn_axes,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1)
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model_a, model_b, gatelbl
            rptbl.Entanglement_fidelity,
            rptbl.Entanglement_infidelity,
            rptbl.Closest_unitary_fidelity,
            rptbl.Fro_diff,
            rptbl.Jt_diff,
            rptbl.HalfDiamondNorm,
            rptbl.Nonunitary_entanglement_infidelity,
            rptbl.Nonunitary_avg_gate_infidelity,
            rptbl.Eigenvalue_nonunitary_entanglement_infidelity,
            rptbl.Eigenvalue_nonunitary_avg_gate_infidelity,
            rptbl.Eigenvalue_entanglement_infidelity,
            rptbl.Eigenvalue_avg_gate_infidelity,
            rptbl.Eigenvalue_diamondnorm,
            rptbl.Eigenvalue_nonunitary_diamondnorm,
            rptbl.Avg_gate_infidelity,
            rptbl.Model_model_angles_btwn_axes,
            rptbl.Rel_eigvals,
            rptbl.Rel_logTiG_eigvals,
            rptbl.Rel_logGTi_eigvals,
            rptbl.Rel_logGmlogT_eigvals,
            rptbl.Rel_gate_eigenvalues,
            rptbl.LogTiG_and_projections,
            rptbl.LogGTi_and_projections,
            rptbl.LogGmlogT_and_projections,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gs2,gl)
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model_a, model_b, synthetic_idle_strs
            rptbl.Robust_LogGTi_and_projections,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gs2, syntheticIdles )
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model_a, model_b
            rptbl.General_decomposition,
            rptbl.Average_gateset_infidelity,
            rptbl.Predicted_rb_number,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gs2)
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model1, model2, label, typ
            rptbl.Vec_fidelity,
            rptbl.Vec_infidelity,
            rptbl.Vec_tr_diff,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,gs2,"rho0","prep")
            rptbl.evaluate(gsf)
            gsf = gsf_factory(gs1,gs2,"Mdefault:0","effect")
            rptbl.evaluate(gsf)


        gatesetfn_factories = ( # model, label, typ
            rptbl.Vec_as_stdmx,
            rptbl.Vec_as_stdmx_eigenvalues,
        )
        for gsf_factory in gatesetfn_factories:
            gsf = gsf_factory(gs1,"rho0","prep")
            rptbl.evaluate(gsf)
            gsf = gsf_factory(gs1,"Mdefault:0","effect")
            rptbl.evaluate(gsf)

    def test_nearby_gatesetfns(self):
        gs1 = std.target_model().depolarize(op_noise=0.1, spam_noise=0.05)
        gs2 = std.target_model()
        opstr = pygsti.obj.Circuit( ('Gx','Gx') )

        fn = rptbl.HalfDiamondNorm(gs1,gs2,'Gx')
        if fn is not None:
            fn.evaluate(gs1)
            fn.evaluate_nearby(gs1)
        else:
            warnings.warn("Can't test HalfDiamondNorm! (probably b/c cvxpy isn't available)")

        fn = rptbl.CircuitHalfDiamondNorm(gs1,gs2,opstr)
        if fn is not None:
            fn.evaluate(gs1)
            fn.evaluate_nearby(gs1)
        else:
            warnings.warn("Can't test CircuitHalfDiamondNorm! (probably b/c cvxpy isn't available)")

    def test_closest_unitary(self):
        gs1 = std.target_model().depolarize(op_noise=0.1, spam_noise=0.05)
        gs2 = std.target_model()
        rptbl.closest_unitary_fidelity(gs1.operations['Gx'], gs2.operations['Gx'], "pp") # op2 is unitary
        rptbl.closest_unitary_fidelity(gs2.operations['Gx'], gs1.operations['Gx'], "pp") # op1 is unitary

    def test_general_decomp(self):
        gs1 = std.target_model().depolarize(op_noise=0.1, spam_noise=0.05)
        gs2 = std.target_model()
        gs1.operations['Gx'] = np.array( [[-1, 0, 0, 0],
                                     [ 0,-1, 0, 0],
                                     [ 0, 0, 1, 0],
                                     [ 0, 0, 0, 1]], 'd') # -1 eigenvalues => use approx log.
        rptbl.general_decomposition(gs1,gs2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
