import pickle

import sys
import numpy as np
from ..util import BaseCase, with_temp_path
from pygsti.modelpacks import smq1Q_XYI as std
from pygsti.modelpacks import smq1Q_XY as std2
from pygsti.modelpacks import smq1Q_XZ as std3
from pygsti.baseobjs import Basis, CompleteElementaryErrorgenBasis
from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model
from pygsti.models import create_cloud_crosstalk_model_from_hops_and_weights, Model


class FogiTester(BaseCase):

    def check_std_fogi_param_count(self, model_type, errgen_types, expected_num_params, expected_num_fogi_params,
                                   expected_num_params_after_setup="same", include_spam=False, reparam=False):
        #Create model and setup FOGI decomp
        model = std.target_model(model_type)
        #target_model = std.target_model('static')
        self.assertEqual(model.num_params, expected_num_params)
        
        #basis = Basis.cast('pp', model.dim)
        basis1q = Basis.cast('pp', 4)
        gauge_basis = CompleteElementaryErrorgenBasis(
            basis1q, model.state_space, elementary_errorgen_types=errgen_types)

        op_abbrevs = {(): 'I',
                     ('Gxpi2', 0): 'Gx',
                     ('Gypi2', 0): 'Gy',
                     ('Gzpi2', 0): 'Gz'}
        model.setup_fogi(gauge_basis, None, op_abbrevs if model.dim == 4 else None,
                         reparameterize=reparam, dependent_fogi_action='drop', include_spam=include_spam)

        #Print FOGI error rates
        labels = model.fogi_errorgen_component_labels(include_fogv=True, typ='normal')
        raw_labels = model.fogi_errorgen_component_labels(include_fogv=True, typ='raw')
        coeffs = model.fogi_errorgen_components_array(include_fogv=True)
        #print("\n".join(["%d: %s = << %s >> = %g" % (i,lbl,raw,coeff)
        #                 for i,(lbl,raw,coeff) in enumerate(zip(labels, raw_labels, coeffs))]))
        #TODO - maybe test labels against known-correct ones?

        # Initialize random FOGI error rates
        base_model_error_strength = 1e-3
        np.random.seed(100)
        ar = model.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=reparam)
        ar = base_model_error_strength * (np.random.rand(len(ar)) - 0.5)
        model.set_fogi_errorgen_components_array(ar, include_fogv=False, normalized_elem_gens=reparam)

        nFOGI = len(ar)
        self.assertEqual(nFOGI, expected_num_fogi_params)
        #print(nFOGI, "FOGI error rates,", len(all_ar)-nFOGI, "gauge")

        if expected_num_params_after_setup == 'same':
            expected_num_params_after_setup = expected_num_params
        self.assertEqual(model.num_params, expected_num_params_after_setup)

        all_ar = model.fogi_errorgen_components_array(include_fogv=True, normalized_elem_gens=reparam)
        ar2 = model.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=reparam)
        self.assertArraysAlmostEqual(ar, ar2)
        self.assertArraysAlmostEqual(ar, all_ar[:nFOGI])

        coeff_dict = model.errorgen_coefficients()
        # TODO: test correctness of above call

        # Test that setting a each component individually works like it should
        include_fogv = True
        N = len(model.fogi_errorgen_component_labels(include_fogv))
        for i in range(N):
            ar = np.zeros(N, 'd')
            ar[i] = 1
            model.set_fogi_errorgen_components_array(ar, include_fogv)
            ar2 = model.fogi_errorgen_components_array(include_fogv)
            #print(i, np.linalg.norm(ar-ar2))
            self.assertArraysAlmostEqual(ar, ar2)

        if reparam:
            # Test if from_vector works too
            w = np.random.rand(model.num_params)
            w2 = model.to_vector().copy()
            self.assertEqual(len(w2), len(w))
            #w[0:nprefix] = 0 # zero out all unused params (these can be SPAM and can't be any value?)
            model.from_vector(w)
            w3 = model.to_vector()
            self.assertArraysAlmostEqual(w, w3)

    def test_std_fogi_GLND(self):
        model_type = "GLND" # "H+s"
        errgen_types = ('H', 'S', 'C', 'A')

        # 60 params, 12+12=24 are SPAM, so 36 gate params; 11 are gauge, 25 FOGI
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=60, expected_num_fogi_params=25,
                                        include_spam=False)

        # 60 params, 31 non-gauge (as in TP case: 12 gauge params in 3 + 4 + 12*3 = 43 total) -> 31 FOGI, 29 gauge
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=60, expected_num_fogi_params=31,
                                        include_spam=True)

        # 60 params, 12+12=24 are SPAM, so 36 gate params; 11 are gauge, 25 FOGI
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=60, expected_num_fogi_params=25,
                                        expected_num_params_after_setup=49,  # 24 SPAM + 25 FOGI
                                        include_spam=False, reparam=True)

        # 60 params, 31 non-gauge (as in TP case: 12 gauge params in 3 + 4 + 12*3 = 43 total) -> 31 FOGI, 29 gauge
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=60, expected_num_fogi_params=31,
                                        expected_num_params_after_setup=31,  # 31 FOGI
                                        include_spam=True, reparam=True)

    def test_std_fogi_GLND(self):
        model_type = "H+s"
        errgen_types = ('H', 'S')

        # 30 params, 6+6=12 are SPAM, so 18 gate params; 5 are gauge, 13 FOGI
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=30, expected_num_fogi_params=13,
                                        include_spam=False)

        # 30 params, 18 non-gauge = 18 FOGI, 12 gauge
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=30, expected_num_fogi_params=18,
                                        include_spam=True)

        # 30 params, 6+6=12 are SPAM, so 18 gate params; 5 are gauge, 13 FOGI
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=30, expected_num_fogi_params=13,
                                        expected_num_params_after_setup=25,  # 12 SPAM + 13 FOGI
                                        include_spam=False, reparam=True)

        # 30 params, 18 non-gauge = 18 FOGI, 12 gauge
        self.check_std_fogi_param_count(model_type, errgen_types,
                                        expected_num_params=30, expected_num_fogi_params=18,
                                        expected_num_params_after_setup=18,  # 18 FOGI
                                        include_spam=True, reparam=True)


    def test_crosstalk_free_fogi(self):
        nQubits = 2
        #pspec = QubitProcessorSpec(nQubits, ['Gxpi2', 'Gypi2', 'Gi'], geometry='line')
        pspec = QubitProcessorSpec(nQubits, ['Gxpi2', 'Gypi2', 'Gcnot'], availability={'Gcnot': [(0,1)]}, geometry='line')

        mdl = create_crosstalk_free_model(pspec, ideal_gate_type='H+s', independent_gates=True, implicit_idle_mode='only_global')
        mdl_no_fogi = mdl.copy()
        print(mdl.num_params, 'parameters')
        self.assertEqual(mdl.num_params, 54)

        # Perform FOGI analysis
        reparam = True
        basis1q = Basis.cast('pp', 4)
        gauge_basis = CompleteElementaryErrorgenBasis(
            basis1q, mdl.state_space, elementary_errorgen_types='HS')
        mdl.setup_fogi(gauge_basis, None, None, reparameterize=reparam, dependent_fogi_action='drop', include_spam=True)

        include_fogv = not reparam
        labels = mdl.fogi_errorgen_component_labels(include_fogv, typ='normal')
        raw_labels = mdl.fogi_errorgen_component_labels(include_fogv, typ='raw')
        coeffs = mdl.fogi_errorgen_components_array(include_fogv)
        #print("\n".join(["%d: %s = %g" % (i,lbl,coeff)
        #                 for i,(lbl,raw,coeff) in enumerate(zip(labels, raw_labels, coeffs))]))

        ar = mdl.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=True)
        nfogi = len(ar)
        self.assertEqual(nfogi, 48)

        ar = np.random.rand(len(ar))
        mdl.set_fogi_errorgen_components_array(ar, include_fogv=False, normalized_elem_gens=True)
        nprefix = mdl.num_params - nfogi  # reparameterization *prefixes* FOGI params with "unused" params
        self.assertEqual(nprefix, 0)  # because include_spam=True above

        temp = mdl.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=True)
        self.assertArraysAlmostEqual(temp, mdl.to_vector()[nprefix:])

        v = mdl.to_vector()  # just test this works

        # Test if from_vector works too
        w = np.random.rand(mdl.num_params)
        w[0:nprefix] = 0 # zero out all unused params (these can be SPAM and can't be any value?)
        mdl.from_vector(w)
        pass


    def test_cloud_crosstalk_fogi(self):
        nQubits = 1
        pspec = QubitProcessorSpec(nQubits, ['Gxpi2', 'Gypi2', 'Gi' ],  # 'Gcnot'
                                   #availability={'Gcnot': [(0,1)]},  # to match smq2Q_XYCNOT
                                   geometry='line')
        mdl = create_cloud_crosstalk_model_from_hops_and_weights(pspec,
                                                                 max_idle_weight=1, max_spam_weight=2,
                                                                 extra_gate_weight=1, maxhops=1,
                                                                 gate_type='H+s', spam_type='H+s',
                                                                 connected_highweight_errors=False)
        print(mdl.num_params)
        self.assertEqual(mdl.num_params, 30)

        reparam = True
        basis1q = Basis.cast('pp', 4)
        gauge_basis = CompleteElementaryErrorgenBasis(
            basis1q, mdl.state_space, elementary_errorgen_types='HS')
        mdl.setup_fogi(gauge_basis, None, None, reparameterize=reparam, dependent_fogi_action='drop', include_spam=True)

        include_fogv = True
        labels = mdl.fogi_errorgen_component_labels(include_fogv, typ='normal')
        raw_labels = mdl.fogi_errorgen_component_labels(include_fogv, typ='raw')
        coeffs = mdl.fogi_errorgen_components_array(include_fogv)
        #print("\n".join(["%d: %s = %g" % (i,lbl,coeff)
        #                 for i,(lbl,raw,coeff) in enumerate(zip(labels, raw_labels, coeffs))]))
        ##print("\n".join(["%d: %s = << %s >> = %g" % (i,lbl,raw,coeff)
        ##                 for i,(lbl,raw,coeff) in enumerate(zip(labels, raw_labels, coeffs))]))

        ar = mdl.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=True)
        nfogi = len(ar)
        self.assertEqual(nfogi, 18)

        ar = np.random.rand(len(ar))
        mdl.set_fogi_errorgen_components_array(ar, include_fogv=False, normalized_elem_gens=True)
        nprefix = mdl.num_params - nfogi  # reparameterization *prefixes* FOGI params with "unused" params

        self.assertEqual(nprefix, 0)  # because include_spam=True above
        temp = mdl.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=True)
        self.assertArraysAlmostEqual(temp, mdl.to_vector()[nprefix:])

        v = mdl.to_vector()  # just test this works

        # Test if from_vector works too
        w = np.random.rand(mdl.num_params)
        w[0:nprefix] = 0 # zero out all unused params (these can be SPAM and can't be any value?)
        mdl.from_vector(w)

    def test_equal_method(self):

        def equal_fogi_models(fogi_model, fogi_model2):
            return fogi_model.fogi_store.__eq__(fogi_model2.fogi_store) and fogi_model.param_interposer.__eq__(fogi_model2.param_interposer)
        
        model = std.target_model('GLND')
        model2 = std2.target_model('GLND')
        model3 = std3.target_model('GLND')

        basis1q = Basis.cast('pp', 4)
        gauge_basis = CompleteElementaryErrorgenBasis(
            basis1q, model.state_space, elementary_errorgen_types='HSCA')
        gauge_basis2 = CompleteElementaryErrorgenBasis(
            basis1q, model2.state_space, elementary_errorgen_types='HSCA')
        gauge_basis3 = CompleteElementaryErrorgenBasis(
            basis1q, model3.state_space, elementary_errorgen_types='HSCA')
        
        model.setup_fogi(gauge_basis, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)
        model2.setup_fogi(gauge_basis2, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)
        model3.setup_fogi(gauge_basis3, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)
        
        msg = 'FOGI models that are the same are identified as different by __eq__ methods'
        self.assertTrue(equal_fogi_models(model, model), msg=msg)
        self.assertTrue(equal_fogi_models(model2, model2), msg=msg)
        self.assertTrue(equal_fogi_models(model3, model3), msg=msg)

        msg = 'FOGI models that are different are not recognized as different by __eq__ methods'
        self.assertFalse(equal_fogi_models(model, model2), msg=msg)
        self.assertFalse(equal_fogi_models(model, model3), msg=msg)
        self.assertFalse(equal_fogi_models(model2, model3), msg=msg)

    #TODO: should this be in test_nice_serialization instead?
    @with_temp_path
    def test_fogi_serialization(self, temp_pth):
        def equal_fogi_models(fogi_model, fogi_model2):
            return fogi_model.fogi_store.__eq__(fogi_model2.fogi_store) and fogi_model.param_interposer.__eq__(fogi_model2.param_interposer)
        
        model = std.target_model('GLND')
        model2 = std2.target_model('GLND')
        model3 = std3.target_model('GLND')

        basis1q = Basis.cast('pp', 4)
        gauge_basis = CompleteElementaryErrorgenBasis(
            basis1q, model.state_space, elementary_errorgen_types='HSCA')
        gauge_basis2 = CompleteElementaryErrorgenBasis(
            basis1q, model2.state_space, elementary_errorgen_types='HSCA')
        gauge_basis3 = CompleteElementaryErrorgenBasis(
            basis1q, model3.state_space, elementary_errorgen_types='HSCA')
        
        model.setup_fogi(gauge_basis, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)
        model2.setup_fogi(gauge_basis2, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)
        model3.setup_fogi(gauge_basis3, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)

        model.write(temp_pth + '.json')
        loaded_model = Model.read(temp_pth + '.json')
        model2.write(temp_pth + '.json')
        loaded_model2 = Model.read(temp_pth + '.json')
        model3.write(temp_pth + '.json')
        loaded_model3 = Model.read(temp_pth + '.json')

        self.assertTrue(equal_fogi_models(model, loaded_model))
        self.assertTrue(equal_fogi_models(model2, loaded_model2))
        self.assertTrue(equal_fogi_models(model3, loaded_model3))
        
    def test_set_param_values(self):
        
        model = std.target_model('GLND')
        cp = model.copy()

        basis1q = Basis.cast('pp', 4)
        gauge_basis = CompleteElementaryErrorgenBasis(
            basis1q, model.state_space, elementary_errorgen_types='HSCA')

        model.setup_fogi(gauge_basis, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)
        cp.setup_fogi(gauge_basis, None, None, reparameterize=True, dependent_fogi_action='drop', include_spam=True)

        test_vec = np.arange(model.num_params) * 1e-3
        cp.set_parameter_values(np.arange(model.num_params), test_vec)
        model.from_vector(test_vec)
        self.assertAlmostEqual(np.linalg.norm(model.to_vector() - cp.to_vector()), 0)
        #TODO: Uncomment when issue #600 is resolved, remove line above
        #self.assertTrue(cp.is_equivalent(cp2))
        
        


