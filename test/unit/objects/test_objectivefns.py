import numpy as np

import pygsti
import pygsti.objectivefns.objectivefns as _objfns
from pygsti.objectivefns.wildcardbudget import PrimitiveOpsWildcardBudget as _PrimitiveOpsWildcardBudget
from . import smqfixtures
from ..util import BaseCase
import unittest


class ObjectiveFunctionData(object):
    """
    Common data for objective function tests
    """

    def setUp(self):
        self.model = smqfixtures.ns.datagen_model.copy()
        self.circuits = smqfixtures.ns.circuits
        self.dataset = smqfixtures.ns.dataset.copy()
        self.sparse_dataset = smqfixtures.ns.sparse_dataset.copy()
        self.perfect_dataset = smqfixtures.ns.perfect_dataset.copy()

        self.aliases = smqfixtures.ns.aliases.copy()
        self.alias_model = smqfixtures.ns.alias_datagen_model.copy()
        self.alias_circuits = smqfixtures.ns.alias_circuits


class ObjectiveFunctionUtilTester(ObjectiveFunctionData, BaseCase):
    """
    Tests for functions in the objectivefns module.
    """

    def test_objfn(self):
        fn = _objfns._objfn(_objfns.Chi2Function, self.model, self.dataset, self.circuits)
        self.assertTrue(isinstance(fn, _objfns.Chi2Function))

        fn = _objfns._objfn(_objfns.PoissonPicDeltaLogLFunction, self.model, self.dataset, self.circuits,
                            regularization={'min_prob_clip': 1e-3}, penalties={'regularize_factor': 1.0})
        self.assertTrue(isinstance(fn, _objfns.PoissonPicDeltaLogLFunction))

        #Test with circuits=None
        fn = _objfns._objfn(_objfns.PoissonPicDeltaLogLFunction, self.model, self.dataset, None)
        self.assertEqual(list(fn.circuits), list(self.dataset.keys()))

        #Test with aliases
        fn = _objfns._objfn(_objfns.PoissonPicDeltaLogLFunction, self.alias_model, self.dataset,
                            self.alias_circuits, op_label_aliases=self.aliases)
        self.assertTrue(isinstance(fn, _objfns.PoissonPicDeltaLogLFunction))


class ObjectiveFunctionBuilderTester(ObjectiveFunctionData, BaseCase):
    """
    Tests for methods in the ObjectiveFunctionBuilder class.
    """

    def test_create_from(self):
        builder1 = _objfns.ObjectiveFunctionBuilder.create_from('chi2')
        builder2 = _objfns.ObjectiveFunctionBuilder.cast(builder1)
        self.assertTrue(builder1 is builder2)

        builder3 = _objfns.ObjectiveFunctionBuilder.cast('chi2')
        builder4 = _objfns.ObjectiveFunctionBuilder.cast({'objective': 'chi2', 'freq_weighted_chi2': True})
        builder5 = _objfns.ObjectiveFunctionBuilder.cast((_objfns.Chi2Function, 'name', 'description',
                                                          {'min_prob_clip_for_weighting': 1e-4}))

        fn = builder3.build(self.model, self.dataset, self.circuits)
        self.assertTrue(isinstance(fn, _objfns.Chi2Function))

        fn = builder4.build(self.model, self.dataset, self.circuits)
        self.assertTrue(isinstance(fn, _objfns.FreqWeightedChi2Function))

        fn = builder5.build(self.model, self.dataset, self.circuits)
        self.assertTrue(isinstance(fn, _objfns.Chi2Function))

    def test_simple_builds(self):
        for objective in ('logl', 'chi2', 'tvd'):
            builder = _objfns.ObjectiveFunctionBuilder.create_from(objective)
            fn = builder.build(self.model, self.dataset, self.circuits)
            self.assertTrue(isinstance(fn, builder.cls_to_build))

        builder = _objfns.ObjectiveFunctionBuilder.create_from('chi2', True)
        fn = builder.build(self.model, self.dataset, self.circuits)
        self.assertTrue(isinstance(fn, builder.cls_to_build))


class RawObjectiveFunctionTesterBase(object):
    """
    Tests for methods in the RawObjectiveFunction class.
    """

    @staticmethod
    def build_objfns(cls):
        raise NotImplementedError()

    @classmethod
    def setUpClass(cls):
        cls.objfns = cls.build_objfns(cls)
        cls.perfect_probs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'd')
        cls.counts = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'd')
        cls.totalcounts = np.array([100] * len(cls.counts), 'd')
        cls.freqs = cls.counts / cls.totalcounts

        cls.probs = np.array([0.05, 0.05, 0.25, 0.25, 0.45, 0.45, 0.65, 0.65, 0.85, 0.85, 0.95], 'd')  # "good" probs
        cls.bad_probs = np.array([1.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, -0.1], 'd')

    def test_value(self):
        for objfn in self.objfns:
            fn = objfn.fn(self.probs, self.counts, self.totalcounts, self.freqs)
            terms = objfn.terms(self.probs, self.counts, self.totalcounts, self.freqs)
            self.assertAlmostEqual(fn, sum(terms))

            if self.computes_lsvec:
                lsvec = objfn.lsvec(self.probs, self.counts, self.totalcounts, self.freqs)
                self.assertArraysAlmostEqual(terms, lsvec**2)

    def test_derivative(self):
        for objfn in self.objfns:
            jac = objfn.jacobian(self.probs, self.counts, self.totalcounts, self.freqs)
            dterms = objfn.dterms(self.probs, self.counts, self.totalcounts, self.freqs)

            eps = 1e-7
            fd_dterms = (objfn.terms(self.probs + eps, self.counts, self.totalcounts, self.freqs)
                         - objfn.terms(self.probs, self.counts, self.totalcounts, self.freqs)) / eps
            norm = np.maximum(np.abs(dterms), 1e-2) * dterms.size  # normalize so test per-element *relative* error
            self.assertArraysAlmostEqual(jac, dterms)  # the same b/c dterms assumes term_i only depends on p_i
            self.assertArraysAlmostEqual(dterms / norm, fd_dterms / norm, places=4)  # compare with finite-difference

            if self.computes_lsvec:
                lsvec = objfn.lsvec(self.probs, self.counts, self.totalcounts, self.freqs)
                dlsvec = objfn.dlsvec(self.probs, self.counts, self.totalcounts, self.freqs)
                dlsvec_chk, lsvec_chk = objfn.dlsvec_and_lsvec(self.probs, self.counts, self.totalcounts, self.freqs)

                fd_dlsvec = (objfn.lsvec(self.probs + eps, self.counts, self.totalcounts, self.freqs) - lsvec) / eps

                norm = np.maximum(np.abs(dlsvec), 1e-2) * dlsvec.size  # normalize so test per-element *relative* error
                self.assertArraysAlmostEqual(lsvec, lsvec_chk)
                self.assertArraysAlmostEqual(dlsvec, dlsvec_chk)
                self.assertArraysAlmostEqual(dlsvec / norm, fd_dlsvec / norm,
                                             places=4)  # compare with finite-difference
                self.assertArraysAlmostEqual(dterms, 2 * lsvec * dlsvec)  # d(terms) = d(lsvec**2) = 2*lsvec*dlsvec

    def test_hessian(self):
        for objfn in self.objfns:
            try:
                jac = objfn.hessian(self.probs, self.counts, self.totalcounts, self.freqs).copy()
                hterms = objfn.hterms(self.probs, self.counts, self.totalcounts, self.freqs).copy()
            except NotImplementedError:
                continue  # ok if hterms is not always implemented

            eps = 1e-7
            fd_hterms = (objfn.dterms(self.probs + eps, self.counts, self.totalcounts, self.freqs).copy()
                         - objfn.dterms(self.probs, self.counts, self.totalcounts, self.freqs).copy()) / eps
            norm = np.maximum(np.abs(hterms), 1e-2) * hterms.size  # normalize so test per-element *relative* error
            self.assertArraysAlmostEqual(jac, hterms)  # the same b/c dterms assumes term_i only depends on p_i
            self.assertArraysAlmostEqual(hterms / norm, fd_hterms / norm, places=4)  # compare with finite-difference

            if self.computes_lsvec:
                lsvec = objfn.lsvec(self.probs, self.counts, self.totalcounts, self.freqs).copy()
                dlsvec = objfn.dlsvec(self.probs, self.counts, self.totalcounts, self.freqs).copy()
                hlsvec = objfn.hlsvec(self.probs, self.counts, self.totalcounts, self.freqs).copy()
                fd_hlsvec = (objfn.dlsvec(self.probs + eps, self.counts, self.totalcounts, self.freqs) - dlsvec) / eps

                norm = np.maximum(np.abs(hlsvec), 1e-2) * hlsvec.size  # normalize so test per-element *relative* error
                self.assertArraysAlmostEqual(hlsvec / norm, fd_hlsvec / norm,
                                             places=4)  # compare with finite-difference

                self.assertArraysAlmostEqual(hterms, 2 * (dlsvec**2 + lsvec * hlsvec))
                # h(terms) = 2 * (dsvec**2 + lsvec * hlsvec)


class RawChi2FunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawChi2Function({'min_prob_clip_for_weighting': 1e-6}, resource_alloc)]


class RawChiAlphaFunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawChiAlphaFunction({'pfratio_stitchpt': 0.1, 'pfratio_derivpt': 0.1, 'fmin': 1e-4},
                                            resource_alloc),
                _objfns.RawChiAlphaFunction({'pfratio_stitchpt': 0.1, 'pfratio_derivpt': 0.1, 'radius': 0.001},
                                            resource_alloc)]

    def test_hessian(self):
        self.skipTest("Hessian for RawChiAlphaFunction isn't implemented yet.")


class RawFreqWeightedChi2FunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawFreqWeightedChi2Function({'min_freq_clip_for_weighting': 1e-4}, resource_alloc)]


class RawPoissonPicDeltaLogLFunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawPoissonPicDeltaLogLFunction({'min_prob_clip': 1e-6, 'radius': 0.001}, resource_alloc),
                _objfns.RawPoissonPicDeltaLogLFunction({'min_prob_clip': None, 'radius': None, 'pfratio_stitchpt': 0.1,
                                                        'pfratio_derivpt': 0.1, 'fmin': 1e-4}, resource_alloc)]


class RawDeltaLogLFunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = False

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawDeltaLogLFunction({'min_prob_clip': 1e-6}, resource_alloc),
                _objfns.RawDeltaLogLFunction({'min_prob_clip': None, 'pfratio_stitchpt': 0.1, 'pfratio_derivpt': 0.1},
                                             resource_alloc)]


class RawMaxLogLFunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = False

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawMaxLogLFunction({}, resource_alloc)]


class RawTVDFunctionTester(RawObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True

    @staticmethod
    def build_objfns(cls):
        resource_alloc = {'mem_limit': None, 'comm': None}
        return [_objfns.RawTVDFunction({}, resource_alloc)]

    def test_derivative(self):
        self.skipTest("Derivatives for RawTVDFunction aren't implemented yet.")

    def test_hessian(self):
        self.skipTest("Derivatives for RawTVDFunction aren't implemented yet.")


class TimeIndependentMDSObjectiveFunctionTesterBase(ObjectiveFunctionData):
    """
    Tests for methods in the TimeIndependentMDSObjectiveFunction class.
    """

    @staticmethod
    def build_objfns(cls):
        raise NotImplementedError()

    @classmethod
    def setUpClass(cls):
        cls.penalty_dicts = [
            {'prob_clip_interval': (-100.0, 100.0)},
            {'regularize_factor': 1e-2},
            {'cptp_penalty_factor': 1.0},
            {'spam_penalty_factor': 1.0},
        ]

    def setUp(self):
        super().setUp()
        self.objfns = self.build_objfns()

    def test_builder(self):
        #All objective function should be of the same type
        cls = self.objfns[0].__class__
        builder = cls.builder("test_name", "test_description")
        self.assertTrue(isinstance(builder, _objfns.ObjectiveFunctionBuilder))

    def test_value(self):
        for objfn in self.objfns:
            terms = objfn.terms().copy()

            if self.computes_lsvec:
                lsvec = objfn.lsvec().copy()
                self.assertArraysAlmostEqual(lsvec**2, terms)

    def test_derivative(self):
        for objfn in self.objfns:
            dterms = objfn.dterms().copy()

            eps = 1e-7
            fd_dterms = np.zeros(dterms.shape, 'd')
            v0 = self.model.to_vector()
            terms0 = objfn.terms(v0).copy()
            for i in range(len(v0)):
                v1 = v0.copy(); v1[i] += eps
                fd_dterms[:, i] = (objfn.terms(v1) - terms0) / eps
            nEls = dterms.size
            self.assertArraysAlmostEqual(dterms / nEls, fd_dterms / nEls,
                                         places=3)  # each *element* should match to 3 places

            if self.computes_lsvec:
                lsvec = objfn.lsvec().copy()
                dlsvec = objfn.dlsvec().copy()
                self.assertArraysAlmostEqual(dterms / nEls, 2 * lsvec[:, None] * dlsvec / nEls,
                                             places=4)  # each *element* should match to 4 places

    def test_approximate_hessian(self):
        if not self.enable_hessian_tests:
            return  # don't test the hessian for this objective function

        for objfn in self.objfns:
            hessian = objfn.approximate_hessian()
            #TODO: how to verify this hessian?

    def test_hessian(self):
        if not self.enable_hessian_tests:
            return  # don't test the hessian for this objective function

        for objfn in self.objfns:
            try:
                hessian = objfn.hessian().copy()
            except NotImplementedError:
                continue  # ok if hessian is not always implemented

            self.assertEqual(hessian.shape, (self.model.num_params, self.model.num_params))

            eps = 1e-7
            fd_hessian = np.zeros(hessian.shape, 'd')
            v0 = self.model.to_vector()
            summed_dterms0 = np.sum(objfn.dterms(v0).copy(), axis=0)  # a 1D array of 1st derivs
            for i in range(len(v0)):
                v1 = v0.copy(); v1[i] += eps
                summed_dterms1 = np.sum(objfn.dterms(v1).copy(), axis=0)
                fd_hessian[:, i] = (summed_dterms1 - summed_dterms0) / eps
            norm = np.maximum(np.abs(hessian), 1e-2) * hessian.size
            self.assertArraysAlmostEqual(hessian / norm, fd_hessian / norm, places=3)


class Chi2FunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True
    enable_hessian_tests = False

    def build_objfns(self):
        return [_objfns.Chi2Function.create_from(self.model, self.dataset, self.circuits, None, penalties, method_names=('terms', 'dterms'))
                for penalties in self.penalty_dicts]


class ChiAlphaFunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True
    enable_hessian_tests = False

    def build_objfns(self):
        return [_objfns.ChiAlphaFunction.create_from(self.model, self.dataset, self.circuits, {'fmin': 1e-4}, None, method_names=('terms', 'dterms'))]


class FreqWeightedChi2FunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True
    enable_hessian_tests = False

    def build_objfns(self):
        return [_objfns.FreqWeightedChi2Function.create_from(self.model, self.dataset, self.circuits, None, None, method_names=('terms', 'dterms'))]


class PoissonPicDeltaLogLFunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True
    enable_hessian_tests = True

    def build_objfns(self):
        return [_objfns.PoissonPicDeltaLogLFunction.create_from(self.model, self.dataset, self.circuits,
                                                    None, penalties, method_names=('terms', 'dterms', 'hessian'))
                for penalties in self.penalty_dicts]


class DeltaLogLFunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = False
    enable_hessian_tests = False

    def build_objfns(self):
        return [_objfns.DeltaLogLFunction.create_from(self.model, self.dataset, self.circuits, None, None, method_names=('terms', 'dterms'))]


class MaxLogLFunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = False
    enable_hessian_tests = False

    def build_objfns(self):
        return [_objfns.MaxLogLFunction.create_from(self.model, self.dataset, self.circuits, None, None, method_names=('terms', 'dterms'))]


class TVDFunctionTester(TimeIndependentMDSObjectiveFunctionTesterBase, BaseCase):
    computes_lsvec = True
    enable_hessian_tests = False

    def build_objfns(self):
        return [_objfns.TVDFunction.create_from(self.model, self.dataset, self.circuits, None, None, method_names=('terms', 'dterms'))]

    def test_derivative(self):
        self.skipTest("Derivatives for TVDFunction aren't implemented yet.")


class TimeDependentMDSObjectiveFunctionTesterBase(ObjectiveFunctionData):
    """
    Tests for methods in the TimeDependentMDSObjectiveFunction class.
    """

    @staticmethod
    def build_objfns(cls):
        raise NotImplementedError() 

    def setUp(self):
        super().setUp()
        self.model.sim = pygsti.forwardsims.MapForwardSimulator(model=self.model, max_cache_size=0)
        self.objfns = self.build_objfns()

    def test_lsvec(self):
        for objfn in self.objfns:
            lsvec = objfn.lsvec()
            #TODO: add validation

    def test_dlsvec(self):
        for objfn in self.objfns:
            dlsvec = objfn.dlsvec()
            #TODO: add validation


class TimeDependentChi2FunctionTester(TimeDependentMDSObjectiveFunctionTesterBase, BaseCase):
    """
    Tests for methods in the TimeDependentChi2Function class.
    """

    def build_objfns(self):
        return [_objfns.TimeDependentChi2Function.create_from(self.model, self.dataset, self.circuits, method_names=('lsvec', 'dlsvec'))]


class TimeDependentPoissonPicLogLFunctionTester(TimeDependentMDSObjectiveFunctionTesterBase, BaseCase):
    """
    Tests for methods in the TimeDependentPoissonPicLogLFunction class.
    """

    def build_objfns(self):
        return [_objfns.TimeDependentPoissonPicLogLFunction.create_from(self.model, self.dataset, self.circuits, method_names=('lsvec', 'dlsvec'))]


class LogLWildcardFunctionTester(ObjectiveFunctionData, BaseCase):
    """
    Tests for methods in the LogLWildcardFunction class.
    """

    def setUp(self):
        super().setUp()
        logl_fn = _objfns.PoissonPicDeltaLogLFunction.create_from(self.model, self.dataset, self.circuits, method_names=('fn', 'terms', 'lsvec'))
        logl_fn.fn()  # evaluate so internals are initialized

        wcbudget = _PrimitiveOpsWildcardBudget(self.model.primitive_op_labels)
        self.pt = wcbudget.to_vector().copy()
        self.objfn = _objfns.LogLWildcardFunction(logl_fn, None, wcbudget)

    def test_values(self):
        fn = self.objfn.fn(self.pt)
        terms = self.objfn.terms(self.pt)
        lsvec = self.objfn.lsvec(self.pt)

        self.assertAlmostEqual(fn, sum(terms))
        self.assertArraysAlmostEqual(terms, lsvec**2)
        #TODO: more validation
