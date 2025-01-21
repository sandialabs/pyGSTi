import numpy as np
import unittest

from ..util import BaseCase

from pygsti.circuits import Circuit
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.states import ComposedState
from pygsti.modelmembers.povms import ComposedPOVM
from pygsti.models.gaugegroup import FullGaugeGroupElement, UnitaryGaugeGroupElement
import pygsti.baseobjs.outcomelabeldict as ld
import pygsti.modelmembers.operations as op
import pygsti.modelmembers.povms as pv
import pygsti.modelmembers.states as sv


# Main test of ComposedSpamvecBase:
# Is the composed SPAM vec equivalent to applying each component separately?
class ComposedSpamvecBase(object):
    base_prep_vec = sv.ComputationalBasisState([0], 'pp', 'default')
    base_noise_op = op.StaticStandardOp('Gxpi2', 'pp', 'default') # X(pi/2) rotation as noise
    base_povm = pv.ComputationalBasisPOVM(1, 'default') # Z-basis measurement
    expected_out = ld.OutcomeLabelDict([(('0',), 0.5), (('1',), 0.5)])

    def setUp(self):
        self.vec = self.build_vec()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.vec.num_params, self.n_params)

    def test_get_dimension(self):
        self.assertEqual(self.vec.dim, 4)
    
    def test_hessian(self):
        if self.n_params:
            self.assertTrue(self.vec.has_nonzero_hessian())
        else:
            self.assertFalse(self.vec.has_nonzero_hessian())

    def test_forward_simulation(self):
        pure_vec = self.vec.state_vec
        noise_op = self.vec.error_map

        # TODO: Would be nice to check more than densitymx evotype
        indep_mdl = ExplicitOpModel(['Q0'], evotype='default')
        indep_mdl['rho0'] = pure_vec
        indep_mdl['G0'] = noise_op
        indep_mdl['Mdefault'] = self.base_povm
        indep_mdl.num_params  # triggers paramvec rebuild
        
        composed_mdl = ExplicitOpModel(['Q0'], evotype='default')
        composed_mdl['rho0'] = self.vec
        composed_mdl['Mdefault'] = self.base_povm
        composed_mdl.num_params  # triggers paramvec rebuild
        
        # Sanity check
        indep_circ = Circuit(['rho0', 'G0', 'Mdefault'])
        indep_probs = indep_mdl.probabilities(indep_circ)
        for k,v in indep_probs.items():
            self.assertAlmostEqual(self.expected_out[k], v)

        composed_circ = Circuit(['rho0', 'Mdefault'])
        composed_probs = composed_mdl.probabilities(composed_circ)
        for k,v in composed_probs.items():
            self.assertAlmostEqual(self.expected_out[k], v)


# For ComposedSpamvec, the spam vec is immutable (set_value),
# but the noise op can be not (transform, depolarize)
class MutableComposedSpamvecBase(ComposedSpamvecBase):
    def test_raises_on_set_value(self):
        v = np.asarray(self.vec)
        with self.assertRaises(ValueError):
            self.vec.set_dense(v)

    @unittest.skip("Transform is expected to fail while spam_transform_inplace is not available")
    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.vec.transform_inplace(S)
        # TODO assert correctness

    def test_depolarize(self):
        self.vec.depolarize(0.9)
        self.vec.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness
    
class ImmutableComposedSpamvecBase(ComposedSpamvecBase):
    def test_raises_on_set_value(self):
        v = np.asarray(self.vec)
        with self.assertRaises(ValueError):
            self.vec.set_dense(v)

    @unittest.skip("Transform is expected to fail while spam_transform_inplace is not available")
    def test_raises_on_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S)

    def test_raises_on_depolarize(self):
        with self.assertRaises(ValueError):
            self.vec.depolarize(0.9)
        # TODO assert correctness

# Cases where noise op is also static acts like an immutable spamvec
class StandardStaticComposedSpamvecTester(ImmutableComposedSpamvecBase, BaseCase):
    n_params = 0

    def build_vec(self):
        return ComposedState(self.base_prep_vec, self.base_noise_op)

class StaticDenseComposedSpamvecTester(ImmutableComposedSpamvecBase, BaseCase):
    n_params = 0

    def build_vec(self):
        sdop = op.StaticArbitraryOp(self.base_noise_op.to_dense())
        return ComposedState(self.base_prep_vec, sdop)

class FullDenseComposedSpamvecTester(MutableComposedSpamvecBase, BaseCase):
    n_params = 16

    def build_vec(self):
        fdop = op.FullArbitraryOp(self.base_noise_op.to_dense())
        return ComposedState(self.base_prep_vec, fdop)

# Currently not inheriting for easy merge, meaning test doesn't quite work
# class LindbladSpamvecTester(MutableComposedSpamvecBase, BaseCase):
#     n_params = 12

#     # Transform cannot be FullGaugeGroup for Lindblad
#     def test_transform(self):
#         S = UnitaryGaugeGroupElement(np.identity(4, 'd'))
#         self.vec.transform_inplace(S, 'prep')
#         self.vec.transform_inplace(S, 'effect')

#     def build_vec(self):
#         lop = op.LindbladDenseOp.from_operation_matrix(self.base_noise_op.to_dense())
#         return sv.LindbladSPAMVec(self.base_prep_vec, lop, 'prep')


## POVM ##

# Main test of ComposedPovmBase:
# Is the composed POVM equivalent to applying each component separately?
class ComposedPovmBase(object):
    base_prep = sv.ComputationalBasisState([0], 'pp', 'default') # 0 state prep
    base_noise_op = op.StaticStandardOp('Gxpi2', 'pp', 'default') # X(pi/2) rotation as noise
    base_povm = pv.ComputationalBasisPOVM(1, 'default') # Z-basis measurement
    expected_out = ld.OutcomeLabelDict([(('0',), 0.5), (('1',), 0.5)])
    
    def setUp(self):
        self.povm = self.build_povm()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.povm.num_params, self.n_params)

    def test_get_dimension(self):
        self.assertEqual(self.povm.state_space.dim, 4)
    
    def test_forward_simulation(self):
        pure_povm = self.povm.base_povm
        noise_op = self.povm.error_map

        # TODO: Would be nice to check more than densitymx evotype
        indep_mdl = ExplicitOpModel(['Q0'], evotype='default')
        indep_mdl['rho0'] = self.base_prep
        indep_mdl['G0'] = noise_op.copy()
        indep_mdl['Mdefault'] = pure_povm
        indep_mdl._clean_paramvec()
        
        composed_mdl = ExplicitOpModel(['Q0'], evotype='default')
        composed_mdl['rho0'] = self.base_prep
        composed_mdl['Mdefault'] = self.povm
        composed_mdl._clean_paramvec()
        
        # Sanity check
        indep_circ = Circuit(['rho0', 'G0', 'Mdefault'])
        indep_probs = indep_mdl.probabilities(indep_circ)
        for k,v in indep_probs.items():
            self.assertAlmostEqual(self.expected_out[k], v)

        composed_circ = Circuit(['rho0', 'Mdefault'])
        composed_probs = composed_mdl.probabilities(composed_circ)
        for k,v in composed_probs.items():
            self.assertAlmostEqual(self.expected_out[k], v)

# For ComposedPOVM, the noise op could be mutable
class MutableComposedPovmBase(ComposedPovmBase):
    def test_vector_conversion(self):
        v = self.povm.to_vector()
        self.povm.from_vector(v)

    @unittest.skip("Transform is expected to fail while spam_transform_inplace is not available")
    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.povm.transform_inplace(S)
        # TODO assert correctness

    def test_depolarize(self):
        self.povm.depolarize(0.9)
        self.povm.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness
    
class ImmutableComposedPovmBase(ComposedPovmBase):
    def test_vector_conversion(self):
        v = self.povm.to_vector()
        self.povm.from_vector(v)
        # TODO: 
        # with self.assertRaises(ValueError):
        #     self.vec.set_dense(v)

    @unittest.skip("Transform is expected to fail while spam_transform_inplace is not available")
    def test_raises_on_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.povm.transform_inplace(S)

    def test_raises_on_depolarize(self):
        with self.assertRaises(ValueError):
            self.povm.depolarize(0.9)
        # TODO assert correctness

class StandardStaticComposedPovmTester(ImmutableComposedPovmBase, BaseCase):
    n_params = 0

    def build_povm(self):
        return ComposedPOVM(self.base_noise_op, self.base_povm, 'pp')

class StaticDenseComposedPovmTester(ImmutableComposedPovmBase, BaseCase):
    n_params = 0

    def build_povm(self):
        sdop = op.StaticArbitraryOp(self.base_noise_op.to_dense())
        return ComposedPOVM(sdop, self.base_povm, 'pp')

class FullDenseComposedPovmTester(MutableComposedPovmBase, BaseCase):
    n_params = 16

    def build_povm(self):
        fdop = op.FullArbitraryOp(self.base_noise_op.to_dense())
        return ComposedPOVM(fdop, self.base_povm, 'pp')

# Currently not inheriting for easy merge, meaning test doesn't quite work
# class LindbladPovmTester(MutableComposedPovmBase, BaseCase):
#     n_params = 12

#     # Transform cannot be FullGaugeGroup for Lindblad
#     def test_transform(self):
#         S = UnitaryGaugeGroupElement(np.identity(4, 'd'))
#         self.povm.transform_inplace(S, 'prep')
#         self.povm.transform_inplace(S, 'effect')

#     def build_povm(self):
#         lop = op.LindbladDenseOp.from_operation_matrix(self.base_noise_op.to_dense())
#         return pv.LindbladPOVM(lop, self.base_povm)

