import numpy as np

from ..util import BaseCase

from pygsti.objects import Circuit, ExplicitOpModel
from pygsti.objects.composed_sv_pv import ComposedSPAMVec, ComposedPOVM
from pygsti.objects.gaugegroup import FullGaugeGroupElement, UnitaryGaugeGroupElement
import pygsti.objects.labeldicts as ld
import pygsti.objects.operation as op
import pygsti.objects.povm as pv
import pygsti.objects.spamvec as sv


# Main test of ComposedSpamvecBase:
# Is the composed SPAM vec equivalent to applying each component separately?
class ComposedSpamvecBase(object):
    base_prep_vec = 1.0/np.sqrt(2) * np.array([1, 0, 0, 1]) # 0 state
    base_noise_op = op.StaticStandardOp('Gxpi2', 'densitymx') # X(pi/2) rotation as noise
    base_povm = pv.ComputationalBasisPOVM(1, 'densitymx') # Z-basis measurement
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
        noise_op = self.vec.noise_op
        typ = self.vec._prep_or_effect
        assert typ == 'prep', "Only prep tested in ComposedSpamvecBase"

        # TODO: Would be nice to check more than densitymx evotype
        indep_mdl = ExplicitOpModel(['Q0'], evotype='densitymx')
        indep_mdl['rho0'] = pure_vec
        indep_mdl['G0'] = noise_op
        indep_mdl['Mdefault'] = self.base_povm
        
        composed_mdl = ExplicitOpModel(['Q0'], evotype='densitymx')
        composed_mdl['rho0'] = self.vec
        composed_mdl['Mdefault'] = self.base_povm
        
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

    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.vec.transform_inplace(S, 'prep')
        self.vec.transform_inplace(S, 'effect')
        # TODO assert correctness

    def test_transform_raises_on_bad_type(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S, 'foobar')

    def test_depolarize(self):
        self.vec.depolarize(0.9)
        self.vec.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness
    
class ImmutableComposedSpamvecBase(ComposedSpamvecBase):
    def test_raises_on_set_value(self):
        v = np.asarray(self.vec)
        with self.assertRaises(ValueError):
            self.vec.set_dense(v)

    def test_raises_on_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S, 'prep')

    def test_raises_on_depolarize(self):
        with self.assertRaises(ValueError):
            self.vec.depolarize(0.9)
        # TODO assert correctness

# Cases where noise op is also static acts like an immutable spamvec
class StandardStaticComposedSpamvecTester(ImmutableComposedSpamvecBase, BaseCase):
    n_params = 0

    def build_vec(self):
        return ComposedSPAMVec(self.base_prep_vec, self.base_noise_op, 'prep')

class StaticDenseComposedSpamvecTester(ImmutableComposedSpamvecBase, BaseCase):
    n_params = 0

    def build_vec(self):
        sdop = op.StaticDenseOp(self.base_noise_op.to_dense())
        return ComposedSPAMVec(self.base_prep_vec, sdop, 'prep')

class FullDenseComposedSpamvecTester(MutableComposedSpamvecBase, BaseCase):
    n_params = 16

    def build_vec(self):
        fdop = op.FullDenseOp(self.base_noise_op.to_dense())
        return ComposedSPAMVec(self.base_prep_vec, fdop, 'prep')

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
    base_prep = sv.ComputationalSPAMVec([0], 'densitymx') # 0 state prep
    base_noise_op = op.StaticStandardOp('Gxpi2', 'densitymx') # X(pi/2) rotation as noise
    base_povm = pv.ComputationalBasisPOVM(1, 'densitymx') # Z-basis measurement
    expected_out = ld.OutcomeLabelDict([(('0',), 0.5), (('1',), 0.5)])
    
    def setUp(self):
        self.povm = self.build_povm()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.povm.num_params, self.n_params)

    def test_get_dimension(self):
        self.assertEqual(self.povm.dim, 4)
    
    def test_forward_simulation(self):
        pure_povm = self.povm.base_povm
        noise_op = self.povm.noise_op

        # TODO: Would be nice to check more than densitymx evotype
        indep_mdl = ExplicitOpModel(['Q0'], evotype='densitymx')
        indep_mdl['rho0'] = self.base_prep
        indep_mdl['G0'] = noise_op.copy()
        indep_mdl['Mdefault'] = pure_povm
        
        composed_mdl = ExplicitOpModel(['Q0'], evotype='densitymx')
        composed_mdl['rho0'] = self.base_prep
        composed_mdl['Mdefault'] = self.povm
        
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
        return ComposedPOVM(self.base_noise_op, self.base_povm)

class StaticDenseComposedPovmTester(ImmutableComposedPovmBase, BaseCase):
    n_params = 0

    def build_povm(self):
        sdop = op.StaticDenseOp(self.base_noise_op.to_dense())
        return ComposedPOVM(sdop, self.base_povm)

class FullDenseComposedPovmTester(MutableComposedPovmBase, BaseCase):
    n_params = 16

    def build_povm(self):
        fdop = op.FullDenseOp(self.base_noise_op.to_dense())
        return ComposedPOVM(fdop, self.base_povm)

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

