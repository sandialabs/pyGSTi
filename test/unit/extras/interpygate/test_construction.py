
import numpy as np
from scipy.interpolate import LinearNDInterpolator as _linND

import pygsti
import pygsti.extras.interpygate as interp
from pygsti.tools.basistools import change_basis
from pygsti.modelpacks import smq1Q_XY
from pathlib import Path
working_dir = Path.cwd()

from ...util import BaseCase

sigI = np.array([[1.,0],[0, 1]], dtype='complex')
sigX = np.array([[0, 1],[1, 0]], dtype='complex')
sigY = np.array([[0,-1],[1, 0]], dtype='complex') * 1.j
sigZ = np.array([[1, 0],[0,-1]], dtype='complex')
sigM = (sigX - 1.j*sigY)/2.
sigP = (sigX + 1.j*sigY)/2.

class SingleQubitTargetOp(pygsti.modelmembers.operations.OpFactory):
    def __init__(self):
        self.process = self.create_target_gate
        pygsti.modelmembers.operations.OpFactory.__init__(self, 1, evotype="densitymx")
        self.dim = 4
    
    def create_target_gate(self, v):
        phi, theta = v
        target_unitary = (np.cos(theta/2) * sigI + 
                          1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
        superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
        return superop
    
    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None)
        mx = self.process([*args])
        return pygsti.modelmembers.operations.StaticArbitraryOp(mx)

class SingleQubitGate(interp.PhysicalProcess):
    def __init__(self, 
                 verbose=False,
                 cont_param_gate = False,
                 num_params = None,
#                  process_shape = (4, 4),
                 item_shape = (4,4),
                 aux_shape = None,
                 num_params_evaluated_as_group = 0,
                 ):

        self.verbose = verbose
        self.cont_param_gate = cont_param_gate
        self.num_params = num_params
        self.item_shape = item_shape
        self.aux_shape = aux_shape
        self.num_params_evaluated_as_group = num_params_evaluated_as_group

    def create_process_matrix(self, v, comm=None, return_generator=False):
        processes = []
        phi, theta, t = v
        theta = theta * t
        target_unitary = (np.cos(theta/2) * sigI + 
                          1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
        superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
        processes += [superop]
        return np.array(processes) if (processes is not None) else None

    def create_aux_info(self, v, comm=None):
        return []  # matches aux_shape=() above
    
    def create_process_matrices(self, v, grouped_v, comm=None):
        assert(len(grouped_v) == 1)  # we expect a single "grouped" parameter
        processes = []
        times = grouped_v[0]
        phi_in, theta_in = v
        for t in times:
            phi = phi_in
            theta = theta_in * t
            target_unitary = (np.cos(theta/2) * sigI + 
                              1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
            superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
            processes += [superop]
        return np.array(processes) if (processes is not None) else None

    def create_aux_infos(self, v, grouped_v, comm=None):
        import numpy as np
        times = grouped_v[0]
        return [ [] for t in times] # list elements must match aux_shape=() above


class InterpygateConstructionTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(InterpygateConstructionTester, cls).setUpClass()
        cls.static_target = np.bmat([[np.eye(2),np.zeros([2,2])],
                                     [np.zeros([2,2]),np.sqrt(2)/2*(sigI++1.j*sigY)]])
        cls.target_op = SingleQubitTargetOp()

        cls.param_ranges = [(0.9,1.1,3)]
        cls.arg_ranges = [2*np.pi*(1+np.cos(np.linspace(np.pi,0, 7)))/2,
                      (0, np.pi, 3)] 
        cls.arg_indices = [0,1]

        cls.gate_process = SingleQubitGate(num_params = 3,num_params_evaluated_as_group = 1)
        
        
    def test_target(self):
        test = self.target_op.create_target_gate([0,np.pi/4])
        self.assertArraysAlmostEqual(test, self.static_target)
        
    def test_create_gate(self):
        test = self.gate_process.create_process_matrices([0,np.pi/4], grouped_v=[[1]])[0]
        self.assertArraysAlmostEqual(test, self.static_target)

    def test_create_opfactory(self):
        opfactory_linear = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
                                self.target_op, self.gate_process, argument_ranges=self.arg_ranges, 
                                parameter_ranges=self.param_ranges, argument_indices=self.arg_indices, 
                                interpolator_and_args='linear')
        op = opfactory_linear.create_op([0,np.pi/4])
        op.from_vector([1])
        self.assertArraysAlmostEqual(op, self.static_target)

        opfactory_spline = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
                                self.target_op, self.gate_process, argument_ranges=self.arg_ranges, 
                                parameter_ranges=self.param_ranges, argument_indices=self.arg_indices, 
                                interpolator_and_args='spline')
        op = opfactory_spline.create_op([0,np.pi/4])
        op.from_vector([1])
        self.assertArraysAlmostEqual(op, self.static_target)

        interpolator_and_args = (_linND, {'rescale': True})
        opfactory_custom = opfactory_spline = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
                                self.target_op, self.gate_process, argument_ranges=self.arg_ranges, 
                                parameter_ranges=self.param_ranges, argument_indices=self.arg_indices, 
                                interpolator_and_args=interpolator_and_args)
        op = opfactory_custom.create_op([0,np.pi/4])
        op.from_vector([1])
        self.assertArraysAlmostEqual(op, self.static_target)



class InterpygateGSTTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(InterpygateGSTTester, cls).setUpClass()
        target_op = SingleQubitTargetOp()
        param_ranges = [(0.9,1.1,3)]
        arg_ranges = [2*np.pi*(1+np.cos(np.linspace(np.pi,0, 7)))/2,
                  (0, np.pi, 3)] 
        arg_indices = [0,1]
        gate_process = SingleQubitGate(num_params = 3,num_params_evaluated_as_group = 1)
        opfactory = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
                                target_op, gate_process, argument_ranges=arg_ranges, 
                                parameter_ranges=param_ranges, argument_indices=arg_indices, 
                                interpolator_and_args='linear')
        x_gate = opfactory.create_op([0,np.pi/4])
        y_gate = opfactory.create_op([np.pi/2,np.pi/4]) 

        cls.model = pygsti.models.ExplicitOpModel([0],'pp')
        cls.model['rho0'] = [ 1/np.sqrt(2), 0, 0, 1/np.sqrt(2) ] # density matrix [[1, 0], [0, 0]] in Pauli basis
        cls.model['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM(
            {'0': [ 1/np.sqrt(2), 0, 0, 1/np.sqrt(2) ],   # projector onto [[1, 0], [0, 0]] in Pauli basis
             '1': [ 1/np.sqrt(2), 0, 0, -1/np.sqrt(2) ] }, evotype="default") # projector onto [[0, 0], [0, 1]] in Pauli basis
        cls.model['Gxpi2',0] = x_gate
        cls.model['Gypi2',0] = y_gate

        
    def test_gpindices(self):
        model = self.model.copy()
        model['rho0'].set_gpindices(slice(0,4),model)
        model['Mdefault'].set_gpindices(slice(4,12),model)
        model['Gxpi2',0].set_gpindices(slice(12,13),model)
        model['Gypi2',0].set_gpindices(slice(12,13),model)
        model._rebuild_paramvec()
        self.assertEqual(model.num_params,13)
        
    def test_circuit_probabilities(self):
        datagen_model = self.model.copy()
        datagen_params = datagen_model.to_vector()
        datagen_params[-2:] = [1.1,1.1]
        datagen_model.from_vector(datagen_params)
        probs = datagen_model.probabilities( (('Gxpi2',0),))
        self.assertEqual(probs['0'],0.8247240241650917)

    def test_germ_selection(self):
        datagen_model = self.model.copy()
        datagen_params = datagen_model.to_vector()
        datagen_params[-2:] = [1.1,1.1]
        datagen_model.from_vector(datagen_params)
        
        target_model = self.model.copy()
        
        final_germs = pygsti.algorithms.germselection.find_germs(
                self.model, randomize=False, force=None, algorithm='greedy', verbosity=4)

        self.assertEqual(final_germs, [pygsti.circuits.circuit.Circuit('Gxpi2:0')])











