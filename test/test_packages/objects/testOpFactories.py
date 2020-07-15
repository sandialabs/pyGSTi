import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os

from pygsti.modelpacks.legacy import std1Q_XYI

from ..testutils import BaseTestCase, compare_files, temp_files


class XRotationOpFactory(pygsti.obj.OpFactory):
    def __init__(self):
        dim = 4
        pygsti.obj.OpFactory.__init__(self, dim, "densitymx")
        
    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None) # we don't use these, and they're only non-None when we're expected to use them
        assert(len(args) == 1)
        theta = float(args[0])/2.0
        print("INIT: theta = ", theta," sslbls=",sslbls)
        b = 2*np.cos(theta)*np.sin(theta)
        c = np.cos(theta)**2 - np.sin(theta)**2
        superop = np.array([[1,   0,   0,   0],
                            [0,   1,   0,   0],
                            [0,   0,   c,  -b],
                            [0,   0,   b,   c]],'d')
        #print("Superop = \n",superop)

        return pygsti.obj.StaticDenseOp(superop)


class XRotationOp(pygsti.obj.DenseOperator):
    def __init__(self, target_angle, initial_params=(0,0)):
        #initialize with no noise
        self.target_angle = target_angle
        super(XRotationOp,self).__init__(np.identity(4,'d'), "densitymx") # this is *super*-operator, so "densitymx"
        self.from_vector(np.array(initial_params,'d'))         

    @property
    def num_params(self): 
        return 2 # we have two parameters
    
    def to_vector(self):
        return np.array([self.depol_amt, self.over_rotation],'d') #our parameter vector
        
    def from_vector(self,v, close=False, dirty_value=True):
        #initialize from parameter vector v
        self.depol_amt = v[0]
        self.over_rotation = v[1]
        
        theta = (self.target_angle + self.over_rotation)/2
        a = 1.0-self.depol_amt
        b = a*2*np.cos(theta)*np.sin(theta)
        c = a*(np.cos(theta)**2 - np.sin(theta)**2)
        
        # .base is a member of DenseOperator and is a numpy array that is 
        # the dense Pauli transfer matrix of this operator
        self.base[:,:] = np.array([[1,   0,   0,   0],
                                   [0,   a,   0,   0],
                                   [0,   0,   c,  -b],
                                   [0,   0,   b,   c]],'d')
        self.dirty = dirty_value

        
class ParamXRotationOpFactory(pygsti.obj.OpFactory):
    def __init__(self):
        dim = 4  # 1-qubit
        self.params = np.array([0,0],'d')  #initialize with no noise
        pygsti.obj.OpFactory.__init__(self, dim, "densitymx")
        
    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None) # we don't use these, and they're only non-None when we're expected to use them
        assert(len(args) == 1)
        return XRotationOp( float(args[0]) ) #no need to set parameters of returned op - done by base class

    @property
    def num_params(self): 
        return len(self.params) # we have two parameters
    
    def to_vector(self):
        return self.params #our parameter vector
        
    def from_vector(self,v, close=False, dirty_value=True):
        self.params[:] = v
        self.dirty = dirty_value

    
class OpFactoryTestCase(BaseTestCase):

    def setUp(self):
        super(OpFactoryTestCase, self).setUp()

    def test_opfactory_simple_1Q(self):
        std_mdl = std1Q_XYI.target_model()
        Gxrot_factory = XRotationOpFactory()

        nQubits = 1
        mdl = pygsti.obj.LocalNoiseModel.from_parameterization(
            nQubits, ('Gi','Gx','Gy'))
        mdl.factories['layers'][('Gxrot',0)] = Gxrot_factory

        c1 = pygsti.obj.Circuit('Gxrot;1.57:0') 
        c2 = pygsti.obj.Circuit([('Gxrot',';',1.57,0)])
        c3 = pygsti.obj.Circuit([('Gy',0),('Gy',0),('Gx',0), ('Gxrot',';',1.25,0),('Gx',0)] )

        p1 = mdl.probabilities(c1)
        p2 = mdl.probabilities(c2)
        p3 = mdl.probabilities(c3)

        self.assertAlmostEqual(p1['0'], 0.5003981633553666)
        self.assertAlmostEqual(p2['0'], 0.5003981633553666)
        self.assertAlmostEqual(p1['1'], 0.499601836644)
        self.assertAlmostEqual(p2['1'], 0.499601836644)
        self.assertAlmostEqual(p3['0'], 0.657661181197634)
        self.assertAlmostEqual(p3['1'], 0.34233881880236)

    def test_embedded_opfactory_2Q(self):
        nQubits = 2
        Gxrot_factory = XRotationOpFactory()
        mdl = pygsti.obj.LocalNoiseModel.from_parameterization(
            nQubits, ('Gi','Gx','Gy'))
        mdl.factories['layers'][('Gxrot',0)] = pygsti.objects.EmbeddedOpFactory((0,1),(0,),Gxrot_factory,dense=True)
        mdl.factories['layers'][('Gxrot',1)] = pygsti.objects.EmbeddedOpFactory((0,1),(1,),Gxrot_factory,dense=True)

        c = pygsti.obj.Circuit( [('Gxrot',';3.14',0),('Gxrot',';1.5',1)] )
        p = mdl.probabilities(c)
        self.assertAlmostEqual(p[('11',)], 0.46463110452654444)

    def test_embedding_opfactory_2Q(self):
        nQubits = 2
        Gxrot_factory = XRotationOpFactory()
        mdl = pygsti.obj.LocalNoiseModel.from_parameterization(
            nQubits, ('Gi','Gx','Gy'))
        mdl.factories['layers']['Gxrot'] = pygsti.objects.EmbeddingOpFactory((0,1),Gxrot_factory,dense=True)

        c = pygsti.obj.Circuit( [('Gxrot',';3.14',0),[('Gxrot',';1.5',1),('Gx',0)]] )

        p = mdl.probabilities(c)
        self.assertAlmostEqual(p[('10',)], 0.2681106285986824)

    def test_parameterized_opfactory(self):
        # check to make sure gpindices is set correctly
        std_mdl = std1Q_XYI.target_model()
        Gxrot_param_factory = ParamXRotationOpFactory()

        nQubits = 1
        mdl = pygsti.obj.LocalNoiseModel.from_parameterization(
            nQubits, ('Gi','Gx','Gy'))
        mdl.factories['layers'][('Gxrot',0)] = Gxrot_param_factory
        self.assertEqual(mdl.num_params, 2)

        #see that parent and gpindices of ops created by factory are correctly set
        mdl.from_vector( np.array([0.1,0.02]) )
        op = mdl.circuit_layer_operator( pygsti.obj.Label(('Gxrot',';1.57',0)), 'op')
        self.assertArraysAlmostEqual( op.to_dense(),
                                      np.array([[1,   0,    0,   0],
                                                [0, 0.9,    0,   0],
                                                [0,   0,   -0.01728224, -0.89983405],
                                                [0,   0,    0.89983405, -0.01728224]],'d'))
        self.assertTrue(op.parent is mdl)
        self.assertEqual(op.gpindices, slice(0,2))
        self.assertArraysAlmostEqual( mdl.to_vector(), np.array([0.1, 0.02]) )
        self.assertArraysAlmostEqual( op.to_vector(), np.array([0.1, 0.02]) )

        c1 = pygsti.obj.Circuit('Gxrot;2.5:0')
        #ALT: c1 = pygsti.obj.Circuit([('Gxrot', ';', np.pi, 0)])

        #Test that probs change appropriately when the model parameters are
        # given new (different) values.
        mdl.from_vector( np.array([0.0,0.0]) )
        p = mdl.probabilities(c1)
        self.assertAlmostEqual(p['0'], 0.09942819222653321)
        self.assertAlmostEqual(p['1'], 0.9005718077734666)

        mdl.from_vector( np.array([0.5,0.02]) )
        p = mdl.probabilities(c1)
        self.assertAlmostEqual(p['0'], 0.2967619907250274)
        self.assertAlmostEqual(p['1'], 0.7032380092749724)

    #TODO: add tests for ComposedOpFactory and UnitaryOpFactory here?
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
