import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti
import numpy as np
from pygsti.modelpacks import smq1Q_XY

from ..testutils import BaseTestCase


#Note: these are the same sample factory classes used elsewhere in the unit tests - TODO: consolidate
class XRotationOp(pygsti.modelmembers.operations.DenseOperator):
    def __init__(self, target_angle, initial_params=(0,0)):
        #initialize with no noise
        self.target_angle = target_angle
        super(XRotationOp,self).__init__(np.identity(4, 'd'), 'pp', "densitymx") # this is *super*-operator, so "densitymx"
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

        # ._ptr is a member of DenseOperator and is a numpy array that is
        # the dense Pauli transfer matrix of this operator
        self._ptr[:] = np.array([[1,   0,   0,   0],
                                 [0,   a,   0,   0],
                                 [0,   0,   c,  -b],
                                 [0,   0,   b,   c]],'d')
        self._ptr_has_changed()  # call this whenever the contents of _ptr may have changed.
        self.dirty = dirty_value


class ParamXRotationOpFactory(pygsti.modelmembers.operations.OpFactory):
    def __init__(self):
        dim = 4  # 1-qubit
        self.params = np.array([0,0],'d')  #initialize with no noise
        pygsti.modelmembers.operations.OpFactory.__init__(self, dim, "densitymx")

    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None) # we don't use these, and they're only non-None when we're expected to use them
        assert(len(args) == 1)
        return XRotationOp( float(args[0]) ) #no need to set parameters of returned op - done by base class

    @property
    def num_params(self):
        return len(self.params) # we have two parameters

    def to_vector(self):
        return self.params #our parameter vector

    def from_vector(self, v, clean=False, dirty_value=True):
        self.params[:] = v
        self.dirty = dirty_value


class ContinuousGatesTestCase(BaseTestCase):

    def setUp(self):
        super(ContinuousGatesTestCase, self).setUp()

    def test_continuous_gates_gst(self):

        nQubits = 1

        #Create some sequences:
        maxLens = [1]
        #use minimally IC set of prep and measurement fiducials
        min_prep_fids = smq1Q_XY.prep_fiducials()[0:4] #Use a minimally informationally complete set of fiducials
        min_meas_fids = smq1Q_XY.meas_fiducials()[0:3]
        seqStructs = pygsti.circuits.create_lsgst_circuit_lists(
            smq1Q_XY.target_model(), min_prep_fids, min_meas_fids, smq1Q_XY.germs(lite=True), maxLens)

        #Add random X-rotations via label arguments
        np.random.seed(1234)
        def sub_Gxrots(circuit):
            ret = circuit.replace_layer( ('Gxpi2',0), ('Gxrot',0,';',np.pi/2 + 0.02*(np.random.random()-0.5)) )
            return ret
        ss0 = seqStructs[0].copy()
        ss1 = ss0.process_circuits(sub_Gxrots)
        allStrs = pygsti.tools.remove_duplicates(ss0[:] + ss1[:])

        print(len(allStrs),"sequences ")
        self.assertEqual(len(allStrs), 47)  # Was 167 when process_circuits acted on *list* rather than individual plaquettes

        #Generate some data for these sequences (simulates an implicit model with factory)
        pspec = pygsti.processors.QubitProcessorSpec(nQubits, ('Gxpi2','Gypi2'))
        mdl_datagen = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='H+S', ideal_spam_type='H+S')

        mdl_datagen.factories['layers'][('Gxrot', 0)] = ParamXRotationOpFactory()
        print(mdl_datagen.num_params, "model params")
        self.assertEqual(mdl_datagen.num_params, 26)

        np.random.seed(4567)
        datagen_vec = 0.001 * np.random.random(mdl_datagen.num_params)
        mdl_datagen.from_vector(datagen_vec)
        ds = pygsti.data.simulate_data(mdl_datagen, allStrs, 10000, seed=1234)

        #Run GST
        mdl = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='H+S', ideal_spam_type='H+S')
        mdl.factories['layers'][('Gxrot', 0)] = ParamXRotationOpFactory()
        mdl.sim = 'map'  # must use map calcs with factories (at least for now, since matrix eval trees don't know about all possible gates?)
        #mdl.from_vector( datagen_vec ) # DEBUG - used to see at where optimization should get us...

        results = pygsti.run_long_sequence_gst_base(ds, mdl, [allStrs], gauge_opt_params=False, verbosity=3)

        _, nSigma, pval = pygsti.two_delta_logl(results.estimates[results.name].models['final iteration estimate'], results.dataset,
                                                dof_calc_method="all")
        self.assertTrue(nSigma < 5.0) # so far we just know that this should roughly work -- how to make TD-GST robus is still an open research topic

if __name__ == "__main__":
    unittest.main(verbosity=2)
