from __future__ import print_function
import unittest
import numpy as np

import pygsti
import pygsti.construction as pc
from pygsti.construction import std2Q_XYCNOT as std
from pygsti.construction import std1Q_XY
from pygsti.objects import Label as L
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files

class CalcMethods2QTestCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        """ 
        Handle all once-per-class (slow) computation and loading,
         to avoid calling it for each test (like setUp).  Store
         results in class variable for use within setUp.
        """
        super(CalcMethods2QTestCase, cls).setUpClass()

        #Note: std is a 2Q gateset
        cls.maxLengths = [1]
        cls.gs_datagen = std.gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
        cls.listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
            std.gs_target, std.prepStrs, std.effectStrs, std.germs, cls.maxLengths)
        cls.ds = pygsti.construction.generate_fake_data(cls.gs_datagen, cls.listOfExperiments,
                                                         nSamples=1000, sampleError="multinomial", seed=1234)

        #Reduced model GST dataset
        cls.nQubits=2
        cls.gs_redmod_datagen = pc.build_nqnoise_gateset(cls.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                    extraWeight1Hops=0, extraGateWeight=1, sparse=False, sim_type="matrix", verbosity=1,
                                                    gateNoise=(1234,0.01), prepNoise=(456,0.01), povmNoise=(789,0.01))

        #Create a reduced set of fiducials and germs
        gateLabels = list(cls.gs_redmod_datagen.gates.keys())
        fids1Q = std1Q_XY.fiducials[0:2] # for speed
        cls.redmod_fiducials = []
        for i in range(cls.nQubits):
            cls.redmod_fiducials.extend( pygsti.construction.manipulate_gatestring_list(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ]) )
        #print(redmod_fiducials, "Fiducials")     
        
        cls.redmod_germs = pygsti.construction.gatestring_list([ (gl,) for gl in gateLabels ])
        cls.redmod_maxLs = [1]
        expList = pygsti.construction.make_lsgst_experiment_list(
            cls.gs_redmod_datagen, cls.redmod_fiducials, cls.redmod_fiducials,
            cls.redmod_germs, cls.redmod_maxLs)
        cls.redmod_ds = pygsti.construction.generate_fake_data(cls.gs_redmod_datagen, expList, 1000, "round", seed=1234)
        #print(len(expList)," reduced model sequences")

        #Random starting points - little kick so we don't get hung up at start
        np.random.seed(1234)
        cls.rand_start18 = np.random.random(18)*1e-6
        cls.rand_start128 = np.random.random(128)*1e-6
        cls.rand_start150 = np.random.random(150)*1e-6
        

    ## GST using "full" (non-embedded/composed) gates
    # All of these calcs use dense matrices; While sparse gate matrices (as Maps) could be used,
    # they'd need to enter as a sparse basis to a LindbladParameterizedGate (maybe add this later?)
    
    def test_stdgst_matrix(self):
        # Using matrix-based calculations
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("CPTP")
        gs_target.set_simtype('matrix') # the default for 1Q, so we could remove this line
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=4)


    def test_stdgst_map(self):
        # Using map-based calculation
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("CPTP")
        gs_target.set_simtype('map')
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=4)

    def test_stdgst_terms(self):
        # Using term-based (path integral) calculation
        # This performs a map-based unitary evolution along each path. 
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("H+S terms")
        gs_target.set_simtype('termorder:1') # this is the default set by set_all_parameterizations above
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=1)    


    # ## GST using "reduced" models
    # Reduced, meaning that we use composed and embedded gates to form a more complex error model with
    # shared parameters and qubit connectivity graphs.  Calculations *can* use dense matrices and matrix calcs,
    # but usually will use sparse mxs and map-based calcs.

    def test_reducedmod_matrix(self):
        # Using dense matrices and matrix-based calcs
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False,
                                             sim_type="matrix", verbosity=1)
        gs_target.from_vector(self.rand_start128)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

    def test_reducedmod_map1(self):
        # Using dense embedded matrices and map-based calcs (maybe not really necessary to include?)
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False,
                                             sim_type="map", verbosity=1)
        gs_target.from_vector(self.rand_start128)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})


    def test_reducedmod_map2(self):
        # Using sparse embedded matrices and map-based calcs
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=True,
                                             sim_type="map", verbosity=1)
        gs_target.from_vector(self.rand_start128)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

    def test_reducedmod_svterm(self):
        # Using term-based calcs using map-based state-vector propagation
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                      extraWeight1Hops=0, extraGateWeight=1, sparse=False, verbosity=1,
                                      sim_type="termorder:1", parameterization="H+S terms")
        gs_target.from_vector(self.rand_start150)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

    def test_reducedmod_cterm(self):
        # Using term-based calcs using map-based stabilizer-state propagation
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False, verbosity=1,
                                             sim_type="termorder:1", parameterization="H+S clifford terms")
        gs_target.from_vector(self.rand_start150)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
