from __future__ import division
import unittest
import pickle
import pygsti
import numpy as np
import warnings
import os
from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files


class TestConfidenceRegionConstructionMethods(BaseTestCase):

    def setUp(self):
        super(TestConfidenceRegionConstructionMethods, self).setUp()

        #OK for these tests, since we test user interface?
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = False
        
        self.gateset = std.gs_target
        self.datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0.1)

        self.fiducials = std.fiducials
        self.germs = std.germs
        self.gateLabels = list(self.gateset.gates.keys()) # also == std.gates

        self.maxLengthList = [0,1,2,4]

        self.lsgstStrings = pygsti.construction.make_lsgst_experiment_list(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLengthList )

        self.ds = pygsti.construction.generate_fake_data(
            self.datagen_gateset, self.lsgstStrings,
            nSamples=1000,sampleError='binomial', seed=100)


    def test_construct_loglCR(self):
        cr = pygsti.construction.logl_confidence_region(self.gateset, self.ds, 95,
                           gatestring_list=self.lsgstStrings, probClipInterval=(-1e6,1e6),
                           minProbClip=1e-4, radius=1e-4, hessianProjection="std",
                           regionType="std", comm=None, memLimit=None,
                           cptp_penalty_factor=None, distributeMethod="deriv",
                           gateLabelAliases=None)

        #also test gatestring_list=None (defaults to dataset keys) behavior
        cr_linresponse = pygsti.construction.logl_confidence_region(self.gateset, self.ds, 95,
                           gatestring_list=None, probClipInterval=(-1e6,1e6),
                           minProbClip=1e-4, radius=1e-4, hessianProjection="linear response",
                           regionType="std", comm=None, memLimit=None,
                           cptp_penalty_factor=None, distributeMethod="deriv",
                           gateLabelAliases=None)

        with self.assertRaises(ValueError):
            pygsti.construction.logl_confidence_region(self.gateset, self.ds, 95,
                                                       regionType="foobar")




    def test_construct_chi2CR(self):
        cr = pygsti.construction.chi2_confidence_region(self.gateset, self.ds, 95,
                           gatestring_list=self.lsgstStrings, probClipInterval=(-1e6,1e6),
                           minProbClipForWeighting=1e-4, hessianProjection="std",
                           regionType='std', comm=None, memLimit=None,
                           gateLabelAliases=None)

        #also test gatestring_list=None (defaults to dataset keys) behavior
        with self.assertRaises(NotImplementedError): # not implemented yet
            cr = pygsti.construction.chi2_confidence_region(self.gateset, self.ds, 95,
                           gatestring_list=None, probClipInterval=(-1e6,1e6),
                           minProbClipForWeighting=1e-4, hessianProjection="linear response",
                           regionType='std', comm=None, memLimit=None,
                           gateLabelAliases=None)

        with self.assertRaises(ValueError):
            cr = pygsti.construction.chi2_confidence_region(self.gateset, self.ds, 95,
                                                            regionType="foobar")



    def test_construct_nonmark_loglCR(self):
        cr = pygsti.construction.logl_confidence_region(self.gateset, self.ds, 95,
                           gatestring_list=self.lsgstStrings, probClipInterval=(-1e6,1e6),
                           minProbClip=1e-4, radius=1e-4, hessianProjection="std",
                           regionType="non-markovian", comm=None, memLimit=None,
                           cptp_penalty_factor=None, distributeMethod="deriv",
                           gateLabelAliases=None)


    def test_construct_nonmark_chi2CR(self):
        cr = pygsti.construction.chi2_confidence_region(self.gateset, self.ds, 95,
                           gatestring_list=self.lsgstStrings, probClipInterval=(-1e6,1e6),
                           minProbClipForWeighting=1e-4, hessianProjection="std",
                           regionType='non-markovian', comm=None, memLimit=None,
                           gateLabelAliases=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
