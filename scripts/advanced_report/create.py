#!/usr/bin/env python3
import pickle

import pygsti
from pygsti.construction import std1Q_XYI

#The usual GST setup: we're going to run GST on the standard XYI 1-qubit gateset
gs_target = std1Q_XYI.gs_target
fiducials = std1Q_XYI.fiducials
germs = std1Q_XYI.germs
maxLengths = [1,2,4,8]
listOfExperiments = pygsti.construction.create_lsgst_circuits(
    gs_target.gates.keys(), fiducials, fiducials, germs, maxLengths)

#Create some datasets for analysis
gs_datagen1 = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
gs_datagen2 = gs_target.depolarize(gate_noise=0.05, spam_noise=0.01).rotate(rotate=0.01)

ds1 = pygsti.construction.simulate_data(gs_datagen1, listOfExperiments, n_samples=1000,
                                        sample_error="binomial", seed=1234)
ds2 = pygsti.construction.simulate_data(gs_datagen2, listOfExperiments, n_samples=1000,
                                        sample_error="binomial", seed=1234)
ds3 = ds1.copy_nonstatic(); ds3.add_counts_from_dataset(ds2); ds3.done_adding_data()

#Run GST on all three datasets
gs_target.set_all_parameterizations("TP")
results1 = pygsti.run_long_sequence_gst(ds1, gs_target, fiducials, fiducials, germs, maxLengths, verbosity=0)
results2 = pygsti.run_long_sequence_gst(ds2, gs_target, fiducials, fiducials, germs, maxLengths, verbosity=0)
results3 = pygsti.run_long_sequence_gst(ds3, gs_target, fiducials, fiducials, germs, maxLengths, verbosity=0)

with open('results.pkl', 'wb') as outfile:
    pickle.dump(results1, outfile)
