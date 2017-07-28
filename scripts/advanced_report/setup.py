import numpy as np
import pygsti
from pygsti.construction import std1Q_XYI

#The usual GST setup: we're going to run GST on the standard XYI 1-qubit gateset
gs_target = std1Q_XYI.gs_target
fiducials = std1Q_XYI.fiducials
germs = std1Q_XYI.germs
maxLengths = [1,2,4,8]
listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
    gs_target.gates.keys(), fiducials, fiducials, germs, maxLengths)

#Create some datasets for analysis
gs_datagen1 = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
gs_datagen2 = gs_target.depolarize(gate_noise=0.05, spam_noise=0.01).rotate(rotate=0.01)

ds1 = pygsti.construction.generate_fake_data(gs_datagen1, listOfExperiments, nSamples=1000,
                                            sampleError="binomial", seed=1234)
ds2 = pygsti.construction.generate_fake_data(gs_datagen2, listOfExperiments, nSamples=1000,
                                            sampleError="binomial", seed=1234)
ds3 = ds1.copy_nonstatic(); ds3.add_counts_from_dataset(ds2); ds3.done_adding_data()

#Run GST on all three datasets
gs_target.set_all_parameterizations("TP")
results1 = pygsti.do_long_sequence_gst(ds1, gs_target, fiducials, fiducials, germs, maxLengths, verbosity=0)
results2 = pygsti.do_long_sequence_gst(ds2, gs_target, fiducials, fiducials, germs, maxLengths, verbosity=0)
results3 = pygsti.do_long_sequence_gst(ds3, gs_target, fiducials, fiducials, germs, maxLengths, verbosity=0)

#make some shorthand variable names for later
tgt = results1.estimates['default'].gatesets['target']

ds1 = results1.dataset
ds2 = results2.dataset
ds3 = results3.dataset

gs1 = results1.estimates['default'].gatesets['go0']
gs2 = results2.estimates['default'].gatesets['go0']
gs3 = results3.estimates['default'].gatesets['go0']

gss = results1.gatestring_structs['final']
