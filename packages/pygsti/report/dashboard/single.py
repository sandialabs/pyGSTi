#!/usr/bin/env python3

import os
import pickle
import pygsti

from pygsti.construction import std1Q_XYI

def create_pickle():
    gs_target = std1Q_XYI.gs_target
    fiducials = std1Q_XYI.fiducials
    germs = std1Q_XYI.germs
    maxLengths = [1,2,4] #,8,16,32,64]
#Generate some data
    gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
    gs_datagen = gs_datagen.rotate(rotate=0.04)
    listOfExperiments = pygsti.construction.make_lsgst_experiment_list(gs_target, fiducials, fiducials, germs, maxLengths)
    ds = pygsti.construction.generate_fake_data(gs_datagen, listOfExperiments, nSamples=1000,
                                                sampleError="binomial", seed=1234)
#Run GST
    gs_target.set_all_parameterizations("TP") #TP-constrained
    results = pygsti.do_long_sequence_gst(ds, gs_target, fiducials, fiducials, germs,
                                          maxLengths, verbosity=0)

    with open('results.pkl', 'wb') as picklefile:
        pickle.dump(results, picklefile)

def main():
    if not os.path.isfile('results.pkl'):
        create_pickle()

    with open('results.pkl', 'rb') as picklefile:
        results = pickle.load(picklefile)

    pygsti.report.create_single_qubit_report(results, "tutorial_files/exampleReport.html",
                                             verbosity=3, auto_open=True, confidenceLevel=95)

if __name__ == '__main__':
    main()
