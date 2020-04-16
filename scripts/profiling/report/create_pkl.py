#!/usr/bin/env python3
import pygsti
import pickle
from pygsti.construction import std1Q_XYI

def main():
    gs_target = std1Q_XYI.gs_target
    fiducials = std1Q_XYI.fiducials
    germs = std1Q_XYI.germs
    maxLengths = [1,2,4]
    #maxLengths = [1, 2, 4, 8, 16, 32, 64]

    #Generate some data
    gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
    gs_datagen = gs_datagen.rotate(rotate=0.04)
    listOfExperiments = pygsti.construction.make_lsgst_experiment_list(gs_target, fiducials, fiducials, germs, maxLengths)
    ds = pygsti.construction.generate_fake_data(gs_datagen, listOfExperiments, n_samples=1000,
                                                sample_error="binomial", seed=1234)
    #Run GST
    gs_target.set_all_parameterizations("TP") #TP-constrained
    results = pygsti.do_long_sequence_gst(ds, gs_target, fiducials, fiducials, germs,
                                          maxLengths, verbosity=0)
    with open('data/example_report_results.pkl', 'wb') as outfile:
        pickle.dump(results, outfile, protocol=2)

    # Case1: TP-constrained GST
    tpTarget = gs_target.copy()
    tpTarget.set_all_parameterizations("TP")
    results_tp = pygsti.do_long_sequence_gst(ds, tpTarget, fiducials, fiducials, germs,
                                          maxLengths, gauge_opt_params=False, verbosity=0)
    # Gauge optimize
    est = results_tp.estimates['default']
    gsFinal = est.gatesets['final iteration estimate']
    gsTarget = est.gatesets['target']
    for spamWt in [1e-4,1e-3,1e-2,1e-1,1.0]:
        gs = pygsti.gaugeopt_to_target(gsFinal,gsTarget,{'gates':1, 'spam':spamWt})
        est.add_gaugeoptimized({'item_weights': {'gates':1, 'spam':spamWt}}, gs, "Spam %g" % spamWt) 

    #Case2: "Full" GST
    fullTarget = gs_target.copy()
    fullTarget.set_all_parameterizations("full")
    results_full = pygsti.do_long_sequence_gst(ds, fullTarget, fiducials, fiducials, germs,
                                          maxLengths, gauge_opt_params=False, verbosity=0)
    #Gauge optimize
    est = results_full.estimates['default']
    gsFinal = est.gatesets['final iteration estimate']
    gsTarget = est.gatesets['target']
    for spamWt in [1e-4,1e-3,1e-2,1e-1,1.0]:
        gs = pygsti.gaugeopt_to_target(gsFinal,gsTarget,{'gates':1, 'spam':spamWt})
        est.add_gaugeoptimized({'item_weights': {'gates':1, 'spam':spamWt}}, gs, "Spam %g" % spamWt)

    with open('data/full_report_results.pkl', 'wb') as outfile:
        pickle.dump((results_tp, results_full), outfile, protocol=2)
        
if __name__ == '__main__':
    main()
