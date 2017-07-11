import pygsti
import pickle

def load():
    with open('gauge_profiling/2qbit_results.pkl', 'rb') as infile:
        results = pickle.load(infile)

    est = results.estimates['default']
    gs_target = est.gatesets['target']
    gs = est.gatesets['final iteration estimate']
    return gs, gs_target
