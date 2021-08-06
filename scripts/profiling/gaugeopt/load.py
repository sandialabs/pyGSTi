import pickle

def load():
    with open('gaugeopt/2qbit_results.pkl', 'rb') as infile:
        results = pickle.load(infile)

    est = results.estimates['default']
    gs_target = est.gatesets['target']
    gs = est.gatesets['final iteration estimate']
    return gs, gs_target

def load_3q():
    with open('gaugeopt/3qbit_results.pkl', 'rb', ) as infile:
        return pickle.load(infile, encoding='latin1')
