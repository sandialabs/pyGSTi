import pickle

with open('results.pkl', 'rb') as infile:
    results = pickle.load(infile)

estimates = results.estimates
default   = estimates['default']
gatesets  = default.gatesets

tgt      = gatesets['target']
ds       = results.dataset
gs       = gatesets['go0']
gs_final = gatesets['final iteration estimate']

cri = None
