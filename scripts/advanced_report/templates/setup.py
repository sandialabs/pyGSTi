import pickle

with open('results.pkl', 'rb') as infile:
    results = pickle.load(infile)

estimates = results.estimates
default   = estimates['default']
gatesets  = default.gatesets

ds        = results.dataset
gs        = gatesets['go0']
gs_final  = gatesets['final iteration estimate']
gs_target = gatesets['target']
#Ls        = [results.gatestring_structs['final'].Ls]
Ls        = results.gatestring_structs['final'].Ls

cri = None
