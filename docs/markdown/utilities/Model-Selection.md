---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Model Selection (alpha)

```{warning}
Currently in development.  This example doesn't work
```

Some parts of `pygsti` are works-in-progress. Here, we investigate how to do the task of "model selection" within GST, essentially answering the question "Can we do a better job of modeling the experiment by changing the assumptions within GST?".

## Testing variable-gateset-dimension GST with model selection

###     Setup

```{code-cell} ipython3
:tags: [nbval-skip]

from __future__ import print_function
```

```{code-cell} ipython3
:tags: [nbval-skip]

import pygsti
from pygsti.construction import std1Q_XYI
```

```{code-cell} ipython3
:tags: [nbval-skip]

#Load gateset and some string lists
gs_target = std1Q_XYI.gs_target
fiducialList = std1Q_XYI.fiducials
germList = std1Q_XYI.germs
specs = pygsti.construction.build_spam_specs(fiducialList)
expList = [1,2,4]
```

```{code-cell} ipython3
:tags: [nbval-skip]

#Create some testing gate string lists
lgstList = pygsti.construction.list_lgst_gatestrings(specs, gs_target.gates.keys())
lsgstLists = [ lgstList[:] ]
for exp in expList:
    gsList = pygsti.construction.create_gatestring_list(
                "f0+germ*exp+f1", f0=fiducialList, f1=fiducialList,
                germ=germList, exp=exp, order=['germ','f0','f1'])
    lsgstLists.append( lsgstLists[-1] +  gsList )
    
dsList = pygsti.remove_duplicates( lsgstLists[-1] )
```

```{code-cell} ipython3
:tags: [nbval-skip]

#Test on fake data by depolarizing target set, increasing its dimension,
# and adding leakage to the gates into the new dimension.

gs_dataGen4 = gs_target.depolarize(gate_noise=0.1)
gs_dataGen5 = gs_dataGen4.increase_dimension(5)
leakGate = pygsti.construction.build_gate( [2,1],[('Q0',),('L0',)] , "LX(pi/4.0,0,2)","gm") # X(pi,Q0)*LX(pi,0,2)

gs_dataGen5['Gx'] = pygsti.objects.compose( gs_dataGen5['Gx'], leakGate, 'gm')
gs_dataGen5['Gy'] = pygsti.objects.compose( gs_dataGen5['Gy'], leakGate, 'gm')
print(gs_dataGen5.gates.keys())

#Some debugging...
#NOTE: with LX(pi,0,2) above, dim 5 test will choose a dimension 3 gateset, which may be sensible
#       looking at the gate matrices in this case... but maybe LX(pi,...) is faulty?
#print(gs_dataGen4)
#print(gs_dataGen5)

#Jmx = GST.JOps.jamiolkowski_iso(gs_dataGen4['Gx'])
#Jmx = GST.JOps.jamiolkowski_iso(gs_dataGen5['Gx'],dimOrStateSpaceDims=[2,1])
#print("J = \n",Jmx)
#print("evals = ",eigvals(Jmx))

dsFake4 = pygsti.construction.simulate_data(gs_dataGen4, dsList, nSamples=1000000, sampleError="binomial", seed=1234)
dsFake5 = pygsti.construction.simulate_data(gs_dataGen5, dsList, nSamples=1000000, sampleError="binomial", seed=1234)
```

```{code-cell} ipython3
:tags: [nbval-skip]

print("Number of gates                        = ",len(gs_target.gates.keys()))
print("Number of fiducials                    =",len(fiducialList))
print("Maximum length for a gate string in ds =",max(map(len,dsList)))
print("Number of LGST strings                 = ",len(lgstList))
print("Number of LSGST strings                = ",map(len,lsgstLists))
```

### Test using dimension-4 fake data

```{code-cell} ipython3
:tags: [nbval-skip]

#Run LGST to get an initial estimate for the gates in gs_target based on the data in ds
# NOTE: with nSamples less than 1M (100K, 10K, 1K) this routine will choose a higher-than-4 dimensional gateset
ds = dsFake4
gs_lgst4 = pygsti.run_lgst(ds, specs, targetGateset=gs_target, svdTruncateTo=4, verbosity=3)
gs_lgst6 = pygsti.run_lgst(ds, specs, targetGateset=gs_target, svdTruncateTo=6, verbosity=3)

#Print chi^2 of 4-dim and 6-dim estimates
chiSq4 = pygsti.chi2(ds, gs_lgst4, lgstList, minProbClipForWeighting=1e-4)
chiSq6 = pygsti.chi2(ds, gs_lgst6, lgstList, minProbClipForWeighting=1e-4)
print("LGST dim=4 chiSq = ", chiSq4)
print("LGST dim=6 chiSq = ", chiSq6)

# Least squares GST with model selection
gs_lsgst = pygsti.do_iterative_mc2gst_with_model_selection(ds, gs_lgst4, 1, lsgstLists, verbosity=2,
                                                           minProbClipForWeighting=1e-3, probClipInterval=(-1e5,1e5))
```

```{code-cell} ipython3
:tags: [nbval-skip]

print(gs_lsgst)
```

### Test using dimension-5 fake data

```{code-cell} ipython3
:tags: [nbval-skip]

#Run LGST to get an initial estimate for the gates in gs_target based on the data in ds
ds = dsFake5
gs_lgst4 = pygsti.run_lgst(ds, specs, targetGateset=gs_target, svdTruncateTo=4, verbosity=3)
gs_lgst6 = pygsti.run_lgst(ds, specs, targetGateset=gs_target, svdTruncateTo=6, verbosity=3)

#Print chi^2 of 4-dim and 6-dim estimates
chiSq4 = pygsti.chi2(ds, gs_lgst4, lgstList, minProbClipForWeighting=1e-2)
chiSq6 = pygsti.chi2(ds, gs_lgst6, lgstList, minProbClipForWeighting=1e-2)
print("LGST dim=4 chiSq = ", chiSq4)
print("LGST dim=6 chiSq = ", chiSq6)

# Least squares GST with model selection
gs_lsgst = pygsti.do_iterative_mc2gst_with_model_selection(ds, gs_lgst4, 1, lsgstLists, verbosity=2, minProbClipForWeighting=1e-3, probClipInterval=(-1e5,1e5), useFreqWeightedChiSq=False, regularizeFactor=1.0, check=False, check_jacobian=False)
```

```{code-cell} ipython3
:tags: [nbval-skip]

print(gs_lsgst)
```
