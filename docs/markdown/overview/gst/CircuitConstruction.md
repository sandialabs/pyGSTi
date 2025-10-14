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

# Circuit Construction
This tutorial discusses the creation of `Circuit` objects for use with Gate Set Tomography (GST).  If you're not sure what a `Circuit` is, you might want to read through the [tutorial on circuits](../Circuit.ipynb) first. 

While pyGSTi allows one to create arbitrary lists of circuits, the GST algorithms have been designed to work well with circuits that have a particular structure.  At the beginning and the end of each string are "preparation fiducial" and "measurement fiducial" circuits (or "sequences"), respectively, whose purpose is to extend the native preparation and measurment operations to informationally complete sets.  In between the fiducial sequences is a "germ" sequence, or just "germ", that is repeated some number of times.  The purpose of the repeated-germ sequence is to amplify one or more particular types of errors.  By considering an entire set of germs (called an "amplificationally complete" set), *all* possible types of errors are amplified, giving the GST algorithms high sensitivity to all errors.  The longer the sequences, that is, the more repetitions of the germs, the higher the sensitivity.  

While this may seem to imply that one should just to incredibly long sequences, there are several caveats.
1) Because the gates are imperfect, there is some practical length (possibly germ-dependent) at which the initial state is so depolarized that it cannot be distinguished from the maximally-mixed state.  This sets an upper bound on the length of useful circuits.
2) Because GST's numerical algorithms use local gradient-based optimization methods, starting with circuits of maximal length would often result in getting trapped in a local minima of the function being optimized (often the likelihood).  The GST algorithms iterate through multiple circuit lists, using the result of the previous iteration to seed the next one.  When operation sequences lists are chosen well they are able guide the GST optimization to a global (or near-global) maximum by *gradually* making the objective function sharper.

In practice, well-chosen lists of operation sequences can be formed as follows.  First, we defne an increasing sequence of **maximum-lengths**, usually powers of 2, e.g. `[1,2,4,8,16,...]`, which terminates with the longest useful sequence length.  Secondly, we select preparation- and measurement-fiducial `Circuit` lists, and a list of germ circuits.  These lists are chosen such that the fiducial sequences result in informationally complete preparations and measurements, and the germs are amplificationally complete.  For each maximum-length $L$, the set of all

`preparation_fiducial + repeated_germ + measurement_fiducial`

circuits, where `preparation_fiducial` ranges over all preparation fiducials, `measurement_fiducial` ranges over all measurement fiducials, and `repeated_germ` ranges over all the germs *repeated such that the length of `repeated_germ` does not exceed $L$*.  To further increase the robustness of the numerics, the circuit list corresponding to $L$ will also contain all of the sequences from earlier (lower) values of $L$.  This procedure thus creates a list of operation sequences for each $L$ that are designed for use by pyGSTi's "long-sequence" GST algorithms.

The fiducial and germ lists are obtained by pyGSTi's "fiducial selection" and "germ selection" algorithms, which are explained in more detail in the [tutorial on fiducial and germ selection](../algorithms/advanced/GST-FiducialAndGermSelection.ipynb).  This this tutorial we'll be using one of pyGSTi's built-in "model packs" (see the [model pack tutorial](ModelPacks.ipynb)), for which fiducial and germ circuits are pre-computed:

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XY #import the standard X(pi/2), Y(pi/2) model info

prep_fiducials = smq1Q_XY.prep_fiducials()
meas_fiducials = smq1Q_XY.meas_fiducials()
germs = smq1Q_XY.germs()

print("Prep fiducials:\n", prep_fiducials)
print("Meas fiducials:\n", meas_fiducials)
print("Germs:\n",germs)
```

We can now construct a the lists of sequences used by a long-sequence GST algorithm by defining a list of maximum lengths and calling `create_lsgst_circuit_lists`:

```{code-cell} ipython3
maxLengths = [1,2,4]
lsgst_lists = pygsti.circuits.create_lsgst_circuit_lists(['Gx','Gy'], prep_fiducials, meas_fiducials, germs, maxLengths)

#Print the result.  Note that larger L lists also contain all the elements of lower-L lists.  
# Note also that germs which are length 4 only show up in the L=4 list, and there are only "repeated" once.
for i,lst in enumerate(lsgst_lists):
    print("\nList %d (max-length L=%d): %d Circuits" % (i,maxLengths[i],len(lst)))
    print('\n'.join([c.str for c in lst])) # Note: use ".str" to get a *single-line* string representation
```

## Why settle for lists, when you can have structures?

While a list-of-lists such as `lsgst_lists` is all that is needed by the core long-sequence GST algorithms, one can also construct, in similar fashion, a list of `pygsti.objects.CircuitStructure` objects.  These objects mimic a list of `Circuits` in many ways, but retain the structure of each circuit so that per-circuit quantities may later be plotted according to their decomposition into fiducials, germ, and $L$.  The `allstrs` member of a `CircuitStructure` allows access to the underlying list of `Circuit` objects.  We'll demonstrate this when we get to plotting.  For now, just think of this as a slightly richer way to generate the circuits needed by a GST algorithm that facilitates post-processing and analysis.

```{code-cell} ipython3
#Note: same arguments as make_lsgst_lists
lsgst_structs = pygsti.circuits.create_lsgst_circuit_lists(['Gx','Gy'], prep_fiducials, meas_fiducials, germs, maxLengths)
print(type(lsgst_structs[0]))
for i,struct in enumerate(lsgst_structs):
    print("Struct %d (L=%d) has %d strings" % (i,maxLengths[i],len(struct)))
```

## List of experiments
If you're taking data, not running a GST algorithm, then you just want a *single* list of all the circuits that the GST algorithm will need.  Since most of the time the lists are *nested*, that is, all of the lower-$L$ (shorter) circuits appear in the higher-$L$ (longer) circuits, the final element of `lsgst_lists` is often the list of required experiments.  However, in advanced usages this is not always the case, and so there is a dedicated function, `make_lsgst_experiment_list` which performs this constrution:

```{code-cell} ipython3
#Note: same arguments as make_lsgst_lists
lsgst_experiment_list = pygsti.circuits.create_lsgst_circuits(['Gx','Gy'], prep_fiducials,
                                                                      meas_fiducials, germs, maxLengths)
print("%d experiments to do..." % len(lsgst_experiment_list))
```

## Example: generating LSGST sequences by hand
As a final full-fledged example we demonstrate the use of pyGSTi's more general `Circuit` creation and processing functions to create the lists of LSGST circuits. The following example functions are very similar to `pygsti.construction.create_lsgst_circuit_lists` except we assume that the preparation and measurement fiducial sequences are the same.

```{code-cell} ipython3
import pygsti.circuits as pc

def my_make_lsgst_lists(gateLabels, fiducialList, germList, maxLengthList):
    singleOps = pc.to_circuits([(g,) for g in gateLabels])
    lgstStrings = pc.create_lgst_circuits(fiducialList, fiducialList, gateLabels)
    lsgst_list = pc.to_circuits([ () ]) #running list of all strings so far
    
    if maxLengthList[0] == 0:
        lsgst_listOfLists = [ lgstStrings ]
        maxLengthList = maxLengthList[1:]
    else: lsgst_listOfLists = [ ]
        
    for maxLen in maxLengthList:
        lsgst_list += pc.create_circuits("f0+R(germ,N)+f1", f0=fiducialList,
                                           f1=fiducialList, germ=germList, N=maxLen,
                                           R=pc.repeat_with_max_length,
                                           order=('germ','f0','f1'))
        lsgst_listOfLists.append( pygsti.remove_duplicates(lgstStrings + lsgst_list) )

    print("%d LSGST sets w/lengths" % len(lsgst_listOfLists), list(map(len,lsgst_listOfLists)))
    return lsgst_listOfLists

my_lsgst_lists = my_make_lsgst_lists(['Gx','Gy'], prep_fiducials, germs, maxLengths)    
print('\n'.join(['%d strings' % len(l) for l in my_lsgst_lists]))
```

```{code-cell} ipython3

```
