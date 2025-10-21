---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: New_FPR
  language: python
  name: new_fpr
---

# Fiducial Pair Reduction
The circuits used in standard Long Sequence GST are more than what are needed to amplify every possible gate error.  (Technically, this is due to the fact that the informationaly complete fiducial sub-sequences allow extraction of each germ's *entire* process matrix, when all that is needed is the part describing the amplified directions in model space.) Because of this over-completeness, fewer sequences, i.e. experiments, may be used whilst retaining the desired Heisenberg-like scaling ($\sim 1/L$, where $L$ is the maximum length sequence).  The over-completeness can still be desirable, however, as it makes the GST optimization more robust to model violation and so can serve to stabilize the GST parameter optimization in the presence of significant non-Markovian noise.  Recall that the form of a GST gate sequence is

$$S = F_i (g_k)^n F_j $$

where $F_i$ is a "preparation fiducial" sequence, $F_j$ is a "measurement fiducial" sequence, and "g_k" is a "germ" sequence.  The repeated germ sequence $(g_k)^n$ we refer to as a "germ-power".  There are currently three different ways to reduce a standard set of GST operation sequences within pyGSTi, each of which removes certain $(F_i,F_j)$ fiducial pairs for certain germ-powers.

- **Global fiducial pair reduction (GFPR)** removes the same intelligently-selected set of fiducial pairs for all germ-powers.  This is a conceptually simple method of reducing the operation sequences, but it is the most computationally intensive since it repeatedly evaluates the number of amplified parameters for en *entire germ set*.  In practice, while it can give very large sequence reductions, its long run can make it prohibitive, and the "per-germ" reduction discussed next is used instead. 
<span style="color:red">Note: this form of FPR is deprecated on the latest versions of pygsti's develop branch. We now recommend using per-germ FPR instead. Also note that the current implementation of per-germ FPR will in most cases return smaller experiment designs than the legacy global FPR does.</span>

- **Per-germ fiducial pair reduction (PFPR)** removes the same intelligently-selected set of fiducial pairs for all powers of a given germ, but different sets are removed for different germs.  Since different germs amplify different directions in model space, it makes intuitive sense to specify different fiducial pair sets for different germs.  Because this method only considers one germ at a time, it is less computationally intensive than GFPR, and thus more practical.

- **Per-germ global fiducial pair reduction (PGGFPR)** removes the same intelligently-selected set of fiducial pairs for all powers of a given germ, but different sets are removed for different germs while also taking into account the amplificational properties of a germ set as a whole. This is a two-step process in which we first identify redundancy within a germ set itself due to overlapping amplified directions in parameter space and identifies a subset of amplified parameters for each germ such that collectively we have sensitivity to every direction. In the second stage we select a subset of fiducial pairs for each germ only requiring sensitivity to the subset of amplified parameters of that germ identified in the first stage. This is currently our most effective form of fiducial pair reduction in terms of potential experimental savings, capable with the right settings of achieving experimental designs approaching information theoretic lower bounds in size, but with fewer fiducial pairs comes the potential for detecting non-Markovian effects and potentially less robustness to those effects (the extent to which this is true, or if it is true at all is an active area of research), so caveat emptor. 

- **Random per-germ power fiducial pair reduction (RFPR)** randomly chooses a different set of fiducial pairs to remove for each germ-power.  It is extremly fast to perform, as pairs are just randomly selected for removal, and in practice works well (i.e. does not impair Heisenberg-scaling) up until some critical fraction of the pairs are removed.  This reflects the fact that the direction detected by a fiducial pairs usually has some non-negligible overlap with each of the directions amplified by a germ, and it is the exceptional case that an amplified direction escapes undetected.  As such, the "critical fraction" which can usually be safely removed equals the ratio of amplified-parameters to germ-process-matrix-elements (typically $\approx 1/d^2$ where $d$ is the Hilbert space dimension, so $1/4 = 25\%$ for 1 qubit and $1/16 = 6.25\%$ for 2 qubits).  RFPR can be combined with GFPR or PFPR so that some number of randomly chosen pairs can be added on top of the "intelligently-chosen" pairs of GFPR or PFPR.  In this way, one can vary the amount of sequence reduction (in order to trade off speed vs. robustness to non-Markovian noise) without inadvertently selecting too few or an especially bad set of random fiducial pairs.

## Preliminaries

We now demonstrate how to invoke each of these methods within pyGSTi for the case of a single qubit, using our standard $X(\pi/2)$, $Y(\pi/2)$, $I$ model.  First, we retrieve a target `Model` as usual, along with corresponding sets of fiducial and germ sequences.  We set the maximum length to be 32, roughly consistent with our data-generating model having gates depolarized by 10%.

```{code-cell} ipython3
#Import pyGSTi and the "stardard 1-qubit quantities for a model with X(pi/2), Y(pi/2)"
import pygsti
import pygsti.circuits as pc
from pygsti.modelpacks import smq1Q_XY
import numpy as np

#Collect a target model, germ and fiducial strings, and set 
# a list of maximum lengths.
target_model = smq1Q_XY.target_model()
prep_fiducials = smq1Q_XY.prep_fiducials()
meas_fiducials = smq1Q_XY.meas_fiducials()
germs = smq1Q_XY.germs()
maxLengths = [1,2,4,8,16,32]

opLabels = list(target_model.operations.keys())
print("Gate operation labels = ", opLabels)
```

## Sequence Reduction

Now let's generate a list of all the operation sequences for each maximum length - so a list of lists.  We'll generate the full lists (without any reduction) and the lists for each of the three reduction types listed above.  In the random reduction case, we'll keep 30% of the fiducial pairs, removing 70% of them.

### No Reduction ("standard" GST)

```{code-cell} ipython3
#Make list-of-lists of GST operation sequences
fullStructs = pc.create_lsgst_circuit_lists(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths)

#Print the number of operation sequences for each maximum length
print("** Without any reduction ** ")
for L,strct in zip(maxLengths,fullStructs):
    print("L=%d: %d operation sequences" % (L,len(strct)))
    
#Make a (single) list of all the GST sequences ever needed,
# that is, the list of all the experiments needed to perform GST.
fullExperiments = pc.create_lsgst_circuits(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths)
print("\n%d experiments to run GST." % len(fullExperiments))
```

### Global Fiducial Pair Reduction (GFPR)

```{code-cell} ipython3
fid_pairs = pygsti.alg.find_sufficient_fiducial_pairs(
            target_model, prep_fiducials, meas_fiducials, germs,
            search_mode="random", n_random=10, seed=1234,
            verbosity=1, mem_limit=int(2*(1024)**3), minimum_pairs=2)

# fid_pairs is a list of (prepIndex,measIndex) 2-tuples, where
# prepIndex indexes prep_fiducials and measIndex indexes meas_fiducials
print("Global FPR says we only need to keep the %d pairs:\n %s\n"
      % (len(fid_pairs),fid_pairs))

gfprStructs = pc.create_lsgst_circuit_lists(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairs)

print("Global FPR reduction")
for L,strct in zip(maxLengths,gfprStructs):
    print("L=%d: %d operation sequences" % (L,len(strct)))
    
gfprExperiments = pc.create_lsgst_circuits(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairs)
print("\n%d experiments to run GST." % len(gfprExperiments))
```

### Per-germ Fiducial Pair Reduction (PFPR)

```{code-cell} ipython3
fid_pairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
                target_model, prep_fiducials, meas_fiducials, germs,
                search_mode="random", constrain_to_tp=True,
                n_random=100, min_iterations=50,
                base_loweig_tol= .25, num_soln_returned=1,
                type_soln_returned= 'best',
                retry_for_smaller=True,
                seed=1234, verbosity=1,
                mem_limit=int(2*(1024)**3))
print("\nPer-germ FPR to keep the pairs:")
for germ,pairsToKeep in fid_pairsDict.items():
    print("%s: %s" % (str(germ),pairsToKeep))

pfprStructs = pc.create_lsgst_circuit_lists(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairsDict) #note: fid_pairs arg can be a dict too!

print("\nPer-germ FPR reduction")
for L,strct in zip(maxLengths,pfprStructs):
    print("L=%d: %d operation sequences" % (L,len(strct)))

pfprExperiments = pc.create_lsgst_circuits(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairsDict)
print("\n%d experiments to run GST." % len(pfprExperiments))
```

### Per-germ Fiducial Pair Reduction (PFPR) with Greedy Search Heuristics

In addition to the implementation of per-germ fiducial pair reduction above, which supports either a brute force sequential or random search heuristic, there is also an implementation using a greedy search heuristic combined with fast low-rank update-based techniques for significantly faster execution, particularly when generating experiment designs for two-or-more qubits.

```{code-cell} ipython3
fid_pairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ_greedy(target_model, prep_fiducials, meas_fiducials,
                                                germs, verbosity=1)
print("\nPer-germ FPR to keep the pairs:")
for germ,pairsToKeep in fid_pairsDict.items():
    print("%s: %s" % (str(germ),pairsToKeep))

pfprStructs_greedy = pc.create_lsgst_circuit_lists(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairsDict) #note: fid_pairs arg can be a dict too!

print("\nPer-germ FPR reduction (greedy heuristic)")
for L,strct in zip(maxLengths,pfprStructs_greedy):
    print("L=%d: %d operation sequences" % (L,len(strct)))

pfprExperiments_greedy = pc.create_lsgst_circuits(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairsDict)
print("\n%d experiments to run GST." % len(pfprExperiments_greedy))
```

### Per-germ Global Fiducial Pair Reduction (PFPR)

As mentioned above, the per-germ global FPR scheme is a two step process. First we identify a reduced set of amplified parameters for each germ to require sensitivity to, and then next we identify reduced sets of fiducials with sensitivity to those particular parameters.

```{code-cell} ipython3
#Note that we are setting the assume_real flag to True below as we know that we am working in the Pauli basis and as such the 
#process matrices for the germs will be real-valued allowing for memory savings and somewhat faster performance. 
#If you're working with a non-hermitian basis or aren't sure keep this set to it's default value of False.
#Likewise, float_type specifies the numpy data type to use, and is primarily useful either in conjunction with
#assume_real, or when needing to fine-tune the memory requirements of the algorithm (running this algorithm for
#more than 2-qubits can be very memory intensive). When running this function for more than two-qubits, consider
#setting the mode kwarg to 'RRQR', which is typically significantly faster for larger qubit counts, but is slightly
#less performant in terms of the cost function of the returned solutions.
germ_set_spanning_vectors, _ = pygsti.alg.germ_set_spanning_vectors(target_model, germs, assume_real=True, float_type= np.double)

#Next use this set of vectors to find a sufficient reduced set of fiducial pairs.
#Alternatively this function can also take as input a list of germs
fid_pairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ_global(target_model, prep_fiducials, meas_fiducials,
                                                germ_vector_spanning_set=germ_set_spanning_vectors, verbosity=2)
print("\nPer-germ Global FPR to keep the pairs:")
for germ,pairsToKeep in fid_pairsDict.items():
    print("%s: %s" % (str(germ),pairsToKeep))

pggfprStructs = pc.create_lsgst_circuit_lists(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairsDict) #note: fid_pairs arg can be a dict too!

print("\nPer-germ Global FPR reduction")
for L,strct in zip(maxLengths,pggfprStructs):
    print("L=%d: %d operation sequences" % (L,len(strct)))

pggfprExperiments = pc.create_lsgst_circuits(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    fid_pairs=fid_pairsDict)
print("\n%d experiments to run GST." % len(pggfprExperiments))
```

### Random Fiducial Pair Reduction (RFPR)

```{code-cell} ipython3
#keep only 30% of the pairs
rfprStructs = pc.create_lsgst_circuit_lists(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    keep_fraction=0.30, keep_seed=1234)

print("Random FPR reduction")
for L,strct in zip(maxLengths,rfprStructs):
    print("L=%d: %d operation sequences" % (L,len(strct)))
    
rfprExperiments = pc.create_lsgst_circuits(
    opLabels, prep_fiducials, meas_fiducials, germs, maxLengths,
    keep_fraction=0.30, keep_seed=1234)
print("\n%d experiments to run GST." % len(rfprExperiments))
```

## Running GST
In each case above, we constructed (1) a list-of-lists giving the GST operation sequences for each maximum-length stage, and (2) a list of the experiments.  In what follows, we'll use the experiment list to generate some simulated ("fake") data for each case, and then run GST on it.  Since this is done in exactly the same way for all three cases, we'll put all of the logic in a function.  Note that the use of fiducial pair redution requires the use of `run_long_sequence_gst_base`, since `run_long_sequence_gst` internally builds a *complete* list of operation sequences.

```{code-cell} ipython3
#use a depolarized version of the target gates to generate the data
mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.001)

def runGST(gstStructs, exptList):
    #Use list of experiments, expList, to generate some data
    ds = pygsti.data.simulate_data(mdl_datagen, exptList,
            num_samples=1000,sample_error="binomial", seed=1234)
    
    #Use "base" driver to directly pass list of circuit structures
    return pygsti.run_long_sequence_gst_base(
        ds, target_model, gstStructs, verbosity=1)

print("\n------ GST with standard (full) sequences ------")
full_results = runGST(fullStructs, fullExperiments)

print("\n------ GST with GFPR sequences ------")
gfpr_results = runGST(gfprStructs, gfprExperiments)

print("\n------ GST with PFPR sequences ------")
pfpr_results = runGST(pfprStructs, pfprExperiments)

print("\n------ GST with PFPR sequences (greedy heuristic) ------")
pfpr_results_greedy = runGST(pfprStructs_greedy, pfprExperiments_greedy)

print("\n------ GST with PGGFPR sequences ------")
pggfpr_results = runGST(pggfprStructs, pggfprExperiments)

print("\n------ GST with RFPR sequences ------")
rfpr_results = runGST(rfprStructs, rfprExperiments)
```

Finally, one can generate reports using GST with reduced-sequences:

```{code-cell} ipython3
pygsti.report.construct_standard_report(full_results, title="Standard GST Strings Example"
                                       ).write_html("../../tutorial_files/example_stdstrs_report")
pygsti.report.construct_standard_report(gfpr_results, title="Global FPR Report Example"
                                        ).write_html("../../tutorial_files/example_gfpr_report")
pygsti.report.construct_standard_report(pfpr_results, title="Per-germ FPR Report Example"
                                        ).write_html("../../tutorial_files/example_pfpr_report")
pygsti.report.construct_standard_report(pfpr_results_greedy, title="Per-germ FPR (Greedy Heuristic) Report Example"
                                        ).write_html("../../tutorial_files/example_pfpr_greedy_report")
pygsti.report.construct_standard_report(pggfpr_results, title="Per-germ Global FPR Report Example"
                                        ).write_html("../../tutorial_files/example_pggfpr_report")
pygsti.report.construct_standard_report(rfpr_results, title="Random FPR Report Example"
                                        ).write_html("../../tutorial_files/example_rfpr_report")
```

If all has gone well, the [Standard GST](../../tutorial_files/example_stdstrs_report/main.html),
[GFPR](../../tutorial_files/example_gfpr_report/main.html),
[PFPR](../../tutorial_files/example_pfpr_report/main.html),
[PFPR (Greedy)](../../tutorial_files/example_pfpr_greedy_report/main.html)
[PGGFPR](../../tutorial_files/example_pggfpr_report/main.html)
and
[RFPR](../../tutorial_files/example_rfpr_report/main.html),
reports may now be viewed.
The only notable difference in the output are "gaps" in the color box plots which plot quantities such as the log-likelihood across all operation sequences, organized by germ and fiducials.


