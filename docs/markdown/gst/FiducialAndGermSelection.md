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

# Fiducial and Germ Selection

This notebook demonstrates how to generate sets of fiducial and germ sequences which form the building blocks of the operation sequences used by long-sequence GST.  As described in [GST circuits tutorial](CircuitConstruction), by structuring the GST sequences as

`preparation_fiducial + repeated_germ + measurement_fiducial`

long-sequence GST is highly sensitive to *all possible* (within the space of allowed models).  Furthermore, by iteratively increase the number of germ repetitions in `repeated_germ`, pyGSTi's iterative algorithms are able to avoid (usually!) local optima.   

Both germ and fiducial sets are determined for a given "target" model.  We currently assume that this model contains unitary gates, such that infinitely many gates may be performed without moving away from a pure state.  It is almost always the case that the desired "target" operations are unitary, so this isn't a burdensome assumption.  If this isn't the case, one should find perform fiducial and germ selection on the nearest unitary model.

## Fiducial Selection: the theory
The purpose of the preparation and measurement fiducial sequences, $\{F_i\}$ and $\{H_i\}$, is to prepare a sufficiently diverse set of input states, and a sufficiently diverse set of measurements, to completely probe an operation of interest - defined as the map that lies *between* the fiducials.  This is achieved if (and only if) the input states $\{\rho_i\}\equiv  \{F_i|\rho\rangle\rangle\}$ and the measurement effects $\{E_j\} \equiv \{\langle\langle E|H_j\}$ are both *informationally complete* (IC).  A set of matrices is IC if and only if it spans the vector space of density matrices.  For a Hilbert space of dimension $d$ this requires at least $d^2$ linearly independent elements.

In general, *any* randomly chosen set of $d^2$ states or effects will be IC.  So, for single-qubit GST, we could choose $d^2=4$ random fiducial sequences.  However, while the resulting $\{\rho_i\}$ and $\{E_j\}$ will almost certainly be linearly independent, they may be *close* to linearly dependent.  

To evaluate a set of fiducials we form a matrix $M$, which will allow us to quantify how linearly independent the resulting $\{\rho_i\}$ or $\{E_j\}$ will be.  If we are evaluating a set of preparation fiducials, then the i^th column of $M$ is $F_i|\rho\rangle\rangle$; if measurement fiducials, the columns are $\langle\langle E|H_i$.  This notation assumes a single native preparation or measurement effect; in the case when there are more this simply adds more columns $M$.

We then form the square matrix $MM^T$.  We either then score the fiducial set as the number of fiducials times the sum of the reciprocals of the eigenvalues of $MM^T$ (`scoreFunc = 'all'`) or by the number of fiducials times the reciprocal of the smallest eigenvalue of $MM^T$ (`scoreFunc = 'worst'`).  In both cases, a lower score is better.

In the `'all'` case, we are attempting to make all the fiducials as uniformly informationally complete as possible, that is, we are trying to make our fiducial set as sensitive as possible to all directions in Hilbert-Schmidt space.  In the `'worst'` case, we are instead attempting minimize our insensitivity to the direction in Hilbert-Schmidt space that we are least sensitive to.

## Germ Selection: the theory
The defining property which makes a set of operation sequences a "complete germ set" is the amplification of *all* possible gate errors.  More precisely, the repetition of a complete set of germs, sandwiched between assumed-IC preparation and measurement fiducial sequences, will yield a sensitivity that scales with the number of times each germ is repeated to *any* direction in the space of `Models` (defined  by the model's parameterization).  This completeness is relative a "target" model, just as in fiducial selection.  While the detailed mathematics behind germ selection is beyond the scope of this tutorial, the essence of the algorithm is as follows.  The Jacobian $J$ of a potential set of germs relative to the target `Model`'s parameters is constructed and the eigenvalues of $J^\dagger J$ are computed.  If the number of large eigenvalues values equals its maximum - the number of non-gauge `Model` parameters - then the germ set is deemed "amplificationally complete", and will amplify *any* gate error.  More specifically, a germ set is scored using a combination of 1) the number of eigenvalues values above some threshold and 2) either the `'all'` or `'worst'` scoring function applied to the eigenvalues of $J^\dagger J$. Several technical points make this slightly more complicated:
- only *gate* errors can be amplified, since only gates can be repeated.  Thus, when computing the number of non-gauge parameters of the target model, we really mean the target model without any SPAM operations.
- typical perfect gates (e.g. $\pi/2$ rotations) may contain symmetries which decrease the number of non-gauge parameters only at that perfect point in gate-set-space.  As such, we add random unitary perturbations to the target `Model` before performing the Jacobian analysis to mitigate the possibility of mischaracterizing a direction as being amplified when it isn't for all `Model`s except the perfect one.
- In the Jacobian analysis, each germ is *twirled* to simulate the effect of it echoing out all directions except those that commute with it.

If not all that made perfect sense, do not worry.  The remainder of this tutorial focuses on how to do fiducial or germ selection using pyGSTi, and does not rely on a rock solid theoretical understanding of the methods.

+++

## Fiducial and Germ selection in practice
The selection of fiducial and germ sequences in pyGSTi is similar in that each uses a numerical optimization which considers different possible "candidate" sets, and tries to find the one which scores the best.  The modules `pygsti.algorithms.fiducialselection` and `pygsti.algorithms.germselection` contain the algorithms relevant to each type of sequence selection.

```{code-cell} ipython3
import pygsti
import pygsti.algorithms.fiducialselection as fidsel
import pygsti.algorithms.germselection as germsel
from pygsti.modelpacks import smq1Q_XYI
import numpy as np
```

We'll begin by constructing a 1-qubit $X(\pi/2)$, $Y(\pi/2)$, $I$ model for which we will find germs and fiducials.

```{code-cell} ipython3
target_model = smq1Q_XYI.target_model('full TP')
```

### Automated "*laissez-faire*" approach

+++

We begin by demonstrating the most automated and hands-off approach to computing fiducials and germs -- by just providing the target model and accepting the defaults for all remaining optional arguments.  Note that one may compute these in either order - fiducial selection is usually much faster, since the required computation is significantly less.

```{code-cell} ipython3
prepFiducials, measFiducials = fidsel.find_fiducials(target_model)
```

```{code-cell} ipython3
germs = germsel.find_germs(target_model, seed = 1234)
```

Now that we have germs and fiducials, we can construct the list of experiments we need to perform in
order to do GST. The only new things to provide at this point are the sizes for the experiments we want
to perform (in this case we want to perform between 0 and 256 gates between fiducial pairs, going up
by a factor of 2 at each stage).

```{code-cell} ipython3
maxLengths = [2**n for n in range(8 + 1)]
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    target_model, prepFiducials, measFiducials, germs, maxLengths)
```

### Less-automated, more control: useful optional arguments

+++

There are many ways you can assume more control over the experiment design process. We'll only demonstrate
a few here, but all options are discussed in the documentation for the various functions we've used.

+++

#### Different algorithms
There are a number of different algorithms available for germ selection. You can choose a non-default
algorithm by specifying the `algorithm` keyword argument.  Each of the available algorithms has a set of keyword arguments of its own with which you can more precisely specify how you want it to behave. These keyword arguments can be passed as a dictionary to `find_germs` through the keyword argument `algorithm_kwargs`.

`find_germs` and `find_fiducials` support supports the algorithms: 'greedy' (default for `find_germs`), 'grasp' (default for `find_fiducials`) and 'slack'.

Each of these algorithms can have different computational performance, and for systems of two-or-more qubits the 'greedy' algorithms are typically the most performant computationally (when run with certain `mode` settings, as discussed in the next section).

```{code-cell} ipython3
graspGerms = germsel.find_germs(target_model, algorithm='grasp', algorithm_kwargs={'iterations': 1}, seed = 1234)
```

```{code-cell} ipython3
slackGerms = germsel.find_germs(target_model, algorithm='slack', algorithm_kwargs={'slack_frac': 0.25}, seed = 1234)
```

Fiducial selection can be controlled in much the same way, using the same algorithms.

```{code-cell} ipython3
graspPrepFids, slackMeasFids = fidsel.find_fiducials(target_model, algorithm='slack',
                                                         algorithm_kwargs={'slack_frac': 0.25})
```

```{code-cell} ipython3
greedyPrepFids, greedyMeasFids = fidsel.find_fiducials(target_model, algorithm='greedy')
```

#### Different Modes
In addition to there being multiple options for the algorithm to use, there are multiple modes that the algorithms can be run in.
For `find_germs` the `mode` kwarg acts as a flag to indicate the caching scheme used for storing the Jacobians for the candidate
germs. Default value of 'allJac' caches all of the Jacobians and requires the most memory. 'singleJac' doesn't cache anything and instead generates these Jacobians on the fly. The final option, 'compactEVD', is currently only configured to work with the greedy search algorithm. When selected the compact eigenvalue decomposition/compact SVD of each of the Jacobians is constructed and is cached. This uses an intermediate amount of memory between 'singleJac' and 'allJac'. When compactEVD mode is selected we also perform the greedy search iterations using an alternative method based on low-rank update techniques, which means in practice this mode can be orders-of-magnitude faster than the other modes, though typically only for two-or-more qubits. This alternative approach means that this mode also only works with the score function option set to 'all'. (Note: this mode can also be a bit more finicky than other modes, so be prepared to tinker a bit, you can see hints of this finickiness below).

`find_germs` also accepts the kwargs `assume_real` and `float_type`. `assume_real` is a flag indicating that the process matrices for germs will be real-valued, as is the case when working with the Pauli basis, which can allow for unlocking certain optimizations in these cases. `float_type` allows fine-tuning the numpy floating point number types used in the computation, which can allow for better computational performance and a lower memory-footprint in the correct circumstances.

```{code-cell} ipython3
greedyGerms_compactEVD = germsel.find_germs(target_model, algorithm='greedy', seed = 1234, mode='compactEVD', verbosity=1,
                                            assume_real=True, float_type=np.double)
```

#### Germ and fiducial lengths
We can also adjust some algorithm-independent parameters for germ and fiducial selection. For instance, all
of the algorithms currently rely on having a pool of circuit from which they construct germs and fiducials.
The size of this pool is set by specifying the longest germ or fiducial to include in this pool.

For germ selection, the default maximum germ length is 6.

+++

We can try and set the maximum germ length to 5 and see what we get.

```{code-cell} ipython3
germsMaxLength5 = germsel.find_germs(target_model, candidate_germ_counts={5: 'all upto'}, seed=1234)
```

If we get too ambitious in shortening the maximum
germ length, germ selection won't be able to find an amplificationally complete germ set. It will send a warning
message to `stderr` if this happens and return `None`.

```{code-cell} ipython3
germsMaxLength3 = germsel.find_germs(target_model, candidate_germ_counts={3: 'all upto'}, seed=1234)
print(germsMaxLength3)
```

As was the case with germ selection, if you are too aggressive in limiting fiducial length you may
constrain the algorithm to the extent that it cannot even find a set of fiducials to generate an
informationally complete set of states and measurements. In that case, it will also send a warning
message to `stderr` and return `None` for the preparation and measurement fiducial sets.

```{code-cell} ipython3
incompletePrepFids, incompleteMeasFids = fidsel.find_fiducials(target_model, candidate_fid_counts={1:'all upto'})
```

```{code-cell} ipython3
print(incompleteMeasFids, incompletePrepFids)
```

#### Set requirements
There are several natural things to require of the returned germ and fiducial sets. For germ sets, you will usually
want the individual gates to be included as germs. If for some reason you don't want this, you can set the
*force* keyword argument to `None`.

```{code-cell} ipython3
nonSingletonGerms = germsel.find_germs(target_model, force=None, candidate_germ_counts={5: 'all upto'},
                                           algorithm='greedy', seed=1234)
```

In fiducial selection, it is likewise natural to require the empty operation sequence to be in the
fiducial set. This requirement may be disabled by setting *forceEmpty* to `False`. It is also
often desireable for identity gates to be left out of fiducials, since they add no diversity
to the set of states and measurements generated. You can allow identity gates in fiducials by
setting *omit_identity* to `False`.

A more common modification to the fiducial set requirements is to leave out additional gates from fiducials.
This might be desireable if you have a multi-qubit system and you expect your 2-qubit gates to be of lower
fidelity than your single-qubit gates. In this case you might want to construct fiducials from only
single-qubit gates. A list of gates that you would like to omit from your fiducials can be provided as a
list of operation labels to the *ops_to_omit* keyword argument.

Our model doesn't have multi-qubit gates, but we can demonstrate several pieces of this
functionality by setting *omit_identity* to `False` and omitting the identity manually using
*ops_to_omit*.

```{code-cell} ipython3
from pygsti.baseobjs import Label
omit_identityPrepFids, omit_identityMeasFids = fidsel.find_fiducials(target_model, omit_identity=False,
                                                                       ops_to_omit=[Label(())])
```

#### The 'Lite'/'Standard' Germ Set

So far we have implicitly been constructing examples of what we call the 'Robust' germ set. This is a germ set designed to be robust against second-order effects that result in a plateuing of our sensitivity at long circuit depths. Unless your system has very low error rates, it is likely that even with this second order effect you'll be decoherence limited long before entering the regime where this effect is significant. By setting the kwarg `randomize` to `False` you can change the behavior of germ selection such that it produces a significantly smaller, but also somewhat less robust, germ set called the 'Standard' or 'Lite' germ set. While not the default behavior of `find_germs`, we've found that for most applications the lite germ set is more than sufficient, so we recommend using it unless there is specific reason to prefer the robust experiment design (e.g. if you need high precision estimates for an idle gate known to have a very high fidelity). For more on these different germ sets see [this paper](https://arxiv.org/abs/2307.15767).

```{code-cell} ipython3
liteGerms = germsel.find_germs(target_model, randomize=False, algorithm='greedy', verbosity=1,
                                            assume_real=True, float_type=np.double)
```

#### Verbosity
The various algorithms can tell you something of what's going on with them while they're running. By default,
this output is silenced, but it can be turned on using the *verbosity* keyword argument.
- A verbosity level of 1 is the default. This prints out what algorithm is being used, the returned set, and the score of that set.
- A verbosity level of 0 silences all output (other than warnings that things have gone wrong).
- A verbosity level of $n+1$ where $n\geq0$ prints the output of verbosity level 1 in addition to the output that the current algorithm displays when its own verbosity is set to $n$.
