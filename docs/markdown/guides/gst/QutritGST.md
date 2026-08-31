---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Qutrit GST

Two ions in one trap, read out by collecting fluorescence from both at once, are not a two-qubit system. The detector counts photons; it cannot say which ion emitted them. Three outcomes come back: no ion bright, one bright, both bright. The states that produce them span the symmetric subspace of the two-qubit Hilbert space, which is three-dimensional, so the thing to characterize is a qutrit, and the third level is a third of the system rather than somewhere population escapes to.

```{code-cell} ipython3
import numpy as np
from numpy import pi

import pygsti
from pygsti.models import qutrit
from pygsti.algorithms.fiducialselection import find_fiducials
from pygsti.algorithms.germselection import find_germs
from pygsti.protocols import ProtocolData, StandardGST, StandardGSTDesign
```

## The target model

`create_qutrit_model` builds a target model for this situation: four operations on a single line label `T0`, three of them two-qubit unitaries projected onto the symmetric subspace. `Gx` is $X(\theta) \otimes X(\theta)$ and `Gy` is $Y(\theta) \otimes Y(\theta)$, one single-ion rotation applied identically to both ions at once. Applying the *same* rotation to both is what keeps the state inside the symmetric subspace, since any $U \otimes U$ commutes with the swap. `Gm` is the Mølmer-Sørensen unitary $\exp(-i \theta\, A \otimes A / 2)$ with $A = \cos\phi\, \sigma_x + \sin\phi\, \sigma_y$, which at `ms_local=0` reduces to $\exp(-i \theta\, \sigma_x \otimes \sigma_x / 2)$. `Gi` is the identity.

State preparation is $|00\rangle$, and the three POVM effects project onto $|00\rangle$, the symmetric combination of $|01\rangle$ and $|10\rangle$, and $|11\rangle$. Level $i$ is therefore $i$ bright ions, as the effect labels below show.

`basis="qt"` selects pyGSTi's qutrit basis, whose nine elements are two-qubit Pauli products projected onto that same symmetric subspace and orthonormalized under the trace inner product. The operator basis is thus built the way the gates are, and its labels are still written in two-qubit Pauli terms.

```{code-cell} ipython3
target_model = qutrit.create_qutrit_model(error_scale=0, x_angle=pi/2, y_angle=pi/2,
                                          ms_global=pi/2, ms_local=0, basis="qt")

print("state space:", target_model.state_space)
print("operations: ", list(target_model.operations.keys()))
print("effects:    ", list(target_model.povms['Mdefault'].keys()))
print("basis:      ", target_model.basis.labels)
```

## Fiducials and germs

None of pyGSTi's current modelpacks describes a qutrit (the one that does, `stdQT_XYIMS`, sits in the deprecated `legacy` package), so this page searches for fiducials and germs rather than importing them. Both searches finish well under a second here. The candidate lists are all circuits up to a fixed length over the gate alphabet, so they grow exponentially in that length.

```{code-cell} ipython3
fiducialPrep, fiducialMeasure = find_fiducials(target_model, candidate_fid_counts={4: 'all upto'},
                                               algorithm='greedy')
germs = find_germs(target_model, randomize=False, candidate_germ_counts={4: 'all upto'},
                   mode='compactEVD', float_type=np.double)
```

```{code-cell} ipython3
print("%d prep fiducials" % len(fiducialPrep))
print("%d meas fiducials" % len(fiducialMeasure))
print("%d germs" % len(germs))
```

## Circuits and data

`StandardGSTDesign` assembles the germ-power circuits sandwiched between fiducials. The dimension travels with the processor specification, which `create_processor_spec` reads off the model as one qudit of dimension three named `T0`.

```{code-cell} ipython3
maxLengths = [1, 2, 4]
design = StandardGSTDesign(target_model.create_processor_spec(), fiducialPrep, fiducialMeasure,
                           germs, maxLengths)
print("%d circuits" % len(design.all_circuits_needing_data))
```

Those circuits are what an experiment would have to run. Written out as an empty dataset they become a template: one row per circuit, three count columns to fill in.

```{code-cell} ipython3
pygsti.io.write_empty_dataset("../../../example_files/dataTemplate_qutrit_maxL=4.txt",
                              design.all_circuits_needing_data,
                              "## Columns = 0bright count, 1bright count, 2bright count")
```

For a real experiment that template is the stopping point: take the data, fill the columns, and read the file back with `pygsti.io.load_dataset`. The rest of this page runs on simulated counts instead, drawn from a depolarized copy of the target model.

```{code-cell} ipython3
mdl_datagen = target_model.depolarize(op_noise=0.05, spam_noise=0.01)
DS = pygsti.data.simulate_data(mdl_datagen, design.all_circuits_needing_data,
                               num_samples=1000, sample_error='multinomial', seed=2018)
data = ProtocolData(design, DS)
```

## Running GST

`StandardGST` in `CPTPLND` mode fits a Lindblad-parameterized CPTP model. Nothing in the call is qutrit-specific; the three levels arrive with the model and the design.

The `optimizer` argument caps the Levenberg-Marquardt iterations at 50 to keep this page quick. Every stage then stops on the cap rather than on its own convergence test, and says so in the output below. Drop the cap when the answer matters.

```{code-cell} ipython3
:tags: [output_scroll]

result = StandardGST(modes=('CPTPLND',), target_model=target_model,
                     optimizer={'maxiter': 50}, verbosity=4).run(data)
```

## The report

The standard report needs nothing qutrit-specific either.

```{code-cell} ipython3
ws = pygsti.report.construct_standard_report(
    result, "Example Qutrit Report", verbosity=3
).write_html('../../../example_files/sampleQutritReport', connected=True, auto_open=False, verbosity=3)
```

Served with these docs: <a href="../../../reports/sampleQutritReport/main.html">sampleQutritReport</a>.
