---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Robust GST using Total Variation Distance (TVD)

This notebook demonstrates a robustness feature in pyGSTi: performing long-sequence GST using different final-stage objective functions.

We compare two loss functions to be used in obtaining the final gate set model, $M$:

- *Negative log-likelihood* (denoted `-logl`)
- *Total variation distance* (denoted `tvd`)

The former loss is the standard, since it corresponds to GST based on maximum liklihood estimation in a Markovian error model.
When there is a Markovian model that accurately models all available data, minimizing ``-logl`` is a good choice.

However, if a dataset contains a small fraction of "bad" circuits (e.g., from data corruption, drift events, or other outliers), likelihood-based objectives can be pulled strongly toward explaining those circuits.

An alternative is objectives based on the *total variation distance* (TVD) of predicted and observed circuit outcome distributions:

$$
\lVert p - q \rVert_{\text{tvd}} = \frac{1}{2}\sum_i |p_i - q_i|.
$$

There are different ways that we can aggregate the per-circuit TVDs into a final loss function.
This notebook uses our default method, which is to just sum over all circuits:

$$
F_{\text{tvd}}(M; \text{data}) = \sum_{\text{circuits } c} \lVert ~ p_{\text{obs}}(\cdot\mid c) - p_{M}(\cdot\mid c)~ \rVert_{\text{tvd}},
$$

This objective can be less sensitive to extreme outliers because it directly penalizes distribution mismatch rather than log-probability of rare events.

We will:
1. Simulate a “clean” dataset from a slightly depolarized 1-qubit gate set.
2. Create a *corrupted* dataset by randomly selecting a small fraction of circuits and replacing their observed outcome frequencies with random distributions.
3. Fit GST models using both objectives (`-logl` and `tvd`) on both datasets.
4. Cross-evaluate the fitted models to show how the objective affects sensitivity to corrupted data.

The next cell defines three utility functions.

- ``run_dataset_and_finalobjective``: run GST on a provided dataset with a chosen final objective (`'logl'` or `'tvd'`). This function might be useful in workflows of your own.
- ``build_all_results_for_demo``, which runs GST and produces two ModelEstimateResults objects, one for each dataset. Both of these ModelEstimateResults objects hold results from training on the original dataset and the corrupted dataset.
- ``print_summary_table_for_demo`` summarizes the output of ``build_all_results_for_demo`` in a way that highlights the effect of training a TVD loss even when we care about robustness *and* log-likelihood. This is an alternative to generating an HTML report.

```{code-cell} ipython3
from typing import Literal

import pygsti
from pygsti.modelpacks import smq1Q_XYI, GSTModelPack
from pygsti.protocols import GateSetTomography, GateSetTomographyDesign, ProtocolData, ModelEstimateResults
from pygsti.models import ExplicitOpModel
from pygsti.data.dataset import DataSet
from pygsti.report import construct_standard_report
import numpy as np
from os import path as os_path
import pandas as pd

def run_dataset_and_finalobjective(
        ds: DataSet, final_objective: Literal['tvd', 'logl'],
        # ^ We'll vary those parameters in the demo.
        edesign: GateSetTomographyDesign, target_model: ExplicitOpModel, verbosity: int, mode: str
        # ^ Those arguments are just here in case you want to repurpose this receipe
        #   for something else.
    ) -> ModelEstimateResults:
    target_model = target_model.copy()
    target_model.convert_members_inplace(mode)
    target_model.default_gauge_group = 'unitary'
    pdata   = ProtocolData(edesign, ds)
    proto   = GateSetTomography(target_model, 'stdgaugeopt', objfn_builders={'objective': final_objective}, name=mode, verbosity=verbosity)
    results = proto.run(pdata, disable_checkpointing=True)
    return results


def build_all_results_for_demo(
        ds_original: DataSet, ds_corrupted: DataSet,
        edesign: GateSetTomographyDesign, target_model: ExplicitOpModel, verbosity: int, mode: str
    ) -> tuple[ModelEstimateResults, ModelEstimateResults]:
    common_args = (edesign, target_model, verbosity, mode)

    ##### train GST models with original data #######
    results_ori = run_dataset_and_finalobjective(  ds_original, 'logl', *common_args )  # type: ignore
    temp        = run_dataset_and_finalobjective(  ds_original,  'tvd', *common_args )  # type: ignore
    results_ori.rename_estimate(mode, 'fit-original-logl')
    results_ori.add_estimate(temp.estimates[mode], 'fit-original-tvd', silent_steal=True)

    ##### train GST models with corrupted data #######
    results_cor = run_dataset_and_finalobjective( ds_corrupted, 'logl', *common_args)  # type: ignore
    temp        = run_dataset_and_finalobjective( ds_corrupted,  'tvd', *common_args)  # type: ignore
    results_cor.rename_estimate(mode, 'fit-corrupted-logl')
    results_cor.add_estimate(temp.estimates[mode], 'fit-corrupted-tvd', silent_steal=True)

    ##### add test models (trained on corrupted data) to results_ori ######
    mdl = results_cor.estimates['fit-corrupted-logl'].models['stdgaugeopt']
    results_ori.add_model_test(target_model, mdl, 'fit-corrupted-logl')
    mdl = results_cor.estimates['fit-corrupted-tvd'].models['stdgaugeopt']
    results_ori.add_model_test(target_model, mdl, 'fit-corrupted-tvd')

    ##### add test models (trained on original data) to results_cor ######
    mdl = results_ori.estimates['fit-original-logl'].models['stdgaugeopt']
    results_cor.add_model_test(target_model, mdl, 'fit-original-logl')
    mdl = results_ori.estimates['fit-original-tvd'].models['stdgaugeopt']
    results_cor.add_model_test(target_model, mdl, 'fit-original-tvd')

    return results_ori, results_cor


def print_summary_table_for_demo(
        results_ori : ModelEstimateResults, results_cor: ModelEstimateResults
    ) -> None:
    f_ori = results_ori.estimates['fit-original-logl'].final_objective_fn()
    f_cor = results_cor.estimates['fit-corrupted-logl'].final_objective_fn()

    modelname = 'final iteration estimate'
    estimates = results_cor.estimates  # results_cor holds ModelTest estimates as well
    vals_cor = [(en, f_cor.fn_from_model(e.models[modelname])) for en,e in estimates.items()  ]
    vals_ori = [(en, f_ori.fn_from_model(e.models[modelname])) for en,e in estimates.items()  ]

    def sub_table_str(dataset_name: Literal['original', 'corrupted']):
        """
        Builds a a string representation of the following Markdown table, where "{ ... }"
        are placeholders for numerical values stored in `vals_cor` and `vals_ori`.

            |  when M = argmin(F(*|{dataset_name}))    |
            |------------------------------------------|
            | test dataset   |   F = -logl |   F = tvd |
            |----------------|-------------|-----------|
            | corrupted      |   { ... }   |  { ... }  |
            | original       |   { ... }   |  { ... }  |
        """
        df = pd.DataFrame([
                [ v  for n,v in vals_cor   if dataset_name in n],
                [ v  for n,v in vals_ori   if dataset_name in n]],
            columns=[n for n,_ in vals_cor if dataset_name in n], 
            index=['corrupted','original']  # row labels
        )
        df.index.name = 'test dataset'
        col_name_map = {f'fit-{dataset_name}-logl': 'F = -logl', f'fit-{dataset_name}-tvd': 'F = tvd'}
        df = df.rename(columns=col_name_map)
        t = df.to_markdown(tablefmt="github")
        lw = len(t.split('\n')[0])
        separator = f'|{ (lw-2)*"-" }|\n'
        caption = f"when M = argmin(F(*|{dataset_name}))"
        formatted_caption = '|' + caption.center(lw-2) + '|\n'
        s = formatted_caption + separator + t + '\n'
        return s
    
    supercaption = "\n"
    supercaption  = "   Negative log-likelihoods -logl(M|test)\n"
    supercaption += "   for different models and test datasets\n"
    print(supercaption)
    print(sub_table_str('original'))
    print(sub_table_str('corrupted'))
    return
```

## Stage data for "normal" GST

We use pyGSTi’s built-in **1-qubit XYI** model pack (`smq1Q_XYI`) to define the target model and a standard GST experiment design

We then create a *slightly noisy* “true” model by depolarizing the target gates (here `op_noise=0.01`) and simulate measurement counts from this model.

Key parameters you may want to vary:
- `max_max_length` (here `64`) controls GST circuit depth/expressivity.
- `num_samples` (here `10_000`) controls shot noise level.
- `op_noise` controls how far the true model deviates from the target.
- `fit_mode` (here, CPTPLND)

```{code-cell} ipython3
mp : GSTModelPack = smq1Q_XYI  # type: ignore
target       = mp.target_model()
circuitlists = mp.create_gst_circuitlists(64)
edesign      = GateSetTomographyDesign(target.create_processor_spec(), circuitlists, nested=True)
depol_model  = target.depolarize(op_noise=0.01)
ds_ori       = pygsti.data.simulate_data(depol_model, circuitlists[-1], num_samples=10_000, seed=0)
fit_mode     = 'CPTPLND'
```

## Inject outliers and run the fits

To test robustness, we intentionally corrupt a fraction of circuits:

- Choose a random subset of circuits (here `prop_corrupt = 0.025`, i.e., 2.5%).
- For each selected circuit, replace its outcome counts with a *random outcome distribution* (keeping the same total shots).

This creates a dataset that is *mostly consistent* with the true depolarized model, but contains a small number of circuits that are strongly inconsistent with any reasonable physical model.

This is a stylized outlier model (not meant to represent a specific device failure mode), but it is useful for probing estimator sensitivity.

```{code-cell} ipython3
prop_corrupt = 0.025

ds_cor = ds_ori.copy_nonstatic()
rng = np.random.default_rng(0)
circuits = list(ds_cor.keys())
num_circs = len(circuits)
selected = rng.choice(np.arange(num_circs), size=int(num_circs*prop_corrupt), replace=False)
selected = [circuits[i] for i in selected]
for c in selected:
    num_shots = ds_cor[c].total
    old_row   = ds_cor[c].to_dict()
    distn     = rng.random(len(old_row))
    distn    /= np.sum(distn)
    new_row   = {k: distn[i]*num_shots for i,k in enumerate(old_row.keys())}
    ds_cor[c] = new_row

results_ori, results_cor = build_all_results_for_demo(
    ds_ori, ds_cor, edesign, target, verbosity=0, mode=fit_mode
)
```

## Evaluating the results

We've fit four models:
- Train on the original data using `-logl`
- Train on the original data using `tvd`
- Train on corrupted data using `-logl`
- Train on corrupted data using `tvd`

Up next, we evaluate each fitted model on both datasets using a *common scoring function*:

$$
-\log \mathcal{L}(M \mid \text{test data}).
$$

Why evaluate with `-logl` even when training used `tvd`?
- It provides a single, interpretable “how well does this model explain these counts?” metric.
- It also highlights whether a training objective produces a model that generalizes well to uncorrupted data.

We summarize the scores below and generate an HTML report.

```{code-cell} ipython3
print_summary_table_for_demo(results_ori, results_cor)

report = construct_standard_report(
    {'eval-original'  : results_ori, 'eval-corrupted' : results_cor},
    advanced_options={'skip_sections': ('colorbox',)},
    title="Total variation distance (TVD) GST", verbosity=0
)
report_dir = 'example_files' if os_path.exists('example_files') else '../../example_files'
report_dir += '/robust-gst-report'
print('HTML report will be written to ... ')
import os
print(os.getcwd() + '/' + report_dir + '/main.html\n\n')
report.write_html(report_dir, verbosity=0)
```
