---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: leak311
  language: python
  name: python3
---

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XY, smq1Q_ZN
from pygsti.tools.leakage import leaky_qubit_model_from_pspec, construct_leakage_report
from pygsti.data import simulate_data
from pygsti.protocols import StandardGST, ProtocolData
```

# Leakage (automatic)

This short notebook shows how (data from) an experiment design for a two-level system can be used to fit a three-level sytem model, and how to generate a special report to provide insights for these models. The report includes special gate error metrics that reflect the distinguished role of the first two levels in the three-level system.

```{code-cell} ipython3
mp = smq1Q_XY
ed = mp.create_gst_experiment_design(max_max_length=32)
tm3 = leaky_qubit_model_from_pspec(mp.processor_spec(), mx_basis='l2p1')
# ^ We could use basis = 'gm' instead of 'l2p1'. We prefer 'l2p1'
#   because it makes process matrices easier to interpret in leakage
#   modeling.
ds = simulate_data(tm3, ed.all_circuits_needing_data, num_samples=1000, seed=1997)
gst = StandardGST( modes=('CPTPLND',), target_model=tm3, verbosity=2)
pd = ProtocolData(ed, ds)
res = gst.run(pd)
```

```{code-cell} ipython3
report_dir = '../../example_files/leakage-report-automagic'
report_object, updated_res = construct_leakage_report(res, title='easy leakage analysis!')
# ^ Each estimate in updated_res has a new gauge-optimized model.
#   The gauge optimization was done to reflect how our target gates
#   are only _really_ defined on the first two levels of our
#   three-level system.
#   
report_object.write_html(report_dir)
```


