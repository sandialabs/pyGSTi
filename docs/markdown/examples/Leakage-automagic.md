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
from pygsti.modelpacks import smq1Q_XYI as mp
from pygsti.tools.leakage import leaky_qubit_model_from_pspec, construct_leakage_report
from pygsti.data import simulate_data
from pygsti.protocols import StandardGST, ProtocolData
import numpy as np
import scipy.linalg as la
```

# Leakage (Automatic)

This short notebook shows how (data from) an experiment design for a two-level system can be used to fit a three-level sytem model, and how to generate a special report to provide insights for these models. The report includes special gate error metrics that reflect the distinguished role of the first two levels in the three-level system.

```{code-cell} ipython3
def with_leaky_gate(m, gate_label, strength):
    rng = np.random.default_rng(0)
    v = np.concatenate([[0.0], rng.standard_normal(size=(2,))])
    v /= la.norm(v)
    H = v.reshape((-1, 1)) @ v.reshape((1, -1))
    H *= strength
    U = la.expm(1j*H)
    m_copy = m.copy()
    G_ideal = m_copy.operations[gate_label]
    from pygsti.modelmembers.operations import ComposedOp, StaticUnitaryOp
    m_copy.operations[gate_label] = ComposedOp([G_ideal, StaticUnitaryOp(U, basis=m.basis)])
    return m_copy, v
```

```{code-cell} ipython3
ed = mp.create_gst_experiment_design(max_max_length=8)
# ^ The default max length is small so we don't have to wait as long 
#   for the GST fit (just for purposes of this notebook).
tm3 = leaky_qubit_model_from_pspec(mp.processor_spec(), mx_basis='l2p1')
# ^ Target model. "Leaky" is a bit of a misnomer here. The returned model
#   is simply a qutrit lift of the qubit model; leakage erorrs in the
#   qubit model can manifest as CPTP Markovian errors in the qutrit model.
dgm3, leaking_state = with_leaky_gate(tm3, ('Gxpi2', 0), strength=0.125)
# ^ Data generating model. 
num_samples = 100_000
# ^ The number of samples is large to compensate for short circuit length.
#   Feel free to change the number of samples to something more "realistic"
#   if you'd like.
if num_samples > 10_000:
    from pygsti.objectivefns import objectivefns
    objectivefns.DEFAULT_MIN_PROB_CLIP = objectivefns.DEFAULT_RADIUS = 1e-12
    # ^ There are numerical thresholding rules in objective function evaluation
    #   that lead to errors when the number of samples is extremely large.
    #   The lines above change those thresholding rules to be appropriate in
    #   the unusual setting that is this notebook.
ds = simulate_data(dgm3, ed.all_circuits_needing_data, num_samples=num_samples, seed=1997)
gst = StandardGST(
    modes=('CPTPLND',), target_model=tm3, verbosity=2,
    badfit_options={'actions': ['wildcard1d'], 'threshold': 0.0}
)
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


