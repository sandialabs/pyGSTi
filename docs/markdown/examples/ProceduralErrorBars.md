---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: pygsti
  language: python
  name: python3
---

# Procedural Error Bars

One other way we can use the `pygsti.report.reportables` module described in the [ModelAnalysisMetrics tutorial](../utilities/ModelAnalysisMetrics) is to procedurally generate error bars for any quantity you want.

First, let's simulate a noisy GST experiment

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XY
from pygsti.report import reportables as rptbl, modelfunction as modelfn
```

```{code-cell} ipython3
target_model = smq1Q_XY.target_model()

L=128
edesign = smq1Q_XY.create_gst_experiment_design(L)

noisy_model = target_model.randomize_with_unitary(.1)
noisy_model = noisy_model.depolarize(.05)

N=64
dataset = pygsti.data.simulate_data(noisy_model,edesign,N)


gst_proto = pygsti.protocols.StandardGST(modes=['full TP','CPTPLND','Target'],verbosity=2)
data = pygsti.protocols.ProtocolData(edesign,dataset)
results = gst_proto.run(data)
```

Now let's compute error bars on the CPTP estimate, and then get a 95% confidence interval "view" from the `ConfidenceRegionFactory`.

```{code-cell} ipython3
crfact = results.estimates['CPTPLND'].add_confidence_region_factory('stdgaugeopt', 'final')
crfact.compute_hessian(comm=None, mem_limit=3.0*(1024.0)**3) #optionally use multiple processors & set memlimit
crfact.project_hessian('intrinsic error')

crf_view = results.estimates['CPTPLND'].confidence_region_factories['stdgaugeopt','final'].view(95)
```

Finally, we can construct `pygsti.report.ModelFunction` objects that take a function which computes some observable from a model and the extracted view from above to compute error bars on that quantity of interest.

One common thing to check is error bars on the process matrices. The `ModelFunction` in this case only needs to return the operation:

```{code-cell} ipython3
final_model = results.estimates['CPTPLND'].models['stdgaugeopt'].copy()
```

```{code-cell} ipython3
def get_op(model, lbl):
    return model[lbl]
get_op_modelfn = modelfn.modelfn_factory(get_op)
```

```{code-cell} ipython3
rptbl.evaluate(get_op_modelfn(final_model, ("Gxpi2", 0)), crf_view)
```

```{code-cell} ipython3
rptbl.evaluate(get_op_modelfn(final_model, ("Gypi2", 0)), crf_view)
```

But we can also create model functions that perform more complicated actions, such as computing other reportables.

```{code-cell} ipython3
# Note that when creating ModelFunctions in this way, the model where you want the quantity evaluated must be the first argument
def ddist(model, ideal_model, lbl, basis):
    return rptbl.half_diamond_norm(model[lbl], ideal_model[lbl], basis)
ddist_modelfn = modelfn.modelfn_factory(ddist)
```

```{code-cell} ipython3
rptbl.evaluate(ddist_modelfn(final_model, target_model, ("Gxpi2", 0), 'pp'), crf_view)
```

```{code-cell} ipython3
rptbl.evaluate(ddist_modelfn(final_model, target_model, ("Gypi2", 0), 'pp'), crf_view)
```
