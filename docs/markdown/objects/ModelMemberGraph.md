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

# ModelMemberGraph (advanced)

Example notebook of ModelMemberGraph functionality

```{code-cell} ipython3
import numpy as np

import pygsti
```

```{code-cell} ipython3
from pygsti.modelpacks import smq2Q_XYICNOT
```

# Similar/Equivalent

```{code-cell} ipython3
ex_mdl1 = smq2Q_XYICNOT.target_model()
ex_mdl2 = ex_mdl1.copy()
```

```{code-cell} ipython3
ex_mmg1 = ex_mdl1.create_modelmember_graph()
```

```{code-cell} ipython3
ex_mmg1.print_graph()
```

```{code-cell} ipython3
ex_mmg1.mm_nodes['operations']['Gxpi2', 0]
```

```{code-cell} ipython3
ex_mmg2 = ex_mdl2.create_modelmember_graph()
print(ex_mmg1.is_similar(ex_mmg2))
print(ex_mmg1.is_equivalent(ex_mmg2))
```

```{code-cell} ipython3
ex_mdl2.operations['Gxpi2', 0][0, 0] = 0.0
ex_mmg2 = ex_mdl2.create_modelmember_graph()
print(ex_mmg1.is_similar(ex_mmg2))
print(ex_mmg1.is_equivalent(ex_mmg2))
```

```{code-cell} ipython3
ex_mdl2.operations['Gxpi2', 0] = pygsti.modelmembers.operations.StaticArbitraryOp(ex_mdl2.operations['Gxpi2', 0])
ex_mmg2 = ex_mdl2.create_modelmember_graph()
print(ex_mmg1.is_similar(ex_mmg2))
print(ex_mmg1.is_equivalent(ex_mmg2))
```

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ['Gi', 'Gxpi2', 'Gypi2', 'mygate'], geometry='line', nonstd_gate_unitaries={'mygate': np.eye(2, dtype='complex')})

ln_mdl1 = pygsti.models.create_crosstalk_free_model(pspec,
                                                    depolarization_strengths={('Gxpi2', 0): 0.1, ('mygate', 0): 0.2},
                                                    lindblad_error_coeffs={('Gypi2', 1): {('H', 'X'): 0.2, ('S', 'Y'): 0.3}})
print(ln_mdl1)
```

```{code-cell} ipython3
ln_mmg1 = ln_mdl1.create_modelmember_graph()
ln_mmg1.print_graph()
```

```{code-cell} ipython3
# Should be exactly the same
ln_mdl2 = pygsti.models.create_crosstalk_free_model(pspec,
                                                    depolarization_strengths={('Gxpi2', 0): 0.1, ('mygate', 0): 0.2},
                                                    lindblad_error_coeffs={('Gypi2', 1): {('H', 'X'): 0.2, ('S', 'Y'): 0.3}})
ln_mmg2 = ln_mdl2.create_modelmember_graph()
print(ln_mmg1.is_similar(ln_mmg2))
print(ln_mmg1.is_equivalent(ln_mmg2))
```

```{code-cell} ipython3
# Should be similar if we change params
ln_mdl3 = pygsti.models.create_crosstalk_free_model(pspec,
                                                    depolarization_strengths={('Gxpi2', 0): 0.01, ('mygate', 0): 0.02},
                                                    lindblad_error_coeffs={('Gypi2', 1): {('H', 'X'): 0.5, ('S', 'Y'): 0.1}})
ln_mmg3 = ln_mdl3.create_modelmember_graph()
print(ln_mmg1.is_similar(ln_mmg3))
print(ln_mmg1.is_equivalent(ln_mmg3))
```

```{code-cell} ipython3
# Should fail both, depolarize is on different gate
ln_mdl4 = pygsti.models.create_crosstalk_free_model(pspec,
                                                    depolarization_strengths={('Gypi2', 0): 0.1},
                                                    lindblad_error_coeffs={('Gypi2', 1): {('H', "X"): 0.2, ('S', 'Y'): 0.3}})
ln_mmg4 = ln_mdl4.create_modelmember_graph()
print(ln_mmg1.is_similar(ln_mmg4))
print(ln_mmg1.is_equivalent(ln_mmg4))
```

# Serialization

```{code-cell} ipython3
ex_mdl1.write('../../example_files/ex_mdl1.json')
```

```{code-cell} ipython3
ln_mdl1.write('../../example_files/ln_mdl1.json')
```

```{code-cell} ipython3

```
