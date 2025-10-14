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

# Custom POVMs
This tutorial will demonstrate how to encode custom POVMs -- such as two-qubit parity measurement into a pyGSTi model -- rather than the standard Z measurement in the computational basis.

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq2Q_XYCNOT as std
import numpy as np
```

## Parity measurement construction

We start with a standard two-qubit model, and replace the default POVM with one that measures the parity instead. We do this by providing the superkets which described the desired measurement. This is straightforward for the parity measurement in the Pauli product basis, as shown below.

```{code-cell} ipython3
parity_model = std.target_model()

# Here, we specify the superkets for the even/odd effects
# This can be done in any basis, but we use Pauli-product here since
# we know the structure of the parity measurements in this basis
even_dmvec = np.zeros(16)
even_dmvec[0] = 1.0  # II element should be 1
even_dmvec[15] = 1.0 # ZZ element should also be 1 for even

odd_dmvec = np.zeros(16)
odd_dmvec[0] = 1.0  # II element is still 1 for odd...
odd_dmvec[15] = -1.0 # ... but ZZ element should be -1 for odd

parity_povm_dict = {'e': even_dmvec, 'o': odd_dmvec}

parity_povm = pygsti.modelmembers.povms.create_from_dmvecs(parity_povm_dict, "full TP",
    basis='pp', evotype=parity_model.evotype, state_space=parity_model.state_space)

parity_model['Mdefault'] = parity_povm
print(parity_model)
```

We can test this by running some simple circuits and seeing what outcomes we observe.

```{code-cell} ipython3
# Idle circuit should give us even outcome
dict(parity_model.probabilities( pygsti.circuits.Circuit([], line_labels=(0,1))))
```

```{code-cell} ipython3
# Partial flip of one qubit gives an equal superposition of odd and even
dict(parity_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0)], line_labels=(0,1))))
```

```{code-cell} ipython3
# Full bitflip of one qubit should give us an odd outcome
dict(parity_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0)], line_labels=(0,1))))
```

```{code-cell} ipython3
# Making a Bell pair (using H = Y(pi/2)X(pi), in operation order) should maintain the even outcome
dict(parity_model.probabilities( pygsti.circuits.Circuit([('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gcnot', 0, 1)], line_labels=(0,1))))
```

```{code-cell} ipython3
# Making a Bell pair and then flipping one qubit should give odd
dict(parity_model.probabilities( pygsti.circuits.Circuit([('Gypi2', 0), ('Gxpi2', 0), ('Gxpi2', 0), ('Gcnot', 0, 1),
                                                          ('Gxpi2', 1), ('Gxpi2', 1)], line_labels=(0,1))))
```

## Combining measurements

It is also possible to use different measurements on different sets of qubits. For example, we can mix computational basis states with our parity measurement from above.

Since we are going up to 3 qubits for this example, we will swap over to using a `QubitProcessorSpec` and `pygsti.modelconstruction` to build our initial `ExplicitModel` rather than loading it from a modelpack.

```{code-cell} ipython3
# Get a basic 3-qubit model
pspec = pygsti.processors.QubitProcessorSpec(3, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')
Z_parity_model = pygsti.models.create_explicit_model(pspec)

# Get a 1-qubit Z basis (computational) measurement
computational_povm = pygsti.modelmembers.povms.ComputationalBasisPOVM(nqubits=1)

# Get a composite POVM that performs Z measurement on qubit 1 and a parity measurement on qubits 2 and 3
# We are using the same parity POVM as the one defined above
Z_parity_povm = pygsti.modelmembers.povms.TensorProductPOVM([computational_povm, parity_povm])

# Override our standard measurement with the composite one
Z_parity_model['Mdefault'] = Z_parity_povm
```

And we can again test this with some simple measurements. Notice that instead of binary bitstrings, the "e"/"o" outcome labels are used as the second part of the outcome labels.

```{code-cell} ipython3
# Idle circuit should give us 0 on first qubit and even parity on second and third qubits
dict(Z_parity_model.probabilities( pygsti.circuits.Circuit([], line_labels=(0,1,2)) ))
```

```{code-cell} ipython3
# We can flip just the first qubit to see a 1 but still even outcome
dict(Z_parity_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0)], line_labels=(0,1,2)) ))
```

```{code-cell} ipython3
# Alternatively we can flip the last qubit to get a 0 but odd outcome
dict(Z_parity_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 2), ('Gxpi2', 2)], line_labels=(0,1,2)) ))
```

```{code-cell} ipython3
# And we can do partial flip of qubits 0 and 1 to get a uniform spread over all outcome possibilities
dict(Z_parity_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 1)], line_labels=(0,1,2)) ))
```

## Multiple custom measurements

The above works nicely if there is only one type of mixed measurement, but what if you have multiple? For example, what if you could measure parity on either pair of neighboring qubits, and also computational basis measurements on all qubits?

In this case, we can just add both POVMs to the model. However, we have to be careful about the "default" measurement of the system. For this example, we will use the computational basis POVM as the default measurement and assign the two parity-containing measurements to other keys. We just have to be careful that we explicitly use the correct POVM key when we want to do a different measurement.

```{code-cell} ipython3
# Get a basic 3-qubit model
mult_meas_model = pygsti.models.create_explicit_model(pspec)

# Note that Mdefault is the 3-qubit computational basis measurement already
print(mult_meas_model['Mdefault'])
```

```{code-cell} ipython3
# Now let's build our two other custom measurements and assign them to other keys
Z_parity_povm = pygsti.modelmembers.povms.TensorProductPOVM([computational_povm, parity_povm])
parity_Z_povm = pygsti.modelmembers.povms.TensorProductPOVM([parity_povm, computational_povm])

mult_meas_model['M_Z_par'] = Z_parity_povm
mult_meas_model['M_par_Z'] = parity_Z_povm

print(mult_meas_model)
```

As usual, let's test with some circuits to see if this has our expected behavior.

```{code-cell} ipython3
# Let's try to run a circuit with a bitflip on qubit 1...
try:
    dict(mult_meas_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0)], line_labels=(0,1,2)) ))
except Exception as e:
    print(e)
```

Notice that this fails! In particular, it tells us that there is not POVM label in the Circuit, and the model does not have a default. This is expected behavior - when models have multiple measurements, pyGSTi does not automatically assume that one is default.

We can fix this by just explicitly adding the Mdefault key.

```{code-cell} ipython3
dict(mult_meas_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0), "Mdefault"], line_labels=(0,1,2)) ))
```

Now, let's run the same circuit but use our other measurements.

```{code-cell} ipython3
# Using the Z-parity should give us 1 on qubit 0 and even for qubits 2 & 3...
dict(mult_meas_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0), "M_Z_par"], line_labels=(0,1,2)) ))
```

```{code-cell} ipython3
# ... while using parity-Z should give us odd for qubits 0 & 1 and 0 for qubit 2
dict(mult_meas_model.probabilities( pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0), "M_par_Z"], line_labels=(0,1,2)) ))
```

```{code-cell} ipython3

```
