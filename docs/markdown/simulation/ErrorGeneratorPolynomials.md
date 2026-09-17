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

# Error Generator Polynomials
In this tutorial we provide an introduction to the functionality available through pyGSTi's `errgenpolytools` module.

The error generator propagation framework described in the companion tutorial {doc}`ErrorGeneratorPropagation` makes it possible to propagate sparse Markovian error generators through Clifford circuits and to construct efficient approximations to the resulting noisy dynamics. The `errgenpolytools` module builds directly on top of that framework and allows one to go one step further: instead of working only with *numerical* propagated error generator rates, we can construct **symbolic polynomial representations** of quantities of interest such as

- effective end-of-circuit error generators,
- corrections to computational basis measurement probabilities, and
- corrections to Pauli observable expectation values.

These polynomial representations are useful when one wants to:

- evaluate the same circuit quantities many times for different noise-parameter values,
- connect circuit-level propagated quantities back to model-level parameters,
- efficiently construct multiple observable corrections while reusing shared intermediate results.

The `errgenpolytools` module provides this additional symbolic layer by representing relevant rates and observable corrections as instances of pyGSTi's `Polynomial` class.

Please note: the functionality described here requires the error generator propagation framework, and thus also requires the `stim` python package.

```{code-cell} ipython3
import pygsti
import stim
import numpy as np
from itertools import product

from pygsti.tools import errgenproptools as eprop
from pygsti.tools import errgenpolytools as epoly
from pygsti.tools.lindbladtools import random_CPTP_error_generator_rates
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
```

## Setup
As in the error generator propagation tutorial, we begin by constructing a target model, a noisy model, and an example random Clifford circuit.

```{code-cell} ipython3
num_qubits = 4
gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
pspec = pygsti.processors.QubitProcessorSpec(num_qubits, gate_names, availability=availability)
target_model = pygsti.models.create_crosstalk_free_model(processor_spec=pspec)
```

For the noisy model we will use a crosstalk-free model with local H+S error generators. To simplify the later discussion of how to handle the aggregation of error generator rates associated with shared model parameters, we'll construct a model where all instances of a given gate type share the same error generator parameters. This is not a fundamental restriction, however; the functionality described in this tutorial is compatible with more general model parameter structures.

```{code-cell} ipython3
error_rates_dict_shared = {
    'Gcphase': random_CPTP_error_generator_rates(2, errorgen_types=('H', 'S'), label_type='local', seed=1234),
    'Gxpi2': random_CPTP_error_generator_rates(1, errorgen_types=('H', 'S'), label_type='local', seed=1235),
    'Gypi2': random_CPTP_error_generator_rates(1, errorgen_types=('H', 'S'), label_type='local', seed=1236)
}
error_model = pygsti.models.create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict_shared)
```

We also create an `ErrorGeneratorPropagator`, which is the object responsible for producing the propagation data structures used by `errgenpolytools`.

```{code-cell} ipython3
errorgen_propagator = ErrorGeneratorPropagator(error_model)
```

Finally, let us generate a random example circuit.

```{code-cell} ipython3
c = pygsti.algorithms.randomcircuit.create_random_circuit(
    pspec, 4, sampler='edgegrab', samplerargs=[0.4], rand_state=12345
)
print(c)
```

For later use we will also create the corresponding `stim.Tableau`.

```{code-cell} ipython3
tableau = c.convert_to_stim_tableau()
```

## From Propagation Maps to Polynomial Variables
The first step in constructing symbolic polynomials is to create a map from elementary error generators to corresponding polynomial variables. The relevant starting point are the error generator transform maps produced by the `ErrorGeneratorPropagator`.

```{code-cell} ipython3
errorgen_transform_map = errorgen_propagator.errorgen_transform_map(c)
errorgen_transform_maps = errorgen_propagator.errorgen_transform_maps(c)
```

These maps give the input-output relationship for an elementary error generator rate following propagation to the end of the circuit `c`. These methods return a dictionary (or dictionaries) with the following structure: Keys are tuples of the form (<original_errorgen_label>, <layer_index>), and values are of the form (<final_errorgen_label>, <overall_phase>), where overall_phase corresponds to the overall sign accumulated on the final error generator rate as a result of propagation. The former method gives this as a single aggregated dictionary for all circuit layers, while the latter gives a list of dictionaries, one per circuit layer.

The method `error_generator_to_polynomial_variable_maps` constructs a map from `(<input_error_generator>, <layer_index>)` pairs to integer variable indices for use in polynomials.

```{code-cell} ipython3
errorgen_to_var_map, var_to_errorgen_map = epoly.error_generator_to_polynomial_variable_maps(
    errorgen_transform_map, return_reverse=True
)
```

The forward map tells us which variable index corresponds to each propagated input error generator, while the reverse map lets us interpret a polynomial variable index in terms of the corresponding error generator and circuit layer.

```{code-cell} ipython3
list(errorgen_to_var_map.items())[:5]
```

```{code-cell} ipython3
list(var_to_errorgen_map.items())[:5]
```

To actually *evaluate* polynomials later on, we will also need a numerical vector of values for these variables. The helper function `construct_polynomial_parameter_vector_from_propagator` constructs exactly this vector.

```{code-cell} ipython3
poly_paramvec = epoly.construct_polynomial_parameter_vector_from_propagator(
    errorgen_propagator, var_to_errorgen_map, c
)
poly_paramvec[:10]
```

This vector can be fed directly into the `.evaluate(...)` method of a `Polynomial`.

## Constructing Polynomial Magnus Expansions
The first major construction provided by `errgenpolytools` is a symbolic Magnus approximation to the effective end-of-circuit error generator. The function `magnus_symbolic_polynomial` takes as input the layer-by-layer transform maps and a variable map and returns a dictionary whose keys are final `LocalStimErrorgenLabel`s and whose values are `Polynomial` objects giving the corresponding rates.

### First-order Magnus
We begin with the first-order Magnus approximation.

```{code-cell} ipython3
first_order_magnus_polys = epoly.magnus_symbolic_polynomial(
    errorgen_transform_maps, errorgen_to_var_map, magnus_order=1
)
```

Let us inspect a few representative terms.

```{code-cell} ipython3
for i, (lbl, poly) in enumerate(first_order_magnus_polys.items()):
    print(lbl, "->", poly)
    if i == 4:
        break
```

Because these are ordinary `Polynomial` objects, we can evaluate them on the parameter vector constructed above.

```{code-cell} ipython3
first_five = list(first_order_magnus_polys.items())[:5]
for lbl, poly in first_five:
    print(lbl, "->", poly.evaluate(poly_paramvec))
```

We can compare these values against the first-order BCH/Magnus propagation result computed using numerical rates.

```{code-cell} ipython3
propagated_errorgen_layer_first_order = errorgen_propagator.propagate_errorgens_bch(c, bch_order=1)
for lbl, poly in first_five:
    print(lbl)
    print("  polynomial evaluation:", poly.evaluate(poly_paramvec))
    print("  numerical BCH result:", propagated_errorgen_layer_first_order[lbl])
```

### Second-order Magnus
The symbolic Magnus construction also supports second order.

```{code-cell} ipython3
second_order_magnus_polys = epoly.magnus_symbolic_polynomial(
    errorgen_transform_maps, errorgen_to_var_map, magnus_order=2
)
```

At second order one may observe the appearance of new contributions to the effective elementary error generator terms arising from non-commutativity of the original elementary error generators.

```{code-cell} ipython3
for i, (lbl, poly) in enumerate(second_order_magnus_polys.items()):
    print(lbl, "->", poly)
    if i == 4:
        break
```

And again we can compare evaluated symbolic results against the numerical second-order BCH result.

```{code-cell} ipython3
propagated_errorgen_layer_second_order = errorgen_propagator.propagate_errorgens_bch(c, bch_order=2)
first_five_second_order = list(second_order_magnus_polys.items())[:5]
for lbl, poly in first_five_second_order:
    print(lbl)
    print("  polynomial evaluation:", poly.evaluate(poly_paramvec))
    print("  numerical BCH result:", propagated_errorgen_layer_second_order[lbl])
```

Please note that the symbolic Magnus implementation currently supports first and second order. 

## Constructing Taylor-Series Polynomials
Once we have a symbolic representation of the effective end-of-circuit error generator, we can form the Taylor series approximation to its exponential using `error_generator_taylor_expansion_symbolic_polynomial`.

This function returns a list of dictionaries, one per Taylor order (excluding zeroth order), with the same key structure as the Magnus dictionary.

### First-order Taylor expansion

```{code-cell} ipython3
first_order_taylor_terms = epoly.error_generator_taylor_expansion_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, order=1
)
first_order_taylor_polys = first_order_taylor_terms[0]
```

```{code-cell} ipython3
for i, (lbl, poly) in enumerate(first_order_taylor_polys.items()):
    print(lbl, "->", poly)
    if i == 4:
        break
```

### Second-order Taylor expansion

```{code-cell} ipython3
second_order_taylor_terms = epoly.error_generator_taylor_expansion_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, order=2
)
second_order_taylor_polys = second_order_taylor_terms[1]
```

```{code-cell} ipython3
for i, (lbl, poly) in enumerate(second_order_taylor_polys.items()):
    print(lbl, "->", poly)
    if i == 4:
        break
```

We can compare evaluated symbolic Taylor terms against the corresponding Taylor expansion with numerical rates from `errgenproptools`.

```{code-cell} ipython3
numeric_taylor_order_1 = eprop.error_generator_taylor_expansion(propagated_errorgen_layer_first_order, order=1)[0]
numeric_taylor_order_2 = eprop.error_generator_taylor_expansion(
    propagated_errorgen_layer_first_order, order=2, truncation_threshold=-1
)[1]
```

```{code-cell} ipython3
for lbl, poly in list(first_order_taylor_polys.items())[:5]:
    print(lbl)
    print("  polynomial evaluation:", poly.evaluate(poly_paramvec))
    print("  numerical Taylor term:", numeric_taylor_order_1[lbl])
```

```{code-cell} ipython3
for lbl, poly in list(second_order_taylor_polys.items())[:5]:
    print(lbl)
    print("  polynomial evaluation:", poly.evaluate(poly_paramvec))
    print("  numerical Taylor term:", numeric_taylor_order_2[lbl])
```

## Probability Correction Polynomials
A major use case for `errgenpolytools` is the construction of symbolic corrections to the
probabilities of computational basis measurement outcomes.

The function `stabilizer_probability_correction_symbolic_polynomial` returns a `Polynomial`
corresponding to the correction to a specified bitstring probability.

### Single-bitstring probability correction
Let us compute the correction polynomial for the output bitstring `'0000'`.

```{code-cell} ipython3
prob_corr_poly_order_1 = epoly.stabilizer_probability_correction_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, tableau, '0000', order=1
)
print(prob_corr_poly_order_1)
```

Evaluating this polynomial gives the numerical first-order correction.

```{code-cell} ipython3
prob_corr_poly_order_1.evaluate(poly_paramvec)
```

We can compare this against the numerical correction computed using `errgenproptools`.

```{code-cell} ipython3
prob_corr_numeric_order_1 = eprop.stabilizer_probability_correction(
    propagated_errorgen_layer_first_order, tableau, '0000', order=1
)
print(prob_corr_numeric_order_1)
```

Similarly for second order:

```{code-cell} ipython3
prob_corr_poly_order_2 = epoly.stabilizer_probability_correction_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, tableau, '0000', order=2
)
prob_corr_numeric_order_2 = eprop.stabilizer_probability_correction(
    propagated_errorgen_layer_first_order, tableau, '0000', order=2
)
print(prob_corr_poly_order_2.evaluate(poly_paramvec))
print(prob_corr_numeric_order_2)
```

### Bulk probability corrections
When one wants probability corrections for many bitstrings, the bulk interface can reuse
intermediate results and is typically much more efficient.

```{code-cell} ipython3
bitstrings_4Q = [''.join(bs) for bs in product(['0', '1'], repeat=4)]
bulk_prob_corr_polys = epoly.bulk_stabilizer_probability_correction_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, tableau, bitstrings_4Q, order=1
)
```

```{code-cell} ipython3
for bs, poly in zip(bitstrings_4Q, bulk_prob_corr_polys):
    print(bs, "->", poly)
    print('------')
```

### Recovering approximate probabilities
To obtain an approximate noisy probability, we add the correction to the ideal stabilizer
probability.

```{code-cell} ipython3
ideal_prob_0000 = eprop.stabilizer_probability(tableau, '0000')
approx_prob_0000 = ideal_prob_0000 + prob_corr_poly_order_2.evaluate(poly_paramvec)
print("Ideal probability:", ideal_prob_0000)
print("Approximate noisy probability:", approx_prob_0000)
```

For a few-qubit example we can compare this against the exact dense forward simulation.

```{code-cell} ipython3
exact_prob_0000 = error_model.sim.probs(c)['0000']
print("Exact noisy probability:", exact_prob_0000)
print("Absolute error:", abs(exact_prob_0000 - approx_prob_0000))
```

## Pauli Expectation Correction Polynomials
In addition to measurement probabilities, `errgenpolytools` can also construct symbolic correction
polynomials for expectation values of Pauli observables.

The relevant functions are:
- `stabilizer_pauli_expectation_correction_symbolic_polynomial`
- `bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial`

### Single-Pauli expectation correction
Let us compute the correction polynomial for the observable `IIXZ`.

```{code-cell} ipython3
pauli = stim.PauliString('IIXZ')
pauli_corr_poly_order_1 = epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, tableau, pauli, order=1
)
print(pauli_corr_poly_order_1)
```

```{code-cell} ipython3
pauli_corr_poly_order_1.evaluate(poly_paramvec)
```

Compare this to the numerical correction.

```{code-cell} ipython3
pauli_corr_numeric_order_1 = eprop.stabilizer_pauli_expectation_correction(
    propagated_errorgen_layer_first_order, tableau, pauli, order=1
)
print(pauli_corr_numeric_order_1)
```

And again at second order:

```{code-cell} ipython3
pauli_corr_poly_order_2 = epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, tableau, pauli, order=2
)
pauli_corr_numeric_order_2 = eprop.stabilizer_pauli_expectation_correction(
    propagated_errorgen_layer_first_order, tableau, pauli, order=2
)
print(pauli_corr_poly_order_2.evaluate(poly_paramvec))
print(pauli_corr_numeric_order_2)
```

## Custom Variable Labels for Polynomial Output
By default, the string representation of a `Polynomial` uses generic variable names such as
`x0`, `x1`, `x2`, and so on. When working with error generator polynomials it is often useful to
replace these generic names with labels that make clear which propagated error generator parameter
each variable corresponds to.

Because the variables used in `errgenpolytools` are indexed by the maps returned from
`error_generator_to_polynomial_variable_maps` (or its aggregated variants), we can construct a
dictionary mapping variable indices to more descriptive labels and pass this into the
`Polynomial.to_string(...)` method.

For example, let us revisit one of the symbolic expectation-value correction polynomials from above.

```{code-cell} ipython3
custom_label_poly = epoly.stabilizer_pauli_expectation_correction_symbolic_polynomial(
    first_order_magnus_polys,
    errorgen_to_var_map,
    tableau,
    stim.PauliString('IIXZ'),
    order=2
)
print(custom_label_poly)
```

The default output is compact, but it does not tell us directly which propagated circuit-layer error
generator each variable corresponds to. We can build a more descriptive variable-label map from the
`errorgen_to_var_map` dictionary:

```{code-cell} ipython3
parameter_label_map = {
    idx: f"({lbl}_{layer_idx})"
    for (lbl, layer_idx), idx in errorgen_to_var_map.items()
}
list(parameter_label_map.items())[:5]
```

Now we can ask the polynomial to render itself using these custom labels.

```{code-cell} ipython3
print(custom_label_poly.to_string(var_labels=parameter_label_map))
```

Please note that `var_labels` can be supplied in several forms supported by the `Polynomial` class,
including a dictionary (as above), a sequence, or a callable. For most `errgenpolytools`
applications, a dictionary keyed by variable index is often the most convenient option.


### Bulk expectation corrections
As with probabilities, a bulk interface is available when expectation corrections for many Pauli
observables are desired.

```{code-cell} ipython3
paulis_4Q = np.fromiter(stim.PauliString.iter_all(num_qubits=4, min_weight=1), dtype=object)
rng = np.random.default_rng(1234)
random_paulis = rng.choice(paulis_4Q, 8, replace=False).tolist()

bulk_pauli_corr_polys = epoly.bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(
    first_order_magnus_polys, errorgen_to_var_map, tableau, random_paulis, order=1
)
```

```{code-cell} ipython3
for p, poly in zip(random_paulis[:5], bulk_pauli_corr_polys[:5]):
    print(p, "->", poly)
```

## Aggregating Variables by Gate or Shared Model Parameters
So far we have treated each propagated input error generator occurrence as an independent polynomial
variable. This is not always the most useful choice.

In many cases we instead want polynomial variables that reflect:
- the parameters associated with a given gate occurrence, or
- shared model parameters reused across multiple gate instances.

The helper function `error_generator_to_polynomial_variable_maps_by_gate` supports this.

### Aggregating by gate
Starting from the raw variable map, we can aggregate variables by gate structure.

```{code-cell} ipython3
errorgen_to_var_map_gate_noagg, var_to_errorgen_map_gate_noagg = epoly.error_generator_to_polynomial_variable_maps_by_gate(
    error_model,
    errorgen_to_var_map,
    c,
    include_spam=True,
    aggregate_shared_parameter_gates=False,
    return_reverse=True
)
```

```{code-cell} ipython3
print("Number of unaggregated variables:", len(var_to_errorgen_map))
print("Number of gate-aggregated variables:", len(var_to_errorgen_map_gate_noagg))
```

### Aggregating across shared-parameter gates
Because our example noisy model shares parameters across all instances of each gate type, we can
further aggregate variables across gates that share the same underlying model parameters.

```{code-cell} ipython3
errorgen_to_var_map_gate_agg, var_to_errorgen_map_gate_agg = epoly.error_generator_to_polynomial_variable_maps_by_gate(
    error_model,
    errorgen_to_var_map,
    c,
    include_spam=True,
    aggregate_shared_parameter_gates=True,
    return_reverse=True
)
```

```{code-cell} ipython3
print("Number of gate-aggregated variables without shared-parameter merging:", len(var_to_errorgen_map_gate_noagg))
print("Number of gate-aggregated variables with shared-parameter merging:", len(var_to_errorgen_map_gate_agg))
```

This can significantly reduce the number of variables appearing in the resulting symbolic polynomials.

### Constructing the matching parameter vector
When using an aggregated map, it is important to evaluate the polynomials on the corresponding aggregated parameter vector.

```{code-cell} ipython3
poly_paramvec_gate_agg = epoly.construct_polynomial_parameter_vector_from_propagator(
    errorgen_propagator, var_to_errorgen_map_gate_agg, c
)
poly_paramvec_gate_agg
```

We can now construct a first-order shared-parameter-aware symbolic Magnus approximation and evaluate it on the aggregated parameter vector.

```{code-cell} ipython3
first_order_magnus_polys_gate_agg = epoly.magnus_symbolic_polynomial(
    errorgen_transform_maps,
    errorgen_to_var_map_gate_agg,
    magnus_order=1
)
```

```{code-cell} ipython3
for lbl, poly in list(first_order_magnus_polys_gate_agg.items())[:5]:
    print(lbl)
    print("  polynomial:", poly)
    print("  evaluated value:", poly.evaluate(poly_paramvec_gate_agg))
```

## End-to-End Example Workflow
A typical `errgenpolytools` workflow looks like this:

1. Build a noisy error-generator-parameterized model and a Clifford circuit.
2. Construct propagation transform maps using `ErrorGeneratorPropagator`.
3. Build a polynomial variable map.
4. Construct symbolic Magnus polynomials.
5. Build symbolic observable correction polynomials.
6. Construct the corresponding parameter vector.
7. Evaluate the resulting polynomials.

Below is a concise example following exactly these steps.

```{code-cell} ipython3
errorgen_transform_maps_demo = errorgen_propagator.errorgen_transform_maps(c)
errorgen_to_var_map_demo, var_to_errorgen_map_demo = epoly.error_generator_to_polynomial_variable_maps(
    errorgen_propagator.errorgen_transform_map(c), return_reverse=True
)
magnus_polys_demo = epoly.magnus_symbolic_polynomial(
    errorgen_transform_maps_demo, errorgen_to_var_map_demo, magnus_order=1
)
prob_poly_demo = epoly.stabilizer_probability_correction_symbolic_polynomial(
    magnus_polys_demo, errorgen_to_var_map_demo, tableau, '0000', order=2
)
paramvec_demo = epoly.construct_polynomial_parameter_vector_from_propagator(
    errorgen_propagator, var_to_errorgen_map_demo, c
)
print("Evaluated probability correction:", prob_poly_demo.evaluate(paramvec_demo))
```

## Limitations and Current Scope
The present polynomial tools inherit several of the practical limitations of the error generator
propagation framework:

- They require Clifford-only circuits for the propagation step.
- The most useful regime is one in which the underlying error generator rates are relatively small, so that low-order BCH and Taylor approximations remain accurate.
- Symbolic expressions can grow significantly in size as one increases approximation order or circuit size.
- The symbolic Magnus implementation currently supports first and second order.
- Aggregation across shared model parameters currently assumes sufficiently local noise structure for the most advanced grouping modes.
