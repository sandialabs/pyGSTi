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

# Error Generator Propagation
In this tutorial we will provide an overview of the core functionality available through pyGSTi's error generator propagation module.

Error generator propagation is a technique which leverages the analytical properties of the error generator formalism to enable efficient forward simulation by propagating general markovian error channels through Clifford circuits. Efficiency of this technique relies on two conditions:

- Sparsity: At most a polynomial number of error generator rates (in the number of qubits) can be nonzero for any given circuit layer.
- Clifford-only: The propagation of error generators relies on the analytic properties of the elementary error generators when conjugated by cliffords.

That is pretty much it though. Coherent errors, non-unital errors (e.g. amplitude damping), dephasing, all fair game. Practically there is a third requirement as well and that is that the error generator rates are relatively small. The larger the error generator rates, the higher-order the approximation you'll require (BCH and/or taylor series) to achieve a given precision target when using the functionality described herein for efficiently performing strong simulation in the error generator propagation framework. 

Please note: The implementation of the error generator propagation framework in pyGSTi requires the `stim` python package, so please ensure this is installed in your environment before proceeding.

```{code-cell} ipython3
import pygsti
import stim
from pygsti.tools import errgenproptools as eprop
from pygsti.tools.lindbladtools import random_error_generator_rates
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
```

To begin we need an error model, and particularly one parameterized using error generators (or otherwise capable of outputing error generators for a circuit layer). For this tutorial we'll work with a 4-qubit crosstalk-free model for a gate set consisting of $\pi/2$ rotations about X and Y on each qubit, and a two-qubit CPHASE gate. 

```{code-cell} ipython3
num_qubits = 4
gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
pspec = pygsti.processors.QubitProcessorSpec(num_qubits, gate_names, availability=availability)
target_model = pygsti.models.create_crosstalk_free_model(processor_spec = pspec)
```

Now that we have a target model we'll also want a noisy model to simulate as well. For this example we'll randomly sample a weight-2 H+S (coherent + pauli stochastic) error model, but the error generator propagation framework can also handle C and A error generators as well (i.e. general lindbladian errors). 
The specific specification we'll need for the model construction routine we're about to use is a dictionary whose keys are gate labels. Each value of this dictionary is itself a dictionary whose keys are elementary error generator labels, and whose values are error generator rates.

```{code-cell} ipython3
qubit_labels = pspec.qubit_labels
error_rates_dict = {}
for gate, availability in pspec.availability.items():
    n = pspec.gate_num_qubits(gate)
    if availability == 'all-edges':
        assert(n == 1), "Currently require all 2-qubit gates have a specified availability!"
        qubits_for_gate = qubit_labels
    else:
        qubits_for_gate = availability  
    for qs in qubits_for_gate:
        label = pygsti.baseobjs.Label(gate, qs)
        # Sample error rates.
        error_rates_dict[label] = random_error_generator_rates(num_qubits=n, errorgen_types=('H', 'S'), label_type='local', seed=1234)
```

```{code-cell} ipython3
error_model = pygsti.models.create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)
```

We'll also need an example circuit for the rest of our examples, so will construct one at random.

```{code-cell} ipython3
c = pygsti.algorithms.randomcircuit.create_random_circuit(pspec, 3, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
```

```{code-cell} ipython3
print(c)
```

## Basic Propagation
In this section we'll introduce the basic syntax of the `ErrorGeneratorPropagator` class and usage of the basic error generator propagation functionality.
Our first step will be to create an instance of the `ErrorGeneratorPropagator` class. This is as simple as passing in our error model into the constructor.

```{code-cell} ipython3
errorgen_propagator = ErrorGeneratorPropagator(error_model)
```

The very first thing we can do is propagate the error generators for each circuit layer to the end of the circuit. This is done using the `propagate_errorgens` method.

```{code-cell} ipython3
propagated_errorgen_layers = errorgen_propagator.propagate_errorgens(c)
```

The output of this method is a list of dictionaries, one for each original error generator layer in the circuit, containing an updated set of elementary error generator coefficients and rates corresponding to the result of propagating each error generator through the circuit. Note this list is returned in circuit ordering, so there is a one-to-one correspondence between the position an error generator appears in the original circuit and where it appears in this final list.

So, to see the result of propagating the error generator corresponding to the noise induced after the first layer of gates to the very end we could query this list as follows:

```{code-cell} ipython3
print(propagated_errorgen_layers[1])
```

There are a few things worth noting at this point. First, we stated we'd be looking at the output of propagating the *first* circuit layer to the end but we indexed into the *second* position of the final list, what gives? This is because by default the `propagate_errorgens` method prepends and appends the error generator layers corresponding to state preparation and measurement respectively *before* beginning the propagation. As such the first layer in the final output corresponds to the error generator associated with state prep, and the final one with measurement. We never actually specified error generator rates for the SPAM, so you'll notice the corresponding dictionaries in the final output are both empty in this case.

```{code-cell} ipython3
print(propagated_errorgen_layers[0])
print(propagated_errorgen_layers[-1])
```

To change this behavior so that the SPAM layers are not included you can set the optional kwarg `include_spam` to `FALSE` in `propgate_errorgens` and other related methods.

+++

The next things worth noting are the keys of the final dictionary. Notice that the basis element labels for each of the elementary error generator coefficient labels are instances of `stim.PauliString`, very much unlike the other elementary error generator labels used in pyGSTi. These labels are instances of the class `LocalStimErrorgenLabel`, a specialized label class with additional metadata and methods used throughout the error generator propagation framework. For applications where you need to take the output of this module and utilize it elsewhere in pyGSTi you can utilize the `to_local_eel` and `to_global_eel` methods of the `LocalStimErrorgenLabel` class to convert these into instances of `LocalElementaryErrorgenLabel` and `GlobalElementaryErrorgenLabel`, respectively, for use within other parts of pyGSTi.

+++

While the output of `propgate_errorgens` is in and of itself incredibly useful, often we want to know more about how specific errors have been transformed by propagation through the circuit. Fortunately the analytic structure of error generator propagation through a clifford operation is such that it acts as a generalized permutation of each elementary error generator within it's sector (i.e. propagation can't in and of itself map H errors to anything other than H errors, for example). To view the input-output corresponding to the transformation of each error generator we can use the `errorgen_transform_map` method.

```{code-cell} ipython3
errorgen_transform_map = errorgen_propagator.errorgen_transform_map(c)
```

This method returns a dictionary with the following structure: Keys are tuples of the form (<original_errorgen_label>, <layer_index>), and values are of the form (<final_errorgen_label>, <overall_phase>), where overall_phase corresponds to the overall sign accumulated on the final error generator rate as a result of propagation. So, for example, we can see that as a result of propagation through the circuit the H(XIII) error generator at circuit layer 1 is mapped to an H(ZIII) error generator accruing and overall phase of -1.

```{code-cell} ipython3
print(errorgen_transform_map[(_LSE('H', [stim.PauliString('XIII')]), 1)])
```

For some purposes it can be useful to go another step further and identity which gate a particular error might be associated with in the original error model. For this purpose `ErrorGeneratorPropagator` has a helper method available called `errorgen_gate_contributors`.

```{code-cell} ipython3
print(errorgen_propagator.errorgen_gate_contributors(_LSE('H', [stim.PauliString('XIII')]), c, layer_idx=1))
```

Here this method returns the fact that in our particular error model the only gate at layer index 1 which could have contributed this particular error generator was the 'Gxpi2' gate acting on qubit 0. In some error models it may be possible for multiple gates to contribute to a particular rate, in which case this method should return all such gates.

+++

## BCH Approximation
In the previous section we showed how to use the `ErrorGeneratorPropagator` class to transform a circuit with a series of post-gate error generators into an equivalent representation of this noisy circuit with instead a series of post-circuit error generator layers. What if we want a single effective end-of-circuit error generator which approximates the overall action of the composition of each of the propagated error generators? To do so the `ErrorGeneratorPropagator` class supports the option to iteratively apply the BCH approximation at various orders to perform this recombination.

The main method for performing propagation together with the iterative application of the BCH approximation is called `propagate_errorgens_bch`.

```{code-cell} ipython3
propagated_errorgen_layer_first_order = errorgen_propagator.propagate_errorgens_bch(c)
```

As before this method propagated all of a circuits error generator layers to the very end, but follows this up with an iterative application of the BCH approximation resulting in a single final error generator. Without any additional optional arguments specified this uses the first-order BCH approximation.

```{code-cell} ipython3
print(propagated_errorgen_layer_first_order)
```

This method supports a number of additional arguments beyond those already for `propagate_errorgens`:
- `bch_order`: An integer from 1 to 5 specifying the order of the BCH approximation to apply (5 is the current maximum). Note that the computational cost of higher order BCH can scale rapidly,    so keep this in mind when balancing the need for accuracy and speed of computation.
- `truncation_threshold`: This argument allows you to specify a minimum threshold (in terms of error generator rate) below which rates are truncated to zero. This can improve performance      by allowing one to skip the computation of terms corresponding to very small corrections.
Some interesting emergent behavior starts to occur when we begin to look at higher-order BCH corrections.

```{code-cell} ipython3
propagated_errorgen_layer_second_order = errorgen_propagator.propagate_errorgens_bch(c, bch_order=2)
```

```{code-cell} ipython3
print(propagated_errorgen_layer_second_order)
```

Aside from the fact that there are now significantly more terms than was found for the first-order BCH approximation, notice that there are also now emergent second (and higher) order contributions due to C and A error generators which arise from the composition of purely H and S error generators. These additional terms arise from the non-commutivity of the elementary error generators, particularly the non-commutivity of H and S elementary error generators. For more on this phenomenon see [insert paper reference here].

+++

## Approximate Probabilities and Expectation Values
Now you have an efficient representation for an approximation to the effective end-of-circuit error generator for your circuit, what can you do with it? In this section we show how to use this sparse representation to efficiently compute corrections to the outcome probability distributions and pauli observable expectation values of noisy clifford circuits.

+++

We'll start off by demonstrating how to perform strong simulation using the results of error generator propagation to estimate the output probabilities for a desired computational basis state. 

To do so we'll be making use of the function `approximate_stabilizer_probability` from the `errgenproptools` module. This function takes as input the following arguments:

- errorgen_dict : A dictionary of elementary error generator coefficients and their corresponding rates (as outputted, for example, by `propagate_errorgens_bch`.
- circuit : The circuit to compute the output probability for. This can by a pyGSTi `Circuit` object, or alternatively a `stim.Tableau`.
- desired_bitstring : A string corresponding to the desired computational basis state.
- order : Order of the taylor series approximation for the exponentiated error generator to use in computing the approximate output probability. In principle this function can compute       arbitary-order approximation (but practically the cost of the computation scales in the order).
- truncation_threshold : As described above, this is a minimum value below which contributions are truncated to zero which can sometimes improve performance by reducing the number of terms   computed with very small overall corrections to the calculated probability. 

Let's use the results of the application of the second-order BCH approximation above and compute the approximate probability of reading out the all-zeros state from our circuit. For the ideal circuit, the probability of observing the all-zeros state is 0.

```{code-cell} ipython3
first_order_approximate_prob = eprop.approximate_stabilizer_probability(propagated_errorgen_layer_second_order, c, '0000', order=1)
print(first_order_approximate_prob)
```

```{code-cell} ipython3
second_order_approximate_prob = eprop.approximate_stabilizer_probability(propagated_errorgen_layer_second_order, c, '0000', order=2)
print(second_order_approximate_prob)
```

In this few qubit test case we also have the luxury compare this to the results of the (effectively) exact forward simulation for the error model:

```{code-cell} ipython3
exact_probability = error_model.sim.probs(c)['0000']
print(exact_probability)
```

```{code-cell} ipython3
print(f'Absolute Error Approx to Exact (First-order Taylor, Second-order BCH): {abs(exact_probability-first_order_approximate_prob)}')
print(f'Absolute Error Approx to Exact (Second-order Taylor, Second-order BCH): {abs(exact_probability-second_order_approximate_prob)}')
```

```{code-cell} ipython3
print(f'Relative Error Approx to Exact (First-order taylor, Second-order BCH): {100*abs(exact_probability-first_order_approximate_prob)/exact_probability}%')
print(f'Relative Error Approx to Exact (Second-order taylor, Second-order BCH): {100*abs(exact_probability-second_order_approximate_prob)/exact_probability}%')
```

Here we can see that with the combination of second-order BCH and second-order taylor approximations our estimated probability is accurate to well below a 1 percent relative error. By going out to higher-order in either approximation one can achieve even higher levels of accuracy.

+++

In addition to strong simulation of the output probabilities of computational basis states, it is also possible to compute approximate values for the expectation values of pauli observables. The main function for doing so is `approximate_stabilizer_pauli_expectation` from the `errgenproptools` module, the signature of which is nearly identical to that of `approximate_stabilizer_probability` described above, except taking instead a desired pauli observable to estimate the expectation value for. Here we'll again use the results of the second-order BCH approximation produced above and look are various order of the taylor series approximation for the pauli expectation value of 'XYZI' (the value for the ideal noise-free circuit is 1).

```{code-cell} ipython3
first_order_approximate_pauli_expectation = eprop.approximate_stabilizer_pauli_expectation(propagated_errorgen_layer_second_order, c, 'XYZI', order=1)
print(first_order_approximate_pauli_expectation)
```

```{code-cell} ipython3
second_order_approximate_pauli_expectation = eprop.approximate_stabilizer_pauli_expectation(propagated_errorgen_layer_second_order, c, 'XYZI', order=2)
print(second_order_approximate_pauli_expectation)
```

There aren't existing built-in functions in pyGSTi for outputing exact pauli expectation values handy, but we can write a short helper function for computing these for the sake of comparison with our above results.

```{code-cell} ipython3
from pygsti.tools.basistools import change_basis
import numpy as np
from pygsti.baseobjs import Label
def pauli_expectation_exact(error_propagator, target_model, circuit, pauli):
    #get the eoc error channel, and the process matrix for the ideal circuit:
    eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    #also get the ideal state prep and povm:
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    
    #finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    #flatten this row-wise
    pauli_vec = np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    #put this in pp basis (since these are paulis themselves I could just read this off directly).
    pauli_vec = change_basis(pauli_vec, 'std', 'pp')
    #print(pauli_vec)
    dense_prep = ideal_prep.to_dense().copy()
    expectation = np.linalg.multi_dot([pauli_vec.reshape((1,len(pauli_vec))), eoc_channel, ideal_channel, dense_prep.reshape((len(dense_prep),1))]).item()
    return expectation
```

```{code-cell} ipython3
exact_pauli_expectation = pauli_expectation_exact(errorgen_propagator, target_model, c, stim.PauliString('XYZI'))
print(exact_pauli_expectation)
```

```{code-cell} ipython3
print(f'Absolute Error Approx to Exact (First-order Taylor, Second-order BCH): {abs(exact_pauli_expectation-first_order_approximate_pauli_expectation)}')
print(f'Absolute Error Approx to Exact (Second-order Taylor, Second-order BCH): {abs(exact_pauli_expectation-second_order_approximate_pauli_expectation)}')
```

```{code-cell} ipython3
print(f'Relative Error Approx to Exact (First-order taylor, Second-order BCH): {100*abs(exact_pauli_expectation-first_order_approximate_pauli_expectation)/exact_pauli_expectation}%')
print(f'Relative Error Approx to Exact (Second-order taylor, Second-order BCH): {100*abs(exact_pauli_expectation-second_order_approximate_pauli_expectation)/exact_pauli_expectation}%')
```

In this case even with the first-order taylor approximation together with the second-order BCH approximation the relative error to the exact expecation value is roughly half a percent, dropping to below a tenth of a percent when we go up to the second order taylor approximation. As before, by going out to higher-order in either approximation one can achieve even higher levels of accuracy.

+++

## Other Helpful Utilities:
In this section we'll highlight a few additional utilities within the error generator propagation related modules which are often useful (some of these you may have even seen us use above!).

We'll specifically cover:
- `eoc_error_channel`
- `errorgen_layer_dict_to_errorgen`
- `approximate_stabilizer_probabilities`
- `error_generator_commutator`
- `error_generator_composition`

+++

#### `eoc_error_channel` : 
This method provides a simple single function call for generating a dense representation of the end-of-circuit error channel (i.e. the exponentiated end-of-circuit error generator). This can be useful in few-qubit testing, but obviously doesn't not scale beyond a few qubits. This end-of-circuit error channel can be produced either exactly or without the BCH approximation. In the former case this is acheived by exponentiating and multiplying together all of the propagated error generator layers.

```{code-cell} ipython3
dense_end_of_circuit_channel_exact = errorgen_propagator.eoc_error_channel(c, use_bch=False)
```

```{code-cell} ipython3
dense_end_of_circuit_channel_first_order_BCH = errorgen_propagator.eoc_error_channel(c, use_bch=True, bch_kwargs={'bch_order':1})
dense_end_of_circuit_channel_second_order_BCH = errorgen_propagator.eoc_error_channel(c, use_bch=True, bch_kwargs={'bch_order':2})
```

This can be useful in testing settings, for example, where we can use these as yet another way to measure the accuracy of our approximation methods.

```{code-cell} ipython3
print(f'Frobenius norm between exact and 1st-order BCH EOC channels: {np.linalg.norm(dense_end_of_circuit_channel_exact-dense_end_of_circuit_channel_first_order_BCH)}')
print(f'Frobenius norm between exact and 2nd-order BCH EOC channels: {np.linalg.norm(dense_end_of_circuit_channel_exact-dense_end_of_circuit_channel_second_order_BCH)}')
```

#### `errorgen_layer_dict_to_errorgen`
Throughout the error generator propagation framework we generate a lot of sparse error generator representations in terms of dictionaries of elementary error generator coefficients and corresponding rates. For testing purposes (with just a few qubits, this obviously does not scale) it is often useful to convert these into a dense representation as a numpy array. This method helps do so in just a single line.

```{code-cell} ipython3
dense_end_of_circuit_errorgen_first_order_BCH = errorgen_propagator.errorgen_layer_dict_to_errorgen(propagated_errorgen_layer_first_order)
```

By default this returns the error generator in the normalized pauli-product basis, but this can be changed using the optional `mx_basis` kwarg.
Note: There is another function called `errorgen_layer_to_matrix` available in the `errgenproptools` module with similar functionality to this method, but with a somewhat different interface. That function can be particularly useful in situations where you may want to compute a lot of dense error generator representations from the outputs of the error generator propagation framework, so check out the documentation of that function for more.

+++

#### `approximate_stabilizer_probabilities`
This one is straightforward. Above we showed the use of the function `approximate_stabilizer_probability` from the `errgenproptools` module for calculating approximate output probabilities for a given computational bitstring. If you happen to want *all* of the bit string probabilities you can save yourself a for loop by using the function `approximate_stabilizer_probabilities` from this module instead!

```{code-cell} ipython3
approximate_probabilities = eprop.approximate_stabilizer_probabilities(propagated_errorgen_layer_first_order, c, order=1)
print(approximate_probabilities)
```

Note the returned values are given in right-LSB convention (i.e. '0000' -> '0001' ->'0010', etc.)

+++

#### `error_generator_commutator` and `error_generator_composition`
These two functions from the `errgenproptools` module return the result of analytically computing the commutator and composition of two elementary error generators, respectively.

```{code-cell} ipython3
errorgen_1 = _LSE('H', [stim.PauliString('X')])
errorgen_2 = _LSE('S', [stim.PauliString('Z')])
```

```{code-cell} ipython3
print(eprop.error_generator_commutator(errorgen_1, errorgen_2))
```

```{code-cell} ipython3
print(eprop.error_generator_composition(errorgen_1, errorgen_2))
```

```{code-cell} ipython3
print(eprop.error_generator_composition(errorgen_1, errorgen_1))
```

Both of these methods return their output as a list of two-element tuples. This list is a specification for the linear combination of elementary error generator coefficients corresponding to the commutator or composition of the two input elementary error generators. (First tuple element is an elementary error generator in the linear combination, and the second element is the coefficient of that elementary error generator in the linear combination).

In the examples above we can see that the commutator of the specified H and S error generators gives rise to a pauli-correlation (C) error generator. This could potentially give rise to emergent C error generators when applying second-or-higher order BCH approximations for the effective end-of-circuit error generator, for example. Likewise the composition of these to error generators is a linear combination of a C error generator and an H error generator. And finally we see that squaring an H error generator (composing it with itself) gives rise to a pauli-stochastic (S) error generator.

+++

There's a whole bunch of other functionality and utilities available, particularly in the `errgenproptools` module which have not been covered in this tutorial, so please check out the documentation for additional capabilities!

```{code-cell} ipython3

```
