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

# Operators

State preparations, measurements, and gate or layer operations are the components a `Model` is built from, and each one carries a set of real-valued *parameters* that say how it is allowed to vary.  This page covers what those objects are, how parameterization works, how to combine small operators into large ones, and how to pick parameterization types when you construct a model.  You need this if you're building your own models, or if you just want to pull specific information out of an existing one.  We use the word *operator* generically for all of these objects, even though gate and layer operators act on vectorized density matrices and are therefore *super-operators*.

State preparations and POVM effects are represented as *vectors*.  For $n$ qubits, these can be either length-$2^n$ complex vectors representing pure states and projections, or length-$4^n$ real vectors representing mixed states (in the Liouville picture, where a $2^n\times 2^n$ density matrix is vectorized into a column or row vector).  Gate and layer operations are represented as *linear maps* on the space of state vectors.  As such they can be viewed as $2^n\times 2^n$ complex matrices in the pure-state case, or $4^n \times 4^n$ real matrices in the mixed-state case.

State and effect vectors are subclasses of `pygsti.modelmembers.states.State` and `pygsti.modelmembers.povms.POVMEffect` respectively.  In both cases the vector is stored as a *column* vector even though effect (co-)vectors are perhaps more properly row vectors; this improves code reuse.  Measurement (POVM) objects, which are basically dictionaries of effect vectors, are subclasses of `pygsti.modelmembers.povms.POVM`.  Gate and layer operator objects are subclasses of `pygsti.modelmembers.operations.LinearOperator`.  All of these classes derive from `ModelMember`, the base for every one of pyGSTi's model components.  Every `ModelMember` has a `state_space` attribute specifying the Hilbert or Hilbert-Schmidt space it acts on.  A state space can describe that space in several ways: by a number of qubits $n$ (`num_qubits`), a unitary operator dimension $2^n$ (`udim`), or a superoperator dimension $4^n$ (`dim`).

## Parameters are the whole point

The fundamental job of a `Model` is to simulate circuits, mapping circuits to outcome probability distributions.  That mapping is *parameterized*: it depends on the values of a vector of real numbers.  A `Model` has a `num_params` attribute holding its parameter count, plus `to_vector` and `from_vector` methods that get and set the parameter vector.  `ModelMember` objects have exactly the same three, and for models that hold members to implement their operations (both explicit and implicit models do), the model's parameterization is the result of combining the parameterizations of its members.  In the simplest case each gate and SPAM vector in an `ExplicitOpModel` is parameterized independently, and the model's parameter vector is just the concatenation of its members' parameter vectors, usually ordered: state preparations, then measurements, then gates.

For an explicit model the parameterization is a mapping from the model's parameter space into the space of $d^2 \times d^2$ operation matrices and length-$d^2$ SPAM vectors.  A model's contents always correspond to some valid set of parameters, and can always be reinitialized from a parameter vector.  The parameter count need not equal, and usually doesn't equal, the total number of matrix and vector elements.  In a TP-parameterized model, for instance, the first row of every operation matrix is pinned at `[1,0,...,0]` no matter what the underlying parameters are.

This matters because one of pyGSTi's primary capabilities is model optimization: fitting a function (often the log-likelihood) over the parameter space of a starting model, often the target model.  Specifying a parameterization specifies the constraints the optimization runs under, or equivalently the space of circuit-to-outcome-distribution mappings that gets searched for a best fit.

## How the classes are organized

Most classes in `pygsti.modelmembers.operations` represent a unique combination of two things:

a. a category of operation that can be represented, and
b. a parameterization of that category.

`FullArbitraryOp` can represent an arbitrary (Markovian) operation and "fully" parameterizes it, exposing every element of the dense process matrix as a parameter.  `StaticCliffordOp` can only represent Clifford operations, and is "static": it exposes no parameters, so an optimizer cannot change it.  The classes in `pygsti.modelmembers.states` and `pygsti.modelmembers.povms` run parallel to these.  `FullState` and `TPState` mirror `FullArbitraryOp` and `FullTPOp`, with `TPState` freezing its first element at $1/\sqrt{d}$ (where the vector has length $d^2$), the value that makes the represented density matrix have unit trace in the Pauli or Gell-Mann basis.

A separate group of classes combines or modifies other operations rather than representing a category of its own; those inherit category and parameterization from whatever they act on.  They're covered further down, under composing and embedding.

## Setup

```{code-cell} ipython3
import pprint

import numpy as np

import pygsti
from pygsti.modelmembers import states, povms, operations as ops
from pygsti.models import modelconstruction as mc
from pygsti.processors import QubitProcessorSpec
```

Before touching the pyGSTi objects, generate some example state vectors and a gate matrix.  These are plain NumPy arrays.  `stdmx_to_ppvec` converts a standard $2^n \times 2^n$ complex Hermitian density matrix into a length-$4^n$ "state vector" of real numbers giving that density matrix's decomposition in the Pauli basis.  `gate_mx` describes how a 1-qubit $X(\pi/2)$ rotation transforms a state vector in the Pauli basis.

```{code-cell} ipython3
gate_mx = np.array([[1,   0,   0,   0],
                    [0,   1,   0,   0],
                    [0,   0,   0,  -1],
                    [0,   0,   1,   0]],'d')
density_mx0 = np.array([[1, 0],
                        [0, 0]], complex)
density_mx1 = np.array([[0, 0],
                        [0, 1]], complex)
state_vec0 = pygsti.tools.stdmx_to_ppvec(density_mx0)
state_vec1 = pygsti.tools.stdmx_to_ppvec(density_mx1)

print(state_vec0) # just a numpy column vector 
print(state_vec0.dtype) # of *real* numbers
```

## Dense operators

The simplest operators look a lot like NumPy arrays in which some elements are read-only.  These derive from `DenseOperator` or `DenseState` and hold a *dense* representation, meaning an actual vector or matrix sits in memory.  Three parameterizations show up constantly:

- **static**: zero parameters, so the object cannot be changed at all.  Static operators are like read-only NumPy arrays.
- **full**: one independent parameter per element of the dense vector or matrix.  Fully parameterized objects behave like ordinary NumPy arrays.
- **trace-preserving (TP)**: like full, except the top row of gate and layer matrices, and the first element of state preparation vectors, is fixed and therefore not a parameter.  A POVM that is trace preserving must have all its effect vectors sum to the identity.

Here's a 1-qubit example of building dense-operator objects:

```{code-cell} ipython3
#Prep vectors
tpSV = states.TPState(state_vec0)
staticSV = states.StaticState(state_vec0)

#Operations
staticOp = ops.StaticArbitraryOp(gate_mx)
fullOp = ops.FullArbitraryOp(gate_mx)
tpOp = ops.FullTPOp(gate_mx)

#Effect vectors - just conjugated state vectors
staticEV = povms.ConjugatedStatePOVMEffect(states.StaticState(state_vec0))
fullEV = povms.ConjugatedStatePOVMEffect(states.FullState(state_vec0))

#POVMs (must specify evotype when constructing using non-POVMEffect objects in 2nd line))
povm = povms.UnconstrainedPOVM( {'outcomeA': staticEV, 'outcomeB':fullEV})
tppovm = povms.TPPOVM( {'0': state_vec0, '1': state_vec1}, evotype='default')

for op in (tpSV,staticSV,staticOp,fullOp,tpOp,staticEV,fullEV,povm,tppovm):
    print("%s object has %d parameters" % (str(type(op)), op.num_params))
```

There are exceptions, but the usual way to set the value of a `State`, `POVM`, or `LinearOperator` is to set its parameter values.  Parameters must be real-valued and are typically allowed to range over all the reals, so you update an operator by passing a real-valued NumPy array (a *parameter vector*) to its `from_vector` method.  The length of that array has to match the operator's `num_params`.

Now set new parameter values on several of the objects created above.  Dense operators have a direct correspondence between parameters and matrix or vector elements, so the parameter vector can be a flattened version of a 2d array of the parameterized element values.

```{code-cell} ipython3
new_vec = np.array([1/np.sqrt(2),0,0],'d')
tpSV.from_vector(new_vec)
print("params = ",tpSV.to_vector())
print(tpSV)

new_mx = np.array([[1,   0,   0,   0],
                   [0,   1,   0,   0],
                   [0,   0,   0,-0.9],
                   [0,   0, 0.9,   0]],'d')
fullOp.from_vector(new_mx.flatten())
print("params = ",fullOp.to_vector())
print(fullOp)

new_mx = np.array([[0,   1,   0,   0],
                   [0,   0,   0,-0.9],
                   [0,   0, 0.9,   0]],'d')
tpOp.from_vector(new_mx.flatten())
print("params = ",tpOp.to_vector())
print(tpOp)


new_vec = np.array([1/np.sqrt(2),1/np.sqrt(2),0,0],'d')
fullEV.from_vector(new_vec)
print("params = ",fullEV.to_vector())
print(fullEV)

new_effect = np.array([1/np.sqrt(2),0.9*1/np.sqrt(2),0,0],'d')
tppovm.from_vector(new_effect)
print("params = ",tppovm.to_vector())
print(tppovm)
```

## Lindblad (CPTP-constrained) operations

A gate or layer operation is guaranteed completely positive and trace preserving (CPTP) if it has the form $\hat{O} = \exp{\mathcal{L}}$ where $\mathcal{L}$ takes the Lindblad form:
$$\mathcal{L}: \rho \rightarrow \sum_i -i\lambda_i[\rho,B_i] + \sum_{ij} \eta_{ij} \left( B_i \rho B_j^\dagger - \frac{1}{2}\left\{ B_i^\dagger B_j, \rho \right\} \right) $$
where $B_i$ ranges over the non-identity elements of the ($n$-qubit) Pauli basis, $\lambda_i$ is real, and $\eta \ge 0$ (the matrix $\eta_{ij}$ is Hermitian and positive semidefinite).  The $\lambda_i$ terms are *Hamiltonian error* terms, and the real $\lambda_i$ are *error rates* or *error coefficients*.  The $\eta_{ij}$ terms are *non-Hamiltonian error* terms.  When $\eta$ is diagonal the terms are called *Pauli stochastic error* terms and the real $\eta_{ii} \ge 0$ are error rates.  A caveat worth stating plainly: while every map of the form $\hat{O}$ is CPTP, not every CPTP map has this form.  $\hat{O}$ is the form of all *infinitesimally-generated* CPTP maps.

Say you want to represent an operation $e^{\mathcal{L}} U_0$, where $U_0$ is a unitary (super-)operator and $\mathcal{L}$ takes the Lindblad form above.  You need three objects: a `LindbladErrorgen` encapsulating the Lindbladian exponent $\mathcal{L}$, an `ExpErrorgenOp` to do the exponentiation, and a `ComposedOp` to combine it with the target unitary $U_0$ (more on `ComposedOp` below).  Lindblad operators are among the most complicated things in pyGSTi, so this section moves slowly.

Start by assuming $U_0 = I$ and making a CPTP operation from a dense gate matrix:

```{code-cell} ipython3
cptpGens = ops.LindbladErrorgen.from_operation_matrix(gate_mx)
cptpOp = ops.ExpErrorgenOp(cptpGens)
```

An `ExpErrorgenOp` does *not* necessarily hold a dense representation of its process matrix (it isn't a `DenseOperator`), so you cannot index into it like a NumPy array.  For a dense representation, call `to_dense()`, which works on dense operators too:

```{code-cell} ipython3
print(cptpOp)
print("dense representation = ")
pygsti.tools.print_mx(cptpOp.to_dense()) # see this equals `gate_mx`
```

Now look at the parameters.  By default, the $\mathcal{L}$ of a `LindbladErrorgen` is parameterized so that $\eta \ge 0$ and the resulting map is CPTP.  Several other parameterizations are available, selected by the `parameterization` argument of construction functions like the `from_operation_matrix` used above:

- `"H"` : Hamiltonian ($\lambda_i$) parameters are allowed.  These model coherent errors.
- `"S"` : Pauli stochastic ($\eta_{ii}$) parameters are allowed, with the constraint $\eta_{ii} \ge 0$ (required to keep the map completely positive).
- `"s"` : Pauli stochastic ($\eta_{ii}$) parameters are allowed without the non-negativity constraint.
- `"D"` and `"d"` : Same as `"S"` and `"s"` except all the parameters are constrained equal to one another, as they would be for depolarizing errors.
- `"A"` : affine parameters are allowed.  These are particular linear combinations of the $\eta_{ij}$ that produce affine errors, generators of the form $$A_i : \rho \rightarrow \mathrm{Tr}(\rho_{target})B_i \otimes \rho_{non-target}$$ where the *target* and *non-target* parts of $\rho$ are the qubits on which $B_i$ is nontrivial and trivial respectively (in an `IXI` term, qubit 2 is the target space and qubits 1 and 3 are the non-target space).  Such an $A_i$ is a linear combination of the non-Hamiltonian ($\eta_{ij}$) terms: since $\rho \rightarrow I$ can be written as $\rho \rightarrow \frac{1}{d^2} \sum_i B_i \rho B_i$ with the sum over *all* Paulis including the identity, a map $\rho \rightarrow B_k$ can be written $\rho \rightarrow \frac{1}{d^2} \sum_i B_i B_k \rho B_i B_k$.  Affine parameters must be accompanied by Pauli stochastic (`"S"`-type) errors.
- `"CPTP"` : All Lindblad parameters ($\lambda_i$ and $\eta_{ij}$) are allowed.  The Hermitian matrix $\eta$ is constrained to be positive semidefinite (required to keep the map completely positive).
- `"GLND"` : All Lindblad parameters ($\lambda_i$ and $\eta_{ij}$) are allowed, with no constraint other than $\eta$ being Hermitian.
- Combinations of the single-letter options above, joined with a plus sign.  `"H+S"` allows Hamiltonian and Pauli stochastic errors.  Since `"A"` can only be used alongside `"S"` or `"s"`, `"H+A"` is invalid while `"H+S+A"` and `"s+A"` are fine.

Pull the parameters out with `to_vector()`:

```{code-cell} ipython3
print("params (%d) = " % cptpOp.num_params, cptpOp.to_vector(),'\n')
```

One parameter is $\pi/4$ and the other eleven are zero.  `from_operation_matrix` finds the error generator whose exponential *is* `gate_mx`; it does not factor out an ideal $U_0$ for you.  The $X(\pi/2)$ rotation therefore lands inside $\mathcal{L}$ as a single Hamiltonian term, which is why `to_dense()` above reproduced `gate_mx` exactly.  If you want an ideal gate with small errors on top, build the $U_0$ factor yourself with a `ComposedOp`, as further down.

Parameters and error coefficients are not the same thing in general.  `errorgen_coefficients` retrieves the coefficients, which is usually more useful than raw parameter values:

```{code-cell} ipython3
coeff_dict, basis = cptpOp.errorgen_coefficients(return_basis=True)
print("Coefficients in (<type>,<basis_labels>) : value form:"); pprint.pprint(coeff_dict)
print("\nBasis containing elements:"); pprint.pprint(basis.labels)
```

`errorgen_coefficients` returns a dictionary, plus a basis if `return_basis=True`.  The keys name individual *elementary* error generators and the values are their coefficients (rates).  Each key printed above is a three-part tuple `(<type>, <basis labels>, <state space labels>)`.  The type is one of four letters, following arXiv:2103.01928, and it determines how the generator acts on a density matrix.  With $P$ and $Q$ the Pauli basis elements named by the basis labels:

- `"H"` (Hamiltonian), one basis label: $$\rho \rightarrow -i[P,\rho]$$
- `"S"` (stochastic), one basis label: $$\rho \rightarrow P \rho P^\dagger - \tfrac{1}{2}\{P^\dagger P, \rho\}$$
- `"C"` (correlation), two basis labels: $$\rho \rightarrow P \rho Q^\dagger + Q \rho P^\dagger - \tfrac{1}{2}\{P^\dagger Q + Q^\dagger P, \rho\}$$
- `"A"` (active), two basis labels: $$\rho \rightarrow i\left(P \rho Q^\dagger - Q \rho P^\dagger + \tfrac{1}{2}\{P^\dagger Q - Q^\dagger P, \rho\}\right)$$

The `"H"` and `"S"` coefficients are the $\lambda_i$ and $\eta_{ii}$ of the Lindblad expansion above; `"C"` and `"A"` together carry the real and imaginary parts of the off-diagonal $\eta_{ij}$, giving a real-valued coordinate system for the Hermitian $\eta$ matrix.  The basis labels are strings over I, X, Y, and Z referencing matrices in the `Basis` that `return_basis=True` gives you.  The state-space labels record which qudits the generator acts on nontrivially.

Be careful with the letter `"A"`: in an elementary error generator label it means *active*, as above, but in a `parameterization` string like `"H+S+A"` it means *affine*.  The two are unrelated.

Most functions that consume these labels also accept a shorter "local" form that drops the state-space labels, so `("H","X")` and `("C","X","Y")` work as input.  A tuple like `("S","X","X")` is a trap: it parses as a stochastic label with two basis elements, which no `LindbladErrorgen` block manages, and the term is silently dropped instead of raising.  Write `("S","X")`.

`set_errorgen_coefficients` does the reverse of `errorgen_coefficients`: it sets parameter values from a dictionary in this format.

You can also initialize a `LindbladErrorgen` directly from such a dictionary.  Below, build one with
$$\mathcal{L} = 0.1 H_X + 0.1 S_X$$
then use it with an `ExpErrorgenOp` and a `ComposedOp` to get an operator corresponding to $e^{\mathcal{L}}U_0$ with $U_0$ an $X(\pi/2)$ rotation.

```{code-cell} ipython3
stdXOp = ops.StaticUnitaryOp.from_standard_gate_name('Gxpi2')
cptpGen2 = ops.LindbladErrorgen.from_elementary_errorgens({('H','X'): 0.1, ('S','X'): 0.1}, state_space=1)
cptpOp2 = ops.ExpErrorgenOp(cptpGen2)
noisyXOp = ops.ComposedOp([stdXOp, cptpOp2])
print(cptpOp2)
print("exp(L) = "); pygsti.tools.print_mx(cptpOp2.to_dense())
print("exp(L) U0 = "); pygsti.tools.print_mx(noisyXOp.to_dense())
```

Check that the operator has the intended error generator coefficients.  This time reach through the `errorgen` member of the `ExpErrorgenOp`:

```{code-cell} ipython3
cptpOp2.errorgen.coefficients() # same as cptpOp2.errorgen_coefficients()
```

An inconvenience arises because an error generator gets *exponentiated* to form a map.  The coefficients of the stochastic generators therefore don't correspond exactly to the error rates of the final map as people usually think of them.  Take a simple case: construct a depolarizing map with a process fidelity of 90%.  You might think this would do it:

```{code-cell} ipython3
test_depol_op = ops.ComposedOp([
    ops.StaticArbitraryOp(np.identity(4)),
    ops.ExpErrorgenOp(
        ops.LindbladErrorgen.from_elementary_errorgens({('S','X'): 0.1/3, ('S','Y'): 0.1/3, ('S','Z'): 0.1/3},
            state_space=1)
    )
])
pygsti.tools.entanglement_fidelity(test_depol_op.to_dense(), np.identity(4))
```

The fidelity isn't quite $0.9$.  The error rate of the *map* whose error generator has all stochastic-term coefficients equal to $C$ is $(1 - e^{-d^2 C}) / d^2$, not $C$.  The `error_rates` and `set_error_rates` methods account for this transformation; they behave like `errorgen_coefficients` and `set_errorgen_coefficients` except that they convert S-values between coefficients and map error rates internally.  Using `set_error_rates` fixes the example:

```{code-cell} ipython3
test_depol_op.set_error_rates({('S','X'): 0.1/3, ('S','Y'): 0.1/3, ('S','Z'): 0.1/3})
pygsti.tools.entanglement_fidelity(test_depol_op.to_dense(), np.identity(4))
```

```{code-cell} ipython3
# And we can see that the errorgen coefficients have been adjusted accordingly
test_depol_op.errorgen_coefficients()
```

### Lindblad state preparations and POVMs

State preparations and POVMs can use error generators too: swap `ComposedOp` for `ComposedState` or `ComposedPOVM`.  These compose an operation (say an $\exp\mathcal{L}$ factor from an `ExpErrorgenOp` and a `LindbladErrorgen`) with an existing "base" state preparation or POVM.  State preparations become $e^{\mathcal{L}} |\rho_0\rangle\rangle$ for a base pure state $|\rho_0\rangle\rangle$, and effect vectors become $\langle\langle E_i | e^{\mathcal{L}}$ for the effects $\langle\langle E_i|$ of a base POVM.

```{code-cell} ipython3
#Spam vectors and POVM
errorgen = ops.LindbladErrorgen.from_elementary_errorgens({('S','X'): 0.1/3, ('S','Y'): 0.1/3, ('S','Z'): 0.1/3}, state_space=1)
cptpSpamVec = states.ComposedState(staticSV, errorgen) # staticSV is the "base" state preparation
cptpPOVM = povms.ComposedPOVM(ops.ExpErrorgenOp(errorgen)) # by default uses the computational-basis POVM
```

## Composing and embedding operators

pyGSTi builds "large" operators (complex, or many-qubit) out of smaller ones.  You saw a modest example above when an `ExpErrorgenOp` was built from a `LindbladErrorgen`.  These classes don't represent a category of operation of their own; they inherit category and parameterization from whatever they wrap:

- `ComposedOp` combines zero or more operations by acting them one after the other.  Its process matrix is the product (in reversed order) of the factors' process matrices.  It does not require its factors to have dense representations; the factors can be *any* `LinearOperator` objects.  Dense versions can be faster when the qubit count is small.
- `ComposedErrorgen` combines zero or more error generators by summing them.
- `EmbeddedOp` maps an operation on a subsystem of a state space into the full state space, taking, for instance, a 1-qubit $X(\pi/2)$ rotation and making a 3-qubit operation that applies it to the second qubit.  This is how layer operations get built in multi-qubit models, where you naturally want to work with 1- and 2-qubit operations and assemble $n$-qubit layers from them.
- `EmbeddedErrorgen` embeds a lower-dimensional error generator into a higher-dimensional space.
- `ExpErrorgenOp` exponentiates an error generator, turning it into a map on quantum states.
- `RepeatedOp` repeats a single operation $k$ times.

The same combinators exist for states, POVMs, and the rest under the other subpackages of `pygsti.modelmembers`.

### Composed operations

Compose several of the dense operations from earlier:

```{code-cell} ipython3
composedOp = ops.ComposedOp((staticOp,tpOp,fullOp))
print(composedOp)
print("Before interacting w/Model:",composedOp.num_params,"params")
```

As expected there are $0+12+16=28$ parameters, the sum of the factors' parameter counts.

### Embedded operations

Here's how to embed a single-qubit operator (`fullOp`, from above) into a 3-qubit state space so that it acts on the second qubit, labelled `"Q1"`.  The parameters of an `EmbeddedOp` are just those of the operator being embedded.

```{code-cell} ipython3
embeddedOp = ops.EmbeddedOp(['Q0','Q1','Q2'],['Q1'],fullOp)
print(embeddedOp)
print("Dimension =",embeddedOp.dim, "(%d qubits!)" % (np.log2(embeddedOp.dim)/2))
print("Number of parameters =",embeddedOp.num_params)
```

### Better together

Combinations of composed and embedded objects give you more complex operations.  Here's a 3-qubit operation that performs three separate 1-qubit operations (`staticOp`, `fullOp`, and `tpOp`) on the three qubits.  These three all *happen* to be $X(\pi/2)$ gates because the examples above were lazy about varying `gate_mx`, but they *could* be entirely different.  The resulting `combinedOp` might represent a layer in which all three gates occur simultaneously.

```{code-cell} ipython3
# use together
mdl_3Q = pygsti.models.ExplicitOpModel(['Q0','Q1','Q2'])
combinedOp = ops.ComposedOp( (ops.EmbeddedOp(['Q0','Q1','Q2'],['Q0'],staticOp),
                             ops.EmbeddedOp(['Q0','Q1','Q2'],['Q1'],fullOp),
                             ops.EmbeddedOp(['Q0','Q1','Q2'],['Q2'],tpOp))
                          )
mdl_3Q.operations[(('Gstatic','Q0'),('Gfull','Q1'),('Gtp','Q2'))] = combinedOp
mdl_3Q.num_params # to recompute & allocate the model's parametes
print(combinedOp)
print("Number of parameters =",combinedOp.num_params)
```

## Choosing types when you build a model

Most of the time you don't construct modelmembers by hand.  The model construction functions take arguments naming the type of modelmember to create, and by choosing a gate's type you select both how it is represented (Clifford gates can be represented much more efficiently than arbitrary ones) and how it is parameterized, which in turn fixes how the whole model is parameterized.

Below is an incomplete list of type strings and the classes they map to.  Most start with `"full"` or `"static"`, indicating whether the members have parameters or not; a type with no prefix is "full" by default.  See the [forward simulation tutorial](../simulation/ForwardSimulators) for how each parameterization relates to the forward simulation types pyGSTi supports.

- gate types, in `pygsti.modelmembers.operations`:
  - `"static"` $\rightarrow$ `StaticArbitraryOp`
  - `"full"` $\rightarrow$ `FullArbitraryOp`
  - `"static clifford"` $\rightarrow$ `StaticCliffordOp`
  - `"static unitary"` $\rightarrow$ `StaticUnitaryOp`
  - `"full unitary"` $\rightarrow$ `FullUnitaryOp`
  - `"full TP"` $\rightarrow$ `FullTPOp`
  - `"CPTP"`, `"H+S"`, etc. $\rightarrow$ `ExpErrorgenOp` + `LindbladErrorgen`
  - `"static standard"` $\rightarrow$ [Deprecated] aliases `("static unitary", "static")`


- state preparation types, in `pygsti.modelmembers.states`:
  - `"computational"` $\rightarrow$ `ComputationalBasisState`
  - `"static pure"` $\rightarrow$ `StaticPureState`
  - `"full pure"` $\rightarrow$ `FullPureState`
  - `"static"` $\rightarrow$ `StaticState`
  - `"full"` $\rightarrow$ `FullState`
  - `"full TP"` $\rightarrow$ `TPState`


- POVM types, in `pygsti.modelmembers.povms`:
  - `"computational"` $\rightarrow$ `ComputationalBasisPOVM`
  - `"static pure"` $\rightarrow$ `UnconstrainedPOVM` + `StaticPOVMPureEffect`
  - `"full pure"` $\rightarrow$ `UnconstrainedPOVM` + `FullPOVMPureEffect`
  - `"static"` $\rightarrow$ `UnconstrainedPOVM` + `StaticPOVMEffect`
  - `"full"` $\rightarrow$ `UnconstrainedPOVM` + `FullPOVMEffect`
  - `"full TP"` $\rightarrow$ `TPPOVM`

Which argument carries which of these depends on the function.  `ExplicitOpModel.set_all_parameterizations` and `create_explicit_model_from_expressions` take separate `gate_type`, `prep_type`, `povm_type`, and `instrument_type` arguments.  `create_explicit_model` and `create_crosstalk_free_model` instead take `ideal_gate_type` and a single `ideal_spam_type` covering both preps and POVMs, because they build operations as compositions of an ideal part and a noise part, and these arguments set the type of the ideal part only (see the [model noise tutorial](../../guides/models/ModelNoise)).

The `prep_type` and `povm_type` arguments also accept `"auto"`, which picks a parameterization from the given gate type.  An incomplete list of that mapping:

- `"auto"`, `"static standard"`, `"static clifford"` $\rightarrow$ `"computational"`
- `"static unitary"` $\rightarrow$ `"static pure"`, `"full unitary"` $\rightarrow$ `"full pure"`
- All others map directly

### Explicit models

```{code-cell} ipython3
pspec = QubitProcessorSpec(1, ['Gi', 'Gxpi2', 'Gypi2'])  # simple single qubit processor
model = mc.create_explicit_model(pspec)
model.print_modelmembers()
print("%d parameters" % model.num_params)
```

By default an explicit model creates static (zero-parameter) operations.  For gates named in the standard gate set, as here, the default `ideal_gate_type='auto'` lands on `StaticUnitaryOp`.  Specify an `ideal_gate_type` to change that:

```{code-cell} ipython3
model = mc.create_explicit_model(pspec, ideal_gate_type="full TP")
model.print_modelmembers()
print("%d parameters" % model.num_params)
```

`set_all_parameterizations` converts an existing model's contents wholesale.  Switching to `"CPTP"` changes the gate type accordingly.  This conversion works best when you supply an `ideal_model` of unitary gates and pure states to target, which avoids branch cuts in the conversion; here that's the default static-unitary model built from the same processor spec.

```{code-cell} ipython3
ideal_model = mc.create_explicit_model(pspec)  # static, unitary ideal gates/SPAM
model.set_all_parameterizations('CPTP', ideal_model=ideal_model)
model.print_modelmembers()
print("%d parameters" % model.num_params)
```

To change an *individual* gate or SPAM vector's parameterization, construct a replacement object of the type you want and assign it into the model.

```{code-cell} ipython3
# Turning ComposedOp into a dense matrix for conversion into a dense FullTPOp
newOp = pygsti.modelmembers.operations.FullTPOp(model[('Gi', 0)].to_dense())
model['Gi'] = newOp
print("model['Gi'] =",model['Gi'])
```

Assignment behaves differently depending on what you assign.  A `ModelMember`-derived object (a `LinearOperator`, `State`, or `POVM`) *replaces* whatever was stored under that key, as above.  Anything else, a bare NumPy array for example, is used to initialize or update the existing object in place rather than replacing it:

```{code-cell} ipython3
numpy_array = np.array( [[1, 0, 0, 0],
                         [0, 0.5, 0, 0],
                         [0, 0, 0.5, 0],
                         [0, 0, 0, 0.5]], 'd')
model['Gi'] = numpy_array # after assignment with a numpy array...
print("model['Gi'] =",model['Gi']) # this is STILL a FullTPOp object

#If you try to assign a gate to something that is either invalid or it doesn't know how
# to deal with, it will raise an exception
invalid_TP_array = np.array( [[2, 1, 3, 0],
                              [0, 0.5, 0, 0],
                              [0, 0, 0.5, 0],
                              [0, 0, 0, 0.5]], 'd')
try:
    model['Gi'] = invalid_TP_array
except ValueError as e:
    print("ERROR!! " + str(e))
```

### Implicit models

The story is similar for implicit models.  Operations are compositions of ideal operations and noise, and `ideal_gate_type` and friends set the type of the ideal part.  Here's a `LocalNoiseModel` with the default static operation type:

```{code-cell} ipython3
mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec)
mdl_locnoise.print_modelmembers()
```

To modify the gate operations you need parameters, so build the model with `ideal_gate_type="full"` and get `FullArbitraryOp` objects:

```{code-cell} ipython3
mdl_locnoise = pygsti.models.create_crosstalk_free_model(pspec, ideal_gate_type='full')
mdl_locnoise.print_modelmembers()
```

Those can be modified by matrix assignment, since their parameters let them take on any process matrix.  Set the process matrix of `"Gxpi2"` (more precisely, the Pauli transfer matrix of the gate) to include some depolarization:

```{code-cell} ipython3
mdl_locnoise.operation_blks['gates']['Gxpi2'] = np.array([[1,   0,   0,   0],
                                                          [0, 0.9,   0,   0],
                                                          [0,   0,-0.9,   0],
                                                          [0,   0,   0,-0.9]],'d')
```

`CloudNoiseModel` objects work differently.  All of the parameterization is inherited from the noise operations, so `create_cloud_crosstalk_model` has no `ideal_gate_type` argument at all; the ideal operations in a cloud noise model are always static.  The [model noise tutorial](../../guides/models/ModelNoise) covers how to set the types of the noise objects.

## What this page doesn't cover

There are more model-building objects than fit here, and the coverage above is not exhaustive even for the classes it names.  For writing your own operator classes see the [custom operator tutorial](CustomOperators); for parameter sharing between members see [tying parameters](TyingParameters).
