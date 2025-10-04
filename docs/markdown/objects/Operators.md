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

# Tutorial on operators (SPAM, gate, and layer operations)

This tutorial explains the objects that represent state preparation, measurement, gate, and layer operations in pyGSTi.  These objects form the essential components of `Model` objects in pyGSTi, and are therefore an important topic to understand if you're creating your own models or just need to know how to extract specific information from a `Model`.  We use the term *operator* generically all such objects, even when gate or layer operators act on vectorized density matrices and are therefore *super-operators*. 

State preparations and POVM effects are represented as *vectors* in pyGSTi.  For $n$ qubits, these can be either length-$2^n$ complex vectors representing pure states/projections or length-$4^n$ real vectors representing mixed states (in the Liouville picture, where we vectorize a $2^n\times 2^n$ density matrix into a column or row vector).  Gate and layer operations are represented as *linear maps* on the space of state vectors.  As such these can be viewed as $2^n\times 2^n$ complex matrices (in the pure-state case) or $4^n \times 4^n$ real matrices (in the mixed-state case).

State and effect vectors are subclasses of `pygsti.modelmembers.states.State` and `pygsti.modelmembers.povms.POVMEffect` respectively.  In both cases the vector is stored as a *column* vector even though effect (co-)vectors are perhaps more properly row vectors (this improves code reuse).  Measurement (POVM) objects, which are basically dictionaries of effect vectors, are subclasses of `pygsti.modelmembers.povms.POVM`.  Gate and layer operator objects are subclasses of `pygsti.modelmembers.operations.LinearOperator`.  All of these classes (`State`, `POVMEffect`, `POVM`, and `LinearOperator`) are derived from `ModelMember` which forms the base for all of pyGSTi's model components.  All `ModelMember` objects have a `state_space` attribute, which specifies the Hilbert or Hilbert-Schmidt space that they act upon.  A state space can describe this space in multiple ways: by a number of qubits $n$ (`num_qubits` attribute), a unitary operator dimension $2^n$ (`udim`) or a superoperator dimension $4^n$ (`dim`).

Let's begin with some familiar imports.

```{code-cell} ipython3
import numpy as np

import pygsti
from pygsti.modelmembers import states, povms, operations as ops
```

Before getting into the pyGSTi objects, let's generate some example state vectors and gate matrix.  These are just NumPy arrays, and we use the `stdmx_to_ppvec` function to convert a standard $2^n \times 2^n$ complex Hermitian densiy matrix to a length $4^n$ "state vector" of real numbers giving the decomposition of this density matrix in the Pauli basis.  The `gate_mx` describes how a 1-qubit $X(\pi/2)$ rotation transforms a state vector in the Pauli basis.

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

The simplest kind of operators look very similar to numpy arrays (like those we created above) in which some of the elements are read-only.  These operators derive from `DenseOperator` or `DenseState` and hold a *dense* representation, meaning the a dense vector or matrix is stored in memory.  **SPAM, gate, and layer operators have parameters which describe how they can be varied**, essentially the "knobs" which you can turn.  `Model` objects also have *parameters* that are essentially inherited from their contained operators.  How an operator is parameterized is particularly relevant for protocols which optimize a `Model` over its parameter space (e.g. Gate Set Tomography).  See the tutorial on [model parameterization](ModelParameterization.ipynb) for more information.  Three common parameterizations are:
- **static**: the object has *no* (zero) parameters, so the object cannot be changed at all.  Static operators are like read-only NumPy arrays.
- **full**: the object has one independent parameter for each element of its (dense) vector or matrix.  Fully parameterized objects are like normal NumPy arrays.
- **trace-preserving (TP)**: similar to full, except the top row of gate/layer matrices and the first element of state preparation vectors is fixed and these elements are therefore not parameters.  (A POVM that is trace preserving must have all of its effect vectors sum to the identity.)

Here's a 1-qubit example of creating dense-operator objects:

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

Although there are certain exceptions, the usual way you set the value of a `State`, `POVM` or `LinearOperator` object is by setting the values of its parameters.  Parameters must be real-valued and are typically allowed to range over all real numbers, so updating an operator's parameter-values is accomplished by passing a real-valued NumPy array of parameter values - a *parameter vector* - to the operators `from_vector` method.  Note that the length of the parameter vector must match the operator's number of parameters (returned by `num_params` as demonstrated above).  

We'll now set new parameter values for several of the operators we created above.  Since for dense operators there's a direct correspondence between parameters and matrix or vector elements, the parameter vector may be a flattened version of a 2d array of the parameterized element values.

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

That a gate or layer operation is completely-positive and trace-preserving (CPTP) can be guaranteed if the operation is given by $\hat{O} = \exp{\mathcal{L}}$ where $\mathcal{L}$ takes the Lindblad form:
$$\mathcal{L}: \rho \rightarrow \sum_i -i\lambda_i[\rho,B_i] + \sum_{ij} \eta_{ij} \left( B_i \rho B_j^\dagger - \frac{1}{2}\left\{ B_i^\dagger B_j, \rho \right\} \right) $$
where $B_i$ range over the non-identity elements of the ($n$-qubit) Pauli basis, $\lambda_i$ is real, and $\eta \ge 0$ (i.e. the matrix $\eta_{ij}$ is Hermitian and positive definite).  We call the $\lambda_i$ terms *Hamiltonian error* terms, and the (real) $\lambda_i$s the *error rates* or *error coefficients*.  Likewise, the $\eta_{ij}$ terms are referred to generically as *non-Hamiltonian error* terms.  In the special case where the $\eta$ matrix is diagonal, the terms are called *Pauli stochastic error* terms and the (real) $\eta_{ii} > 0$ are error rates.  **Technical note:** While all maps of the above form ($\hat{O}$) are CPTP, not all CPTP maps are of this form.  $\hat{O}$ is the form of all *infinitesimally-generated* CPTP maps.  

Say we want to repsent an operation $e^{\mathcal{L}} U_0$, where $U_0$ is a unitary (super-)operator and $\mathcal{L}$ takes the Lindblad form given above.  The way to do this is using a `LindbladErrorgen` object that encapsulates the Lindbladian exponent $\mathcal{L}$, an `ExpErrorgenOp` to do the exponentiation, and a `ComposedOp` to combine it with the target unitary $U_0$ (more on the `ComposedOp` below). Lindblad operators are among the most complicated of all the operators in pyGSTi, so bear with us as we try to present things in an organized and comprehensible way. 

Let's start by assuming $U_0 = I$ and making a CPTP operation using `LindbladErrorgen` and `ExpErrorgenOp` from a dense gate matrix:

```{code-cell} ipython3
cptpGens = ops.LindbladErrorgen.from_operation_matrix(gate_mx)
cptpOp = ops.ExpErrorgenOp(cptpGens)
```

A `LindbladOp` does *not* necessarily hold a dense representation of its process matrix (it's not a `DenseOperator`), and so you cannot access it like a Numpy array.  If you want a dense representation, you can call the `to_dense()` method (which works on dense operators too!):

```{code-cell} ipython3
print(cptpOp)
print("dense representation = ")
pygsti.tools.print_mx(cptpOp.to_dense()) # see this equals `gate_mx`
```

Now let's look at the parameters of `cptpOp`.  By default, the $\mathcal{L}$ of a `LindbladErrorgen` is parameterized such that $\eta \ge 0$ and the `LindbladOp` map is CPTP.  There are several other ways a $\mathcal{L}$ can be parameterized, and these are specified by the values of the `parameterization` argument of construction functions like `from_operation_matrix` we used above.  Here's a quick rundown on these options:

- `"H"` : Hamiltonian ($\lambda_i$) parameters are allowed.  These model coherent errors.
- `"S"` : Pauli stochastic ($\eta_{ii}$) parameters are allowed, with the constraint $\eta_{ii} \ge 0$ (required to keep the map completely positive).
- `"s"` : Pauli stochastic ($\eta_{ii}$) parameters are allowed without non-negativivty constraint.
- `"D"` and `"d"` : Same as `"S"` and `"s"` except all of the parameters are constrained to be equal to one another, as they would be for depolarizing errors.
- `"A"` : affine parameters are allowed.  These are particular linear combinations of the $\eta_ij$ that produce affine errors.  These must be accompanied by Pauli stochastic (`"S"`-type) errors.
- `"CPTP"` : All Lindblad parameters ($\lambda_i$ and $\eta_{ij}$) are allowed.  The Hermitian matrix $\eta$ is constrained to be positive semidefinite (required to keep the map completely positive).
- `"GLND"` : All Lindblad parameters ($\lambda_i$ and $\eta_{ij}$) are allowed.  No constraints other than that $\eta$ be Hermitian.
- Combinations of the single-letter options above, joined with a plus (+) sign combine these types.  For example, `"H+S"` allows Hamiltonian and Pauli stochastic errors.  As stated above `"A"` can only be used along with `"S"` or `"s"`, so `"H+A"` is an invalid parameterization but `"H+S+A"` or `"s+A"`, for example, are fine.

Let's get these parameters using `cptpOp.to_vector()` and print them:

```{code-cell} ipython3
print("params (%d) = " % cptpOp.num_params, cptpOp.to_vector(),'\n')
```

All 12 parameters are essentially 0 because `gate_mx` represents a (unitary) $X(\pi/2)$ rotation and $U_0$ is automatically set to this unitary so that $\exp\mathcal{L} = \mathbb{1}$.  This means that all the error coefficients are zero, and this translates into all the parameters being zero.  Note, however, that error coefficients are not always the same as parameters. The `errorgen_coefficients` retrieves the error coefficients, which is often more useful than the raw parameter values: 

```{code-cell} ipython3
import pprint
coeff_dict, basis = cptpOp.errorgen_coefficients(return_basis=True)
print("Coefficients in (<type>,<basis_labels>) : value form:"); pprint.pprint(coeff_dict)
print("\nBasis containing elements:"); pprint.pprint(basis.labels)
```

`errorgen_coefficients` returns a dictionary (and a basis if `return_basis=True`).  The dictionary maps a shorthand description of the error term to value of the term's coefficient (rate).  This shorthand description is a tuple starting with `"H"`, `"S"`, or `"A"` to indicate the *type* of error term: Hamiltonian, non-Hamiltonian/stochastic, or affine.  Additional elements in the tuple are basis-element labels (often strings of I, X, Y, and Z), which reference basis matrices in the `Basis` that is returned when `return_basis=True`.  Hamiltonian (`"H"`) errors are described by a single basis element (the single index of $\lambda_i$). The non-Hamiltonian (`"S"`) errors in the Lindblad expansion can be described either by a single basis element (indicating a diagonal element $\eta_{ii}$, also referred to as a *stochastic* error rate) or by two basis elements (the two indices of $\eta_{ij}$).  The affine (`"A"`) errors correspond to particular linear combinations of the non-Hamiltonian errors, and can only be used in conjuction with diagonal non-Hamiltonian errors. 

If we write the above expansion for $\mathcal{L}$ compactly as $\mathcal{L} = \sum_i \lambda_i F_i + \sum_{ij} \eta_{ij}F_{ij}$, then the elements of the error-coefficient dictionary returned from `errorgen_coefficients` correspond to:

- Hamiltonian (e.g. `("H","X")`) elements specify $\lambda_i$, the coefficients of $$F_i : \rho \rightarrow -i[B_i,\rho]$$
- Stochastic, i.e. diagonal non-Hamiltonian, elements (e.g. `("S","X")`) specify $\eta_{ii}$, the coefficients of $$F_{ii} : \rho \rightarrow B_i \rho B_i - \rho$$
- Off-diagonal non-Hamiltonian elements (e.g. `("S","X","Y")`) specify $\eta_{ij}$, the potentially complex coefficients of: $$F_{ij} : \rho \rightarrow B_i \rho B_j^\dagger - \frac{1}{2}\left\{ B_i^\dagger B_j, \rho \right\}$$
- Affine elements (e.g. `("A","X")`) specify the coefficients of: $$A_i : \rho \rightarrow \mathrm{Tr}(\rho_{target})B_i \otimes \rho_{non-target}.$$  Here the *target* vs *non-target* parts of $\rho$ refer to the qubits on $B_i$ is nontrivial and trivial respectively (e.g. in `("A","IXI")` the second qubit is the "target" space and qubits 1 and 3 are the "non-target" space).  This $A_i$ can be written as a linear combination of the $F_{ij}$. Because the map $\rho \rightarrow I$ can be written as $\rho \rightarrow \frac{1}{d^2} \sum_i B_i \rho B_i$, where the sum loops over *all* the Paulis (including the identity), this means that a map like $\rho \rightarrow B_k$ can be written as $\rho \rightarrow \frac{1}{d^2} \sum_i B_i B_k \rho B_i B_k$, which can be expressed as a sum of the $F_{ij}$ terms.

The `set_errorgen_coefficients` function does the reverse of `errorgen_coefficients`: it sets the values of a `LindbladOp`'s parameters based on a description of the errors in terms of the above-described dictionary and a `Basis` object that can resolve the basis labels.

Finally, we can also initialize a `LindbladErrorgen` using a dictionary in this format.  Below we construct a `LindbladErrorgen` with 
$$\mathcal{L} = 0.1 H_X + 0.1 S_X$$
where $H_X: \rho \rightarrow -i[\rho,X]$ and $S_X: \rho \rightarrow X\rho X - \rho$ are Hamiltonian and Pauli-stochastic errors, respectively.  We then use this error generator to create, using a `ExpErrogenOp` and `ComposedOp`, an operator corresponding to $e^{\mathcal{L}}U_0$, where $U_0$ is a $X(\pi/2)$ rotation.

```{code-cell} ipython3
staticOp = ops.StaticStandardOp('Gxpi2')
cptpGen2 = ops.LindbladErrorgen.from_elementary_errorgens({('H','X'): 0.1, ('S','X','X'): 0.1}, state_space=1)
cptpOp2 = ops.ExpErrorgenOp(cptpGen2)
composedOp = ops.ComposedOp([staticOp, cptpOp2])
print(cptpOp2)
pygsti.tools.print_mx(cptpOp2.to_dense())
```

We can check that this operator has the right error generator coefficients.  This time we do things slightly differently by accessing the `errorgen` member of the operator of the `ExpErrorgenOp`:

```{code-cell} ipython3
cptpOp2.errorgen.coefficients() # same as cptpOp2.errorgen_coefficients()
```

An inconvenience arises because an error generator is *exponentiated* to form a map.  This means that the coefficients of the stochastic generators don't exactly correspond to error rates of the final map as they're often thought of.  Take a simple example where we want to construct a depolarizing map with a process fidelity of 90%.  One might think that this would achieve that:

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

But as we see, the fidelity isn't quite $0.9$.  This is because the error rate of the *map* whose error generator has  all stochastic-term coefficients equal to $C$ is $(1 - e^{-d^2 C}) / d^2$, not just $C$.  This transformation is accounted for in the `error_rates` and `set_error_rates` Lindblad operator methods, which behave just like `errorgen_coefficients` and `set_errorgen_coefficients` except that they internally transform the S-values between coefficients and (map) error rates.  Our simple example is fixed by using `set_error_rates`:

```{code-cell} ipython3
test_depol_op.set_error_rates({('S','X'): 0.1/3, ('S','Y'): 0.1/3, ('S','Z'): 0.1/3})
pygsti.tools.entanglement_fidelity(test_depol_op.to_dense(), np.identity(4))
```

```{code-cell} ipython3
# And we can see that the errorgen coefficients have been adjusted accordingly
test_depol_op.errorgen_coefficients()
```

### Lindblad state preparations and POVMs

It is possible to create state preparations and POVMs that use error generators by replacing `ComposedOp` in the construction above with `ComposedState` or `ComposedPOVM`.  These simply compose an operations (e.g. a $\exp\mathcal{L}$ factor from a `ExpErrorgenOp` and `LindbladErrorgen`) with an existing "base" state preparation or POVM.  That is, state preparations are $e^{\mathcal{L}} |\rho_0\rangle\rangle$, where $|\rho_0\rangle\rangle$ represents a "base" pure state, and effect vectors are $\langle\langle E_i | e^{\mathcal{L}}$ where $\langle\langle E_i|$ are the effects of a "base" POVM.  

```{code-cell} ipython3
#Spam vectors and POVM
errorgen = ops.LindbladErrorgen.from_elementary_errorgens({('S','X'): 0.1/3, ('S','Y'): 0.1/3, ('S','Z'): 0.1/3}, state_space=1)
cptpSpamVec = states.ComposedState(staticSV, errorgen) # staticSV is the "base" state preparation
cptpPOVM = povms.ComposedPOVM(ops.ExpErrorgenOp(errorgen)) # by default uses the computational-basis POVM
```

## Embedded and Composed operations

PyGSTi makes it possible to build up "large" (e.g. complex or many-qubit) operators from other "smaller" ones. We have already seen a modest example of this above when an `ExpErrorgenOp` was constructed from a `LindbladErrorgen` object.  Two common and useful actions for building large operators are:
1. **Composition**: A `ComposedOp` composes zero or more other operators, and therefore it's action is the sequential application of each of its *factors*.  The dense representation of a `ComposedOp` is equal to the product (in reversed order!) of the dense representations of its factors.  Note that a `ComposedOp` does not, however, require that its factors have dense representations - they can be *any* `LinearOperator` objects. The dense versions of operators can sometimes result in faster calculations when the system size (qubit number) is small.

2. **Embedding**: An `EmbeddedOp` maps an operator on a subsystem of a state space to the full state space.  For example, it could take a 1-qubit $X(\pi/2)$ rotation and make a 3-qubit operation in which this operation is applied to the 2nd qubit.  Embedded operators are very useful for constructing layer operations in multi-qubit models, where we naturally prefer to work with the lower-dimensional (typically 1- and 2-qubit) operations and need to build up $n$-qubit *layer* operations.

### Composed operations
We'll being by creating an operation that composes several of the dense operations we made earlier:

```{code-cell} ipython3
composedOp = ops.ComposedOp((staticOp,tpOp,fullOp))
print(composedOp)
print("Before interacting w/Model:",composedOp.num_params,"params")
```

This all looks good.  As we expect, there are $0+12+16=28$ parameters (the sum of the parameter-counts of the factors).  

+++

### Embedded operations
Here's how to embed a single-qubit operator (`fullOp`, created above) into a 3-qubit state space, and have `fullOp` act on the second qubit (labelled `"Q1"`).  Note that the parameters of an `EmbeddedOp` are just those of the underlying operator (the one that has been embedded).

```{code-cell} ipython3
embeddedOp = ops.EmbeddedOp(['Q0','Q1','Q2'],['Q1'],fullOp)
print(embeddedOp)
print("Dimension =",embeddedOp.dim, "(%d qubits!)" % (np.log2(embeddedOp.dim)/2))
print("Number of parameters =",embeddedOp.num_params)
```

### Better together
We can design even more complex operations using combinations of composed and embedded objects.  For example, here's a 3-qubit operation that performs three separate 1-qubit operations (`staticOp`, `fullOp`, and `tpOp`) on each of the three qubits.  (These three operations *happen* to all be $X(\pi/2)$ gates because we're lazy and didn't bother to use `gate_mx` values in our examples above, but they *could* be entirely different.)  The resulting `combinedOp` might represent a layer in which all three gates occur simultaneously. 

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

## More coming (soon?) ...
While this tutorial covers the main ones, there're even more model-building-related objects that we haven't had time to cover here.  We plan to update this tutorial, making it more comprehensive, in future versions of pyGSTi.
