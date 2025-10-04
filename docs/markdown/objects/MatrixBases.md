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

# Matrix Basis Tutorial

Consider the space of density matrices corresponding to a Hilbert space $\mathcal{H}$ of dimension $d$.  The basis used for this Hilbert-Schmidt space, $B(\mathcal{H})$, can be any set of $d\times d$ matrices which span the density matrix space.  pyGSTi supports arbitrary bases by deriving from the `pygsti.tools.Basis` class, and constains built-in support for the following basis sets:

- the matrix unit, or "standard" basis, consisting of the matrices with a single unit (1.0) element and otherwise zero.  This basis is selected by passing `"std"` to appropriate function arguments.
- the Pauli-product basis, consisting of tensor products of the four Pauli matrices {I, X, Y, Z} normalized so that $Tr(B_i B_j) = \delta_{ij}$.  All of these matrices are Hermitian, so that Hilbert-Schmidt vectors and matrices are real when this basis is used. This basis can only be used when the $d = 4^i$ for integer $i$, and is selected using the string `"pp"`.
- the Gell-Mann basis, consisting of the normalized Gell-Mann matrices (see Wikipedia if you don't know what these are).  Similar to the Pauli-product case, these matrices are also Hermitian, so that Hilbert-Schmidt vectors and matrices are real when this basis is used.  Unlike the Pauli-product case, since Gell-Mann matrices are well defined in any dimension, the Gell-Mann basis is *not* restricted to cases when $d=4^i$.  This basis is selected using the string `"gm"`.
- a special basis of $3 \times 3$ matricies designed for Qutrit systems formed by taking the symmetric subspace of a 2-qubit system.  This basis is selected using the string `"qt"`. 

Various functions and objects within pyGSTi require knowledge of what Hilbert-Schmidt basis is being used.  The `pygsti.objects.Basis` object encapsulates a basis, and is the most flexible way of specifying a basis in pyGSTi.  Alternatively, many functions also accept the short strings `"std"`, `"gm"`, `"pp"`, and `"qt"` to select one of the standard bases.  In this tutorial, we'll demonstrate how to create a `Basis` object and use it and related functions to obtain and change the basis of the operation matrices and SPAM vectors stored in a `Model`.

The most straightforward way to create a `Basis` object is to provide its short name and dimension to the `Basis.cast` function, which "casts" various things as a basis object.  PyGSTi contains built-in support for bases consisting of the tensor product of Pauli matrices (or just the Pauli matrices in the case of 1 qubit), named `"pp"`, as well as the Gell-Mann matrices, named `"gm"`.  It also contains a special "qutrit" basis, named `"qt"`, for the case of 3-level quantum systems.  In cases when there are an integral number of qubits, and the dimension equals $4^N$, the `"pp"` basis is usually preferred since it is more intuitive.  In other cases, where the Hilbert space includes non-qubit (e.g. environmental) degrees of freedom, the Gell-Mann basis may be useful since it can be used in any dimension.  Note that both the Gell-Mann and Pauli-Product bases reduce to the usual Pauli matrices plus identity in when the dimension equals 4 (1 qubit).

Here are some examples:

```{code-cell} ipython3
import pygsti
from pygsti.baseobjs import Basis
pp  = Basis.cast('pp',  4) # Pauli-product (in this dim=4 case, just the Paulis)
std = Basis.cast('std', 4) # "standard" basis of matrix units
gm  = Basis.cast('gm',  4) # Gell-Mann
qt  = Basis.cast('qt',  9) # qutrit - must be dim 9
bases = [pp, std, gm, qt]
```

Each of the `pp`, `std`, and `gm` bases created will have $4$ $2x2$ matrices each. The `qt` basis has $9$ $3x3$ matrices instead:

```{code-cell} ipython3
for basis in bases:
    print('\n{} basis (dim {}):'.format(basis.name, basis.dim))
    print('{} elements:'.format(len(basis)))
    for element in basis:
        pygsti.tools.print_mx(element)
```

However, custom "explicit" bases, which expicitly hold a set of basis-element matrices, can be easily created by supplying a list of the elements:

```{code-cell} ipython3
import numpy as np
from pygsti.baseobjs import ExplicitBasis
std2x2Matrices = [
        np.array([[1, 0],
                  [0, 0]]),
        np.array([[0, 1],
                  [0, 0]]),
        np.array([[0, 0],
                  [1, 0]]),
        np.array([[0, 0],
                  [0, 1]])]

alt_standard = ExplicitBasis(std2x2Matrices, ["myElement%d" % i for i in range(4)],
                     name='std', longname='Standard')
print(alt_standard)
```

More complex bases can be created by chaining the elements of other bases together along the diagonal.  This yields a basis for the direct-sum of the spaces spanned by the original basis. For example, a composition of the $2x2$ `std` basis with the $1x1$ `std` basis leads to a basis with state vectors of length $5$, or $5x5$ matrices:

```{code-cell} ipython3
from pygsti.baseobjs import DirectSumBasis
comp = Basis.cast('std', [4, 1])
comp = Basis.cast([('std', 4), ('std', 1)])
comp = DirectSumBasis([ Basis.cast('std', 4), Basis.cast('std', 1)])

#All three comps above give the same final basis
print(comp)
for element in comp.elements:
    print(element)
```

### Basis usage
Once created, bases are used to manipulate matrices and vectors within pygsti:

```{code-cell} ipython3
from pygsti.tools import change_basis, flexible_change_basis

mx = np.array([[1, 0, 0, 1],
               [0, 1, 2, 0],
               [0, 2, 1, 0],
               [1, 0, 0, 1]])

change_basis(mx, 'std', 'gm') # shortname lookup
change_basis(mx, std, gm)     # object only
change_basis(mx, std, 'gm')   # combination
```

Composite bases can be converted between expanded and contracted forms:

```{code-cell} ipython3
mxInStdBasis = np.array([[1,0,0,2],
                         [0,0,0,0],
                         [0,0,0,0],
                         [3,0,0,4]],'d')

begin = Basis.cast('std', 4)
end   = Basis.cast('std', [1,1])
mxInReducedBasis = flexible_change_basis(mxInStdBasis, begin, end)
print(mxInReducedBasis)
original         = flexible_change_basis(mxInReducedBasis, end, begin)
```

```{code-cell} ipython3

```
