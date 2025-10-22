---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# State Space Labeling

Instances of `pygsti.baseobjs.StateSpace` describe the structure of a model's state space and associate labels with the parts of that structure.  This is particularly useful when dealing with multiple qubits or a qubit and its environment, as it can be useful to reference subspaces or subsystems of the entire quantum state space.

In general, a state space is the direct sum of one or more *tensor product blocks*, each of which is the tensor product of one or more *factors*: 

$$ \mbox{State space} = (\mathcal{H}_1^A \otimes \mathcal{H}_2^A \otimes \cdots) \oplus (\mathcal{H}_1^B \otimes \mathcal{H}_2^B \otimes \cdots) \oplus \cdots$$

In the above expression the tensor product blocks are in parenthesis and labelled by $A$, $B$, etc., and the $\mathcal{H}_i^X$ are the factors.  We can initialize a `StateSpace` object using a list of tuples containing labels and dimensions which mirror this structure, i.e.

~~~
StateSpace( [(H1A_label, H2A_label, ...), ((H1B_label, H2B_label, ...), ... ],
                  [(H1A_dim  , H2A_dim, ...),   ((H1B_dim  , H2B_dim, ...), ... ])
~~~

There are currently two main types of `StateSpace` objects: `ExplicitStateSpace`, where the labels and dimensions must be explicitly defined, and `QubitStateSpace`, which is a product of Hilbert spaces with dimension 2 (qubits).

```{code-cell} ipython3
import pygsti
from pygsti.baseobjs import ExplicitStateSpace

lbls = ExplicitStateSpace([('H0','H1')], [(2,3)])
print(lbls) # label(dim) notation, '*' means 'otimes', '+' means 'oplus'

lbls2 = ExplicitStateSpace(('H0','H1'), (2,3)) # same as above - a *single* tensor product block
print(lbls2)

lbls3 = ExplicitStateSpace([('H0',), ('H1',)], [(2,),(3,)]) # direct sum
print(lbls3)

lbls4 = ExplicitStateSpace([('H1a','H2a'), ('H1b','H2b')], [(2,1),(3,4)])
print(lbls4)
```

Since we're often dealing with qubits (dimension = 2 factors), the labels beginning with 'Q' or that are integers default to dimension 2.  Similarly, labels beginning with 'L' default dimension 1 (an additional "Level").  If all the labels in the first argument passed to the `StateSpaceLabels` constructor have defaults, then the **second argument (the dimensions) may be omitted**.  For example:

```{code-cell} ipython3
from pygsti.baseobjs import QubitSpace

lbls5 = QubitSpace(['Q0','Q1']) # 2 qubits
print(lbls5)

lbls6 = QubitSpace(3) # 3 qubits
print(lbls6)
```

Rather than explicitly constructing either one of the `StateSpace`-derived classes, you can also use the `cast` class method to automatically generate the correct type, based on the provided labels. The labels must start with a tag that indicates how many levels/the dimension of that space. Allowed values include:

- `"Q"`: qubit (dimension 2)
- `"T"`: qutrit (dimension 3)
- `"L"`: single level (dimension 1)
- `"C"`: classical bit (for use with instruments)
- `int`: Defines a tensor product block with that many qubits

```{code-cell} ipython3
from pygsti.baseobjs import StateSpace

# Defines a "kite" with 2 tensor product blocks - one with 2 qubits and then a single leakage level
lbls7 = StateSpace.cast([('Q0','Q1'),('Leakage',)])
print(lbls7)
```

Internally, the data in a `StateSpace` is kept in terms of the tensor product blocks. The accessors for `labels` and `dimensions` must be done on a tensor product block basis:

```{code-cell} ipython3
print("Number of tensor product blocks = ",lbls7.num_tensor_product_blocks)
print("The labels in the 0th tensor product block are: ",lbls7.tensor_product_block_labels(0))
print("The dimensions corresponding to those labels are: ",lbls7.tensor_product_block_dimensions(0))
print("The 'Q0' labels exists in the tensor product block w/index=",lbls7.label_tensor_product_block_index('Q0'))
```

**That's it!** You know all there is to know about the `StateSpace` object.  Remember you can pass a `StateSpace` object to `pygsti.models.create_explicit_model_from_expressions` to create a model which operates on the given state space.


