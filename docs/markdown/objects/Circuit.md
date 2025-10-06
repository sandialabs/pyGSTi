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

# Circuits
This tutorial will show you how to create and use `Circuit` objects, which represent (suprise, suprise) quantum circuits.  Noteable among their features is the ability to interface pyGSTi with other quantum circuit standards (e.g., conversion to [OpenQasm](https://arxiv.org/abs/1707.03429))

First let's get the usual imports out of the way.

```{code-cell} ipython3
import pygsti
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label as L
```

## Labels
Let's begin by discussing gate and layer labels, which we'll use to build circuits.

### Gate Labels
Gate labels represent a single gate within a circuit, like a CNOT operation between two qubits. A gate label has two parts: a `str`-type name and a tuple of line labels.  Gate names typically begin with 'G' because this is expected when we parse circuits from text files.  The line labels assign the gate to those lines in the circuit.  For example, `"Gx"` or `"Gcnot"` are common gate names, and the integers 0 to $n$ might be the available line labels.  We can make a proper gate label by creating a instance of the `Label` class:

```{code-cell} ipython3
myGateLabel = L('Gcnot',(0,1))
```

But in nearly all scenarios it's also fine to use the Python tuple `('Gcnot',0,1)` as shorthand - this will get converted into the `Label` object above as needed within pyGSTi.

```{code-cell} ipython3
myJustAsGoodGateLabel = ('Gcnot',0,1)
```

As a **special case**, the tuple of line labels can be `None`.  This is interpreted to mean that the gate acts on *all* the available lines.  When just a string is used as a gate label it acts as though it's line labels are `None`.  So these are also valid gate labels:

```{code-cell} ipython3
mySpecialGateLabel = L('Gi')
myJustAsGoodSpecialGateLabel = 'Gi'
```

When dealing with actual `Label` objects you can access the name and line labels of a gate label via the `.name` and `.sslbls` (short for "state space labels", which are the same as line labels as we'll see) members:

```{code-cell} ipython3
print("name = ", myGateLabel.name, " sslbls = ", myGateLabel.sslbls)
print("name = ", mySpecialGateLabel.name, " sslbls = ", mySpecialGateLabel.sslbls)
```

Simple enough; now let's move on to layer labels:

### Layer labels

Layer labels represent an entire layer of a circuit.  A layer label can either be a single gate label or a sequence of gate labels.  In the former case, the layer is interpreted to have just a single gate in it.  In the latter case, all of the gate labels comprising the layer label are interpreted as occurring simultaneously (in parallel) during the given circuit layer.  Again, there's a proper way to make a layer label using a `Label` object, and a number of shorthand ways which are almost always equivalent: 

```{code-cell} ipython3
layerLabel1 = myGateLabel            # single-gate layer using Label object
layerLabel2 = myJustAsGoodGateLabel  # single-gate layer using tuple
layerLabel3 = 'Gi'                   # single-gate layer using a string
layerLabel4 = L( [L('Gx',0), L('Gcnot',(0,1))] ) # multi-gate layer as Label object, from Label objects
layerLabel5 = L( [('Gx',0),('Gcnot',0,1)] )      # multi-gate layer as Label object, from tuple objects
layerLabel6 = L( [('Gx',0),L('Gcnot',(0,1))] )   # multi-gate layer as Label object, from mixed objects
layerLabel7 = [('Gx',0),('Gcnot',0,1)]         # multi-gate layer as a list of tuples
layerLable8 = L( [] )  # *empty* gate layer - useful to mean the identity on all qubits
# etc, etc. -- anything reasonable works like it should
```

Notice that the same `Label` object used for gate labels is used for layer labels.  This is natural when gates and layers are thought of more broadly as "operations" (e.g. a layer of an $n$-qubit circuit is just a $n$-qubit gate).  Thus, you can access the `.name` and `.sslbls` of a layer too (though the name is given the default value "COMPOUND"):

```{code-cell} ipython3
print("name = ", layerLabel5.name, " sslbls = ", layerLabel5.sslbls)
```

A couple tricks:
- when you're not sure whether a layer `Label` object has a multiple gates or is just a single simple gate label, you can iterate over the `.components` member of a `Label`.  This iterates over the gate labels for a multi-gate layer label and just over the label itself for a simple gate label.  For example:

```{code-cell} ipython3
print( list(L([('Gx',0),('Gcnot',0,1)]).components) )
print( list(L('Gx',0).components) )
```

- you can use `lbl.qubits` as an alias for `lbl.sslbls`, and `lbl.num_qubits` instead of `len(lbl.sslbls)`.  These can improve code legibility when dealing a system of qubits (as opposed to qutrits, etc.).  **Beware**: both of these quantities can be `None`, just like `lbl.sslbls`.

```{code-cell} ipython3
lbl = L('Gcnot',(0,1))
print("The label %s applies to %d qubits: %s" % (str(lbl), lbl.num_qubits, str(lbl.qubits)))
lbl = L('Gi')
print("The label %s applies to %s qubits: %s" % (str(lbl), lbl.num_qubits, str(lbl.qubits)))
```

## Circuits

The `Circuit` object encapsulates a quantum circuit as a sequence of *layer labels*, each of which contains zero or more non-identity *gate lables*.  A `Circuit` has some number of labeled *lines* which should have a one-to-one correspondence with the factors $\mathcal{H}_i$ when the quantum-state space is written as a tensor product: $\mathcal{H}_1 \otimes \mathcal{H}_2 \cdots \otimes \mathcal{H}_n$.  Line labels can be integers or strings (in the above examples we used the integers 0 and 1). 

### Construction
We initialize a `Circuit` with a sequence of *layer labels*, and either:
- a sequence of line labels, as `line_labels`, or
- the number of lines for the circuit, as `num_lines`, in which case the line labels are taken to be integers starting at 0.

```{code-cell} ipython3
c = Circuit( [('Gx',0),('Gcnot',0,1),(),'Gall',('Gy',3)], line_labels=[0,1,2,3])
c2 = Circuit( [('Gx',0),('Gcnot',0,1),(),'Gall',('Gy',3)], num_lines=4) # equivalent to above
c3 = Circuit( [('Gx',0),('Gcnot',0,1),(),'Gall',('Gy',3)] ) # NOT equivalent, because line 2 is never used (see below)

print(c)   # You can print circuits to get a
print(c2)  #  text-art version.
print(c3)
```

In the **case of 1 or 2 qubits**, it can be more convenient to dispense with the line labels entirely and just equate gates with circuit layers and represent them with simple Python strings.  If we initialize a `Circuit` without specifying the line labels (either by `line_labels` or by `num_lines`) *and* the layer labels don't contain any non-`None` line labels, then a `Circuit` is created which has a single special **'\*'-line** which indicates that this circuit doesn't contain any explicit lines:  

```{code-cell} ipython3
c4 = Circuit( ('Gx','Gy','Gi') )
print(c4)
```

Using circuits with '\*'-lines is fine (and is the default type of circuit for the "standard" 1- and 2-qubit modules located at `pygsti.construction.std*`); one just needs to be careful not to combine these circuits with other non-'\*'-line circuits.

**1Q Note:** Particularly within the 1-qubit context the **ordering direction** is important.  The elements of a `Circuit` are read from **left-to-right**, meaning the first (left-most) layer is performed first.  This is very natural for experiments since one can read the operation sequence as a script, executing each gate as one reads from left to right.  It's also natural for 2+ qubit circuits which similar to standard quantum circuit diagrams.  However, for 1-qubit circuits, since we insist on "normal" matrix multiplication conventions, the fact that the ordering of matrix products is *reversed* from that of operation sequences may be confusing.  For example, the circuit `('Ga','Gb','Gc')`, in which Ga is performed first, corresponds to the matrix product $G_c G_b G_a$.  The probability of this operation sequence for a SPAM label associated with the (column) vectors ($\rho_0$,$E_0$) is given by $E_0^T G_c G_b G_a \rho_0$, which can be interpreted as "prepare state 0 first, then apply gate A, then B, then C, and finally measure effect 0".  While this nuance is typically hidden from the user (the `Model` functions which compute products and probabilities from `Circuit` objects perform the order reversal internally), it becomes very important if you plan to perform matrix products by hand. 

### Implied SPAM
A `Circuit` may optionally begin with an explicit state-preparation and end with an explicit measurement, but these may be omitted when used with `Model` objects which have only a single state-preparation and/or POVM.  Usually state preparation and measurement operations are represented by `"rho"`- and `"M"`-prefixed `str`-type labels that therefore act on all circuit lines.  For example:

```{code-cell} ipython3
c5 = Circuit( ['rho010',('Gz',1),[('Gswap',0,1),('Gy',2)],'Mx'] , line_labels=[0,1,2])
print(c5)
```

### Basic member access
The basic member variables and functions of a `Circuit` that you may be interested in are demonstrated below: 

```{code-cell} ipython3
print("depth = ", c.depth)    # circuit depth, i.e., the number of layers
print("tup = ", c.tup)          # circuit as a tuple of layer-labels (elements are *always* Label objects)
print("str = ", c.str)          # circuit as a single-line string
print("lines = ",c.line_labels) # tuple of line labels
print("#lines = ",c.num_lines) #number of line labels
print("#multi-qubit gates = ", c.num_multiq_gates)
c_copy = c.copy()               #copies the circuit
```

### Indexing and Slicing
When a `Circuit` is indexed as if it were a tuple of layer-labels.  The index must be an integer and a `Label` object is returned.  Once can also access a particular gate label by providing a second index which is either a single line label, a tuple of line labels, or, *if the line labels are integers*, a slice of line labels:

```{code-cell} ipython3
c = Circuit( [('Gx',0),[('Gcnot',0,1),('Gz',3)],(),'Gall',('Gy',3)], line_labels=[0,1,2,3])
print(c)
print('c[1] = ',c[1])       # layer
print('c[0,0] = ',c[0,0])   # gate at layer=0, line=0 (Gx)
print('c[0,2] = ',c[0,2])   # gate at layer=0, line=2 (nothing)
print('c[1,0] = ',c[1,0])   # gate at layer=0, line=0 (NOTE: nothing because CNOT doesn't *only* occupy line 0)
print('c[1,(0,1)] = ',c[1,(0,1)]) # gate at layer=0, lines=0&1 (Gcnot)
print('c[1,(0,1,3)] = ', c[1,(0,1,3)]) # layer-label restricted to lines 0,1,&3
print('c[1,0:3] = ', c[1,0:3]) # layer-label restricted to lines 0,1,&2 (line-label slices OK b/c ints)
print('c[3,0] = ',c[3,0])
print('c[3,:] = ',c[3,:]) # DEBUG!
```

If the first index is a tuple or slice of layer indices, a `Circuit` is returned which contains only the indexed layers.  This indexing may be combined with the line-label indexing described above.  Here are some examples:

```{code-cell} ipython3
print(c)
print('c[1:3] = ');print(c[1:3])      # Circuit formed from layers 1 & 2 of original circuit
print('c[2:3] = ');print(c[1:2])      # Layer 1 but as a circuit (not the same as c[1], which is a Label)
print('c[0:2,(0,1)] = ');print(c[0:2,(0,1)]) # upper left "box" of circuit
print('c[(0,3,4),(0,3)] = ');print(c[(0,3,4),(0,3)]) #Note: gates only partially in the selected "box" are omitted
```

### Editing Circuits
**Circuits are by default created as read-only objects**.  This is because making them read-only allows them to be hashed (e.g. used as the keys of a dictionary) and there are many tasks that don't require them being editable.  That said, it's easy to get an editable `Circuit`: just create one or make a copy of one with `editable=True`:

```{code-cell} ipython3
ecircuit1 = Circuit( [('Gx',0),('Gcnot',0,1),(),'Gall',('Gy',3)], num_lines=4, editable=True)
ecircuit2 = c.copy(editable=True)
print(ecircuit1)
```

When a circuit is editable, you can perform additional operations that alter the circuit in place (see below).  When you're done, call `.done_editing()` to change the `Circuit` into read-only mode.  Once in read-only mode, a `Circuit` cannot be changed back into editable-mode, you must make an editable *copy* of the circuit.  

As you may have guessed, you're allowed to *assign* the layers or labels of an editable circuit by indexing:

```{code-cell} ipython3
ecircuit1[0,(2,3)] = ('Gcnot',2,3)
print(ecircuit1)

ecircuit1[2,1] = 'Gz' # interpreted as ('Gz',1) 
print(ecircuit1)

ecircuit1[2:4] = [[('Gx',1),('Gcnot',3,2)],('Gy',1)] #assigns to layers 2 & 3
print(ecircuit1)
```

There are also methods for inserting and removing lines and layers:

```{code-cell} ipython3
ecircuit1.append_circuit( Circuit([('Gx',0),'Gi'], num_lines=4) )
print(ecircuit1)

ecircuit1.insert_circuit( Circuit([('Gx',0),('Gx',1),('Gx',2),('Gx',3)], num_lines=4), 1)
print(ecircuit1)

ecircuit1.insert_layer( L( (L('Gz',0),L('Gz',3)) ), 0) #expects something like a *label*
print(ecircuit1)

ecircuit1.delete_layers([2,3])
print(ecircuit1)

ecircuit1.insert_idling_lines(2, ['N1','N2'])
print(ecircuit1)

ecircuit1.delete_lines(['N1','N2'])
print(ecircuit1)
```

Finally, there are more complex methods which do fancy things to `Circuit`s:

```{code-cell} ipython3
ecircuit1.compress_depth_inplace()
print(ecircuit1)
```

```{code-cell} ipython3
ecircuit1.change_gate_library({('Gx',0) : [('Gx2',0)]}, allow_unchanged_gates=True)
print(ecircuit1)
```

```{code-cell} ipython3
ecircuit1.done_editing()
```

### Circuits as tuples
In many ways `Circuit` objects behave as a tuple of layer labels.  We've already shown how indexing and slicing mimic this behavior.  You can also add circuits together and multiply them by integers:

```{code-cell} ipython3
c2 = Circuit([('Gx',0),('Gx',1),('Gx',2),('Gx',3)], num_lines=4)
print(c)
print(c+c2)
print(c*2)
```

There are also methods to "parallelize" and "serialize" circuits, which are available to read-only circuits too because they return new `Circuit` objects and don't modify anything in place:

```{code-cell} ipython3
c2 = Circuit([[('Gx',0),('Gx',1)],('Gx',2),('Gx',3)], num_lines=4)
print(c2)
print(c2.parallelize())
print(c2.serialize())
```

### String representations
`Circuit` objects carry along with them a string representation, accessible via the `.str` member.  This is intended to hold a compact human-readable expression for the circuit that can be parsed, using pyGSTi's standard circuit format and conventions, to reconstruct the circuit.  This isn't quite true because the line-labels are not currently contained in the string representation, but this will likely change in future releases.

Here's how you can construct a `Circuit` with or from a string representation which thereafter illustrates how you can print different representation of a `Circuit`.  Note that two `Circuits` may be equal even if their string representations are different.

```{code-cell} ipython3
#Construction of a Circuit
c1 = Circuit( ('Gx','Gx') ) # from a tuple
c2 = Circuit( ('Gx','Gx'), stringrep="Gx^2" ) # from tuple and string representations (must match!)
c3 = Circuit( "Gx^2" ) # from just a string representation

#All of these are equivalent (even though their string representations aren't -- only tuples are compared)
assert(c1 == c2 == c3)

#Printing displays the Circuit representation
print("Printing as string (multi-line string rep)")
print("c1 = %s" % c1)
print("c2 = %s" % c2)
print("c3 = %s" % c3, end='\n\n')

#Printing displays the Circuit representation
print("Printing .str (single-line string rep)")
print("c1 = %s" % c1.str)
print("c2 = %s" % c2.str)
print("c3 = %s" % c3.str, end='\n\n')

#Casting to tuple displays the tuple representation
print("Printing tuple(.) (tuple rep)")
print("c1 =", tuple(c1))
print("c2 =", tuple(c2))
print("c3 =", tuple(c3), end='\n\n')

#Operations
assert(c1 == ('Gx','Gx')) #can compare with tuples
c4 = c1+c2 #addition (note this concatenates string reps)
c5 = c1*3  #integer-multplication (note this exponentiates in string rep)
print("c1 + c2 = ",c4.str, ", tuple = ", tuple(c4))
print("c1*3    = ",c5.str, ", tuple = ", tuple(c5), end='\n\n')
```

### Barriers and compilable regions
Each layer in a `Circuit` is marked as to whether it can be compiled with the layer following it.  The default is that a layer is *not* able to be compiled, effectively placing compiler *barriers* after each layer.  By toggling one or more layers to being compilable, users can mark regions of a circuit (or the entire circuit) as fair game to be modified by a compiler program before being run on hardware.

Currently, tracking compilable layers is not used much by other pyGSTi routines - it's just bookkeeping metadata that is stored with a circuit for convenience.

When building a circuit, the `compilable_layer_indices` specifies which layers can be compiled with their subsequent layer.

```{code-cell} ipython3
c = Circuit(('Gx', 'Gx', 'Gy', 'Gy'), compilable_layer_indices=(1,2) )
```

This information can be retrieved directly throught a circuit's `compilable_layer_indices` attribute, or a boolean mask for whether each layer is compilable is given by the `compilable_by_layer` attribute:

```{code-cell} ipython3
print("c.compilable_layer_indices = ", c.compilable_layer_indices)
print("c.compilable_by_layer = ", c.compilable_by_layer)
```

In the string representation of circuits, there are two mutually exclusive ways to indicate which layers are "compilable".

1. the tilde (`~`) character can be used between layers to indicate that compilation is allowed between the adjacent layers.  Layers not joined by a tilde are interpreted as *not* compilable. 
2. the pipe (`|`) character can be used between layers to indicate that compilation is forbidden between the adjacent layers.  Layers not joined by a pipe are interpreted as *compilable*.

Using both pipes and tildes in a circuit string is invalid.  If neither is used, case 1 is the default and so all circuit layers are *not* compilable.  The examples below illustrate this.  Note that printing the "fancy"/ascii-art version of the circuit does not indicate which indices are compilable - the tildes and pipes are only implemented in the single-line string representations of the circuit so far.

```{code-cell} ipython3
print(c.str)  # shows that layers 1 and 2 are compilable with the next layer
```

```{code-cell} ipython3
print(c)  # ascii-art does NOT indicate compilable layers
```

```{code-cell} ipython3
c = Circuit("Gx~GxGx~Gx")
c.compilable_layer_indices
```

```{code-cell} ipython3
c2 = Circuit("GxGx|GxGx|")
c2.compilable_layer_indices
```

```{code-cell} ipython3
c == c2
```

### File I/O
Circuits can be saved to and read from their single-line string format, which uses square brackets to enclose each layer of the circuit. See the lines of [MyCircuits.txt](../tutorial_files/MyCircuits.txt), which we read in below, for examples.  Note that a `Circuit`'s line labels are not included in their single-line-string format, and so to reliably import circuits the line labels should be supplied separately to the `read_circuit_list` function: 

```{code-cell} ipython3
circuitList = pygsti.io.read_circuit_list("../tutorial_files/MyCircuits.txt", line_labels=[0,1,2,3,4])
for c in circuitList:
    print(c)
```

### Converting circuits external formats

`Circuit` objects can be easily converted to [OpenQasm](https://arxiv.org/abs/1707.03429) or [Quil](https://arxiv.org/pdf/1608.03355.pdf) strings, using the `convert_to_openqasm()` and `convert_to_quil()` methods. This conversion is automatic for circuits that containing only gates with name that are in-built into `pyGSTi` (the docstring of `pygsti.tools.internalgates.standard_gatename_unitaries()`). This is with some exceptions in the case of Quil: currently not all of the in-built gate names can be converted to quil gate names automatically, but this will be fixed in the future. 

For other gate names (or even more crucially, if you have re-purposed any of the gate names that `pyGSTi` knows for a different unitary), the desired gate name conversation must be specified as an optional argument for both `convert_to_openqasm()` and `convert_to_quil()`. 

Circuits with line labels that are *integers* or of the form 'Q*integer*' are auto-converted to the corresponding integer. If either of these labelling conventions is used but the mapping should be different, or if the qubit labelling in the circuit is not of one of these two forms, the mapping should be handed to these conversion methods.

```{code-cell} ipython3
label_lst = [ [L('Gh','Q0'),L('Gh','Q1')], L('Gcphase',('Q0','Q1')), [L('Gh','Q0'),L('Gh','Q1')]]
c = Circuit(label_lst, line_labels=['Q0','Q1'])

print(c)
openqasm = c.convert_to_openqasm()
print(openqasm)
```

```{code-cell} ipython3
label_lst = [L('Gxpi2','Q0'),L('Gcnot',('Q0','Q1')),L('Gypi2','Q1')]
c2 = Circuit(label_lst, line_labels=['Q0','Q1'])
quil = c2.convert_to_quil()
print(quil)
```

### Simulating circuits
`Model` objects in pyGSTi are able to *simulate*, or "generate the outcome probabilities for", circuits.  To demonstrate, let's create a circuit and a model (see the tutorials on ["explicit" models](ExplicitModel.ipynb) and ["implicit" models](ImplicitModel.ipynb) for more information on model creation):

```{code-cell} ipython3
clifford_circuit = Circuit([ [L('Gh','Q0'),L('Gh','Q1')],
                              L('Gcphase',('Q0','Q1')),
                             [L('Gh','Q0'),L('Gh','Q1')]],
                            line_labels=['Q0','Q1'])
pspec = pygsti.processors.QubitProcessorSpec(2, ['Gh', 'Gcphase'], geometry='line', qubit_labels=['Q0', 'Q1'])
model = pygsti.models.create_crosstalk_free_model(pspec)
```

Then circuit outcome probabilities can be computed using either the `model.probabilities(circuit)` or `circuit.simulate(model)`, whichever is more convenient: 

```{code-cell} ipython3
out1 = model.probabilities(clifford_circuit)
out2 = clifford_circuit.simulate(model)
```

The output is simply a dictionary of outcome probabilities:

```{code-cell} ipython3
out1
```

The keys of the outcome dictionary `out` are things like `('00',)` instead of just `'00'` because of possible *intermediate* outcomes.  See the [Instruments tutorial](advanced/Instruments.ipynb) if you're interested in learning more about intermediate outcomes.

Computation of outcome probabilities may be done in a variety of ways, and `Model` objects are associated with a *forward simulator* that supplies the core computational routines for generating outcome probabilities.  In the example above the simulation was performed by multiplying together process matrices.  For more information on the types of forward simulators in pyGSTi and how to use them, see the [forward simulators tutorial](../algorithms/advanced/ForwardSimulationTypes.ipynb).

+++

## Conclusion
This concludes our detailed look into the `Circuit` object.  If you're intersted in using circuits for specific applications, you might want to check out the [tutorial on circuit lists](advanced/CircuitLists.ipynb) or the [tutorial on constructing GST circuits](advanced/GSTCircuitConstruction.ipynb)
