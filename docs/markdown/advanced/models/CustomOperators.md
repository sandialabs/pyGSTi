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

# Custom operators and factories

A gate (or layer) operation in pyGSTi is just a parameterized process matrix: a mapping that sends a vector of parameter values to a process matrix. That mapping lives in a `LinearOperator`-derived class. pyGSTi ships a lot of these (`FullArbitraryOp` and friends, catalogued in the [operators tutorial](Operators)), but nothing stops you from writing your own, and there are good reasons to.

The usual reason is physics. You have a specific model of how a gate misbehaves on your device, and none of the generic built-in classes captures it. A close second is instrumentation: you want the operator's parameters to be exactly the knobs you can turn in the lab, so that a GST fit hands back numbers you can act on.

This page walks three levels of customization, in increasing order of machinery:

- A custom **dense operator**, where you write `from_vector` yourself and pyGSTi treats the result like any other gate.
- An **operation factory**, which builds an operator on demand from arguments carried by the circuit label. This is how continuously parameterized gates work.
- An **interpolated** operator or factory, for when the process matrix comes from a physics simulation too slow to evaluate inside an optimizer loop.

## Setup

```{code-cell} ipython3
import numpy as np
from scipy.linalg import expm

import pygsti
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.modelmembers import operations as op
import pygsti.extras.interpygate as interp

try:
    # See if MPI is available.
    #   For robustness, force the Libfabric/OFI provider to fallback to standard `sockets`.
    #   This prevents a hard C-level process abort on some platforms. One should remove
    #   this environment-variable assignment unless actually needed on your machine.
    import os
    os.environ['FI_PROVIDER'] = 'sockets'
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except (ImportError, RuntimeError): # no mpi module will result in a Runtime error and not an ImportError
    comm = None
```

The MPI communicator is only used by the interpolation section at the end of this page. Physical-process simulation is the one part of this workflow that routinely wants more than one core.

## Writing a custom dense operator

The example here is a 1-qubit $X(\pi/2)$ rotation that can suffer depolarization and "on-axis" overrotation, and nothing else. Two imperfections, so two parameters.

Subclass `DenseOperator` and implement three things: `num_params`, `to_vector`, and `from_vector`. The comments explain the rest.

```{code-cell} ipython3
class MyXPi2Operator(op.DenseOperator):
    def __init__(self):
        #initialize with no noise
        super(MyXPi2Operator,self).__init__(np.identity(4,'d'), 'pp', "densitymx") # this is *super*-operator, so "densitymx"
        self.from_vector([0,0]) 
    
    @property
    def num_params(self): 
        return 2 # we have two parameters
    
    def to_vector(self):
        return np.array([self.depol_amt, self.over_rotation],'d') #our parameter vector
        
    def from_vector(self, v, close=False, dirty_value=True):
        #initialize from parameter vector v
        self.depol_amt = v[0]
        self.over_rotation = v[1]
        
        theta = (np.pi/2 + self.over_rotation)/2
        a = 1.0-self.depol_amt
        b = a*np.sin(2*theta)
        c = a*np.cos(2*theta)
        
        # ._ptr is a member of DenseOperator and is a numpy array that is 
        # the dense Pauli transfer matrix of this operator
        # Technical note: use [:,:] instead of direct assignment so id of self._ptr doesn't change
        self._ptr[:,:] = np.array([[1,   0,   0,   0],
                                  [0,   a,   0,   0],
                                  [0,   0,   c,  -b],
                                  [0,   0,   b,   c]],'d')
        self.dirty = dirty_value  # mark that parameter vector may have changed
        
    def transform(self, S):
        # Update self with inverse(S) * self * S (used in gauge optimization)
        raise NotImplementedError("MyXPi2Operator cannot be transformed!")
```

Drop an instance in as the `("Gxpi2",0)` gate of pyGSTi's {Idle, $X(\pi/2)$, $Y(\pi/2)$} modelpack (see the [modelpacks tutorial](../../guides/workflow/TargetModels) for what modelpacks are).

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
---
from pygsti.modelpacks import smq1Q_XYI
mdl = smq1Q_XYI.target_model()
mdl.operations[('Gxpi2',0)] = MyXPi2Operator()
print(mdl)
```

### Running GST with it

Nothing about the custom class needs special handling downstream, so run gate set tomography on the model directly (the [GST overview tutorial](../../start/FirstGST) covers what each of these calls does).

There is one wrinkle. GST by default gauge optimizes its final estimate toward the target model, and that requires every operator in the model to implement `transform`. `MyXPi2Operator` deliberately doesn't, so turn gauge optimization off with `gauge_opt_suite_name='none'`. See the [gauge optimization tutorial](../../guides/analysis/GaugeFreedom) for why you'd normally want it.

`gauge_opt_suite_name='none'` is not the same as `gauge_opt_suite_name=None`. The string explicitly skips gauge optimization; `None` defers the decision to downstream functions.

```{code-cell} ipython3
# Generate "fake" data from a depolarized version of the target (ideal) model
maxLengths = [1,2,4,8,16]
mdl_datagen = smq1Q_XYI.target_model().depolarize(op_noise=0.01, spam_noise=0.001)
listOfExperiments = pygsti.circuits.create_lsgst_circuits(
    mdl_datagen, smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(), smq1Q_XYI.germs(), maxLengths)
ds = pygsti.data.simulate_data(mdl_datagen, listOfExperiments, num_samples=1000,
                                            sample_error="binomial", seed=1234)

#Run GST *without* gauge optimization
results = pygsti.run_long_sequence_gst(ds, mdl, smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials(),
                                      smq1Q_XYI.germs(), maxLengths, gauge_opt_suite_name='none')
```

That's it. GST just ran with a custom operation.

The model fits the data well; compare the actual and expected $2\Delta \log \mathcal{L}$ values printed above. That should not surprise you, because the data came from a model with only depolarizing gate errors, and `MyXPi2Operator` can represent exactly that. The generating depolarization was $0.01$ with zero overrotation, so those are the numbers to look for:

```{code-cell} ipython3
mdl_estimate = results.estimates['GateSetTomography'].models['final iteration estimate']
print(mdl_estimate[('Gxpi2',0)])
est_depol, est_overrotation = mdl_estimate[('Gxpi2',0)].to_vector()
print("Estimated Gx depolarization =",est_depol)
print("Estimated Gx over-rotation =",est_overrotation)
```

They don't come out exactly right. Finite sampling accounts for most of the discrepancy, and residual gauge freedom for the rest.

## Gate labels with arguments

Some gates are not one gate. A continuously parameterized $X(\theta)$ rotation is a whole family, and which member you get depends on the circuit, not on a fit. pyGSTi handles this by letting gate labels carry **arguments**.

Arguments are tags attached to a gate label. They can be continuous values such as a rotation angle. They are held separately from the label's *state space labels* (usually qubit labels), which say which qubits the gate targets and therefore which lines it is drawn on in a circuit diagram.

Arguments are written after a semicolon. Several spellings produce the same label:

```{code-cell} ipython3
#Different ways of creating a gate label that contains a single argument
l = Label('Ga',args=(1.4,1.2))
l2 = Label(('Ga',';1.4',';1.2')) #Note: in this case the arguments are *strings*, not floats
l3 = Label(('Ga',';',1.4,';',1.2))
```

The compact semicolon notation also works when you build `Circuit`s from tuples or strings:

```{code-cell} ipython3
# standard 1Q circuit, just for reference
c = Circuit( ('Gx','Gy') )
print(c)

# 1Q circuit with explicit qubit label
c = Circuit( [('Gx',0),('Gy',0)] )
print(c)

# adding arguments
c = Circuit( [('Gx',0,';1.4'),('Gy',';1.2',0)] )
print(c)

#Or like this:
c = Circuit("Gx;1.4:0*Gy;1.2:0")
print(c)
```

## Operation factories

A gate label without arguments resolves to an operator object. A label *with* arguments resolves to a factory. A factory builds operator objects on demand from the arguments it finds in the circuit label. The method you implement is `create_object`, which receives a tuple as `args` and returns a gate object.

Keep arguments and parameters straight. Parameters are what GST twiddles during a fit. Arguments are fixed by the circuit. The factory below returns a `StaticArbitraryOp`, meaning zero parameters, because how the produced gate is parameterized is a separate question from how it is built.

```{code-cell} ipython3
class XRotationOpFactory(op.OpFactory):
    def __init__(self):
        op.OpFactory.__init__(self, state_space=1, evotype="densitymx")
        
    def create_object(self, args=None, sslbls=None):
        # Note: don't worry about sslbls (unused) -- this argument allow factories to create different operations on different target qubits
        assert(len(args) == 1)
        theta = float(args[0])/2.0  #note we convert to float b/c the args can be strings depending on how the circuit is specified
        b = 2*np.cos(theta)*np.sin(theta)
        c = np.cos(theta)**2 - np.sin(theta)**2
        superop = np.array([[1,   0,   0,   0],
                            [0,   1,   0,   0],
                            [0,   0,   c,  -b],
                            [0,   0,   b,   c]],'d')
        return op.StaticArbitraryOp(superop)
```

Models store factories in a `factories` dictionary. Implicit models key it by layer block, so a `LocalNoiseModel` wants `factories['layers']`; explicit models use a flat `factories` dict. Build a crosstalk-free (local noise) model for one qubit with the standard X and Y gates, then attach the factory under the label `"Ga"`. See the [implicit model tutorial](../../guides/models/MultiQubitModels) for more on this model type.

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(1, ['Gx', 'Gy'])
mdl_1q = pygsti.models.create_crosstalk_free_model(pspec)

Ga_factory = XRotationOpFactory()
mdl_1q.factories['layers'][('Ga',0)] = Ga_factory
```

That model computes outcome probabilities for circuits containing `Gx`, `Gy`, or `Ga;<ANGLE>` on any qubit, where ANGLE is an angle in radians handed to `create_object`. Note the explicit qubit label, `0`: local noise models build gates using multi-qubit conventions.

```{code-cell} ipython3
c1 = pygsti.circuits.Circuit('Gx:0*Ga;3.1:0*Gx:0')
print(c1)
mdl_1q.probabilities(c1)
```

### More than one qubit

The extension to larger systems is mechanical. The only real addition is that a factory producing 1-qubit gates has to be *embedded* in the larger qubit space to yield an n-qubit gate factory. `EmbeddedOpFactory` does that: give it a tuple of all the qubits, say `(0,1)`, and a tuple of the subset to embed into, say `(0,)`.

```{code-cell} ipython3
pspec2 = pygsti.processors.QubitProcessorSpec(2, ('Gx','Gy','Gcnot'), geometry='line')
mdl_2q = pygsti.models.create_crosstalk_free_model(pspec2)

Ga_factory = XRotationOpFactory()
mdl_2q.factories['layers'][('Ga',0)] = op.EmbeddedOpFactory((0,1),(0,),Ga_factory)
mdl_2q.factories['layers'][('Ga',1)] = op.EmbeddedOpFactory((0,1),(1,),Ga_factory)

c2 = pygsti.circuits.Circuit("[Gx:0Ga;1.2:1][Ga;1.4:0][Gcnot:0:1][Gy:0Ga;0.3:1]" )
print(c2)

mdl_2q.probabilities(c2)
```

## Interpolating a physical process

Everything above assumes the process matrix is cheap to compute. Often it isn't. If your gate model is a Lindblad evolution, or a solver over a full device Hamiltonian, evaluating it once per optimizer step is hopeless.

The fix is to do the physics up front on a grid of parameter values and interpolate between those points afterward. `InterpolatedDenseOp` and `InterpolatedOpFactory` implement this. You write a `PhysicalProcess` subclass, interpolate it once over the parameter ranges you care about, and the resulting object evaluates quickly at any point inside those ranges. Interpolated objects can be saved and reloaded, so the expensive step happens once ever, not once per session.

This machinery lives in `pygsti.extras.interpygate`, imported as `interp` in the setup section above.

### Defining a physical process

Derive from `PhysicalProcess` and implement `create_process_matrix`. That is the expensive method: it takes a parameter vector and returns a process matrix. Every physical process declares a fixed number of parameters (the space to be interpolated over) and a fixed `process_shape`, almost always a square matrix of dimension $4^n$ for $n$ qubits. The returned matrix must be in whatever basis the eventual `Model` operations use, usually the Pauli-product basis `"pp"`.

Give an `aux_shape` and implement `create_aux_info` if you want extra floating-point values describing the process to be interpolated alongside it.

The process below evolves a quantum state for some time under a parameterized Lindbladian, then runs process tomography on the evolution to get a process matrix. Six parameters, one of which is the evolution time. The generators are pulled out to module level because a second version of this process appears shortly.

```{code-cell} ipython3
Hx = np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, -1],
               [0, 0, 1, 0]], dtype='float')
Hy = np.array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, -1, 0, 0]], dtype='float')
Hz = np.array([[0, 0, 0, 0],
               [0, 0, -1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]], dtype='float')

dephasing_generator = np.diag([0, -1, -1, 0])
decoherence_generator = np.diag([0, -1, -1, -1])

def lindbladian(omega, phase, detuning, dephasing, decoherence):
    H = (omega * np.cos(phase) * Hx + omega * np.sin(phase) * Hy + detuning * Hz)
    L = dephasing * dephasing_generator + decoherence * decoherence_generator
    return H + L
```

```{code-cell} ipython3
class ExampleProcess(interp.PhysicalProcess):
    def __init__(self):
        num_params = 6  # omega (0), phase (1), detuning (2), dephasing (3), decoherence (4), time (5)
        process_shape = (4, 4)
        super().__init__(num_params, process_shape, 
                         aux_shape=())  # our auxiliary information is a single float (None means no info)
                            
    def advance(self, state, v):
        """ Evolves `state` in time """
        state = np.array(state, dtype='complex')
        omega, phase, detuning, dephasing, decoherence, t = v  #Here are all our parameters

        process = pygsti.tools.change_basis(
            expm(lindbladian(omega, phase, detuning, dephasing, decoherence) * t), 'pp', 'col')
        dim = state.size
        vec_density_in  = np.outer(state, state.conj()).ravel(order='F')
        vec_density_out = process @ vec_density_in
        state = vec_density_out.reshape((dim, dim), order='F')
        return state

    def create_process_matrix(self, v, comm=None):
        def state_to_process_mxs(state):
            return self.advance(state, v)
        processes = interp.run_process_tomography(state_to_process_mxs, n_qubits=1,
                                                  basis='pp', comm=comm)  # returns None on all but root processor
        return np.array(processes) if (processes is not None) else None
    
    def create_aux_info(self, v, comm=None):
        omega, phase, detuning, dephasing, decoherence, t = v
        return t*omega  # matches aux_shape=() above
```

Call `create_process_matrix` directly to get a process matrix at one point in parameter space. Choosing the error-free parameters gives the ideal "target" operation.

```{code-cell} ipython3
example_process = ExampleProcess()
target_mx = example_process.create_process_matrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi/2]), comm=comm)
target_op = pygsti.modelmembers.operations.StaticArbitraryOp(target_mx)
print(target_op)
```

### Evaluating some parameters as a group

The process above is an evolution in time, and process matrices at different *times* are unusually cheap to get together: one evolution can produce the whole range in a single shot. Recomputing from scratch at every time value wastes most of the work.

`PhysicalProcess` supports this through the `num_params_evaluated_as_group` constructor argument. It defaults to 0. Set it to $k$ and the last $k$ parameters, counting backward from the end, get evaluated within a single call. Once it is nonzero, implement `create_process_matrices` and (optionally) `create_aux_infos` in place of the singular versions. These take an extra `grouped_v` argument holding *arrays* of values for the grouped parameters, and return arrays of process matrices with a matching leading index per grouped parameter.

Here is the same physics with *time* handled as a group of one:

```{code-cell} ipython3
class ExampleProcess_GroupTime(interp.PhysicalProcess):
    def __init__(self):
        num_params = 6  # omega (0), phase (1), detuning (2), dephasing (3), decoherence (4), time (5)
        process_shape = (4, 4)
        super().__init__(num_params, process_shape, 
                         aux_shape=(),  # a single float
                         num_params_evaluated_as_group=1)  # time values can be evaluated all at once

    def advance(self, state, v, times):
        state = np.array(state, dtype='complex')
        omega, phase, detuning, dephasing, decoherence = v

        G = lindbladian(omega, phase, detuning, dephasing, decoherence)
        processes = [pygsti.tools.change_basis(expm(G * t), 'pp', 'col') for t in times]
        vec_density_in  = np.outer(state, state.conj()).ravel(order='F')
        dim = state.size
        states = []
        for process in processes:
            vec_density_out = process @ vec_density_in
            state = vec_density_out.reshape((dim, dim), order='F')
            states.append(state)
        return states

    def create_process_matrices(self, v, grouped_v, comm=None):
        assert(len(grouped_v) == 1)  # we expect a single "grouped" parameter
        times = grouped_v[0]
        def state_to_process_mxs(state):
            return self.advance(state, v, times)
        processes = interp.run_process_tomography(state_to_process_mxs, n_qubits=1,
                                                  basis='pp', time_dependent=True, comm=comm)
        return np.array(processes) if (processes is not None) else None
    
    def create_aux_infos(self, v, grouped_v, comm=None):
        omega, phase, detuning, dephasing, decoherence = v
        times = grouped_v[0]
        return np.array([t*omega for t in times], 'd')
```

Getting a target operation out of this one works the same way, except you pass a list of times and index into the result.

```{code-cell} ipython3
example_process = ExampleProcess_GroupTime()
target_mx = example_process.create_process_matrices(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), [[np.pi/2]], comm=comm)[0]
target_op = pygsti.modelmembers.operations.StaticArbitraryOp(target_mx)
print(target_op)
```

### Building an interpolated operation

With the physical process written, the operator is nearly free. `InterpolatedDenseOp.create_by_interpolating_physical_process` evaluates the process on a grid and interpolates between the grid points. Its `from_vector` then works at any point inside the interpolated ranges.

The parameters of the resulting `InterpolatedDenseOp` are the parameters of the underlying `PhysicalProcess`. Specify each range as a *(min, max, num_points)* tuple or as an explicit array of values. The grid below uses only 2 points in most directions to keep the runtime down; a real application would want considerably more, and the cost grows as the product over directions.

You also pass a target operation. This is required because what actually gets interpolated is the *error generator*, not the process matrix. The target may be parameterized by any contiguous subset of the physical process's parameters starting from the first. Here `target_op` is a `StaticArbitraryOp` with 0 parameters, which reads as "the first 0 parameters of the process".

```{code-cell} ipython3
param_ranges = ([(0.9, 1.1, 2),  # omega
                 (-.1, .1, 2),   # phase
                 (-.1, .1, 2),   # detuning
                 (0, 0.1, 2),    # dephasing
                 (0, 0.1, 2),    # decoherence
                 np.linspace(np.pi / 2, np.pi / 2 + .5, 10)  # time
                ])

interp_op = interp.InterpolatedDenseOp.create_by_interpolating_physical_process(
    target_op, example_process, param_ranges, comm=comm)
```

Evaluate it anywhere in that box, quickly:

```{code-cell} ipython3
interp_op.from_vector([1.1, 0.01, 0.01, 0.055, 0.055, 1.59])
interp_op.to_dense()
```

The auxiliary information, if the process defines any, is on the `aux_info` attribute and reflects the last point evaluated.

```{code-cell} ipython3
interp_op.aux_info
```

### Building an interpolated operation factory

`InterpolatedOpFactory` interpolates a physical process the same way, but splits the process's parameters into two groups: *factory arguments* and *operation parameters*. Arguments range over intended (target) operations and are supplied by the circuit, as in the factory section above. Parameters stay unknown and get fit to data.

To build one, first write a factory class that produces the target operation for a given set of arguments. As with `InterpolatedDenseOp`, the target operations may be parameterized by any contiguous subset of the factory's parameters starting with the first.

The factory below takes *time* and *omega* as its arguments.

```{code-cell} ipython3
class TargetOpFactory(pygsti.modelmembers.operations.OpFactory):
    def __init__(self):
        self.process = ExampleProcess_GroupTime()
        pygsti.modelmembers.operations.OpFactory.__init__(self, state_space=1, evotype="densitymx")
        
    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None)
        assert(len(args) == 2)  # t (time), omega
        t, omega = args
        mx = self.process.create_process_matrices(np.array([omega, 0.0, 0.0, 0.0, 0.0]), [[t]], comm=None)[0]
        #mx = self.process.create_process_matrix(np.array([omega, 0.0, 0.0, 0.0, 0.0, t]), comm=None)  # Use this if using our initial ExampleProcess above.
        return pygsti.modelmembers.operations.StaticArbitraryOp(mx)
```

Construction mirrors the `InterpolatedDenseOp` case, with argument ranges and parameter ranges given separately, plus `arg_indices` saying which of the physical process's parameters become factory arguments.

```{code-cell} ipython3
arg_ranges = [np.linspace(np.pi / 2, np.pi / 2 + .5, 10),  # time
              (0.9, 1.1, 2)  # omega
             ]

param_ranges = [(-.1, .1, 2),  # phase
                (-.1, .1, 2),  # detuning
                (0, 0.1, 2),   # dephasing
                (0, 0.1, 2)    # decoherence
               ]
arg_indices = [5, 0]  #indices for time and omega within ExampleProcess_GroupTime's parameters

opfactory = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
                TargetOpFactory(), example_process, arg_ranges, param_ranges, arg_indices, comm=comm)
```

The factory has 4 parameters where the physical process and the interpolated operator built earlier have 6. Two of the six became factory arguments.

```{code-cell} ipython3
print(opfactory.num_params)
print(interp_op.num_params)
print(example_process.num_params)
```

Set the parameters, then ask the factory for the operation at a given *time* and *omega*:

```{code-cell} ipython3
opfactory.from_vector(np.array([0.01, 0.01, 0.055, 0.055]))
factory_op = opfactory.create_op((1.59, 1.1))
factory_op.to_dense()
```

```{code-cell} ipython3
factory_op.aux_info
```

## Where to go next

- The [operators tutorial](Operators) covers pyGSTi's existing operations, which are worth exhausting before you write your own.
- The [POVM tutorial](CustomPOVMs) does the same job as this page for terminating measurements.
- The [instrument tutorial](../../guides/gst/MidCircuitMeasurement) covers mid-circuit measurements.
