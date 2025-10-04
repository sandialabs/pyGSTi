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

# How to create gates from physical processes
This tutorial shows how to use the `InterpolatedDenseOp` and `InterpolatedOpFactory` to create quick-to-evaluate operations by interpolating between the discrete points at quick a more computationally-intensive process is performed.  Often the computationally intensive process simulates the physics of a qubit gate, and would not practially work as a custom model operation because of the time required to evaluate it.

In order to turn such physical processes into gates, you should implement a custom `PhysicalProcess` object and then use the `InterpolatedDenseOp` or `InterpolatedOpFactory` class to interpolate the values of the custom process on a set of pre-defined points.  All the physics simulation is then done at the time of creating the interpolated operation (or factory), after which the object can be saved for later use.  An `InterpolatedDenseOp` or `InterpolatedOpFactory` object can be evaluated at any parameter-space point within the ranges over which the initial interpolation was performed.

All of this functionality is currently provided within the `pygsti.extras.interpygate` sub-package.  This tutorial demonstrates how to setup a custom physical process and create an interpolated gate and factory object from it.

We'll begin by some standard imports and by importing the `interpygate` sub-package.  We get a MPI communicator if we can, as usually the physical simulation is performed using multiple processors.

```{code-cell} ipython3
import numpy as np
from scipy.linalg import expm

import pygsti
import pygsti.extras.interpygate as interp

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None
```

## Defining a physical process
We create a physical process simulator by deriving from the `PhysicalProcess` class and implementing its `create_process_matrix` function.  This is the computationally intensive method that generates a process matrix based on some set of parameters.  Every physical process has a fixed number of parameters that define the space that will be interpolated over.  The generated process matrix is expected to be in whatever basis the ultimate `Model` operations will be in - usually the Pauli-product basis specified by `"pp"` - and have a fixed shape.  This shape, given by `process_shape` below, is almost always a square matrix of dimension $4^n$ where $n$ is the number of qubits.  Specifying an auxiliary information shape (`aux_shape` below) and implementing the `create_aux_info` will allow additional (floating point) values that describe the process to be interpolated. 

Below we create a physical process that evolves a quantum state for some time (also a parameter) using a parameterized Lindbladian.  Process tomography is used to construct a process matrix from the state evolution.  The process has 6 parameters.

```{code-cell} ipython3
class ExampleProcess(interp.PhysicalProcess):
    def __init__(self):
        self.Hx = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, -1],
                            [0, 0, 1, 0]], dtype='float')
        self.Hy = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, -1, 0, 0]], dtype='float')
        self.Hz = np.array([[0, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]], dtype='float')

        self.dephasing_generator = np.diag([0, -1, -1, 0])
        self.decoherence_generator = np.diag([0, -1, -1, -1])
        num_params = 6  # omega (0), phase (1), detuning (2), dephasing (3), decoherence (4), time (5)
        process_shape = (4, 4)
        super().__init__(num_params, process_shape, 
                         aux_shape=())  # our auxiliary information is a single float (None means no info)
                            
    def advance(self, state, v):
        """ Evolves `state` in time """
        state = np.array(state, dtype='complex')
        omega, phase, detuning, dephasing, decoherence, t = v  #Here are all our parameters

        H = (omega * np.cos(phase) * self.Hx + omega * np.sin(phase) * self.Hy + detuning * self.Hz)
        L = dephasing * self.dephasing_generator + decoherence * self.decoherence_generator

        process = pygsti.tools.change_basis(expm((H + L) * t),'pp', 'col')
        state = interp.unvec(np.dot(process, interp.vec(np.outer(state, state.conj()))))
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

We can call `create_process_matrix` to generate a process matrix at a given set of parameters.  Below we compute the ideal "target" operation by choosing the parameters corresponding to no errors.

```{code-cell} ipython3
example_process = ExampleProcess()
target_mx = example_process.create_process_matrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi/2]), comm=comm)
target_op = pygsti.modelmembers.operations.StaticArbitraryOp(target_mx)
print(target_op)
```

### Making things more efficient

We note that since our physical process is just an evolution in time, process matrices corresponding to different values of (just) the *time* parameter are especially easy to compute - a single evolution could compute, in one shot, the process matrices for an entire range of times.  

The `PhysicalProcess` class contains support for such "easy-to-compute" parameters via the `num_params_evaluated_as_group` argument to its constructor.  This argument defaults to 0, and specifies how many of the parameters, starting with the last one and working backward, should be evaluated within the same function call.  If `num_params_evaluated_as_group` is set higher than 0, the derived class must implement the `create_process_matrices` and (optionally) `create_aux_infos` methods instead of `create_process_matrix` and `create_aux_info`.  These methods take an additional `grouped_v` argument that contains *arrays* of values for the final `num_params_evaluated_as_group` parameters, and are expected return arrays of process matrices with corresponding shape (i.e., there is a leading index in the retured values for each "grouped" parameter).

We demonstrate this more complex usage below, where values for our final *time* argument are handled all at once.

```{code-cell} ipython3
class ExampleProcess_GroupTime(interp.PhysicalProcess):
    def __init__(self):
        self.Hx = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, -1],
                            [0, 0, 1, 0]], dtype='float')
        self.Hy = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, -1, 0, 0]], dtype='float')
        self.Hz = np.array([[0, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]], dtype='float')

        self.dephasing_generator = np.diag([0, -1, -1, 0])
        self.decoherence_generator = np.diag([0, -1, -1, -1])
        num_params = 6  # omega (0), phase (1), detuning (2), dephasing (3), decoherence (4), time (5)
        process_shape = (4, 4)
        super().__init__(num_params, process_shape, 
                         aux_shape=(),  # a single float
                         num_params_evaluated_as_group=1)  # time values can be evaluated all at once

    
    def advance(self, state, v, times):
        state = np.array(state, dtype='complex')
        omega, phase, detuning, dephasing, decoherence = v

        H = (omega * np.cos(phase) * self.Hx + omega * np.sin(phase) * self.Hy + detuning * self.Hz)
        L = dephasing * self.dephasing_generator + decoherence * self.decoherence_generator

        processes = [pygsti.tools.change_basis(expm((H + L) * t),'pp', 'col') for t in times]
        states = [interp.unvec(np.dot(process, interp.vec(np.outer(state, state.conj())))) for process in processes]
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

We can similarly create a target operation from this physical process, but now we must specify a list of times.

```{code-cell} ipython3
example_process = ExampleProcess_GroupTime()
target_mx = example_process.create_process_matrices(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), [[np.pi/2]], comm=comm)[0]
target_op = pygsti.modelmembers.operations.StaticArbitraryOp(target_mx)
print(target_op)
```

## Creating an interpolated operation (gate)
Now that we've done the hard work of creating the physical process, it's easy to create an operator that evaluates the physical process on a grid of points and interpolates between them.  The resulting `InterpolatedDenseOp` can be evaluated (i.e. `from_vector` can be invoked) at any point within the range being interpolated.

The parameters of the resulting `InterpolatedDenseOp` are the same as those of the underlying `PhysicalProcess`, and ranges are specified using either a *(min, max, num_points)* tuple or an array of values.  Below we use only 2 points in most directions so it doesn't take too long to run. 

Creating the object also requires a target operation, for which we use `target_op` as defined above.  This is required because internally it is the *error generator* rather than the process matrix itself that is interpolated. The target operation can be parameterized by any contiguous subset of the physical process's parameters, starting with the first one. In our example, `target_op` is a `StaticArbitraryOp` and so takes 0 parameters.  This should be interpreted as the "first 0 parameters of our example process".

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

The created `interp_op` can then be evaluated (quickly) at points in parameter space.

```{code-cell} ipython3
interp_op.from_vector([1.1, 0.01, 0.01, 0.055, 0.055, 1.59])
interp_op.to_dense()
```

The auxiliary information can be retrieved from any interpolated operator via its `aux_info` attribute.

```{code-cell} ipython3
interp_op.aux_info
```

## Creating an interpolated operation factory
Operation factories in pyGSTi take "arguments" provided by in-circuit labels and produce operations.  For example, the value of the rotation angle might be specified over a continuous interval by the algorithm being run, rather than being noise parameter that is fit to data when a model is optimized (e.g. in GST).

The `InterpolatedOpFactory` object interpolates a physical process, similar to `InterpolatedDenseOp`, but allows the user to divide the parameters of the physical process into *factory arguments* and *operation parameters*.  The first group is meant to range over different intended (target) operations, and the latter group is meant to be unkonwn quantities determined by fitting a model to data.  To create an `InterpolatedOpFactory`, we must first create a custom factory class that creates the target operation corresponding to a given set of arguments.  As in the case of `InterpolatedDenseOp`, the target operations can be parameterized by any contiguous subset of the factory's parameters, starting with the first one. 

We choose to make a factory that takes as arguments the *time* and *omega* physical process parameters. 

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

We can then create an `InterpolatedOpFactory` similarly to how we created an `InterpolatedDenseOp` except now we separately specify factory argument and optimization parameter ranges, and specify which of the underlying physical process's parameters are turned into factory arguments (`arg_indices` below).

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

Note that the factory has only 4 parameters (whereas the physical process and the interpolated operator we made above have 6).  This is because 2 of the physical process parameters have been turned into factory arguments.

```{code-cell} ipython3
print(opfactory.num_params)
print(interp_op.num_params)
print(example_process.num_params)
```

We can use the factory to create an `InterpolatedDenseOp` operation at a given *time* and *omega* pair:

```{code-cell} ipython3
opfactory.from_vector(np.array([0.01, 0.01, 0.055, 0.055]))
op = opfactory.create_op((1.59, 1.1))
op.to_dense()
```

```{code-cell} ipython3
op.aux_info
```

```{code-cell} ipython3

```
