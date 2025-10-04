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

# Tutorial on creating a custom operator (e.g. gate)

This tutorial demonstrates the process of creating your own gate operation.  One can view gate (or layer) operations in pyGSTi as simply parameterized process matrices: a mapping that associates with any given set of parameter values a process matrix.  This mapping is encapsulated by a `LinearOperator`-derived class in pyGSTi, and in addition to using the ones included with pyGSTi (e.g. `FullArbitraryOp`, see the [Operator tutorial](Operators.ipynb) for more examples) you're free to make your own.  That's exactly what we'll be doing here.

There are lots of good reasons for doing this, the foremost being that you have a specific way you want to model a gate operation that is specific to your system's physics and not captured by pyGSTi's more generic built-in operation classes.  You also may want to make an operation whose parameters are exactly the "knobs" that you have access to in the lab.  Whatever the reason, pyGSTi has been designed to make the creation of custom operator types easy and straightforward.

In this example, we'll be creating a custom 1-qubit gate operation.  It will be a $X(\pi/2)$-rotation that may have some amount of depolarization and "on-axis" overrotation, but no other imperfections.  Thus, it will only have to parameters: the depolarization and the overrotation amounts.

Here's a class which implements this operation.  The comments explain what different parts do.

```{code-cell} ipython3
import pygsti
import numpy as np

class MyXPi2Operator(pygsti.modelmembers.operations.DenseOperator):
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

We'll add a `MyXPi2Operator` instance as the `("Gxpi2",0)` gate in pyGSTi's {Idle, $X(\pi/2)$, $Y(\pi/2)$} modelpack (see the [modelpacks tutorial](ModelPacks.ipynb) for more information on modelpacks).

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

Next, to demonstrate everything is working like it should, we'll optimize this model using Gate Set tomography (see the [GST overview tutorial](../../algorithms/GST-Overview.ipynb) for the details on what all this stuff does).  GST by default attempts to gauge optimize its final estimate to look like the target model (see the [gauge optimization tutorial](../../algorithms/advanced/GaugeOpt.ipynb) for details).  This would requires all of the operators in our model to implement the (gauge) `transform` method.  Because `MyXPi2Operator` doesn't, we tell GST not to perform any gauge optimization by setting `gauge_opt_params=False` below.

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
                                      smq1Q_XYI.germs(), maxLengths, gauge_opt_params=False)
```

**That's it!  We just ran GST with a custom operation.**

Our `MyXPi2Operator`-containing model fits the data pretty well (compare the actual and expected $2\Delta \log \mathcal{L}$ values printed above).  This makes sense because the data was generated by a model containing only depolarization errors on the gates, and our custom gate class can model this type of noise.  We expect, since we know how the data was generated, that the `"Gx"` gate depolarizes with magnitude $0.01$ and has no (zero) over-rotation.  Indeed, this is what we find when we look at `"Gx"` of the estimated model: 

```{code-cell} ipython3
mdl_estimate = results.estimates['GateSetTomography'].models['final iteration estimate']
print(mdl_estimate[('Gxpi2',0)])
est_depol, est_overrotation = mdl_estimate[('Gxpi2',0)].to_vector()
print("Estimated Gx depolarization =",est_depol)
print("Estimated Gx over-rotation =",est_overrotation)
```

The reason these values aren't exactly $0.01$ and $0$ are due to the finite number of samples, and to a lesser extent  gauge degrees of freedom.

## What's next?
This tutorial showed you how to create a custom *dense* operation (a subclass of `DenseOperator`).  We'll be adding demonstrations of more complex custom operations in the future.  Here are some places you might want to go next:
- The [operators tutorial](Operators.ipynb) explains and shows examples of pyGSTi's existing operations.
- MORE TODO

```{code-cell} ipython3

```
