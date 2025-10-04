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

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
np.set_printoptions(precision=4, linewidth=120, floatmode='maxprec_equal')
```

```{code-cell} ipython3
import pygsti
import pygsti.extras.interpygate as interp
from pygsti.tools.basistools import change_basis
from pygsti.modelpacks import smq1Q_XY
```

```{code-cell} ipython3
from pathlib import Path
working_dir = Path.cwd()
```

## Build model gate

```{code-cell} ipython3
sigI = np.array([[1.,0],[0, 1]], dtype='complex')
sigX = np.array([[0, 1],[1, 0]], dtype='complex')
sigY = np.array([[0,-1],[1, 0]], dtype='complex') * 1.j
sigZ = np.array([[1, 0],[0,-1]], dtype='complex')
sigM = (sigX - 1.j*sigY)/2.
sigP = (sigX + 1.j*sigY)/2.
```

```{code-cell} ipython3
class SingleQubitTargetOp(pygsti.modelmembers.operations.OpFactory):

    def __init__(self):
        self.process = self.create_target_gate
        pygsti.modelmembers.operations.OpFactory.__init__(self, 1, evotype="densitymx")
        self.dim = 4

    def create_target_gate(self, v):
        
        phi, theta = v
        target_unitary = (np.cos(theta/2) * sigI + 
                          1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
        superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')

        return superop
    
    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None)
        mx = self.process([*args])
        return pygsti.modelmembers.operations.StaticArbitraryOp(mx)
```

```{code-cell} ipython3
class SingleQubitGate(interp.PhysicalProcess):
    def __init__(self, 
                 verbose=False,
                 cont_param_gate = False,
                 num_params = None,
#                  process_shape = (4, 4),
                 item_shape = (4,4),
                 aux_shape = None,
                 num_params_evaluated_as_group = 0,
                 ):

        self.verbose = verbose

        self.cont_param_gate = cont_param_gate

        self.num_params = num_params
        self.item_shape = item_shape

        self.aux_shape = aux_shape
        self.num_params_evaluated_as_group = num_params_evaluated_as_group

   
    def create_process_matrix(self, v, comm=None, return_generator=False):                                                                                                                                                                                             

        processes = []
        phi, theta, t = v
        theta = theta * t
        target_unitary = (np.cos(theta/2) * sigI + 
                          1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
        superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
        processes += [superop]
        return np.array(processes) if (processes is not None) else None

    def create_aux_info(self, v, comm=None):
        return []  # matches aux_shape=() above
    
    def create_process_matrices(self, v, grouped_v, comm=None):
        assert(len(grouped_v) == 1)  # we expect a single "grouped" parameter

        processes = []
        times = grouped_v[0]
        phi_in, theta_in = v
        for t in times:
            phi = phi_in
            theta = theta_in * t
            target_unitary = (np.cos(theta/2) * sigI + 
                              1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
            superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
            processes += [superop]
        return np.array(processes) if (processes is not None) else None

    def create_aux_infos(self, v, grouped_v, comm=None):
        import numpy as np
        times = grouped_v[0]
        return [ [] for t in times] # list elements must match aux_shape=() above
```

```{code-cell} ipython3
param_ranges = [(0.9,1.1,3)]

arg_ranges = [2*np.pi*(1+np.cos(np.linspace(np.pi,0, 7)))/2,
              (0, np.pi, 3)] 
arg_indices = [0,1]


target_op = SingleQubitTargetOp()
gate_process = SingleQubitGate(num_params = 3,num_params_evaluated_as_group = 1)
```

```{code-cell} ipython3
opfactory_linear = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
    target_op, gate_process, argument_ranges=arg_ranges, parameter_ranges=param_ranges, 
    argument_indices=arg_indices, interpolator_and_args='linear')

opfactory_spline = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
    target_op, gate_process, argument_ranges=arg_ranges, parameter_ranges=param_ranges, 
    argument_indices=arg_indices, interpolator_and_args='spline')
```

### Check that the interpolator is working

```{code-cell} ipython3
if False:
    indices = (2,3)
    nparams = 30

    x = np.linspace(0,2*np.pi, nparams)
    y = np.linspace(0, np.pi, nparams)
    for p in np.linspace(.9,1.1,5):

        def interp_linear(x, y):    
            op = opfactory_linear.create_op([x, y])
            return op.base_interpolator([x,y,p])[indices]

        def interp_spline(x, y):    
            op = opfactory_spline.create_op([x, y])
            return op.base_interpolator([x,y,p])[indices]

        def truth(x, y):
            tmp_gate = gate_process.create_process_matrix([x,y,p])[0]
            tar_gate = target_op.create_target_gate([x,y])
            return pygsti.error_generator(tmp_gate, tar_gate, 'pp', 'logGTi')[indices]


        X, Y = np.meshgrid(x, y, indexing='ij')
        Z_linear = np.zeros([nparams, nparams])
        Z_spline = np.zeros([nparams, nparams])
        Z_truth  = np.zeros([nparams, nparams])
        for idx, xx in enumerate(x):
            for idy, yy in enumerate(y):
                Z_linear[idx,idy] = interp_linear(xx,yy)
                Z_spline[idx,idy] = interp_spline(xx,yy)
                Z_truth[idx,idy]  = truth(xx,yy)

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z_linear-Z_truth, rstride=1, cstride=1,
                        edgecolor='none', alpha=.8)
        ax.plot_surface(X, Y, Z_spline-Z_truth, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', alpha=.8)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    #     ax.set_zlim(-1,1)
        plt.show()
```

# Build a model from this gate

```{code-cell} ipython3
x_gate = opfactory_spline.create_op([0,np.pi/4])
y_gate = opfactory_spline.create_op([np.pi/2,np.pi/4])
```

```{code-cell} ipython3
x_gate.from_vector([1.03])
y_gate.from_vector([1.0])
print(np.round_(x_gate,4))
print()
print(np.round_(y_gate,4))
```

```{code-cell} ipython3
x_gate.parameter_bounds = np.array([[0.91, 1.09]])
y_gate.parameter_bounds = np.array([[0.91, 1.09]])
```

## Make a fake dataset

```{code-cell} ipython3
#Model only has Gx and Gy gates.  Let's rename them.

model = pygsti.models.ExplicitOpModel([0],'pp')

model['rho0'] = [ 1/np.sqrt(2), 0, 0, 1/np.sqrt(2) ] # density matrix [[1, 0], [0, 0]] in Pauli basis
model['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM(
    {'0': [ 1/np.sqrt(2), 0, 0, 1/np.sqrt(2) ],   # projector onto [[1, 0], [0, 0]] in Pauli basis
     '1': [ 1/np.sqrt(2), 0, 0, -1/np.sqrt(2) ] }, evotype="default") # projector onto [[0, 0], [0, 1]] in Pauli basis
model['Gxpi2',0] = x_gate
model['Gypi2',0] = y_gate
```

```{code-cell} ipython3
model.num_params
```

```{code-cell} ipython3
# Define the error model used to generate data
datagen_model = model.copy()
datagen_params = datagen_model.to_vector()
datagen_params[-2:] = [1.03,1.00]
datagen_model.from_vector(datagen_params)
datagen_model.probabilities( (('Gxpi2',0),('Gypi2',0),))
```

```{code-cell} ipython3
model.parameter_labels
```

```{code-cell} ipython3
# # Link the over-rotation errors on Gx and Gy
#model.collect_parameters(model.parameter_labels[-2:], 'Shared Gx/Gy physical parameter')
#print(model.parameter_labels)
print(model.num_params)

# # Define the error model used to generate data
# datagen_model = model.copy()
# datagen_params = datagen_model.to_vector()
# datagen_params[-1:] = [1.02]
# datagen_model.from_vector(datagen_params)
# datagen_model.probabilities( (('Gxpi2',0),('Gypi2',0),))
```

```{code-cell} ipython3
# Define the perfect target model
target_model = model.copy()
target_params = target_model.to_vector()
target_params[-2:] = [1,1]
# target_model.from_vector(target_params)
target_model.probabilities( (('Gxpi2',0),('Gypi2',0),))
```

### Germ and fiducial selection

```{code-cell} ipython3
final_germs = pygsti.algorithms.germselection.find_germs(
                target_model, randomize=False, force=None, algorithm='greedy', 
                verbosity=4, num_nongauge_params=2)
```

```{code-cell} ipython3
fiducial_pairs = pygsti.algorithms.fiducialpairreduction.find_sufficient_fiducial_pairs_per_germ(
                                model, 
                                smq1Q_XY.prep_fiducials(),
                                smq1Q_XY.meas_fiducials(), 
                                final_germs)
```

```{code-cell} ipython3
# # Reduce the number of fiducial pairs by hand, if you want

# fiducial_pairs2 = fiducial_pairs.copy()
# for key in fiducial_pairs2.keys():
#     fiducial_pairs2[key] = fiducial_pairs2[key][0:2]
# fiducial_pairs = fiducial_pairs2

# print(fiducial_pairs)
```

```{code-cell} ipython3
# Use fiducial pair reductions
exp_design = pygsti.protocols.StandardGSTDesign(model, 
                                                smq1Q_XY.prep_fiducials(), 
                                                smq1Q_XY.meas_fiducials(), 
                                                final_germs, 
                                                max_lengths=[1,2,4,8,16,32,64,128,256], 
                                                fiducial_pairs=fiducial_pairs,
                                                include_lgst=False)

dataset = pygsti.data.simulate_data(datagen_model, exp_design.all_circuits_needing_data,
                                    num_samples=1000, seed=1234)

data = pygsti.protocols.ProtocolData(exp_design, dataset)
```

```{code-cell} ipython3
len(data.dataset)
```

## Fisher information matrix

```{code-cell} ipython3
fim = pygsti.tools.edesigntools.calculate_fisher_information_matrix(model,
                                                                    exp_design.all_circuits_needing_data)
```

```{code-cell} ipython3
np.log(np.linalg.inv(fim))
```

```{code-cell} ipython3
plt.matshow(np.linalg.inv(fim))
```

# Run GST on the dataset

```{code-cell} ipython3
proto = pygsti.protocols.GateSetTomography(model, gaugeopt_suite=None)
results = proto.run(data)
```

```{code-cell} ipython3
# What is the estimated value of the error parameter?

final_model = results.estimates['GateSetTomography'].models['final iteration estimate']
print('Actual: ', datagen_model.to_vector()[-2:])
print('Estimated: ', final_model.to_vector()[-2:])
```

```{code-cell} ipython3
pprint(np.sqrt(2)*final_model.to_vector()[0:4])
pprint(np.sqrt(2)*final_model.to_vector()[4:8])
pprint(np.sqrt(2)*final_model.to_vector()[8:12])
```

```{code-cell} ipython3

```
