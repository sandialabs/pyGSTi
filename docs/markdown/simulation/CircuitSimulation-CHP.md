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

# CHP Interface (legacy)

```{warning} This notebook is under construction and will have more description in the near future.```

```{code-cell} ipython3
from __future__ import print_function #python 2 & 3 compatibility
import pygsti

import numpy as np

from pygsti.modelmembers.operations import LinearOperator, StaticStandardOp, StochasticNoiseOp, DepolarizeOp, ComposedOp, EmbeddedOp
```

## LinearOperator and StaticStandardOp

Now with 'chp' evotype.

```{code-cell} ipython3
Gx = StaticStandardOp('Gxpi', evotype='chp')
print(Gx)
print(Gx._rep._chp_ops())
```

```{code-cell} ipython3
Gx.evotype.name
```

```{code-cell} ipython3
# Can also make custom CHP operations
# Here I'm making a (deterministic) Hadamard on qubit 0 and CNOT on qubits 1 and 2
rep = pygsti.evotypes.chp.opreps.OpRep(['h 0', 'c 1 2'], state_space=3)
c = LinearOperator(rep, 'chp')
```

```{code-cell} ipython3
print(c)
print(c._rep._chp_ops())
```

```{code-cell} ipython3
print(StaticStandardOp('Gc20', evotype='chp'))
```

## StochasticNoiseOp and DepolarizeOp

Now with 'chp' evotype

```{code-cell} ipython3
nqubits = 1
scop = StochasticNoiseOp(nqubits, basis='pp', evotype='chp', initial_rates=[0.5, 0.1, 0.1], seed_or_state=2021)
print(scop)
for _ in range(4):
    print(scop._rep._chp_ops())
```

```{code-cell} ipython3
nqubits = 1
dop = DepolarizeOp(nqubits, basis='pp', evotype='chp', initial_rate=0.7, seed_or_state=2021)
print(dop)
for _ in range(4): # With seed 2021, pulls Z, I (no output), X, Y
    print(dop._rep._chp_ops())
```

## ComposedOp + EmbeddedOp

```{code-cell} ipython3
# ComposedOp
Gzx_composed = ComposedOp([StaticStandardOp('Gzpi', evotype='chp'), StaticStandardOp('Gxpi', evotype='chp')])
print(Gzx_composed)
print(Gzx_composed._rep._chp_ops())
#print(Gzx_composed.get_chp_str([2]))
```

```{code-cell} ipython3
# EmbeddedOp
Gxi_embedded = EmbeddedOp(['Q0', 'Q1'], ['Q0'], StaticStandardOp('Gxpi', evotype='chp'))
print(Gxi_embedded)
print(Gxi_embedded._rep._chp_ops())
#print(Gxi_embedded.get_chp_str([5,7]))
```

```{code-cell} ipython3
Gix_embedded = EmbeddedOp(['Q0', 'Q1'], ['Q1'], StaticStandardOp('Gxpi', evotype='chp'))
print(Gix_embedded)
print(Gix_embedded._rep._chp_ops())
#print(Gix_embedded.get_chp_str([5,7]))
```

```{code-cell} ipython3
# EmbeddedOp made of ComposedOps
Gzx_comp_embed = EmbeddedOp(['Q0', 'Q1', 'Q2', 'Q3'], ['Q1'], Gzx_composed)
print(Gzx_comp_embed)
print(Gzx_comp_embed._rep._chp_ops())
#print(Gzx_comp_embed.get_chp_str([5, 6, 7, 8]))
```

## CHPForwardSimulator + Explicit Model

```{code-cell} ipython3
#This is the directory where the chp directory lives during github testing, replace this with the 
#correct directory for your personal installation
pygsti.evotypes.chp.chpexe = 'chp'
sim = pygsti.forwardsims.WeakForwardSimulator(shots=100)
```

```{code-cell} ipython3
#Initialize an empty Model object
model = pygsti.models.ExplicitOpModel(['Q0', 'Q1'], simulator=sim, evotype='chp')

def make_2Q_op(name0, name1):
    return ComposedOp([
        EmbeddedOp(['Q0', 'Q1'], ['Q0'], StaticStandardOp(name0, evotype='chp')),
        EmbeddedOp(['Q0', 'Q1'], ['Q1'], StaticStandardOp(name1, evotype='chp')),
    ])

#Populate the Model object with states, effects, gates
# For CHP, prep must be all-zero ComputationalSPAMVec
# and povm must be ComputationalBasisPOVM
model['rho0'] = pygsti.modelmembers.states.ComputationalBasisState([0, 0], evotype='chp')
model['Mdefault'] = pygsti.modelmembers.povms.ComputationalBasisPOVM(2, evotype='chp')

model['Gii'] = make_2Q_op('Gi', 'Gi')
model['Gxi'] = make_2Q_op('Gxpi', 'Gi')
model['Gix'] = make_2Q_op('Gi', 'Gxpi')
model['Gxx'] = make_2Q_op('Gxpi', 'Gxpi')
model['Gyi'] = make_2Q_op('Gypi', 'Gi')
model['Giy'] = make_2Q_op('Gi', 'Gypi')
model['Gyy'] = make_2Q_op('Gypi', 'Gypi')

print(model)
```

```{code-cell} ipython3
circ = pygsti.circuits.Circuit(['Gix'])
model.probabilities(circ)
```

```{code-cell} ipython3
circ = pygsti.circuits.Circuit(['Gix', 'Gxi'])
model.probabilities(circ)
```

```{code-cell} ipython3
circ = pygsti.circuits.Circuit(['rho0', 'Gxx', 'Mdefault'])
model.probabilities(circ)
```

## Advanced State Prep and Measurement

<font color='red'>TODO: This section does not work due to non-CHP related issues. Come back to this once other issues are fixed.</font>

### State Prep

```{code-cell} ipython3
#Initialize an empty Model object
prep01_model = pygsti.models.ExplicitOpModel(['Q0', 'Q1'], simulator=sim, evotype='chp')

# Make a ComputationalSPAMVec with one bit in 1 state
prep01_model.preps['rho0'] = pygsti.modelmembers.states.ComputationalBasisState([0, 1], evotype='chp')
prep01_model.povms['Mdefault'] = pygsti.modelmembers.povms.ComputationalBasisPOVM(2, evotype='chp')

circ = pygsti.circuits.Circuit([])
prep01_model.probabilities(circ)
```

```{code-cell} ipython3
#Initialize an empty Model object
prep00noise_model = pygsti.models.ExplicitOpModel(['Q0', 'Q1'], simulator=sim, evotype='chp')

# Make a ComposedSPAMVec where second qubit has X error
rho0 = pygsti.modelmembers.states.ComposedState(
    pygsti.modelmembers.states.ComputationalBasisState([0, 0], evotype='chp'), # Pure SPAM vec is 00 state
    make_2Q_op('Gi', 'Gxpi2')) # Second qubit has X(pi/2) error (partial flip on qubit 1)

prep00noise_model['rho0'] = rho0
prep00noise_model['Mdefault'] = pygsti.modelmembers.povms.ComputationalBasisPOVM(2, 'chp')

circ = pygsti.circuits.Circuit([])
prep00noise_model.probabilities(circ)
```

```{code-cell} ipython3
#Initialize an empty Model object
prep11noise_model = pygsti.models.ExplicitOpModel(['Q0', 'Q1'], simulator=sim, evotype='chp')

# Make a ComposedSPAMVec where second qubit has X error AND is initialized to 1 state
rho0 = pygsti.modelmembers.states.ComposedState(
    pygsti.modelmembers.states.ComputationalBasisState([1, 1], evotype='chp'), # Pure SPAM vec is 00 state
    make_2Q_op('Gi', 'Gxpi2')) # Second qubit has X(pi/2) error (partial flip on qubit 1)

prep11noise_model['rho0'] = rho0
prep11noise_model['Mdefault'] = pygsti.modelmembers.povms.ComputationalBasisPOVM(2, 'chp')

circ = pygsti.circuits.Circuit([])
prep11noise_model.probabilities(circ)
```

### Measurement

```{code-cell} ipython3
make_2Q_op('Gi', 'Gxpi2')._rep
```

```{code-cell} ipython3
##Initialize an empty Model object
#povm01_model = pygsti.models.ExplicitOpModel(['Q0', 'Q1'], simulator=sim, evotype='chp')
#
## Make a measurement with a bitflip error on qubit 1
#povm01_model.preps['rho0'] = pygsti.modelmembers.states.ComputationalBasisState([0, 1], evotype='chp')
#povm01_model.povms['Mdefault'] = pygsti.modelmembers.povms.ComposedPOVM(
#    make_2Q_op('Gi', 'Gxpi2'),
#    pygsti.modelmembers.povms.ComputationalBasisPOVM(2, evotype='chp'),
#    mx_basis='pp')
#
#povm01_model._primitive_povm_label_dict['Mdefault'] = povm01_model['Mdefault']
#
#circ = pygsti.circuits.Circuit([])
#povm01_model.probabilities(circ)
```

## CHPForwardSimulator + LocalNoiseModel

```{code-cell} ipython3
# Step 1: Define stochastic Pauli noise operators
# Note that the probabilities here are the "error rates" that would be model parameters (currently just static)
noise_1q = StochasticNoiseOp(1, basis='pp', evotype='chp', initial_rates=[0.1, 0.01, 0.01], seed_or_state=2021)

# Also need two-qubit version
# Here we just make it independent stochastic Pauli noise
noise_2q = ComposedOp([EmbeddedOp([0, 1], [0], noise_1q), EmbeddedOp([0, 1], [1], noise_1q)])
```

```{code-cell} ipython3
# Step 2: Define gate dict of noisy gates
# Using equivalent of XYICNOT modelpack
gatedict = {}
gatedict['Gi'] = noise_1q
gatedict['Gx'] = ComposedOp([StaticStandardOp('Gxpi', evotype='chp'), noise_1q])
gatedict['Gy'] = ComposedOp([StaticStandardOp('Gypi', evotype='chp'), noise_1q])
# Note that first Gcnot is now key in model, whereas second Gcnot is a standard gatename known to CHPOp constructor
gatedict['Gcnot'] = ComposedOp([StaticStandardOp('Gcnot', evotype='chp'), noise_2q])
```

```{code-cell} ipython3
from pygsti.models.localnoisemodel import LocalNoiseModel
from pygsti.modelmembers.states import ComputationalBasisState
from pygsti.modelmembers.povms import ComputationalBasisPOVM
from pygsti.processors import QubitProcessorSpec

pspec = QubitProcessorSpec(4, list(gatedict.keys()), geometry='line',
                           availability={'Gcnot': [(0,1),(1,2),(2,3)]})

rho0 = ComputationalBasisState([0,]*4, evotype='chp')
Mdefault = ComputationalBasisPOVM(4, evotype='chp')

ln_model = LocalNoiseModel(pspec, gatedict=gatedict, prep_layers=[rho0], povm_layers=[Mdefault],
                           simulator=sim, evotype='chp')
```

```{code-cell} ipython3
# Step 4: Profit?? Worked way too quickly...
def print_implicit_model_blocks(mdl, showSPAM=False):
    if showSPAM:
        print('State prep building blocks (.prep_blks):')
        for blk_lbl,blk in mdl.prep_blks.items():
            print(" " + blk_lbl, ": ", ', '.join(map(str,blk.keys())))
        print()

        print('POVM building blocks (.povm_blks):')
        for blk_lbl,blk in mdl.povm_blks.items():
            print(" "  + blk_lbl, ": ", ', '.join(map(str,blk.keys())))
        print()
    
    print('Operation building blocks (.operation_blks):')
    for blk_lbl,blk in mdl.operation_blks.items():
        print(" " + blk_lbl, ": ", ', '.join(map(str,blk.keys())))
    print()

print_implicit_model_blocks(ln_model, showSPAM=True)
```

```{code-cell} ipython3
print(ln_model.prep_blks['layers']['rho0'])
```

```{code-cell} ipython3
print(ln_model.operation_blks['gates']['Gx'])
```

```{code-cell} ipython3
Gcnot_layer_op = ln_model.operation_blks['layers']['Gcnot', 1, 2]
print(ln_model.operation_blks['layers']['Gcnot', 1, 2])
```

```{code-cell} ipython3
# Step 5: Actually run circuits with local noise model
circ = pygsti.circuits.Circuit([('Gx', 1)], num_lines=4)
ln_model.probabilities(circ)
```

```{code-cell} ipython3
circ = pygsti.circuits.Circuit([('Gx', 1), ('Gcnot', 1, 2)], num_lines=4)
ln_model.probabilities(circ)
```

```{code-cell} ipython3
# Could also define correlated noise for 2-qubit error?
pp = pygsti.baseobjs.Basis.cast('pp', 16)
rates_2q = [0.01,]*15
rates_2q[pp.labels.index('XX')] = 0.1 # Set XX to much higher

noise_2q_correlated = StochasticNoiseOp(2, basis='pp', evotype='chp', initial_rates=rates_2q, seed_or_state=2021)

gatedict = {}
gatedict['Gi'] = noise_1q
gatedict['Gx'] = ComposedOp([StaticStandardOp('Gxpi', evotype='chp'), noise_1q])
gatedict['Gy'] = ComposedOp([StaticStandardOp('Gypi', evotype='chp'), noise_1q])
# Note that first Gcnot is now key in model, whereas second Gcnot is a standard gatename known to CHPOp constructor
gatedict['Gcnot'] = ComposedOp([StaticStandardOp('Gcnot', evotype='chp'), noise_2q_correlated])
```

```{code-cell} ipython3
rho0 = ComputationalBasisState([0,]*4, evotype='chp')
Mdefault = ComputationalBasisPOVM(4, evotype='chp')

sim = pygsti.forwardsims.WeakForwardSimulator(shots=100)

ln_model_corr = LocalNoiseModel(pspec, gatedict=gatedict, prep_layers=[rho0], povm_layers=[Mdefault],
                                simulator=sim, evotype='chp')
```

```{code-cell} ipython3
# Now the CNOT gates have a 2-qubit stochastic gate instead of independent 1-qubit ones
print(ln_model_corr.operation_blks['layers']['Gcnot', 1, 2])
```

```{code-cell} ipython3
circ = pygsti.circuits.Circuit([('Gx', 1)], num_lines=4)
ln_model_corr.probabilities(circ)
```

```{code-cell} ipython3
circ = pygsti.circuits.Circuit([('Gx', 1), ('Gcnot', 1, 2)], num_lines=4)
ln_model_corr.probabilities(circ)
```

## Crosstalk-Free Model Construction

```{code-cell} ipython3
#import pygsti.models.modelconstruction as mc
#
#sim = pygsti.forwardsims.WeakForwardSimulator(shots=100, base_seed=2021)
#
#pspec = QubitProcessorSpec(4, ['Gi', 'Gxpi', 'Gypi', 'Gcnot'], availability={'Gcnot': [(0,1),(1,2),(2,3)]})
#
## Use the same 2-qubit stochastic noise for CNOT as above
#ctf_model = mc.create_crosstalk_free_model(pspec,
#    depolarization_strengths={'Gi': 0.1, 'Gxpi': 0.1},
#    stochastic_error_probs={'Gypi': [0.1, 0.1, 0.1], 'Gcnot': rates_2q},
#    simulator=sim, evotype='chp')
#
#print_implicit_model_blocks(ctf_model, showSPAM=True)
```

```{code-cell} ipython3
#for name, gate in ctf_model.operation_blks['gates'].items():
#    print(f'Gate {name}')
#    print(gate)
#    print()
```

```{code-cell} ipython3
#circ = pygsti.circuits.Circuit([('Gxpi', 1)], num_lines=4)
#ctf_model.probabilities(circ)
```

```{code-cell} ipython3
#circ = pygsti.circuits.Circuit([('Gxpi', 1), ('Gcnot', 1, 2)], num_lines=4)
#ctf_model.probabilities(circ)
```

```{code-cell} ipython3
# Marginalized POVMs now work!
#circ = pygsti.circuits.Circuit([('Gxpi', 1), ('Gcnot', 1, 2)])
#ctf_model.probabilities(circ)
```

```{code-cell} ipython3
# Let's try a model with only readout error
#sim = pygsti.forwardsims.CHPForwardSimulator(chpexe, shots=1000) # Bump up shots for better noise resolution
#
#ctf_povm_model = mc.create_crosstalk_free_model(pspec,
#    stochastic_error_probs={'povm': [0.05, 0.0, 0.0]}, # 5% X error on prep
#    simulator=sim, evotype='chp')
```

```{code-cell} ipython3
#circ = pygsti.circuits.Circuit([])
#ctf_povm_model.probabilities(circ) # Expect about 80% all 0, 5% on weight one errors, 0.25% on weight 2, etc.
```

```{code-cell} ipython3

```
