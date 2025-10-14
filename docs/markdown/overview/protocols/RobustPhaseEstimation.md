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

# Robust Phase Estimation
This notebook demonstrates how to use Robust Phase Estimation (RPE) to estimate certain parameters of a standard single-qubit model.  The RPE protocol is contained within the `extras` package of pyGSTi.

```{code-cell} ipython3
#Import relevant namespaces.

import pygsti
from pygsti.modelpacks.legacy import std1Q_XY as Std1Q_XY
from pygsti.extras import rpe

import numpy as np
```

```{code-cell} ipython3
#Declare the particular RPE instance we are interested in
#(X and Y pi/2 rotations)
#(Prep and measurement are for the |0> state.   See below for prep and measure in |0> and |1>, respectively.)
rpeconfig_inst = rpe.rpeconfig_GxPi2_GyPi2_00
```

```{code-cell} ipython3
#Declare a variety of relevant parameters

target_model = Std1Q_XY.target_model()
target_model.set_all_parameterizations('full TP')
maxLengths_1024 = [1,2,4,8,16,32,64,128,256,512,1024]

stringListsRPE = rpe.rpeconstruction.create_rpe_angle_circuits_dict(10,rpeconfig_inst)

angleList = ['alpha','epsilon','theta']

numStrsD = {}
numStrsD['RPE'] = [6*i for i in np.arange(1,12)]
```

```{code-cell} ipython3
#Create noisy model
mdl_real = target_model.randomize_with_unitary(.01,seed=0)
```

```{code-cell} ipython3
#Extract noisy model angles
true_alpha = rpe.extract_alpha(mdl_real,rpeconfig_inst)
true_epsilon = rpe.extract_epsilon(mdl_real,rpeconfig_inst)
true_theta = rpe.extract_theta(mdl_real,rpeconfig_inst)
```

```{code-cell} ipython3
#Simulate dataset
N=1000
DS = pygsti.data.simulate_data(mdl_real,stringListsRPE['totalStrList'],N,sample_error='binomial',seed=1)
```

```{code-cell} ipython3
#Analyze dataset
resultsRPE = rpe.analyze_rpe_data(DS,mdl_real,stringListsRPE,rpeconfig_inst)
```

```{code-cell} ipython3
#Print results
print('alpha_true - pi/2 =',true_alpha-np.pi/2)
print('epsilon_true - pi/2 =',true_epsilon-np.pi/2)
print('theta_true =',true_theta)
print()
print('alpha_true - alpha_est_final =',resultsRPE['alphaErrorList'][-1])
print('epsilon_true - epsilon_est_final =',resultsRPE['epsilonErrorList'][-1])
print('theta_true - theta_est_final =',resultsRPE['thetaErrorList'][-1])
```

```{code-cell} ipython3
#Repeat above with prep and measure in |0> and |1>, respectively.)
rpeconfig_inst = rpe.rpeconfig_GxPi2_GyPi2_UpDn
```

```{code-cell} ipython3
#Declare a variety of relevant parameters
target_model = pygsti.models.create_explicit_model_from_expressions([('Q0',)], ['Gx','Gy'],
                                  [ "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                  effect_expressions=['1','0'])
target_model.set_all_parameterizations('full TP')
maxLengths_1024 = [1,2,4,8,16,32,64,128,256,512,1024]

stringListsRPE = rpe.rpeconstruction.create_rpe_angle_circuits_dict(10,rpeconfig_inst)

angleList = ['alpha','epsilon','theta']

numStrsD = {}
numStrsD['RPE'] = [6*i for i in np.arange(1,12)]
```

```{code-cell} ipython3
#Create noisy model
mdl_real = target_model.randomize_with_unitary(.01,seed=0)
```

```{code-cell} ipython3
#Extract noisy model angles
true_alpha = rpe.extract_alpha(mdl_real,rpeconfig_inst)
true_epsilon = rpe.extract_epsilon(mdl_real,rpeconfig_inst)
true_theta = rpe.extract_theta(mdl_real,rpeconfig_inst)
```

```{code-cell} ipython3
#Simulate dataset
N=1000
DS = pygsti.data.simulate_data(mdl_real,stringListsRPE['totalStrList'],N,sample_error='binomial',seed=1)
```

```{code-cell} ipython3
#Analyze dataset
resultsRPE = rpe.analyze_rpe_data(DS,mdl_real,stringListsRPE,rpeconfig_inst)
```

```{code-cell} ipython3
#Print results
print('alpha_true - pi/2 =',true_alpha-np.pi/2)
print('epsilon_true - pi/2 =',true_epsilon-np.pi/2)
print('theta_true =',true_theta)
print()
print('alpha_true - alpha_est_final =',resultsRPE['alphaErrorList'][-1])
print('epsilon_true - epsilon_est_final =',resultsRPE['epsilonErrorList'][-1])
print('theta_true - theta_est_final =',resultsRPE['thetaErrorList'][-1])
```

```{code-cell} ipython3

```
