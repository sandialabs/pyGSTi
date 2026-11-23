---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Robust phase estimation
This notebook demonstrates how to use Robust Phase Estimation (RPE) to estimate certain parameters of a standard single-qubit model.  RPE determines the phase of a target gate $U$ by iterated action of that gate on a superposition of eigenstates of $U$, and tolerates upwards of 30% error in counts, whatever its cause (statistical noise, calibration error).

## With the RPE protocol

For the $\pi/2$ phase of an $X$ rotation, the `smq1Q_Xpi2_rpe` model pack provides an experiment design, and the `RobustPhaseEstimation` protocol runs on its data like any other pyGSTi protocol. It needs only noisy $X_{\pi/2}$ gates and SPAM.

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_Xpi2_rpe, smq1Q_XY

exp_design = smq1Q_Xpi2_rpe.create_rpe_experiment_design(max_max_length=64)

# write an empty data object (creates a template to fill in)
pygsti.io.write_empty_protocol_data('../../../tutorial_files/test_rpe_dir', exp_design, clobber_ok=True)

# fill in the template with simulated data (you would run the experiment and use actual data)
pygsti.io.fill_in_empty_dataset_with_fake_data(
    "../../../tutorial_files/test_rpe_dir/data/dataset.txt",
    smq1Q_XY.target_model().depolarize(op_noise=0.01, spam_noise=0.1),
    num_samples=1000, seed=1234)

# read the data object back in, now with the experimental data
data = pygsti.io.read_data_from_dir('../../../tutorial_files/test_rpe_dir/')

results = pygsti.protocols.rpe.RobustPhaseEstimation().run(data)
print(results.angle_estimate)
```

## With the lower-level functions

The functions in `pygsti.extras.rpe` expose the individual steps: circuit construction for a chosen RPE configuration, angle extraction from a model, and analysis of a data set. The rest of this notebook uses them to estimate the angles of an $X$/$Y$ gate set.

```{code-cell} ipython3
#Import relevant namespaces.

import pygsti
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

target_model = pygsti.models.create_explicit_model_from_expressions([('Q0',)], ['Gx','Gy'],
                                  [ "X(pi/2,Q0)", "Y(pi/2,Q0)"])
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
