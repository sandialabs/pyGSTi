---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: pygsti
  language: python
  name: python3
---

# Leakage (manual)

This tutorial demonstrates how to perform GST on a "leaky-qubit" described by a 3-level (instead of the desired 2-level) system.

```{code-cell} ipython3
import pygsti
import pygsti.modelpacks.smq1Q_XYI as smq1Q
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
import numpy as np
import scipy.linalg as sla
#import pickle
```

```{code-cell} ipython3
def to_3level_unitary(U_2level):
    U_3level = np.zeros((3,3),complex)
    U_3level[0:2,0:2] = U_2level
    U_3level[2,2] = 1.0
    return U_3level

def unitary_to_gmgate(U):
    return pygsti.tools.change_basis( 
        pygsti.tools.unitary_to_std_process_mx(U), 'std','gm')

def state_to_gmvec(state):
    pygsti.tools.stdmx_to_gmvec

Us = pygsti.tools.internalgates.standard_gatename_unitaries()
```

```{code-cell} ipython3
mdl_2level_ideal = smq1Q.target_model(qubit_labels=["Qubit"])
```

```{code-cell} ipython3
rho0 = np.array( [[1,0,0],
                  [0,0,0],
                  [0,0,0]], complex)
E0 = rho0
E1 = np.array( [[0,0,0],
                [0,1,0],
                [0,0,1]], complex)

sslbls = pygsti.baseobjs.ExplicitStateSpace(['Qubit_leakage'],[3])
mdl_3level_ideal = pygsti.models.ExplicitOpModel(sslbls, 'gm', simulator='matrix')
mdl_3level_ideal['rho0'] = pygsti.tools.stdmx_to_gmvec(rho0)
mdl_3level_ideal['Mdefault'] = pygsti.modelmembers.povms.TPPOVM([('0',pygsti.tools.stdmx_to_gmvec(E0)),
                                                                 ('1',pygsti.tools.stdmx_to_gmvec(E1))],
                                                                evotype='default')

mdl_3level_ideal[tuple()] = unitary_to_gmgate( to_3level_unitary(Us['Gi']))
mdl_3level_ideal['Gxpi2', 'Qubit_leakage'] = unitary_to_gmgate( to_3level_unitary(Us['Gxpi2']))
mdl_3level_ideal['Gypi2', 'Qubit_leakage'] = unitary_to_gmgate( to_3level_unitary(Us['Gypi2']))
```

```{code-cell} ipython3
sigmaX = np.array([[0,1],[1,0]],complex)
rot = sla.expm(1j * 0.1 * sigmaX)
Uleakage = np.identity(3,complex)
Uleakage[1:3,1:3] = rot
leakageOp = unitary_to_gmgate(Uleakage)
#print(Uleakage)

#Guess of a model w/just unitary leakage
mdl_3level_guess = mdl_3level_ideal.copy()
mdl_3level_guess[tuple()] = np.dot(leakageOp, mdl_3level_guess[tuple()])
#mdl_3level_guess['Gxpi2', 'Qubit_leakage'] = np.dot(leakageOp, mdl_3level_guess['Gxpi2', 'Qubit_leakage'])
#mdl_3level_guess['Gypi2', 'Qubit_leakage'] = np.dot(leakageOp, mdl_3level_guess['Gypi2', 'Qubit_leakage'])

#Actual model used for data generation (some depolarization too)
mdl_3level_noisy = mdl_3level_ideal.depolarize(op_noise=0.005, spam_noise=0.01)
mdl_3level_noisy[tuple()] = np.dot(leakageOp, mdl_3level_noisy[tuple()])
#mdl_3level_noisy['Gxpi2', 'Qubit_leakage'] = np.dot(leakageOp, mdl_3level_noisy['Gxpi2', 'Qubit_leakage'])
#mdl_3level_noisy['Gypi2', 'Qubit_leakage'] = np.dot(leakageOp, mdl_3level_noisy['Gypi2', 'Qubit_leakage'])
```

```{code-cell} ipython3
#print(mdl_3level_guess)
```

```{code-cell} ipython3
# get sequences using expected model
find_fiducials = True

if find_fiducials:
    prepfids, measfids = pygsti.algorithms.find_fiducials(
        mdl_3level_guess, omit_identity=False, candidate_fid_counts={4: "all upto"}, verbosity=4)
    pygsti.io.write_circuit_list("../../example_files/leakage_prepfids.txt", prepfids)
    pygsti.io.write_circuit_list("../../example_files/leakage_measfids.txt", measfids)
```

```{code-cell} ipython3
# If files missing, run previous cell at least once with find_fiducials = True
prepfids = pygsti.io.read_circuit_list("../../example_files/leakage_prepfids.txt")
measfids = pygsti.io.read_circuit_list("../../example_files/leakage_measfids.txt")
germs = smq1Q.germs(qubit_labels=["Qubit_leakage"])
maxLengths = [1,]
expList = pygsti.circuits.create_lsgst_circuits(mdl_3level_noisy, prepfids, measfids, germs, maxLengths)
ds = pygsti.data.simulate_data(mdl_3level_noisy, expList, 1000, 'binomial', seed=1234)
```

```{code-cell} ipython3
# We have found out prep fids, meas fids, and germs, as well as simulated noisy data, for the 3 level model
# If we want to run GST on another model, we need to get versions of the circuits will the correct state space labels

def map_2level_sslbls(circuit):
    sslbl_map = {'Qubit_leakage': 'Qubit'}
    return circuit.map_state_space_labels(sslbl_map)

prepfids_2level = [map_2level_sslbls(c) for c in prepfids]
measfids_2level = [map_2level_sslbls(c) for c in measfids]
germs_2level = [map_2level_sslbls(c) for c in germs]
ds_2level = ds.process_circuits(map_2level_sslbls)

results_2level = pygsti.run_stdpractice_gst(ds_2level, mdl_2level_ideal, prepfids_2level, measfids_2level,
                                           germs_2level, maxLengths, modes="CPTPLND", verbosity=3)
```

```{code-cell} ipython3
pygsti.report.construct_standard_report(results_2level, "2-level Leakage Example Report").write_html('../../example_files/leakage_report_2level')
```

Open the report [here](../../example_files/leakage_report_2level/main.html)

```{code-cell} ipython3
results_3level = pygsti.run_stdpractice_gst(ds, mdl_3level_ideal, prepfids, measfids,
                                           germs, maxLengths, modes=["CPTPLND","True"],
                                           models_to_test={'True': mdl_3level_noisy}, 
                                           verbosity=4, advanced_options={'all': {'tolerance': 1e-2}})
```

```{code-cell} ipython3
pygsti.report.construct_standard_report(results_3level, "3-level Leakage Example Report").write_html('../../example_files/leakage_report')
```

Open the report [here](../../example_files/leakage_report/main.html)

```{code-cell} ipython3
#try a different basis:
gm_basis = pygsti.baseobjs.Basis.cast('gm',9)
   
leakage_basis_mxs = [ np.sqrt(2)/3*(np.sqrt(3)*gm_basis[0] + 0.5*np.sqrt(6)*gm_basis[8]),
                      gm_basis[1], gm_basis[4], gm_basis[7],
                     gm_basis[2], gm_basis[3], gm_basis[5], gm_basis[6],
                     1/3*(np.sqrt(3)*gm_basis[0] - np.sqrt(6)*gm_basis[8]) ]
#for mx in leakage_basis_mxs:
#    pygsti.tools.print_mx(mx)

check = np.zeros( (9,9), complex)
for i,m1 in enumerate(leakage_basis_mxs):
    for j,m2 in enumerate(leakage_basis_mxs):
        check[i,j] = np.trace(np.dot(m1,m2))
assert(np.allclose(check, np.identity(9,complex)))

leakage_basis = pygsti.baseobjs.ExplicitBasis(leakage_basis_mxs, name="LeakageBasis",  
                                        longname="2+1 level leakage basis", real=True,
                                        labels=['I','X','Y','Z','LX0','LX1','LY0','LY1','L'])

def changebasis_3level_model(mdl):
    new_mdl = mdl.copy()
    new_mdl.preps['rho0'] = pygsti.modelmembers.states.FullState(
        pygsti.tools.change_basis(mdl.preps['rho0'].to_dense(), gm_basis, leakage_basis))
    new_mdl.povms['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM(
        [('0', pygsti.tools.change_basis(mdl.povms['Mdefault']['0'].to_dense(), gm_basis, leakage_basis)),
         ('1', pygsti.tools.change_basis(mdl.povms['Mdefault']['1'].to_dense(), gm_basis, leakage_basis))],
        evotype='default')
    
    for lbl,op in mdl.operations.items():
        new_mdl.operations[lbl] = pygsti.modelmembers.operations.FullArbitraryOp(
            pygsti.tools.change_basis(op.to_dense(), gm_basis, leakage_basis))
    new_mdl.basis = leakage_basis
    return new_mdl

def changebasis_3level_results(results):
    new_results = results.copy()
    for estlbl,est in results.estimates.items():
        for mlbl,mdl in est.models.items():
            if isinstance(mdl,(list,tuple)): #assume a list/tuple of models
                new_results.estimates[estlbl].models[mlbl] = \
                    [ changebasis_3level_model(m) for m in mdl ]
            else:
                new_results.estimates[estlbl].models[mlbl] = changebasis_3level_model(mdl)
    return new_results
    
```

```{code-cell} ipython3
results_3level_leakage_basis = changebasis_3level_results( results_3level )    
```

```{code-cell} ipython3
pygsti.report.construct_standard_report(results_3level_leakage_basis, "3-level with Basis Change Leakage Example Report"
                                        ).write_html('../../example_files/leakage_report_basis')
```

Open the report [here](../../example_files/leakage_report_basis/main.html)

```{code-cell} ipython3
# use "kite" density-matrix structure
def to_2plus1_superop(superop_2level):
    ret = np.zeros((5,5),'d')
    ret[0:4,0:4] = superop_2level.to_dense()
    ret[4,4] = 1.0 #leave leakage population where it is
    return ret

#Tack on a single extra "0" for the 5-th dimension corresponding
# to the classical leakage level population.
eps = 0.01 # ideally zero, a smallish number to seed the GST optimiation away from 0-leakage so it doesn't get stuck there.
rho0 = np.concatenate( (mdl_2level_ideal.preps['rho0'].to_dense(),[eps]), axis=0)
E0 = np.concatenate( (mdl_2level_ideal.povms['Mdefault']['0'].to_dense(),[eps]), axis=0)
E1 = np.concatenate( (mdl_2level_ideal.povms['Mdefault']['1'].to_dense(),[eps]), axis=0)


statespace = pygsti.baseobjs.ExplicitStateSpace([('Qubit',),('Leakage',)], [(2,), (1,)])
mdl_2plus1_ideal = pygsti.models.ExplicitOpModel(statespace, 'gm', simulator='matrix')
mdl_2plus1_ideal['rho0'] = rho0
mdl_2plus1_ideal['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM([('0',E0),('1',E1)],
                                                                           evotype='default', state_space=statespace)

mdl_2plus1_ideal[tuple()] = to_2plus1_superop(mdl_2level_ideal[tuple()])
mdl_2plus1_ideal['Gxpi2'] = to_2plus1_superop(mdl_2level_ideal['Gxpi2', 'Qubit'])
mdl_2plus1_ideal['Gypi2'] = to_2plus1_superop(mdl_2level_ideal['Gypi2', 'Qubit'])
```

```{code-cell} ipython3
# We have found out prep fids, meas fids, and germs, as well as simulated noisy data, for the 3 level model
# If we want to run GST on another model, we need to get versions of the circuits will the correct state space labels

# We do this in a slightly different/awkward way here for this case since our state space labels are not a single entry
# This would not be necessary if we were rebuilding the circuits/dataset from scratch, only hacky since we are reusing the 3-level information
def map_2plus1_circuit_linelabels(circuit):
    return Circuit([Label(l.name) if l.name != "COMPOUND" else tuple() for l in circuit.layertup],
                   "*", None, not circuit._static)

prepfids_2plus1 = [map_2plus1_circuit_linelabels(c) for c in prepfids]
measfids_2plus1 = [map_2plus1_circuit_linelabels(c) for c in measfids]
germs_2plus1 = [map_2plus1_circuit_linelabels(c) for c in germs]
ds_2plus1 = ds.process_circuits(map_2plus1_circuit_linelabels)

results_2plus1 = pygsti.run_long_sequence_gst(ds_2plus1, mdl_2plus1_ideal, prepfids_2plus1, measfids_2plus1,
                                             germs_2plus1, maxLengths, verbosity=2,
                                             advanced_options={"starting_point": "target",
                                                               "tolerance": 1e-8,  # (lowering tolerance from 1e-6 gave a better fit)
                                                               "estimate_label": "kite"})
```

```{code-cell} ipython3
:tags: [nbval-skip]

# TODO: This is currently broken
pygsti.report.construct_standard_report(results_2plus1,"2+1 Leakage Example Report"
).write_html('../../example_files/leakage_report_2plus1', autosize='none')
```

Open the report [here](../../example_files/leakage_report/main.html)
