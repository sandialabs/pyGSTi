import numpy as np
from pygsti.baseobjs import Basis

np.set_printoptions(precision=4, linewidth=1000)

import pygsti
from pygsti.extras import idletomography as idt
from pygsti.modelmembers.operations import (
    LindbladErrorgen,
    ExpErrorgenOp,
    StaticArbitraryOp,
    ComposedOp,
)
from pygsti.baseobjs import QubitSpace, Label
from pygsti.circuits import Circuit
from pygsti.modelpacks import smq1Q_XYI

from pygsti.extras.idletomography.pauliobjs import NQPauliState

from pygsti.extras.idletomography.idtcore import idle_tomography_fidpairs

n_qubits = 2

fid_pairs = idle_tomography_fidpairs(n_qubits)
print(fid_pairs)
print(len(fid_pairs))

if n_qubits == 1:
    huh = [
        (NQPauliState("X", (1,)), NQPauliState("X", (1,))),
        (NQPauliState("X", (1,)), NQPauliState("Y", (1,))),
        (NQPauliState("X", (1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Y", (1,)), NQPauliState("X", (1,))),
        (NQPauliState("Y", (1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Y", (1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Z", (1,)), NQPauliState("X", (1,))),
        (NQPauliState("Z", (1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Z", (1,)), NQPauliState("Z", (1,))),
        (NQPauliState("X", (-1,)), NQPauliState("X", (1,))),
        (NQPauliState("X", (-1,)), NQPauliState("Y", (1,))),
        (NQPauliState("X", (-1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Y", (-1,)), NQPauliState("X", (1,))),
        (NQPauliState("Y", (-1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Y", (-1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Z", (-1,)), NQPauliState("X", (1,))),
        (NQPauliState("Z", (-1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Z", (-1,)), NQPauliState("Z", (1,))),
    ]
    fid_pairs = huh

gates = ["Gi", "Gx", "Gy", "Gcnot"]
max_lengths = [1, 2]

pspec = pygsti.processors.QubitProcessorSpec(
    n_qubits, gates, geometry="line", nonstd_gate_unitaries={(): 1}
)
mdl_target = pygsti.models.create_crosstalk_free_model(pspec)
paulidicts = idt.determine_paulidicts(mdl_target)

idle_experiments = idt.make_idle_tomography_list(
    n_qubits,
    max_lengths,
    paulidicts,
    maxweight=1,
    force_fid_pairs=fid_pairs,
)

print(len(idle_experiments), "idle tomography experiments for %d qubits" % n_qubits)
from pygsti.baseobjs import Label

if n_qubits > 1:
    updated_ckt_list = []
    for ckt in idle_experiments:
        new_ckt = ckt.copy(editable=True)
        for i, lbl in enumerate(ckt):
            if lbl == Label(()):
                new_ckt[i] = [Label(("Gi", i)) for i in range(n_qubits)]
                # new_ckt[i] = Label(("Gi", 0))
        updated_ckt_list.append(new_ckt)
else:
    updated_ckt_list = []
    for ckt in idle_experiments:
        new_ckt = ckt.copy(editable=True)

        for i, lbl in enumerate(ckt):
            if lbl == Label(()):
                new_ckt[i] = Label(("Gi", 0))
        updated_ckt_list.append(new_ckt)

err_str = "HX"
term_dict = {("H", "X"): 0.001}
# state_space = QubitSpace(n_qubits)
# test_error_gen = LindbladErrorgen.from_elementary_errorgens(term_dict, state_space=state_space, parameterization='GLND')
# test_error_gen.to_dense()
# test_exp_error_gen = ExpErrorgenOp(test_error_gen)
# ideal_idle = StaticArbitraryOp(np.eye(4))
# noisy_idle = ComposedOp([ideal_idle, test_exp_error_gen])
# noisy_idle.to_dense()
# noise_model = smq1Q_XYI.target_model()
# # del noise_model.operations['Gxpi2',0], noise_model.operations['Gypi2',0]
# Circuit.replace_gatename_inplace(noise_model.operations['Gypi2',0], "Gypi2", "Gy")
# Circuit.replace_gatename_inplace(noise_model.operations['Gxpi2',0], "Gxpi2", "Gx")
# noise_model.operations[Label(("Gi", 0))] = noisy_idle
# noise_model._rebuild_paramvec()

mdl_datagen = pygsti.models.create_crosstalk_free_model(
    pspec, lindblad_error_coeffs={"Gi": term_dict}, lindblad_parameterization="GLND"
)
# Error models! Random with right CP constraints from Taxonomy paper
ds = pygsti.data.simulate_data(
    mdl_datagen, updated_ckt_list, 1, seed=8675309, sample_error="none"
)

if n_qubits == 2:
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        maxweight=1,
        advanced_options={"jacobian mode": "together", "pauli_fidpairs": fid_pairs},
        idle_string="[Gi:0Gi:1]",
    )
else:
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        maxweight=1,
        advanced_options={"jacobian mode": "together", "pauli_fidpairs": fid_pairs},
        idle_string="Gi:0",
    )


# print(f'{results.observed_rate_infos=}')
# print(f'{results.intrinsic_rates=}')

output_str = "../1qTestReports/" + err_str
name_str = "Test idle tomography example report: 1q, " + err_str

idt.create_idletomography_report(results, output_str, name_str, auto_open=True)

results.error_list
# ws = pygsti.report.Workspace()
# ws.init_notebook_mode(autodisplay=True)
# print(results)
