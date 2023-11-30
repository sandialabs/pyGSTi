import numpy as np
from pygsti.baseobjs import Basis

np.set_printoptions(precision=1, linewidth=1000)

import pygsti
from pygsti.extras import idletomography as idt

from pygsti.extras.idletomography.pauliobjs import NQPauliState

from pygsti.extras.idletomography.idtcore import idle_tomography_fidpairs
n_qubits = 2

fid_pairs = idle_tomography_fidpairs(2)
print(fid_pairs)

gates = ["Gi", "Gx", "Gy", "Gcnot"]
max_lengths = [1, 2, 4, 8]

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

updated_ckt_list = []
for ckt in idle_experiments:
    new_ckt = ckt.copy(editable=True)
    for i, lbl in enumerate(ckt):
        if lbl == Label(()):
            new_ckt[i] = [Label(("Gi", i)) for i in range(n_qubits)]
            #new_ckt[i] = Label(("Gi", 0))
    updated_ckt_list.append(new_ckt)



mdl_datagen = pygsti.models.create_crosstalk_free_model(
    pspec, lindblad_error_coeffs={"Gi": {"SX": 0.01}}
)
# Error models! Random with right CP constraints from Taxonomy paper
ds = pygsti.data.simulate_data(
    mdl_datagen, updated_ckt_list, 100000, seed=8675309, sample_error="none"
)

results = idt.do_idle_tomography(
    n_qubits,
    ds,
    max_lengths,
    paulidicts,
    maxweight=1,
    advanced_options={"jacobian mode": "together", "pauli_fidpairs": fid_pairs},
    idle_string="[Gi:0Gi:1]",
)


print(results.observed_rate_infos)
print(results.intrinsic_rates)

idt.create_idletomography_report(
    results, "../IDTTestReport", "Test idle tomography example report", auto_open=True
)

results.error_list
# ws = pygsti.report.Workspace()
# ws.init_notebook_mode(autodisplay=True)
print(results)
