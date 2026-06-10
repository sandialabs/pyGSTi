"""Golden-circuit definitions, shared by generate_circuit_golden.py (one-time
fixture generator) and test_circuit_golden_fixtures.py (the loading test).

DO NOT casually edit: the committed binary fixtures under golden/ were built
from exactly these constructions. Editing a construction without regenerating
fixtures (a deliberate, reviewed act) will break the golden tests.
"""
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit


def build_golden_circuits():
    """Ordered dict of key -> static Circuit covering the identity-contract surface."""
    three_layers = [('Gx', 0), ('Gy', 0), ('Gz', 0)]

    d = {}
    d['simple_1q']        = Circuit('GxGxGy')
    d['explicit_lines']   = Circuit([('Gx', 0), ('Gy', 1)],                line_labels=(0, 1))
    d['parallel_layer']   = Circuit([[('Gx', 0), ('Gy', 1)], [('Gz', 0)]], line_labels=(0, 1))
    d['string_lines']     = Circuit('Gx:Q0Gy:Q1@(Q0,Q1)')
    d['occurrence']       = Circuit([('Gx', 0)], line_labels=(0,), occurrence=2)
    d['compilable_tilde'] = Circuit(three_layers, line_labels=(0,), compilable_layer_indices=(1,))
    d['compilable_pipe']  = Circuit(three_layers, line_labels=(0,), compilable_layer_indices=(0, 2))
    d['empty']            = Circuit('{}')
    d['empty_with_lines'] = Circuit('{}@(0,1)')
    d['idle_layer']       = Circuit([Label(())])
    d['subcircuit_label'] = Circuit('(GxGy)^2', expand_subcircuits=False)
    d['args_and_time']    = Circuit('Gx;theta:0!1.5@(0)')
    d['long_periodic']    = Circuit([('Gx', 0), ('Gy', 0)] * 15, line_labels=(0,))
    return d
