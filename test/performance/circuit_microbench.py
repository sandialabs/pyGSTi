#!/usr/bin/env python
"""Microbenchmarks for Circuit identity/creation hot paths.

Produces the before/after evidence table required for any PR touching circuit.py
performance paths (format precedent: PRs #445 / #692). Not run in CI — run
manually on a quiet machine, same interpreter, before and after a change.

Usage:
    python test/performance/circuit_microbench.py [--repeats 7] [--json OUT.json]

Reference numbers from this script (2026-06-10, dev container, linux/aarch64, py313,
production code identical to develop@3e7dd411e):
    depth-30 2Q __init__ check=True ~141us | check=False ~124us | _fastinit ~0.76us
    static copy ~0.25us | cached hash ~50ns | __add__ ~16us
    smq1Q_XYI ML=64 edesign: ~0.13s wall, Circuit.__add__ ~73% of wall
"""
import argparse
import cProfile
import json
import pstats
import statistics
import time
import timeit

from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.io import stdinput


def depth30_2q_layers():
    return [[('Gx', 0), ('Gy', 1)] if i % 2 else [('Gcnot', 0, 1)] for i in range(30)]


def _label_layers(layers):
    return tuple(Label(layer) for layer in layers)


def run_microbenches(repeats):
    layers = depth30_2q_layers()
    lbl_layers = _label_layers(layers)
    static_c = Circuit(layers, line_labels=(0, 1))
    half_a, half_b = static_c[0:15], static_c[15:30]
    cstr = static_c.str
    sip = stdinput.StdInputParser()
    sip.use_global_parse_cache = False  # otherwise the parse bench times a dict lookup

    benches = [
        ('init_check_true',     lambda: Circuit(layers, line_labels=(0, 1), check=True),                200   ),
        ('init_check_false',    lambda: Circuit(layers, line_labels=(0, 1), check=False),               500   ),
        ('fastinit',            lambda: Circuit._fastinit(lbl_layers, (0, 1), False),                   5000  ),
        ('static_copy',         lambda: static_c.copy(),                                                10000 ),
        ('cached_hash',         lambda: hash(static_c),                                                 100000),
        ('parse_via_init',      lambda: Circuit(cstr),                                                  200   ),
        ('parse_via_stdinput',  lambda: sip.parse_circuit(cstr, create_subcircuits=False),              500   ),
        ('add',                 lambda: half_a + half_b,                                                2000  ),
        ('layer_slice',         lambda: static_c[5:25],                                                 2000  ),
    ]
    results = {}
    for name, fn, number in benches:
        timer     = timeit.Timer(fn)
        raw_times = timer.repeat(repeat=repeats, number=number)
        per_call  = [t / number for t in raw_times]
        best_us   = min(per_call) * 1e6
        median_us = statistics.median(per_call) * 1e6
        results[name] = {'best_us': best_us, 'median_us': median_us, 'n': number, 'repeats': repeats}
    return results


def run_edesign_macro():
    """Wall time + __add__ share for a real experiment-design creation."""
    from pygsti.modelpacks import smq1Q_XYI
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    design = smq1Q_XYI.create_gst_experiment_design(64)
    profiler.disable()
    wall = time.perf_counter() - t0
    n_circuits = len(design.all_circuits_needing_data)

    stats = pstats.Stats(profiler)
    add_cumtime = 0.0
    for func_key, func_stats in stats.stats.items():
        filename, _line, funcname = func_key
        cumtime = func_stats[3]
        if funcname == '__add__' and filename.endswith('circuit.py'):
            add_cumtime += cumtime

    add_share = (add_cumtime / wall) if wall else 0.0
    macro = {
        'wall_s':                 wall,
        'n_circuits':             n_circuits,
        'circuit_add_cumtime_s':  add_cumtime,
        'circuit_add_share':      add_share,
    }
    return macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()

    micro = run_microbenches(args.repeats)
    macro = run_edesign_macro()

    print('\n| benchmark | best | median |')
    print('|---|---|---|')
    for name, r in micro.items():
        print(f"| {name} | {r['best_us']:.3f} us | {r['median_us']:.3f} us |")

    n_circuits  = macro['n_circuits']
    wall_s      = macro['wall_s']
    add_cum_s   = macro['circuit_add_cumtime_s']
    add_share   = macro['circuit_add_share']
    print(f"\nsmq1Q_XYI ML=64 edesign: {n_circuits} circuits in {wall_s:.2f}s")
    print(f"Circuit.__add__ cumtime: {add_cum_s:.2f}s ({add_share:.0%} of wall)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'micro': micro, 'macro': macro}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()
