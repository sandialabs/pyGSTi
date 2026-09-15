#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Wall-clock scaling and algorithm-selection suite for edge coloring.

Every test in this module asserts on wall-clock behavior: each algorithm runs in
a subprocess under a hard timeout, and the tests assert that it finished inside
that budget. That makes them sensitive to whatever else the machine is doing --
on a busy box a healthy algorithm gets killed on the budget and the test reports
a failure that says nothing about the code. They live under ``test/performance/``
for that reason, and are not part of the ordinary unit suite.

The correctness tests for the same algorithms -- which run in-process, assert
nothing about time, and are safe under load -- stay in
``test/unit/tools/graphs/test_coloring.py``.

Run them on an otherwise-idle machine:

    pytest -s test/performance/test_coloring_scaling.py

``-s`` shows the comparison tables, which are the real deliverable of the
scaling suite. CI does not run ``test/performance/`` as a matter of course; it
runs this module only when a branch changes the graphs toolkit (see the
graphs steps in ``.github/workflows/reuseable-main.yml``).
"""

import multiprocessing as _mp
import os
import time as _time
import unittest

import pytest

from pygsti.tools.graphs.coloring import switchboard_find_edge_coloring

from ..helpers.coloring_graphs import (
    ALL_ALGORITHMS,
    DETERMINISTIC_EXACT_ALGORITHMS,
    SPARSE_SAFE,
    assess_coloring,
    make_complete_graph,
    make_cycle_graph,
    make_grid_graph,
    make_path_graph,
    make_random_regular_graph,
)


# GitHub Actions (like most CI providers) sets the ``CI`` environment variable
# automatically. This suite is meant to be run locally, so we skip it in CI via
# this environment-variable ``skipif`` -- the same pattern used elsewhere in the
# test suite (e.g. the ``PYGSTI_NO_CUSTOMLM_SIGINT`` check in
# test/unit/optimize/test_sigint.py and the ``needs_*`` helpers in
# test/unit/util.py). To also exclude it locally, use ``-m "not slow"``.
#
# However, CI (see .github/workflows/reuseable-main.yml) additionally sets
# ``GRAPHS_CHANGED=true`` when a push/PR touches pygsti/tools/graphs/,
# this module, or the shared graph fixtures, so that regressions there are still
# caught automatically instead of relying solely on local runs.
skip_in_ci = pytest.mark.skipif(
    'CI' in os.environ and os.environ.get('GRAPHS_CHANGED', '').lower() != 'true',
    reason="benchmark-style scaling test; run locally only, or in CI when "
           "pygsti/tools/graphs/ or this module has changed "
           "(see GRAPHS_CHANGED in reuseable-main.yml)")


# Small enough that even the good algorithms finish comfortably, large enough to
# expose the blow-ups. Kept modest so the whole suite runs in a few seconds.
PER_ALGO_TIMEOUT = 5.0


# Dense complete graphs used as regression guards for the deterministic
# (Vizing-chain) algorithms, which previously hung or left edges uncolored here.
DENSE_REGRESSION_GRAPHS = [
    ("K6", make_complete_graph(6)),
    ("K9", make_complete_graph(9)),
    ("K12", make_complete_graph(12)),
]


# ---------------------------------------------------------------------------
# Timeout-guarded runner. Some randomized algorithms can, on dense graphs, run
# for a very long time, so we run each attempt in a separate process and
# hard-kill it if it exceeds the budget. This keeps the scaling suite from ever
# hanging.
# ---------------------------------------------------------------------------
def _worker(algorithm_name, deg, vertices, edges, neighbors, seed, q):
    try:
        cp = switchboard_find_edge_coloring(algorithm_name, deg, vertices, edges, neighbors, seed=seed)
        q.put(("ok", cp))
    except Exception as ex:  # noqa: BLE001 -- we intentionally capture everything
        q.put(("error", "%s: %s" % (type(ex).__name__, ex)))


def run_with_timeout(algorithm_name, graph, seed=0, timeout=PER_ALGO_TIMEOUT):
    """Run one algorithm on one graph under a wall-clock budget.

    Returns a result dict with keys:
        status   : 'ok' | 'error' | 'timeout'
        seconds  : elapsed wall time (== timeout if it timed out)
        colors   : number of colors used (None unless status=='ok' and proper+complete)
        proper   : bool or None
        complete : bool or None
        detail   : error string or None
    """
    vertices, edges, neighbors, deg = graph
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(algorithm_name, deg, vertices, edges, neighbors, seed, q))
    start = _time.perf_counter()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return dict(status="timeout", seconds=timeout, colors=None,
                    proper=None, complete=None, detail="exceeded %.1fs" % timeout)
    elapsed = _time.perf_counter() - start
    try:
        kind, payload = q.get_nowait()
    except Exception:  # queue empty -> process died without reporting
        return dict(status="error", seconds=elapsed, colors=None,
                    proper=None, complete=None, detail="worker produced no result")
    if kind == "error":
        return dict(status="error", seconds=elapsed, colors=None,
                    proper=None, complete=None, detail=payload)
    proper, complete, ncolors = assess_coloring(payload, edges)
    return dict(status="ok", seconds=elapsed,
                colors=ncolors if (proper and complete) else None,
                proper=proper, complete=complete, detail=None)


def _print_table(title, graph, results):
    """Print a benchmark comparison table (visible with `pytest -s`)."""
    vertices, edges, neighbors, deg = graph
    print("\n" + "=" * 78)
    print(f"{title}  (|V|={len(vertices)}, |E|={len(edges)}, max_degree={deg})")
    print("-" * 78)
    print(f"{'algorithm':16s} {'status':9s} {'colors':>7s} {'time(ms)':>10s}  notes")
    for algo in ALL_ALGORITHMS:
        r = results[algo]
        colors = "-" if r["colors"] is None else str(r["colors"])
        note = ""
        if r["status"] == "ok" and not (r["proper"] and r["complete"]):
            note = f"INVALID (proper={r['proper']}, complete={r['complete']})"
        elif r["detail"]:
            note = r["detail"]
        print(f"{algo:16s} {r['status']:9s} {colors:>7s} {r['seconds'] * 1000:>10.1f}  {note}")
    print("=" * 78)


@skip_in_ci
@pytest.mark.slow
@pytest.mark.parametrize("name,graph", DENSE_REGRESSION_GRAPHS)
@pytest.mark.parametrize("algorithm", DETERMINISTIC_EXACT_ALGORITHMS)
def test_deterministic_algorithm_colors_dense_graph(algorithm, name, graph):
    """Regression guard: the deterministic (Vizing-chain) algorithms must
    terminate and produce a proper, complete, (deg+1)-color coloring on dense
    complete graphs.

    misra_gries used to raise a KeyError or hang on non-sparse graphs, and
    vizing's complex ("Vizing chain") case was an unimplemented placeholder that
    silently left edges uncolored. Both were fixed (vizing's chain now delegates
    to the canonical Misra-Gries procedure), so both are deterministic, always
    terminate, and are near-optimal even on complete graphs.

    This asserts termination inside ``PER_ALGO_TIMEOUT``, which is why it lives
    here rather than in the unit suite.
    """
    _vertices, _edges, _neighbors, deg = graph
    r = run_with_timeout(algorithm, graph, seed=0, timeout=PER_ALGO_TIMEOUT)
    assert r["status"] == "ok", \
        f"{algorithm} did not finish on {name}: {r['status']} ({r['detail']})"
    assert r["proper"] and r["complete"], \
        (f"{algorithm} produced an invalid coloring on {name} "
         f"(proper={r['proper']}, complete={r['complete']})")
    assert r["colors"] <= deg + 1, \
        f"{algorithm} used {r['colors']} colors on {name} (budget {deg + 1})"


# ---------------------------------------------------------------------------
# Scaling / algorithm-selection suite.
#
# These tests characterize *which algorithm to use in which situation*. They are
# not pure correctness tests (those live in test/unit/tools/test_graphcoloring.py);
# instead they run every algorithm across graph families and sizes, recording
# runtime, color count (quality), and whether the algorithm produced a proper+
# complete coloring at all. Findings are printed as a table (visible with
# `pytest -s`).
#
# Because some algorithms can loop for a very long time on dense graphs, every
# run is guarded by a hard wall-clock timeout in a separate process.
#
# Per-algorithm expectations, derived from measured behavior on this codebase:
#   - vizing        : deterministic; always terminates; proper+complete and
#                     near-optimal (<= deg+1 colors) on every family tested,
#                     including dense complete graphs and grids. It uses a greedy
#                     simple-case fast path and falls back to a Vizing-chain step
#                     (the Misra-Gries procedure) for the hard case.
#   - sinnamon      : Sinnamon (2019)'s deterministic Greedy-Euler-Color.
#                     Always terminates, proper+complete, and guaranteed to use
#                     at most 2*deg-1 colors -- but that budget is itself much
#                     looser than deg+1, so it uses noticeably more colors than
#                     the other algorithms here; avoid when color count matters.
#   - random_euler_color : Sinnamon (2019)'s randomized Random-Euler-Color.
#                     Seedable/reproducible; always terminates, proper+complete,
#                     and guaranteed to use at most deg+1 colors (matching
#                     vizing/misra_gries's quality) in expected O(m*sqrt(n)) time.
#   - misra_gries   : deterministic; always terminates; proper+complete and
#                     near-optimal (<= deg+1 colors) on every family tested,
#                     including dense complete graphs and grids. Fast and the most
#                     reliable algorithm here -- a good default.
#   - auto          : the recommended default algorithm. It checks for canonical
#                     topologies (and applies a fast closed-form optimal coloring),
#                     then falls back to bipartite-optimal randomized coloring (using
#                     seed) on bipartite graphs, and to `vizing` otherwise.
# ---------------------------------------------------------------------------
@skip_in_ci
@pytest.mark.slow
class GraphColoringScalingTester(unittest.TestCase):
    """Benchmark-style tests to guide algorithm selection.

    Each test builds a graph, runs every algorithm under a timeout, prints a
    comparison table, and asserts only the invariants we are confident about
    (namely: the algorithms in SPARSE_SAFE must succeed, terminate, and be
    near-optimal on the low-degree families). The slower or incomplete
    algorithms are recorded but not asserted on, so the table stays informative
    without making the suite flaky.
    """

    def _run_all(self, graph, timeout=PER_ALGO_TIMEOUT):
        return {algo: run_with_timeout(algo, graph, seed=0, timeout=timeout)
                for algo in ALL_ALGORITHMS}

    def _assert_timeout_guard_held(self, results, graph_name):
        """No algorithm may run unbounded: each finishes or is killed on budget."""
        for algo in ALL_ALGORITHMS:
            self.assertLessEqual(
                results[algo]["seconds"], PER_ALGO_TIMEOUT + 3.0,
                f"{algo} was not bounded by the timeout on {graph_name}")

    def _assert_sparse_safe_algos_good(self, graph, results, color_slack=2):
        """The sparse-safe algorithms must succeed, terminate, and be near-optimal.

        By Vizing's theorem a simple graph needs deg or deg+1 colors; we allow a
        small slack. This is asserted only for low-degree families, where these
        algorithms were observed to be reliable across runs.
        """
        _vertices, _edges, _neighbors, deg = graph
        budget = deg + 1 + color_slack
        for algo in SPARSE_SAFE:
            r = results[algo]
            self.assertEqual(r["status"], "ok",
                             f"{algo} did not finish: {r['status']} ({r['detail']})")
            self.assertTrue(
                r["proper"] and r["complete"],
                f"{algo} produced an invalid coloring (proper={r['proper']}, complete={r['complete']})")
            self.assertLessEqual(
                r["colors"], budget,
                f"{algo} used {r['colors']} colors on a max-degree-{deg} graph (budget {budget})")

    def _assert_at_least_one_valid(self, graph, results, graph_name):
        """Sanity floor: on any graph, *some* algorithm must produce a valid coloring."""
        valid = [a for a in ALL_ALGORITHMS
                 if results[a]["status"] == "ok"
                 and results[a]["proper"] and results[a]["complete"]]
        self.assertTrue(valid, f"No algorithm produced a proper+complete coloring on {graph_name}")

    def test_scaling_cycle_graphs(self):
        """Sparse (deg=2) family across increasing size."""
        for n in [10, 50, 200]:
            graph = make_cycle_graph(n)
            results = self._run_all(graph)
            _print_table(f"cycle C_{n}", graph, results)
            self._assert_timeout_guard_held(results, f"C_{n}")
            self._assert_sparse_safe_algos_good(graph, results)

    def test_scaling_path_graphs(self):
        """Sparse path family (deg<=2)."""
        for n in [10, 50, 200]:
            graph = make_path_graph(n)
            results = self._run_all(graph)
            _print_table(f"path P_{n}", graph, results)
            self._assert_timeout_guard_held(results, f"P_{n}")
            self._assert_sparse_safe_algos_good(graph, results)

    def test_scaling_grid_graphs(self):
        """2D lattice (deg<=4) -- representative of a planar QPU connectivity."""
        for rows, cols in [(3, 3), (5, 5), (8, 8)]:
            graph = make_grid_graph(rows, cols)
            results = self._run_all(graph)
            _print_table(f"grid {rows}x{cols}", graph, results)
            self._assert_timeout_guard_held(results, f"grid {rows}x{cols}")
            self._assert_sparse_safe_algos_good(graph, results)

    def test_scaling_random_regular_graphs(self):
        """Random ~d-regular graphs at moderate degree."""
        for n, d in [(12, 3), (20, 4), (30, 5)]:
            graph = make_random_regular_graph(n, d, seed=1234)
            results = self._run_all(graph)
            _print_table(f"random ~{d}-regular on {n} verts", graph, results)
            self._assert_timeout_guard_held(results, f"random ~{d}-regular n={n}")
            self._assert_at_least_one_valid(graph, results, f"random ~{d}-regular n={n}")

    def test_scaling_complete_graphs_dense_stress(self):
        """Dense stress test (K_n, deg=n-1).

        This is where the algorithms diverge most sharply: some hang, some fail
        to converge, some explode the color count. We record everything (the
        printed table is the deliverable) and assert only that the timeout guard
        protected the run and that at least one algorithm succeeded.
        """
        for n in [6, 10, 16]:
            graph = make_complete_graph(n)
            results = self._run_all(graph)
            _print_table(f"complete K_{n}", graph, results)
            self._assert_timeout_guard_held(results, f"K_{n}")
            self._assert_at_least_one_valid(graph, results, f"K_{n}")
