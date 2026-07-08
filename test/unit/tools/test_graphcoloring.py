
import multiprocessing as _mp
import time as _time

import pytest
from pygsti.tools.graphcoloring import switchboard_find_edge_coloring, check_valid_edge_coloring, find_edge_coloring
import numpy as np


# ALGORITHMS = ["new_bipartite", "assadi", "vizing", "sinnamon", "moser_tardos", "misra-gries"]
# ALGORITHMS = ["sinnamon", "misra_gries"]
ALGORITHMS = ["new_bipartite", "vizing", "sinnamon", "misra_gries"]

# Every edge-coloring algorithm exposed by the switchboard. This is deliberately
# the *full* list (not the curated ALGORITHMS above) because the scaling suite's
# whole point is to characterize which algorithms are usable in which regime --
# including the ones that are known to be slow, suboptimal, or outright broken on
# certain graph families.
ALL_ALGORITHMS = ["new_bipartite", "vizing", "sinnamon", "misra_gries", "moser_tardos", "assadi"]


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------
def _finalize(vertices, edges):
    """Build the (vertices, edges, neighbors, deg) 4-tuple the API expects.

    ``edges`` is a list of undirected ``(u, v)`` pairs (each listed once). The
    neighbor lists are symmetric, and ``deg`` is the true maximum degree.
    """
    neighbors = {v: [] for v in vertices}
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    deg = max((len(neighbors[v]) for v in vertices), default=0)
    return list(vertices), list(edges), neighbors, deg


def make_cycle_graph(n):
    """Cycle C_n: n vertices, n edges, max degree 2."""
    return _finalize(range(n), [(i, (i + 1) % n) for i in range(n)])


def make_path_graph(n):
    """Path P_n: n vertices, n-1 edges, max degree 2."""
    return _finalize(range(n), [(i, i + 1) for i in range(n - 1)])


def make_complete_graph(n):
    """Complete graph K_n: max degree n-1, the hardest dense case."""
    return _finalize(range(n), [(i, j) for i in range(n) for j in range(i + 1, n)])


def make_grid_graph(rows, cols):
    """2D lattice (rows x cols), max degree 4 -- models a planar QPU layout."""
    def idx(r, c):
        return r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return _finalize(range(rows * cols), edges)


def make_random_regular_graph(n, d, seed):
    """(Approximately) d-regular random graph on n vertices via a pairing model.

    Falls back to whatever simple graph the pairing produces; the returned
    ``deg`` is the realized maximum degree, which is <= d.
    """
    rng = np.random.default_rng(seed)
    stubs = []
    for v in range(n):
        stubs.extend([v] * d)
    edge_set = set()
    for _ in range(10):
        rng.shuffle(stubs)
        edge_set.clear()
        ok = True
        for i in range(0, len(stubs) - 1, 2):
            u, w = stubs[i], stubs[i + 1]
            if u == w or (min(u, w), max(u, w)) in edge_set:
                ok = False
                break
            edge_set.add((min(u, w), max(u, w)))
        if ok:
            break
    return _finalize(range(n), sorted(edge_set))


# ---------------------------------------------------------------------------
# Coloring-quality checks (stronger than check_valid_edge_coloring, which only
# verifies each color class is a matching -- it does NOT verify completeness).
# ---------------------------------------------------------------------------
def assess_coloring(color_patches, edges):
    """Return (is_proper, is_complete, num_colors) for a coloring.

    * proper   : no edge appears twice AND no two edges sharing a vertex share a color
    * complete : every input edge received exactly one color
    """
    all_edges = {tuple(sorted(e)) for e in edges}
    seen = set()
    is_proper = True
    for _color, patch in color_patches.items():
        touched = set()
        for u, v in patch:
            e = tuple(sorted((u, v)))
            if e in seen:
                is_proper = False  # edge colored more than once
            seen.add(e)
            if u in touched or v in touched:
                is_proper = False  # adjacent edges share this color
            touched.add(u)
            touched.add(v)
    is_complete = (seen == all_edges)
    return is_proper, is_complete, len(color_patches)


# ---------------------------------------------------------------------------
# Timeout-guarded runner. Some algorithms (notably misra_gries on dense graphs)
# can loop effectively forever, so we run each attempt in a separate process and
# hard-kill it if it exceeds the budget. This keeps the scaling suite from ever
# hanging CI.
# ---------------------------------------------------------------------------
def _worker(algorithm_name, deg, vertices, edges, neighbors, seed, q):
    try:
        cp = switchboard_find_edge_coloring(algorithm_name, deg, vertices, edges, neighbors, seed=seed)
        q.put(("ok", cp))
    except Exception as ex:  # noqa: BLE001 -- we intentionally capture everything
        q.put(("error", "%s: %s" % (type(ex).__name__, ex)))


def run_with_timeout(algorithm_name, graph, seed=0, timeout=5.0):
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

class TestGraphColoring(object):

    def test_find_edge_coloring_cycle_graph(self):
        # Define a cycle graph with 10 vertices
        num_vertices = 10
        vertices = list(range(num_vertices))
        edges = []
        neighbors = {i: [] for i in vertices}

        for i in range(num_vertices):
            u, v = i, (i + 1) % num_vertices
            edges.append((u, v))
            neighbors[u].append(v)
            neighbors[v].append(u)
        
        # Max degree for a cycle graph is 2
        deg = 2

        color_patches = find_edge_coloring(deg, vertices, edges, neighbors)

        # 1. Verify that each edge is colored exactly once
        # Collect all colored edges from color_patches
        colored_edges_from_patches = []
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                colored_edges_from_patches.append(tuple(sorted((u, v))))

        # Original unique edges (non-symmetric)
        original_unique_edges = set()
        for u, v in edges:
            original_unique_edges.add(tuple(sorted((u, v))))
        
        assert len(colored_edges_from_patches) == len(original_unique_edges), "Not all edges were colored or some were colored multiple times."
        assert len(set(colored_edges_from_patches)) == len(original_unique_edges), "Some edges were colored multiple times."

        # 2. Verify that no two adjacent edges have the same color
        # This means, for any vertex, all edges incident to it must have different colors.
        
        # Reconstruct edge_colors for easy lookup
        edge_colors = {}
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                edge_colors[tuple(sorted((u, v)))] = color
        
        for vertex in vertices:
            incident_edges = []
            for neighbor in neighbors[vertex]:
                incident_edges.append(tuple(sorted((vertex, neighbor))))
            
            # Get colors of incident edges
            incident_edge_colors = []
            for edge in incident_edges:
                # Some edges might not be present in edge_colors if they were implicitly handled
                # by symmetric pairs. We need to handle this.
                if edge in edge_colors:
                    incident_edge_colors.append(edge_colors[edge])
            
            # Check for duplicate colors among incident edges
            assert len(incident_edge_colors) == len(set(incident_edge_colors)), f"Vertex {vertex} has adjacent edges with the same color."

        # 3. Use the internal check_valid_edge_coloring for a quick verification
        check_valid_edge_coloring(color_patches)

        print(f"Edge coloring for cycle graph with {num_vertices} vertices passed all checks.")

    def test_find_edge_coloring_path_graph(self):
        # Define a path graph with 10 vertices
        num_vertices = 10
        vertices = list(range(num_vertices))
        edges = []
        neighbors = {i: [] for i in vertices}

        for i in range(num_vertices - 1):
            u, v = i, i + 1
            edges.append((u, v))
            neighbors[u].append(v)
            neighbors[v].append(u)

        # Max degree for a path graph is 2 (for internal vertices)
        # End vertices have degree 1
        deg = 2

        color_patches = find_edge_coloring(deg, vertices, edges, neighbors)

        # 1. Verify that each edge is colored exactly once
        colored_edges_from_patches = []
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                colored_edges_from_patches.append(tuple(sorted((u, v))))

        original_unique_edges = set()
        for u, v in edges:
            original_unique_edges.add(tuple(sorted((u, v))))

        assert len(colored_edges_from_patches) == len(original_unique_edges), "Not all edges were colored or some were colored multiple times."
        assert len(set(colored_edges_from_patches)) == len(original_unique_edges), "Some edges were colored multiple times."

        # 2. Verify that no two adjacent edges have the same color
        edge_colors = {}
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                edge_colors[tuple(sorted((u, v)))] = color

        for vertex in vertices:
            incident_edges = []
            for neighbor in neighbors[vertex]:
                incident_edges.append(tuple(sorted((vertex, neighbor))))

            incident_edge_colors = []
            for edge in incident_edges:
                if edge in edge_colors:
                    incident_edge_colors.append(edge_colors[edge])

            print(incident_edge_colors)
            assert len(incident_edge_colors) == len(set(incident_edge_colors)), f"Vertex {vertex} has adjacent edges with the same color."

        # 3. Use the internal check_valid_edge_coloring for a quick verification
        check_valid_edge_coloring(color_patches)

        print(f"Edge coloring for path graph with {num_vertices} vertices passed all checks.")

    def test_find_edge_coloring_high_degree_graph(self):
        # Define a graph with 10 vertices and a max degree of 5
        num_vertices = 10
        vertices = list(range(num_vertices))
        edges = []
        neighbors = {i: [] for i in vertices}

        # Connect vertex 0 to 5 other vertices to ensure a high degree
        high_degree_vertex = 0
        for i in range(1, 6):
            u, v = high_degree_vertex, i
            edges.append((u, v))
            neighbors[u].append(v)
            neighbors[v].append(u)
        
        # Add some more edges to other vertices
        edges_to_add = [
            (1, 6), (6, 1),
            (2, 7), (7, 2),
            (3, 8), (8, 3),
            (4, 9), (9, 4),
        ]
        for u, v in edges_to_add:
            if (u, v) not in edges and (v, u) not in edges: # Avoid duplicate edges
                edges.append((u, v))
                neighbors[u].append(v)
                neighbors[v].append(u)

        # Calculate the maximum degree dynamically
        deg = max(len(neighbors[v]) for v in vertices)

        color_patches = find_edge_coloring(deg, vertices, edges, neighbors)

        # 1. Verify that each edge is colored exactly once
        colored_edges_from_patches = []
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                colored_edges_from_patches.append(tuple(sorted((u, v))))

        original_unique_edges = set()
        for u, v in edges:
            original_unique_edges.add(tuple(sorted((u, v))))

        assert len(colored_edges_from_patches) == len(original_unique_edges), "Not all edges were colored or some were colored multiple times."
        assert len(set(colored_edges_from_patches)) == len(original_unique_edges), "Some edges were colored multiple times."

        # 2. Verify that no two adjacent edges have the same color
        edge_colors = {}
        for color, patch_edges in color_patches.items():
            for u, v in patch_edges:
                edge_colors[tuple(sorted((u, v)))] = color

        for vertex in vertices:
            incident_edges = []
            for neighbor in neighbors[vertex]:
                incident_edges.append(tuple(sorted((vertex, neighbor))))

            incident_edge_colors = []
            for edge in incident_edges:
                if edge in edge_colors:
                    incident_edge_colors.append(edge_colors[edge])

            assert len(incident_edge_colors) == len(set(incident_edge_colors)), f"Vertex {vertex} has adjacent edges with the same color. \n {incident_edge_colors} \n {set(incident_edge_colors)}"

        # 3. Use the internal check_valid_edge_coloring for a quick verification
        check_valid_edge_coloring(color_patches)

        print(f"Edge coloring for high degree graph with {num_vertices} vertices and max degree {deg} passed all checks.")


# ---------------------------------------------------------------------------
# Scaling / algorithm-selection suite.
#
# These tests characterize *which algorithm to use in which situation*. They are
# not pure correctness tests (those live in TestGraphColoring above); instead
# they run every algorithm across graph families and sizes, recording runtime,
# color count (quality), and whether the algorithm produced a proper+complete
# coloring at all. Findings are printed as a table (visible with `pytest -s`).
#
# Because some algorithms can loop for a very long time on dense graphs, every
# run is guarded by a hard wall-clock timeout in a separate process.
#
# The suite is marked `slow` so the default fast unit run can skip it via
# `-m "not slow"`; run it explicitly with `pytest -m slow -s`.
# ---------------------------------------------------------------------------

# Per-algorithm expectations, derived from measured behavior on this codebase.
# These notes are the whole point of the suite -- they tell a caller which
# algorithm to reach for in which regime. They are intentionally conservative:
# several algorithms rely on Python's global `random` module internally, so their
# output (and even whether the coloring is *complete*) can vary run-to-run. We
# therefore assert only on invariants that held across repeated runs, and treat
# color-count / completeness as *reported* metrics rather than hard guarantees.
#
#   - vizing        : deterministic; always terminates; proper+complete and
#                     near-optimal (<= deg+1 colors) on every family tested,
#                     including dense complete graphs and grids. It uses a greedy
#                     simple-case fast path and falls back to a Vizing-chain step
#                     for the hard case. (That chain step is the Misra-Gries
#                     procedure -- the standard formalization of Vizing's chain --
#                     which vizing reuses; previously the chain case was an
#                     unimplemented placeholder that silently left edges
#                     uncolored on dense graphs.)
#   - new_bipartite : fast, but its core popularization step is unimplemented, so
#                     it leaves edges uncolored on dense/grid graphs -- only
#                     trust it after verifying completeness.
#   - sinnamon      : always terminates and is proper, but uses many extra
#                     colors (poor quality); avoid when color count matters.
#   - misra_gries   : deterministic; always terminates; proper+complete and
#                     near-optimal (<= deg+1 colors, per Vizing) on every family
#                     tested, including dense complete graphs and grids. Fast and
#                     the most reliable algorithm here -- a good default. (It was
#                     previously buggy: it could raise a KeyError or hang, caused
#                     by drifting free-color bookkeeping and unordered edge keys;
#                     it has since been reimplemented to derive missing colors
#                     directly from the current edge coloring.)
#   - moser_tardos  : randomized, now seedable (numpy Generator via `seed`), and
#                     reproducible for a fixed seed. Fixed to always terminate:
#                     it uses deg+1 colors and a bounded resample budget, then
#                     retries and finally raises rather than looping forever.
#                     Produces proper+complete near-optimal colorings on sparse
#                     and moderate graphs; still times out on large dense graphs
#                     (e.g. K_10) where the resampling cannot converge in budget.
#   - assadi        : randomized, now seedable (numpy Generator via `seed`) and
#                     reproducible for a fixed seed. Fixed to always terminate
#                     (bounded retry budget on the underlying line-graph vertex
#                     colorer, then raises). Proper+complete and near-optimal on
#                     sparse/moderate graphs, but the local-search solver is slow
#                     at high degree, so it can time out as degree grows (e.g. on
#                     complete graphs, and even large grids under a tight budget).
#
# Reproducibility: the four randomized algorithms (moser_tardos, assadi,
# sinnamon, and find_edge_coloring) all draw from a numpy Generator seeded via
# the `seed` argument, so passing the same seed yields identical colorings. See
# TestGraphColoringReproducibility for the invariants this suite relies on.
#
# "SPARSE_SAFE" = algorithms that reliably produced a proper & complete coloring
# on the low-degree (deg<=4) families in this suite, across runs and within the
# per-algorithm timeout. `vizing` and `misra_gries` qualify (both deterministic,
# fast, and near-optimal here; misra_gries is in fact reliable on *dense* graphs
# too). moser_tardos and assadi are now correct and terminate, but at the suite's
# larger low-degree sizes (e.g. the 8x8 grid) they can exceed the timeout, so
# they are recorded but not asserted on.
SPARSE_SAFE = ["vizing", "misra_gries"]

# Small enough that even the good algorithms finish comfortably, large enough to
# expose the blow-ups. Kept modest so the whole suite runs in a few seconds.
PER_ALGO_TIMEOUT = 5.0


def _print_table(title, graph, results):
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
        print(f"{algo:16s} {r['status']:9s} {colors:>7s} {r['seconds']*1000:>10.1f}  {note}")
    print("=" * 78)


@pytest.mark.slow
class TestGraphColoringScaling(object):
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
            assert results[algo]["seconds"] <= PER_ALGO_TIMEOUT + 3.0, \
                f"{algo} was not bounded by the timeout on {graph_name}"

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
            assert r["status"] == "ok", f"{algo} did not finish: {r['status']} ({r['detail']})"
            assert r["proper"] and r["complete"], \
                f"{algo} produced an invalid coloring (proper={r['proper']}, complete={r['complete']})"
            assert r["colors"] <= budget, \
                f"{algo} used {r['colors']} colors on a max-degree-{deg} graph (budget {budget})"

    def _assert_at_least_one_valid(self, graph, results, graph_name):
        """Sanity floor: on any graph, *some* algorithm must produce a valid coloring."""
        valid = [a for a in ALL_ALGORITHMS
                 if results[a]["status"] == "ok"
                 and results[a]["proper"] and results[a]["complete"]]
        assert valid, f"No algorithm produced a proper+complete coloring on {graph_name}"

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

    def test_misra_gries_colors_dense_graphs(self):
        """Regression guard: misra_gries must terminate and produce a proper,
        complete, (deg+1)-color coloring on dense complete graphs.

        Earlier this implementation would raise a KeyError or hang on non-sparse
        graphs. It has since been reimplemented to follow the canonical
        Misra-Gries algorithm (deriving missing colors from the current edge
        coloring), so it is now deterministic, always terminates, and is
        near-optimal even on complete graphs. This test pins that behavior.
        """
        for n in [6, 9, 12]:
            graph = make_complete_graph(n)
            _vertices, _edges, _neighbors, deg = graph
            r = run_with_timeout("misra_gries", graph, seed=0, timeout=5.0)
            assert r["status"] == "ok", \
                f"misra_gries did not finish on K_{n}: {r['status']} ({r['detail']})"
            assert r["proper"] and r["complete"], \
                (f"misra_gries produced an invalid coloring on K_{n} "
                 f"(proper={r['proper']}, complete={r['complete']})")
            assert r["colors"] <= deg + 1, \
                f"misra_gries used {r['colors']} colors on K_{n} (budget {deg + 1})"

    def test_vizing_colors_dense_graphs(self):
        """Regression guard: vizing must terminate and produce a proper,
        complete, (deg+1)-color coloring on dense complete graphs.

        vizing's complex ("Vizing chain") case used to be an unimplemented
        placeholder, so it silently left edges uncolored on graphs where its
        greedy simple case did not suffice (e.g. complete graphs). The chain
        case now delegates to the Misra-Gries procedure (the standard
        formalization of Vizing's chain), so vizing is deterministic, always
        terminates, and is near-optimal even on complete graphs. This test pins
        that behavior.
        """
        for n in [6, 9, 12]:
            graph = make_complete_graph(n)
            _vertices, _edges, _neighbors, deg = graph
            r = run_with_timeout("vizing", graph, seed=0, timeout=5.0)
            assert r["status"] == "ok", \
                f"vizing did not finish on K_{n}: {r['status']} ({r['detail']})"
            assert r["proper"] and r["complete"], \
                (f"vizing produced an invalid coloring on K_{n} "
                 f"(proper={r['proper']}, complete={r['complete']})")
            assert r["colors"] <= deg + 1, \
                f"vizing used {r['colors']} colors on K_{n} (budget {deg + 1})"


# ---------------------------------------------------------------------------
# Seeded-reproducibility suite for the randomized edge-coloring algorithms.
#
# The randomized algorithms now draw from a numpy Generator seeded via the
# `seed` argument, so that:
#   * passing the same integer seed yields identical colorings (reproducible),
#   * they still produce proper AND complete colorings, and
#   * a numpy.random.Generator object is accepted directly.
#
# These graphs are small and sparse enough that the fixed randomized algorithms
# terminate quickly, so the tests run in-process (no timeout guard needed).
# ---------------------------------------------------------------------------

# Randomized algorithms that were fixed to terminate and produce valid colorings.
RANDOMIZED_ALGORITHMS = ["assadi", "moser_tardos", "sinnamon"]


def _canonical_coloring(color_patches):
    """A hashable, order-independent representation of a coloring for equality.

    Two colorings compare equal iff they assign the same set of edges to the
    same set of colors (edge order within a color and dict order are ignored).
    """
    return tuple(sorted(
        (color, tuple(sorted(tuple(sorted(e)) for e in patch)))
        for color, patch in color_patches.items() if patch
    ))


class TestGraphColoringReproducibility(object):
    """Verify seed-controlled reproducibility of the randomized algorithms."""

    GRAPHS = [
        ("cycle_C6", make_cycle_graph(6)),
        ("path_P8", make_path_graph(8)),
        ("grid_3x3", make_grid_graph(3, 3)),
        ("random_3reg_n12", make_random_regular_graph(12, 3, seed=1234)),
    ]

    @pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
    def test_same_seed_is_reproducible(self, algorithm):
        """Same integer seed => byte-for-byte identical coloring."""
        for name, graph in self.GRAPHS:
            vertices, edges, neighbors, deg = graph
            r1 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
            r2 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
            assert _canonical_coloring(r1) == _canonical_coloring(r2), \
                f"{algorithm} not reproducible with a fixed seed on {name}"

    @pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
    def test_seeded_output_is_proper_and_complete(self, algorithm):
        """A seeded run must still be a proper, complete edge coloring."""
        for name, graph in self.GRAPHS:
            vertices, edges, neighbors, deg = graph
            cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
            proper, complete, _ncolors = assess_coloring(cp, edges)
            assert proper, f"{algorithm} produced an improper coloring on {name}"
            assert complete, f"{algorithm} left an edge uncolored on {name}"
            # check_valid_edge_coloring raises on an improper coloring.
            assert check_valid_edge_coloring(cp, ret_false_on_error=True), \
                f"{algorithm} failed check_valid_edge_coloring on {name}"

    @pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
    def test_accepts_generator_object(self, algorithm):
        """A numpy.random.Generator may be passed directly as the seed, and two
        generators created from the same seed reproduce each other."""
        name, graph = self.GRAPHS[0]
        vertices, edges, neighbors, deg = graph
        r1 = switchboard_find_edge_coloring(
            algorithm, deg, vertices, edges, neighbors, seed=np.random.default_rng(7))
        r2 = switchboard_find_edge_coloring(
            algorithm, deg, vertices, edges, neighbors, seed=np.random.default_rng(7))
        assert _canonical_coloring(r1) == _canonical_coloring(r2), \
            f"{algorithm} not reproducible from equivalent Generator objects on {name}"

    @pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
    def test_no_seed_still_valid(self, algorithm):
        """Omitting the seed still yields a proper, complete coloring."""
        name, graph = self.GRAPHS[0]
        vertices, edges, neighbors, deg = graph
        cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors)
        proper, complete, _ncolors = assess_coloring(cp, edges)
        assert proper and complete, \
            f"{algorithm} produced an invalid coloring without a seed on {name}"

    def test_different_seeds_can_differ(self):
        """Different seeds should generally produce different colorings.

        This is a soft check: for a sufficiently non-trivial graph at least one
        randomized algorithm must yield distinct output for two different seeds.
        (Individual algorithms may coincidentally match on small graphs, so we
        only require that *some* algorithm distinguishes the seeds.)
        """
        vertices, edges, neighbors, deg = make_grid_graph(4, 4)
        any_differs = False
        for algorithm in RANDOMIZED_ALGORITHMS:
            a = _canonical_coloring(
                switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=1))
            b = _canonical_coloring(
                switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=999))
            if a != b:
                any_differs = True
        assert any_differs, "no randomized algorithm distinguished two different seeds"

    def test_find_edge_coloring_is_reproducible(self):
        """The networkx-based find_edge_coloring is also seedable/reproducible."""
        vertices, edges, neighbors, deg = make_grid_graph(4, 4)
        r1 = find_edge_coloring(deg, vertices, edges, neighbors, seed=123)
        r2 = find_edge_coloring(deg, vertices, edges, neighbors, seed=123)
        assert _canonical_coloring(r1) == _canonical_coloring(r2), \
            "find_edge_coloring not reproducible with a fixed seed"
        proper, complete, _ = assess_coloring(r1, edges)
        assert proper and complete
