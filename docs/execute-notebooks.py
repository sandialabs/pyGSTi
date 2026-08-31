#!/usr/bin/env python
"""Execute the jupytext-paired ``.ipynb`` under ``docs/markdown`` in place.

Run from the repository root, after materializing the pairs and the shared
fixtures the cross-notebook inputs depend on::

    jupytext --sync 'docs/markdown/**/*.md'
    python -c "import sys; sys.path.insert(0,'docs'); import conftest; conftest._generate_shared_fixtures()"
    python docs/execute-notebooks.py --jobs 4 --timeout 1800 --allow-errors

``EXCLUDE`` lists notebooks that ship deliberately without output (none today),
overridable with ``--exclude``. Excluded notebooks are still normalized, because
they are committed too.

Why this rather than ``jupyter nbconvert --execute --inplace`` directly:

* **cwd.** The tutorials use relative paths like ``../../tutorial_files/foo``, so
  every notebook must run with its own directory as the working directory. This
  driver sets ``resources={"metadata": {"path": <notebook's dir>}}`` explicitly
  rather than relying on nbconvert's implicit behaviour.
* **One failure must not lose the rest.** A single ``nbconvert`` invocation over
  many notebooks aborts at the first failing one and writes *nothing* -- not even
  for the notebooks that already succeeded. Here each notebook is independent and
  its partial outputs are always persisted, so a broken notebook costs you that
  notebook's tail, not the whole run.
* **Timeouts.** ``jupytext --execute`` hardcodes ``timeout=None`` and exposes no
  flag, so a wedged cell runs until the CI job's own wall-clock limit. This
  driver takes a per-cell timeout and (with ``--allow-errors``) turns a timeout
  into a recoverable ``KeyboardInterrupt`` cell output.
* **Error policy is decoupled from output recording.** ``nbconvert
  --allow-errors`` exits 0, so recording a traceback and failing the job are
  mutually exclusive there. Here ``--allow-errors`` controls only whether
  execution *continues*; the exit status always reflects whether any cell errored.
* **No new dependency.** ``.[docs]`` pulls in ``nbclient`` and ``ipykernel``
  (via ``jupyter-book`` -> ``myst-nb``) but **not** ``nbconvert``.

Normalization: executed notebooks are rewritten to strip run-to-run noise that
would otherwise churn the committed diff -- see ``normalize()`` and
``path_scrubs()``. Measured over ten full runs (2026-08-20): 48 of 76 notebooks
come out byte-identical across two independent cold runs. The remaining 28 differ
for reasons outside this driver's reach -- unseeded randomness inside pyGSTi
reached through call sites the tutorials do not control, and one page whose MPI
rank output interleaves nondeterministically. That was measured and accepted
rather than chased: an independently re-executed release costs about 2.4 MB of
packed git history against 0 for a byte-identical one.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import os as _os
import pathlib
import re
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

# --------------------------------------------------------------------------
# Notebooks that read or write the shared docs/tutorial_files and
# docs/example_files trees. They run as one serial chain, in the order a reader
# would run them (producers before consumers); everything else runs in parallel.
#
# Derived from a static scan of the ../..[/..]/tutorial_files and
# ../..[/..]/example_files references in docs/markdown. Re-derive after adding a
# notebook that touches either tree.
# --------------------------------------------------------------------------
SERIAL_CHAIN = [
    "guides/workflow/DataSets",        # W Example_Dataset{,_LowCnts}.txt, Example_GST_Data
    "guides/gst/RunningGST",           # W Example_GST_Data (+ results)
    "guides/analysis/Results",         # R Example_GST_Data/results
    "advanced/extending/LowLevelGST",  # R Example_Dataset{,_LowCnts}.txt
    "guides/analysis/Reports",         # R Example_Dataset.txt
    "start/FirstGST",                  # W gettingStartedReport, test_gst_dir
    "guides/workflow/Workflow",        # W gettingStartedReport, test_gst_dir, test_rb_dir
    "guides/rb/HowRBWorks",            # W test_rb_dir
]

# --------------------------------------------------------------------------
# Notebooks to ship deliberately without output, as paths under docs/markdown
# with no .ipynb suffix. Empty today: Switchboards and ComparingDataSets sat
# here for a while -- their ColorBoxPlot hover text executes to 23 MB and
# 9.5 MB of output respectively -- but interactive figures are those pages'
# entire subject, so they now execute like everything else.
#
# This list is the *only* mechanism that can exclude a page.
# `nb_execution_excludepatterns` is a myst-nb key and takes effect only when
# myst-nb executes; the hosted build sets `execute_notebooks: 'off'` and
# execution happens here instead, so that key would be inert.
# --------------------------------------------------------------------------
EXCLUDE = []

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Any CPython default repr ending in " at 0x...>". Deliberately broad: the
# earlier `<[\w.]+ (?:object )?at ` form missed qualified names such as
# `<generator object DataSet.keys at 0x...>` and `<bound method M of <...>>`.
ADDR = re.compile(r"(<[^<>\n]*? at )0x[0-9a-fA-F]+(>)")
UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                  r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
# pygsti.report.workspace.random_id() == str(int(1e6 * random.random())), used as
# the DOM id of every plot / table / switchboard that a Workspace renders.
PYGSTI_ID = re.compile(r"(plot|table|switchboard)_(\d{1,7})(?![0-9])")
# tqdm's elapsed/remaining/rate bracket: "[00:02<00:00, 643.82it/s]". Wall-clock
# by definition, so it cannot be reproducible.
TQDM = re.compile(r"\[\d+:\d+<[^\],]*,\s*[^\]]*\]")


def path_scrubs(repo_root: str | None) -> list:
    """Absolute paths that must never survive into a committed notebook.

    `--repo-root` alone is not enough. pyGSTi is typically an *editable* install
    pointing outside the build tree, and warning tracebacks name files under
    site-packages, so a run leaks both the developer's checkout and their
    environment (and on CI, the runner's paths) into the published docs.

    Longest path first, so a root nested inside another is replaced by its own
    tag rather than by the shorter one.
    """
    pairs = []
    if repo_root:
        pairs.append((_os.path.abspath(repo_root), "<repo>"))
    try:
        spec = importlib.util.find_spec("pygsti")
        if spec and spec.origin:                      # .../pygsti/__init__.py
            pairs.append((_os.path.dirname(_os.path.dirname(_os.path.abspath(spec.origin))),
                          "<pygsti>"))
    except Exception:
        pass                                          # not importable here; skip
    for prefix in (sys.prefix, sys.exec_prefix):
        if prefix:
            pairs.append((_os.path.abspath(prefix), "<env>"))
    home = _os.path.expanduser("~")
    if home and home not in ("/", ""):
        pairs.append((home, "<home>"))                # catch-all, deliberately last

    seen, out = set(), []
    for path, tag in sorted(pairs, key=lambda t: -len(t[0])):
        if path and path not in seen:
            seen.add(path)
            out.append((path, tag))
    return out


def normalize(nb, paths: list) -> None:
    """Strip the run-to-run noise that would otherwise churn the committed diff.

    * deterministic cell ids -- ``jupytext --sync`` mints a fresh ``uuid4().hex[:8]``
      per cell on every cold sync, so without this *every* cell of *every*
      notebook changes on every rebuild;
    * drop ``cell.metadata.execution`` (iopub/shell ISO timestamps);
    * drop ``metadata.language_info.version`` (runner patch-version churn);
    * canonicalize pyGSTi's random DOM ids and plotly's uuid div ids;
    * canonicalize ``<... object at 0x...>`` repr addresses;
    * strip ANSI escapes and every absolute path in ``paths``.
    """
    seen: dict[str, int] = {}
    ids: dict[str, str] = {}

    def scrub(s: str) -> str:
        # Apply terminal carriage-return semantics: text after a \r overwrites
        # the line, so only the final frame is what a reader would have seen.
        # tqdm emits one frame per update, sampled on wall-clock, so the
        # intermediate frames differ every run while the last one is stable.
        # A \r immediately before \n is a pty line ending, not an overwrite --
        # IPython's `!` shell escape runs through pexpect and yields \r\n --
        # so fold those first or every such line collapses to nothing.
        if "\r" in s:
            s = s.replace("\r\n", "\n")
            s = "\n".join(ln.rsplit("\r", 1)[-1] for ln in s.split("\n"))
        s = TQDM.sub("[00:00<00:00, 0.00it/s]", s)
        s = ANSI.sub("", s)
        s = ADDR.sub(r"\g<1>0xADDRESS\g<2>", s)
        for _path, _tag in paths:
            s = s.replace(_path, _tag)

        def _uuid(m):
            ids.setdefault(m.group(0), f"{len(ids):08x}-0000-4000-8000-{'0' * 12}")
            return ids[m.group(0)]

        def _pyg(m):
            ids.setdefault(m.group(0), f"{m.group(1)}_{len(ids):06d}")
            return ids[m.group(0)]

        return PYGSTI_ID.sub(_pyg, UUID.sub(_uuid, s))

    for cell in nb.cells:
        src = "".join(cell.get("source", ""))
        h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:8]
        n = seen.get(h, 0)
        seen[h] = n + 1
        cell["id"] = h if n == 0 else f"{h}-{n}"
        cell.get("metadata", {}).pop("execution", None)

        # Coalesce consecutive same-stream outputs. The kernel flushes stdout in
        # timing-dependent chunks, so two runs of an identical cell can split the
        # same text across a different number of output entries. Merging first
        # (rather than after scrubbing) also lets the regexes below match ids that
        # happened to straddle a chunk boundary.
        outs = cell.get("outputs") or []
        merged: list = []
        for out in outs:
            prev = merged[-1] if merged else None
            if (prev is not None and out.get("output_type") == "stream"
                    and prev.get("output_type") == "stream"
                    and prev.get("name") == out.get("name")):
                a, b = prev.get("text", ""), out.get("text", "")
                prev["text"] = ("".join(a) if isinstance(a, list) else a) + \
                               ("".join(b) if isinstance(b, list) else b)
            else:
                merged.append(out)
        if outs:
            cell["outputs"] = merged

        for out in cell.get("outputs", []):
            for key in ("text", "traceback"):
                if key in out:
                    # nbformat stores stream `text` as a single string and
                    # `traceback` as a list of strings. Iterating the string
                    # would scrub it one character at a time, which both
                    # explodes it into a list of 1-char strings and defeats
                    # every regex below (none can match across the split).
                    val = out[key]
                    out[key] = (scrub(val) if isinstance(val, str)
                                else [scrub(t) for t in val])
            for mime, payload in (out.get("data") or {}).items():
                if not mime.startswith(("text/", "application/json",
                                        "application/javascript")):
                    continue
                out["data"][mime] = (scrub(payload) if isinstance(payload, str)
                                     else [scrub(t) for t in payload])

    nb.metadata.get("language_info", {}).pop("version", None)


def run_one(path: pathlib.Path, timeout: int, allow_errors: bool,
            logdir: pathlib.Path, paths: list) -> dict:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,                        # PER CELL, not per notebook
        startup_timeout=120,
        kernel_name="python3",
        allow_errors=allow_errors,
        interrupt_on_timeout=allow_errors,      # otherwise a timeout is fatal
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    t0 = time.time()
    status, detail = "ok", ""
    try:
        client.execute()
    except CellTimeoutError as e:
        status, detail = "timeout", str(e).splitlines()[0]
    except CellExecutionError as e:
        status, detail = "error", str(e).splitlines()[0]
    except Exception as e:                       # kernel died, OOM, ...
        status, detail = "crash", f"{type(e).__name__}: {e}"
    finally:
        normalize(nb, paths)
        nbformat.write(nb, path)                 # always persist what did run

    bad = sorted({i for i, c in enumerate(nb.cells)
                  for o in c.get("outputs", []) if o.get("output_type") == "error"})
    rec = {"notebook": str(path), "status": status, "detail": detail,
           "seconds": round(time.time() - t0, 1), "error_cells": bad}
    (logdir / (str(path).replace("/", "_") + ".json")).write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel notebooks; each is a full kernel + pyGSTi import")
    ap.add_argument("--timeout", type=int, default=1800, help="PER-CELL seconds")
    ap.add_argument("--logdir", default="nb-exec-logs")
    ap.add_argument("--allow-errors", action="store_true",
                    help="keep going past a failing cell and record its traceback "
                         "in the notebook (the exit status still reports failure)")
    ap.add_argument("--serial-chain", nargs="*", default=SERIAL_CHAIN,
                    help="notebooks touching shared state, in dependency order")
    ap.add_argument("--exclude", nargs="*", default=EXCLUDE,
                    help="notebooks to leave outputless (paths under docs/markdown, "
                         "no .ipynb suffix); pass with no values to execute everything. "
                         "REPLACES the built-in list -- to add to it, use --exclude-also")
    ap.add_argument("--exclude-also", nargs="*", default=[],
                    help="additional notebooks to leave outputless, on top of whatever "
                         "--exclude resolved to. For environment-driven exclusions (a "
                         "missing optional dependency, say) that should not silently "
                         "drop the deliberate ones")
    ap.add_argument("--repo-root", default=os.environ.get("GITHUB_WORKSPACE") or os.getcwd())
    a = ap.parse_args()

    # Kernels are subprocesses and inherit this. Without a fixed hash seed,
    # iteration order over sets and dicts keyed by strings (or tuples of them)
    # varies per process, so any cell that prints such a collection produces a
    # different ordering every run -- observed in advanced/models/Operators,
    # where elementary error-generator labels came out in a different order.
    os.environ["PYTHONHASHSEED"] = "0"

    # Also inherited by the kernels: write the tutorials' HTML reports with
    # figures split into per-figure files loaded on demand, instead of pyGSTi's
    # default of embedding everything in main.html. Split reports load far
    # faster but only work served over HTTP -- fine for these, which are served
    # from the docs site (collect-reports.py keeps each report directory
    # intact). pyGSTi's own default stays "embed", so a reader running the same
    # notebook locally still gets a double-clickable report.
    os.environ["PYGSTI_REPORT_EMBED_FIGURE_DEFAULT"] = "false"

    # A caller's environment-driven exclusions add to the deliberate ones rather
    # than replacing them; see --exclude-also.
    a.exclude = list(dict.fromkeys(list(a.exclude) + list(a.exclude_also)))

    scrubs = path_scrubs(a.repo_root)

    md = pathlib.Path(a.docs) / "markdown"
    logdir = pathlib.Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    everything = sorted(p for p in md.rglob("*.ipynb")
                        if ".ipynb_checkpoints" not in p.parts)
    excluded = {md / f"{n}.ipynb" for n in a.exclude}
    skipped = sorted(p for p in everything if p in excluded)
    everything = [p for p in everything if p not in excluded]
    chain = [p for p in (md / f"{n}.ipynb" for n in a.serial_chain)
             if p.exists() and p not in excluded]
    pool = [p for p in everything if p not in set(chain)]
    print(f"{len(everything)} notebooks: {len(chain)} serial (shared tutorial_files/"
          f"example_files state), {len(pool)} parallel at {a.jobs} jobs; "
          f"per-cell timeout {a.timeout}s", flush=True)
    # Never let an exclusion be silent -- a page that ships outputless should say
    # so in the log, not just fail to appear.
    for p in skipped:
        # Still normalize: these are committed too, and jupytext mints fresh
        # uuid4 cell ids on every cold sync, so without this the excluded pages
        # churn their whole cell list on every rebuild despite never running.
        nb = nbformat.read(p, as_version=4)
        # Strip outputs, don't just decline to execute: on a workstation the
        # .ipynb may carry outputs from an earlier run, and skipping without
        # stripping would silently ship those stale outputs (it once did).
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        normalize(nb, scrubs)
        nbformat.write(nb, p)
        print(f"  [SKIP] excluded, will ship without output: {p}", flush=True)
    for n in a.exclude:
        if md / f"{n}.ipynb" not in set(skipped):
            print(f"  [WARN] --exclude {n!r} matched no notebook", file=sys.stderr)

    results: list[dict] = []

    def show(r):
        results.append(r)
        ok = r["status"] == "ok" and not r["error_cells"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {r['seconds']:7.1f}s  {r['notebook']}"
              + ("" if ok else f"   ({r['status']}; error cells {r['error_cells']})"),
              flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.jobs + 1) as ex:
        serial = ex.submit(lambda: [run_one(p, a.timeout, a.allow_errors, logdir,
                                            scrubs) for p in chain])
        futs = [ex.submit(run_one, p, a.timeout, a.allow_errors, logdir, scrubs)
                for p in pool]
        for f in cf.as_completed(futs):
            show(f.result())
        for r in serial.result():
            show(r)

    bad = [r for r in results if r["status"] != "ok" or r["error_cells"]]
    if bad:
        print("\n=== NOTEBOOKS WITH ERRORS ===", file=sys.stderr)
        for r in sorted(bad, key=lambda r: r["notebook"]):
            print(f"  {r['notebook']}  [{r['status']}] cells {r['error_cells']}"
                  f"\n      {r['detail']}", file=sys.stderr)
    print(f"\n{len(results) - len(bad)}/{len(results)} notebooks executed cleanly")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
