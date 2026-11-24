#!/usr/bin/env python3
"""Build a local preview of the docs with executed notebooks and example reports.

    python docs/local-preview.py full     # execute every notebook, collect reports, build
    python docs/local-preview.py update   # after editing pages: incremental rebuild

Run from a source checkout whose extensions are built in place, with ``PYTHONPATH``
set to the checkout (see ``docs/README.md``). Both commands use the interpreter
that runs this script, for execution and for the build.

``full`` converts the paired Markdown pages to notebooks (leaving landing and prose
pages alone), generates the shared fixtures, runs ``execute-notebooks.py``, packs
the reports with ``collect-reports.py``, and then does what ``update`` does. Expect
it to take a while: on an 8-core workstation, about 35 minutes. QPANN is skipped
when TensorFlow isn't importable, as in CI. Extra arguments after ``--`` go to
``execute-notebooks.py`` (e.g. ``-- --jobs 4 --exclude-also guides/gst/running/Parallelism``).

``update`` builds from a mirror of ``docs/`` in ``docs/_build/executed``, in which
each paired page's ``.md`` is replaced by its executed ``.ipynb``. It copies only
files that changed and lets Sphinx rebuild incrementally, so an edit to one page
takes seconds rather than minutes, and the served site stays up during the
rebuild. A paired page edited after execution is pushed into its notebook with
``jupytext --update``, which keeps the existing outputs; cells whose code changed
show stale output until the page is re-executed.

The example reports are copied into the HTML output by this script rather than
through ``html_extra_path``: Sphinx 7 compares every extra file byte by byte on
every build, which for ~800 MB of reports costs minutes per rebuild. (Sphinx
warns that ``_extra`` doesn't exist; that is expected here.)

Serve the result with::

    python -m http.server 8950 --bind 127.0.0.1 --directory docs/_build/executed/_build/prose/_build/html

Some pages need more than ``.[docs]``; see ``docs/README.md`` and the pages'
own notes (MPI, Qiskit, TeX for PDF output).
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
MD = DOCS / "markdown"
DST = DOCS / "_build" / "executed"
HTML = DST / "_build" / "prose" / "_build" / "html"
PY = sys.executable

_spec = importlib.util.spec_from_file_location("stage", DOCS / "stage-rtd-branch.py")
_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stage)
is_paired = _stage.is_paired


# Tools installed beside this interpreter (jb, jupytext) come first on PATH.
ENV = dict(os.environ, PATH=os.pathsep.join([str(Path(PY).parent), os.environ.get("PATH", "")]))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, env=ENV, **kw)


def same(a: Path, b: Path) -> bool:
    if not b.exists():
        return False
    sa, sb = a.stat(), b.stat()
    return sa.st_size == sb.st_size and sa.st_mtime_ns == sb.st_mtime_ns


def paired_pages():
    return sorted(p for p in MD.rglob("*.md") if is_paired(p))


def full(extra_args):
    # Notebooks for paired pages only; drop any stray notebook made from a prose page.
    for nb in MD.rglob("*.ipynb"):
        md = nb.with_suffix(".md")
        if ".ipynb_checkpoints" not in nb.parts and not (md.exists() and is_paired(md)):
            nb.unlink()
    run([PY, "-m", "jupytext", "--quiet", "--to", "notebook", "--update", *paired_pages()], check=True)
    run([PY, "-c", "import sys; sys.path.insert(0, 'docs'); import conftest; "
                   "conftest._generate_shared_fixtures()"], check=True)
    args = list(extra_args)
    if importlib.util.find_spec("tensorflow") is None:
        print("TensorFlow unavailable; models/QPANN will be left without output", flush=True)
        args += ["--exclude-also", "models/QPANN"]
    rc = run([PY, "docs/execute-notebooks.py", "--allow-errors",
              "--logdir", "docs/_build/nb-exec-logs", *args]).returncode
    if rc:
        print("execute-notebooks.py reported failing cells; building anyway", flush=True)
    run([PY, "docs/collect-reports.py", "collect"], check=True)
    return update() or rc


def wanted_files():
    """Map preview-relative path -> source path for the preview tree."""
    out = {}
    for root, dirs, files in os.walk(DOCS):
        rel_root = Path(root).relative_to(DOCS)
        if rel_root.parts[:1] in (("_build",), ("_extra",)):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in (".ipynb_checkpoints", "__pycache__")
                   and not (rel_root == Path(".") and d in ("_build", "_extra"))]
        for name in files:
            src = Path(root) / name
            rel = rel_root / name
            in_md = rel.parts[:1] == ("markdown",)
            if in_md and src.suffix == ".ipynb":
                md = src.with_suffix(".md")
                if md.exists() and is_paired(md):
                    out[rel] = src          # the executed notebook stands in for the page
                continue
            if in_md and src.suffix == ".md" and is_paired(src) and src.with_suffix(".ipynb").exists():
                continue                    # replaced by its notebook
            out[rel] = src
    return out


def update():
    stale = [md for md in paired_pages() if md.with_suffix(".ipynb").exists()
             and md.stat().st_mtime > md.with_suffix(".ipynb").stat().st_mtime]
    if stale:
        run([PY, "-m", "jupytext", "--quiet", "--update", "--to", "notebook", *stale], check=True)

    want = wanted_files()
    copied = removed = 0
    for rel, src in want.items():
        dst = DST / rel
        if not same(src, dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    for root, dirs, files in os.walk(DST):
        rel_root = Path(root).relative_to(DST)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d != "_build"]
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if rel_root / name not in want:
                (DST / rel_root / name).unlink()
                removed += 1
    shutil.rmtree(DST / "_extra", ignore_errors=True)
    print(f"preview source: {copied} files copied, {removed} removed", flush=True)

    rc = run([PY, DST / "build-prose.py"]).returncode
    if rc:
        return rc

    # What html_extra_path would do, copying only what changed.
    extra = DOCS / "_extra"
    if extra.exists():
        want_extra = set()
        for src in extra.rglob("*"):
            if src.is_file():
                rel = src.relative_to(extra)
                want_extra.add(rel)
                if not same(src, HTML / rel):
                    (HTML / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, HTML / rel)
        for top in {r.parts[0] for r in want_extra}:
            for dst in (HTML / top).rglob("*"):
                if dst.is_file() and dst.relative_to(HTML) not in want_extra:
                    dst.unlink()
    else:
        print("no docs/_extra: run collect-reports.py (or `full`) for the example reports", flush=True)
    print(f"\npreview: {HTML / 'index.html'}")
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("full", "update"):
        sys.exit(__doc__)
    extra = sys.argv[2:]
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if sys.argv[1] == "update" and extra:
        sys.exit("update takes no arguments")
    return full(extra) if sys.argv[1] == "full" else update()


if __name__ == "__main__":
    sys.exit(main())
