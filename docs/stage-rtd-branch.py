#!/usr/bin/env python
"""Assemble the branch ReadTheDocs builds the hosted docs from.

ReadTheDocs does not execute notebooks (``docs/_config.yml`` sets
``execute_notebooks: 'off'``), so the branch it builds has to already carry the
executed output. This script builds such a branch: it takes a source commit plus
a tree of executed ``.ipynb``, replaces each notebook page's ``.md`` with the
``.ipynb`` holding that page's output, adds the collected HTML reports,
retargets ``docs/_config.yml``, and commits the result as one atomic commit.

It is the same work the CI job does, in a form a human can run. Executing the 76
notebooks is the expensive half (~40 minutes on a 2-core GitHub runner) and any
modern laptop beats that badly, so a local run is often the faster way to get a
preview onto RTD -- and the only way that doesn't cost a CI slot.

The full local recipe, from a clean checkout of the branch you want staged::

    jupytext --sync 'docs/markdown/**/*.md'
    python -c "import sys; sys.path.insert(0,'docs'); import conftest; conftest._generate_shared_fixtures()"
    python docs/execute-notebooks.py --jobs 8 --timeout 1800 --allow-errors
    python docs/collect-reports.py collect
    python docs/stage-rtd-branch.py --target-branch docs-preview --push

Prerequisites for the execution step, beyond ``pip install -e .[docs]``:
``.[chp]`` for the Clifford-simulation page, an MPI launcher for the Parallelism
page, and ``.[ml]`` for the QPANN page. Anything missing costs you that page's
output, not the run -- pass the page to ``execute-notebooks.py --exclude-also``
to skip it deliberately instead of shipping a traceback.

Local execution is as reproducible as CI execution. ``execute-notebooks.py``
scrubs the repo root, the pyGSTi install directory, ``sys.prefix`` and ``$HOME``
out of every recorded output, so nothing machine-specific survives into a
committed notebook. The run-to-run churn that does remain (28 of 76 notebooks)
comes from unseeded randomness inside pyGSTi and from MPI rank interleaving,
neither of which cares which machine you are on.

Executing the notebooks also writes ~25 pyGSTi HTML reports into the scratch
directories; ``collect-reports.py`` packs those into ``docs/reports.tar.xz``,
which this script commits alongside the notebooks so that the reports and the
pages linking to them always ship together. Pass ``--allow-missing-reports`` to
stage a branch without them, accepting that every report link will 404.

Nothing is pushed without ``--push``, and the protected branch names are refused
outright.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

#: Never force-push these, whatever is passed. This script's normal mode of
#: operation is `git push --force`, which on any of these would destroy work.
PROTECTED = {"master", "main", "develop", "beta", "bugfix"}

def run(cmd, cwd=None, check=True, capture=True):
    """Thin subprocess wrapper that reports the command on failure."""
    p = subprocess.run(cmd, cwd=cwd, check=False, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    if check and p.returncode != 0:
        sys.exit(f"error: {' '.join(cmd)} failed ({p.returncode})\n{p.stdout or ''}")
    return (p.stdout or "").strip()


def find_notebooks(root: pathlib.Path) -> list:
    """Executed notebooks under `root`, excluding Jupyter's checkpoint copies."""
    return sorted(p for p in root.rglob("*.ipynb")
                  if ".ipynb_checkpoints" not in p.parts)


def is_paired(md: pathlib.Path) -> bool:
    """Whether `md` is a jupytext-paired notebook page rather than plain markdown.

    Reads the leading ``---`` front matter only. Grepping the whole file for
    ``^jupytext:`` gives false positives on any page that happens to discuss the
    pairing setup, which is exactly the kind of page the docs contain.
    """
    try:
        lines = md.read_text(errors="replace").splitlines()
    except OSError:
        return False
    if not lines or lines[0].strip() != "---":
        return False
    for line in lines[1:60]:
        if line.strip() == "---":
            return False
        if line.startswith("jupytext:"):
            return True
    return False


def check_staleness(notebooks: list) -> list:
    """Notebooks older than the ``.md`` they were generated from.

    A stale pair means the page was executed, then its source was edited: the
    committed output would show results for code that is no longer on the page.
    Cheap to detect by mtime and worth detecting, because nothing downstream
    would ever notice -- the site renders, the outputs just describe the wrong
    source.
    """
    stale = []
    for nb in notebooks:
        md = nb.with_suffix(".md")
        if md.exists() and md.stat().st_mtime > nb.stat().st_mtime:
            stale.append(nb)
    return stale


def check_reports(repo: pathlib.Path) -> pathlib.Path | None:
    """The report tarball, if it is present and not older than the reports.

    Same failure the notebook staleness check guards against, one level up: the
    notebooks can be re-executed without anyone re-running collect-reports.py,
    and the site would then serve reports describing an older run. Returns None
    when there is no tarball at all, which the caller decides what to do about.
    """
    tarball = repo / "docs" / "reports.tar.xz"
    if not tarball.exists():
        return None
    packed = tarball.stat().st_mtime
    newer = sorted(m.parent.name
                   for root in ("tutorial_files", "example_files")
                   for m in (repo / "docs" / root).glob("*/main.html")
                   if m.stat().st_mtime > packed)
    if newer:
        sys.exit(f"error: {len(newer)} report(s) newer than docs/reports.tar.xz:\n  "
                 + "\n  ".join(newer)
                 + "\n  Re-run docs/collect-reports.py collect.")
    return tarball


def retarget_config(cfg: pathlib.Path, target_branch: str, edit_branch: str):
    """Point the launch buttons at `target_branch`, the source buttons at `edit_branch`.

    Both substitutions match on the KEY and replace whatever value is there,
    rather than matching the old value: a rewrite anchored on the literal
    ``develop`` silently no-ops when staging from anywhere else and leaves the
    buttons pointing at the wrong branch. The indent distinguishes them --
    two-space ``branch:`` is ``repository.branch``, four-space
    ``pygsti_edit_branch:`` is under ``sphinx.config`` -- and each is unique in
    the file. Trailing comments survive.
    """
    text = cfg.read_text()
    text, n_repo = re.subn(r"^(  branch: )[^ #\n]+", rf"\g<1>{target_branch}",
                           text, count=1, flags=re.M)
    text, n_edit = re.subn(r"^(    pygsti_edit_branch: )[^ #\n]+", rf"\g<1>{edit_branch}",
                           text, count=1, flags=re.M)
    if n_repo != 1 or n_edit != 1:
        sys.exit(f"error: _config.yml rewrite missed "
                 f"(repository.branch x{n_repo}, pygsti_edit_branch x{n_edit}); "
                 f"the keys or their indentation must have changed")
    cfg.write_text(text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-branch", required=True,
                    help="branch to build (and, with --push, force-push). Overwritten.")
    ap.add_argument("--edit-branch", default=None,
                    help="branch the Show source / Suggest edit buttons point at, "
                         "i.e. where the human-editable .md lives. Defaults to the "
                         "current branch, which is what you want when previewing "
                         "work that is not on develop yet.")
    ap.add_argument("--source", default="HEAD",
                    help="commit to build the staged branch from (default: HEAD)")
    ap.add_argument("--notebooks-from", default="docs/markdown",
                    help="tree of executed .ipynb to stage (default: docs/markdown). "
                         "Point this at an unpacked CI artifact to stage output "
                         "someone else executed.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="stage even if some paired page has no executed .ipynb. "
                         "Those pages ship as unexecuted source, which renders "
                         "without ever reporting a problem, so this is off by default.")
    ap.add_argument("--allow-stale", action="store_true",
                    help="stage even if some .md is newer than its executed .ipynb")
    ap.add_argument("--allow-missing-reports", action="store_true",
                    help="stage even if docs/reports.tar.xz is absent. The site "
                         "then builds without the example reports, and every link "
                         "to one 404s, so this is off by default.")
    ap.add_argument("--push", action="store_true",
                    help="force-push the result. Without this the branch is built "
                         "locally and left for you to inspect.")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--keep-branch", action="store_true",
                    help="keep the local branch after pushing (default: delete it, "
                         "since it is a generated artifact and not somewhere to work)")
    a = ap.parse_args()

    if a.target_branch in PROTECTED:
        sys.exit(f"error: refusing to stage onto protected branch '{a.target_branch}'")

    repo = pathlib.Path(run(["git", "rev-parse", "--show-toplevel"]))
    nb_root = (repo / a.notebooks_from) if not pathlib.Path(a.notebooks_from).is_absolute() \
        else pathlib.Path(a.notebooks_from)
    if not nb_root.is_dir():
        sys.exit(f"error: {nb_root} is not a directory")

    edit_branch = a.edit_branch or run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    if edit_branch == "HEAD":
        sys.exit("error: detached HEAD, so --edit-branch cannot be inferred; pass it")
    source_sha = run(["git", "rev-parse", a.source], cwd=repo)

    notebooks = find_notebooks(nb_root)
    print(f"source commit    {source_sha[:12]} ({a.source})")
    print(f"executed .ipynb  {len(notebooks)} under {nb_root}")
    if not notebooks:
        sys.exit(f"error: no executed notebooks under {nb_root}. "
                 f"Run docs/execute-notebooks.py first.")

    tarball = check_reports(repo)
    if tarball is None:
        msg = ("no docs/reports.tar.xz; run docs/collect-reports.py collect "
               "after executing the notebooks")
        if not a.allow_missing_reports:
            sys.exit(f"error: {msg}, or pass --allow-missing-reports.")
        print(f"warning: {msg}. Report links will 404.", file=sys.stderr)
    else:
        print(f"report tarball   {tarball.stat().st_size / 2**20:.1f} MB")

    stale = check_staleness(notebooks)
    if stale:
        rel = "\n  ".join(str(p.relative_to(repo)) for p in stale[:10])
        more = f"\n  ... and {len(stale) - 10} more" if len(stale) > 10 else ""
        msg = (f"{len(stale)} notebook(s) older than the .md they came from:\n  "
               f"{rel}{more}\nTheir committed output describes source that has since "
               f"changed.")
        if not a.allow_stale:
            sys.exit(f"error: {msg}\nRe-execute them, or pass --allow-stale.")
        print(f"warning: {msg}")

    # Build in a throwaway worktree rather than switching the current checkout's
    # branch. This script is meant to be run from a working tree someone is
    # actively using; `git checkout -B` there would move their HEAD out from
    # under them, and a failure partway through would strand them on a generated
    # branch. The worktree is removed on the way out either way.
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="rtd-stage-"))
    wt = tmp / "wt"
    try:
        run(["git", "worktree", "add", "--detach", "--quiet", str(wt), source_sha], cwd=repo)

        # Every paired page must have an executed notebook. A count floor cannot
        # do this job: the number of pages changes as docs are written, so any
        # fixed threshold is either slack enough to miss a real gap or tight
        # enough to need editing whenever a page is added. Compare against the
        # source commit instead, which knows exactly how many paired pages it has.
        #
        # This matters because the failure is invisible downstream. A paired page
        # with no .ipynb keeps its .md, renders as unexecuted source, and reports
        # nothing wrong anywhere -- the site just quietly stops showing results
        # for that page.
        src_md = wt / "docs" / "markdown"
        expected = {p.relative_to(src_md).with_suffix("")
                    for p in src_md.rglob("*.md") if is_paired(p)}
        have = {p.relative_to(nb_root).with_suffix("") for p in notebooks}
        missing = sorted(expected - have)
        print(f"paired pages     {len(expected)} on {source_sha[:12]}, "
              f"{len(have & expected)} with executed output")
        if missing:
            listed = "\n  ".join(str(m) for m in missing[:10])
            more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
            msg = (f"{len(missing)} paired page(s) have no executed notebook:\n  "
                   f"{listed}{more}")
            if not a.allow_missing:
                sys.exit(f"error: {msg}\nRun `jupytext --sync 'docs/markdown/**/*.md'` "
                         f"then docs/execute-notebooks.py, or pass --allow-missing.")
            print(f"warning: {msg}\n  these pages will ship as unexecuted source")

        # Copy the executed notebooks in, then drop the .md each one supersedes.
        # Sphinx picks ONE source per document and jupyter-book warns "multiple
        # files found for the document" when both exist -- a warning logged
        # without a type=, so suppress_warnings cannot silence it and any -W
        # build fails on it. Plain-markdown pages (index and landing pages, which
        # carry no jupytext header and so have no paired .ipynb) are left alone.
        dest_root = wt / "docs" / "markdown"
        staged = 0
        for nb in notebooks:
            dest = dest_root / nb.relative_to(nb_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nb, dest)
            dest.with_suffix(".md").unlink(missing_ok=True)
            staged += 1

        if tarball is not None:
            shutil.copy2(tarball, wt / "docs" / tarball.name)

        retarget_config(wt / "docs" / "_config.yml", a.target_branch, edit_branch)

        run(["git", "checkout", "--quiet", "-B", a.target_branch], cwd=wt)
        # -f because .ipynb is gitignored on the source branch.
        run(["git", "add", "-A", "-f", "docs/markdown"], cwd=wt)
        run(["git", "add", "docs/_config.yml"], cwd=wt)
        if tarball is not None:
            # -f for the same reason as the notebooks: generated, and so
            # gitignored on every branch a human works on.
            run(["git", "add", "-f", f"docs/{tarball.name}"], cwd=wt)
        # [skip ci] so pushing a branch of generated notebook output does not
        # kick off a test matrix for source that was already tested.
        run(["git", "-c", "user.name=PyGSTi", "-c", "user.email=pygsti@noreply.github.com",
             "commit", "--quiet", "-m", "Auto-build notebooks for ReadTheDocs [skip ci]"],
            cwd=wt)
        head = run(["git", "rev-parse", "HEAD"], cwd=wt)

        print(f"staged           {staged} notebooks"
              + (" + reports" if tarball is not None else "")
              + f" onto '{a.target_branch}' ({head[:12]})")
        print(f"launch buttons   -> {a.target_branch}")
        print(f"source buttons   -> {edit_branch}")

        if a.push:
            run(["git", "push", "--force", a.remote,
                 f"{a.target_branch}:{a.target_branch}"], cwd=wt, capture=False)
            print(f"pushed           {a.remote}/{a.target_branch}")
            if not a.keep_branch:
                run(["git", "branch", "-D", a.target_branch], cwd=repo, check=False)
        else:
            print(f"\nNot pushed. Inspect it with:\n"
                  f"  git log -1 --stat {a.target_branch}\n"
                  f"Then publish with:\n"
                  f"  git push --force {a.remote} {a.target_branch}\n"
                  f"or re-run this with --push.")
    finally:
        run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, check=False)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
