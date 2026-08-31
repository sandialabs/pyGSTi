#!/usr/bin/env python3
"""Gather the HTML reports the tutorials generate, for publication with the docs.

Executing the notebook pages (``docs/execute-notebooks.py``) leaves ~25 pyGSTi
HTML reports behind in the gitignored scratch directories ``docs/tutorial_files``
and ``docs/example_files``. Those reports are worth serving from the docs site:
a reader gets to click through a real GST report instead of reading a page that
describes one.

This script collects them into ``docs/_extra/reports`` -- which ``_config.yml``
publishes via ``html_extra_path``, so it lands at ``/reports/`` on the site --
and packs the result into ``docs/reports.tar.xz`` for transport.

Two shapes come out of ``write_html``:

* A report whose directory holds only ``main.html`` is flattened to
  ``reports/<name>.html``; the flattened reports share one ``reports/offline/``
  directory, which is where the ``./offline/...`` references inside every report
  resolve to (the logo, the CSS duplicates that back up the CDN stylesheets, and
  the fallback used when a CDN is unreachable).
* A report with sidecar files keeps its directory -- ``reports/<name>/main.html``,
  with its own copy of ``offline`` since the shared one is no longer a sibling.
  Because ``execute-notebooks.py`` runs the tutorials with
  ``PYGSTI_REPORT_EMBED_FIGURE_DEFAULT=false``, this is every report: each
  carries a ``figures/`` directory of lazily-fetched figure files (plus, with
  ``link_to=``, pickled tables and LaTeX source), linked by relative path from
  inside the HTML.

Why a tarball rather than committing ``_extra`` itself: the reports are 575 MB
uncompressed and 5.4 MB as ``.tar.xz``, because 25 reports of the same template
share almost everything. ``stage-rtd-branch.py`` commits the tarball onto the
generated branch alongside the executed notebooks, and ReadTheDocs unpacks it
before the Sphinx build.

Usage::

    python docs/collect-reports.py collect    # scratch dirs -> _extra + tarball
    python docs/collect-reports.py extract    # tarball -> _extra
    python docs/collect-reports.py check      # built site -> report links resolve

``extract`` is what runs on ReadTheDocs, and what you want locally before
``jb build docs/`` if you did not just run ``collect`` yourself. ``check`` runs
against ``docs/_build/html`` after a build: the pages link to the reports by
relative path, and nothing in the Sphinx build validates a path that resolves
outside the document tree.
"""

import argparse
import html
import lzma
import pathlib
import re
import shutil
import sys
import tarfile

DOCS = pathlib.Path(__file__).resolve().parent

# Where the notebooks write. Relative to DOCS; both are gitignored scratch.
SCRATCH = ("tutorial_files", "example_files")

EXTRA = DOCS / "_extra"          # html_extra_path root; contents land at site root
DEST = EXTRA / "reports"         # -> /reports/ on the built site
TARBALL = DOCS / "reports.tar.xz"

# `write_html("<path>/<name>"` and `write_html('<path>/<name>'`, to label each
# report with the page that generated it. Reports written through a variable
# (Leakage, RobustGST-TVD) simply go unlabelled; the index still lists them.
WRITE_HTML = re.compile(r"""write_html\(\s*(['"])(?P<path>[^'"]+)\1""")


def find_reports(docs: pathlib.Path) -> list:
    """Every directory under the scratch roots holding a `main.html`."""
    found = []
    for root in SCRATCH:
        base = docs / root
        if not base.is_dir():
            continue
        for main in sorted(base.glob("*/main.html")):
            found.append(main.parent)
    return found


def source_pages(docs: pathlib.Path, names: list) -> dict:
    """Map report name -> page path, by reading the pages that write them."""
    pages = {}
    md_root = docs / "markdown"
    texts = {md: md.read_text(encoding="utf-8") for md in sorted(md_root.rglob("*.md"))}

    for md, text in texts.items():
        for m in WRITE_HTML.finditer(text):
            name = pathlib.PurePosixPath(m.group("path")).name
            # First writer wins: gettingStartedReport is written by both
            # FirstGST and Workflow, and FirstGST is where a reader meets it.
            pages.setdefault(name, str(md.relative_to(md_root).with_suffix("")))

    # Leakage and RobustGST-TVD pass a variable to write_html, so the regex above
    # cannot see the path. Fall back to whichever page mentions the output
    # directory at all -- that is where the variable is assigned.
    for name in names:
        if name in pages:
            continue
        for md, text in texts.items():
            if re.search(rf"""['"][^'"]*/{re.escape(name)}['"]""", text):
                pages[name] = str(md.relative_to(md_root).with_suffix(""))
                break
    return pages


def offline_source() -> pathlib.Path:
    """The `offline` template directory inside the installed pyGSTi."""
    try:
        import pygsti.report as _r
    except ImportError:
        sys.exit("error: pyGSTi must be importable to place the offline directory.")
    d = pathlib.Path(_r.__file__).parent / "templates" / "offline"
    if not d.is_dir():
        sys.exit(f"error: no offline templates at {d}")
    return d


def place_offline(dest: pathlib.Path) -> None:
    """Copy pyGSTi's shared JS/CSS in beside the flat reports."""
    src = offline_source()
    shutil.rmtree(dest / "offline", ignore_errors=True)
    shutil.copytree(src, dest / "offline")


def sidecars(report: pathlib.Path) -> list:
    """Entries a report needs alongside main.html, `offline` aside."""
    return sorted(p for p in report.iterdir()
                  if p.name not in ("main.html", "offline"))


def dir_size(p: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_index(dest: pathlib.Path, entries: list) -> None:
    """A plain listing at /reports/, so the collection is browsable on its own."""
    rows = []
    for name, href, size, page in entries:
        page_cell = (f'<a href="../markdown/{html.escape(page)}.html">{html.escape(page)}</a>'
                     if page else "<span class=none>&mdash;</span>")
        rows.append(
            f'<tr><td><a href="{html.escape(href)}">{html.escape(name)}</a></td>'
            f'<td class=num>{size / 2**20:.1f} MB</td><td>{page_cell}</td></tr>'
        )
    total = sum(e[2] for e in entries)
    (dest / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pyGSTi example reports</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 60rem;
         margin: 3rem auto; padding: 0 1.5rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .25rem; }}
  p {{ max-width: 42rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 2rem; }}
  th, td {{ text-align: left; padding: .5rem .75rem;
            border-bottom: 1px solid rgba(128,128,128,.35); }}
  th {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums;
          white-space: nowrap; }}
  .none {{ opacity: .5; }}
</style></head><body>
<h1>pyGSTi example reports</h1>
<p>The HTML reports generated by the example notebooks in this documentation,
served here so you can read one without running anything. Each is the output of
a <code>write_html</code> call on the page listed beside it.</p>
<table>
<thead><tr><th>Report</th><th class=num>Size</th><th>Generated by</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
<p style="margin-top:2rem;opacity:.7;font-size:.9rem">
{len(entries)} reports, {total / 2**20:.0f} MB.</p>
</body></html>
""", encoding="utf-8")


def collect(docs: pathlib.Path) -> int:
    reports = find_reports(docs)
    if not reports:
        sys.exit("error: no reports found. Run docs/execute-notebooks.py first.")

    seen = {}
    for r in reports:
        if r.name in seen:
            sys.exit(f"error: two reports named {r.name!r}:\n  {seen[r.name]}\n  {r}")
        seen[r.name] = r

    shutil.rmtree(EXTRA, ignore_errors=True)
    DEST.mkdir(parents=True)
    pages = source_pages(docs, [r.name for r in reports])

    entries, flat, nested = [], 0, 0
    for r in reports:
        extra = sidecars(r)
        if extra:
            # Keep the directory: the sidecars are linked by relative path.
            out = DEST / r.name
            out.mkdir()
            shutil.copy2(r / "main.html", out / "main.html")
            for p in extra:
                (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / p.name)
            href, nested = f"{r.name}/main.html", nested + 1
        else:
            shutil.copy2(r / "main.html", DEST / f"{r.name}.html")
            href, flat = f"{r.name}.html", flat + 1
        entries.append((r.name, href, (r / "main.html").stat().st_size,
                        pages.get(r.name)))

    write_index(DEST, entries)

    print(f"collected        {len(entries)} reports ({flat} flat, {nested} with sidecars)")
    unlabelled = [e[0] for e in entries if not e[3]]
    if unlabelled:
        # No page writes these, so they are almost certainly left over from an
        # earlier version of the docs -- the scratch directories are never
        # cleared by anything. They would still be published, unreferenced.
        print(f"warning: {len(unlabelled)} report(s) that no page generates:\n  "
              + "\n  ".join(unlabelled)
              + "\n  Clear the scratch directories and re-execute to drop them.",
              file=sys.stderr)

    TARBALL.unlink(missing_ok=True)
    # preset 9|EXTREME: ~30 s for this corpus and worth it, since the payload
    # rides a git branch that is force-pushed on every docs staging run.
    with lzma.open(TARBALL, "wb", preset=9 | lzma.PRESET_EXTREME) as fh:
        with tarfile.open(fileobj=fh, mode="w") as tar:
            tar.add(EXTRA, arcname=EXTRA.name)
    raw = dir_size(EXTRA)
    packed = TARBALL.stat().st_size
    print(f"packed           {TARBALL.relative_to(docs)} "
          f"({packed / 2**20:.1f} MB, {raw / packed:.0f}x)")

    # After packing, so the offline libraries stay out of the tarball: they are
    # ~9 MB that every install of pyGSTi already carries, and regenerating them
    # on extract keeps them matched to the installed version.
    populate_offline()
    print(f"staged           {EXTRA.relative_to(docs)} ({dir_size(EXTRA) / 2**20:.0f} MB)")
    return 0


def populate_offline() -> None:
    """Place pyGSTi's JS/CSS: one shared copy, plus one per directory report."""
    place_offline(DEST)
    for main in sorted(DEST.glob("*/main.html")):
        place_offline(main.parent)


def extract(docs: pathlib.Path) -> int:
    if not TARBALL.exists():
        sys.exit(f"error: {TARBALL} not found. Run `collect-reports.py collect` "
                 f"after executing the notebooks, or fetch the staged branch.")
    shutil.rmtree(EXTRA, ignore_errors=True)
    with lzma.open(TARBALL, "rb") as fh:
        with tarfile.open(fileobj=fh, mode="r") as tar:
            # filter="data" refuses absolute paths, `..` and special files.
            tar.extractall(docs, filter="data")
    populate_offline()
    n = len(list(DEST.glob("*.html"))) - 1 + len(list(DEST.glob("*/main.html")))
    print(f"extracted        {n} reports into {EXTRA.relative_to(docs)} "
          f"({dir_size(EXTRA) / 2**20:.0f} MB)")
    return 0


# href="../../reports/<name>.html" as written by the pages, plus the directory
# form for a report with sidecars.
REPORT_HREF = re.compile(r'href="([^"]*reports/[^"#?]+\.html)"')


def check(docs: pathlib.Path, html: pathlib.Path) -> int:
    """Every report link on a built page resolves, and every report is linked."""
    if not html.is_dir():
        sys.exit(f"error: {html} is not a directory. Build the docs first.")
    pages = html / "markdown"
    served = html / "reports"
    if not served.is_dir():
        sys.exit(f"error: no {served}. Run `collect-reports.py extract`, then rebuild.")

    broken, hit = [], set()
    for page in sorted(pages.rglob("*.html")):
        for m in REPORT_HREF.finditer(page.read_text(encoding="utf-8", errors="replace")):
            target = (page.parent / m.group(1)).resolve()
            if target.is_file():
                hit.add(target)
            else:
                broken.append(f"{page.relative_to(html)} -> {m.group(1)}")

    everything = {p.resolve() for p in served.glob("*.html")
                  if p.name != "index.html"} | \
                 {p.resolve() for p in served.glob("*/main.html")}
    unlinked = sorted(p.stem if p.name != "main.html" else p.parent.name
                      for p in everything - hit)

    print(f"linked           {len(hit)} of {len(everything)} reports")
    if unlinked:
        # Not fatal: the index page lists them, and a report can be worth
        # serving without a page that points at it.
        print(f"warning: {len(unlinked)} report(s) no page links to:\n  "
              + "\n  ".join(unlinked), file=sys.stderr)
    if broken:
        sys.exit(f"error: {len(broken)} report link(s) do not resolve:\n  "
                 + "\n  ".join(broken))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("collect", "extract", "check"))
    ap.add_argument("--html", type=pathlib.Path, default=DOCS / "_build" / "html",
                    help="built site to check (default: docs/_build/html)")
    a = ap.parse_args()
    if a.action == "collect":
        return collect(DOCS)
    if a.action == "extract":
        return extract(DOCS)
    return check(DOCS, a.html)


if __name__ == "__main__":
    sys.exit(main())
