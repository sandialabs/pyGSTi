"""Local Sphinx extension: emit HTML redirect stubs for relocated pages.

The 2026 documentation restructure moved essentially every tutorial page into one
of three reader-tier directories (``start/``, ``guides/``, ``advanced/``). Papers,
issues and third-party posts cite the old URLs, so those paths must keep
resolving rather than 404.

``docs/_redirects.yml`` maps each retired page path (relative to the HTML root,
without extension) to the path of the page that replaces it (relative to the
retired page's own directory, *with* ``.html``). At the end of a build this
extension writes a minimal stub at each retired path containing a
``<meta http-equiv="refresh">``, a ``<link rel="canonical">`` so search engines
transfer authority to the new page, and a visible link for anyone whose browser
declines the refresh.

Deliberately dependency-free: the alternative, ``sphinx-reredirects``, would add a
package to the ``docs`` extra for ~30 lines of behavior.

A stub is never written over a real page produced by the build. That check asks
Sphinx which documents this build actually contains (``env.found_docs``) rather
than which files happen to sit in the output directory: ``jb build`` is
incremental, so after a restructure the output directory still holds the HTML of
every page that moved. A filesystem check would see those leftovers, decline to
write a single stub, and leave the old URLs quietly serving stale content.
Overwriting them is the point.
"""

import os

import yaml
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={target}">
    <link rel="canonical" href="{target}">
    <meta name="robots" content="noindex">
    <title>Page moved</title>
  </head>
  <body>
    <p>This page has moved to <a href="{target}">{target}</a>.</p>
  </body>
</html>
"""


def _emit(app, exception):
    if exception is not None or app.builder.name != "html":
        return

    src = os.path.join(app.srcdir, "_redirects.yml")
    if not os.path.exists(src):
        LOGGER.info("[page_redirects] no _redirects.yml; nothing to do")
        return

    with open(src) as handle:
        mapping = yaml.safe_load(handle) or {}

    real_docs = set(app.env.found_docs)

    written = skipped = 0
    for old, target in mapping.items():
        if old in real_docs:
            # This path is a live page in this build; a redirect would shadow it.
            LOGGER.warning(
                "[page_redirects] %s is a real page in this build; skipping redirect", old
            )
            skipped += 1
            continue
        out = os.path.join(app.outdir, f"{old}.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as handle:
            handle.write(_TEMPLATE.format(target=target))
        written += 1

    LOGGER.info("[page_redirects] wrote %d redirect stubs (%d skipped)", written, skipped)


def setup(app):
    app.connect("build-finished", _emit)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
