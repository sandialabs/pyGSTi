#!/usr/bin/env python3
"""Build the prose documentation without the API reference.

A full ``jb build docs`` also renders roughly 1200 stub pages that
``sphinx.ext.autosummary`` generates from pyGSTi's docstrings. Those pages
dominate the build time and are irrelevant while you are editing the tutorials,
so this script builds everything except them. From a cold start that is a few
seconds rather than the better part of half an hour.

    python docs/build-prose.py            # incremental; re-reads changed pages only
    python docs/build-prose.py --all      # force a full re-read

Any further arguments are passed through to ``jb build``.

The config and table of contents are derived from ``_config.yml`` and
``_toc.yml`` at run time, so there is nothing here to keep in sync when those
change. The derivation drops the "API reference" part from the table of
contents, excludes ``api.rst`` and the generated ``_autosummary`` tree, and
turns off ``autosummary_generate``.

The cost is that this build has no API reference: no API section in its sidebar,
and any cross-reference from a tutorial into the ``pygsti`` API pages will not
resolve. No tutorial makes such a reference today. Use ``jb build docs`` for a
build that includes the API reference, and before you trust a link check.
"""
import subprocess
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).parent.resolve()
OUT = DOCS / "_build" / "prose"
EXTRA_EXCLUDES = ["api.rst", "_autosummary/**"]

toc = yaml.safe_load((DOCS / "_toc.yml").read_text())
toc["parts"] = [part for part in toc["parts"] if part.get("caption") != "API reference"]

config = yaml.safe_load((DOCS / "_config.yml").read_text())
sphinx = config.setdefault("sphinx", {})
# local_extensions paths are resolved relative to the config file. This config is
# written somewhere other than docs/, so they have to be made absolute first.
sphinx["local_extensions"] = {
    name: str((DOCS / path).resolve())
    for name, path in sphinx.get("local_extensions", {}).items()
}
sphinx_config = sphinx.setdefault("config", {})
sphinx_config["exclude_patterns"] = sphinx_config.get("exclude_patterns", []) + EXTRA_EXCLUDES
sphinx_config["autosummary_generate"] = False

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "_toc.yml").write_text(yaml.safe_dump(toc, sort_keys=False))
(OUT / "_config.yml").write_text(yaml.safe_dump(config, sort_keys=False))

returncode = subprocess.call(
    ["jb", "build", str(DOCS),
     "--config", str(OUT / "_config.yml"),
     "--toc", str(OUT / "_toc.yml"),
     "--path-output", str(OUT)] + sys.argv[1:]
)
if returncode == 0:
    print(f"\nProse-only build (no API reference): {OUT / '_build' / 'html' / 'index.html'}")
sys.exit(returncode)
