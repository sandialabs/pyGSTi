"""Drop duplicate ``<script>`` tags from the rendered page context.

``sphinx_thebe`` and ``sphinx_togglebutton`` both call ``app.add_js_file()`` from
``env-before-read-docs``.  Sphinx allows ``add_js_file`` only from ``setup()`` or
``builder-inited``; from a read-phase event it runs again whenever Sphinx reads
docs more than once, and each run appends another copy.

The copies are not equal objects -- the first carries ``filename=None`` and the
second ``filename=''`` -- so Sphinx's own de-duplication does not catch them, and
every page ends up emitting both.  For ``sphinx_togglebutton`` that is harmless,
since its payload is a ``var`` declaration and redeclaring a ``var`` is legal.
For ``sphinx_thebe`` it is not: the payload opens with ``const THEBE_JS_URL``,
and the second copy throws

    SyntaxError: Identifier 'THEBE_JS_URL' has already been declared

on every page of the site, which aborts that script tag.  Thebe itself still
loads from its own ``<script src=...>``, so nothing visibly breaks; it is a
console error on every page.

This reproduces on a stock ``jupyter-book`` build with a default ``_config.yml``,
so it is upstream rather than anything in this repo's configuration.  Until it is
fixed there, collapse the duplicates on the way out: a file-backed script is keyed
by its filename, an inline one by its body, and the first occurrence of each key
wins so that ordering is preserved.
"""
from __future__ import annotations


def _key(script):
    filename = getattr(script, "filename", None)
    if filename:
        return ("file", filename)
    # Inline <script>: sphinx stashes the source in the 'body' attribute.
    return ("body", getattr(script, "attributes", {}).get("body", ""))


def _dedupe_scripts(app, pagename, templatename, context, doctree):
    scripts = context.get("script_files")
    if not scripts:
        return
    seen, kept = set(), []
    for script in scripts:
        key = _key(script)
        if key in seen:
            continue
        seen.add(key)
        kept.append(script)
    if len(kept) != len(scripts):
        context["script_files"] = kept


def setup(app):
    app.connect("html-page-context", _dedupe_scripts)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
