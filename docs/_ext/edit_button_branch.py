"""Local Sphinx extension: pin the source/edit buttons to a fixed branch.

The hosted docs are built from an auto-generated ``develop-with-notebooks``
branch that carries committed ``.ipynb`` companions, so the Colab/Binder launch
links resolve to real notebooks on GitHub. sphinx-book-theme derives a single
``repository_branch`` for *both* the launch buttons *and* the "Show source" /
"Suggest edit" buttons, so without intervention those edit links would send
contributors to the throwaway build branch (where hand edits are clobbered on the
next sync).

This extension rewrites only the source/edit button URLs back to the canonical
source branch named by the ``pygsti_edit_branch`` config value, leaving the
launch buttons on the build branch.

Retargeting the branch is not sufficient on its own. The build branch renders each
notebook page from its ``.ipynb``, so that is the suffix sphinx-book-theme puts in
the URL -- but ``.ipynb`` is gitignored on the canonical branch, where the page
exists only as the jupytext-paired ``.md`` it was generated from. A branch-only
rewrite would therefore point every notebook page's Show source / Suggest edit
button at a 404. The suffix is swapped along with the branch, which is the same
translation: build-branch URL to edit-branch URL.

The two rewrites are independently conditional. The *branch* rewrite is a no-op
when ``pygsti_edit_branch`` is unset or already equals the build branch. The
*suffix* rewrite is not conditional on either: a page's canonical source is its
``.md`` on every branch, so a build that renders from ``.ipynb`` while pointing
its buttons at a branch where only the ``.md`` is committed needs the swap
regardless of which branch that is. A local preview is exactly that case -- the
paired ``.ipynb`` exists in the working tree, so pages render from it, while
``repository.branch`` is still ``develop``, where it is gitignored.

The whole extension is a no-op when the source/edit buttons are not enabled.
"""

from pydata_sphinx_theme.utils import get_theme_options_dict
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

#: Labels (set by sphinx_book_theme) of the buttons whose URLs embed the branch.
_BRANCHED_BUTTON_LABELS = ("source-file-button", "source-edit-button")


def _to_paired_source(url):
    """Point an ``.ipynb`` URL at the ``.md`` it is generated from.

    The suffix sits at the end of the path, which may be followed by a query
    (sphinx-book-theme appends ``?plain=1`` to Show source URLs), so split the
    query off before touching it.
    """
    path, sep, query = url.partition("?")
    if path.endswith(".ipynb"):
        path = path[: -len(".ipynb")] + ".md"
    return path + sep + query


def _rewrite(buttons, build_branch, edit_branch):
    """Recursively rewrite branch and suffix in source/edit button URLs.

    Launch buttons (Binder/Colab) are deliberately untouched: those *should*
    track the build branch and its committed ``.ipynb``.
    """
    retarget = bool(build_branch and edit_branch and build_branch != edit_branch)
    for button in buttons:
        if button.get("type") == "group":
            _rewrite(button.get("buttons", []), build_branch, edit_branch)
        elif button.get("label") in _BRANCHED_BUTTON_LABELS:
            url = button.get("url", "")
            if retarget:
                for segment in ("/edit/", "/blob/"):
                    url = url.replace(
                        f"{segment}{build_branch}/", f"{segment}{edit_branch}/"
                    )
            button["url"] = _to_paired_source(url)


def _on_html_page_context(app, pagename, templatename, context, doctree):
    header_buttons = context.get("header_buttons")
    if not header_buttons:
        return
    _rewrite(
        header_buttons,
        get_theme_options_dict(app).get("repository_branch"),
        app.config.pygsti_edit_branch,
    )


def setup(app):
    app.add_config_value("pygsti_edit_branch", None, "html")
    # priority > 501 so this runs after sphinx_book_theme.add_source_buttons,
    # which is connected to html-page-context at priority 501.
    app.connect("html-page-context", _on_html_page_context, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
