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
launch buttons on the build branch. It is a no-op when:
  * ``pygsti_edit_branch`` is unset, or
  * the build branch already equals ``pygsti_edit_branch`` (e.g. local builds or
    a plain ``develop`` build), or
  * the source/edit buttons are not enabled.
"""

from pydata_sphinx_theme.utils import get_theme_options_dict
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

#: Labels (set by sphinx_book_theme) of the buttons whose URLs embed the branch.
_BRANCHED_BUTTON_LABELS = ("source-file-button", "source-edit-button")


def _rewrite(buttons, build_branch, edit_branch):
    """Recursively rewrite branch segments in source/edit button URLs."""
    for button in buttons:
        if button.get("type") == "group":
            _rewrite(button.get("buttons", []), build_branch, edit_branch)
        elif button.get("label") in _BRANCHED_BUTTON_LABELS:
            url = button.get("url", "")
            for segment in ("/edit/", "/blob/"):
                url = url.replace(
                    f"{segment}{build_branch}/", f"{segment}{edit_branch}/"
                )
            button["url"] = url


def _on_html_page_context(app, pagename, templatename, context, doctree):
    edit_branch = app.config.pygsti_edit_branch
    if not edit_branch:
        return
    build_branch = get_theme_options_dict(app).get("repository_branch")
    if not build_branch or build_branch == edit_branch:
        return
    header_buttons = context.get("header_buttons")
    if not header_buttons:
        return
    _rewrite(header_buttons, build_branch, edit_branch)


def setup(app):
    app.add_config_value("pygsti_edit_branch", None, "html")
    # priority > 501 so this runs after sphinx_book_theme.add_source_buttons,
    # which is connected to html-page-context at priority 501.
    app.connect("html-page-context", _on_html_page_context, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
