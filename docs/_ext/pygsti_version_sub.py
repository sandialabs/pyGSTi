"""Local Sphinx extension: expose the installed pyGSTi version to the docs.

ReadTheDocs (and a local ``pip install``) freezes the ``setuptools_scm`` version
into the package metadata at install time, so we can read it back at doc-build
time without needing the git tree. We strip the PEP 440 *local* segment (the
volatile ``+g<hash>.<branch>.<date>`` part) and publish the result as the MyST
substitution ``{{ pygsti_version }}`` (and as Sphinx ``version``/``release``).
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version


def _pygsti_version():
    """Return the public pyGSTi version, e.g. ``0.9.14.1.post1.dev99``."""
    raw = None
    try:
        raw = _dist_version("pyGSTi")
    except PackageNotFoundError:
        try:
            import pygsti

            raw = getattr(pygsti, "__version__", None)
        except Exception:
            raw = None
    if not raw:
        return "unknown"
    # Drop the PEP 440 local version segment (everything after "+").
    return raw.split("+", 1)[0]


def _on_config_inited(app, config):
    ver = _pygsti_version()
    subs = dict(getattr(config, "myst_substitutions", None) or {})
    subs.setdefault("pygsti_version", ver)
    config.myst_substitutions = subs
    # Keep Sphinx's own version/release in sync for any |release| usage.
    config.version = ver
    config.release = ver


def setup(app):
    app.connect("config-inited", _on_config_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
