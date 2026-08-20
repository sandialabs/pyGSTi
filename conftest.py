"""Repo-wide pytest configuration.

Keeps protocol checkpoints out of the source tree during test runs.

``GateSetTomography``, ``StandardGST``, ``ModelTest`` and ``IBMQExperiment``
each default their checkpoint location to a directory *relative to the current
working directory* (``./gst_checkpoints/``, ``./standard_gst_checkpoints/``,
``./model_test_checkpoints/``, ``./ibmqexperiment_checkpoint/``). That's a
reasonable default for someone running a fit interactively, but under pytest it
means ~50 tests deposit checkpoint JSON wherever pytest happened to be invoked
from — normally the repo root — and a full notebook run scatters ~150 MB across
the ``docs/markdown`` source directories.

This plugin redirects only the *default*: when a caller passes
``checkpoint_path=None`` (i.e. expresses no opinion) and hasn't disabled
checkpointing, the path is rewritten into a throwaway directory outside the
source tree. Callers that pass an explicit ``checkpoint_path`` are untouched, so
the tests that deliberately exercise checkpoint write/resume — notably
``test/test_packages/drivers/test_drivers.py`` — keep testing exactly what they
tested before. Checkpointing itself still runs; only the destination changes.

Disable with ``--checkpoint-redirect=off``. Set
``PYGSTI_TEST_KEEP_CHECKPOINTS=1`` to leave the directory behind for inspection
instead of deleting it at session end.

Note that this cannot reach the notebook suite: nbval executes each notebook in
a separate kernel subprocess, which this process's monkeypatching never sees.
Notebook checkpoints stay under ``docs/`` and are gitignored.
"""
import functools
import inspect
import itertools
import pathlib
import shutil
import tempfile

_root = []            # [pathlib.Path] once configured
_cleanup = [False]
_counter = itertools.count()


def _sanitize(name):
    """Make `name` safe as a path stem.

    Dots matter: the protocols run the path through ``Path(...).with_suffix('')``,
    which would silently truncate a stem containing one.
    """
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)) or "checkpoint"


def _fresh_path(kind, name=None, as_directory=False):
    """A unique checkpoint destination under the throwaway root.

    The protocols expect ``{dir}/{stem}`` and create ``{dir}`` themselves, so we
    don't create anything here. ``IBMQExperiment`` instead wants a directory that
    does *not* yet exist (it refuses to clobber an existing checkpoint), hence
    ``as_directory``.
    """
    n = next(_counter)
    if as_directory:
        return str(_root[0] / f"{kind}_{n}")
    return str(_root[0] / f"{kind}_{n}" / _sanitize(name or kind))


def _redirect(owner, attrname, kind, as_directory=False):
    """Wrap ``owner.attrname`` so a ``checkpoint_path`` of None becomes a temp path."""
    fn = getattr(owner, attrname, None)
    if fn is None or getattr(fn, "_pygsti_checkpoint_redirected", False):
        return
    sig = inspect.signature(fn)
    if "checkpoint_path" not in sig.parameters:
        return

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if (arguments.get("checkpoint_path") is None
                and not arguments.get("disable_checkpointing", False)):
            # `self` carries the protocol name, which makes the temp tree readable.
            name = getattr(args[0], "name", None) if args else None
            arguments["checkpoint_path"] = _fresh_path(kind, name, as_directory)
        return fn(*bound.args, **bound.kwargs)

    wrapper._pygsti_checkpoint_redirected = True
    setattr(owner, attrname, wrapper)


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint-redirect", action="store", default="on", choices=["on", "off"],
        help="Redirect default protocol checkpoint paths out of the source tree "
             "(default: on).")


def pytest_configure(config):
    if config.getoption("--checkpoint-redirect") == "off":
        return

    # Prefer pytest's own basetemp when the user asked for one, so checkpoints
    # follow the same lifecycle as everything else pytest writes. Otherwise use a
    # private temp dir we clean up ourselves. Either way each xdist worker runs
    # this hook separately and gets its own directory, so there's no contention.
    basetemp = config.getoption("basetemp", None)
    if basetemp:
        _root.append(pathlib.Path(basetemp) / "pygsti_checkpoints")
        _root[0].mkdir(parents=True, exist_ok=True)
    else:
        _root.append(pathlib.Path(tempfile.mkdtemp(prefix="pygsti-test-checkpoints-")))
        _cleanup[0] = True

    from pygsti.protocols.gst import GateSetTomography, StandardGST
    from pygsti.protocols.modeltest import ModelTest

    _redirect(GateSetTomography, "run", "gst")
    _redirect(StandardGST, "run", "standard_gst")
    _redirect(ModelTest, "run", "model_test")

    try:  # optional dependency: only present when qiskit is installed
        from pygsti.extras.ibmq.ibmqexperiment import IBMQExperiment
    except ImportError:
        pass
    else:
        _redirect(IBMQExperiment, "__init__", "ibmqexperiment", as_directory=True)


def pytest_report_header(config):
    if _root:
        return f"protocol checkpoints redirected to: {_root[0]}"
    return None


def pytest_unconfigure(config):
    import os
    if _root and _cleanup[0] and not os.environ.get("PYGSTI_TEST_KEEP_CHECKPOINTS"):
        shutil.rmtree(_root[0], ignore_errors=True)
