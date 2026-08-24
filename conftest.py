"""Repo-wide pytest configuration.

This file is the rootdir-level hook point for the whole repository: anything
that has to apply to every suite at once belongs here. At the moment it carries
exactly one such concern — keeping protocol checkpoints out of the source tree —
which lives in the ``CheckpointRedirect`` namespace below.

New concerns should get their own namespace class rather than adding free
functions here, so the pytest hooks at the bottom of the file stay a short,
readable index of everything this conftest does.
"""
import functools
import inspect
import itertools
import os
import pathlib
import shutil
import tempfile


class CheckpointRedirect:
    """Keeps protocol checkpoints out of the source tree during test runs.

    ``GateSetTomography``, ``StandardGST``, ``ModelTest`` and ``IBMQExperiment``
    each default their checkpoint location to a directory *relative to the
    current working directory* (``./gst_checkpoints/``,
    ``./standard_gst_checkpoints/``, ``./model_test_checkpoints/``,
    ``./ibmqexperiment_checkpoint/``). That's a reasonable default for someone
    running a fit interactively, but under pytest it means ~50 tests deposit
    checkpoint JSON wherever pytest happened to be invoked from — normally the
    repo root — and a full notebook run scatters ~150 MB across the
    ``docs/markdown`` source directories.

    This redirects only the *default*: when a caller passes
    ``checkpoint_path=None`` (i.e. expresses no opinion) and hasn't disabled
    checkpointing, the path is rewritten into a throwaway directory outside the
    source tree. Callers that pass an explicit ``checkpoint_path`` are untouched,
    so the tests that deliberately exercise checkpoint write/resume — notably
    ``test/test_packages/drivers/test_drivers.py`` — keep testing exactly what
    they tested before. Checkpointing itself still runs; only the destination
    changes.

    Disable with ``--checkpoint-redirect=off``. Set
    ``PYGSTI_TEST_KEEP_CHECKPOINTS=1`` to leave the directory behind for
    inspection instead of deleting it at session end.

    Note that this cannot reach the notebook suite: nbval executes each notebook
    in a separate kernel subprocess, which this process's monkeypatching never
    sees. Notebook checkpoints stay under ``docs/`` and are gitignored.
    """

    OPTION = "--checkpoint-redirect"
    KEEP_ENVIRONMENT_VARIABLE = "PYGSTI_TEST_KEEP_CHECKPOINTS"
    WRAPPED_MARKER = "_pygsti_checkpoint_redirected"
    SAFE_PUNCTUATION = "-_"

    root = None         # pathlib.Path, once `choose_root` has run
    owns_root = False   # True when we created the root and must delete it again
    counter = itertools.count()

    @staticmethod
    def add_option(parser):
        parser.addoption(
            CheckpointRedirect.OPTION, action="store", default="on", choices=["on", "off"],
            help="Redirect default protocol checkpoint paths out of the source tree "
                 "(default: on).")

    @staticmethod
    def enabled(config):
        return config.getoption(CheckpointRedirect.OPTION) == "on"

    @staticmethod
    def choose_root(config):
        """Pick the throwaway directory that all redirected checkpoints go under.

        Prefer pytest's own basetemp when the user asked for one, so checkpoints
        follow the same lifecycle as everything else pytest writes. Otherwise use
        a private temp dir we clean up ourselves. Either way each xdist worker
        runs the configure hook separately and gets its own directory, so there
        is no contention.
        """
        basetemp = config.getoption("basetemp", None)
        if basetemp:
            CheckpointRedirect.root = pathlib.Path(basetemp) / "pygsti_checkpoints"
            CheckpointRedirect.root.mkdir(parents=True, exist_ok=True)
            CheckpointRedirect.owns_root = False
            return

        private_directory = tempfile.mkdtemp(prefix="pygsti-test-checkpoints-")
        CheckpointRedirect.root = pathlib.Path(private_directory)
        CheckpointRedirect.owns_root = True

    @staticmethod
    def sanitize(name):
        """Make `name` safe as a path stem.

        Dots matter: the protocols run the path through
        ``Path(...).with_suffix('')``, which would silently truncate a stem
        containing one.
        """
        def safe(character):
            keep = character.isalnum() or character in CheckpointRedirect.SAFE_PUNCTUATION
            return character if keep else "_"

        sanitized = "".join(safe(character) for character in str(name))
        return sanitized or "checkpoint"

    @staticmethod
    def fresh_path(kind, name, as_directory):
        """A unique checkpoint destination under the throwaway root.

        The protocols expect ``{dir}/{stem}`` and create ``{dir}`` themselves, so
        we don't create anything here. ``IBMQExperiment`` instead wants a
        directory that does *not* yet exist (it refuses to clobber an existing
        checkpoint), hence ``as_directory``.
        """
        index = next(CheckpointRedirect.counter)
        directory = CheckpointRedirect.root / f"{kind}_{index}"
        if as_directory:
            return str(directory)

        stem = CheckpointRedirect.sanitize(name or kind)
        return str(directory / stem)

    @staticmethod
    def wrap(owner, attribute_name, kind, as_directory=False):
        """Wrap ``owner.attribute_name`` so a ``checkpoint_path`` of None becomes a temp path."""
        original = getattr(owner, attribute_name, None)
        if original is None:
            return
        if getattr(original, CheckpointRedirect.WRAPPED_MARKER, False):
            return

        signature = inspect.signature(original)
        if "checkpoint_path" not in signature.parameters:
            return

        wrapper = CheckpointRedirect.build_wrapper(original, signature, kind, as_directory)
        setattr(owner, attribute_name, wrapper)

    @staticmethod
    def build_wrapper(original, signature, kind, as_directory):
        """The replacement callable installed by `wrap`."""
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments

            # Test on the *value*, not on whether the argument was supplied:
            # `pygsti/drivers/longsequence.py` forwards `checkpoint_path=None`
            # explicitly, and that accounts for most of the calls we care about.
            caller_chose_a_path = arguments.get("checkpoint_path") is not None
            checkpointing_is_off = bool(arguments.get("disable_checkpointing", False))
            if caller_chose_a_path or checkpointing_is_off:
                return original(*bound.args, **bound.kwargs)

            # `self` carries the protocol name, which makes the temp tree readable.
            protocol = args[0] if args else None
            name = getattr(protocol, "name", None)
            arguments["checkpoint_path"] = CheckpointRedirect.fresh_path(kind, name, as_directory)
            return original(*bound.args, **bound.kwargs)

        setattr(wrapper, CheckpointRedirect.WRAPPED_MARKER, True)
        return wrapper

    @staticmethod
    def find_ibmq_experiment():
        """``IBMQExperiment``, or None when qiskit isn't installed."""
        try:
            from pygsti.extras.ibmq.ibmqexperiment import IBMQExperiment
        except ImportError:
            return None
        return IBMQExperiment

    @staticmethod
    def install(config):
        if not CheckpointRedirect.enabled(config):
            return

        CheckpointRedirect.choose_root(config)

        from pygsti.protocols.gst import GateSetTomography, StandardGST
        from pygsti.protocols.modeltest import ModelTest

        CheckpointRedirect.wrap(GateSetTomography, "run", "gst")
        CheckpointRedirect.wrap(StandardGST, "run", "standard_gst")
        CheckpointRedirect.wrap(ModelTest, "run", "model_test")

        ibmq_experiment = CheckpointRedirect.find_ibmq_experiment()
        if ibmq_experiment is None:
            return

        CheckpointRedirect.wrap(
            ibmq_experiment, "__init__", "ibmqexperiment", as_directory=True)

    @staticmethod
    def report_header():
        if CheckpointRedirect.root is None:
            return None
        return f"protocol checkpoints redirected to: {CheckpointRedirect.root}"

    @staticmethod
    def remove_root():
        if CheckpointRedirect.root is None:
            return
        if not CheckpointRedirect.owns_root:
            return
        if os.environ.get(CheckpointRedirect.KEEP_ENVIRONMENT_VARIABLE):
            return

        shutil.rmtree(CheckpointRedirect.root, ignore_errors=True)


def pytest_addoption(parser):
    CheckpointRedirect.add_option(parser)


def pytest_configure(config):
    CheckpointRedirect.install(config)


def pytest_report_header(config):
    return CheckpointRedirect.report_header()


def pytest_unconfigure(config):
    CheckpointRedirect.remove_root()
