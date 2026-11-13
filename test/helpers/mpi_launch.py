"""Launcher arguments for tests that start MPI jobs through mpiexec.

Tests often ask for more ranks than a CI runner has cores. Open MPI refuses
that unless it is given ``--oversubscribe``, a flag that Intel MPI (Hydra) and
MPICH don't recognise, so we only pass it when Open MPI is detected.
"""

import os
import shutil
import subprocess


def find_mpi_launcher():
    """Return the path of ``mpiexec`` (or ``mpirun``) on PATH, or None."""
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _version_output(executable):
    try:
        completed = subprocess.run(
            [executable, '--version'], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ''
    return completed.stdout + completed.stderr


def is_open_mpi(launcher):
    """Return True if ``launcher`` belongs to an Open MPI installation.

    ``ompi_info --version`` is checked first, looking only beside the launcher
    so that an unrelated Open MPI elsewhere on PATH can't be mistaken for it.
    The launcher's own ``--version`` is the fallback, but it is not enough on
    its own: Ubuntu 26.04's Open MPI 5 package omits PRRTE's help files, and
    there ``mpiexec --version`` prints a "couldn't open the help file" message
    that never mentions Open MPI.
    """
    directories = dict.fromkeys(
        os.path.dirname(path) for path in (launcher, os.path.realpath(launcher)))
    for directory in directories:
        ompi_info = os.path.join(directory, 'ompi_info')
        if os.path.isfile(ompi_info) and 'Open MPI' in _version_output(ompi_info):
            return True
    output = _version_output(launcher)
    return 'Open MPI' in output or 'OpenRTE' in output


def openmpi_extra_args(launcher=None):
    """Return ``['--oversubscribe']`` for an Open MPI launcher, else ``[]``.

    ``launcher`` defaults to :func:`find_mpi_launcher`.
    """
    launcher = launcher or find_mpi_launcher()
    if launcher is None:
        return []
    return ['--oversubscribe'] if is_open_mpi(launcher) else []
