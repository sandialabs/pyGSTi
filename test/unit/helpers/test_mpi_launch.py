"""Tests for Open MPI detection in test.helpers.mpi_launch.

Each test lays out fake MPI binaries in a temporary directory and replaces the
subprocess call with canned ``--version`` output, so no MPI install is needed.
"""

import os

import test.helpers.mpi_launch as mpi_launch

# What Ubuntu 26.04's Open MPI 5.0.10 prints for `mpiexec --version`: it can't
# find PRRTE's help files, so it never says "Open MPI".
UBUNTU_2604_MPIEXEC = """\
--------------------------------------------------------------------------
Sorry!  You were supposed to get help about:
    version
But I couldn't open the help file:
    /usr/lib/x86_64-linux-gnu/prrte3/share/prte/help-prterun.txt: No such file or directory.
Sorry!
--------------------------------------------------------------------------
"""


def _install_fakes(monkeypatch, directory, outputs):
    """Create one empty file per name in `outputs`; `--version` on it prints the mapped text."""
    for name in outputs:
        (directory / name).touch()
    monkeypatch.setattr(
        mpi_launch, '_version_output', lambda executable: outputs[os.path.basename(executable)])
    return str(directory / 'mpiexec')


def test_open_mpi_5_on_ubuntu_2604_is_detected_through_ompi_info(monkeypatch, tmp_path):
    launcher = _install_fakes(monkeypatch, tmp_path, {
        'mpiexec': UBUNTU_2604_MPIEXEC,
        'ompi_info': 'Open MPI v5.0.10\n\nhttps://www.open-mpi.org/community/help/\n',
    })
    assert mpi_launch.openmpi_extra_args(launcher) == ['--oversubscribe']


def test_launcher_version_is_the_fallback_when_there_is_no_ompi_info(monkeypatch, tmp_path):
    launcher = _install_fakes(monkeypatch, tmp_path, {'mpiexec': 'mpiexec (Open MPI) 5.0.10\n'})
    assert mpi_launch.openmpi_extra_args(launcher) == ['--oversubscribe']


def test_open_mpi_4_launcher_is_detected(monkeypatch, tmp_path):
    launcher = _install_fakes(monkeypatch, tmp_path, {'mpiexec': 'mpiexec (OpenRTE) 4.1.6\n'})
    assert mpi_launch.openmpi_extra_args(launcher) == ['--oversubscribe']


def test_mpich_gets_no_extra_arguments(monkeypatch, tmp_path):
    launcher = _install_fakes(monkeypatch, tmp_path, {
        'mpiexec': 'HYDRA build details:\n    Version: 4.2.0\n',
    })
    assert mpi_launch.openmpi_extra_args(launcher) == []


def test_ompi_info_elsewhere_on_path_does_not_count(monkeypatch, tmp_path):
    mpich_dir = tmp_path / 'mpich'
    ompi_dir = tmp_path / 'ompi'
    mpich_dir.mkdir()
    ompi_dir.mkdir()
    (ompi_dir / 'ompi_info').touch()
    launcher = _install_fakes(monkeypatch, mpich_dir, {
        'mpiexec': 'HYDRA build details:\n    Version: 4.2.0\n',
        'ompi_info': 'Open MPI v5.0.10\n',
    })
    (mpich_dir / 'ompi_info').unlink()
    monkeypatch.setenv('PATH', os.pathsep.join([str(mpich_dir), str(ompi_dir)]))
    assert mpi_launch.openmpi_extra_args(launcher) == []


def test_no_launcher_means_no_extra_arguments(monkeypatch):
    monkeypatch.setattr(mpi_launch, 'find_mpi_launcher', lambda: None)
    assert mpi_launch.openmpi_extra_args() == []
