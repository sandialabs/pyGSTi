import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from notebookstestcase import _PYGSTI_ROOT, notebooks_in_path, _make_test

# All tutorials to be tested are under this directory
_TUTORIALS_ROOT = _PYGSTI_ROOT / 'jupyter_notebooks' / 'Tutorials'

# File resources to be copied to the workdir before testing
_TUTORIAL_FILES = [
    'tutorial_files/MyCircuits.txt',
    'tutorial_files/timestamped_dataset.txt',
    'tutorial_files/Example_GST_Data'
]


def test_tutorials():
    """
    WARNING: this is a factory function that generates tests for PyTest.
    Unfortunately, that functionality was removed in PyTest 4.1. Quoting from the release notes ...

        issue #3079: Removed support for yield tests - they are fundamentally broken because they
        don’t support fixtures properly since collection and test execution were separated.

    TODO: redesign this part of the testing infrastructure for modern pytest.
    """
    logging.getLogger('traitlets').setLevel(logging.CRITICAL)
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy tutorial file resources
        for f_path in _TUTORIAL_FILES:
            src = _TUTORIALS_ROOT / f_path
            dest = tmp_path / f_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                shutil.copy(src, dest)

        # Emit a test for each notebook
        for nb_path in notebooks_in_path(_TUTORIALS_ROOT):
            yield _make_test(nb_path, tmp_path, _TUTORIALS_ROOT)
