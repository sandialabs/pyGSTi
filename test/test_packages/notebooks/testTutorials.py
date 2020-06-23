from notebookstestcase import _PYGSTI_ROOT, notebooks_in_path, _make_test
from nose.plugins.attrib import attr
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import logging


# All tutorials to be tested are under this directory
_TUTORIALS_ROOT = _PYGSTI_ROOT / 'jupyter_notebooks' / 'Tutorials'

# File resources to be copied to the workdir before testing
_TUTORIAL_FILES = [
    'tutorial_files/MyCircuits.txt',
    'tutorial_files/timestamped_dataset.txt',
    'tutorial_files/Example_GST_Data'
]


def test_tutorials():
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
