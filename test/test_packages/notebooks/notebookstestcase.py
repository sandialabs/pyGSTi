import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from tempfile import TemporaryDirectory
from nose.plugins.attrib import attr


_DEFAULT_IPYNB_VERSION = 4
_DEFAULT_TIMEOUT = 60 * 10  # 10 minutes

# hardcoded path to pyGSTi root
# XXX change if refactoring
_PYGSTI_ROOT = Path(__file__).absolute().parent.parent.parent.parent


def run_notebook(path, workdir=None, timeout=_DEFAULT_TIMEOUT):
    resources = {
        'metadata': {
            'path': workdir
        }
    } if workdir is not None else None

    with open(path, 'r') as f:
        nb = nbformat.read(f, as_version=_DEFAULT_IPYNB_VERSION)
    ep = ExecutePreprocessor(timeout=timeout)
    ep.preprocess(nb, resources=resources)


def notebooks_in_path(root_path, ignore_checkpoints=True):
    for path, _, files in os.walk(root_path, topdown=True, followlinks=False):
        if '.ipynb_checkpoints' not in path:
            for ipynb in filter(lambda s: s.endswith('.ipynb'), sorted(files)):
                yield Path(path) / ipynb


def _test_notebooks_in_path(root_path):
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for nb_path in notebooks_in_path(root_path):
            rel_path = nb_path.relative_to(root_path)
            workdir = tmp_path / rel_path.parent
            workdir.mkdir(parents=True, exist_ok=True)
            description = "Running notebook {}".format(rel_path)
            yield attr(description=description)(run_notebook), nb_path, workdir
