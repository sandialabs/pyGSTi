import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from tempfile import TemporaryDirectory
from nose.plugins.attrib import attr


_DEFAULT_IPYNB_VERSION = 4
_DEFAULT_TIMEOUT = 60 * 20  # 20 minutes

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

    # Some notebooks may generate and automatically open reports in a web browser.
    # This is inconvenient in an automated test suite, so let's disable it.
    # Overwriting $BROWSER with a dummy command will keep the notebook
    # kernel from being able to open a web browser on Linux.
    # TODO find platform-neutral solution to suppress webbrowser.open
    browser = os.environ.get('BROWSER', None)
    os.environ['BROWSER'] = 'echo %s'
    ep.preprocess(nb, resources=resources)
    os.environ['BROWSER'] = browser


def notebooks_in_path(root_path, ignore_checkpoints=True):
    for path, _, files in os.walk(root_path, topdown=True, followlinks=False):
        if '.ipynb_checkpoints' not in path:
            for ipynb in filter(lambda s: s.endswith('.ipynb'), sorted(files)):
                yield Path(path) / ipynb


def _make_test(nb_path, tmp_path, root_path):
    rel_path = nb_path.relative_to(root_path)
    workdir = tmp_path / rel_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    @attr(description="Running notebook {}".format(rel_path))
    def test_wrapper():
        run_notebook(nb_path, workdir)
    return test_wrapper


def _test_notebooks_in_path(root_path):
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for nb_path in notebooks_in_path(root_path):
            yield _make_test(nb_path, tmp_path, root_path)
