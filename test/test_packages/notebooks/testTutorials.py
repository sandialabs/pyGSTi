from notebookstestcase import _PYGSTI_ROOT, _test_notebooks_in_path

_TUTORIALS_ROOT = _PYGSTI_ROOT / 'jupyter_notebooks' / 'Tutorials'


def test_tutorials():
    yield from _test_notebooks_in_path(_TUTORIALS_ROOT)
