from notebookstestcase import _PYGSTI_ROOT, _test_notebooks_in_path

_EXAMPLES_ROOT = _PYGSTI_ROOT / 'jupyter_notebooks' / 'Examples'


def test_examples():
    yield from _test_notebooks_in_path(_EXAMPLES_ROOT)
