import logging

from notebookstestcase import _PYGSTI_ROOT, _test_notebooks_in_path

_EXAMPLES_ROOT = _PYGSTI_ROOT / 'jupyter_notebooks' / 'Examples'


def test_examples():
    logging.getLogger('traitlets').setLevel(logging.CRITICAL)
    yield from _test_notebooks_in_path(_EXAMPLES_ROOT)
