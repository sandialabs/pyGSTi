import logging

from notebookstestcase import _PYGSTI_ROOT, _test_notebooks_in_path

_EXAMPLES_ROOT = _PYGSTI_ROOT / 'jupyter_notebooks' / 'Examples'


def test_examples():
    """
    WARNING: this is a factory function that generates tests for PyTest.
    Unfortunately, that functionality was removed in PyTest 4.1. Quoting from the release notes ...

        issue #3079: Removed support for yield tests - they are fundamentally broken because they
        don’t support fixtures properly since collection and test execution were separated.

    TODO: redesign this part of the testing infrastructure for modern pytest.
    """
    logging.getLogger('traitlets').setLevel(logging.CRITICAL)
    yield from _test_notebooks_in_path(_EXAMPLES_ROOT)
