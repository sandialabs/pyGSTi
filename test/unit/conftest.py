# pytest configuration

# https://stackoverflow.com/a/75438209 for making pytest work with VSCode debugging better
import sys
import pytest

def is_debugging():
    if 'debugpy' in sys.modules:
        return True
    return False

# enable_stop_on_exceptions if the debugger is running during a test
if is_debugging():
    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value