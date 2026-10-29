"""
Tests for pygsti.tools._static_tls: explaining imports that fail because glibc's static TLS
reserve is exhausted.

The real failure needs particular shared libraries loaded in a particular process, so these tests
fake it with a meta path finder whose loader raises the dynamic loader's error message.
"""
import importlib.abc
import importlib.util
import sys
import warnings

import pytest

from pygsti.tools import _static_tls
from pygsti.tools._qiskit_interop import check_qiskit_version
from pygsti.tools.exceptions import StaticTLSImportError, StaticTLSWarning

GLIBC_ERROR = ("/opt/site-packages/qiskit_aer.libs/libgomp-a34b3233.so.1.0.0: "
               "cannot allocate memory in static TLS block")
FAKE_PACKAGE = '_pygsti_fake_static_tls_package'


class _FailingLoader(importlib.abc.Loader):
    def __init__(self, message):
        self.message = message

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(self.message)


class _FailingFinder(importlib.abc.MetaPathFinder):
    def __init__(self, names, message):
        self.names = set(names)
        self.message = message

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names:
            return importlib.util.spec_from_loader(fullname, _FailingLoader(self.message))
        return None


@pytest.fixture
def fail_import(monkeypatch):
    """Make importing the given top-level modules fail with `message` (the glibc error by default)."""
    def install(*names, message=GLIBC_ERROR):
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, 'meta_path', [_FailingFinder(names, message)] + sys.meta_path)
    monkeypatch.delenv('GLIBC_TUNABLES', raising=False)
    return install


def test_is_static_tls_error():
    assert _static_tls.is_static_tls_error(ImportError(GLIBC_ERROR))
    assert not _static_tls.is_static_tls_error(ImportError("No module named 'torch'"))
    assert not _static_tls.is_static_tls_error(OSError(GLIBC_ERROR))


def test_import_explains_static_tls_failure(fail_import):
    fail_import(FAKE_PACKAGE)
    with pytest.raises(StaticTLSImportError) as excinfo:
        _static_tls.import_explaining_static_tls(FAKE_PACKAGE)
    message = str(excinfo.value)
    assert message.startswith(f"{FAKE_PACKAGE} is installed but could not be imported")
    assert "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096" in message
    assert GLIBC_ERROR in message
    assert excinfo.value.name == FAKE_PACKAGE
    assert isinstance(excinfo.value.__cause__, ImportError)
    assert str(excinfo.value.__cause__) == GLIBC_ERROR


def test_import_passes_other_errors_through(fail_import):
    fail_import(FAKE_PACKAGE, message="some unrelated import failure")
    with pytest.raises(ImportError, match="some unrelated import failure") as excinfo:
        _static_tls.import_explaining_static_tls(FAKE_PACKAGE)
    assert not isinstance(excinfo.value, StaticTLSImportError)


def test_import_torch_explains_static_tls_failure(fail_import):
    fail_import('torch')
    with pytest.raises(StaticTLSImportError, match=r"^torch is installed but could not be imported"):
        _static_tls.import_torch()


def test_recommendation_keeps_other_tunables(monkeypatch):
    monkeypatch.setenv('GLIBC_TUNABLES', 'glibc.malloc.arena_max=2')
    message = _static_tls.static_tls_explanation('torch', ImportError(GLIBC_ERROR))
    assert "GLIBC_TUNABLES=glibc.malloc.arena_max=2:glibc.rtld.optional_static_tls=4096\n" in message
    assert "started with GLIBC_TUNABLES=glibc.malloc.arena_max=2," in message


def test_recommendation_grows_an_insufficient_setting(monkeypatch):
    monkeypatch.setenv('GLIBC_TUNABLES', 'glibc.rtld.optional_static_tls=4096')
    message = _static_tls.static_tls_explanation('torch', ImportError(GLIBC_ERROR))
    assert "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192\n" in message

    monkeypatch.setenv('GLIBC_TUNABLES', 'glibc.rtld.optional_static_tls=512')
    message = _static_tls.static_tls_explanation('torch', ImportError(GLIBC_ERROR))
    assert "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096\n" in message


def test_warn_if_static_tls():
    with pytest.warns(StaticTLSWarning, match=r"^qiskit_aer is installed but could not be imported"):
        _static_tls.warn_if_static_tls(ImportError(GLIBC_ERROR), 'qiskit_aer')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _static_tls.warn_if_static_tls(ImportError("No module named 'qiskit_aer'"), 'qiskit_aer')


@pytest.mark.parametrize('required', [True, False])
def test_check_qiskit_version_explains_static_tls_failure(fail_import, required):
    # An installed qiskit that can't be loaded is not a missing dependency, so this raises
    # even when `required` is False.
    fail_import('qiskit')
    with pytest.raises(StaticTLSImportError, match=r"^qiskit is installed but could not be imported"):
        check_qiskit_version('test', required=required)


def test_torch_forward_simulator_explains_static_tls_failure(fail_import):
    from pygsti.forwardsims import TorchForwardSimulator
    if not TorchForwardSimulator.ENABLED:
        pytest.skip("PyTorch is not installed.")
    fail_import('torch')
    with pytest.raises(StaticTLSImportError, match=r"^torch is installed but could not be imported"):
        TorchForwardSimulator()
