import pytest

from pygsti.baseobjs.statespace import QubitSpace
from pygsti.evotypes import Evotype


@pytest.mark.parametrize('name, nqubits, expected', [
    ('densitymx_slow', 4, True),
    ('densitymx_slow', 5, False),
    ('statevec_slow', 4, False),
    ('default', 4, True),
])
def test_prefer_dense_reps_threshold(monkeypatch, name, nqubits, expected):
    if name == 'default':
        monkeypatch.setattr(Evotype, 'default_evotype', 'densitymx_slow')
    evotype = Evotype.cast(name, state_space=QubitSpace(nqubits))
    assert evotype.prefer_dense_reps is expected
