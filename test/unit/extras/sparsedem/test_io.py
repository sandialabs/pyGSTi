import pytest
import numpy as np
import stim

from pygsti.extras.sparsedem.io import (
    dem_from_str,
    dem_to_dict,
    dem_from_dict,
    dem_to_event_probabilities,
    dem_from_event_probabilities,
)

def test_dem_to_dict():
    dem_str = """
    error(0.02) D0
    error(0.03) D0 D1
    error(0.05) D0 D1 D2
    """
    expected = {1: 0.02, 3: 0.03, 7: 0.05}
    dem = dem_from_str(dem_str)
    result = dem_to_dict(dem)
    assert result == expected

    dem_str = """
    error(0.02) D0
    error(0.05) D0
    """
    p1 = 0.02; p2 = 0.05
    p_flip = (1-p1)*p2 + p1*(1-p2)
    expected = {1: p_flip}
    dem = dem_from_str(dem_str)
    result = dem_to_dict(dem)
    assert result == expected

def test_dem_from_dict():
    dem_dict = {1: 0.02, 3: 0.03, 7: 0.05}
    dem = dem_from_dict(dem_dict)
    expected = dem_from_str("""
    error(0.02) D0
    error(0.03) D0 D1
    error(0.05) D0 D1 D2
    """)
    assert dem.approx_equals(expected, atol=0.0)

def test_dem_to_event_probabilities():
    p0 = np.random.random() / 10
    p1 = np.random.random() / 10
    p01 = np.random.random() / 10
    dem_str = f"""
    error({p0}) D0
    error({p1}) D1
    error({p01}) D0 D1
    """
    dem = dem_from_str(dem_str)
    expected = np.array([0, p0, p1, p01])
    result = dem_to_event_probabilities(dem)
    assert np.allclose(result, expected, atol=1e-8)

def test_dem_from_event_probabilities():
    p0 = np.random.random() / 10
    p1 = np.random.random() / 10
    p01 = np.random.random() / 10
    dem_str = f"""
    error({p0}) D0
    error({p1}) D1
    error({p01}) D0 D1
    """
    dem = dem_from_str(dem_str)
    probs = dem_to_event_probabilities(dem)

    # Test unmasked
    inferred = dem_from_event_probabilities(probs)
    assert dem.__repr__() == inferred.__repr__()

    # Test masked
    inferred_masked = dem_from_event_probabilities(probs[1:], [1, 2, 3])
    assert dem.__repr__() == inferred_masked.__repr__()
