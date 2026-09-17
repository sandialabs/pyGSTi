"""
Backward-compatibility shims for leakage routines relocated to :mod:`pygsti.leakage`.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import warnings as _warnings
from pygsti.tools.exceptions import pyGSTiDeprecationWarning

import pygsti.leakage as _leakage

# Names that used to live in ``pygsti.tools`` and are now re-exported from
# ``pygsti.leakage``.  Calling any of these through ``pygsti.tools`` is
# deprecated; the shims below emit a ``DeprecationWarning`` at call time.
RELOCATED_NAMES = (
    'computational_effect',
    'computational_superkets',
    'tensorized_teststate_density',
    'apply_tensorized_to_teststate',
    'choi_state',
    'subspace_entanglement_fidelity',
    'subspace_jtracedist',
    'computational_projector',
    'subspace_superop_fro_dist',
    'subspace_diamonddist',
    'pop_transport_profile',
    'gate_leakage_profile',
    'gate_seepage_profile',
    'leaky_qubit_model_from_pspec',
    'promote_bb_to_bt',
    'lagoified_gopparams_dicts',
    'std_lago_gopsuite',
    'add_lago_models',
    'construct_leakage_report',
)

_shim_cache = {}


def _make_shim(name):
    target = getattr(_leakage, name)

    @_functools.wraps(target)
    def shim(*args, **kwargs):
        _warnings.warn(
            f"pygsti.tools.{name} is deprecated and will be removed in a future release; "
            f"use pygsti.leakage.{name} instead.",
            pyGSTiDeprecationWarning,
            stacklevel=2,
        )
        return target(*args, **kwargs)

    shim.__doc__ = (
        f"Deprecated alias for :func:`pygsti.leakage.{name}`.\n\n" + (target.__doc__ or "")
    )
    return shim


def get_leakage_shim(name):
    """
    Return a deprecation-warning shim for a leakage routine relocated to ``pygsti.leakage``.

    Parameters
    ----------
    name : str
        The name of the relocated routine.

    Returns
    -------
    callable
        A wrapper that emits a ``DeprecationWarning`` and forwards to
        ``pygsti.leakage.<name>``.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the relocated leakage routines.
    """
    if name not in RELOCATED_NAMES:
        raise AttributeError(name)
    if name not in _shim_cache:
        _shim_cache[name] = _make_shim(name)
    return _shim_cache[name]
