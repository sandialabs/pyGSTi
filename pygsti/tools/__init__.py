"""
pyGSTi Tools Python Package
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .basistools import *
from .chi2fns import *
from .edesigntools import *
from .exceptions import *
from . import graphcoloring
from .hypothesis import *
# Import the most important/useful routines of each module into
# the package namespace
from .jamiolkowski import *
from .legacytools import *
from .likelihoodfns import *
from pygsti.tools._leakage import RELOCATED_NAMES as _LEAKAGE_NAMES, get_leakage_shim as _get_leakage_shim
# ^ Leakage functions moved to pygsti.leakage.  They remain accessible here for
#   backward compatibility, but accessing them via pygsti.tools is deprecated: the
#   shims in pygsti.tools._leakage emit a DeprecationWarning when called.  Using a
#   module-level __getattr__ (PEP 562) keeps these names out of the tools namespace
#   until requested, so the shim is what callers actually get.
#
from .lindbladtools import *
from .listtools import *
from .matrixmod2 import *
from .matrixtools import *
from .mpitools import parallel_apply, mpi4py_comm
from .mpitools import resolve_mpiexec, compute_blas_threads, write_mpi_runner_artifacts, build_slurm_script
from .mptools import starmap_with_kwargs
from .nameddict import NamedDict
from .optools import *
from .gatetools import *
from .opttools import *
from .pdftools import *
from .rbtheory import *
from .slicetools import *
from .symplectic import *
from .typeddict import TypedDict


def __getattr__(name):
    # PEP 562 hook: serve deprecation shims for leakage routines relocated to
    # pygsti.leakage.  Only fires for names not otherwise defined in this module.
    if name in _LEAKAGE_NAMES:
        return _get_leakage_shim(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LEAKAGE_NAMES))
