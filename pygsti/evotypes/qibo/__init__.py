"""
An evolution type that uses the 3rd-party 'qibo' package.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

densitymx_mode = False
nshots = 1000


def _get_densitymx_mode():
    return densitymx_mode


def _get_nshots():
    return nshots


def _get_minimal_space():
    return minimal_space


minimal_space = 'Hilbert'
from .povmreps import *
from .effectreps import *
from .opreps import *
from .statereps import *
