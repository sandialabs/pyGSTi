"""
pyGSTi Input/Output Python Package
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import json
# Import the most important/useful routines of each module into
# the package namespace
from .circuitparser import CircuitParser
#from .legacyio import enable_no_cython_unpickling
#from .legacyio import enable_old_object_unpickling  # , disable_old_object_unpickling
from .loaders import *
from .metadir import *
from .stdinput import *
from .writers import *

#Users may not have msgpack, which is fine.
try: from . import msgpack
except ImportError: pass
