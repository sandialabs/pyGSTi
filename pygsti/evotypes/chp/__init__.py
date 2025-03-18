"""
The CHP ("chp") evolution type
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

chpexe = None


def chpexe_path():
    from pathlib import Path as _Path
    if chpexe is None:
        raise ValueError(("To use 'chp' evotype, please set `pygsti.evotypes.chp.chpexe`"
                          "to the path to your chp executable."))
    return _Path(chpexe)


from .povmreps import *
from .effectreps import *
from .opreps import *
from .statereps import *
