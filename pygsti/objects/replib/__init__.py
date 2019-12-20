"""Implementations of calculation "representation" objects"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

warn_msg = """
An optimized Cython-based implementation of `{module}` is available as
an extension, but couldn't be imported. This might happen if the
extension has not been built. `pip install cython`, then reinstall
pyGSTi to build Cython extensions. Alternatively, setting the
environment variable `PYGSTI_NO_CYTHON_WARNING` will suppress this
message.
""".format(module=__name__)

try:
    # Import cython implementation if it's been built...
    from .fastreplib import *
except ImportError:
    # ... If not, fall back to the python implementation, with a warning.
    import os as _os
    import warnings as _warnings

    if 'PYGSTI_NO_CYTHON_WARNING' not in _os.environ:
        _warnings.warn(warn_msg)

    from .slowreplib import *
