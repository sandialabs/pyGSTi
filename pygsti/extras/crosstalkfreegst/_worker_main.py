#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Entry point of a :class:`~pygsti.extras.crosstalkfreegst.parallel.LanePool` worker process.

Kept apart from :mod:`.parallel`, which the package imports, so that running this module with
``python -m`` does not import the same module twice.
"""
import sys

from multiprocessing.connection import Client

from pygsti.extras.crosstalkfreegst.parallel import _worker

if __name__ == '__main__':
    _worker(Client(sys.argv[1], authkey=bytes.fromhex(sys.argv[2])))
