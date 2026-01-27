""" Interpygate Sub-package """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .core import PhysicalProcess, InterpolatedDenseOp, InterpolatedOpFactory
from .process_tomography import vec, unvec, run_process_tomography

# Note from Riley on September, 2024:
#
#   vec is deprecated, and shouldn't be called anywhere in the codebase.
#
#   unvec is deprecated and replaced with unvec_square; the latter function
#   isn't imported here because we don't want people to access it just from
#   the pygsti.extras.interpygate namespace.
#
#   Ideally we'd remove vec and unvec from the pygsti.extras.interpygate namespace
#   and only have them available in pygsti.extras.interpygate.process_tomography.
#
