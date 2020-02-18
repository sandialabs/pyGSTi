#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Dynamic package versioning using setuptools_scm """

__version__ = "unknown"

try:
    from pkg_resources import get_distribution, DistributionNotFound
    __version__ = get_distribution('pygsti').version
except DistributionNotFound:
    # package not installed
    try:
        from inspect import getfile, currentframe
        this_file = getfile(currentframe())
        from os.path import abspath
        from setuptools_scm import get_version

        __version__ = get_version(root='../..', relative_to=abspath(this_file))
    except Exception:
        pass
