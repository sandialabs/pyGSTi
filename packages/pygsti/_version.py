#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
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
