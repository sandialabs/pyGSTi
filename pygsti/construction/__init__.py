"""
pyGSTi Object Construction Python Package
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# import importlib as _importlib
# import sys as _sys
#
# _modelpack_names = [
#     "std1Q_Cliffords",
#     "std1Q_pi4_pi2_XZ",
#     "std1Q_XYI",
#     "std1Q_XY",
#     "std1Q_XYZI",
#     "std1Q_XZ",
#     "std1Q_ZN",
#     "std2Q_XXII",
#     "std2Q_XXYYII",
#     "std2Q_XYCNOT",
#     "std2Q_XYCPHASE",
#     "std2Q_XYI1",
#     "std2Q_XYI2",
#     "std2Q_XYICNOT",
#     "std2Q_XYICPHASE",
#     "std2Q_XYI",
#     "std2Q_XY",
#     "std2Q_XYZICNOT",
#     "stdQT_XYIMS"
# ]
#
# warn_msg = ("`pygsti.construction.{name}` has been moved to `pygsti.modelpacks.legacy`. Future versions of pyGSTi "
#             "will drop support for importing this module from the deprecated path.")
#
# if _sys.version_info < (3, 7):
#     # Note that this will make ALL attribute lookup substantially slower
#     replacement_map = {name: lambda: _importlib.import_module('pygsti.modelpacks.legacy.{}'.format(name))
#                        for name in _modelpack_names}
#     from ..tools.legacytools import deprecate_imports
#     deprecate_imports(__name__, replacement_map, warn_msg)
# else:
#     # Module-level __getattr__ does exactly what we need but was only introduced in python 3.7
#     from warnings import warn
#
#     def __getattr__(name):
#         if name in _modelpack_names:
#             warn(warn_msg.format(name=name))
#             return _importlib.import_module('pygsti.modelpacks.legacy.{}'.format(name))
#         else:
#             raise AttributeError("cannot import name '{name}' from '{module}'".format(name=name, module=__name__))
