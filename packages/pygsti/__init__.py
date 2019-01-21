# *****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" A Python implementation of LinearOperator Set Tomography """

#Import the most important/useful routines of each module/sub-package
# into the package namespace
from ._version import __version__

#TEMPORARY WARNING B/C OF BACKCOMPAT BREAKING
import os as _os
import logging as _logging
val = _os.environ.get('PYGSTI_BACKCOMPAT_WARNING',None)

if val not in ("0","False","FALSE","false","No","no","NO"):
    _logging.warning(("\n"
                 "Welcome to pygsti version 0.9.7!\nThere have been some major changes between this "
                 "version and 0.9.6 - ones that break backward compatibility.  If you're trying "
                 "to run an old script and nothing works, DON'T PANIC; we've tried to make the transition "
                 "easy.  More often then not, you can just run `pyGSTi/scripts/upgrade2v0.9.7.py` on your old "
                 "script or notebook files and you'll be up and running again.  For more information, see "
                 "the pyGSTi FAQ.ipynb.\n\nIf this warning annoys you run:\n"
                 "  `export PYGSTI_BACKCOMPAT_WARNING=0` from the command line or\n"
                 "  `import os; os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'` in a script or\n"
                 "     notebook *before* importing pygsti and the the madness will stop."))

from . import algorithms as alg
from . import construction as cst
from . import objects as obj
from . import report as rpt

from .algorithms.core import *
from .algorithms.gaugeopt import *
from .algorithms.contract import *
from .algorithms.grammatrix import *
from .construction.gateconstruction import * # *_qubit_gate fns
from .objects import Basis
from .tools import *
from .drivers import *

#NUMPY BUG FIX (imported from tools)
from .tools.compattools import _numpy14einsumfix
_numpy14einsumfix()
