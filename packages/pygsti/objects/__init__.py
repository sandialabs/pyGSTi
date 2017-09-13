from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Gate Set Tomography Objects Python Package """

#Import the most important/useful routines of each module into
# the package namespace

#Import Objects at package level
from .confidenceregion import ConfidenceRegion
from .dataset import DataSet
from .tddataset import TDDataSet
from .exceptions import *
from .evaltree import EvalTree
from .matrixevaltree import MatrixEvalTree
from .mapevaltree import MapEvalTree
from .gate import Gate
from .gate import GateMatrix
from .gate import LinearlyParameterizedGate
from .gate import FullyParameterizedGate
from .gate import TPParameterizedGate
from .gate import StaticGate
from .gate import EigenvalueParameterizedGate
from .gate import LindbladParameterizedGate
from .spamvec import SPAMVec
from .spamvec import FullyParameterizedSPAMVec
from .spamvec import TPParameterizedSPAMVec
from .spamvec import CPTPParameterizedSPAMVec
from .spamvec import StaticSPAMVec

from .gateset import GateSet
from .gatestring import GateString
from .gatestring import WeightedGateString
from .gatestringstructure import GatestringStructure
from .gatestringstructure import LsGermsStructure
from .multidataset import MultiDataSet
from .spamspec import SpamSpec
from .profiler import Profiler
from .profiler import DummyProfiler
from .datacomparator import DataComparator

from .gaugegroup import FullGaugeGroup, TPGaugeGroup, \
    DiagGaugeGroup, TPDiagGaugeGroup, UnitaryGaugeGroup

#from gigateset import GaugeInvGateSet
#Experimental only: don't import in production pyGSTi
#from gigateset import GaugeInvGateSet

#Functions
from .gate import compose, optimize_gate, finite_difference_deriv_wrt_params
from .verbosityprinter import VerbosityPrinter
from ..tools.smartcache import SmartCache, CustomDigestError, smart_cached

# To prevent circular imports. In all respects, Basis is an object, but it needs to live in tools so that there are no circular imports or backwards dependencies.
# An alternative would be to move certain modules that depend on the Basis object out of tools, but moving the Basis object to tools works fine.
from ..tools import basis
from ..tools.basis import *
