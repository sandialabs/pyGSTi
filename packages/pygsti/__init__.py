# *****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" A Python implementation of Gate Set Tomography """

#Import the most important/useful routines of each module/sub-package
# into the package namespace
from ._version import __version__

from . import algorithms as alg
from . import construction as cst
from . import objects as obj
from . import report as rpt

from .algorithms.core import *
from .algorithms.gaugeopt import *
from .algorithms.contract import *
from .algorithms.grammatrix import *
from .tools import *
from .drivers import *


#OLD
#from Core import doLGST,  doExLGST, \
#    doIterativeExLGST, doLSGST, doLSGSTwithModelSelection, doIterativeLSGST, \
#    doIterativeLSGSTwithModelSelection, doMLEGST, doIterativeMLEGST, optimizeGauge, contract, \
#    printGatesetInfo, getRhoAndESpecs, getRhoAndEStrs
##gramRankAndEvals, listStringsLGSTcanEstimate,
#
#from MatrixOps import printMx
#from GateSetConstruction import buildGate, buildVector, buildGateset, buildIdentityVector
#from GateOps import Fidelity
#from JamiolkowskiOps import JTraceDistance
##from ListTools import remove_duplicates, remove_duplicates_in_place
#from GateStringTools import createGateStringList, listLGSTGateStrings, gateStringList
#from Loaders import loadDataset, loadMultiDataset, loadGateset, \
#                    loadGatestringDict, loadGatestringList
#from Writers import writeGateset, writeEmptyDatasetFile, writeDatasetFile, writeGatestringList
#from GateSetTools import generateFakeData
#
##Import Objects at package level
#from evaltree import EvalTree
#from gateset import GateSet
#from gatestring import GateString
#from gatestring import WeightedGateString
#from dataset import DataSet  #, UpgradeOldDataSet
#from multidataset import MultiDataSet
#from outputdata import OutputData #, UpgradeOldDataSets
#from results import Results
#
##Import modules with shortened names for convenience
#
#import LikelihoodFunctions as LF
#import MatrixOps as MOps
#import JamiolkowskiOps as JOps
#import BasisTools as BT
#import AnalysisTools as AT
#import GateStringTools as ST
#import ReportGeneration as RG
#
#
##StdInputParser?
