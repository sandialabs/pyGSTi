"""
pyGSTi Objects Python Package
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

#Import the most important/useful routines of each module into
# the package namespace

#Import Objects at package level
from .confidenceregionfactory import ConfidenceRegionFactory
from .dataset import DataSet
from .operation import LinearOperator
from .operation import DenseOperator
from .operation import LinearlyParamDenseOp
from .operation import FullDenseOp
from .operation import TPDenseOp
from .operation import StaticDenseOp
from .operation import EigenvalueParamDenseOp
from .operation import LindbladDenseOp
from .operation import LindbladOp
from .operation import TPInstrumentOp
from .operation import EmbeddedOp
from .operation import EmbeddedDenseOp
from .operation import ComposedOp
from .operation import ComposedDenseOp
from .operation import ExponentiatedOp
from .operation import CliffordOp
from .operation import LindbladErrorgen
from .operation import ComposedErrorgen
from .operation import EmbeddedErrorgen
from .operation import StochasticNoiseOp
from .operation import DepolarizeOp
from .spamvec import SPAMVec
from .spamvec import DenseSPAMVec
from .spamvec import FullSPAMVec
from .spamvec import TPSPAMVec
from .spamvec import CPTPSPAMVec
from .spamvec import ComplementSPAMVec
from .spamvec import StaticSPAMVec
from .spamvec import TensorProdSPAMVec
from .spamvec import PureStateSPAMVec
from .spamvec import LindbladSPAMVec
from .spamvec import ComputationalSPAMVec
from .povm import POVM
from .povm import TPPOVM
from .povm import UnconstrainedPOVM
from .povm import TensorProdPOVM
from .povm import ComputationalBasisPOVM
from .povm import LindbladPOVM
from .povm import MarginalizedPOVM
from .instrument import Instrument
from .instrument import TPInstrument
from .opfactory import OpFactory
from .opfactory import EmbeddedOpFactory
from .opfactory import EmbeddingOpFactory

from .model import Model
from .explicitmodel import ExplicitOpModel
from .explicitmodel import ExplicitOpModel as GateSet  # alias
from .implicitmodel import ImplicitOpModel
from .localnoisemodel import LocalNoiseModel
from .cloudnoisemodel import CloudNoiseModel
from .oplessmodel import OplessModel
from .oplessmodel import SuccessFailModel
from .circuitstructure import CircuitPlaquette, FiducialPairPlaquette, GermFiducialPairPlaquette
from .circuitstructure import PlaquetteGridCircuitStructure
from .circuit import Circuit
from .multidataset import MultiDataSet
from .datacomparator import DataComparator
from .compilationlibrary import CompilationLibrary
from .processorspec import ProcessorSpec
from .stabilizer import StabilizerFrame
from .qubitgraph import QubitGraph
from .hypothesistest import HypothesisTest

from .forwardsim import ForwardSimulator
from .matrixforwardsim import MatrixForwardSimulator
from .mapforwardsim import MapForwardSimulator
from .termforwardsim import TermForwardSimulator

from .gaugegroup import FullGaugeGroup, FullGaugeGroupElement
from .gaugegroup import TPGaugeGroup, TPGaugeGroupElement
from .gaugegroup import DiagGaugeGroup, DiagGaugeGroupElement
from .gaugegroup import TPDiagGaugeGroup, TPDiagGaugeGroupElement
from .gaugegroup import UnitaryGaugeGroup, UnitaryGaugeGroupElement
from .gaugegroup import SpamGaugeGroup, SpamGaugeGroupElement
from .gaugegroup import TPSpamGaugeGroup, TPSpamGaugeGroupElement
from .gaugegroup import TrivialGaugeGroup, TrivialGaugeGroupElement
from .labeldicts import StateSpaceLabels

from .bulkcircuitlist import BulkCircuitList
from .resourceallocation import ResourceAllocation
from .objectivefns import ObjectiveFunctionBuilder, \
    ObjectiveFunction, RawChi2Function, RawChiAlphaFunction, RawFreqWeightedChi2Function, \
    RawPoissonPicDeltaLogLFunction, RawDeltaLogLFunction, RawMaxLogLFunction, RawTVDFunction, \
    Chi2Function, ChiAlphaFunction, FreqWeightedChi2Function, PoissonPicDeltaLogLFunction, DeltaLogLFunction, \
    MaxLogLFunction, TVDFunction, TimeDependentChi2Function, TimeDependentPoissonPicLogLFunction, LogLWildcardFunction

from .objectivefns import ModelDatasetCircuitsStore, EvaluatedModelDatasetCircuitsStore

#from .results import Results  # REMOVE
from .operation import compose, optimize_operation, finite_difference_deriv_wrt_params
from .smartcache import SmartCache
from .verbosityprinter import VerbosityPrinter
from .profiler import Profiler
from .basis import Basis, BuiltinBasis, ExplicitBasis, TensorProdBasis, DirectSumBasis
from .label import Label, CircuitLabel
