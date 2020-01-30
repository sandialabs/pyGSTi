""" ModelTest Protocol objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import os as _os
import numpy as _np
import pickle as _pickle
import collections as _collections
import warnings as _warnings
import scipy.optimize as _spo
from scipy.stats import chi2 as _chi2

from . import protocol as _proto
from .. import objects as _objs
from .. import algorithms as _alg
from .. import construction as _construction
from .. import io as _io
from .. import tools as _tools

from ..objects import wildcardbudget as _wild
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects import objectivefns as _objfns


class ModelTest(_proto.Protocol):
    """A protocol that tests how well a model agrees with a given set of data."""

    def __init__(self, model_to_test, target_model=None, gaugeopt_suite=None,
                 gaugeopt_target=None, advancedOptions=None, output_pkl=None,
                 verbosity=2, name=None):

        if advancedOptions is None: advancedOptions = {}
        if advancedOptions.get('set trivial gauge group', True):
            model_to_test = model_to_test.copy()
            model_to_test.default_gauge_group = _objs.TrivialGaugeGroup(model_to_test.dim)  # so no gauge opt is done

        super().__init__(name)
        self.model_to_test = model_to_test
        self.target_model = target_model
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.advancedOptions = advancedOptions
        self.output_pkl = output_pkl
        self.verbosity = verbosity

        self.auxfile_types['model_to_test'] = 'pickle'
        self.auxfile_types['target_model'] = 'pickle'
        self.auxfile_types['gaugeopt_suite'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['gaugeopt_target'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['advancedOptions'] = 'pickle'  # TODO - better later? - json?

    #def run_using_germs_and_fiducials(self, model, dataset, target_model, prep_fiducials,
    #                                  meas_fiducials, germs, maxLengths):
    #    from .gst import StandardGSTDesign as _StandardGSTDesign
    #    design = _StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
    #    return self.run(_proto.ProtocolData(design, dataset))

    def run(self, data, memlimit=None, comm=None):
        the_model = self.model_to_test
        advancedOptions = self.advancedOptions

        if isinstance(data.edesign, _proto.CircuitListsDesign):
            lsgstLists = data.edesign.circuit_lists
        else:
            lsgstLists = [data.edesign.all_circuits_needing_data]

        if self.target_model is not None:
            target_model = self.target_model
        elif hasattr(data.edesign, 'target_model'):
            target_model = data.edesign.target_model
        else:
            target_model = None  # target model isn't necessary

        mdl_lsgst_list = [the_model] * len(lsgstLists)

        #Create profiler
        profile = advancedOptions.get('profile', 1)
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _objs.Profiler(comm, False)
        elif profile == 2: profiler = _objs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        parameters = _collections.OrderedDict()
        parameters['objective'] = advancedOptions.get('objective', 'logl')
        if parameters['objective'] == 'logl':
            parameters['minProbClip'] = advancedOptions.get('minProbClip', 1e-4)
            parameters['radius'] = advancedOptions.get('radius', 1e-4)
        elif parameters['objective'] == 'chi2':
            parameters['minProbClipForWeighting'] = advancedOptions.get(
                'minProbClipForWeighting', 1e-4)
        else:
            raise ValueError("Invalid objective: %s" % parameters['objective'])

        parameters['profiler'] = profiler
        parameters['opLabelAliases'] = advancedOptions.get('opLabelAliases', None)
        parameters['truncScheme'] = advancedOptions.get('truncScheme', "whole germ powers")
        parameters['weights'] = None

        #Set a different default for onBadFit: don't do anything
        if 'onBadFit' not in advancedOptions:
            advancedOptions['onBadFit'] = []  # empty list => 'do nothing'

        from .gst import _package_into_results
        return _package_into_results(self, data, target_model, the_model,
                                     lsgstLists, parameters, None, mdl_lsgst_list,
                                     self.gaugeopt_suite, self.gaugeopt_target, advancedOptions, comm,
                                     memlimit, self.output_pkl, self.verbosity, profiler)
