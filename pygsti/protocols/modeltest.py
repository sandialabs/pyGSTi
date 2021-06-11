"""
ModelTest Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections

from pygsti.baseobjs.profiler import DummyProfiler as _DummyProfiler
from pygsti.objectivefns.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitStore
from pygsti.protocols.estimate import Estimate as _Estimate
from . import protocol as _proto
from .. import baseobjs as _baseobjs
from .. import models as _models
from ..objectivefns import objectivefns as _objfns
from ..circuits.circuitlist import CircuitList as _CircuitList
from ..baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation


class ModelTest(_proto.Protocol):
    """
    A protocol that tests how well a model agrees with a given set of data.

    Parameters
    ----------
    model_to_test : Model
        The model to compare with data when :method:`run` is called.

    target_model : Model, optional
        The ideal or desired model of perfect operations.  It is often useful to bundle this
        together with `model_to_test` so that comparison metrics can be easily computed.

    gaugeopt_suite : str or list or dict, optional
        Specifies which gauge optimizations to perform.  Often set to `None` to indicate
        no gauge optimization.  See :class:`GateSetTomography` for more details.

    gaugeopt_target : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates,
        which are used as the default when `gaugeopt_target` is None.

    objfn_builder : ObjectiveFunctionBuilder
        The objective function (builder) that is used to compare the model to data,
        i.e. the objective function that defines this model test.

    badfit_options : GSTBadFitOptions
        Options specifing what constitutes a "bad fit" (or "failed test") and what
        additional actions to take if and when this occurs.

    set_trivial_gauge_group : bool, optional
        A convenience flag that updates the default gauge group of `model_to_test`
        to the trivial gauge group before performing the test, so that no actual gauge
        optimization is performed (even if `gaugeopt_suite` is non-None).

    verbosity : int, optional
        Level of detail printed to stdout.

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """

    @classmethod
    def create_objective_builder(cls, obj):
        """
        Creates objective function builders from `obj` that are commonly used in model tests.

        Parameters
        ----------
        obj : object
            If `obj` is already an :class:`ObjectiveFunctionBuilder` it is used directly.  A
            dictionary is assumed to hold arguments of :method:`ObjectiveFunctionBuilder.simple`.
            A list or tuple is assumed to hold positional arguments of
            :method:`ObjectiveFunctionBuilder.__init__`.

        Returns
        -------
        ObjectiveFunctionBuilder
        """
        builder_cls = _objfns.ObjectiveFunctionBuilder
        if isinstance(obj, builder_cls): return obj
        elif obj is None: return builder_cls.create_from()
        elif isinstance(obj, dict): return builder_cls.create_from(**obj)
        elif isinstance(obj, (list, tuple)): return builder_cls(*obj)
        else: raise ValueError("Cannot build a objective-fn builder from '%s'" % str(type(obj)))

    def __init__(self, model_to_test, target_model=None, gaugeopt_suite=None,
                 gaugeopt_target=None, objfn_builder=None, badfit_options=None,
                 set_trivial_gauge_group=True, verbosity=2, name=None):

        from .gst import GSTBadFitOptions as _GSTBadFitOptions

        if set_trivial_gauge_group:
            model_to_test = model_to_test.copy()
            model_to_test.default_gauge_group = _models.gaugegroup.TrivialGaugeGroup(model_to_test.state_space)
            # ==  no gauge opt

        super().__init__(name)
        self.model_to_test = model_to_test
        self.target_model = target_model
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.badfit_options = _GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        self.objfn_builders = [self.create_objective_builder(objfn_builder)]

        self.auxfile_types['model_to_test'] = 'pickle'
        self.auxfile_types['target_model'] = 'pickle'
        self.auxfile_types['gaugeopt_suite'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['gaugeopt_target'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['badfit_options'] = 'pickle'  # SS: Had issues using json, unclear what was not serializable
        self.auxfile_types['objfn_builders'] = 'pickle'

        #Advanced options that could be changed by users who know what they're doing
        self.profile = 1
        self.oplabel_aliases = None
        self.circuit_weights = None
        self.unreliable_ops = ('Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz')

    #def run_using_germs_and_fiducials(self, model, dataset, target_model, prep_fiducials,
    #                                  meas_fiducials, germs, maxLengths):
    #    from .gst import StandardGSTDesign as _StandardGSTDesign
    #    design = _StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
    #    return self.run(_proto.ProtocolData(design, dataset))

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        ModelEstimateResults
        """
        the_model = self.model_to_test

        if self.target_model is not None:
            target_model = self.target_model
        elif hasattr(data.edesign, 'target_model'):
            target_model = data.edesign.target_model
        else:
            target_model = None  # target model isn't necessary

        #Create profiler
        profile = self.profile
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _baseobjs.Profiler(comm, False)
        elif profile == 2: profiler = _baseobjs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _baseobjs.VerbosityPrinter.create_printer(self.verbosity, comm)
        resource_alloc = _ResourceAllocation(comm, memlimit, profiler, distribute_method='default')

        try:  # take lists if available
            circuit_lists = data.edesign.circuit_lists
        except:
            circuit_lists = [data.edesign.all_circuits_needing_data]
        aliases = circuit_lists[-1].op_label_aliases if isinstance(circuit_lists[-1], _CircuitList) else None
        ds = data.dataset

        if self.oplabel_aliases:  # override any other aliases with ones specifically given
            aliases = self.oplabel_aliases

        bulk_circuit_lists = [_CircuitList(lst, aliases, self.circuit_weights)
                              if not isinstance(lst, _CircuitList) else lst
                              for lst in circuit_lists]
        objfn_vals = []
        chi2k_distributed_vals = []
        assert(len(self.objfn_builders) == 1), "Only support for a single objective function so far."
        for circuit_list in bulk_circuit_lists:
            objective = self.objfn_builders[0].build(the_model, ds, circuit_list, resource_alloc, printer - 1)
            f = objective.fn(the_model.to_vector())
            objfn_vals.append(f)
            chi2k_distributed_vals.append(objective.chi2k_distributed_qty(f))

        mdc_store = _ModelDatasetCircuitStore(the_model, ds, bulk_circuit_lists[-1], resource_alloc)
        parameters = _collections.OrderedDict()
        parameters['raw_objective_values'] = objfn_vals
        parameters['model_test_values'] = chi2k_distributed_vals
        parameters['final_objfn_builder'] = self.objfn_builders[-1]
        parameters['final_mdc_store'] = mdc_store
        parameters['profiler'] = profiler

        from .gst import _add_gaugeopt_and_badfit
        from .gst import ModelEstimateResults as _ModelEstimateResults

        ret = _ModelEstimateResults(data, self)
        models = {'final iteration estimate': the_model, 'iteration estimates': [the_model] * len(bulk_circuit_lists)}
        # TODO: come up with better key names? and must we have iteration_estimates?
        if target_model is not None:
            models['target'] = target_model
        ret.add_estimate(_Estimate(ret, models, parameters), estimate_key=self.name)
        return _add_gaugeopt_and_badfit(ret, self.name, target_model, self.gaugeopt_suite,
                                        self.gaugeopt_target, self.unreliable_ops, self.badfit_options,
                                        None, resource_alloc, printer)
