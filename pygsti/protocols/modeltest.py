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
import warnings as _warnings
import pathlib as _pathlib
from pygsti.baseobjs.profiler import DummyProfiler as _DummyProfiler
from pygsti.objectivefns.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitStore
from pygsti.protocols.estimate import Estimate as _Estimate
from pygsti.protocols import protocol as _proto
from pygsti import baseobjs as _baseobjs
from pygsti import models as _models
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.circuits import Circuit
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation


class ModelTest(_proto.Protocol):
    """
    A protocol that tests how well a model agrees with a given set of data.

    Parameters
    ----------
    model_to_test : Model
        The model to compare with data when :meth:`run` is called.

    target_model : Model, optional
        The ideal or desired model of perfect operations.  It is often useful to bundle this
        together with `model_to_test` so that comparison metrics can be easily computed.

    gaugeopt_suite : GSTGaugeOptSuite, optional
        Specifies which gauge optimizations to perform on each estimate.  Can also
        be any object that can be cast to a :class:`GSTGaugeOptSuite` object, such
        as a string or list of strings (see below) specifying built-in sets of gauge
        optimizations.  This object also optionally stores an alternate target model
        for gauge optimization.  This model is used as the "target" for gauge-
        optimization (only), and is useful when you want to gauge optimize toward
        something other than the *ideal* target gates.

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
            dictionary is assumed to hold arguments of :meth:`ObjectiveFunctionBuilder.simple`.
            A list or tuple is assumed to hold positional arguments of
            :meth:`ObjectiveFunctionBuilder.__init__`.

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
                 objfn_builder=None, badfit_options=None,
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
        self.badfit_options = _GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        self.objfn_builders = [self.create_objective_builder(objfn_builder)]

        self.auxfile_types['model_to_test'] = 'serialized-object'
        self.auxfile_types['target_model'] = 'serialized-object'
        self.auxfile_types['gaugeopt_suite'] = 'serialized-object'
        self.auxfile_types['badfit_options'] = 'serialized-object'
        self.auxfile_types['objfn_builders'] = 'list:serialized-object'

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

    def run(self, data, memlimit=None, comm=None, checkpoint=None, checkpoint_path=None, disable_checkpointing= False):
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

        checkpoint : ModelTestCheckpoint, optional (default None)
            If specified use a previously generated checkpoint object to restart
            or warm start this run part way through.

        checkpoint_path : str, optional (default None)
            A string for the path/name to use for writing intermediate checkpoint
            files to disk. Format is {path}/{name}, without inclusion of the json
            file extension. This {path}/{name} combination will have the latest
            completed iteration number appended to it before writing it to disk.
            If none, the value of {name} will be set to the name of the protocol
            being run.
        
        disable_checkpointing : bool, optional (default False)
            When set to True checkpoint objects will not be constructed and written
            to disk during the course of this protocol. It is strongly recommended
            that this be kept set to False without good reason to disable the checkpoints.

        Returns
        -------
        ModelEstimateResults
        """
        the_model = self.model_to_test
        
        target_model = self.target_model  # can be None; target model isn't necessary

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
        
        if not disable_checkpointing:
            #Set the checkpoint_path variable if None
            if checkpoint_path is None:
                checkpoint_path = _pathlib.Path('./model_test_checkpoints/' + self.name)
            else:
                #cast this to a pathlib path with the file extension (suffix) dropped
                checkpoint_path = _pathlib.Path(checkpoint_path).with_suffix('')
            
            #create the parent directory of the checkpoint if needed:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            #If there is no checkpoint we should start from with the first circuit list,
            #otherwise we should load in the cached results and start from the point
            # in the objective function calculation we left off.
            if checkpoint is None:
                objfn_vals = []
                chi2k_distributed_vals = []
                checkpoint = ModelTestCheckpoint()
            elif isinstance(checkpoint, ModelTestCheckpoint):
                objfn_vals = checkpoint.objfn_vals
                chi2k_distributed_vals = checkpoint.chi2k_distributed_vals
            else:
                NotImplementedError('The only currently valid checkpoint inputs are None and ModelTestCheckpoint.')

            #Check the last completed iteration identified in the checkpoint and set that as the
            #starting point for the iteration through bulk_circuit_lists. This starts at -1 for
            #a freshly initialized checkpoint.
            starting_idx = checkpoint.last_completed_iter + 1
        else:
            starting_idx = 0
            objfn_vals = []
            chi2k_distributed_vals = []

        assert(len(self.objfn_builders) == 1), "Only support for a single objective function so far."
        for i in range(starting_idx, len(bulk_circuit_lists)):
            circuit_list = bulk_circuit_lists[i]
            objective = self.objfn_builders[0].build(the_model, ds, circuit_list, resource_alloc, printer - 1)
            f = objective.fn(the_model.to_vector())
            objfn_vals.append(f)
            chi2k_distributed_vals.append(objective.chi2k_distributed_qty(f))

            if not disable_checkpointing:
                #Update the checkpoint:
                checkpoint.objfn_vals = objfn_vals
                checkpoint.chi2k_distributed_vals = chi2k_distributed_vals
                checkpoint.last_completed_iter += 1
                checkpoint.last_completed_circuit_list= circuit_list
                #write the updated checkpoint to disk:
                if resource_alloc.comm_rank == 0:
                    checkpoint.write(f'{checkpoint_path}_iteration_{i}.json')

        mdc_store = _ModelDatasetCircuitStore(the_model, ds, bulk_circuit_lists[-1], resource_alloc)
        parameters = _collections.OrderedDict()
        parameters['final_objfn_builder'] = self.objfn_builders[-1]
        parameters['final_mdc_store'] = mdc_store
        parameters['profiler'] = profiler

        #Separate these out for now, as the parameters arg will be done away with in future
        # - these "extra" params must be straightforward to serialize
        extra_parameters = _collections.OrderedDict()
        extra_parameters['raw_objective_values'] = objfn_vals
        extra_parameters['model_test_values'] = chi2k_distributed_vals

        from .gst import _add_gaugeopt_and_badfit
        from .gst import ModelEstimateResults as _ModelEstimateResults

        ret = _ModelEstimateResults(data, self)
        models = {'final iteration estimate': the_model}
        models.update({('iteration %d estimate' % k): the_model for k in range(len(bulk_circuit_lists))})
        # TODO: come up with better key names? and must we have iteration_estimates?
        if target_model is not None:
            models['target'] = target_model
        ret.add_estimate(_Estimate(ret, models, parameters, extra_parameters=extra_parameters), estimate_key=self.name)

        #Add some better handling for when gauge optimization is turned off (current code path isn't working.
        if self.gaugeopt_suite is not None:
            ret= _add_gaugeopt_and_badfit(ret, self.name, target_model, self.gaugeopt_suite,
                                            self.unreliable_ops, self.badfit_options,
                                            None, resource_alloc, printer)
        else:
            #add a model to the estimate that we'll call the trivial gauge optimized model which
            #will be set to be equal to the final iteration estimate.
            ret.estimates[self.name].models['trivial_gauge_opt']= the_model
            #and add a key for this to the goparameters dict (this is what the report
            #generation looks at to determine the names of the gauge optimized models).
            #Set the value to None as a placeholder.
            from .gst import GSTGaugeOptSuite
            ret.estimates[self.name].goparameters['trivial_gauge_opt']= None
        return ret


class ModelTestCheckpoint(_proto.ProtocolCheckpoint):
    """
    A class for storing intermediate results associated with running
    a ModelTest protocol's run method to allow for restarting
    that method partway through.

    Parameters
    ----------
    last_completed_iter : int, optional (default -1)
        Index of the last iteration what was successfully completed.

    last_completed_circuit_list : list of Circuit objects, CircuitList or equivalent, optional (default None)
        A list of Circuit objects corresponding to the last iteration successfully completed.

    objfn_vals : list, optional (default None)
        A list of the current objective function values for each iteration/circuit list
        evaluated during the ModelTest protocol.

    chi2k_distributed_vals : list, optional (default None)
        A list of the current objective function values for each iteration/circuit list
         evaluated during the ModelTest protocol rescaled so as to have an expected chi-squared
         distribution under the null hypothesis of Wilks' theorem.
        
    name : str, optional (default None)
        An optional name for the checkpoint. Note this is not necessarily the name used in the
        automatic generation of filenames when written to disk. 
    
    parent : ProtocolCheckpoint, optional (default None)
        When specified this checkpoint object is treated as the child of another ProtocolCheckpoint
        object that acts as the parent. When present, the parent's `write` method supersedes
        the child objects and is called when calling `write` on the child. Currently only used
        in the implementation of StandardGSTCheckpoint.

    """

    def __init__(self, last_completed_iter = -1, 
                 last_completed_circuit_list = None, objfn_vals = None,
                 chi2k_distributed_vals=None, name= None, parent = None):
        self.last_completed_iter = last_completed_iter
        self.last_completed_circuit_list = last_completed_circuit_list if last_completed_circuit_list is not None else []
        self.objfn_vals = objfn_vals if objfn_vals is not None else []
        self.chi2k_distributed_vals = chi2k_distributed_vals if chi2k_distributed_vals is not None else []

        super().__init__(name, parent)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'last_completed_iter': self.last_completed_iter,
                      'last_completed_circuit_list': [ckt.str for ckt in self.last_completed_circuit_list],
                      'objfn_vals': self.objfn_vals,
                      'chi2k_distributed_vals': self.chi2k_distributed_vals,
                      'name': self.name
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        last_completed_iter = state['last_completed_iter']
        last_completed_circuit_list = [Circuit(ckt_str) for ckt_str in state['last_completed_circuit_list']]
        objfn_vals = state['objfn_vals']
        chi2k_distributed_vals = state['chi2k_distributed_vals']
        name = state['name']
        return cls(last_completed_iter, last_completed_circuit_list, 
                   objfn_vals, chi2k_distributed_vals, name)
