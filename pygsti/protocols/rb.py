""" RB Protocol objects """
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
from .modeltest import ModelTest as _ModelTest
from .. import objects as _objs
from .. import algorithms as _alg
from .. import construction as _construction
from .. import io as _io
from .. import tools as _tools

from ..objects import wildcardbudget as _wild
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects import objectivefns as _objfns

from ..extras import rb as _rb


#Useful to have a base class?
#class RBInput(_proto.ProtocolInput):
#    pass

#Structure:
# MultiInput -> specifies multiple circuit structures on (possibly subsets of) the same data (e.g. collecting into one large dataset the data for multiple protocols)
# MultiProtocol -> runs, on same input circuit structure & data, multiple protocols (e.g. different GST & model tests on same GST data)
#   if that one input is a MultiInput, then it must have the same number of inputs as there are protocols and each protocol is run on the corresponding input.
#   if that one input is a normal input, then the protocols can cache information in a Results object that is handed down.
# SimultaneousInput -- specifies a "qubit structure" for each sub-input
# SimultaneousProtocol -> runs multiple protocols on the same data, but "trims" circuits and data before running sub-protocols
#  (e.g. Volumetric or randomized benchmarks on different subsets of qubits) -- only accepts SimultaneousInputs.

#Inputs:
# Simultaneous: (spec1)
#    Q1: ByDepthData
#    Q2,Q3: ByDepthData
#    Q4: ByDepthData

#Protocols:
# Simultaneous:
#    Q1: Multi
#      VB (aggregate)
#      PredictedModelA
#      PredictedModelB
#      Datasetcomp (between passes)
#    Q2,Q3: VB
#    Q4: VB

#OR: so that auto-running performs the above protocols:
#Inputs:
# Simultaneous: (spec1)
#    Q1: MultiInput(VB, PredictedModelA, PredictedModelB) - or MultiBenchmark?
#       ByDepthData
#    Q2,Q3: VB-ByDepthData
#    Q4: VB-ByDepthData


class ByDepthInput(_proto.CircuitListsInput):
    def __init__(self, depths, circuit_lists, qubit_labels=None):
        assert(len(depths) == len(circuit_lists)), \
            "Number of depths must equal the number of circuit lists!"
        super().__init__(circuit_lists, qubit_labels=qubit_labels)
        self.depths = depths

        
class BenchmarkingInput(ByDepthInput):
    def __init__(self, depths, circuit_lists, ideal_outs, qubit_labels=None):
        assert(len(depths) == len(ideal_outs))
        super().__init__(depths, circuit_lists, qubit_labels)
        self.idealout_lists = ideal_outs


class CliffordRBInput(BenchmarkingInput):

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, randomizeout=False,
                 citerations=20, compilerargs=[], descriptor='A Clifford RB experiment',
                 verbosity=1):
        #Translated from clifford_rb_experiment
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []
    
        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at CRB length {} ({} of {} depths)'.format(circuits_per_depth, l,
                                                                                         lnum + 1, len(depths)))
                print('  - Number of circuits sampled = ', end='')
            circuits_at_depth = []
            idealouts_at_depth = []
            for j in range(circuits_per_depth):
                c, iout = _rb.sample.clifford_rb_circuit(pspec, l, subsetQs=qubit_labels, randomizeout=randomizeout,
                                                  citerations=citerations, compilerargs=compilerargs)
                circuits_at_depth.append(c)
                idealouts_at_depth.append(iout)
                if verbosity > 0: print(j + 1, end=',')
            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)
            if verbosity > 0: print('')

        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels)
        self.circuits_per_depth = circuits_per_depth
        self.randomizeout = randomizeout
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.descriptor = descriptor


class DirectRBInput(BenchmarkingInput):

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, sampler='Qelimination', samplerargs=[],
                 addlocal=False, lsargs=[], randomizeout=False, cliffordtwirl=True, conditionaltwirl=True,
                 citerations=20, compilerargs=[], partitioned=False, descriptor='A DRB experiment',
                 verbosity=1):

        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []

        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at DRB length {} ({} of {} depths)'.format(circuits_per_depth, l,
                                                                                         lnum + 1, len(depths)))
                print('  - Number of circuits sampled = ', end='')
            circuits_at_depth = []
            idealouts_at_depth = []
            for j in range(circuits_per_depth):
                c, iout = _rb.sample.direct_rb_circuit(
                    pspec, l, subsetQs=qubit_labels, sampler=sampler, samplerargs=samplerargs,
                    addlocal=addlocal, lsargs=lsargs, randomizeout=randomizeout,
                    cliffordtwirl=cliffordtwirl, conditionaltwirl=conditionaltwirl,
                    citerations=citerations, compilerargs=compilerargs,
                    partitioned=partitioned)
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))
                if verbosity > 0: print(j + 1, end=',')
            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)
            if verbosity > 0: print('')

        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels)
        self.circuits_per_depth = circuits_per_depth
        self.sampler = sampler
        self.samplerargs = samplerargs
        self.addlocal = addlocal
        self.lsargs = lsargs
        self.randomizeout = randomizeout
        self.cliffordtwirl = cliffordtwirl
        self.conditionaltwirl = conditionaltwirl
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.partitioned = partitioned
        self.descriptor = descriptor


#TODO: maybe need more input types for simultaneous RB and mirrorRB "experiments"

#class MultiPassProtocol(_proto.Protocol):
#    # expects a MultiDataSet of passes and maybe adds data comparison (?) - probably not RB specific
#
#    def run_multipass(self, data):
#
#        assert(isinstance(data.dataset, _objs.MultiDataSet))
#        inp = data.input
#        multids = data.dataset
#        #numpasses = len(multids.keys())
#
#        final_result = _proto.MultiPassResults()
#        for ds_name, ds in multids.items():
#            #TODO: print progress: pass X of Y, etc
#            data = _proto.ProtocolData(inp, ds)
#            results = self.run(data)
#            final_result.add_pass_results(results, ds_name)  #some Result objects have this method??
#
#        return final_result


class Benchmark(_proto.Protocol):

    summary_datatypes = ('success_counts', 'total_counts', 'hamming_distance_counts', 'success_probabilities')
    dscmp_datatypes =  ('tvds', 'pvals', 'jsds', 'llrs', 'sstvds')

    def compute_summary_data(self, data):

        def success_counts(dsrow, circ, idealout):
            if dsrow.total == 0: return 0  # shortcut?
            return dsrow[idealout]

        def hamming_distance_counts(dsrow, circ, idealout):
            nQ = len(circ.line_labels)  # number of qubits
            assert(nQ == len(idealout[-1]))
            hamming_distance_counts = _np.zeros(nQ + 1, float)
            if dsrow.total > 0:
                for outcome_lbl, counts in dsrow.counts.items():
                    outbitstring = outcome_lbl[-1]
                    hamming_distance_counts[_rb.analysis.hamming_distance(outbitstring, idealout[-1])] += counts
            return list(hamming_distance_counts)  # why a list?

        def get_summary_values(icirc, circ, dsrow, idealout):
            sc = success_counts(dsrow, circ, idealout)
            tc = dsrow.total
            ret = {'success_counts': sc,
                   'total_counts': tc,
                   'success_probabilities': _np.nan if tc == 0 else sc / tc,
                   'hamming_distance_counts': hamming_distance_counts(dsrow, circ, idealout)}
            return ret

        return self.compute_dict(data, self.summary_datatypes,
                                 get_summary_values, for_passes='all')

    def compute_circuit_data(self, data):
        names = ['success_counts', 'total_counts', 'hamming_distance_counts', 'success_probabilities']

        def get_circuit_values(icirc, circ, dsrow, idealout):
            ret = {'twoQgate_count': circ.twoQgate_count(),
                   'depth': circ.depth(),
                   'target': idealout,
                   'circuit_index': icirc,
                   'width': len(circ.line_labels)}
            ret.update(dsrow.aux)  # note: will only get aux data from *first* pass in multi-pass data
            return ret

        return self.compute_dict(data, names, get_circuit_values, for_passes="first")

    def compute_dscmp_data(self, data, dscomparator):

        def get_dscmp_values(icirc, circ, dsrow, idealout):
            ret = {'tvds':  dscomparator.tvds.get(circ, _np.nan),
                   'pvals': dscomparator.pVals.get(circ, _np.nan),
                   'jsds':  dscomparator.jsds.get(circ, _np.nan),
                   'llrs':  dscomparator.llrs.get(circ, _np.nan)}
            return ret

        return self.compute_dict(data, "dscmpdata", self.dsmp_datatypes, get_dscmp_values, for_passes="none")

    def compute_predicted_probs(self, data, model):

        def get_success_prob(icirc, circ, dsrow, idealout):
            #if set(circ.line_labels) != set(qubits):
            #    trimmedcirc = circ.copy(editable=True)
            #    for q in circ.line_labels:
            #        if q not in qubits:
            #            trimmedcirc.delete_lines(q)
            #else:
            #    trimmedcirc = circ
            return {'success_probabilities': model.probs(circ)[('success',)]}

        return self.compute_dict(data, ('success_probabilities',),
                                 get_success_prob, for_passes="none")

    #def compute_results_qty(self, results, qtyname, component_names, compute_fn, force=False, for_passes="all"):
    def compute_dict(self, data, component_names, compute_fn, for_passes="all"):

        if isinstance(data.dataset, _objs.MultiDataSet):
            multids = data.dataset
            if for_passes == "all":
                passes = list(multids.items())
            elif for_passes == "first":  # only run for *first* dataset in multidataset
                passes = [(None, multids[multids.keys()[0]])]
            elif for_passes == "none":  # don't run on any data
                passes = [(None, None)]
            else:
                raise ValueError("Invalid 'for_passes' arg!")
        else:
            passes = [(None, data.dataset)]
            assert(for_passes in ("all", "first")), "'for_passes' can only be something other than 'all' for multi-pass data!"

        inp = data.input

        if (passes[0][0] is None):  # whether we have multiple passes
            def passdata(): return []
        else:
            def passdata(): return {passname: [] for passname, ds in passes}

        depths = inp.depths
        qty_data = {comp: {depth: passdata() for depth in depths}
                    for comp in component_names}

        for passname, ds in passes:
            #loop over all circuits
            for depth, circuits_at_depth, idealouts_at_depth in zip(depths, inp.circuit_lists, inp.idealout_lists):
                for icirc, (circ, idealout) in enumerate(zip(circuits_at_depth, idealouts_at_depth)):
                    dsrow = ds[circ] if (ds is not None) else None  # stripOccurrenceTags=True ??
                    # -- this is where Tim thinks there's a bottleneck, as these loops will be called for each
                    # member of a simultaneous experiment separately instead of having an inner-more iteration
                    # that loops over the "structure", i.e. the simultaneous qubit sectors.
                    #TODO: <print percentage>

                    for component_name, val in compute_fn(icirc, circ, dsrow, idealout).items():
                        if passname is None:
                            qty_data[component_name][depth].append(val)  # maybe use a pandas dataframe here?
                        else:
                            qty_data[component_name][depth][passname].append(val)  # maybe use a pandas dataframe here?

        return qty_data


class PassStabilityTest(_proto.Protocol):
    pass

class VolumetricBenchmarkGrid(Benchmark):
    pass  #object that can take, e.g. a multiinput or simultaneous input and create a results object with the desired grid of width vs depth numbers

class VolumetricBenchmark(Benchmark):

    def __init__(self, depths='all', datatype='success_probabilities',
                 statistic='mean', aggregate=True, rescaler='auto', dscomparator=None):

        assert(statistic in ('max', 'mean', 'min', 'dist', 'maxmax', 'minmin'))

        super().__init__()
        self.depths = depths
        #self.widths = widths  # widths='all',
        self.datatype = datatype
        self.statistic = statistic
        self.aggregate = aggregate
        self.dscomparator = dscomparator

        if isinstance(rescaler, str):
            if rescaler == 'auto':
                if datatype == 'success_probabilities':
                    def rescale_function(data, width):
                        return list((_np.array(data) - 1 / 2**width) / (1 - 1 / 2**width))
                else:
                    def rescale_function(data, width):
                        return data
            elif rescaler == 'none':
                def rescale_function(data, width):
                    return data
            else:
                raise ValueError("Unknown rescaling option!")
        else:
            rescale_function = rescaler
        self.rescale_function = rescale_function

    def run(self, data):

        inp = data.input

        #Note: can only take/put things ("results") in data.cache that *only* depend on the input
        # and dataset (i.e. the DataProtocol object).  Protocols must be careful of this in their implementation!
        if self.datatype in self.summary_datatypes:
            if self.datatype not in data.cache:
                summary_data_dict = self.compute_summary_data(data)
                data.cache.update(summary_data_dict)
            src_data = data.cache[self.datatype]
            passnames = list(data.dataset.keys()) if isinstance(data.dataset, _objs.MultiDataSet) else None
        elif self.datatype in self.dscmp_datatypes:
            if self.datatype not in data.cache:
                dscmp_data = self.compute_dscmp_data(data, self.dscomparator)
                data.cache.update(dscmp_data)
            src_data = data.cache[self.datatype]
            passnames = None

        #self.compute_circuit_data(results)
        #self.compute_predicted_probs(results, qtyname, model)

        #Get function to aggregate the different per-circuit datatype values
        if self.statistic == 'max' or self.statistic == 'maxmax':
            np_fn = _np.nanmax
        elif self.statistic == 'mean':
            np_fn = _np.nanmean
        elif self.statistic == 'min' or self.statistic == 'minmin':
            np_fn = _np.nanmin
        elif self.statistic == 'dist':
            def np_fn(v): return v  # identity
        else: raise ValueError("Invalid statistic '%s'!" % self.statistic)

        def agg_fn(percircuitdata, width, rescale=True):
            """ Aggregates datatype-data for all circuits at same depth """
            rescaled = self.rescale_function(percircuitdata, width) if rescale else percircuitdata
            if _np.isnan(rescaled).all():
                return _np.nan
            else:
                return np_fn(rescaled)

        def failcnt_fn(percircuitdata):
            """ Returns (nSucceeded, nFailed) for all circuits at same depth """
            nCircuits = len(percircuitdata)
            failcount = _np.sum(_np.isnan(percircuitdata))
            return (nCircuits - failcount, failcount)

        def new_passdata(depths, widths):
            return {depth: {width: None for width in widths} for depth in depths}

        #TODO REMOVE
        #BEFORE SimultaneousInputs: for qubits in inp.get_structure():        
        #width = len(qubits)

        passdata_per_depth = src_data
        if self.depths == 'all':
            depths = passdata_per_depth.keys()
        else:
            depths = filter(lambda d: d in passdata_per_depth, self.depths)
        width = len(inp.qubit_labels)

        if passnames is None:
            vb = new_passdata(depths, (width,))
            fails = new_passdata(depths, (width,))
        else:
            vb = {passname: new_passdata(depths, (width,)) for passname in passnames}
            fails = {passname: new_passdata(depths, (width,)) for passname in passnames}

        for depth in depths:
            passdata = passdata_per_depth[depth]

            if passnames is None:
                percircuitdata = passdata
                fails[depth][width] = failcnt_fn(percircuitdata)
                vb[depth][width] = agg_fn(percircuitdata, width)
            else:
                failcounts = [failcnt_fn(passdata[passname]) for passname in passnames]
                vbvals = [agg_fn(passdata[passname], width) for passname in passnames]

                if not self.aggregate:
                    for i, passname in enumerate(passnames):
                        vb[passname][depth][width] = vbvals[i]
                        fails[passname][depth][width] = failcounts[i]

                else:  # aggregate pass data
                    successcount = 0
                    failcount = 0
                    for (successcountpass, failcountpass) in failcounts:
                        successcount += successcountpass
                        failcount += failcountpass
                    fails[depth][width] = (successcount, failcount)

                    if self.statistic == 'dist':
                        vb[depth][width] = [item for sublist in vbvals for item in sublist]
                    else:
                        vb[depth][width] = agg_fn(vbvals, width, rescale=False)

        if self.statistic in ('minmin', 'maxmax'):
            raise NotImplementedError("TODO")  # need a new MultiVolumetricBenchmark protocol?
            #def agg_singlepass_vb(vb):
            #    for d in vb.keys():
            #        for w in vb[d].keys():
            #            for d2 in vb.keys():
            #                for w2 in vb[d2].keys():
            #                    if self.statistic == 'minmin' and d2 <= d and w2 <= w and vb[d2][w2] < vb[d][w]:
            #                        vb[d][w] = vb[d2][w2]
            #                    if self.statistic == 'maxmax' and d2 >= d and w2 >= w and vb[d2][w2] > vb[d][w]:
            #                        vb[d][w] = vb[d2][w2]
            #if self.aggregate:
            #    agg_singlepass_vb(vb)
            #else:
            #    for passname in passnames:
            #        agg_singlepass_vb(vb[passname])

        results = _proto.ProtocolResults(data)
        results.qtys['volumetric_benchmarks'] = vb
        results.qtys['failure_counts'] = fails
        return results


class PredictedData(_proto.Protocol):
    #maybe just a function??
    def run(self, data):

        for i, ((circ, dsrow), auxdict, (pcirc, pdsrow)) in enumerate(iterator):
            if pcirc is not None:
                if not circ == pcirc:
                    print('-{}-'.format(i))
                    pdsrow = predds[circ]
                    _warnings.warn("Predicted DataSet is ordered differently to the main DataSet!"
                                   + "Reverting to potentially slow dictionary hashing!")

        #Similar to above but only on first dataset... see create_summary_data
            

class RB(_proto.Protocol):
    pass

