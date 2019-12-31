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


#Useful to have a base class?
#class RBInput(_proto.ProtocolInput):
#    pass


class CliffordRBInput(_proto.ProtocolInput):
    
    def __init__(self, pspec, depths, circuits_per_length, subsetQs=None, randomizeout=False,
                 citerations=20, compilerargs=[], descriptor='A Clifford RB experiment',
                 verbosity=1):
        #Translated from clifford_rb_experiment
        self.spec = {}
        self.spec['depths'] = depths
        self.spec['circuits_per_depth'] = circuits_per_depth
        self.spec['subsetQs'] = subsetQs
        self.spec['randomizeout'] = randomizeout
        self.spec['citerations'] = citerations
        self.spec['compilerargs'] = compilerargs
        self.spec['descriptor'] = descriptor
        
        if subsetQs is not None: self.qubitordering = tuple(subsetQs)
        else: self.qubitordering = tuple(pspec.qubit_labels)

        self.circuits = []
        self.idealouts = []
    
        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at CRB length {} ({} of {} depths)'.format(circuits_per_length, l,
                                                                                         lnum + 1, len(depths)))
                print('  - Number of circuits sampled = ', end='')
            circuits_at_depth = []
            idealouts_at_depth = []
            for j in range(circuits_per_depth):
                c, iout = _rb.clifford_rb_circuit(pspec, l, subsetQs=subsetQs, randomizeout=randomizeout,
                                                  citerations=citerations, compilerargs=compilerargs)
                circuits_at_depth.append(c)
                idealouts_at_depth.append(iout)
                if verbosity > 0: print(j + 1, end=',')
            self.circuits.append(circuits_at_depth)
            self.idealouts.append(idealouts_at_depth)
            if verbosity > 0: print('')
            

class DirectRBInput(_proto.ProtocolInput):

    def __init__(self, pspec, depths, circuits_per_length, subsetQs=None, sampler='Qelimination', samplerargs=[],
                 addlocal=False, lsargs=[], randomizeout=False, cliffordtwirl=True, conditionaltwirl=True,
                 citerations=20, compilerargs=[], partitioned=False, descriptor='A DRB experiment',
                 verbosity=1):
        self.spec = {}
        self.spec['depths'] = depths
        self.spec['circuits_per_length'] = circuits_per_length
        self.spec['subsetQs'] = subsetQs
        self.spec['sampler'] = sampler
        self.spec['samplerargs'] = samplerargs
        self.spec['addlocal'] = addlocal
        self.spec['lsargs'] = lsargs
        self.spec['randomizeout'] = randomizeout
        self.spec['cliffordtwirl'] = cliffordtwirl
        self.spec['conditionaltwirl'] = conditionaltwirl
        self.spec['citerations'] = citerations
        self.spec['compilerargs'] = compilerargs
        self.spec['partitioned'] = partitioned
        self.spec['descriptor'] = descriptor
    
        if subsetQs is not None: self.qubitordering = tuple(subsetQs)
        else: self.qubitordering = tuple(pspec.qubit_labels)
    
        self.circuits = {}
        self.target = {}
    
        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at DRB length {} ({} of {} depths)'.format(circuits_per_length, l,
                                                                                         lnum + 1, len(depths)))
                print('  - Number of circuits sampled = ', end='')
            for j in range(circuits_per_length):
                circuit, idealout = _rb.direct_rb_circuit(
                    pspec, l, subsetQs=subsetQs, sampler=sampler, samplerargs=samplerargs,
                    addlocal=addlocal, lsargs=lsargs, randomizeout=randomizeout,
                    cliffordtwirl=cliffordtwirl, conditionaltwirl=conditionaltwirl,
                    citerations=citerations, compilerargs=compilerargs,
                    partitioned=partitioned)
                self.circuits[l, j] = circuit
                self.target[l, j] = idealout
                if verbosity > 0: print(j + 1, end=',')
            if verbosity > 0: print('')

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

    def compute_summary_data(self, results):
        def get_summary_values(icirc, circ, dsrow, idealout, qubits):
            ret = {'success_counts': _analysis.marginalized_success_counts(dsrow, circ, idealout, qubits),
                   'total_counts': dsrow.total,
                   'hamming_distance_counts': _analysis.marginalized_hamming_distance_counts(
                       dsrow, circ, idealout, qubits) }
            sc = _analysis.marginalized_success_counts(dsrow, circ, idealout, qubits)
            tc = dsrow.total
            ret['success_probabilities'] = _np.nan if tc == 0 else sc / tc
            return ret

        return self.compute_results_qty(results, "summarydata", self.summary_datatypes,
                                        get_summary_values, for_passes='all')

    def compute_circuit_data(self, results):
        names = ['success_counts', 'total_counts', 'hamming_distance_counts', 'success_probabilities']

        def get_circuit_values(icirc, circ, dsrow, idealout, qubits):
            ret = {'twoQgate_count': circ.twoQgate_count(),
                   'depth': circ.depth(),
                   'target': idealout,
                   'circuit_index': icirc,
                   'width': len(qubits)}
            ret.update(dsrow.aux)  # note: will only get aux data from *first* pass in multi-pass data
            return ret

        return self.compute_results_qty(results, "circuitdata", names, get_circuit_values, for_passes="first")

    def compute_dscmp_data(self, results, dscomparator):

        def get_dscmp_values(icirc, circ, dsrow, idealout, qubits):
            ret = {'tvds':  dscomparator.tvds.get(circ, _np.nan),
                   'pvals': dscomparator.pVals.get(circ, _np.nan),
                   'jsds':  dscomparator.jsds.get(circ, _np.nan),
                   'llrs':  dscomparator.llrs.get(circ, _np.nan)}
            return ret

        return self.compute_results_qty(results, "dscmpdata", self.dsmp_datatypes, get_dscmp_values, for_passes="none")

    def compute_predicted_probs(self, results, qtyname, model):

        def get_success_prob(icirc, circ, dsrow, idealout, qubits):
            if set(circ.line_labels) != set(qubits):
                trimmedcirc = circ.copy(editable=True)
                for q in circ.line_labels:
                    if q not in qubits:
                        trimmedcirc.delete_lines(q)
            else:
                trimmedcirc = circ
            return {'success_probabilities': model.probs(trimmedcirc)[('success',)]}

        return self.compute_results_qty(results, qtyname, ('success_probabilities',),
                                        get_success_prob, for_passes="none")

    def compute_results_qty(self, results, qtyname, component_names, compute_fn, force=False, for_passes="all"):

        if results.has_qty(qtyname) and force is False: return

        if isinstance(results.dataset, _objs.MultiDataSet):
            multids = results.dataset
            if for_passes == "all":
                 passes = list(multids.items())
            elif for_passes == "first":  # only run for *first* dataset in multidataset
                passes = [(None, multids[multids.keys()[0]])]
            elif for_passes == "none":  # don't run on any data
                passes = [(None, None)]
            else:
                raise ValueError("Invalid 'for_passes' arg!")
        else:
            passes = [(None, results.dataset)]
            assert(for_passes in ("all", "first")), "'for_passes' can only be something other than 'all' for multi-pass data!"

        inp = results.input
        structure = inp.get_structure()

        if (passes[0][0] is None):  # whether we have multiple passes
            def passdata(): return []
        else:
            def passdata(): return {passname: [] for passname, ds in passes}

        depths = inp.spec['depths']
        qty_data = {qubits:
                    {comp:
                     {depth: passdata() for depth in depths}
                     for comp in component_names}
                    for qubits in structure}

        for passname, ds in passes:
            #loop over all circuits
            for depth, circuits_at_depth, idealouts_at_depth in zip(depths, inp.circuits, inp.idealouts):
                for icirc, (circ, idealout) in enumerate(zip(circuits_at_depth, idealouts_at_depth)):
                    dsrow = ds[circ] if (ds is not None) else None  # stripOccurrenceTags=True ?? -- this is where Tim thinks there's a bottleneck
                    #TODO: <print percentage>

                    for qubits in structure:  # a "simultaneous benchmark" experiment -- maybe all are like this?
                        for component_name, val in compute_fn(icirc, circ, dsrow, idealout, qubits).items():
                            if passname is None:
                                qty_data[qubits][component_name][depth].append(val)  # maybe use a pandas dataframe here?
                            else:
                                qty_data[qubits][component_name][depth][passname].append(val)  # maybe use a pandas dataframe here?

        results.add_qty(qty_data, qtyname)

class PassStabilityTest(_proto.Protocol):
    pass

class VolumetricBenchmark(_proto.Protocol):

    def __init__(self, depths, datatype='success_probabilities',
                 statistic='mean', aggregate=True, rescaler='auto'):

        assert(statistic in ('max', 'mean', 'min', 'dist', 'maxmax', 'minmin'))

        super()
        self.depths = depths
        #self.widths = widths  # widths='all',
        self.datatype = datatype
        self.statistic = statistic
        self.aggregate = aggregate
        self.rescaler = rescaler

    def run(self, data):

        inp = data.input

        if isinstance(data, _proto.ProtocolResults):
            results = data
        else:
            results = _proto.ProtocolResults(data)

        if self.datatype in self.summary_datatypes:
            self.compute_summary_data(results)
            src_data = results.get_qty("summarydata")
            passnames = list(results.dataset.keys()) if isinstance(results.dataset, _objs.MultiDataset) else [None]
        elif self.datatype in self.dscmp_datatypes:
            self.compute_dscmp_data(results, dscomparator)
            src_data = results.get_qty("dscmpdata")
            passnames = [None]

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
            
        for qubits in inp.get_structure():
            passdata_per_depth = src_data[qubits][self.datatype]
            width = len(qubits)
            
            for depth in self.depths:
                if depth in passdata_per_depth:
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
                                
        if self.statistic in ('minmin','maxmax'):
            def agg_singlepass_vb(vb):
                for d in vb.keys():
                    for w in vb[d].keys():
                        for d2 in vb.keys():
                            for w2 in vb[d2].keys():
                                if self.statistic == 'minmin' and d2 <= d and w2 <= w and vb[d2][w2] < vb[d][w]:
                                    vb[d][w] = vb[d2][w2]
                                if self.statistic == 'maxmax' and d2 >= d and w2 >= w and vb[d2][w2] > vb[d][w]:
                                    vb[d][w] = vb[d2][w2]
            if self.aggregate:
                agg_singlepass_vb(vb)
            else:
                for passname in passnames:
                    agg_singlepass_vb(vb[passname])

        results.add_qty(vb, 'volumetric_benchmarks')
        results.add_qty(fails, 'failure_counts')
        return results


class PredictedData(_protocol.Protocol):
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

