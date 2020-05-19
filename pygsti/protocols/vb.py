""" Volumetric Benchmarking Protocol objects """
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
import copy as _copy
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
from ..algorithms import randomcircuit as _rc


class ByDepthDesign(_proto.CircuitListsDesign):
    """ Experiment design that holds circuits organized by depth """

    def __init__(self, depths, circuit_lists, qubit_labels=None, remove_duplicates=True):
        assert(len(depths) == len(circuit_lists)), \
            "Number of depths must equal the number of circuit lists!"
        super().__init__(circuit_lists, qubit_labels=qubit_labels, remove_duplicates=remove_duplicates)
        self.depths = depths


class BenchmarkingDesign(ByDepthDesign):
    """
    Experiment design that holds benchmarking data, i.e. definite-outcome
    circuits organized by depth along with their corresponding ideal outcomes.
    """

    def __init__(self, depths, circuit_lists, ideal_outs, qubit_labels=None, remove_duplicates=False):
        assert(len(depths) == len(ideal_outs))
        super().__init__(depths, circuit_lists, qubit_labels, remove_duplicates)
        self.idealout_lists = ideal_outs
        self.auxfile_types['idealout_lists'] = 'json'


class PeriodicMirrorCircuitDesign(BenchmarkingDesign):
    """ This Design is a temporary hack for a periodic mirror circuits experiment"""
    @classmethod
    def from_existing_circuits(cls, circuits_and_idealouts_by_depth, qubit_labels=None,
                               sampler='edgegrab', samplerargs=(1 / 4), localclifford=True,
                               paulirandomize=True, fixed_versus_depth=False,
                               descriptor='A random germ mirror circuit experiment'):
        depths = sorted(list(circuits_and_idealouts_by_depth.keys()))
        circuit_lists = [[x[0] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        ideal_outs = [[x[1] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        circuits_per_depth = [len(circuits_and_idealouts_by_depth[d]) for d in depths]
        self = cls.__new__(cls)
        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, localclifford, paulirandomize, fixed_versus_depth, descriptor)
        return self

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, sampler='edgegrab', samplerargs=(1 / 4,),
                 localclifford=True, paulirandomize=True, fixed_versus_depth=False,
                 descriptor='A random germ mirror circuit experiment'):

        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = [[] for d in depths]
        ideal_outs = [[] for d in depths]

        for j in range(circuits_per_depth):
            circtemp, outtemp, junk = _rc.random_germpower_mirror_circuits(pspec, depths, qubit_labels=qubit_labels,
                                                                           localclifford=localclifford,
                                                                           paulirandomize=paulirandomize,
                                                                           interactingQs_density=samplerargs[0],
                                                                           fixed_versus_depth=fixed_versus_depth)
            for ind in range(len(depths)):
                circuit_lists[ind].append(circtemp[ind])
                ideal_outs[ind].append(outtemp[ind])

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, localclifford, paulirandomize, fixed_versus_depth, descriptor)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         sampler, samplerargs, localclifford, paulirandomize, fixed_versus_depth, descriptor):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor
        self.sampler = sampler
        self.samplerargs = samplerargs
        self.localclifford = localclifford
        self.paulirandomize = paulirandomize
        self.fixed_versus_depth = fixed_versus_depth


class SummaryStatistics(_proto.Protocol):
    """
    A protocol that can construct "summary" quantities from the raw data.

    """
    summary_statistics = ('success_counts', 'total_counts', 'hamming_distance_counts',
                         'success_probabilities', 'polarization', 'adjusted_success_probabilities')
    circuit_statistics = ('twoQgate_count', 'circuit_depth', 'idealout', 'circuit_index', 'circuit_width')
    # dscmp_statistics = ('tvds', 'pvals', 'jsds', 'llrs', 'sstvds')

    def __init__(self, name):
        super().__init__(name)

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
                    hamming_distance_counts[_tools.rbtools.hamming_distance(outbitstring, idealout[-1])] += counts
            return list(hamming_distance_counts)  # why a list?

        def adjusted_success_probability(hamming_distance_counts):
            """ TODO: docstring """
            hamming_distance_pdf = _np.array(hamming_distance_counts) / _np.sum(hamming_distance_counts)
            adjSP = _np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])
            return adjSP

        def get_summary_values(icirc, circ, dsrow, idealout):
            sc = success_counts(dsrow, circ, idealout)
            tc = dsrow.total
            hdc = hamming_distance_counts(dsrow, circ, idealout)
            sp = _np.nan if tc == 0 else sc / tc
            nQ = len(circ.line_labels)
            pol = (sp - 1 / 2**nQ) / (1 - 1 / 2**nQ)
            ret = {'success_counts': sc,
                   'total_counts': tc,
                   'success_probabilities': sp,
                   'polarization': pol,
                   'hamming_distance_counts': hdc,
                   'adjusted_success_probabilities': adjusted_success_probability(hdc)}
            return ret

        return self.compute_dict(data, self.summary_statistics,
                                 get_summary_values, for_passes='all')

    def compute_circuit_data(self, data):

        def get_circuit_values(icirc, circ, dsrow, idealout):
            ret = {'twoQgate_count': circ.two_q_gate_count(),
                   'circuit_depth': circ.depth(),
                   'idealout': idealout,
                   'circuit_index': icirc,
                   'circuit_width': len(circ.line_labels)}
            ret.update(dsrow.aux)  # note: will only get aux data from *first* pass in multi-pass data
            return ret

        return self.compute_dict(data, self.circuit_statistics, get_circuit_values, for_passes="first")

    # def compute_dscmp_data(self, data, dscomparator):

    #     def get_dscmp_values(icirc, circ, dsrow, idealout):
    #         ret = {'tvds': dscomparator.tvds.get(circ, _np.nan),
    #                'pvals': dscomparator.pVals.get(circ, _np.nan),
    #                'jsds': dscomparator.jsds.get(circ, _np.nan),
    #                'llrs': dscomparator.llrs.get(circ, _np.nan)}
    #         return ret

    #     return self.compute_dict(data, "dscmpdata", self.dsmp_statistics, get_dscmp_values, for_passes="none")

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

        design = data.edesign
        ds = data.dataset

        depths = design.depths
        qty_data = _tools.NamedDict('Datatype', 'category', None, None,
                                    {comp: _tools.NamedDict('Depth', 'int', 'Value', 'float', {depth: [] for depth in depths})
                                     for comp in component_names})

        #loop over all circuits
        for depth, circuits_at_depth, idealouts_at_depth in zip(depths, design.circuit_lists, design.idealout_lists):
            for icirc, (circ, idealout) in enumerate(zip(circuits_at_depth, idealouts_at_depth)):
                dsrow = ds[circ] if (ds is not None) else None  # stripOccurrenceTags=True ??
                # -- this is where Tim thinks there's a bottleneck, as these loops will be called for each
                # member of a simultaneous experiment separately instead of having an inner-more iteration
                # that loops over the "structure", i.e. the simultaneous qubit sectors.
                #TODO: <print percentage>

                for component_name, val in compute_fn(icirc, circ, dsrow, idealout).items():
                    qty_data[component_name][depth].append(val)  # maybe use a pandas dataframe here?

        return qty_data

    def create_depthwidth_dict(self, depths, widths, fillfn, seriestype):
        return _tools.NamedDict(
            'Depth', 'int', None, None, {depth: _tools.NamedDict(
                'Width', 'int', 'Value', seriestype, {width: fillfn() for width in widths}) for depth in depths})

    def add_bootstrap_qtys(self, data_cache, num_qtys, finitecounts=True):
        """
        Adds bootstrapped "summary datasets". The bootstrap is over both the finite counts of each
        circuit and over the circuits at each length.

        Note: only adds quantities if they're needed.

        Parameters
        ----------
        num_qtys : int, optional
            The number of bootstrapped datasets to construct.

        Returns
        -------
        None
        """
        key = 'bootstraps' if finitecounts else 'infbootstraps'
        if key in data_cache:
            num_existing = len(data_cache['bootstraps'])
        else:
            data_cache[key] = []
            num_existing = 0

        #extract "base" values from cache, to base boostrap off of
        success_probabilities = data_cache['success_probabilities']
        total_counts = data_cache['total_counts']
        hamming_distance_counts = data_cache['hamming_distance_counts']
        depths = list(success_probabilities.keys())

        for i in range(num_existing, num_qtys):

            component_names = self.summary_statistics
            bcache = _tools.NamedDict(
                'Datatype', 'category', None, None,
                {comp: _tools.NamedDict('Depth', 'int', 'Value', 'float', {depth: [] for depth in depths})
                 for comp in component_names})  # ~= "RB summary dataset"

            for depth, SPs in success_probabilities.items():
                numcircuits = len(SPs)
                for k in range(numcircuits):
                    ind = _np.random.randint(numcircuits)
                    sampledSP = SPs[ind]
                    totalcounts = total_counts[depth][ind] if finitecounts else None
                    bcache['success_probabilities'][depth].append(sampledSP)
                    if finitecounts:
                        bcache['success_counts'][depth].append(_np.random.binomial(totalcounts, sampledSP))
                        bcache['total_counts'][depth].append(totalcounts)
                    else:
                        bcache['success_counts'][depth].append(sampledSP)

                    #ind = _np.random.randint(numcircuits)  # note: old code picked different random ints
                    #totalcounts = total_counts[depth][ind] if finitecounts else None  # need this if a new randint
                    sampledHDcounts = hamming_distance_counts[depth][ind]
                    sampledHDpdf = _np.array(sampledHDcounts) / _np.sum(sampledHDcounts)

                    if finitecounts:
                        bcache['hamming_distance_counts'][depth].append(
                            list(_np.random.multinomial(totalcounts, sampledHDpdf)))
                    else:
                        bcache['hamming_distance_counts'][depth].append(sampledHDpdf)

                    # replicates adjusted_success_probability function above
                    adjSP = _np.sum([(-1 / 2)**n * sampledHDpdf[n] for n in range(len(sampledHDpdf))])
                    bcache['adjusted_success_probabilities'][depth].append(adjSP)

            data_cache[key].append(bcache)


class ByDepthSummaryStatistics(SummaryStatistics):
    """
    TODO
    """
    def __init__(self, depths='all', statistics_to_compute='polarization', names_to_compute=None,
                 custom_data_src=None, name=None):
        """
        todo

        """
        super().__init__(name)
        self.depths = depths
        self.statistics_to_compute = statistics_to_compute
        self.names_to_compute = statistics_to_compute if (names_to_compute is None) else names_to_compute
        self.custom_data_src = custom_data_src
        # because this *could* be a model or a qty dict (or just a string?)
        self.auxfile_types['custom_data_src'] = 'pickle'

    def _get_statistic_per_depth(self, statistic, data):
        design = data.edesign

        if self.custom_data_src is None:  # then use the data in `data`
            #Note: can only take/put things ("results") in data.cache that *only* depend on the exp. design
            # and dataset (i.e. the DataProtocol object).  Protocols must be careful of this in their implementation!
            if statistic in self.summary_statistics:
                if statistic not in data.cache:
                    summary_data_dict = self.compute_summary_data(data)
                    data.cache.update(summary_data_dict)
            # Code currently doesn't work with a dscmp, so commented out.
            # elif statistic in self.dscmp_statistics:
            #     if statistic not in data.cache:
            #         dscmp_data = self.compute_dscmp_data(data, dscomparator)
            #         data.cache.update(dscmp_data)
            elif statistic in self.circuit_statistics:
                if statistic not in data.cache:
                    circuit_data = self.compute_circuit_data(data)
                    data.cache.update(circuit_data)
            else:
                raise ValueError("Invalid statistic: %s" % statistic)

            statistic_per_depth = data.cache[statistic]

        elif isinstance(self.custom_data_src, _objs.SuccessFailModel):  # then simulate all the circuits in `data`
            assert(statistic in ('success_probabilities', 'polarization')), \
                "Only success probabilities or polarizations can be simulated!"
            sfmodel = self.custom_data_src
            depths = design.depths if self.depths == 'all' else self.depths
            statistic_per_depth = _tools.NamedDict('Depth', 'int', 'Value', 'float', {depth: [] for depth in depths})
            circuit_lists_for_depths = {depth: lst for depth, lst in zip(design.depths, design.circuit_lists)}

            for depth in depths:
                for circ in circuit_lists_for_depths[depth]:
                    predicted_success_prob = sfmodel.probs(circ)[('success',)]
                    if statistic == 'success_probabilities':
                        statistic_per_depth[depth].append(predicted_success_prob)
                    elif statistic == 'polarization':
                        nQ = len(circ.line_labels)
                        pol = (predicted_success_prob - 1 / 2**nQ) / (1 - 1 / 2**nQ)
                        statistic_per_depth[depth].append(pol)

        # Note sure this is used anywhere, so commented out for now.
        #elif isinstance(custom_data_src, dict):  # Assume this is a "summary dataset"
        #    summary_data = self.custom_data_src
        #    statistic_per_depth = summary_data['success_probabilities']

        else:
            raise ValueError("Invalid 'custom_data_src' of type: %s" % str(type(self.custom_data_src)))

        return statistic_per_depth

    def run(self, data, memlimit=None, comm=None, dscomparator=None):
        """
        TODO

        """
        design = data.edesign
        width = len(design.qubit_labels)
        results = SummaryStatisticsResults(data, self)

        for statistic, statistic_nm in zip(self.statistics_to_compute, self.names_to_compute):
            statistic_per_depth = self._get_statistic_per_depth(statistic, data)
            depths = statistic_per_depth.keys() if self.depths == 'all' else \
                filter(lambda d: d in statistic_per_depth, self.depths)
            statistic_per_dwc = self.create_depthwidth_dict(depths, (width,), lambda: None, 'float')
            # a nested NamedDict with indices: depth, width, circuit_index (width only has single value though)

            for depth in depths:
                percircuitdata = statistic_per_depth[depth]
                statistic_per_dwc[depth][width] = \
                    _tools.NamedDict('CircuitIndex', 'int', 'Value', 'float',
                                     {i: j for i, j in enumerate(percircuitdata)})
            results.statistics[statistic_nm] = statistic_per_dwc
        return results


# This is currently not used I think
# class PredictedByDepthSummaryStatsConstructor(ByDepthSummaryStatsConstructor):
#     """
#     Runs a volumetric benchmark on success/fail data predicted from a model

#     """
#     def __init__(self, model_or_summary_data, depths='all', statistic='mean',
#                  dscomparator=None, name=None):
#         super().__init__(depths, 'success_probabilities', statistic,
#                          dscomparator, model_or_summary_data, name)


class SummaryStatisticsResults(_proto.ProtocolResults):
    """
    The results from running a volumetric benchmark protocol

    """
    def __init__(self, data, protocol_instance):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)
        self.statistics = {}
        self.auxfile_types['statistics'] = 'pickle'  # b/c NamedDicts don't json

    def _my_attributes_as_nameddict(self):
        """Overrides base class behavior so elements of self.statistics form top-level NamedDict"""
        stats = _tools.NamedDict('ValueName', 'category')
        for k, v in self.statistics.items():
            assert(isinstance(v, _tools.NamedDict)), \
                "SummaryStatisticsResults.statistics dict should be populated with NamedDicts, not %s" % str(type(v))
            stats[k] = v
        return stats


#BDB = ByDepthBenchmark
#VBGrid = VolumetricBenchmarkGrid
#VBResults = VolumetricBenchmarkingResults  # shorthand

#Add something like this?
#class PassStabilityTest(_proto.Protocol):
#    pass

# Commented out as we are not using this currently. todo: revive or delete this in the future.
# class VolumetricBenchmarkGrid(Benchmark):
#     """ A protocol that creates an entire depth vs. width grid of volumetric benchmark values """

#     def __init__(self, depths='all', widths='all', datatype='success_probabilities',
#                  paths='all', statistic='mean', aggregate=True, rescaler='auto',
#                  dscomparator=None, name=None):

#         super().__init__(name)
#         self.postproc = VolumetricBenchmarkGridPP(depths, widths, datatype, paths, statistic, aggregate, self.name)
#         self.dscomparator = dscomparator
#         self.rescaler = rescaler

#         self.auxfile_types['postproc'] = 'protocolobj'
#         self.auxfile_types['dscomparator'] = 'pickle'
#         self.auxfile_types['rescaler'] = 'reset'  # punt for now - fix later

#     def run(self, data, memlimit=None, comm=None):
#         #Since we know that VolumetricBenchmark protocol objects Create a single results just fill
#         # in data under the result object's 'volumetric_benchmarks' and 'failure_counts'
#         # keys, and these are indexed by width and depth (even though each VolumetricBenchmark
#         # only contains data for a single width), we can just "merge" the VB results of all
#         # the underlying by-depth datas, so long as they're all for different widths.

#         #Then run resulting data normally, giving a results object
#         # with "top level" dicts correpsonding to different paths
#         VB = ByDepthBenchmark(self.postproc.depths, self.postproc.datatype, self.postproc.statistic,
#                               self.rescaler, self.dscomparator, name=self.name)
#         separate_results = _proto.SimpleRunner(VB).run(data, memlimit, comm)
#         pp_results = self.postproc.run(separate_results, memlimit, comm)
#         pp_results.protocol = self
#         return pp_results


# Commented out as we are not using this currently. todo: revive this in the future.
# class VolumetricBenchmark(_proto.ProtocolPostProcessor):
#     """ A postprocesor that constructs a volumetric benchmark from existing results. """

#     def __init__(self, depths='all', widths='all', datatype='polarization',
#                  statistic='mean', paths='all', edesigntype=None, aggregate=True,
#                  name=None):

#         super().__init__(name)
#         self.depths = depths
#         self.widths = widths
#         self.datatype = datatype
#         self.paths = paths if paths == 'all' else sorted(paths)  # need to ensure paths are grouped by common prefix
#         self.statistic = statistic
#         self.aggregate = aggregate
#         self.edesigntype = edesigntype

#     def run(self, results, memlimit=None, comm=None):
#         data = results.data
#         paths = results.get_tree_paths() if self.paths == 'all' else self.paths
#         #Note: above won't work if given just a results object - needs a dir

#         #Process results
#         #Merge/flatten the data from different paths into one depth vs width grid
#         passnames = list(data.passes.keys()) if data.is_multipass() else [None]
#         passresults = []
#         for passname in passnames:
#             vb = _tools.NamedDict('Depth', 'int', None, None)
#             fails = _tools.NamedDict('Depth', 'int', None, None)
#             path_for_gridloc = {}
#             for path in paths:
#                 #TODO: need to be able to filter based on widths... - maybe replace .update calls
#                 # with something more complicated when width != 'all'
#                 #print("Aggregating path = ", path)  #TODO - show progress something like this later?

#                 #Traverse path to get to root of VB data
#                 root = results
#                 for key in path:
#                     root = root[key]
#                 root = root.for_protocol.get(self.name, None)
#                 if root is None: continue

#                 if passname:  # then we expect final Results are MultiPassResults
#                     root = root.passes[passname]  # now root should be a BenchmarkingResults
#                 assert(isinstance(root, VolumetricBenchmarkingResults))
#                 if self.edesigntype is None:
#                     assert(isinstance(root.data.edesign, ByDepthDesign)), \
#                         "All paths must lead to by-depth exp. design, not %s!" % str(type(root.data.edesign))
#                 else:
#                     if not isinstance(root.data.edsign, self.edesigntype):
#                         continue

#                 #Get the list of depths we'll extract from this (`root`) sub-results
#                 depths = root.data.edesign.depths if (self.depths == 'all') else \
#                     filter(lambda d: d in self.depths, root.data.edesign.depths)
#                 width = len(root.data.edesign.qubit_labels)  # sub-results contains only a single width
#                 if self.widths != 'all' and width not in self.widths: continue  # skip this one

#                 for depth in depths:
#                     if depth not in vb:  # and depth not in fails
#                         vb[depth] = _tools.NamedDict('Width', 'int', 'Value', 'float')
#                         fails[depth] = _tools.NamedDict('Width', 'int', 'Value', None)
#                         path_for_gridloc[depth] = {}  # just used for meaningful error message

#                     if width in path_for_gridloc[depth]:
#                         raise ValueError(("Paths %s and %s both give data for depth=%d, width=%d!  Set the `paths`"
#                                           " argument of this VolumetricBenchmarkGrid to avoid this.") %
#                                          (str(path_for_gridloc[depth][width]), str(path), depth, width))

#                     vb[depth][width] = root.volumetric_benchmarks[depth][width]
#                     fails[depth][width] = root.failure_counts[depth][width]
#                     path_for_gridloc[depth][width] = path

#             if self.statistic in ('minmin', 'maxmax') and not self.aggregate:
#                 self._update_vb_minmin_maxmax(vb)   # aggregate now since we won't aggregate over passes

#             #Create Results
#             results = VolumetricBenchmarkingResults(data, self)
#             results.volumetric_benchmarks = vb
#             results.failure_counts = fails
#             passresults.append(results)

#         agg_fn = _get_statistic_function(self.statistic)

#         if self.aggregate and len(passnames) > 1:  # aggregate pass data into a single set of qty dicts
#             agg_vb = _tools.NamedDict('Depth', 'int', None, None)
#             agg_fails = _tools.NamedDict('Depth', 'int', None, None)
#             template = passresults[0].volumetric_benchmarks  # to get widths and depths

#             for depth, template_by_width_data in template.items():
#                 agg_vb[depth] = _tools.NamedDict('Width', 'int', 'Value', 'float')
#                 agg_fails[depth] = _tools.NamedDict('Width', 'int', 'Value', None)

#                 for width in template_by_width_data.keys():
#                     # ppd = "per pass data"
#                     vb_ppd = [r.volumetric_benchmarks[depth][width] for r in passresults]
#                     fail_ppd = [r.failure_counts[depth][width] for r in passresults]

#                     successcount = 0
#                     failcount = 0
#                     for (successcountpass, failcountpass) in fail_ppd:
#                         successcount += successcountpass
#                         failcount += failcountpass
#                     agg_fails[depth][width] = (successcount, failcount)

#                     if self.statistic == 'dist':
#                         agg_vb[depth][width] = [item for sublist in vb_ppd for item in sublist]
#                     else:
#                         agg_vb[depth][width] = agg_fn(vb_ppd)

#             aggregated_results = VolumetricBenchmarkingResults(data, self)
#             aggregated_results.volumetric_benchmarks = agg_vb
#             aggregated_results.failure_counts = agg_fails

#             if self.statistic in ('minmin', 'maxmax'):
#                 self._update_vb_minmin_maxmax(aggregated_results.qtys['volumetric_benchmarks'])
#             return aggregated_results  # replace per-pass results with aggregated results
#         elif len(passnames) > 1:
#             multipass_results = _proto.MultiPassResults(data, self)
#             multipass_results.passes.update({passname: r for passname, r in zip(passnames, passresults)})
#             return multipass_results
#         else:
#             return passresults[0]

#     def _update_vb_minmin_maxmax(self, vb):
#         for d in vb.keys():
#             for w in vb[d].keys():
#                 for d2 in vb.keys():
#                     for w2 in vb[d2].keys():
#                         if self.statistic == 'minmin' and d2 <= d and w2 <= w and vb[d2][w2] < vb[d][w]:
#                             vb[d][w] = vb[d2][w2]
#                         if self.statistic == 'maxmax' and d2 >= d and w2 >= w and vb[d2][w2] > vb[d][w]:
#                             vb[d][w] = vb[d2][w2]
