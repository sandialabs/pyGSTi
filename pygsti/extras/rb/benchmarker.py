""" Encapsulates RB results and dataset objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import warnings as _warnings
from itertools import cycle as _cycle

import numpy as _np

from pygsti.data import dataset as _stdds, multidataset as _multids, datacomparator as _dcomp
from pygsti.models import oplessmodel as _oplessmodel

#from . import analysis as _analysis
_analysis = None  # MOVED - and this module is deprecated & broken now, so just set to None


class Benchmarker(object):
    """
    todo

    """

    def __init__(self, specs, ds=None, summary_data=None, predicted_summary_data=None,
                 dstype='standard', success_outcome='success', success_key='target',
                 dscomparator=None):
        """
        todo

        dstype : ('success-fail', 'standard')

        specs: dictionary of (name, RBSpec) key-value pairs. The names are arbitrary

        """
        if ds is not None:
            assert(dstype in ('success-fail', 'standard', 'dict')), "Unknown format for the dataset!"
            self.dstype = dstype
            if self.dstype == 'success-fail' or self.dstype == 'dict':
                self.success_outcome = success_outcome
            else:
                self.success_outcome = None
            if self.dstype == 'standard' or self.dstype == 'dict':
                self.success_key = success_key
            else:
                self.success_key = None

            if dstype == 'dict':
                assert('standard' in ds.keys() and 'success-fail' in ds.keys())
                self.multids = ds
            else:
                self.multids = {}
                if isinstance(ds, _stdds.DataSet):
                    self.multids[self.dstype] = _multids.MultiDataSet()
                    self.multids[self.dstype].add_dataset(0, ds)
                elif isinstance(ds, list):
                    self.multids[self.dstype] = _multids.MultiDataSet()
                    for i, subds in enumerate(ds):
                        self.multids[self.dstype].add_dataset(i, ds)
                elif isinstance(ds, _multids.MultiDataSet):
                    self.multids[self.dstype] = ds
                else:
                    raise ValueError("If specified, `ds` must be a DataSet, a list of DataSets,"
                                     + " a MultiDataSet or a dictionary of MultiDataSets!")

            self.numpasses = len(self.multids[list(self.multids.keys())[0]])
        else:
            assert(summary_data is not None), "Must specify one or more DataSets or a summary data dict!"
            self.multids = None
            self.success_outcome = None
            self.success_key = None
            self

        self.dscomparator = _copy.deepcopy(dscomparator)

        self._specs = tuple(specs.values())
        self._speckeys = tuple(specs.keys())

        if summary_data is None:
            self.pass_summary_data = {}
            self.global_summary_data = {}
            self.aux = {}
        else:
            assert(isinstance(summary_data, dict)), "The summary data must be a dictionary"
            self.pass_summary_data = summary_data['pass'].copy()
            self.global_summary_data = summary_data['global'].copy()
            self.aux = summary_data.get('aux', {}).copy()
            if self.multids is None:
                arbqubits = self._specs[0].get_structure()[0]
                arbkey = list(self.pass_summary_data[0][arbqubits].keys())[0]
                arbdepth = list(self.pass_summary_data[0][arbqubits][arbkey].keys())[0]
                self.numpasses = len(self.pass_summary_data[0][arbqubits][arbkey][arbdepth])

        if predicted_summary_data is None:
            self.predicted_summary_data = {}
        else:
            self.predicted_summary_data = predicted_summary_data.copy()

    def select_volumetric_benchmark_regions(self, depths, boundary, widths='all', datatype='success_probabilities',
                                            statistic='mean', merit='aboveboundary', specs=None, aggregate=True,
                                            passnum=None, rescaler='auto'):

        # Selected regions encodes the selected regions, but in the slighty obtuse format of a dictionary of spec
        # indices and a list of tuples of qubit regions. (so, e.g., if 1- and 2-qubit circuit are run in parallel
        # the width-1 and width-2 spec chosen could by encoded as the index of that spec and a length-2 list of those
        # regions.). A less obtuse way to represent the region selection should maybe be used in the future.
        selected_regions = {}
        assert(statistic in ('max', 'mean', 'min'))

        if specs is None:
            specs = self._specs

        specsbywidth = {}
        for ind, structure in specs.items():
            for qs in structure:
                w = len(qs)
                if widths == 'all' or w in widths:
                    if w not in specsbywidth.keys():
                        specsbywidth[w] = []
                    specsbywidth[w].append((ind, qs))

        if not aggregate:
            assert(passnum is not None), "Must specify the passnumber data to use for selection if not aggregating!"

        for w, specsforw in specsbywidth.items():

            if len(specsforw) == 1:  # There's no decision to make: only one benchmark of one region of the size w.
                (ind, qs) = specsforw[0]
                if ind not in selected_regions:
                    selected_regions[ind] = [qs, ]
                else:
                    selected_regions[ind].append(qs)

            else:  # There's data for more than one region (and/or multiple benchmarks of a single region) of size w
                best_boundary_index = 0
                best_vb_at_best_boundary_index = None
                for (ind, qs) in specsforw:
                    vbdata = self.volumetric_benchmark_data(depths, widths=[w, ], datatype=datatype,
                                                            statistic=statistic, specs={ind: [qs, ]},
                                                            aggregate=aggregate, rescaler=rescaler)['data']
                    # Only looking at 1 width, so drop the width key, and keep only the depths with data
                    if not aggregate:
                        vbdata = {d: vbdata[d][w][passnum] for d in vbdata.keys() if w in vbdata[d].keys()}
                    else:
                        vbdata = {d: vbdata[d][w] for d in vbdata.keys() if w in vbdata[d].keys()}

                    # We calcluate the depth index of the largest depth at which the data is above/below the boundary,
                    # ignoring cases where there's data missing at some depths as long as we're still above/below the
                    # boundard at a larger depth.
                    if merit == 'aboveboundary':
                        x = [vbdata[d] > boundary if d in vbdata.keys() else None for d in depths]
                    if merit == 'belowboundary':
                        x = [vbdata[d] < boundary if d in vbdata.keys() else None for d in depths]
                    try:
                        x = x[:x.index(False)]
                    except:
                        pass
                    x.reverse()
                    try:
                        boundary_index = len(x) - 1 - x.index(True)
                        #print("There's a non-zero boundary!", str(w), qs)
                    except:
                        boundary_index = 0
                        #print("Zero boundary!", str(w), qs)

                    if boundary_index > best_boundary_index:
                        best_boundary_index = boundary_index
                        selected_region_at_w = (ind, qs)
                        best_vb_at_best_boundary_index = vbdata[depths[boundary_index]]
                    elif boundary_index == best_boundary_index:
                        if best_vb_at_best_boundary_index is None:
                            # On first run through we automatically select that region
                            selected_region_at_w = (ind, qs)
                            best_vb_at_best_boundary_index = vbdata[depths[boundary_index]]
                        else:
                            if merit == 'aboveboundary' \
                               and vbdata[depths[boundary_index]] > best_vb_at_best_boundary_index:
                                selected_region_at_w = (ind, qs)
                                best_vb_at_best_boundary_index = vbdata[depths[boundary_index]]
                            if merit == 'belowboundary' \
                               and vbdata[depths[boundary_index]] < best_vb_at_best_boundary_index:
                                selected_region_at_w = (ind, qs)
                                best_vb_at_best_boundary_index = vbdata[depths[boundary_index]]
                    else:
                        pass

                (ind, qs) = selected_region_at_w
                if ind not in selected_regions:
                    selected_regions[ind] = [qs, ]
                else:
                    selected_regions[ind].append(qs)

        return selected_regions

    def volumetric_benchmark_data(self, depths, widths='all', datatype='success_probabilities',
                                  statistic='mean', specs=None, aggregate=True, rescaler='auto'):

        # maxmax : max over all depths/widths larger or equal
        # minmin : min over all deoths/widths smaller or equal.

        assert(statistic in ('max', 'mean', 'min', 'dist', 'maxmax', 'minmin'))

        if isinstance(widths, str):
            assert(widths == 'all')
        else:
            assert(isinstance(widths, list) or isinstance(widths, tuple))

        if specs is None:  # If we're not given a filter, we use all of the data.
            specs = {i: [qs for qs in spec.get_structure()] for i, spec in enumerate(self._specs)}

        width_to_spec = {}
        for i, structure in specs.items():
            for qs in structure:
                w = len(qs)
                if widths == 'all' or w in widths:
                    if w not in width_to_spec:
                        width_to_spec[w] = (i, qs)
                    else:
                        raise ValueError(("There are multiple qubit subsets of size {} benchmarked! "
                                          "Cannot have specs as None!").format(w))

        if widths == 'all':
            widths = list(width_to_spec.keys())
            widths.sort()
        else:
            assert(set(widths) == set(list(width_to_spec.keys())))

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

        # if samecircuitpredictions:
        #     predvb = {d: {} for d in depths}
        # else:
        #     predvb = None

        qs = self._specs[0].get_structure()[0]  # An arbitrary key
        if datatype in self.pass_summary_data[0][qs].keys():
            datadict = self.pass_summary_data
            globaldata = False
        elif datatype in self.global_summary_data[0][qs].keys():
            datadict = self.global_summary_data
            globaldata = True
        else:
            raise ValueError("Unknown datatype!")

        if aggregate or globaldata:
            vb = {d: {} for d in depths}
            fails = {d: {} for d in depths}
        else:
            vb = [{d: {} for d in depths} for i in range(self.numpasses)]
            fails = [{d: {} for d in depths} for i in range(self.numpasses)]

        if len(self.predicted_summary_data) > 0:
            arbkey = list(self.predicted_summary_data.keys())[0]
            dopredictions = datatype in self.predicted_summary_data[arbkey][0][qs].keys()
            if dopredictions:
                pkeys = self.predicted_summary_data.keys()
                predictedvb = {pkey: {d: {} for d in depths} for pkey in pkeys}
            else:
                predictedvb = {pkey: None for pkey in self.predicted_summary_data.keys()}

        for w in widths:
            (i, qs) = width_to_spec[w]
            data = datadict[i][qs][datatype]
            if dopredictions:
                preddata = {pkey: self.predicted_summary_data[pkey][i][qs][datatype] for pkey in pkeys}
            for d in depths:
                if d in data.keys():

                    dline = data[d]

                    if globaldata:

                        failcount = _np.sum(_np.isnan(dline))
                        fails[d][w] = (len(dline) - failcount, failcount)

                        if statistic == 'dist':
                            vb[d][w] = rescale_function(dline, w)
                        else:
                            if not _np.isnan(rescale_function(dline, w)).all():
                                if statistic == 'max' or statistic == 'maxmax':
                                    vb[d][w] = _np.nanmax(rescale_function(dline, w))
                                elif statistic == 'mean':
                                    vb[d][w] = _np.nanmean(rescale_function(dline, w))
                                elif statistic == 'min' or statistic == 'minmin':
                                    vb[d][w] = _np.nanmin(rescale_function(dline, w))
                            else:
                                vb[d][w] = _np.nan

                    else:
                        failline = [(len(dpass) - _np.sum(_np.isnan(dpass)), _np.sum(_np.isnan(dpass)))
                                    for dpass in dline]

                        if statistic == 'max' or statistic == 'maxmax':
                            vbdataline = [_np.nanmax(rescale_function(dpass, w))
                                          if not _np.isnan(rescale_function(dpass, w)).all() else _np.nan
                                          for dpass in dline]
                        elif statistic == 'mean':
                            vbdataline = [_np.nanmean(rescale_function(dpass, w))
                                          if not _np.isnan(rescale_function(dpass, w)).all() else _np.nan
                                          for dpass in dline]
                        elif statistic == 'min' or statistic == 'minmin':
                            vbdataline = [_np.nanmin(rescale_function(dpass, w))
                                          if not _np.isnan(rescale_function(dpass, w)).all() else _np.nan
                                          for dpass in dline]
                        elif statistic == 'dist':
                            vbdataline = [rescale_function(dpass, w) for dpass in dline]

                        if not aggregate:
                            for i in range(len(vb)):
                                vb[i][d][w] = vbdataline[i]
                                fails[i][d][w] = failline[i]

                        if aggregate:

                            successcount = 0
                            failcount = 0
                            for (successcountpass, failcountpass) in failline:
                                successcount += successcountpass
                                failcount += failcountpass
                                fails[d][w] = (successcount, failcount)

                            if statistic == 'dist':
                                vb[d][w] = [item for sublist in vbdataline for item in sublist]
                            else:
                                if not _np.isnan(vbdataline).all():
                                    if statistic == 'max' or statistic == 'maxmax':
                                        vb[d][w] = _np.nanmax(vbdataline)
                                    elif statistic == 'mean':
                                        vb[d][w] = _np.nanmean(vbdataline)
                                    elif statistic == 'min' or statistic == 'minmin':
                                        vb[d][w] = _np.nanmin(vbdataline)
                                else:
                                    vb[d][w] = _np.nan

                    # Repeat the process for the predictions, but with simpler code as don't have to
                    # deal with passes or NaNs.
                    if dopredictions:
                        pdline = {pkey: preddata[pkey][d] for pkey in pkeys}
                        for pkey in pkeys:
                            if statistic == 'dist':
                                predictedvb[pkey][d][w] = rescale_function(pdline[pkey], w)
                            if statistic == 'max' or statistic == 'maxmax':
                                predictedvb[pkey][d][w] = _np.max(rescale_function(pdline[pkey], w))
                            if statistic == 'mean':
                                predictedvb[pkey][d][w] = _np.mean(rescale_function(pdline[pkey], w))
                            if statistic == 'min' or statistic == 'minmin':
                                predictedvb[pkey][d][w] = _np.min(rescale_function(pdline[pkey], w))

        if statistic == 'minmin' or statistic == 'maxmax':
            if aggregate:
                for d in vb.keys():
                    for w in vb[d].keys():
                        for d2 in vb.keys():
                            for w2 in vb[d2].keys():
                                if statistic == 'minmin' and d2 <= d and w2 <= w and vb[d2][w2] < vb[d][w]:
                                    vb[d][w] = vb[d2][w2]
                                if statistic == 'maxmax' and d2 >= d and w2 >= w and vb[d2][w2] > vb[d][w]:
                                    vb[d][w] = vb[d2][w2]
            else:
                for i in range(self.numpasses):
                    for d in vb[i].keys():
                        for w in vb[i][d].keys():
                            for d2 in vb[i].keys():
                                for w2 in vb[i][d2].keys():
                                    if statistic == 'minmin' and d2 <= d and w2 <= w and vb[i][d2][w2] < vb[i][d][w]:
                                        vb[i][d][w] = vb[i][d2][w2]
                                    if statistic == 'maxmax' and d2 >= d and w2 >= w and vb[i][d2][w2] > vb[i][d][w]:
                                        vb[i][d][w] = vb[i][d2][w2]

        out = {'data': vb, 'fails': fails, 'predictions': predictedvb}

        return out

    def flattened_data(self, specs=None, aggregate=True):

        flattened_data = {}

        if specs is None:
            specs = self.filter_experiments()

        qubits = self._specs[0].get_structure()[0]  # An arbitrary key in the dict of the summary data.
        if aggregate:
            flattened_data = {dtype: [] for dtype in self.pass_summary_data[0][qubits].keys()}
        else:
            flattened_data = {dtype: [[] for i in range(self.numpasses)]
                              for dtype in self.pass_summary_data[0][qubits].keys()}
        flattened_data.update({dtype: [] for dtype in self.global_summary_data[0][qubits].keys()})
        flattened_data.update({dtype: [] for dtype in self.aux[0][qubits].keys()})
        flattened_data.update({'predictions': {pkey: {'success_probabilities': []}
                                               for pkey in self.predicted_summary_data.keys()}})

        for specind, structure in specs.items():
            for qubits in structure:
                for dtype, data in self.pass_summary_data[specind][qubits].items():
                    for depth, dataline in data.items():
                        #print(specind, qubits, dtype, depth)
                        if aggregate:
                            aggregatedata = _np.array(dataline[0])
                            # print(aggregatedata)
                            # print(type(aggregatedata))
                            # print(type(aggregatedata[0]))
                            for i in range(1, self.numpasses):
                                # print(dataline[i])
                                # print(type(dataline[i]))
                                # print(type(dataline[i][0]))
                                aggregatedata = aggregatedata + _np.array(dataline[i])
                            flattened_data[dtype] += list(aggregatedata)
                        else:
                            for i in range(self.numpasses):
                                flattened_data[dtype][i] += dataline[i]

                for dtype, data in self.global_summary_data[specind][qubits].items():
                    for depth, dataline in data.items():
                        flattened_data[dtype] += dataline
                for dtype, data in self.aux[specind][qubits].items():
                    for depth, dataline in data.items():
                        flattened_data[dtype] += dataline
                for pkey in self.predicted_summary_data.keys():
                    data = self.predicted_summary_data[pkey][specind][qubits]
                    if 'success_probabilities' in data.keys():
                        for depth, dataline in data['success_probabilities'].items():
                            flattened_data['predictions'][pkey]['success_probabilities'] += dataline
                    else:
                        for (depth, dataline1), dataline2 in zip(data['success_counts'].items(),
                                                                 data['total_counts'].values()):
                            flattened_data['predictions'][pkey]['success_probabilities'] += list(
                                _np.array(dataline1) / _np.array(dataline2))

        #  Only do this if we've not already stored the success probabilities in the benchamrker.
        if ('success_counts' in flattened_data) and ('total_counts' in flattened_data) \
           and ('success_probabilities' not in flattened_data):
            if aggregate:
                flattened_data['success_probabilities'] = [sc / tc if tc > 0 else _np.nan for sc,
                                                           tc in zip(flattened_data['success_counts'],
                                                                     flattened_data['total_counts'])]
            else:
                flattened_data['success_probabilities'] = [[sc / tc if tc > 0 else _np.nan for sc, tc in zip(
                    scpass, tcpass)] for scpass, tcpass in zip(flattened_data['success_counts'],
                                                               flattened_data['total_counts'])]

        return flattened_data

    def test_pass_stability(self, formatdata=False, verbosity=1):

        assert(self.multids is not None), \
            "Can only run the stability analysis if a MultiDataSet is contained in this Benchmarker!"

        if not formatdata:
            assert('success-fail' in self.multids.keys()), "Must have generated/imported a success-fail format DataSet!"
        else:
            if 'success-fail' not in self.multids.keys():
                if verbosity > 0:
                    print("No success/fail dataset found, so first creating this dataset from the full data...", end='')
                self.generate_success_or_fail_dataset()
                if verbosity > 0:
                    print("complete.")

        if len(self.multids['success-fail']) > 1:
            self.dscomparator = _dcomp.DataComparator(self.multids['success-fail'], allow_bad_circuits=True)
            self.dscomparator.run(verbosity=verbosity)

    def generate_success_or_fail_dataset(self, overwrite=False):
        """
        """

        assert('standard' in self.multids.keys())
        if not overwrite:
            assert('success-fail' not in self.multids.keys())

        sfmultids = _multids.MultiDataSet()

        for ds_ind, ds in self.multids['standard'].items():
            sfds = _stdds.DataSet(outcome_labels=['success', 'fail'], collision_action=ds.collisionAction)
            for circ, dsrow in ds.items(strip_occurrence_tags=True):
                try:
                    scounts = dsrow[dsrow.aux[self.success_key]]
                except:
                    scounts = 0
                tcounts = dsrow.total
                sfds.add_count_dict(circ, {'success': scounts, 'fail': tcounts - scounts}, aux=dsrow.aux)

            sfds.done_adding_data()
            sfmultids.add_dataset(ds_ind, sfds)

        self.multids['success-fail'] = sfmultids

    # def get_all_data(self):

    #     for circ

    def summary_data(self, datatype, specindex, qubits=None):

        spec = self._specs[specindex]
        structure = spec.get_structure()
        if len(structure) == 1:
            if qubits is None:
                qubits = structure[0]

        assert(qubits in structure), "Invalid choice of qubits for this spec!"

        return self.pass_summary_data[specindex][qubits][datatype]

    #def getauxillary_data(self, datatype, specindex, qubits=None):

    #def get_predicted_summary_data(self, prediction, datatype, specindex, qubits=None):

    def create_summary_data(self, predictions=None, verbosity=2, auxtypes=None):
        """
        todo
        """
        if predictions is None:
            predictions = dict()
        if auxtypes is None:
            auxtypes = []
        assert(self.multids is not None), "Cannot generate summary data without a DataSet!"
        assert('standard' in self.multids.keys()), "Currently only works for standard dataset!"
        useds = 'standard'
        # We can't use the success-fail dataset if there's any simultaneous benchmarking. Not in
        # it's current format anyway.

        summarydata = {}
        aux = {}
        globalsummarydata = {}
        predsummarydata = {}
        predds = None
        preddskey = None
        for pkey in predictions.keys():
            predsummarydata[pkey] = {}
            if isinstance(predictions[pkey], _stdds.DataSet):
                assert(predds is None), "Can't have two DataSet predictions!"
                predds = predictions[pkey]
                preddskey = pkey
            else:
                assert(isinstance(predictions[pkey], _oplessmodel.SuccessFailModel)
                       ), "If not a DataSet must be an ErrorRatesModel!"

        datatypes = ['success_counts', 'total_counts', 'hamming_distance_counts', 'success_probabilities']
        if self.dscomparator is not None:
            stabdatatypes = ['tvds', 'pvals', 'jsds', 'llrs', 'sstvds']
        else:
            stabdatatypes = []

        #preddtypes = ('success_probabilities', )
        auxtypes = ['twoQgate_count', 'depth', 'target', 'width', 'circuit_index'] + auxtypes

        def _get_datatype(datatype, dsrow, circ, target, qubits):

            if datatype == 'success_counts':
                return _analysis.marginalized_success_counts(dsrow, circ, target, qubits)
            elif datatype == 'total_counts':
                return dsrow.total
            elif datatype == 'hamming_distance_counts':
                return _analysis.marginalized_hamming_distance_counts(dsrow, circ, target, qubits)
            elif datatype == 'success_probabilities':
                sc = _analysis.marginalized_success_counts(dsrow, circ, target, qubits)
                tc = dsrow.total
                if tc == 0:
                    return _np.nan
                else:
                    return sc / tc
            else:
                raise ValueError("Unknown data type!")

        numpasses = len(self.multids[useds].keys())

        for ds_ind in self.multids[useds].keys():

            if verbosity > 0:
                print(" - Processing data from pass {} of {}. Percent complete:".format(ds_ind + 1,
                                                                                        len(self.multids[useds])))

            #circuits = {}
            numcircuits = len(self.multids[useds][ds_ind].keys())
            percent = 0

            if preddskey is None or ds_ind > 0:
                iterator = zip(self.multids[useds][ds_ind].items(strip_occurrence_tags=True),
                               self.multids[useds].auxInfo.values(), _cycle(zip([None, ], [None, ])))
            else:
                iterator = zip(self.multids[useds][ds_ind].items(strip_occurrence_tags=True),
                               self.multids[useds].auxInfo.values(),
                               predds.items(strip_occurrence_tags=True))

            for i, ((circ, dsrow), auxdict, (pcirc, pdsrow)) in enumerate(iterator):

                if pcirc is not None:
                    if not circ == pcirc:
                        print('-{}-'.format(i))
                        pdsrow = predds[circ]
                        _warnings.warn("Predicted DataSet is ordered differently to the main DataSet!"
                                       + "Reverting to potentially slow dictionary hashing!")

                if verbosity > 0:
                    if _np.floor(100 * i / numcircuits) >= percent:
                        percent += 1
                        if percent in (1, 26, 51, 76):
                            print("\n    {},".format(percent), end='')
                        else:
                            print("{},".format(percent), end='')
                        if percent == 100:
                            print('')

                speckeys = auxdict['spec']
                try:
                    depth = auxdict['depth']
                except:
                    depth = auxdict['length']
                target = auxdict['target']

                if isinstance(speckeys, str):
                    speckeys = [speckeys]

                for speckey in speckeys:
                    specind = self._speckeys.index(speckey)
                    spec = self._specs[specind]
                    structure = spec.get_structure()

                    # If we've not yet encountered this specind, we create the required dictionaries to store the
                    # summary data from the circuits associated with that spec.
                    if specind not in summarydata.keys():

                        assert(ds_ind == 0)
                        summarydata[specind] = {qubits: {datatype: {}
                                                         for datatype in datatypes} for qubits in structure}
                        aux[specind] = {qubits: {auxtype: {} for auxtype in auxtypes} for qubits in structure}

                        # Only do predictions on the first pass dataset.
                        for pkey in predictions.keys():
                            predsummarydata[pkey][specind] = {}
                            for pkey in predictions.keys():
                                if pkey == preddskey:
                                    predsummarydata[pkey][specind] = {qubits: {datatype: {} for datatype in datatypes}
                                                                      for qubits in structure}
                                else:
                                    predsummarydata[pkey][specind] = {
                                        qubits: {'success_probabilities': {}} for qubits in structure}

                        globalsummarydata[specind] = {qubits: {datatype: {}
                                                               for datatype in stabdatatypes} for qubits in structure}

                    # If we've not yet encountered this depth, we create the list where the data for that depth
                    # is stored.
                    for qubits in structure:
                        if depth not in summarydata[specind][qubits][datatypes[0]].keys():

                            assert(ds_ind == 0)
                            for datatype in datatypes:
                                summarydata[specind][qubits][datatype][depth] = [[] for i in range(numpasses)]
                            for auxtype in auxtypes:
                                aux[specind][qubits][auxtype][depth] = []

                            for pkey in predictions.keys():
                                if pkey == preddskey:
                                    for datatype in datatypes:
                                        predsummarydata[pkey][specind][qubits][datatype][depth] = []
                                else:
                                    predsummarydata[pkey][specind][qubits]['success_probabilities'][depth] = []

                            for datatype in stabdatatypes:
                                globalsummarydata[specind][qubits][datatype][depth] = []

                    #print('---', i)
                    for qubits_ind, qubits in enumerate(structure):
                        for datatype in datatypes:
                            x = _get_datatype(datatype, dsrow, circ, target, qubits)
                            summarydata[specind][qubits][datatype][depth][ds_ind].append(x)
                            # Only do predictions on the first pass dataset.
                            if preddskey is not None and ds_ind == 0:
                                x = _get_datatype(datatype, pdsrow, circ, target, qubits)
                                predsummarydata[preddskey][specind][qubits][datatype][depth].append(x)

                        # Only do predictions and aux on the first pass dataset.
                        if ds_ind == 0:
                            for auxtype in auxtypes:
                                if auxtype == 'twoQgate_count':
                                    auxdata = circ.two_q_gate_count()
                                elif auxtype == 'depth':
                                    auxdata = circ.depth
                                elif auxtype == 'target':
                                    auxdata = target
                                elif auxtype == 'circuit_index':
                                    auxdata = i
                                elif auxtype == 'width':
                                    auxdata = len(qubits)
                                else:
                                    auxdata = auxdict.get(auxtype, None)

                                aux[specind][qubits][auxtype][depth].append(auxdata)

                            for pkey, predmodel in predictions.items():
                                if pkey != preddskey:
                                    if set(circ.line_labels) != set(qubits):
                                        trimmedcirc = circ.copy(editable=True)
                                        for q in circ.line_labels:
                                            if q not in qubits:
                                                trimmedcirc.delete_lines(q)
                                    else:
                                        trimmedcirc = circ

                                    predsp = predmodel.probabilities(trimmedcirc)[('success',)]
                                    predsummarydata[pkey][specind][qubits]['success_probabilities'][depth].append(
                                        predsp)

                            for datatype in stabdatatypes:
                                if datatype == 'tvds':
                                    x = self.dscomparator.tvds.get(circ, _np.nan)
                                elif datatype == 'pvals':
                                    x = self.dscomparator.pVals.get(circ, _np.nan)
                                elif datatype == 'jsds':
                                    x = self.dscomparator.jsds.get(circ, _np.nan)
                                elif datatype == 'llrs':
                                    x = self.dscomparator.llrs.get(circ, _np.nan)
                                globalsummarydata[specind][qubits][datatype][depth].append(x)

            if verbosity > 0:
                print('')

        #  Record the data in the object at the end.
        self.predicted_summary_data = predsummarydata
        self.pass_summary_data = summarydata
        self.global_summary_data = globalsummarydata
        self.aux = aux

    def analyze(self, specindices=None, analysis='adjusted', bootstraps=200, verbosity=1):
        """
        todo

        todo: this partly ignores specindices
        """
        #self.create_summary_data(specindices=specindices, datatype=analysis, verbosity=verbosity)

        for i, rbdatadict in self._summary_data.items():
            #if not isinstance(rbdata, dict):
            #    self._rbresults[i] = rb.analysis.std_practice_analysis(rbdata)
            #else:
            #self._rbresults[i] = {}
            #for key in rbdata.items():
            if verbosity > 0:
                print('- Running analysis for {} of {}'.format(i, len(self._summary_data)))
            self._rbresults['adjusted'][i] = {}
            self._rbresults['raw'][i] = {}
            for j, (key, rbdata) in enumerate(rbdatadict.items()):
                if verbosity > 1:
                    print('   - Running analysis for qubits {} ({} of {})'.format(key, j, len(rbdatadict)))
                if analysis == 'all' or analysis == 'raw':
                    self._rbresults['raw'][i][key] = _analysis.std_practice_analysis(
                        rbdata, bootstrap_samples=bootstraps, datatype='raw')
                if (analysis == 'all' and rbdata.datatype == 'hamming_distance_counts') or analysis == 'adjusted':
                    self._rbresults['adjusted'][i][key] = _analysis.std_practice_analysis(
                        rbdata, bootstrap_samples=bootstraps, datatype='adjusted')

    def filter_experiments(self, numqubits=None, containqubits=None, onqubits=None, sampler=None,
                           two_qubit_gate_prob=None, prefilter=None, benchmarktype=None):
        """
        todo

        """

        kept = {}
        for i, spec in enumerate(self._specs):
            structures = spec.get_structure()
            for qubits in structures:

                keep = True

                if keep:
                    if benchmarktype is not None:
                        if spec.type != benchmarktype:
                            keep = False

                if keep:
                    if numqubits is not None:
                        if len(qubits) != numqubits:
                            keep = False

                if keep:
                    if containqubits is not None:
                        if not set(containqubits).issubset(qubits):
                            keep = False

                if keep:
                    if onqubits is not None:
                        if set(qubits) != set(onqubits):
                            keep = False

                if keep:
                    if sampler is not None:
                        if not spec._sampler == sampler:
                            keep = False

                if keep:
                    if two_qubit_gate_prob is not None:
                        if not _np.allclose(two_qubit_gate_prob, spec.get_twoQgate_rate()):
                            keep = False

                if keep:
                    if i not in kept.keys():
                        kept[i] = []
                    kept[i].append(qubits)

        if prefilter is not None:
            dellist = []
            for key in kept.keys():
                if key not in prefilter.keys():
                    dellist.append(key)
                else:
                    newlist = []
                    for qubits in kept[key]:
                        if qubits in prefilter[key]:
                            newlist.append(qubits)
                    if len(newlist) == 0:
                        dellist.append(key)
                    else:
                        kept[key] = newlist

            for key in dellist:
                del kept[key]

        return kept

        # for i, rbdata in self._adjusted_summary_data.items():
        #     #if not isinstance(rbdata, dict):
        #     #    self._rbresults[i] = rb.analysis.std_practice_analysis(rbdata)
        #     #else:
        #     #self._rbresults[i] = {}
        #     #for key in rbdata.items():
        #     self._adjusted_rbresults[i] = rb.analysis.std_practice_analysis(rbdata, bootstrap_samples=0,
        #                                                                     asymptote=1/4**rbdata.number_of_qubits)


# class RBResults(object):
#     """
#     An object to contain the results of an RB analysis
#     """

#     def __init__(self, data, rtype, fits):
#         """
#         Initialize an RBResults object.

#         Parameters
#         ----------
#         data : RBSummaryDataset
#             The RB summary data that the analysis was performed for.

#         rtype : {'IE','AGI'}
#             The type of RB error rate, corresponding to different dimension-dependent
#             re-scalings of (1-p), where p is the RB decay constant in A + B*p^m.

#         fits : dict
#             A dictionary containing FitResults objects, obtained from one or more
#             fits of the data (e.g., a fit with all A, B and p as free parameters and
#             a fit with A fixed to 1/2^n).
#         """
#         self.data = data
#         self.rtype = rtype
#         self.fits = fits

#     def plot(self, fitkey=None, decay=True, success_probabilities=True, size=(8, 5), ylim=None, xlim=None,
#              legend=True, title=None, figpath=None):
#         """
#         Plots RB data and, optionally, a fitted exponential decay.

#         Parameters
#         ----------
#         fitkey : dict key, optional
#             The key of the self.fits dictionary to plot the fit for. If None, will
#             look for a 'full' key (the key for a full fit to A + Bp^m if the standard
#             analysis functions are used) and plot this if possible. It otherwise checks
#             that there is only one key in the dict and defaults to this. If there are
#             multiple keys and none of them are 'full', `fitkey` must be specified when
#             `decay` is True.

#         decay : bool, optional
#             Whether to plot a fit, or just the data.

#         success_probabilities : bool, optional
#             Whether to plot the success probabilities distribution, as a violin plot. (as well
#             as the *average* success probabilities at each length).

#         size : tuple, optional
#             The figure size

#         ylim, xlim : tuple, optional
#             The x and y limits for the figure.

#         legend : bool, optional
#             Whether to show a legend.

#         title : str, optional
#             A title to put on the figure.

#         figpath : str, optional
#             If specified, the figure is saved with this filename.
#         """

#         # Future : change to a plotly plot.
#         try: import matplotlib.pyplot as _plt
#         except ImportError: raise ValueError("This function requires you to install matplotlib!")

#         if decay and fitkey is None:
#             allfitkeys = list(self.fits.keys())
#             if 'full' in allfitkeys: fitkey = 'full'
#             else:
#                 assert(len(allfitkeys) == 1), \
#                     "There are multiple fits and none have the key 'full'. Please specify the fit to plot!"
#                 fitkey = allfitkeys[0]

#         _plt.figure(figsize=size)
#         _plt.plot(self.data.lengths, self.data.ASPs, 'o', label='Average success probabilities')

#         if decay:
#             lengths = _np.linspace(0, max(self.data.lengths), 200)
#             A = self.fits[fitkey].estimates['A']
#             B = self.fits[fitkey].estimates['B']
#             p = self.fits[fitkey].estimates['p']
#             _plt.plot(lengths, A + B * p**lengths,
#                       label='Fit, r = {:.2} +/- {:.1}'.format(self.fits[fitkey].estimates['r'],
#                                                               self.fits[fitkey].stds['r']))

#         if success_probabilities:
#             _plt.violinplot(list(self.data.success_probabilities), self.data.lengths, points=10, widths=1.,
#                             showmeans=False, showextrema=False, showmedians=False)  # , label='Success probabilities')

#         if title is not None: _plt.title(title)
#         _plt.ylabel("Success probability")
#         _plt.xlabel("RB sequence length $(m)$")
#         _plt.ylim(ylim)
#         _plt.xlim(xlim)

#         if legend: _plt.legend()

#         if figpath is not None: _plt.savefig(figpath, dpi=1000)
#         else: _plt.show()

#         return
