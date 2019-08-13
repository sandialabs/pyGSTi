""" Encapsulates RB results and dataset objects """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy
from . import analysis as _analysis
from . import dataset as _dataset


class RBAnalyzer(object):
    """
    todo

    """
    def __init__(self, specs, ds=None, summary_data=None):
        """
        todo

        """
        if ds is not None:
            self.ds = ds.copy()
        else:
            self.ds = None

        self._specs = tuple(specs)

        if summary_data is None:
            self._summary_data = {}
        else:
            self._summary_data = _copy.deepcopy(summary_data)

        self._rbresults = {}
        self._rbresults['raw'] = {}
        self._rbresults['adjusted'] = {}

    def discard_dataset(self):
        """
        todo

        """
        self.ds = None

    def add_dataset(self, ds):
        """
        todo

        """
        self.ds = ds.copy()

    # def discard_dataset_and_circuits(self):

    #     self.ds = None


    def create_summary_data(self, specindices=None, datatype='adjusted', method='fast', verbosity=2):
        """
        analysis : 'all', 'hamming', 'raw'.

        """
        if method == 'simple':
            if specindices is None:
                specindices = range(len(self._specs))

            for specind in specindices:
                spec = self._specs[specind]

                if specind not in self._summary_data.keys():

                    if verbosity > 0:
                        print(" - Creating summary data {} of {} ...".format(specind + 1, len(specindices)), end='')
                    if verbosity > 1: print("")

                    self._summary_data[specind] = _dataset.create_summary_datasets(self.ds, spec, datatype=datatype,
                                                                                   verbosity=verbosity)

                    if verbosity == 1: print("complete.")

                else:
                    if verbosity > 0:
                        print(" - Summary data already extant for {} of {}".format(specind + 1, len(specindices)))

        elif method == 'fast':
            assert(specindices is None), "The 'fast' method cannot format a subset of the data!"

            summarydata = {}
            numcircuits = len(self.ds.keys())
            percent = 0
            for i, circ in enumerate(self.ds.keys()):

                if verbosity > 0:
                    if _np.floor(100 * i/numcircuits) >= percent:
                        percent += 1
                        print("{} percent complete".format(percent))

                specindices = self.ds.auxInfo[circ]['specindices']
                length = self.ds.auxInfo[circ]['length']
                target = self.ds.auxInfo[circ]['target']

                for specind in specindices:
                    spec = self._specs[specind]
                    structure = spec.get_structure()
                    if specind not in summarydata.keys():

                        summarydata[specind] = {}
                        for qubits in structure:
                            summarydata[specind][qubits] = {}
                            summarydata[specind][qubits]['success_counts'] = {}
                            summarydata[specind][qubits]['total_counts'] = {}
                            summarydata[specind][qubits]['hamming_distance_counts'] = {}

                    for qubits in structure:
                        if length not in summarydata[specind][qubits]['success_counts'].keys():
                            summarydata[specind][qubits]['success_counts'][length] = []
                            summarydata[specind][qubits]['total_counts'][length] = []
                            summarydata[specind][qubits]['hamming_distance_counts'][length] = []

                    dsrow = self.ds[circ]
                    for qubits in structure:
                        if datatype == 'raw':
                            summarydata[specind][qubits]['success_counts'][length].append(_analysis.marginalized_success_counts(dsrow, circ, target, qubits))
                            summarydata[specind][qubits]['total_counts'][length].append(dsrow.total)
                        elif datatype == 'adjusted':
                            summarydata[specind][qubits]['hamming_distance_counts'][length].append(_analysis.marginalized_hamming_distance_counts(dsrow, circ, target, qubits))


            for specind in summarydata.keys():
                spec = self._specs[specind]
                structure = spec.get_structure()
                self._summary_data[specind] = {}
                for qubits in structure:
                    if datatype == 'raw':
                        summarydata[specind][qubits]['hamming_distance_counts'] = None
                    elif datatype == 'adjusted':
                        summarydata[specind][qubits]['success_counts'] = None
                        summarydata[specind][qubits]['total_counts'] = None
                    
                    self._summary_data[specind][qubits] = _dataset.RBSummaryDataset(len(qubits), success_counts=summarydata[specind][qubits]['success_counts'],
                                                total_counts=summarydata[specind][qubits]['total_counts'],
                                                hamming_distance_counts=summarydata[specind][qubits]['hamming_distance_counts'])
       

        else:
            raise ValueError("Input `method` must be 'fast' or 'simple'!")

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
                    self._rbresults['raw'][i][key] = _analysis.std_practice_analysis(rbdata, bootstrap_samples=bootstraps, datatype='raw')
                if (analysis == 'all' and rbdata.datatype == 'hamming_distance_counts') or analysis == 'adjusted':
                    self._rbresults['adjusted'][i][key] = _analysis.std_practice_analysis(rbdata, bootstrap_samples=bootstraps, datatype='adjusted')

    def filter_experiments(self, numqubits=None, containqubits=None, onqubits=None, twoQgateprob=None,
                           prefilter=None):
        """
        todo

        """

        kept = {}
        for i, spec in enumerate(self._specs):
            structures = spec.get_structure()
            for qubits in structures:

                keep = True

                if numqubits is not None:
                    if len(qubits) != numqubits:
                        keep = False

                if containqubits is not None:
                    if not set(containqubits).issubset(qubits):
                        keep = False

                if onqubits is not None:
                    if set(qubits) != set(onqubits):
                        keep = False

                if twoQgateprob is not None:
                    if not _np.allclose(twoQgateprob, spec.get_twoQgate_rate()):
                        keep = False

                # Once all filters are applied, check whether we're keeping this (i,qubits) pair
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
