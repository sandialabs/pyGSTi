from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Classes for evaluating germ set performance."""

import numpy as _np
import matplotlib.pyplot as plt

from ..algorithms import germselection as germsel


class GermSetEval:
    def __init__(self, germset=None, gatesets=None, resultDict=None,
                 errorDict=None):
        self.germset = germset
        self.gatesets = gatesets
        self.resultDict = resultDict
        self.errorDict = errorDict

    def calc_marg_pows(self):
        if self.errorDict is None:
            raise ValueError('Dictionary of estimate errors not present, so '
                             'cannot calculate marginal powers!')
        marginalPowers = {}
        for (gsNum, numClicks, run), convergence in self.errorDict.items():
            for (error0, L0), (error1, L1) in zip(convergence[:-1],
                                                  convergence[1:]):
                if (numClicks, L1) in marginalPowers:
                    marginalPowers[numClicks, L1].append(_np.log(error1/error0)
                                                         / _np.log(L1/L0))
                else:
                    marginalPowers[numClicks, L1] = [_np.log(error1/error0)
                                                     / _np.log(L1/L0)]
        return marginalPowers

    def plot_marg_pows(self, axs=None):
        if self.errorDict is None:
            raise ValueError('Dictionary of estimate errors not present, so '
                             'cannot plot marginal powers!')

        clickNums = set()
        for trueGatesetNum, numClicks, run in self.errorDict:
            clickNums.add(numClicks)
        clickNums = sorted(list(clickNums))
        numClickNums = len(clickNums)
        if axs is None:
            fig = plt.figure(figsize=(6, 4*numClickNums))
            axs = []
            for row in range(1, numClickNums + 1):
                axs.append(fig.add_subplot(numClickNums, 1, row))
        elif len(axs) != numClickNums:
            raise ValueError("The number of axs provided must be equal to the "
                             "number of unique click numbers ({})!"
                             .format(numClickNums))

        marginalPowers = self.calc_marg_pows()
        for row, (ax, currentNumClicks) in enumerate(zip(axs, clickNums)):
            Ls = sorted([L for numClicks, L in marginalPowers.keys()
                         if numClicks == currentNumClicks])
            ax.boxplot([marginalPowers[currentNumClicks, L] for L in Ls],
                       whis=_np.inf)
            ax.set_ylabel(r'Marginal power ($\alpha$)', fontsize=18)
            ax.set_xticklabels(Ls)
            if row == numClickNums - 1:
                ax.set_xlabel('Sequence length ($L$)', fontsize=18)
            ax.set_title(r'{} clicks'.format(currentNumClicks), fontsize=18)

        return axs

    def plot_spectrum(self, spectrum, numGaugeParams=None, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            return_fig_ax = True
        else:
            return_fig_ax = False

        if numGaugeParams is None:
            legend = False
            numGaugeParams = 0
        else:
            legend = True

        ax.semilogy(range(numGaugeParams + 1, len(spectrum) + 1),
                    spectrum[numGaugeParams:], 'o',
                    label='Physical parameters')
        ax.semilogy(range(1, numGaugeParams + 1), spectrum[:numGaugeParams],
                    'o', label='Gauge parameters', color='r')
        if legend:
            ax.legend(loc=4, framealpha=0.5)
        if return_fig_ax:
            return {'fig': fig, 'ax': ax}

    def plot_spectra(self, axs=None):
        missing = [key for key in self.__dict__
                   if key in ['germset', 'gatesets']
                   and self.__dict__[key] is None]
        if len(missing) > 0:
            raise ValueError('Missing {}, so cannot plot spectra!'
                             .format(' and '.join(missing)))

        numGatesets = len(self.gatesets)
        numGaugeParams = germsel.removeSPAMVectors(
            self.gatesets[0]).num_gauge_params()
        if axs is None:
            fig = plt.figure(figsize=(6, 4*numGatesets))
            axs = []
            for row in range(1, numGatesets + 1):
                axs.append(fig.add_subplot(numGatesets, 1, row))
        elif len(axs) != numGatesets:
            raise ValueError("The number of axs provided must be equal to the "
                             "number of gatesets ({})!".format(numGatesets))

        for gatesetNum, (ax, gateset) in enumerate(zip(axs, self.gatesets)):
            spectrum = germsel.test_germ_list_infl(gateset, self.germset,
                                                   scoreFunc='all',
                                                   returnSpectrum=True)[1]
            self.plot_spectrum(spectrum, numGaugeParams, ax)
            ax.set_title('GateSet {}'.format(gatesetNum))

        return axs


    def calc_ranges(self):
        if self.errorDict is None:
            raise ValueError('Dictionary of estimate errors not present, so '
                             'cannot calculate error ranges!')
        errorsAtL = {}
        for (gsNum, numClicks, run), convergence in self.errorDict.items():
            for error, L in convergence:
                if (numClicks, L) in errorsAtL:
                    errorsAtL[numClicks, L].append(error)
                else:
                    errorsAtL[numClicks, L] = [error]

        ranges = {key: _np.percentile(_np.array(value), [0.0, 100.0])
                  for key, value in errorsAtL.items()}

        return ranges


    def plot_ranges(self, axs=None):
        if self.errorDict is None:
            raise ValueError('Dictionary of estimate errors not present, so '
                             'cannot plot error ranges!')

        clickNums = set()
        for trueGatesetNum, numClicks, run in self.errorDict:
            clickNums.add(numClicks)
        clickNums = sorted(list(clickNums))
        numClickNums = len(clickNums)
        if axs is None:
            fig = plt.figure(figsize=(6, 4*numClickNums))
            axs = []
            for row in range(1, numClickNums + 1):
                axs.append(fig.add_subplot(numClickNums, 1, row))
        elif len(axs) != numClickNums:
            raise ValueError("The number of axs provided must be equal to the "
                             "number of unique click numbers ({})!"
                             .format(numClickNums))

        ranges = self.calc_ranges()
        for row, (ax, currentNumClicks) in enumerate(zip(axs, clickNums)):
            Ls = sorted([L for numClicks, L in ranges.keys()
                         if numClicks == currentNumClicks])
            min_dists, max_dists = zip(*[ranges[currentNumClicks, L]
                                         for L in Ls])
            ax.fill_between(Ls, min_dists, max_dists, alpha=0.5,
                            label='{} clicks'.format(int(currentNumClicks)),
                            linewidth=0)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'Distance to truth', fontsize=18)
            ax.set_xticks(Ls)
            ax.set_xticklabels(Ls)
            if row == numClickNums - 1:
                ax.set_xlabel('Sequence length ($L$)', fontsize=18)
            ax.set_title(r'{} clicks'.format(currentNumClicks), fontsize=18)

        return axs
