"""Canned routines for detecting and characterizing instability ("drift") using time-stamped data"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from . import stabilityanalyzer as _sa
from .. import rb as _rb

import numpy as _np
import warnings as _warnings
import itertools as _itertools
import copy as _copy


def do_stability_analysis(ds, significance=0.05, transform='auto', marginalize='auto', mergeoutcomes=None,
                          constnumtimes='auto', ids=False, frequencies='auto', freqpointers={},
                          freqstest=None, tests='auto', inclass_correction={}, betweenclass_weighting='auto',
                          estimator='auto', modelselector=(None, None), verbosity=1):
    """
    Implements instability ("drift") detection and characterization on timeseries data from *any* set of
    quantum circuits on *any* number of qubits. This uses the StabilityAnalyzer object, and directly
    accessing that object allows for some more complex analyzes to be performed.

    Parameters
    ----------

    Returns
    -------
    StabilityAnalyzers
        An object containing the results of the instability detection and characterization. This can
        be used to, e.g., generate plots showing any detected drift, and it can also form the basis
        of further analysis.

    """
    if verbosity > 0: print(" - Formatting the data...", end='')
    results = _sa.StabilityAnalyzer(ds, transform=transform, marginalize=marginalize, mergeoutcomes=mergeoutcomes,
                                    constnumtimes=constnumtimes, ids=ids)
    if verbosity > 0: print("done!")

    # Calculate the power spectra.
    if verbosity > 0: print(" - Calculating power spectra...",end='')
    results.generate_spectra(frequencies=frequencies, freqpointers=freqpointers)
    if verbosity > 0: print("done!")

    # Implement the drift detection with statistical hypothesis testing on the power spectra.
    if verbosity > 0: print(" - Running instability detection...", end='')
    if verbosity > 1: print('')
    results.do_instability_detection(significance=significance, freqstest=freqstest, tests=tests,
                                     inclass_correction=inclass_correction, betweenclass_weighting=betweenclass_weighting,
                                     saveas='default', default=True, overwrite=False, verbosity=verbosity - 1)
    if verbosity == 1: print("done!")

    # Estimate the drifting probabilities.
    if verbosity > 0: print(" - Running instability characterization...", end='')
    if verbosity > 1: print('')

    results.do_instability_characterization(estimator=estimator, modelselector=modelselector, default=True,
                                            verbosity=verbosity - 1)
    if verbosity == 1: print("done!")

    return results


def do_time_resolved_rb(ds, timeslices='auto', significance=0.05, transform='auto', constnumtimes='auto',
                        frequencies='auto', freqpointers={}, freqtest=None, estimator='auto', verbosity=1):
    """
    Implements a time-resolved randomized benchmarking (RB) analysis, on time-series RB data. This data can
    be from any form of RB in which the observed sucess/survial probability is fit to the standard
    exponential form Pm = A + Bp^m.

    """
    mergeoutcomes = todo
    trrb_tests = ((),)
    trrb_inclass_correction = {}
    trrb_modelselector = ('default', ((),))

    stabilityanalyzer = do_stability_analysis(ds, significance=significance, transform=transform,
                                              mergeoutcomes=rb_mergeoutcomes, constnumtimes=constnumtimes, ids=True,
                                              frequencies=frequencies, freqpointers=freqpointers, freqstest=freqtest,
                                              tests=trrb_tests, inclass_correction=trrb_inclass_correction,
                                              betweenclass_weighting='auto', estimator=estimator,
                                              modelselector=trrb_modelselector, verbosity=verbosity - 1)


    return None
