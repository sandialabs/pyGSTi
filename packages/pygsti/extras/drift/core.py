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
    quantum circuits on *any* number of qubits.

    This is a wrap-around of the StabilityAnalyzer object. More complex analyzes can be implemented
    by directly using that object.

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


def do_time_resolved_rb():
    """

    """
    # codetodo
    return None
